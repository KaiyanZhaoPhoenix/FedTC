"""Macro-step dual interval actor/critic, literal eq:dual_td and eq:fallback_gate."""
from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .forward import CONSERVATIVE_ACTION, validate_action


@dataclass
class ControllerConfig:
    td_mode: str = "paper"
    hidden: int = 64
    gamma: float = .9
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    target_rate: float = .005
    popart_beta: float = .05
    popart_min_std: float = 1e-4
    batch_size: int = 16
    replay_capacity: int = 10000
    updates_per_macro: int = 4
    warmup_macros: int = 4
    omega0: float = .5
    kappa: float = .1
    tolerance: float = .05
    exploration_std: float = 0.
    disable_utility: bool = False

    def validate(self):
        if self.td_mode not in {"paper", "consistent"}:
            raise ValueError("td_mode must be paper or consistent.")
        if not (0 <= self.gamma < 1 and 0 < self.target_rate <= 1 and 0 < self.popart_beta <= 1):
            raise ValueError("Invalid controller discount/EMA rates.")
        for name in ("hidden", "batch_size", "updates_per_macro", "replay_capacity", "warmup_macros"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
        if min(self.hidden, self.batch_size, self.updates_per_macro) < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("Invalid controller network/replay sizes.")
        if min(self.actor_lr, self.critic_lr, self.popart_min_std) <= 0:
            raise ValueError("Learning rates and PopArt minimum std must be positive.")
        if min(self.warmup_macros, self.kappa, self.tolerance, self.exploration_std) < 0:
            raise ValueError("Controller margins and warmup cannot be negative.")
        if not .5 <= self.omega0 <= 1:
            raise ValueError("omega0 must be in [.5, 1].")
        if any(not math.isfinite(v) for v in asdict(self).values() if not isinstance(v, str)):
            raise ValueError("Controller parameters must be finite.")


class Actor(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 2)
        nn.init.uniform_(self.head.weight, -.003, .003)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-2., math.log(.4 / .6)]))

    def forward(self, state):
        unit = torch.sigmoid(self.head(self.features(state)))
        return unit * unit.new_tensor([.5, .75]) + unit.new_tensor([0., 1.05])


class Critic(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(10, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, state, action):
        return self.head(self.features(torch.cat((state, action), dim=-1)))


@dataclass
class PopArt:
    mean: float = 0.
    second: float = 1.
    std: float = 1.


class IntervalController:
    def __init__(self, config: ControllerConfig, seed: int = 0):
        config.validate()
        self.config = config
        # Small networks run on CPU, independent of vision training device.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = Actor(config.hidden)
            self.low = Critic(config.hidden)
            self.high = Critic(config.hidden)
        self.target_low, self.target_high = deepcopy(self.low), deepcopy(self.high)
        self.target_low.requires_grad_(False)
        self.target_high.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.low.parameters()) + list(self.high.parameters()),
                                                 lr=config.critic_lr)
        self.popart = PopArt()
        self.replay = deque(maxlen=config.replay_capacity)
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0

    def normalize_state(self, raw):
        state = torch.as_tensor(raw, dtype=torch.float32, device="cpu").clone()
        state[..., 5] = (state[..., 5] - self.popart.mean) / self.popart.std
        state[..., 6] /= self.popart.std
        return state

    def actor_action(self, state):
        action = self.actor(state)
        if self.config.disable_utility:
            action = torch.stack((torch.zeros_like(action[..., 0]), action[..., 1]), dim=-1)
        return action

    @torch.no_grad()
    def update_statistics(self, midpoint: float):
        """Preserve raw-state actions and denormalized Q predictions under PopArt.

        Reparameterize state input columns too because two state coordinates
        use these same running statistics. Replay stores raw states/rewards.
        """
        if not math.isfinite(midpoint):
            raise ValueError("PopArt midpoint must be finite.")
        old_mean, old_std = self.popart.mean, self.popart.std
        beta = self.config.popart_beta
        self.popart.mean = (1 - beta) * old_mean + beta * midpoint
        self.popart.second = (1 - beta) * self.popart.second + beta * midpoint**2
        self.popart.std = max(self.config.popart_min_std,
                             math.sqrt(max(0., self.popart.second - self.popart.mean**2)))
        new_mean, new_std = self.popart.mean, self.popart.std
        for network in (self.actor, self.low, self.high, self.target_low, self.target_high):
            first = network.features[0]
            first.bias.add_(first.weight[:, 5] * ((new_mean - old_mean) / old_std))
            first.weight[:, 5:7].mul_(new_std / old_std)
            if isinstance(network, Critic):
                network.head.weight.mul_(old_std / new_std)
                network.head.bias.mul_(old_std).add_(old_mean - new_mean).div_(new_std)
            # Adam moments are in the previous parameter coordinates. Reset only
            # transformed layers, retaining optimizer state for all other layers.
            optimizer = self.actor_optimizer if network is self.actor else self.critic_optimizer
            transformed = list(first.parameters())
            if isinstance(network, Critic):
                transformed += list(network.head.parameters())
            for parameter in transformed:
                optimizer.state.pop(parameter, None)

    @torch.no_grad()
    def select_action(self, raw_state, macro_index: int):
        state = self.normalize_state(raw_state).unsqueeze(0)
        candidate = self.actor_action(state)[0]
        if self.config.exploration_std:
            candidate += torch.tensor(self.rng.normal(0, self.config.exploration_std, 2), dtype=candidate.dtype)
            candidate = candidate.maximum(candidate.new_tensor([0., 1.05])).minimum(candidate.new_tensor([.5, 1.8]))
            if self.config.disable_utility:
                candidate[0] = 0
        conservative = torch.tensor(CONSERVATIVE_ACTION).unsqueeze(0)
        lower = float(self.low(state, candidate.unsqueeze(0)))
        upper_reference = float(self.high(state, conservative))
        warmup = macro_index < self.config.warmup_macros or self.train_steps == 0
        accepted = not warmup and lower >= upper_reference - self.config.tolerance
        if accepted:
            # float32(1.05) is slightly smaller than the Python float endpoint.
            # Canonicalize returned scalars without weakening domain validation.
            action = [min(.5, max(0., float(candidate[0]))), min(1.8, max(1.05, float(candidate[1])))]
        else:
            action = list(CONSERVATIVE_ACTION)
        validate_action(action)
        return action, {"candidate": candidate.tolist(), "candidate_low": lower,
                        "reference_high": upper_reference, "fallback": not accepted,
                        "gate_reason": "warmup" if warmup else ("accepted" if accepted else "value_gap")}

    def remember(self, state, executed_action, low, high, next_state, done):
        validate_action(executed_action)
        if low > high or not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError("Replay reward must be an ordered finite interval.")
        self.replay.append((state.clone().cpu(), list(executed_action), float(low),
                            float(high), next_state.clone().cpu(), bool(done)))

    def train(self):
        if len(self.replay) < self.config.batch_size:
            return {}
        result = {}
        for _ in range(self.config.updates_per_macro):
            ids = self.rng.choice(len(self.replay), self.config.batch_size, replace=False)
            states, actions, lows, highs, next_states, dones = zip(*(self.replay[i] for i in ids))
            s, ns = self.normalize_state(torch.stack(states)), self.normalize_state(torch.stack(next_states))
            a = torch.tensor(actions, dtype=torch.float32)
            low_r = (torch.tensor(lows).unsqueeze(1) - self.popart.mean) / self.popart.std
            high_r = (torch.tensor(highs).unsqueeze(1) - self.popart.mean) / self.popart.std
            continuation = 1 - torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            # Manuscript uses current pi_phi and target critics, without gate in the TD target.
            with torch.no_grad():
                na = self.actor_action(ns)
                y_low = low_r + self.config.gamma * continuation * self.target_low(ns, na)
                y_high = high_r + self.config.gamma * continuation * self.target_high(ns, na)
                if self.config.td_mode == "consistent":
                    # Q_raw = std * Q_normalized + mean. Explicit audited variant.
                    correction = self.config.gamma * continuation * self.popart.mean / self.popart.std
                    y_low += correction
                    y_high += correction
            loss_low = nn.functional.mse_loss(self.low(s, a), y_low)
            loss_high = nn.functional.mse_loss(self.high(s, a), y_high)
            critic_loss = loss_low + loss_high
            if not torch.isfinite(critic_loss):
                raise FloatingPointError("Nonfinite critic loss.")
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(self.low.parameters()) + list(self.high.parameters()), 10.)
            self.critic_optimizer.step()
            self.low.requires_grad_(False)
            self.high.requires_grad_(False)
            omega = (self.config.omega0 + self.config.kappa * s[:, 6:7]).clamp(.5, 1.)
            policy_action = self.actor_action(s)
            actor_loss = -(omega * self.low(s, policy_action) + (1 - omega) * self.high(s, policy_action)).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Nonfinite actor loss.")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_optimizer.step()
            self.low.requires_grad_(True)
            self.high.requires_grad_(True)
            with torch.no_grad():
                for target, online in ((self.target_low, self.low), (self.target_high, self.high)):
                    for dst, src in zip(target.parameters(), online.parameters()):
                        dst.lerp_(src, self.config.target_rate)
            self.train_steps += 1
            result = {"critic_loss_low": float(loss_low.detach()), "critic_loss_high": float(loss_high.detach()),
                      "actor_loss": float(actor_loss.detach()), "omega": float(omega.mean()),
                      "train_steps": self.train_steps}
        return result

    def state_dict(self):
        names = ("actor", "low", "high", "target_low", "target_high", "actor_optimizer", "critic_optimizer")
        return {**{name: getattr(self, name).state_dict() for name in names},
                "config": asdict(self.config), "popart": asdict(self.popart),
                "replay": list(self.replay), "rng": self.rng.bit_generator.state,
                "train_steps": self.train_steps}

    def load_state_dict(self, state):
        if state["config"] != asdict(self.config):
            raise ValueError("Checkpoint controller configuration differs from this run.")
        for name in ("actor", "low", "high", "target_low", "target_high", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.popart = PopArt(**state["popart"])
        self.replay = deque(state["replay"], maxlen=self.config.replay_capacity)
        self.rng.bit_generator.state = state["rng"]
        self.train_steps = state["train_steps"]
