"""SAC proposals with an optional, separate reward-only interval gate.

The two soft Q networks estimate the SAME midpoint-reward objective. Two other
networks fit interval endpoints without entropy. Their gate is a heuristic, not
a confidence guarantee. Both SAC variants train all four networks; only action
execution differs. Entropy is defined in normalized [-1,1]^2 action coordinates.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .controller import Critic
from .forward import CONSERVATIVE_ACTION, validate_action


@dataclass
class SACConfig:
    hidden: int = 64
    gamma: float = .9
    actor_lr: float = .0003
    critic_lr: float = .0003
    alpha_lr: float = .0003
    initial_alpha: float = .1
    target_entropy: float = -2.
    target_rate: float = .005
    batch_size: int = 16
    replay_capacity: int = 10000
    updates_per_macro: int = 4
    warmup_macros: int = 16
    tolerance: float = .05
    state_loss_scale: float = 5.
    use_gate: bool = False

    def validate(self):
        for name in ('hidden', 'batch_size', 'replay_capacity', 'updates_per_macro', 'warmup_macros'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == 'warmup_macros' else 1):
                raise ValueError(f'Invalid {name}.')
        if self.replay_capacity < self.batch_size or not 0 <= self.gamma < 1:
            raise ValueError('Invalid replay capacity or discount.')
        if not 0 < self.target_rate <= 1 or self.tolerance < 0:
            raise ValueError('Invalid target rate or tolerance.')
        if min(self.actor_lr, self.critic_lr, self.alpha_lr, self.initial_alpha, self.state_loss_scale) <= 0:
            raise ValueError('SAC rates/scales must be positive.')
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('SAC configuration must be finite.')


class GaussianActor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mean = nn.Linear(hidden, 2)
        self.log_std = nn.Linear(hidden, 2)
        nn.init.uniform_(self.mean.weight, -.003, .003)
        nn.init.zeros_(self.log_std.weight)
        with torch.no_grad():
            self.mean.bias.copy_(torch.tensor([math.atanh(-.76), math.atanh(-.2)]))
            self.log_std.bias.fill_(-.5)

    def sample(self, state, generator, deterministic=False):
        features = self.features(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(-5., 2.)
        noise = torch.zeros_like(mean) if deterministic else torch.randn(mean.shape, generator=generator)
        raw = mean + log_std.exp() * noise
        # Stable tanh Jacobian, even for highly saturated actions.
        log_prob = (-.5 * (noise.square() + 2*log_std + math.log(2*math.pi))
                    - 2*(math.log(2) - raw - F.softplus(-2*raw))).sum(-1, keepdim=True)
        unit = raw.tanh()
        physical = unit * unit.new_tensor([.25, .375]) + unit.new_tensor([.25, 1.425])
        return physical, log_prob


def backup_targets(midpoint, lower, upper, continuation, next_soft_q, next_log_prob,
                   next_low, next_high, gamma, alpha):
    """All returns are in raw loss-improvement units; only SAC Q includes entropy."""
    return (midpoint + gamma*continuation*(next_soft_q-alpha*next_log_prob),
            lower + gamma*continuation*next_low,
            upper + gamma*continuation*next_high)


class SACController:
    def __init__(self, config, seed=0):
        config.validate()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.actor = GaussianActor(config.hidden)
            self.q1, self.q2 = Critic(config.hidden), Critic(config.hidden)
            self.low, self.high = Critic(config.hidden), Critic(config.hidden)
        for name in ('q1', 'q2', 'low', 'high'):
            target = deepcopy(getattr(self, name)).requires_grad_(False)
            setattr(self, 'target_'+name, target)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        parameters = [p for name in ('q1', 'q2', 'low', 'high') for p in getattr(self, name).parameters()]
        self.critic_optimizer = torch.optim.Adam(parameters, lr=config.critic_lr)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config.initial_alpha)))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.action_rng = torch.Generator().manual_seed(seed+401)
        self.learning_rng = torch.Generator().manual_seed(seed+402)
        self.replay_rng = np.random.default_rng(seed+403)
        self.replay = deque(maxlen=config.replay_capacity)
        self.train_steps = 0

    def normalize_state(self, raw):
        state = torch.as_tensor(raw, dtype=torch.float32).clone()
        state[..., 5:7] /= self.config.state_loss_scale
        return state

    def update_statistics(self, midpoint):
        # Fixed input scaling; raw returns. No moving PopArt transform in SAC.
        if not math.isfinite(midpoint):
            raise ValueError('Nonfinite feedback.')

    @torch.no_grad()
    def select_action(self, raw_state, macro_index):
        state = self.normalize_state(raw_state).unsqueeze(0)
        candidate, log_prob = self.actor.sample(state, self.action_rng)
        conservative = candidate.new_tensor(CONSERVATIVE_ACTION).unsqueeze(0)
        lower = float(self.low(state, candidate))
        reference = float(self.high(state, conservative))
        would_accept = lower >= reference-self.config.tolerance
        warmup = macro_index < self.config.warmup_macros or self.train_steps == 0
        fallback = warmup or (self.config.use_gate and not would_accept)
        action = list(CONSERVATIVE_ACTION) if fallback else [
            min(.5, max(0., float(candidate[0, 0]))),
            min(1.8, max(1.05, float(candidate[0, 1])))]
        validate_action(action)
        reason = 'warmup' if warmup else ('accepted' if self.config.use_gate and would_accept else
                                        ('value_gap' if self.config.use_gate else 'sac_no_gate'))
        return action, {'candidate': candidate[0].tolist(), 'candidate_low': lower,
                        'reference_high': reference, 'gate_margin': lower-reference+self.config.tolerance,
                        'gate_value_units': 'raw_loss_return_without_entropy',
                        'gate_would_accept': would_accept, 'fallback': fallback, 'gate_reason': reason,
                        'proposal_log_prob_unit_action': float(log_prob), 'sac_alpha': float(self.log_alpha.exp())}

    def remember(self, state, executed_action, low, high, next_state, done):
        validate_action(executed_action)
        if low > high or not math.isfinite(low+high):
            raise ValueError('Reward must be a finite ordered interval.')
        self.replay.append((state.clone().cpu(), list(executed_action), float(low), float(high),
                            next_state.clone().cpu(), bool(done)))

    def train(self):
        if len(self.replay) < self.config.batch_size:
            return {}
        result = {}
        critics = [self.q1, self.q2, self.low, self.high]
        for _ in range(self.config.updates_per_macro):
            ids = self.replay_rng.choice(len(self.replay), self.config.batch_size, replace=False)
            states, actions, lows, highs, next_states, dones = zip(*(self.replay[i] for i in ids))
            s, ns = self.normalize_state(torch.stack(states)), self.normalize_state(torch.stack(next_states))
            a = torch.tensor(actions, dtype=torch.float32)
            lower, upper = torch.tensor(lows).unsqueeze(1), torch.tensor(highs).unsqueeze(1)
            midpoint = (lower+upper)/2
            continuation = 1-torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                na, logp = self.actor.sample(ns, self.learning_rng)
                soft, low, high = backup_targets(
                    midpoint, lower, upper, continuation,
                    torch.minimum(self.target_q1(ns, na), self.target_q2(ns, na)), logp,
                    self.target_low(ns, na), self.target_high(ns, na),
                    self.config.gamma, self.log_alpha.exp())
            losses = [F.mse_loss(net(s, a), target) for net, target in zip(critics, (soft, soft, low, high))]
            loss = sum(losses)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite SAC critic loss.')
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip independently: wide interval losses must not rescale SAC Q gradients.
            for net in critics:
                nn.utils.clip_grad_norm_(net.parameters(), 10.)
            self.critic_optimizer.step()
            for net in critics:
                net.requires_grad_(False)
            pa, logp = self.actor.sample(s, self.learning_rng)
            actor_loss = (self.log_alpha.exp().detach()*logp-torch.minimum(self.q1(s, pa), self.q2(s, pa))).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError('Nonfinite SAC actor loss.')
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_optimizer.step()
            for net in critics:
                net.requires_grad_(True)
            alpha_loss = -(self.log_alpha*(logp.detach()+self.config.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                for name in ('q1', 'q2', 'low', 'high'):
                    for target, online in zip(getattr(self, 'target_'+name).parameters(), getattr(self, name).parameters()):
                        target.lerp_(online, self.config.target_rate)
            self.train_steps += 1
            result = {name: float(value.detach()) for name, value in zip(
                ('sac_q1_loss', 'sac_q2_loss', 'critic_loss_low', 'critic_loss_high'), losses)}
            result.update(actor_loss=float(actor_loss.detach()), sac_alpha=float(self.log_alpha.exp().detach()),
                          entropy_unit_action=float(-logp.detach().mean()), train_steps=self.train_steps)
        return result

    def state_dict(self):
        names = ('actor', 'q1', 'q2', 'low', 'high', 'target_q1', 'target_q2', 'target_low', 'target_high',
                 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer')
        return {**{name: getattr(self, name).state_dict() for name in names},
                'config': asdict(self.config), 'log_alpha': self.log_alpha.detach().clone(),
                'action_rng': self.action_rng.get_state(), 'learning_rng': self.learning_rng.get_state(),
                'replay_rng': self.replay_rng.bit_generator.state, 'replay': list(self.replay),
                'train_steps': self.train_steps}

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('Checkpoint SAC configuration differs.')
        for name in ('actor', 'q1', 'q2', 'low', 'high', 'target_q1', 'target_q2', 'target_low', 'target_high',
                     'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        self.action_rng.set_state(state['action_rng'])
        self.learning_rng.set_state(state['learning_rng'])
        self.replay_rng.bit_generator.state = state['replay_rng']
        self.replay = deque(state['replay'], maxlen=self.config.replay_capacity)
        self.train_steps = state['train_steps']
