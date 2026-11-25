"""Five-round paired-action pilot on the native FedAA MNIST partition.

Run from code/: python -m diagnostics.run_paired_mnist --output runs/paired_mnist
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import json
from pathlib import Path
import time

import numpy as np
import torch

from algorithms.fedtc.forward import CONSERVATIVE_ACTION, calibrate, macro_state
from main import parser as training_parser, prepare
from servers.serverShared import SharedServer
from utils.experiment import append_json, fingerprint, model_fingerprint, set_vector, vector, write_json

CANDIDATES = (1.25, CONSERVATIVE_ACTION[1])
PROBE = 1.15
REFERENCE = CANDIDATES[-1]
HORIZON = 5
SCENARIOS = ("none", "delta_sign_flip")
FEATURES = ("log_mean_scale", "relative_distance_iqr", "fraction_distance_above_2",
            "utility_distance_rank_correlation", "fraction_positive_weight",
            "previous_validation_mean_improvement", "previous_validation_improvement_sd", "progress")
POLICIES = {"fixed": None, "round_only": None, "bandit": 0., "q_learning": .9}
RIDGE = 8.
EXPLORATION = .2


@dataclass
class Branch:
    parameters: torch.Tensor
    losses: np.ndarray
    geometries: list
    rounds: list
    model_sha256: str


class PairedMNIST(SharedServer):
    def __init__(self, args):
        super().__init__(args)
        # The native MNIST MLP has no persistent buffers. Plain SGD carries no
        # momentum, and scheduled learning-rate decay is disabled in the pilot.
        if list(self.global_model.buffers()) or args.learning_rate_decay:
            raise ValueError("This diagnostic requires parameter-only MNIST with a fixed local LR.")
        if any(group["momentum"] for client in self.clients for group in client.optimizer.param_groups):
            raise ValueError("This diagnostic requires stateless local SGD.")
        self.schedule = [
            self.cohort_rng.choice(self.num_clients, self.num_sel_clients, replace=False).tolist()
            for _ in range(self.global_rounds)
        ]

    def validation_losses(self):
        return np.array([client.bounded_loss(self.global_model, self.args.loss_cap)
                         for client in self.clients])

    def aggregation_audit(self, result):
        return {}

    def run_branch(self, start, round_start, tau):
        set_vector(self.global_model, start)
        geometries, rounds = [], []
        for index in range(round_start, round_start + HORIZON):
            cohort = [self.clients[i] for i in self.schedule[index]]
            self.send_models()
            parameters = vector(self.global_model)
            utilities = self.train_clients(cohort, index, parameters)
            updates = torch.stack([vector(client.model) - parameters for client in cohort])
            result = calibrate(updates, updates.new_tensor(utilities), (0., tau))
            set_vector(self.global_model, parameters + result.delta)
            geometries.append(result.geometry)
            rounds.append({
                "round": index + 1,
                "upload_clients": self.schedule[index],
                "sgd_seeds": [self.seed + 100003 * index + 97 * client.id for client in cohort],
                "updates_sha256": hashlib.sha256(updates.numpy().tobytes()).hexdigest(),
                "positive_weights": int((result.weights > 0).sum()),
                "weight_sum": float(result.weights.sum()),
                "cutoff_radius": result.cutoff_radius,
                **self.aggregation_audit(result),
            })
        if any(client.optimizer.state for client in self.clients):
            raise RuntimeError("Local optimizer unexpectedly acquired persistent state.")
        return Branch(vector(self.global_model), self.validation_losses(), geometries, rounds,
                      model_fingerprint(self.global_model))

    def paired_branches(self, start_round, actions):
        start = vector(self.global_model)
        starting_hash = model_fingerprint(self.global_model)
        branches = {tau: self.run_branch(start, start_round, tau) for tau in actions}
        reference = branches[REFERENCE]
        # All first-round uploads must be byte-identical. Later updates diverge
        # because the global parameters differ, but use the same RNG schedule.
        for branch in branches.values():
            if branch.rounds[0]["updates_sha256"] != reference.rounds[0]["updates_sha256"]:
                raise RuntimeError("First-round uploads differ from the same snapshot.")
            if [(r["upload_clients"], r["sgd_seeds"]) for r in branch.rounds] != [
                    (r["upload_clients"], r["sgd_seeds"]) for r in reference.rounds]:
                raise RuntimeError("Paired branch schedules differ.")
        return starting_hash, branches

    def test_accuracy(self):
        self.send_models()
        metrics = [client.test_metrics() for client in self.clients]
        by_client = sorted(correct / total for correct, total, _ in metrics)
        worst_count = math.ceil(.1 * len(by_client))
        return {
            "accuracy": sum(row[0] for row in metrics) / sum(row[1] for row in metrics),
            "worst10": float(np.mean(by_client[:worst_count])),
            "worst_client_count": worst_count,
        }


def next_state(branch, previous_losses, round_end, total):
    improvements = previous_losses - branch.losses
    mean, sd = float(improvements.mean()), float(improvements.std())
    return macro_state(branch.geometries, mean, mean + sd, round_end / total).numpy()


def make_world(config, seed, attack, folder, world_type=PairedMNIST):
    args = training_parser().parse_args([
        "--algorithm", "FedTCFixed", "--global_rounds", str(config["rounds"]),
        "--num_clients", str(config["clients"]), "--num_sel_clients", str(config["clients"]),
        "--aggre_num", str(config["clients"]), "--dataset", "mnist", "--device", "cpu",
        "--local_epochs", "1", "--batch_size", "16", "--macro-length", str(HORIZON),
        "--seed", str(seed), "--attack", attack, "--ad_ratio", ".2" if attack != "none" else "0",
        "--output", str(folder), "--data-root", config["data_root"],
    ])
    return world_type(prepare(args))


def branch_log(branch, start_losses, reference_losses):
    gains = reference_losses - branch.losses
    return {
        "end_model_sha256": branch.model_sha256,
        "end_validation_losses": branch.losses.tolist(),
        "paired_client_gains": gains.tolist(),
        "paired_reward": float(gains.mean()),
        "absolute_validation_improvement": float((start_losses - branch.losses).mean()),
        "rounds": branch.rounds,
    }


def collect(config, seed, attack, folder, measure_test):
    world = make_world(config, seed, attack, folder)
    state = macro_state([], 0., 0., 0.).numpy()
    losses = world.validation_losses()
    records = []
    for index in range(0, world.global_rounds, HORIZON):
        start_hash, branches = world.paired_branches(index, (*CANDIDATES, PROBE))
        executed = branches[REFERENCE]
        following = next_state(executed, losses, index + HORIZON, world.global_rounds)
        record = {
            "round_start": index, "round_end": index + HORIZON, "state": state.tolist(),
            "start_model_sha256": start_hash, "start_validation_losses": losses.tolist(),
            "executed_tau": REFERENCE, "next_state": following.tolist(),
            "branches": {str(tau): branch_log(branch, losses, executed.losses)
                         for tau, branch in branches.items()},
        }
        set_vector(world.global_model, executed.parameters)
        if measure_test:
            record["test"] = world.test_accuracy()
        append_json(folder / "macros.jsonl", record)
        records.append(record)
        state, losses = following, executed.losses
    print(f"Paired probes complete: seed={seed} scenario={attack} ({len(records)} snapshots)", flush=True)
    return records


def fit_ridge(features, targets):
    penalty = np.eye(features.shape[1]) * RIDGE
    penalty[0, 0] = 0.
    return np.linalg.solve(features.T @ features + penalty, features.T @ targets)


class ActionValues:
    def __init__(self, training, gamma):
        states = np.array([row["state"] for row in training])
        self.center = states.mean(axis=0)
        self.scale = states.std(axis=0).clip(min=1e-6)
        targets = np.array([[reward(row, tau) for tau in CANDIDATES] for row in training])
        self.prior = fit_ridge(self.features(states), targets)
        self.correction = np.zeros_like(self.prior)
        self.gamma = gamma
        self.replay = []

    def features(self, states):
        normalized = ((np.atleast_2d(states) - self.center) / self.scale).clip(-3., 3.)
        return np.column_stack((np.ones(len(normalized)), normalized))

    def values(self, state):
        return (self.features(state) @ (self.prior + self.correction))[0]

    def learn(self, transition):
        self.replay.append(transition)
        previous = self.prior + self.correction
        correction = np.zeros_like(self.correction)
        for action in range(len(CANDIDATES)):
            rows = [row for row in self.replay if row["action"] == action]
            if not rows:
                continue
            states = self.features([row["state"] for row in rows])
            following = self.features([row["next_state"] for row in rows])
            continuation = np.array([not row["done"] for row in rows])
            targets = np.array([row["reward"] for row in rows])
            targets += self.gamma * continuation * (following @ previous).max(axis=1)
            # Shrink the online fit towards the paired one-step warm start.
            # This leaves the prior unchanged in unsupported directions.
            residuals = targets - states @ self.prior[:, action]
            correction[:, action] = np.linalg.solve(
                states.T @ states + RIDGE * np.eye(states.shape[1]), states.T @ residuals)
        self.correction = correction


def reward(record, tau):
    return record["branches"][str(tau)]["paired_reward"]


def choose(scores):
    # On a tie prefer the conservative action.
    return 0 if scores[0] > scores[1] else 1


def selections(training):
    moments = sorted({row["round_start"] for row in training})
    by_round = {
        index: CANDIDATES[choose([np.mean([reward(row, tau) for row in training
                                           if row["round_start"] == index])
                                  for tau in CANDIDATES])]
        for index in moments
    }
    fixed = CANDIDATES[choose([np.mean([reward(row, tau) for row in training])
                              for tau in CANDIDATES])]
    return fixed, by_round


def rollout(config, seed, attack, folder, name, training, fixed, by_round):
    world = make_world(config, seed, attack, folder)
    gamma = POLICIES[name]
    policy = ActionValues(training, gamma) if gamma is not None else None
    rng = np.random.default_rng(np.random.SeedSequence([seed, 710]))
    state = macro_state([], 0., 0., 0.).numpy()
    losses = world.validation_losses()
    records = []
    for index in range(0, world.global_rounds, HORIZON):
        values, exploring = None, False
        if policy is None:
            action = CANDIDATES.index(fixed if name == "fixed" else by_round[index])
        else:
            values = policy.values(state).tolist()
            exploring = rng.random() < EXPLORATION
            exploratory_action = int(rng.integers(len(CANDIDATES)))
            action = exploratory_action if exploring else choose(values)
        tau = CANDIDATES[action]
        actions = (REFERENCE,) if tau == REFERENCE else (REFERENCE, tau)
        start_hash, branches = world.paired_branches(index, actions)
        executed, reference = branches[tau], branches[REFERENCE]
        following = next_state(executed, losses, index + HORIZON, world.global_rounds)
        transition = {
            "state": state.tolist(), "action": action,
            "reward": float((reference.losses - executed.losses).mean()),
            "next_state": following.tolist(), "done": index + HORIZON == world.global_rounds,
        }
        if policy is not None:
            policy.learn(transition)
        set_vector(world.global_model, executed.parameters)
        record = {
            "round_start": index, "round_end": index + HORIZON, "start_model_sha256": start_hash,
            "executed_tau": tau, "predicted_values": values, "exploring": exploring,
            "transition": transition, "replay_size": len(policy.replay) if policy is not None else 0,
            "start_validation_losses": losses.tolist(),
            "branches": {str(t): branch_log(branch, losses, reference.losses)
                         for t, branch in branches.items()},
            "test": world.test_accuracy(),
        }
        append_json(folder / "macros.jsonl", record)
        records.append(record)
        state, losses = following, executed.losses
    if policy is not None:
        write_json(folder / "policy.json", {
            "gamma": gamma, "center": policy.center.tolist(), "scale": policy.scale.tolist(),
            "prior": policy.prior.tolist(), "online_correction": policy.correction.tolist(),
        })
    print(f"Rollout complete: {name} scenario={attack} accuracy={records[-1]['test']['accuracy']:.4f}",
          flush=True)
    return records


def heldout_comparison(records, training, fixed, by_round):
    policy = ActionValues(training, 0.)
    choices = {
        "fixed": [fixed] * len(records),
        "round_only": [by_round[row["round_start"]] for row in records],
        "state": [CANDIDATES[choose(policy.values(row["state"]))] for row in records],
        "oracle": [CANDIDATES[choose([reward(row, tau) for tau in CANDIDATES])] for row in records],
    }
    return {
        "fixed_1.15_probe": float(np.mean([reward(row, PROBE) for row in records])),
        **{name: {
            "mean_paired_gain": float(np.mean([reward(row, tau) for row, tau in zip(records, actions)])),
            "tau_1.25_count": actions.count(CANDIDATES[0]),
            "oracle_agreements": sum(tau == best for tau, best in zip(actions, choices["oracle"])),
            "selected_tau_by_round": dict(zip([row["round_start"] for row in records], actions)),
        } for name, actions in choices.items()}
    }


def summarize_rollout(records):
    end = records[-1]["round_end"]
    tail = [row for row in records if row["round_end"] > end - 50]
    return {
        "tail_evaluations": [row["round_end"] for row in tail],
        "tail_accuracy": float(np.mean([row["test"]["accuracy"] for row in tail])),
        "tail_worst10": float(np.mean([row["test"]["worst10"] for row in tail])),
        "final_accuracy": records[-1]["test"]["accuracy"],
        "final_worst10": records[-1]["test"]["worst10"],
        "tau_1.25_count": sum(row["executed_tau"] == CANDIDATES[0] for row in records),
        "exploratory_macros": sum(row.get("exploring", False) for row in records),
        "main_communication_rounds": end,
        "extra_shadow_rounds": HORIZON * sum(len(row["branches"]) - 1 for row in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clients", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    args = parser.parse_args()
    if not (2 <= args.clients <= 100 and args.rounds >= HORIZON and args.rounds % HORIZON == 0
            and args.jobs > 0 and 0 <= args.train_seed != args.eval_seed >= 0):
        parser.error("Use 2–100 clients, a positive multiple of five rounds, distinct seeds and positive jobs.")
    started = time.monotonic()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = {"clients": args.clients, "rounds": args.rounds, "data_root": str(args.data_root.resolve())}
    write_json(root / "plan.json", {
        **config, "train_seed": args.train_seed, "evaluation_seed": args.eval_seed,
        "scenarios": SCENARIOS, "actions": CANDIDATES, "diagnostic_only_tau": PROBE,
        "macro_rounds": HORIZON, "state_features": FEATURES, "online_exploration": EXPLORATION,
        "ridge": RIDGE, "policies": POLICIES,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reward": "Macro client mean of reference-end minus executed-end capped private validation loss.",
        "test_scope": "All clients, including clients whose uploads were perturbed. Tests never select actions.",
    })
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        probes = {
            (seed, attack): pool.submit(collect, config, seed, attack, root / "probes" / str(seed) / attack,
                                        seed == args.eval_seed)
            for seed in (args.train_seed, args.eval_seed) for attack in SCENARIOS
        }
        paired = {key: job.result() for key, job in probes.items()}
        training = [row for attack in SCENARIOS for row in paired[args.train_seed, attack]]
        fixed, schedule = selections(training)
        write_json(root / "selection.json", {"fixed_tau": fixed, "round_only_tau": schedule,
                                              "training_probe_count": len(training)})
        ranking = {attack: heldout_comparison(paired[args.eval_seed, attack], training, fixed, schedule)
                   for attack in SCENARIOS}
        write_json(root / "heldout_ranking.json", ranking)
        outcomes = {
            (attack, name): pool.submit(rollout, config, args.eval_seed, attack,
                                        root / "rollouts" / attack / name, name, training, fixed, schedule)
            for attack in SCENARIOS for name in POLICIES
        }
        summaries = {}
        for attack in SCENARIOS:
            reference = paired[args.eval_seed, attack]
            source = root / "probes" / str(args.eval_seed) / attack / "manifest.json"
            manifest = json.loads(source.read_text())
            expected = [r["upload_clients"] for record in reference
                        for r in record["branches"][str(REFERENCE)]["rounds"]]
            summaries[attack] = {"conservative": summarize_rollout(reference)}
            for name in POLICIES:
                records = outcomes[attack, name].result()
                path = root / "rollouts" / attack / name / "manifest.json"
                compared = json.loads(path.read_text())
                for key in ("native_data_sha256", "private_splits_sha256", "initial_model_sha256",
                            "malicious_clients_harness_only"):
                    if compared[key] != manifest[key]:
                        raise RuntimeError(f"{attack}/{name} has unmatched {key}.")
                actual = [r["upload_clients"] for record in records for r in
                          record["branches"][str(record["executed_tau"])]["rounds"]]
                if actual != expected or records[0]["start_model_sha256"] != manifest["initial_model_sha256"]:
                    raise RuntimeError(f"{attack}/{name} has unmatched initialization or uploads.")
                summaries[attack][name] = summarize_rollout(records)
    write_json(root / "summary.json", {
        "fixed_tau": fixed, "round_only_tau": schedule, "heldout_ranking": ranking,
        "rollouts": summaries, "calibration_sha256": fingerprint(training),
        "elapsed_seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    main()
