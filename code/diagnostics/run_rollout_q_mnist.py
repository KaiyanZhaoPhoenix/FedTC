"""Compare five- and fifteen-round action values against a fixed MNIST policy.

Run from code/: python -m diagnostics.run_rollout_q_mnist --output runs/rollout_q
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from algorithms.fedtc.forward import macro_state
from diagnostics.run_paired_mnist import (
    CANDIDATES, FEATURES, HORIZON, SCENARIOS, fit_ridge, make_world, next_state,
)
from utils.experiment import append_json, model_fingerprint, set_vector, vector, write_json

BASELINE, CANDIDATE = CANDIDATES
DISCOUNT = .9
MAX_MACROS = 3
STRATEGIES = ("round_only", "state_5", "state_15")


def context(state, losses, weights):
    mean = np.dot(weights, losses)
    deviation = np.sqrt(np.dot(weights, (losses - mean) ** 2))
    return [*state.tolist(), float(mean), float(deviation)]


def observed_gain(gains):
    return float(np.average(gains, weights=DISCOUNT ** np.arange(len(gains))))


def collect(config, seed, attack, folder, evaluate):
    world = make_world(config, seed, attack, folder)
    sizes = np.array([len(client.validation_indices) for client in world.clients])
    weights = sizes / sizes.sum()
    losses = world.validation_losses()
    state = macro_state([], 0., 0., 0.).numpy()
    snapshots, baseline = [], []
    for start in range(0, world.global_rounds, HORIZON):
        parameters = vector(world.global_model)
        snapshots.append((parameters, model_fingerprint(world.global_model),
                          context(state, losses, weights)))
        branch = world.run_branch(parameters, start, BASELINE)
        state = next_state(branch, losses, start + HORIZON, world.global_rounds)
        losses = branch.losses
        row = {"round_end": start + HORIZON, "validation_loss": float(weights @ losses),
               "model_sha256": branch.model_sha256}
        if evaluate:
            row["test"] = world.test_accuracy()
        baseline.append((branch, row))
        append_json(folder / "baseline.jsonl", row)

    samples = []
    for i, (parameters, start_hash, features) in enumerate(snapshots):
        start = i * HORIZON
        candidate = world.run_branch(parameters, start, CANDIDATE)
        if candidate.rounds[0]["updates_sha256"] != baseline[i][0].rounds[0]["updates_sha256"]:
            raise RuntimeError("The paired branches have different initial uploads.")
        gains, endpoints = [], []
        for j in range(i, min(i + MAX_MACROS, len(snapshots))):
            if j > i:
                candidate = world.run_branch(candidate.parameters, j * HORIZON, BASELINE)
            reference = baseline[j][0]
            candidate_schedule = [(r["upload_clients"], r["sgd_seeds"]) for r in candidate.rounds]
            reference_schedule = [(r["upload_clients"], r["sgd_seeds"]) for r in reference.rounds]
            if candidate_schedule != reference_schedule:
                raise RuntimeError("The candidate and baseline continuation schedules differ.")
            per_client = reference.losses - candidate.losses
            gains.append(float(weights @ per_client))
            endpoints.append({
                "round_end": (j + 1) * HORIZON,
                "candidate_model_sha256": candidate.model_sha256,
                "reference_model_sha256": reference.model_sha256,
                "client_gains": per_client.tolist(),
                "weighted_loss_gain": gains[-1],
                "candidate_rounds": candidate.rounds,
                "reference_rounds": reference.rounds,
            })
        sample = {"round_start": start, "context": features, "start_model_sha256": start_hash,
                  "gain_5": gains[0], "gain_15": observed_gain(gains),
                  "lookahead_rounds": len(gains) * HORIZON, "endpoints": endpoints}
        append_json(folder / "action_values.jsonl", sample)
        samples.append(sample)
    print(f"Measured action values: seed={seed}, scenario={attack}, n={len(samples)}", flush=True)
    return samples, [row for _, row in baseline]


class Gains:
    def __init__(self, samples):
        inputs = np.array([row["context"] for row in samples])
        self.center = inputs.mean(axis=0)
        self.scale = inputs.std(axis=0).clip(min=1e-6)
        features = self.features(inputs)
        targets = np.array([[row["gain_5"], row["gain_15"]] for row in samples])
        self.weights = fit_ridge(features, targets)
        rounds = sorted({row["round_start"] for row in samples})
        self.by_round = {
            index: float(np.mean([row["gain_15"] for row in samples if row["round_start"] == index]))
            for index in rounds
        }

    def features(self, inputs):
        values = ((np.atleast_2d(inputs) - self.center) / self.scale).clip(-3., 3.)
        return np.column_stack((np.ones(len(values)), values))

    def score(self, strategy, index, inputs):
        if strategy == "round_only":
            return self.by_round[index]
        return float((self.features(inputs) @ self.weights)[0, int(strategy == "state_15")])


def test_policy(config, seed, attack, folder, strategy, policy):
    world = make_world(config, seed, attack, folder)
    sizes = np.array([len(client.validation_indices) for client in world.clients])
    weights = sizes / sizes.sum()
    losses = world.validation_losses()
    state = macro_state([], 0., 0., 0.).numpy()
    records = []
    for start in range(0, world.global_rounds, HORIZON):
        features = context(state, losses, weights)
        predicted = policy.score(strategy, start, features)
        tau = CANDIDATE if predicted > 0 else BASELINE
        branch = world.run_branch(vector(world.global_model), start, tau)
        state = next_state(branch, losses, start + HORIZON, world.global_rounds)
        losses = branch.losses
        row = {
            "round_start": start, "round_end": start + HORIZON, "context": features,
            "predicted_gain": predicted, "tau": tau, "rounds": branch.rounds,
            "model_sha256": branch.model_sha256, "validation_loss": float(weights @ losses),
            "test": world.test_accuracy(),
        }
        append_json(folder / "trajectory.jsonl", row)
        records.append(row)
    print(f"Completed {strategy}: seed={seed}, scenario={attack}", flush=True)
    return records


def heldout_scores(samples, policy):
    targets = np.array([row["gain_15"] for row in samples])
    result = {"oracle_gain_15": float(np.maximum(targets, 0.).mean()),
              "first_vs_fifteenth_round_sign_changes": sum(
                  (row["gain_5"] > 0) != (row["gain_15"] > 0) for row in samples)}
    for strategy in STRATEGIES:
        predicted = np.array([policy.score(strategy, row["round_start"], row["context"])
                              for row in samples])
        selected = predicted > 0.
        result[strategy] = {"switches": int(selected.sum()),
                            "gain_15": float(np.where(selected, targets, 0.).mean()),
                            "correct_sign": int((selected == (targets > 0.)).sum())}
    return result


def summarize(records):
    final = records[-1]["round_end"]
    tail = [row for row in records if row["round_end"] > final - 50]
    return {
        "tail_accuracy": float(np.mean([row["test"]["accuracy"] for row in tail])),
        "tail_worst10": float(np.mean([row["test"]["worst10"] for row in tail])),
        "tail_private_validation_loss": float(np.mean([row["validation_loss"] for row in tail])),
        "final_accuracy": records[-1]["test"]["accuracy"],
        "switches": sum(row.get("tau", BASELINE) == CANDIDATE for row in records),
        "final_model_sha256": records[-1]["model_sha256"],
    }


def verify_worlds(reference, compared):
    ref = json.loads((reference / "manifest.json").read_text())
    actual = json.loads((compared / "manifest.json").read_text())
    keys = ("initial_model_sha256", "native_data_sha256", "private_splits_sha256",
            "malicious_clients_harness_only")
    if any(actual[key] != ref[key] for key in keys):
        raise RuntimeError(f"Comparison provenance differs in {compared}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--clients", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--eval-seed", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    seeds = [*args.train_seeds, args.eval_seed]
    if (len(set(seeds)) != len(seeds) or min(seeds) < 0 or args.jobs < 1
            or not 2 <= args.clients <= 100 or args.rounds < HORIZON * MAX_MACROS
            or args.rounds % HORIZON):
        parser.error("Use distinct nonnegative seeds, 2–100 clients, positive jobs and ≥15 rounds divisible by 5.")

    started = time.monotonic()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = {"clients": args.clients, "rounds": args.rounds, "data_root": str(args.data_root.resolve())}
    write_json(root / "plan.json", {
        **config, "train_seeds": args.train_seeds, "eval_seed": args.eval_seed, "scenarios": SCENARIOS,
        "baseline_tau": BASELINE, "candidate_tau": CANDIDATE, "lambda": 0.,
        "horizons": [HORIZON, HORIZON * MAX_MACROS], "discount": DISCOUNT,
        "state_features": [*FEATURES, "weighted_validation_loss", "weighted_validation_loss_sd"],
        "target": "Discounted mean sample-weighted private validation loss gain vs fixed baseline.",
        "strategy": "Frozen policy improvement against fixed baseline using measured continuations.",
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          (Path(__file__), Path(__file__).with_name("run_paired_mnist.py"))},
    })
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        probes = {(seed, attack): pool.submit(collect, config, seed, attack,
                                             root / "probes" / str(seed) / attack, seed == args.eval_seed)
                  for seed in seeds for attack in SCENARIOS}
        calibration = [row for seed in args.train_seeds for attack in SCENARIOS
                       for row in probes[seed, attack].result()[0]]
        policy = Gains(calibration)
        write_json(root / "policy.json", {
            "center": policy.center.tolist(), "scale": policy.scale.tolist(),
            "weights": policy.weights.tolist(), "long_gain_by_round": policy.by_round,
            "training_snapshots": len(calibration),
        })
        jobs = {(attack, strategy): pool.submit(test_policy, config, args.eval_seed, attack,
                                               root / "policies" / attack / strategy, strategy, policy)
                for attack in SCENARIOS for strategy in STRATEGIES}
        summary = {}
        for attack in SCENARIOS:
            samples, baseline = probes[args.eval_seed, attack].result()
            reference = root / "probes" / str(args.eval_seed) / attack
            policy_results = {"fixed_125": summarize(baseline)}
            for strategy in STRATEGIES:
                records = jobs[attack, strategy].result()
                verify_worlds(reference, root / "policies" / attack / strategy)
                for sample, row in zip(samples, records):
                    expected = sample["endpoints"][0]["reference_rounds"]
                    if [r["upload_clients"] for r in expected] != [r["upload_clients"] for r in row["rounds"]]:
                        raise RuntimeError(f"{attack}/{strategy} uploaded on a different schedule.")
                policy_results[strategy] = summarize(records)
            summary[attack] = {"heldout": heldout_scores(samples, policy), "policies": policy_results}
    write_json(root / "summary.json", {"scenarios": summary, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
