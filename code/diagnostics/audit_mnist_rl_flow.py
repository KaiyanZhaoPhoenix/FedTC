"""Trace FedTC actions from upload weights through paired gains to policy decisions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from algorithms.fedtc.forward import SIGMA, macro_state
from diagnostics.run_paired_mnist import FEATURES, HORIZON, SCENARIOS, PairedMNIST, make_world, next_state
from diagnostics.run_rollout_q_mnist import (
    BASELINE, CANDIDATE, DISCOUNT, MAX_MACROS, Gains, context, observed_gain, summarize, verify_worlds,
)
from utils.experiment import append_json, model_fingerprint, set_vector, vector, write_json

GEOMETRY = ("cutoff_band_fraction", "cutoff_band_baseline_weight", "weight_total_variation")
ACTIONS = (BASELINE, CANDIDATE)
CUTS = tuple(SIGMA * math.atanh(1 / tau) for tau in ACTIONS)
STRATEGIES = ("fixed_125", "fixed_135", "round_only", "geometry_5", "geometry_15")
EFFECT = .001


class AuditedMNIST(PairedMNIST):
    def aggregation_audit(self, result):
        z = result.deviations
        band = (z >= CUTS[1]) & (z < CUTS[0])
        weights = []
        for tau, cutoff in zip(ACTIONS, CUTS):
            credibility = (1 - tau * torch.tanh(z / SIGMA)).clamp_min(0)
            credibility[z >= cutoff] = 0
            weights.append(credibility / credibility.sum())
        return {
            "normalized_distance": z.tolist(),
            "executed_weights": result.weights.tolist(),
            "baseline_weights_on_uploads": weights[0].tolist(),
            "candidate_weights_on_uploads": weights[1].tolist(),
            GEOMETRY[0]: float(band.float().mean()),
            GEOMETRY[1]: float(weights[0][band].sum()),
            GEOMETRY[2]: float((weights[0] - weights[1]).abs().sum() / 2),
        }


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_audited_world(config, seed, scenario, folder):
    world = make_world(config, seed, scenario, folder, world_type=AuditedMNIST)
    sizes = np.array([len(client.validation_indices) for client in world.clients])
    return world, sizes / sizes.sum()


def initial_state():
    return macro_state([], 0., 0., 0.).numpy()


def input_features(state, losses, weights, previous):
    geometry = [float(np.mean([row[key] for row in previous.rounds])) if previous else 0.
                for key in GEOMETRY]
    return context(state, losses, weights) + geometry


def require_matched(left, right):
    if ([(r["round"], r["upload_clients"], r["sgd_seeds"]) for r in left.rounds] !=
            [(r["round"], r["upload_clients"], r["sgd_seeds"]) for r in right.rounds]):
        raise RuntimeError("Paired branches used different random schedules.")


def compare_branches(base, candidate, weights):
    require_matched(base, candidate)
    values = base.losses - candidate.losses
    return {
        "gain": float(weights @ values),
        "gains_by_client": values.tolist(),
        "baseline_losses_by_client": base.losses.tolist(),
        "candidate_losses_by_client": candidate.losses.tolist(),
        "model_parameter_l2_gap": float((base.parameters - candidate.parameters).norm()),
        "baseline_model_sha256": base.model_sha256,
        "candidate_model_sha256": candidate.model_sha256,
        "baseline_rounds": base.rounds,
        "candidate_rounds": candidate.rounds,
    }


@dataclass
class Visit:
    start: torch.Tensor
    branch: object
    row: dict


def collect(config, seed, scenario, folder, prior, evaluate):
    world, weights = make_audited_world(config, seed, scenario, folder)
    state, losses, previous = initial_state(), world.validation_losses(), None
    rng = np.random.default_rng(np.random.SeedSequence([seed, 831]))
    visits = []
    for start in range(0, world.global_rounds, HORIZON):
        features = input_features(state, losses, weights, previous)
        score = prior.score("state_5", start, features[:10])
        explore = not evaluate and rng.random() < .25
        tau = ACTIONS[int(rng.integers(2))] if explore else (CANDIDATE if score > 0 else BASELINE)
        parameters = vector(world.global_model)
        start_hash = model_fingerprint(world.global_model)
        branch = world.run_branch(parameters, start, tau)
        row = {
            "round_start": start, "round_end": start + HORIZON, "context": features,
            "start_model_sha256": start_hash, "tau": tau, "exploration": explore,
            "prior_score_5": score, "prior_score_15": prior.score("state_15", start, features[:10]),
            "start_validation_losses": losses.tolist(), "validation_loss": float(weights @ branch.losses),
            "model_sha256": branch.model_sha256, "rounds": branch.rounds,
        }
        if evaluate:
            row["test"] = world.test_accuracy()
        append_json(folder / "behavior.jsonl", row)
        visits.append(Visit(parameters, branch, row))
        state = next_state(branch, losses, start + HORIZON, world.global_rounds)
        losses, previous = branch.losses, branch

    samples = []
    for i, visit in enumerate(visits):
        start = visit.row["round_start"]
        main_tau = visit.row["tau"]
        branches = {main_tau: visit.branch}
        other = BASELINE if main_tau == CANDIDATE else CANDIDATE
        branches[other] = world.run_branch(visit.start, start, other)
        if branches[BASELINE].rounds[0]["updates_sha256"] != branches[CANDIDATE].rounds[0]["updates_sha256"]:
            raise RuntimeError("First-round uploads differ across the paired branches.")
        on_behavior = {tau: tau == main_tau for tau in ACTIONS}
        endpoints = []
        for j in range(i, min(i + MAX_MACROS, len(visits))):
            if j != i:
                for tau in ACTIONS:
                    on_behavior[tau] = on_behavior[tau] and visits[j].row["tau"] == BASELINE
                    branches[tau] = (visits[j].branch if on_behavior[tau] else
                                     world.run_branch(branches[tau].parameters, j * HORIZON, BASELINE))
            endpoint = compare_branches(branches[BASELINE], branches[CANDIDATE], weights)
            endpoints.append({"round_end": (j + 1) * HORIZON, **endpoint})
        sample = {
            **visit.row, "gain_5": endpoints[0]["gain"],
            "gain_15": observed_gain([step["gain"] for step in endpoints]),
            "endpoints": endpoints,
        }
        append_json(folder / "decisions.jsonl", sample)
        samples.append(sample)
    print(f"Audited {len(samples)} visited states, seed {seed}, {scenario}", flush=True)
    return samples


def narrow(samples):
    return [{**row, "context": row["context"][:10]} for row in samples]


def decisions(samples, prior, visited, geometry):
    result = []
    for row in samples:
        start, features = row["round_start"], row["context"]
        scores = {
            "fixed_only_5": prior.score("state_5", start, features[:10]),
            "fixed_only_15": prior.score("state_15", start, features[:10]),
            "visited_5": visited.score("state_5", start, features[:10]),
            "visited_15": visited.score("state_15", start, features[:10]),
            "geometry_5": geometry.score("state_5", start, features),
            "geometry_15": geometry.score("state_15", start, features),
            "round_only": geometry.score("round_only", start, features),
        }
        result.append({"round_start": start, "actual_behavior_tau": row["tau"],
                       "gain_5": row["gain_5"], "gain_15": row["gain_15"], "predictions": scores})
    return result


def scorecard(rows):
    five = np.array([row["gain_5"] for row in rows])
    long = np.array([row["gain_15"] for row in rows])
    selected = {}
    for name in rows[0]["predictions"]:
        prediction = np.array([row["predictions"][name] for row in rows])
        choose_candidate = prediction > 0
        # Evaluate both horizons and the regret to each same-state oracle.
        entry = {"switches": int(choose_candidate.sum())}
        for key, target in (("5", five), ("15", long)):
            obtained = np.where(choose_candidate, target, 0.)
            entry[f"mean_gain_{key}"] = float(obtained.mean())
            entry[f"mean_regret_{key}"] = float((np.maximum(target, 0) - obtained).mean())
            entry[f"correct_sign_{key}"] = int((choose_candidate == (target > 0)).sum())
        selected[name] = entry
    return {
        "count": len(rows), "oracle_gain_5": float(np.maximum(five, 0).mean()),
        "oracle_gain_15": float(np.maximum(long, 0).mean()),
        "material_positive_15": int((long >= EFFECT).sum()),
        "material_negative_15": int((long <= -EFFECT).sum()),
        "sign_flips_5_to_15": int(((five > 0) != (long > 0)).sum()),
        "methods": selected,
    }


def evaluate_strategy(config, seed, scenario, folder, strategy, policy):
    world, weights = make_audited_world(config, seed, scenario, folder)
    state, losses, previous = initial_state(), world.validation_losses(), None
    rows = []
    for start in range(0, world.global_rounds, HORIZON):
        features = input_features(state, losses, weights, previous)
        score = None
        if strategy.startswith("fixed"):
            tau = BASELINE if strategy == "fixed_125" else CANDIDATE
        else:
            score = policy.score(strategy.replace("geometry", "state"), start, features)
            tau = CANDIDATE if score > 0 else BASELINE
        start_hash = model_fingerprint(world.global_model)
        start_vector = vector(world.global_model)
        branch = world.run_branch(start_vector, start, tau)
        comparison = None
        if strategy.startswith("geometry"):
            other = BASELINE if tau == CANDIDATE else CANDIDATE
            alternate = world.run_branch(start_vector, start, other)
            if branch.rounds[0]["updates_sha256"] != alternate.rounds[0]["updates_sha256"]:
                raise RuntimeError("Policy audit first-round uploads differ.")
            base, candidate = (branch, alternate) if tau == BASELINE else (alternate, branch)
            comparison = compare_branches(base, candidate, weights)
            set_vector(world.global_model, branch.parameters)
        state = next_state(branch, losses, start + HORIZON, world.global_rounds)
        row = {"round_start": start, "round_end": start + HORIZON, "context": features,
               "start_model_sha256": start_hash, "tau": tau, "prediction": score,
               "paired_5_round_outcome": comparison, "rounds": branch.rounds,
               "model_sha256": branch.model_sha256, "validation_loss": float(weights @ branch.losses),
               "test": world.test_accuracy()}
        append_json(folder / "trajectory.jsonl", row)
        rows.append(row)
        losses, previous = branch.losses, branch
    print(f"Evaluated {strategy}, seed {seed}, {scenario}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--eval-seed", default=4, type=int)
    parser.add_argument("--jobs", default=6, type=int)
    args = parser.parse_args()
    source = args.source.resolve()
    previous = json.loads((source / "plan.json").read_text())
    seeds = previous["train_seeds"]
    if args.eval_seed in (*seeds, previous["eval_seed"]) or args.eval_seed < 0 or args.jobs < 1:
        parser.error("Use a fresh held-out seed and at least one job.")
    config = {key: previous[key] for key in ("clients", "rounds", "data_root")}
    prior_samples = []
    hashes = {}
    for seed in seeds:
        for scenario in SCENARIOS:
            path = source / "probes" / str(seed) / scenario / "action_values.jsonl"
            hashes[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
            prior_samples.extend(read_rows(path))
    prior = Gains(prior_samples)
    started = time.monotonic()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", {
        **config, "train_seeds": seeds, "eval_seed": args.eval_seed,
        "scenarios": SCENARIOS, "actions": ACTIONS, "cutoffs": CUTS, "discount": DISCOUNT,
        "state": [*FEATURES, "weighted_validation_loss", "weighted_validation_loss_sd", *GEOMETRY],
        "behavior": "Prior 5-round policy, with 25% exploration for calibration trajectories only.",
        "paired_continuation": BASELINE, "fixed_calibration_source": str(source),
        "fixed_calibration_sha256": hashes, "descriptive_gain_threshold": EFFECT,
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in (Path(__file__).name, "run_paired_mnist.py", "run_rollout_q_mnist.py")},
    })
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {(seed, scenario): pool.submit(collect, config, seed, scenario,
                                                  root / "visits" / str(seed) / scenario,
                                                  prior, seed == args.eval_seed)
                   for seed in (*seeds, args.eval_seed) for scenario in SCENARIOS}
        calibration = [row for seed in seeds for scenario in SCENARIOS
                       for row in futures[seed, scenario].result()]
        fitted = Gains(calibration)
        visited = Gains(narrow(calibration))
        write_json(root / "scorer.json", {
            "center": fitted.center.tolist(), "scale": fitted.scale.tolist(),
            "coefficients": fitted.weights.tolist(), "gain_15_by_round": fitted.by_round,
        })
        jobs = {(scenario, name): pool.submit(evaluate_strategy, config, args.eval_seed, scenario,
                                               root / "rollouts" / scenario / name, name, fitted)
                for scenario in SCENARIOS for name in STRATEGIES}
        report = {}
        for scenario in SCENARIOS:
            probe = root / "visits" / str(args.eval_seed) / scenario
            samples = futures[args.eval_seed, scenario].result()
            predictions = decisions(samples, prior, visited, fitted)
            write_json(probe / "prediction_audit.json", predictions)
            report[scenario] = {
                "heldout_decisions": scorecard(predictions),
                "rollouts": {"fixed_only_5": summarize(samples)},
            }
            for name in STRATEGIES:
                rows = jobs[scenario, name].result()
                verify_worlds(probe, root / "rollouts" / scenario / name)
                for sample, row in zip(samples, rows):
                    if ([(r["upload_clients"], r["sgd_seeds"]) for r in sample["rounds"]] !=
                            [(r["upload_clients"], r["sgd_seeds"]) for r in row["rounds"]]):
                        raise RuntimeError(f"Unmatched control schedule: {scenario}/{name}")
                report[scenario]["rollouts"][name] = summarize(rows)
    write_json(root / "summary.json", {"scenarios": report, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
