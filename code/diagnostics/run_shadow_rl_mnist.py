"""Evaluate rollout value learning from same-upload MNIST shadow models."""
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
from torch.nn import functional as F

from algorithms.fedtc.forward import EPS_SCALE, SIGMA, calibrate, validate_action
from diagnostics.run_paired_mnist import PairedMNIST, make_world
from utils.experiment import append_json, fingerprint, model_fingerprint, set_vector, vector, write_json

DISCOUNT = .9
REFERENCE = 1.25
EVAL_GAP = 5
METHODS = ("round_only", "greedy", "learned", "rollout")


@dataclass
class Shadows:
    start: torch.Tensor
    parameters: list
    losses: np.ndarray
    features: np.ndarray
    weights: np.ndarray
    upload_hash: str
    cohort: list


class ShadowMNIST(PairedMNIST):
    def __init__(self, args):
        super().__init__(args)
        # With lambda=0 the utility rank contributes zero to every credibility.
        # Reuse native local SGD without running the training-utility forwards.
        self.algorithm = "ShadowTauStudy"
        self.private = [self.stack(client._full_train_data, client.validation_indices)
                        for client in self.clients]
        self.sizes = np.array([len(y) for _, y in self.private])
        self.private_x = torch.cat([x for x, _ in self.private])
        self.private_y = torch.cat([y for _, y in self.private])
        self.sample_weights = self.sizes / self.sizes.sum()

    @staticmethod
    def stack(data, indices):
        x, y = zip(*(data[i] for i in indices))
        return torch.stack(x), torch.stack(y)

    @torch.no_grad()
    def private_losses(self, parameters):
        set_vector(self.global_model, parameters)
        losses = F.cross_entropy(self.global_model(self.private_x), self.private_y,
                                 reduction="none").clamp(max=self.args.loss_cap)
        return np.array([float(chunk.mean()) for chunk in losses.split(self.sizes.tolist())])

    def attack_schedule(self, scenario):
        if scenario == "none":
            return ["none"] * self.global_rounds
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 921]))
        blocks = [name for _ in range(math.ceil(self.global_rounds / 10))
                  for name in rng.permutation(["none", "delta_sign_flip"]).tolist()]
        return [name for name in blocks for _ in range(5)][:self.global_rounds]

    @torch.no_grad()
    def aggregate_options(self, updates, start, actions):
        reference = calibrate(updates, updates.new_zeros(len(updates)), (0., REFERENCE))
        values, weights = [], []
        slope = torch.tanh(reference.deviations / SIGMA)
        for tau in actions:
            credibility = (1 - tau * slope).clamp_min(0)
            cutoff = SIGMA * (reference.scale + EPS_SCALE) * math.atanh(1 / tau)
            credibility[reference.distances >= cutoff] = 0
            weight = credibility / credibility.sum()
            if tau == REFERENCE:
                torch.testing.assert_close(weight, reference.weights, rtol=0, atol=1e-7)
            values.append(start + updates.T.mv(weight))
            weights.append(weight.numpy())
        return values, np.array(weights), reference.geometry[:3].numpy()

    def collect_uploads(self, start, index, schedule):
        set_vector(self.global_model, start)
        self.attack = schedule[index]
        cohort = [self.clients[i] for i in self.schedule[index]]
        self.send_models()
        self.train_clients(cohort, index, start)
        return torch.stack([vector(client.model) - start for client in cohort])

    def shadows(self, start, index, actions, schedule):
        updates = self.collect_uploads(start, index, schedule)
        parameters, weights, geometry = self.aggregate_options(updates, start, actions)
        losses = np.array([self.private_losses(p) for p in parameters])
        ref = actions.index(REFERENCE)
        differences = losses[ref] - losses
        mean_gain = differences @ self.sample_weights
        spread = np.sqrt((differences - mean_gain[:, None]) ** 2 @ self.sample_weights)
        downsides = (differences < 0) @ self.sample_weights
        weight_tv = np.abs(weights - weights[ref]).sum(axis=1) / 2
        context = [index / self.global_rounds, float(losses[ref] @ self.sample_weights),
                   float(geometry[0]), float(geometry[1]), float(geometry[2])]
        features = np.column_stack((
            mean_gain, spread, downsides, weight_tv, weights.min(axis=1),
            np.repeat(np.array(context)[None, :], len(actions), axis=0),
        ))
        return Shadows(start, parameters, losses, features, weights,
                       hashlib.sha256(updates.numpy().tobytes()).hexdigest(), self.schedule[index])

    def record_test(self, params):
        set_vector(self.global_model, params)
        return self.test_accuracy()


def build_world(config, seed, scenario, folder):
    attack = "none" if scenario == "none" else "delta_sign_flip"
    world = make_world(config, seed, attack, folder, world_type=ShadowMNIST)
    schedule = world.attack_schedule(scenario)
    provenance = {"attack_schedule_harness_only": schedule, "upload_schedule_sha256": fingerprint(world.schedule),
                  "actions": config["actions"], "utility_weight": 0, "test_policy_input": False}
    write_json(folder / "study_manifest.json", provenance)
    return world, schedule


def continuation_values(world, shadow, index, config, schedule):
    actions = config["actions"]
    values = []
    for p, losses in zip(shadow.parameters, shadow.losses):
        accum = [float(losses @ world.sample_weights)]
        for j in range(index + 1, min(index + config["horizon"], world.global_rounds)):
            future = world.shadows(p, j, actions, schedule)
            cost = future.losses @ world.sample_weights
            best = int(cost.argmin())
            p = future.parameters[best]
            accum.append(float(cost[best]))
        values.append(accum)
    costs = np.array(values)
    discounts = DISCOUNT ** np.arange(costs.shape[1])
    return costs, costs @ discounts / discounts.sum()


def log_action(world, shadow, action, index, actions):
    ref = actions.index(REFERENCE)
    return {
        "round": index + 1, "tau": actions[action], "action": action,
        "upload_clients": shadow.cohort, "upload_sha256": shadow.upload_hash,
        "sgd_seeds": [world.seed + 100003 * index + 97 * i for i in shadow.cohort],
        "weights_by_action": shadow.weights.tolist(),
        "private_losses_by_action": shadow.losses.tolist(),
        "private_losses_by_client_vs_reference": (shadow.losses[ref] - shadow.losses).tolist(),
        "selected_validation_loss": float(shadow.losses[action] @ world.sample_weights),
        "features": shadow.features.tolist(),
    }


def collect(config, seed, scenario, folder):
    world, schedule = build_world(config, seed, scenario, folder)
    actions = config["actions"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 713]))
    current = vector(world.global_model)
    examples = []
    for index in range(world.global_rounds):
        shadow = world.shadows(current, index, actions, schedule)
        greedy = int(shadow.features[:, 0].argmax())
        exploring = rng.random() < .25
        action = int(rng.integers(len(actions))) if exploring else greedy
        row = {**log_action(world, shadow, action, index, actions), "exploring": exploring}
        if index % config["stride"] == 0:
            costs, discounted = continuation_values(world, shadow, index, config, schedule)
            ref = actions.index(REFERENCE)
            row["rollout_costs"] = costs.tolist()
            row["gain_long"] = (discounted[ref] - discounted).tolist()
            examples.append(row)
        append_json(folder / "decisions.jsonl", row)
        current = shadow.parameters[action]
    print(f"Calibration: {scenario} seed {seed}, {len(examples)} paired snapshots", flush=True)
    return examples


class GainModel:
    def __init__(self, examples, actions, regularization=12., blend=1., target_horizon=None):
        self.ref = actions.index(REFERENCE)
        self.blend = blend
        features = np.array([row["features"] for row in examples])
        gains = np.array([self.target(row, target_horizon) for row in examples])
        self.center = features.reshape(-1, features.shape[-1]).mean(axis=0)
        self.scale = features.reshape(-1, features.shape[-1]).std(axis=0).clip(min=1e-6)
        x = np.array([self.design(row) for row in features])
        # Lightly shrink each fitted correction toward the observed immediate gain.
        # Intercept is penalized too, so small calibration samples stay conservative.
        self.coefs = []
        for i in range(len(actions)):
            v = x[:, i, :]
            error = gains[:, i] - features[:, i, 0]
            regularizer = regularization * np.eye(v.shape[1])
            self.coefs.append(np.linalg.solve(v.T @ v + regularizer, v.T @ error))
        self.coefs = np.array(self.coefs)
        self.rounds = sorted({row["round"] for row in examples})
        self.round_gains = {
            r: np.mean([row["gain_long"] for row in examples if row["round"] == r], axis=0)
            for r in self.rounds
        }

    def design(self, features):
        z = ((features - self.center) / self.scale).clip(-3, 3)
        return np.column_stack((np.ones(len(z)), z))

    def values(self, features):
        gains = features[:, 0] + getattr(self, "blend", 1.) * np.sum(
            self.design(features) * self.coefs, axis=1)
        return gains - gains[self.ref]

    def target(self, row, horizon):
        if horizon is None:
            return row["gain_long"]
        costs = np.array(row["rollout_costs"])[:, :horizon]
        discount = DISCOUNT ** np.arange(costs.shape[1])
        value = costs @ discount / discount.sum()
        return value[self.ref] - value


def tune_model(groups, actions, horizon):
    """Tune only on held-out calibration seeds, never on test trajectories."""
    if len(groups) < 2:
        raise ValueError("Tuning requires at least two calibration seeds.")
    candidates = []
    for steps in sorted({2, horizon}):
        for ridge in (4., 12., 48.):
            for blend in (.25, .5, 1.):
                lifts = []
                for seed, validation in groups.items():
                    train = [row for other, rows in groups.items() if other != seed for row in rows]
                    model = GainModel(train, actions, ridge, blend, steps)
                    differences = []
                    for row in validation:
                        x = np.array(row["features"])
                        gain = row["gain_long"]
                        differences.append(gain[int(model.values(x).argmax())] - gain[int(x[:, 0].argmax())])
                    lifts.append(float(np.mean(differences)))
                candidates.append({"regularization": ridge, "blend": blend, "target_horizon": steps,
                                   "mean_gain_vs_greedy": float(np.mean(lifts)), "lift_by_seed": lifts})
    choice = max(candidates, key=lambda item: item["mean_gain_vs_greedy"])
    return choice, candidates


def add_planning_calibration(groups, source, cases, actions, rounds, horizon, heldout_seeds):
    hashes = {}
    for case in cases:
        folder = source / case
        manifest = json.loads((folder/"manifest.json").read_text())
        schedule = json.loads((folder/"study_manifest.json").read_text())
        seed = manifest["args"]["seed"]
        if seed in heldout_seeds or schedule["actions"] != list(actions):
            raise ValueError("Planning calibration must use different seeds and the same action space.")
        path = folder/"decisions.jsonl"
        hashes[case] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["round"] for row in rows] != list(range(1, rounds + 1)):
            raise ValueError(f"Incomplete planning calibration {case}.")
        for row in rows:
            costs = np.array(row["rollout_costs"])
            if costs.shape != (len(actions), min(horizon, rounds - row["round"] + 1)):
                raise ValueError(f"Unmatched planning horizon {case}.")
            discount = DISCOUNT ** np.arange(costs.shape[1])
            average = costs @ discount / discount.sum()
            row["gain_long"] = (average[actions.index(REFERENCE)] - average).tolist()
        groups.setdefault(seed, []).extend(rows)
    return hashes


def evaluate(config, seed, scenario, folder, method, model):
    world, schedule = build_world(config, seed, scenario, folder)
    current = vector(world.global_model)
    actions = config["actions"]
    rows = []
    for index in range(world.global_rounds):
        shadow = world.shadows(current, index, actions, schedule)
        values = None
        oracle = None
        if method.startswith("fixed_"):
            action = actions.index(float(method[len("fixed_"):]))
        elif method == "round_only":
            closest = min(model.rounds, key=lambda r: abs(r - 1 - index))
            values = model.round_gains[closest]
            action = int(values.argmax())
        elif method == "greedy":
            values = shadow.features[:, 0]
            action = int(values.argmax())
        elif method == "learned":
            values = model.values(shadow.features)
            action = int(values.argmax())
        else:
            oracle, costs = continuation_values(world, shadow, index, config, schedule)
            values = costs[actions.index(REFERENCE)] - costs
            action = int(values.argmax())
        current = shadow.parameters[action]
        row = {**log_action(world, shadow, action, index, actions), "scores": None if values is None else values.tolist()}
        if oracle is not None:
            row["rollout_costs"] = oracle.tolist()
        if (index + 1) % EVAL_GAP == 0 or index + 1 == world.global_rounds:
            row["test"] = world.record_test(current)
            row["model_sha256"] = model_fingerprint(world.global_model)
        rows.append(row)
        append_json(folder / "decisions.jsonl", row)
    print(f"Evaluation: {scenario} seed {seed}, {method}, final={100 * rows[-1]['test']['accuracy']:.2f}%", flush=True)
    return rows


def summary(rows):
    tail = [row for row in rows if "test" in row and row["round"] > .5 * rows[-1]["round"]]
    return {
        "tail_accuracy": float(np.mean([row["test"]["accuracy"] for row in tail])),
        "tail_worst10": float(np.mean([row["test"]["worst10"] for row in tail])),
        "tail_validation": float(np.mean([row["selected_validation_loss"] for row in tail])),
        "final_accuracy": rows[-1]["test"]["accuracy"],
        "actions": {str(tau): sum(row["tau"] == tau for row in rows) for tau in sorted({r["tau"] for r in rows})},
        "greedy_disagreements": sum(
            row["action"] != int(np.argmax(np.array(row["features"])[:, 0])) for row in rows),
    }


def summarize_group(root, case, seed, methods, rounds):
    result = {}
    expected = None
    for method in methods:
        folder = root / "evaluation" / case / str(seed) / method
        records = [json.loads(line) for line in (folder/"decisions.jsonl").read_text().splitlines()]
        if [record["round"] for record in records] != list(range(1, rounds + 1)):
            raise RuntimeError(f"Incomplete run: {case}, seed {seed}, {method}.")
        source = json.loads((folder/"manifest.json").read_text())
        provenance = [source[key] for key in ("native_data_sha256", "private_splits_sha256",
                                              "initial_model_sha256", "malicious_clients_harness_only")]
        schedule = json.loads((folder/"study_manifest.json").read_text())
        provenance.extend([schedule, [(r["upload_clients"], r["sgd_seeds"]) for r in records]])
        if expected is None:
            expected = provenance
        elif expected != provenance:
            raise RuntimeError(f"Unmatched comparison: {case}, seed {seed}, {method}.")
        result[method] = summary(records)
    return result


def summarize_existing(root):
    plan = json.loads((root/"plan.json").read_text())
    results = {
        case: {seed: summarize_group(root, case, seed, plan["methods"], plan["rounds"])
               for seed in plan["eval_seeds"]}
        for case in plan["scenarios"]
    }
    write_json(root/"summary.json", {"results": results, "reconstructed_from_raw": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", type=float, default=[1.15, 1.25, 1.35, 1.55])
    parser.add_argument("--scenarios", nargs="+", choices=("none", "switched"), default=["none", "switched"])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[5, 6, 7])
    parser.add_argument("--clients", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--with-rollout", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--calibration-source", type=Path)
    parser.add_argument("--planning-calibration-source", type=Path)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1]/"dataset")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_existing(args.output.resolve())
        return
    seeds = args.train_seeds + args.eval_seeds
    actions = tuple(args.actions)
    if (len(seeds) != len(set(seeds)) or min(seeds) < 0 or len(actions) < 2
            or len(actions) != len(set(actions)) or REFERENCE not in actions
            or len(args.scenarios) != len(set(args.scenarios))
            or min(args.jobs, args.stride) < 1 or args.horizon < 2
            or not 2 <= args.clients <= 100 or args.rounds < max(EVAL_GAP, args.horizon)
            or args.rounds % EVAL_GAP):
        parser.error("Require unique seeds/actions/scenarios, tau 1.25, positive jobs/stride, "
                     "a horizon >=2 and enough rounds in multiples of five.")
    for tau in actions:
        validate_action((0., tau))
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config = {"actions": actions, "clients": args.clients, "rounds": args.rounds,
              "horizon": args.horizon, "stride": args.stride, "data_root": str(args.data_root.resolve())}
    methods = [f"fixed_{tau}" for tau in actions] + list(METHODS[:-1])
    if args.with_rollout:
        methods.append("rollout")
    write_json(root/"plan.json", {
        **config, "train_seeds": args.train_seeds, "eval_seeds": args.eval_seeds, "scenarios": args.scenarios,
        "methods": methods, "discount": DISCOUNT, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target": "Discounted private loss difference versus tau 1.25 following greedy continuation.",
        "criterion": "On a scenario, learned tail test accuracy must exceed best prespecified fixed and shadow greedy "
                     "by at least 0.10 percentage points on average and beat both on at least 2/3 evaluation seeds.",
        "tune_on_calibration_seeds": args.tune,
        "reused_calibration": str(args.calibration_source.resolve()) if args.calibration_source else None,
        "planning_calibration": (str(args.planning_calibration_source.resolve())
                                 if args.planning_calibration_source else None),
    })
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if args.calibration_source:
            source = args.calibration_source.resolve()
            original = json.loads((source/"plan.json").read_text())
            expected = {**config, "actions": list(actions), "train_seeds": args.train_seeds}
            if any(original[key] != value for key, value in expected.items()):
                raise ValueError("Existing calibration has different data, actions, rounds, or seeds.")
            groups = {seed: [] for seed in args.train_seeds}
            for case in args.scenarios:
                for seed in args.train_seeds:
                    path = source/"calibration"/case/str(seed)/"decisions.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if [row["round"] for row in rows] != list(range(1, args.rounds + 1)):
                        raise ValueError(f"Incomplete calibration: {path}.")
                    groups[seed].extend(row for row in rows if "gain_long" in row)
        else:
            collected = {(case, seed): pool.submit(collect, config, seed, case,
                                                    root/"calibration"/case/str(seed))
                         for case in args.scenarios for seed in args.train_seeds}
            groups = {seed: [row for case in args.scenarios for row in collected[case, seed].result()]
                      for seed in args.train_seeds}
        if args.planning_calibration_source:
            hashes = add_planning_calibration(
                groups, args.planning_calibration_source, args.scenarios, actions, args.rounds,
                args.horizon, args.eval_seeds)
            write_json(root/"planning_calibration.json", {"source_file_sha256": hashes,
                                                          "samples_per_seed": {seed:len(rows) for seed, rows in groups.items()}})
        examples = [row for rows in groups.values() for row in rows]
        options = {}
        if args.tune:
            choice, grid = tune_model(groups, actions, args.horizon)
            options = {key: choice[key] for key in ("regularization", "blend", "target_horizon")}
            write_json(root/"selection.json", {"selected": choice, "grid": grid})
        model = GainModel(examples, actions, **options)
        write_json(root/"model.json", {
            "mean": model.center.tolist(), "scale": model.scale.tolist(), "corrections": model.coefs.tolist(),
            "examples": len(examples), "blend": model.blend,
        })
        # Run one complete seed/scenario group at a time in submission order.
        completed = {(case, seed, method): pool.submit(
            evaluate, config, seed, case, root/"evaluation"/case/str(seed)/method, method, model)
                     for case in args.scenarios for seed in args.eval_seeds for method in methods}
        results = {}
        for case in args.scenarios:
            results[case] = {}
            for seed in args.eval_seeds:
                for method in methods:
                    completed[case, seed, method].result()
                results[case][seed] = summarize_group(root, case, seed, methods, config["rounds"])
                write_json(root/"progress.json", results)
    write_json(root/"summary.json", {"results": results, "elapsed_seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
