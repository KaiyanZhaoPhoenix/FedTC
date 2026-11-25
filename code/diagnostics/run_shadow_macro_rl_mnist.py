"""Compare short and long returns when a chosen tau is held for five rounds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.fedtc.forward import calibrate
from diagnostics.run_shadow_rl_mnist import (
    DISCOUNT, REFERENCE, GainModel, ShadowMNIST, build_world, log_action,
)
from main import parser as training_parser, prepare
from utils.experiment import append_json, fingerprint, model_fingerprint, set_vector, vector, write_json

H = 5
SCENARIO = "switched"
ACTIONS = [1.15, 1.25, 1.55]
METHODS = ("fixed_1.15", "fixed_1.25", "fixed_1.55", "round_only", "greedy", "short", "long")


class AnnouncedCohortsMNIST(ShadowMNIST):
    classification_cost = False

    def private_logits(self):
        if self.args.dataset == "Cifar10":
            # Only the ten scores per image need to coexist for the client reductions.
            return torch.cat([self.global_model(batch) for batch in self.private_x.split(128)])
        return self.global_model(self.private_x)

    @torch.no_grad()
    def private_losses(self, parameters):
        if not self.classification_cost:
            return super().private_losses(parameters)
        set_vector(self.global_model, parameters)
        logits = self.private_logits()
        errors = (logits.argmax(dim=1) != self.private_y).float()
        confidence = F.cross_entropy(logits, self.private_y, reduction="none").clamp(max=self.args.loss_cap)
        combined = errors + .05 * confidence
        return np.array([float(chunk.mean()) for chunk in combined.split(self.sizes.tolist())])

    def shadows(self, start, index, actions, schedule):
        shadow = super().shadows(start, index, actions, schedule)
        following = min(index+H, self.global_rounds-1)
        future_ids = np.array(self.schedule[following])
        forthcoming = np.isin(np.arange(self.num_clients), future_ids)
        ref_losses = shadow.losses[actions.index(REFERENCE)]
        gains = ref_losses - shadow.losses
        groups = [forthcoming, ~forthcoming]
        # These upload slots are server-announced protocol metadata.
        # The state never reads cohort construction labels or test samples.
        group_gains = [gains[:, ids] @ (self.sample_weights[ids] / self.sample_weights[ids].sum())
                       for ids in groups]
        group_losses = [float(ref_losses[ids] @ (self.sample_weights[ids] / self.sample_weights[ids].sum()))
                        for ids in groups]
        shadow.features = np.column_stack((shadow.features, *group_gains,
                                          np.tile(group_losses, (len(actions), 1))))
        return shadow


def open_world(config, seed, folder):
    if not config.get("announced_cohorts", False):
        return build_world(config, seed, SCENARIO, folder)
    count = config["clients"]
    args = training_parser().parse_args([
        "--algorithm", "FedTCFixed", "--dataset", config.get("dataset", "mnist"), "--num_clients", str(count),
        "--num_sel_clients", str(count//2), "--aggre_num", str(count//2),
        "--training-cohort", "sampled", "--batch_size", "16", "--local_epochs", "1",
        "--global_rounds", str(config["rounds"]), "--macro-length", str(H),
        "--validation-fraction", str(config.get("validation_fraction", .1)),
        "--attack", "none", "--ad_ratio", "0", "--seed", str(seed), "--device", "cpu",
        "--output", str(folder), "--data-root", config["data_root"],
        *(["--split-seed", str(config["split_seed"])] if "split_seed" in config else []),
    ])
    world = AnnouncedCohortsMNIST(prepare(args))
    world.classification_cost = config.get("classification_cost", False)
    # Build a benchmark availability shift from training-only label heterogeneity.
    def dominant(client):
        y = torch.stack([client._full_train_data[i][1] for i in client.train_indices])
        return int(torch.bincount(y).argmax())
    order = sorted(range(count), key=lambda i: dominant(world.clients[i]))
    cohorts = (order[:count//2], order[count//2:])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 960]))
    blocks = [int(value) for _ in range((world.global_rounds + 2*H - 1)//(2*H))
              for value in rng.permutation(2)]
    world.schedule = [rng.permutation(cohorts[blocks[index//H]]).tolist()
                      for index in range(world.global_rounds)]
    schedule = ["none"] * world.global_rounds
    write_json(folder/"study_manifest.json", {
        "attack_schedule_harness_only": schedule,
        "upload_schedule_sha256": fingerprint(world.schedule), "announced_upload_schedule": world.schedule,
        "actions": config["actions"], "utility_weight": 0, "test_policy_input": False,
        **({"action_kind": "server_step", "tau": 1.15, "final_tau_candidates": [1.15, 1.55]}
           if config.get("action_kind") == "server_step" else {}),
        "private_cost": "classification error + 0.05 capped cross entropy" if world.classification_cost
                        else "capped cross entropy",
        "validation_fraction": config.get("validation_fraction", .1),
    })
    return world, schedule


def commit(world, parameters, index, tau, schedule):
    """Continue a first-round shadow along the same action for a complete macro."""
    return continue_rounds(world, parameters, index + 1, min(index + H, world.global_rounds), tau, schedule)


def continue_rounds(world, parameters, start, stop, tau, schedule):
    for rnd in range(start, stop):
        set_vector(world.global_model, parameters)
        world.attack = schedule[rnd]
        clients = [world.clients[i] for i in world.schedule[rnd]]
        world.send_models()
        world.train_clients(clients, rnd, parameters)
        updates = torch.stack([vector(client.model) - parameters for client in clients])
        result = calibrate(updates, updates.new_zeros(len(updates)), (0., tau))
        parameters = parameters + result.delta
    return parameters


def next_greedy(world, params, index, schedule):
    shadow = world.shadows(params, index, ACTIONS, schedule)
    action = int(shadow.features[:, 0].argmax())
    return commit(world, shadow.parameters[action], index, ACTIONS[action], schedule)


def exact_values(world, shadow, index, schedule, reference, long):
    successors = [commit(world, p, index, tau, schedule) for p, tau in zip(shadow.parameters, ACTIONS)]
    values = []
    for endpoint in successors:
        params = endpoint
        costs = [float(world.private_losses(params) @ world.sample_weights)]
        if long:
            for following in range(index+H, world.global_rounds, H):
                future = world.shadows(params, following, [REFERENCE, reference]
                                       if reference != REFERENCE else [REFERENCE], schedule)
                chosen = -1
                params = commit(world, future.parameters[chosen], following, reference, schedule)
                costs.append(float(world.private_losses(params) @ world.sample_weights))
        values.append(-float(np.mean(costs)))
    return successors, np.array(values)


def collect(config, seed, folder):
    world, schedule = open_world(config, seed, folder)
    current = vector(world.global_model)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 415]))
    rows = []
    for index in range(0, world.global_rounds, H):
        shadow = world.shadows(current, index, ACTIONS, schedule)
        if config.get("calibrate_reference") and index < config["warmup_rounds"]:
            tau = config["reference_tau"]
            action = ACTIONS.index(tau)
            current = commit(world, shadow.parameters[action], index, tau, schedule)
            append_json(folder/"decisions.jsonl", {
                **log_action(world, shadow, action, index, ACTIONS), "round_end": index + H, "warmup": True
            })
            continue
        successors = [commit(world, p, index, tau, schedule) for p, tau in zip(shadow.parameters, ACTIONS)]
        costs = []
        for successor in successors:
            c = [float(world.private_losses(successor) @ world.sample_weights)]
            if index + H < world.global_rounds:
                future = next_greedy(world, successor, index + H, schedule)
                c.append(float(world.private_losses(future) @ world.sample_weights))
            costs.append(c)
        cost = np.array(costs)
        discounts = DISCOUNT ** np.arange(cost.shape[1])
        target = cost @ discounts / discounts.sum()
        exploring = rng.random() < .3
        action = int(rng.integers(len(ACTIONS))) if exploring else int(cost[:, 0].argmin())
        row = {
            **log_action(world, shadow, action, index, ACTIONS),
            "round_end": index + H, "exploring": exploring, "rollout_costs": cost.tolist(),
            "gain_long": (target[ACTIONS.index(REFERENCE)] - target).tolist(),
            "selected_macro_validation_loss": float(cost[action, 0]),
        }
        append_json(folder/"decisions.jsonl", row)
        rows.append(row)
        current = successors[action]
    print(f"Macro calibration seed {seed}: {len(rows)} fully paired snapshots", flush=True)
    return rows


def evaluate(config, seed, folder, method, scorer, test=True):
    world, schedule = open_world(config, seed, folder)
    current = vector(world.global_model)
    rows = []
    reference = ACTIONS.index(config.get("reference_tau", REFERENCE))
    for index in range(0, world.global_rounds, H):
        shadow = world.shadows(current, index, ACTIONS, schedule)
        successors = None
        warmup = index < config.get("warmup_rounds", 0) and not method.startswith("fixed_")
        if method.startswith("fixed_"):
            values = None
            action = ACTIONS.index(float(method[6:]))
        elif warmup:
            values = None
            action = reference
        else:
            if method == "greedy":
                values = shadow.features[:, 0]
            elif method == "round_only":
                values = scorer.round_gains[index+1]
            elif method in ("exact_short", "exact_long"):
                successors, values = exact_values(
                    world, shadow, index, schedule, ACTIONS[reference], method == "exact_long")
            else:
                values = scorer.values(shadow.features)
                if config.get("calibrate_reference"):
                    values = values - .001
                    values[reference] += .001
            action = int(values.argmax())
        current = (successors[action] if successors is not None else
                   commit(world, shadow.parameters[action], index, ACTIONS[action], schedule))
        row = {
            **log_action(world, shadow, action, index, ACTIONS),
            "round_end": index + H, "warmup": warmup,
            "scores": None if values is None else values.tolist(),
            "selected_macro_validation_loss": float(world.private_losses(current) @ world.sample_weights),
            "model_sha256": model_fingerprint(world.global_model),
        }
        if test:
            row["test"] = world.record_test(current)
        append_json(folder/"decisions.jsonl", row)
        rows.append(row)
    if test:
        print(f"Macro evaluation seed {seed}: {method}, final={100*rows[-1]['test']['accuracy']:.2f}%", flush=True)
    return rows


def metrics(rows, rounds):
    tail = [row for row in rows if row["round_end"] > rounds/2]
    return {"tail_accuracy": float(np.mean([r["test"]["accuracy"] for r in tail])),
            "tail_worst10": float(np.mean([r["test"]["worst10"] for r in tail])),
            "tail_validation": float(np.mean([r["selected_macro_validation_loss"] for r in tail])),
            "final_accuracy": rows[-1]["test"]["accuracy"],
            "first_round_actions": [r["tau"] for r in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1]/"dataset")
    parser.add_argument("--clients", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[12, 13, 14])
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--announced-cohorts", action="store_true")
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--calibration-source", type=Path)
    parser.add_argument("--calibrate-reference", action="store_true")
    parser.add_argument("--classification-cost", action="store_true")
    parser.add_argument("--planner-baseline", type=float, choices=ACTIONS)
    parser.add_argument("--planner-mode", choices=("both", "short"), default="both")
    parser.add_argument("--validation-fraction", type=float, default=.1)
    args = parser.parse_args()
    seeds = args.train_seeds + args.eval_seeds
    if (len(set(seeds)) != len(seeds) or min(seeds) < 0 or args.rounds % H
            or args.rounds < 2*H or args.jobs < 1 or not 2 <= args.clients <= 100
            or not 0 <= args.warmup_rounds < args.rounds or args.warmup_rounds % H
            or (args.calibrate_reference and (args.calibration_source or not args.warmup_rounds))
            or (args.classification_cost and not args.announced_cohorts)
            or not 0 < args.validation_fraction < 1
            or (args.validation_fraction != .1 and not args.announced_cohorts)
            or (args.planner_baseline is not None and
                (not args.announced_cohorts or args.calibrate_reference))
            or (args.announced_cohorts and args.clients % 2)):
        parser.error("Require disjoint nonnegative seeds, positive jobs and rounds divisible by five.")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config = {"actions": ACTIONS, "clients": args.clients, "rounds": args.rounds,
              "data_root": str(args.data_root.resolve()), "announced_cohorts": args.announced_cohorts,
              "warmup_rounds": args.warmup_rounds, "calibrate_reference": args.calibrate_reference,
              "classification_cost": args.classification_cost}
    if args.validation_fraction != .1:
        config["validation_fraction"] = args.validation_fraction
    methods = METHODS
    if args.planner_baseline is not None:
        config["reference_tau"] = args.planner_baseline
        methods += ("exact_short",)
        if args.planner_mode == "both":
            methods += ("exact_long",)
    write_json(root/"plan.json", {
        **config, "train_seeds": args.train_seeds, "eval_seeds": args.eval_seeds, "methods": methods,
        "scenario": "announced_availability" if args.announced_cohorts else SCENARIO,
        "action_duration": H, "rollout_macros": 2,
        "reused_calibration": str(args.calibration_source.resolve()) if args.calibration_source else None,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "criterion": "Long policy gains >=0.10 pp on average vs strongest fixed, greedy, and "
                     "short learned policy, and beats all three in at least two held-out seeds.",
        "planner_criterion": {
            "both": "Exact long improves >=0.10 pp vs strongest fixed, immediate greedy and exact short, "
                    "with joint wins on >=2 held-out seeds.",
            "short": "Exact short improves >=0.10 pp vs strongest fixed, immediate greedy and time-only policy, "
                     "with joint wins on >=2 held-out seeds.",
        }[args.planner_mode] if args.planner_baseline is not None else None,
    })
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if args.calibrate_reference:
            controls = {(tau, seed): pool.submit(
                evaluate, config, seed, root/"reference_calibration"/str(seed)/str(tau),
                f"fixed_{tau}", None, False) for tau in ACTIONS for seed in args.train_seeds}
            validation_losses = {
                tau: float(np.mean([row["selected_macro_validation_loss"]
                                    for seed in args.train_seeds for row in controls[tau,seed].result()
                                    if row["round_end"] > args.rounds/2])) for tau in ACTIONS
            }
            config["reference_tau"] = min(validation_losses, key=validation_losses.get)
            write_json(root/"reference_selection.json", {"value": config["reference_tau"],
                                                        "calibration_validation_losses": validation_losses,
                                                        "candidate_score_margin": .001})
        if args.calibration_source:
            input_root = args.calibration_source.resolve()
            input_plan = json.loads((input_root/"plan.json").read_text())
            expected = {key: value for key, value in config.items()
                        if key not in ("warmup_rounds", "calibrate_reference", "reference_tau")}
            if any(input_plan.get(key, False) != value for key, value in expected.items()):
                raise ValueError("Calibration data or protocol does not match this study.")
            training = []
            input_hashes = {}
            for seed in args.train_seeds:
                source = input_root/"calibration"/str(seed)/"decisions.jsonl"
                input_hashes[str(seed)] = hashlib.sha256(source.read_bytes()).hexdigest()
                records = [json.loads(line) for line in source.read_text().splitlines()]
                if [row["round_end"] for row in records] != list(range(H, args.rounds+1, H)):
                    raise ValueError(f"Incomplete calibration: {source}.")
                training.extend(records)
            write_json(root/"calibration_sources.json", input_hashes)
        else:
            fitted = [pool.submit(collect, config, seed, root/"calibration"/str(seed))
                      for seed in args.train_seeds]
            training = [row for future in fitted for row in future.result()]
        models = {"long": GainModel(training, ACTIONS), "short": GainModel(training, ACTIONS, target_horizon=1)}
        for name, model in models.items():
            write_json(root/f"{name}.json", {"center": model.center.tolist(), "scale": model.scale.tolist(),
                                             "coefs": model.coefs.tolist()})
        outcomes = {(seed, method): pool.submit(evaluate, config, seed,
                                               root/"evaluation"/str(seed)/method, method,
                                               models["short" if method == "short" else "long"])
                    for seed in args.eval_seeds for method in methods}
        results = {}
        for seed in args.eval_seeds:
            results[seed], matched, first = {}, None, None
            for method in methods:
                series = outcomes[seed,method].result()
                folder = root/"evaluation"/str(seed)/method
                manifest = json.loads((folder/"manifest.json").read_text())
                schedule = json.loads((folder/"study_manifest.json").read_text())
                provenance = [manifest[key] for key in ("native_data_sha256", "private_splits_sha256",
                                                        "initial_model_sha256",
                                                        "malicious_clients_harness_only")]
                if matched is None:
                    matched = [provenance, schedule]
                    first = series[0]["upload_sha256"]
                elif matched != [provenance, schedule] or first != series[0]["upload_sha256"]:
                    raise RuntimeError(f"Provenance mismatch at seed {seed}, {method}.")
                results[seed][method] = metrics(series, args.rounds)
            write_json(root/"progress.json", results)
    write_json(root/"summary.json", {"results": results, "elapsed_seconds": time.monotonic()-start})


if __name__ == "__main__":
    main()
