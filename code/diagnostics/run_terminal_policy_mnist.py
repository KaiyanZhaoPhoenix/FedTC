"""Fit a two-decision MNIST policy by backward induction on paired training trees."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.run_shadow_macro_rl_mnist import ACTIONS, H, commit, evaluate, metrics, open_world
from diagnostics.run_shadow_rl_mnist import GainModel
from utils.experiment import fingerprint, set_vector, vector, write_json

BASE = 0  # tau=1.15, the strong baseline from the preceding MNIST study.


def read(path):
    return json.loads(path.read_text())


@torch.no_grad()
def private_error(world, parameters):
    set_vector(world.global_model, parameters)
    logits = world.global_model(world.private_x)
    return float((logits.argmax(dim=1) != world.private_y).float().mean())


def terminal_cost(world, parameters):
    ce = float(world.private_losses(parameters) @ world.sample_weights)
    return private_error(world, parameters) + .05 * ce


def state_features(world, shadow):
    errors = np.array([private_error(world, p) for p in shadow.parameters])
    return np.column_stack((shadow.features, errors[BASE] - errors))


def successors(world, shadow, index, schedule):
    return [commit(world, p, index, tau, schedule) for tau, p in zip(ACTIONS, shadow.parameters)]


def collect_tree(config, seed, folder, measure_test=False):
    world, schedule = open_world(config, seed, folder)
    current = vector(world.global_model)
    for index in range(0, config['rounds']-2*H, H):
        shadow = world.shadows(current, index, ACTIONS, schedule)
        current = commit(world, shadow.parameters[BASE], index, ACTIONS[BASE], schedule)
    index = config['rounds']-2*H
    first = world.shadows(current, index, ACTIONS, schedule)
    first_features = state_features(world, first)
    leaves = []
    for p in successors(world, first, index, schedule):
        last = world.shadows(p, index+H, ACTIONS, schedule)
        features = state_features(world, last)
        ends = successors(world, last, index+H, schedule)
        costs = [terminal_cost(world, end) for end in ends]
        leaf = dict(features=features.tolist(), costs=costs, upload_hash=last.upload_hash)
        if measure_test:
            leaf['test'] = [world.record_test(end) for end in ends]
        leaves.append(leaf)
    tree = dict(seed=seed, features=first_features.tolist(), upload_hash=first.upload_hash, leaves=leaves)
    write_json(folder/'tree.json', tree)
    return tree


def fit_head(features, costs):
    # Fit advantage against the anchor, in raw units of terminal private cost.
    x, costs = np.asarray(features), np.asarray(costs)
    gain = costs[:, :1] - costs
    center = x.mean(axis=0)
    scale = x.std(axis=0).clip(1e-6)
    design = ((x-center)/scale).clip(-3,3)
    coefficients = []
    for a in range(len(ACTIONS)):
        v = design[:,a]
        coefficients.append(np.linalg.solve(v.T@v + 16*np.eye(v.shape[1]), v.T@(gain[:,a]-gain[:,a].mean())))
    return dict(center=center.tolist(), scale=scale.tolist(), coef=np.array(coefficients).tolist(),
                intercept=gain.mean(axis=0).tolist())


def advantages(head, features):
    if head.get("anchor_differences"):
        features = np.asarray(features) - np.asarray(features)[BASE]
    x = ((np.asarray(features)-head['center'])/head['scale']).clip(-3,3)
    return np.asarray(head['intercept']) + np.sum(x*head['coef'],axis=1)


def select(head, features, margin):
    score = advantages(head, features)
    action = int(score.argmax())
    return action if score[action] > margin else BASE


def train(trees, margin):
    # Freeze the final decision rule before constructing the previous decision's targets.
    leaves = [leaf for tree in trees for leaf in tree['leaves']]
    last = fit_head([leaf['features'] for leaf in leaves], [leaf['costs'] for leaf in leaves])
    targets = [[leaf['costs'][select(last, leaf['features'], margin)] for leaf in tree['leaves']]
               for tree in trees]
    first = fit_head([tree['features'] for tree in trees], targets)
    return dict(first=first, last=last, margin=margin)


def decide(tree, policy):
    first = select(policy['first'], tree['features'], policy['margin'])
    leaf = tree['leaves'][first]
    last = select(policy['last'], leaf['features'], policy['margin'])
    return first, last


def choose_policy(trees, validation):
    choices = []
    for margin in (0., .0025, .005, .01):
        policy = train(trees, margin)
        lift = []
        for tree in validation:
            a,b = decide(tree, policy)
            lift.append(tree['leaves'][BASE]['costs'][BASE] - tree['leaves'][a]['costs'][b])
        # Penalize dispersion across independent training/validation trajectories.
        criterion = np.mean(lift) - np.std(lift, ddof=1)/np.sqrt(len(lift))
        choices.append(dict(margin=margin, gains=lift, criterion=float(criterion)))
    best = max(choices, key=lambda row:(row['criterion'],row['margin']))
    return train(trees, best['margin']), choices


def record_round(world, parameters, shadow, index, action, scores):
    return {
        "round_end": index + H,
        "tau": ACTIONS[action],
        "scores": scores,
        "upload_sha256": shadow.upload_hash,
        "selected_macro_validation_loss": float(world.private_losses(parameters) @ world.sample_weights),
        "test": world.record_test(parameters),
    }


def run_policy(config, seed, folder, policy, verify_final=False, save_model=False):
    world, schedule = open_world(config, seed, folder)
    parameters = vector(world.global_model)
    reference = None
    reference_rows = []
    rows = []

    for index in range(0, config["rounds"], H):
        shadow = world.shadows(parameters, index, ACTIONS, schedule)
        remaining = config["rounds"] - index
        action, scores = BASE, None
        if remaining <= 2 * H:
            if policy.get("timing_only") and reference is not None:
                action = ACTIONS.index(1.55)
            elif "schedule" in policy:
                action = policy["schedule"][1 if remaining == H else 0]
            else:
                head = policy["last" if remaining == H else "first"]
                features = state_features(world, shadow)
                scores = advantages(head, features).tolist()
                action = select(head, features, policy["margin"])
                if policy.get("timing_only") and action != BASE:
                    action = ACTIONS.index(1.55)

        # Both branches share the prefix and the uploads at the fork.
        if verify_final and (reference is not None or action != BASE):
            baseline_shadow = shadow if reference is None else world.shadows(reference, index, ACTIONS, schedule)
            reference = commit(world, baseline_shadow.parameters[BASE], index, ACTIONS[BASE], schedule)
            reference_rows.append(record_round(world, reference, baseline_shadow, index, BASE, None))

        parameters = commit(world, shadow.parameters[action], index, ACTIONS[action], schedule)
        rows.append(record_round(world, parameters, shadow, index, action, scores))

    if verify_final:
        candidate_cost = terminal_cost(world, parameters)
        reference_cost = terminal_cost(world, reference) if reference is not None else candidate_cost
        keep_candidate = reference is None or candidate_cost < reference_cost
        proposed = [row["tau"] for row in rows]
        if not keep_candidate:
            write_json(folder / "candidate_decisions.json", rows)
            rows[-len(reference_rows):] = reference_rows
            parameters = reference
        write_json(folder / "terminal_selection.json", {
            "candidate_cost": candidate_cost,
            "anchor_cost": reference_cost,
            "fork_rounds": H * len(reference_rows),
            "kept_candidate": keep_candidate,
            "proposed_actions": proposed,
            "executed_actions_in_returned_model": [row["tau"] for row in rows],
            "final_test": rows[-1]["test"],
        })
    set_vector(world.global_model, parameters)
    if save_model:
        torch.save(world.global_model.state_dict(), folder / "global_model.pt")
    write_json(folder / "decisions.json", rows)
    return rows


def previous_policy(source):
    saved = read(source / "long.json")
    policy = GainModel.__new__(GainModel)
    policy.ref = ACTIONS.index(1.25)
    for name in ("center", "scale", "coefs"):
        setattr(policy, name, np.asarray(saved[name]))
    return policy


def paired_summary(results):
    fields = ("final_accuracy", "tail_accuracy", "tail_worst10")
    means = {
        method: {key: float(np.mean([row[method][key] for row in results.values()])) for key in fields}
        for method in next(iter(results.values()))
    }
    best = max((m for m in means if m.startswith("fixed_")), key=lambda m: means[m]["final_accuracy"])
    differences = {}
    for method in dict.fromkeys((best, "old_long", "fixed_1.15")):
        delta = [100 * (r["terminal"]["final_accuracy"] - r[method]["final_accuracy"])
                 for r in results.values()]
        differences[method] = dict(mean_pp=float(np.mean(delta)), wins=sum(d > 0 for d in delta),
                                   ties=sum(d == 0 for d in delta), per_seed_pp=delta)
    return dict(means=means, best_fixed=best, paired=differences)


def evaluate_bank(pool, config, seeds, folder, policy, old):
    methods = ["terminal", "old_long"] + [f"fixed_{tau}" for tau in ACTIONS]
    jobs = {}
    for seed in seeds:
        for method in methods:
            run = folder / str(seed) / method
            if method == "terminal":
                jobs[seed, method] = pool.submit(run_policy, config, seed, run, policy)
            else:
                jobs[seed, method] = pool.submit(evaluate, config, seed, run, method, old)
    results = {}
    for seed in seeds:
        reference = None
        results[seed] = {}
        for method in methods:
            rows = jobs[seed, method].result()
            run = folder / str(seed) / method
            manifest = read(run / "manifest.json")
            proof = [manifest[k] for k in ("native_data_sha256", "private_splits_sha256", "initial_model_sha256")]
            proof.extend([read(run / "study_manifest.json"), rows[0]["upload_sha256"]])
            if reference is not None and proof != reference:
                raise ValueError(f"Unmatched evaluation at {seed}, {method}.")
            reference = proof
            results[seed][method] = metrics(rows, config["rounds"])
        write_json(folder / "results.json", results)
    write_json(folder / "comparison.json", paired_summary(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Previous long-policy study for the matched baseline")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--fit-only", action="store_true",
                        help="Fit the two heads and save the verified policy without evaluating test seeds")
    parser.add_argument("--policy", type=Path, help="Run a single frozen policy from JSON")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-seed", type=int, help="Private split for a frozen-policy replay")
    parser.add_argument("--verify-final", action="store_true",
                        help="Keep a fixed branch after the first deviation and select by terminal private cost")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    args = parser.parse_args()
    if args.jobs < 1 or (args.policy is None) != (args.seed is None):
        parser.error("Use --policy and --seed together, with a positive number of jobs.")
    if args.seed is not None and args.seed < 0:
        parser.error("Use a nonnegative seed.")
    if args.split_seed is not None and (args.policy is None or args.split_seed < 0):
        parser.error("--split-seed requires a frozen policy and a nonnegative split seed.")
    if args.verify_final and args.policy is None:
        parser.error("--verify-final applies to a single frozen policy run.")
    if args.fit_only and args.policy is not None:
        parser.error("--fit-only trains a new policy and cannot load --policy.")
    if args.source is None and args.policy is None and not args.fit_only:
        parser.error("--source is required for the comparison against the previous controller.")

    expected = dict(rounds=40, clients=24, actions=ACTIONS, announced_cohorts=True, validation_fraction=.25)
    config = read(args.source / "plan.json") if args.source else expected.copy()
    if any(config.get(k) != v for k, v in expected.items()) or config.get("classification_cost", False):
        raise ValueError("Expected the matched 40-round CE availability benchmark.")
    config["data_root"] = str(args.data_root.resolve())
    if args.split_seed is not None:
        config["split_seed"] = args.split_seed
    out = args.output.resolve()
    if args.policy is not None:
        policy = read(args.policy)
        out.mkdir(parents=True, exist_ok=False)
        run_policy(config, args.seed, out, policy, args.verify_final or policy.get("verify_final", False),
                   save_model=True)
        return

    fit, tune = list(range(1, 17)), list(range(17, 25))
    held, confirm = list(range(42, 54)), list(range(54, 66))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", dict(
        fit=fit, tune=tune, held=[] if args.fit_only else held, confirm=[] if args.fit_only else confirm, config=config,
        primary="round-40 test accuracy, +0.10 pp vs strongest fixed and wins on a majority of seeds in each bank",
        secondary="tail accuracy and worst 10%; harmful switches measured by terminal private cost",
        warmup_tau=ACTIONS[BASE], warmup_rounds=30,
    ))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = {s: pool.submit(collect_tree, config, s, out / "calibration" / str(s)) for s in fit + tune}
        trees = {s: task.result() for s, task in jobs.items()}
        policy, choices = choose_policy([trees[s] for s in fit], [trees[s] for s in tune])
        write_json(out / "policy.json", policy)
        write_json(out / "selection.json", choices)
        write_json(out / "verified_terminal_policy.json", dict(policy, verify_final=True))
        if args.fit_only:
            return
        old = previous_policy(args.source)
        evaluate_bank(pool, config, held, out / "held", policy, old)
        evaluate_bank(pool, config, confirm, out / "confirm", policy, old)
        print("Frozen policy:", fingerprint(policy), flush=True)


if __name__ == "__main__":
    main()
