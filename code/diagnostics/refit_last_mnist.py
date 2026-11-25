"""Refit only round 36 across client pools and prospectively evaluate on a fourth pool."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np

from diagnostics.compare_terminal_mnist import score_path
from diagnostics.run_terminal_policy_mnist import (
    ACTIONS, advantages, collect_tree, decide, fit_head, read, select, write_json,
)


def pool_config(data, output, offset, split_seed, dataset="mnist"):
    root = output / f"clients{offset}" / "dataset"
    for part in ("train", "test"):
        folder = root / dataset / part
        folder.mkdir(parents=True)
        for i in range(24):
            source = data / dataset / part / f"{offset+i}.npz"
            (folder / f"{i}.npz").symlink_to(source.resolve(strict=True))
    return dict(rounds=40, clients=24, actions=ACTIONS, announced_cohorts=True,
                validation_fraction=.25, data_root=str(root), split_seed=split_seed, dataset=dataset)


def fit_last(trees, method):
    leaves = [leaf for tree in trees for leaf in tree["leaves"]]
    features = np.array([leaf["features"] for leaf in leaves])
    costs = np.array([leaf["costs"] for leaf in leaves])
    if method == "pooled_ridge":
        return fit_head(features, costs)

    # Center within each state. The same change in a candidate's features has
    # the same coefficient regardless of its absolute tau or client pool.
    delta = features - features[:, :1]
    targets = costs[:, :1] - costs
    scale = np.sqrt(np.mean(delta[:, 1:]**2, axis=(0, 1))).clip(1e-6)
    x = (delta[:, 1:]/scale).clip(-3, 3).reshape(-1, features.shape[-1])
    beta = np.linalg.solve(x.T @ x + 16*np.eye(x.shape[1]), x.T @ targets[:, 1:].ravel())
    return dict(anchor_differences=True, center=np.zeros(x.shape[1]).tolist(), scale=scale.tolist(),
                intercept=[0.]*len(ACTIONS), coef=[beta.tolist() for _ in ACTIONS])


def reached(tree, policy):
    a = select(policy["first"], tree["features"], policy["margin"])
    return tree["leaves"][a]


def calibrate(pools, original):
    scores = {}
    for method in ("pooled_ridge", "shared_difference_ridge"):
        fold = []
        for held in range(len(pools)):
            train = [tree for i, group in enumerate(pools) if i != held for tree in group]
            head = fit_last(train, method)
            regrets = []
            for tree in pools[held]:
                leaf = reached(tree, original)
                a = select(head, leaf["features"], original["margin"])
                regrets.append(leaf["costs"][a] - min(leaf["costs"]))
            fold.append(mean(regrets))
        scores[method] = fold
    winner = min(scores, key=lambda method: mean(scores[method]))
    trees = [tree for group in pools for tree in group]
    constant = min(range(len(ACTIONS)), key=lambda a: mean(reached(t, original)["costs"][a] for t in trees))
    return dict(original, last=fit_last(trees, winner)), dict(
        held_pool_regret=scores, selected=winner, constant_last_index=constant)


def assess(tree, original, revised, constant):
    first, action = decide(tree, revised)
    leaf = tree["leaves"][first]
    x = np.asarray(leaf["features"])
    # Single-upload proxy for the same error + .05 CE terminal criterion.
    greedy = int((x[:, -1] + .05*(x[:, 0]-x[0, 0])).argmax())
    best = int(np.argmin(leaf["costs"]))
    # Same first action, same fork opening (or no fork) and same verification.
    strong = 2 if first or action else 0
    actions = dict(original=decide(tree, original)[1], revised=action, matched_155=strong,
                   calibrated_constant=constant, immediate=greedy, hindsight_private=best)
    methods = {}
    for name, last in actions.items():
        path = (first, last)
        methods[name] = dict(last=ACTIONS[last], regret=leaf["costs"][last]-min(leaf["costs"]),
                             direct=score_path(tree, path, False), verified=score_path(tree, path, True))
    return dict(seed=tree["seed"], first=ACTIONS[first], fixed=score_path(tree, (0, 0), False),
                predicted_gain=advantages(revised["last"], x).tolist(),
                measured_gain=[leaf["costs"][0]-v for v in leaf["costs"]], methods=methods)


def summarize(rows):
    means = {}
    for name in rows[0]["methods"]:
        methods = [r["methods"][name] for r in rows]
        means[name] = dict(hits=sum(m["regret"] == 0 for m in methods), regret=mean(m["regret"] for m in methods),
                           accuracy=mean(m["direct"]["accuracy"] for m in methods),
                           worst10=mean(m["direct"]["worst10"] for m in methods),
                           verified_accuracy=mean(m["verified"]["accuracy"] for m in methods),
                           verified_worst10=mean(m["verified"]["worst10"] for m in methods),
                           fork_rounds=mean(m["verified"]["extra_rounds"] for m in methods))
    differences = {}
    for name in ("original", "matched_155", "calibrated_constant", "immediate", "fixed"):
        differences[name] = {}
        for mode in ("direct", "verified"):
            deltas = []
            for row in rows:
                control = row["fixed"] if name == "fixed" else row["methods"][name][mode]
                deltas.append(100*(row["methods"]["revised"][mode]["accuracy"]-control["accuracy"]))
            differences[name][mode] = dict(mean_pp=mean(deltas), wins=sum(d>0 for d in deltas),
                                            ties=deltas.count(0.), losses=sum(d<0 for d in deltas))
    return dict(fixed_accuracy=mean(r["fixed"]["accuracy"] for r in rows), means=means,
                revised_minus=differences, per_seed=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-policy", type=Path, required=True)
    parser.add_argument("--original-trees", type=Path, required=True, help="Private fit trees 1..16")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("Use at least one worker")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = args.data_root.resolve()
    original = read(args.original_policy)
    old = [read(args.original_trees / str(s) / "tree.json") for s in range(1, 17)]
    pools = [old]
    fit = list(range(201, 217))
    test = list(range(321, 337))
    write_json(root / "protocol.json", dict(first_policy=original["first"], margin=original["margin"],
               fit_pools=[0,24,48], split_seeds=[2026,3055,3079,3103], original_fit=list(range(1,17)),
               added_fit=fit, held_pool=72, held_seeds=test,
               head_selection="min mean private regret, leave one fit pool out",
               endpoints="round 36 private regret and round 40 direct accuracy vs original and matched 1.55"))
    with ProcessPoolExecutor(args.jobs) as executor:
        groups = []
        for offset, split in ((24, 3055), (48, 3079)):
            config = pool_config(data, root, offset, split)
            groups.append([executor.submit(collect_tree, config, s, root/f"clients{offset}"/str(s))
                           for s in fit])
        pools.extend([[job.result() for job in group] for group in groups])
        policy, cv = calibrate(pools, original)
        write_json(root/"selection.json", cv)
        write_json(root/"policy.json", policy)
        print("Frozen before held-pool evaluation:", cv, flush=True)
        config = pool_config(data, root, 72, 3103)
        jobs = [executor.submit(collect_tree, config, s, root/"clients72"/str(s), True) for s in test]
        rows = [assess(job.result(), original, policy, cv["constant_last_index"]) for job in jobs]
    write_json(root/"evaluation.json", summarize(rows))


if __name__ == "__main__":
    main()
