"""Does a second observed local-training round improve the last five-round decision?"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np

from diagnostics.refit_last_mnist import pool_config
from diagnostics.run_shadow_macro_rl_mnist import commit, continue_rounds, open_world
from diagnostics.run_terminal_policy_mnist import (
    ACTIONS, private_error, read, select, state_features, terminal_cost, vector, write_json,
)


def gain_features(world, parameters):
    values = np.array([[private_error(world, p), world.private_losses(p) @ world.sample_weights]
                       for p in parameters])
    return values[0] - values


def fit_probe(rows):
    x = np.array([r["features"] for r in rows])[:, 1:]
    x = x.reshape(-1, x.shape[-1])
    y = np.array([r["gain"] for r in rows])[:, 1:].ravel()
    scale = np.sqrt(np.mean(x*x, axis=0)).clip(1e-6)
    x = (x/scale).clip(-3,3)
    coef = np.linalg.solve(x.T@x + 16*np.eye(x.shape[1]), x.T@y)
    return dict(scale=scale.tolist(), coef=coef.tolist())


def probe(config, seed, folder, first_policy, *, tree_path=None, head=None):
    world, schedule = open_world(config, seed, folder)
    params = vector(world.global_model)
    for index in range(0, 30, 5):
        shadow = world.shadows(params, index, ACTIONS, schedule)
        params = commit(world, shadow.parameters[0], index, ACTIONS[0], schedule)
    first = world.shadows(params, 30, ACTIONS, schedule)
    a = select(first_policy["first"], state_features(world, first), first_policy["margin"])
    params = commit(world, first.parameters[a], 30, ACTIONS[a], schedule)
    last = world.shadows(params, 35, ACTIONS, schedule)
    g1 = gain_features(world, last.parameters)
    second = [continue_rounds(world, p, 36, 37, tau, schedule) for tau, p in zip(ACTIONS, last.parameters)]
    g2 = gain_features(world, second)
    features = np.column_stack((g1, g2))
    row = dict(seed=seed, first=ACTIONS[a], features=features.tolist(), upload_sha256=last.upload_hash)
    if tree_path is not None:
        # Reuse terminal labels only after proving this is exactly the fitted prefix.
        tree = read(tree_path)
        leaf = tree["leaves"][a]
        if first.upload_hash != tree["upload_hash"] or last.upload_hash != leaf["upload_hash"]:
            raise ValueError("Probe and target came from different states")
        costs = np.asarray(leaf["costs"])
    else:
        scores = (features / head["scale"]).clip(-3,3) @ head["coef"]
        row["scores"] = scores.tolist()
        row["choice"] = dict(fixed_last=0, fixed_strong=2,
                             greedy1=int((g1 @ [1., .05]).argmax()),
                             greedy2=int((g2 @ [1., .05]).argmax()), learned2=int(scores.argmax()))
        ends = [continue_rounds(world, p, 37, 40, tau, schedule) for tau, p in zip(ACTIONS, second)]
        costs = np.array([terminal_cost(world, end) for end in ends])
        row["test"] = [world.record_test(end) for end in ends]
    row["gain"] = (costs[0]-costs).tolist()
    write_json(folder/"probe.json", row)
    return row


def summary(rows):
    result = {}
    for name in rows[0]["choice"]:
        choices = [r["choice"][name] for r in rows]
        regrets = [max(r["gain"])-r["gain"][a] for r,a in zip(rows, choices)]
        result[name] = dict(accuracy=mean(r["test"][a]["accuracy"] for r,a in zip(rows,choices)),
                            worst10=mean(r["test"][a]["worst10"] for r,a in zip(rows,choices)),
                            hits=regrets.count(0.), regret=mean(regrets))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-policy", required=True, type=Path)
    p.add_argument("--original-trees", required=True, type=Path)
    p.add_argument("--development-run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data-root", default=Path("dataset"),type=Path)
    p.add_argument("--jobs", default=4, type=int)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("Use at least one worker")
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    original = read(args.original_policy)
    held_seeds = list(range(341,357))
    write_json(root/"protocol.json", dict(fit_pools=[0,24,48],fit_seeds_per_pool=12,held_pool=72,
               held_seeds=held_seeds, score="shared ridge on candidate minus anchor error/CE gains at 36,37",
               comparison="raw round-40 accuracy vs greedy at same two-round probe cost; first policy frozen",
               candidate_extra_client_rounds=2))
    with ProcessPoolExecutor(args.jobs) as executor:
        jobs = []
        for offset, split in ((0,2026),(24,3055),(48,3079)):
            config = pool_config(args.data_root, root, offset, split)
            trees = args.original_trees if offset == 0 else args.development_run/f"clients{offset}"
            for s in (range(1,13) if offset==0 else range(201,213)):
                jobs.append(executor.submit(probe,config,s,root/f"clients{offset}"/str(s),original,
                                            tree_path=trees/str(s)/"tree.json"))
        head = fit_probe([j.result() for j in jobs])
        write_json(root/"probe_head.json",head)
        config = pool_config(args.data_root, root, 72, 3103)
        jobs = [executor.submit(probe,config,s,root/"clients72"/str(s),original,head=head) for s in held_seeds]
        rows = [j.result() for j in jobs]
    write_json(root/"comparison.json",dict(means=summary(rows),per_seed=rows))


if __name__ == "__main__":
    main()
