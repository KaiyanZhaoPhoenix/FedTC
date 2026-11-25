"""Choose round 31 for the actual continuation that verifies two alternatives at round 40."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

import numpy as np

from diagnostics.refit_last_mnist import pool_config
from diagnostics.run_terminal_policy_mnist import ACTIONS, collect_tree, fit_head, read, select, write_json


def terminal_choice(leaf):
    return 2 if leaf["costs"][2] < leaf["costs"][0] else 0


def continuation_costs(tree):
    return [leaf["costs"][terminal_choice(leaf)] for leaf in tree["leaves"]]


def evaluate(tree, head, constant):
    x = np.asarray(tree["features"])
    costs = continuation_costs(tree)
    choices = dict(fitted=select(head, x, 0.), fixed_115=0, calibrated_constant=constant,
                   immediate=int((x[:, -1]+.05*(x[:,0]-x[0,0])).argmax()))
    methods = {}
    for method, action in choices.items():
        leaf = tree["leaves"][action]
        last = terminal_choice(leaf)
        methods[method] = dict(first_tau=ACTIONS[action], last_tau=ACTIONS[last],
                               regret=costs[action]-min(costs), **leaf["test"][last])
    return dict(seed=tree["seed"], methods=methods)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-trees", type=Path, required=True)
    p.add_argument("--development-run", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("dataset"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("Use at least one worker")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    train = []
    for offset in (0,24,48):
        src = args.original_trees if offset == 0 else args.development_run/f"clients{offset}"
        train.extend(read(src/str(s)/"tree.json") for s in (range(1,17) if offset==0 else range(201,217)))
    costs = np.array([continuation_costs(t) for t in train])
    head = fit_head([t["features"] for t in train], costs)
    constant = int(costs.mean(axis=0).argmin())
    seeds = list(range(361,377))
    write_json(root/"first_head.json", head)
    write_json(root/"protocol.json", dict(holdout=seeds, constant_tau=ACTIONS[constant], extra_rounds=5,
               training_pools=[0,24,48], held_pool=72, split_seed=3103,
               target="lower private cost of late 1.15/1.55 from each first-action successor",
               criterion="round-40 accuracy at the same 5 extra branch rounds vs fixed and immediate"))
    config = pool_config(args.data_root, root, 72, 3103)
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs = [pool.submit(collect_tree, config, s, root/"clients72"/str(s), True) for s in seeds]
        rows = [evaluate(j.result(),head,constant) for j in jobs]
    means = {name:{field:mean(r["methods"][name][field] for r in rows)
                   for field in ("accuracy","worst10","regret")} for name in rows[0]["methods"]}
    write_json(root/"comparison.json",dict(means=means, per_seed=rows))


if __name__ == "__main__":
    main()
