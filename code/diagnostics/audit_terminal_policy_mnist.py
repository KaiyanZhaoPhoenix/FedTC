"""Replay frozen learned and round-only policies from matched round-30 snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from diagnostics.run_terminal_policy_mnist import (
    ACTIONS, BASE, choose_policy, collect_tree, decide, previous_policy, read, run_policy, write_json,
)


def private_cost(tree, pair):
    a, b = pair
    return tree["leaves"][a]["costs"][b]


def keep_candidate(tree, pair):
    anchor = (BASE, BASE)
    return pair if private_cost(tree, pair) < private_cost(tree, anchor) else anchor


def with_terminal_verification(tree):
    limit = private_cost(tree, (BASE, BASE))
    leaves = [dict(leaf, costs=[min(limit, c) for c in leaf["costs"]]) for leaf in tree["leaves"]]
    return dict(tree, leaves=leaves)


def table(rows, methods):
    comparison = {}
    for method in methods:
        controls = ("fixed_1.15", "terminal", "verified_schedule", "matched_budget_schedule", "terminal_budget_schedule")
        differences = {}
        for control in controls:
            gains = np.array([r[method]["accuracy"] - r[control]["accuracy"] for r in rows]) * 100
            differences[control] = dict(mean_pp=float(gains.mean()),wins=int((gains>0).sum()),
                                        ties=int((gains==0).sum()), per_seed_pp=gains.tolist())
        gains = [r[method]["private_gain"] for r in rows]
        proposal_rounds = [r[method]["extra_rounds"] for r in rows]
        comparison[method] = dict(
            mean_accuracy=float(np.mean([r[method]["accuracy"] for r in rows])),
            mean_worst10=float(np.mean([r[method]["worst10"] for r in rows])),
            harmful_private_departures=sum(g < 0 for g in gains),
            mean_private_gain=float(np.mean(gains)),
            mean_extra_rounds=float(np.mean(proposal_rounds)),
            comparisons=differences,
        )
    return comparison


def measure(tree, policy, old, schedule, guarded_schedule, option_policy, late_schedule):
    # Only features and private costs are read while constructing decisions.
    def greedy_old(features):
        return int(old.values(np.asarray(features)[:, :-1]).argmax())

    old_first = greedy_old(tree["features"])
    candidate = decide(tree, policy)
    option = decide(tree,option_policy)
    def match_budget(proposal):
        if proposal[0] != BASE:
            return guarded_schedule
        return late_schedule if proposal[1] != BASE else (BASE, BASE)

    verified_proposals = {
        "verified_schedule": guarded_schedule,
        "verified_terminal": candidate,
        "verified_option": option,
        "matched_budget_schedule": match_budget(option),
        "terminal_budget_schedule": match_budget(candidate),
    }
    pairs = {
        "fixed_1.15": (BASE,BASE),
        "old_last_two": (old_first,greedy_old(tree["leaves"][old_first]["features"])),
        "scheduled": schedule,
        "terminal": candidate,
        **{name:keep_candidate(tree,pair) for name,pair in verified_proposals.items()},
    }
    row = {}
    for name,(a,b) in pairs.items():
        leaf = tree["leaves"][a]
        proposal = verified_proposals.get(name, (BASE,BASE))
        extra = next((5*(2-i) for i,k in enumerate(proposal) if k != BASE), 0)
        row[name] = dict(accuracy=leaf["test"][b]["accuracy"], worst10=leaf["test"][b]["worst10"],
                         actions=[ACTIONS[a],ACTIONS[b]], extra_rounds=extra,
                         private_gain=private_cost(tree,(BASE,BASE))-leaf["costs"][b])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reuse-trees", action="store_true",
                        help="Score already-completed diagnostic trees without repeating local SGD")
    parser.add_argument("--verify-execution", action="store_true",
                        help="Also run the verified controller directly and check each retained endpoint")
    parser.add_argument("--bank", choices=("held", "confirm"), help="Score one bank with --reuse-trees")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.bank and not args.reuse_trees:
        parser.error("--bank requires --reuse-trees.")
    root = args.study.resolve()
    out = root/"decision_audit"
    if not args.reuse_trees:
        out.mkdir(exist_ok=False)
    plan = read(root/"plan.json")
    policy = read(root/"policy.json")
    write_json(out / "verified_terminal_policy.json", dict(policy, verify_final=True))
    old = previous_policy(args.source)
    training = [read(root/"calibration"/str(s)/"tree.json") for s in plan["fit"]]
    tuning = [read(root/"calibration"/str(s)/"tree.json") for s in plan["tune"]]
    option_policy, selection = choose_policy([with_terminal_verification(t) for t in training],
                                             [with_terminal_verification(t) for t in tuning])
    option_policy["verify_final"] = True
    write_json(out/"option_policy.json", option_policy)
    write_json(out/"option_selection.json", selection)
    all_pairs = list(product(range(len(ACTIONS)),repeat=2))
    schedule = min(all_pairs,key=lambda p:np.mean([private_cost(t,p) for t in training]))
    guarded_schedule = min(all_pairs,key=lambda p:np.mean([private_cost(t,keep_candidate(t,p)) for t in training]))
    write_json(out/"schedule_policy.json",dict(schedule=guarded_schedule,verify_final=True))
    late_schedule = min((p for p in all_pairs if p[0] == BASE and p[1] != BASE),
                        key=lambda p:np.mean([private_cost(t,keep_candidate(t,p)) for t in training]))
    write_json(out/"plan.json",dict(policy=policy, schedule=schedule, guarded_schedule=guarded_schedule,
        late_schedule=late_schedule,
        verification="one optional parallel fixed trajectory starting at round 30; choose by terminal private cost",
        comparison="paired same-budget schedule selected using calibration only; test never used for selection"))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for bank in ((args.bank,) if args.bank else ("held","confirm")):
            rows = []
            matched = read(root/bank/"results.json") if args.reuse_trees else None
            seeds = plan[bank]
            if args.reuse_trees:
                futures = [pool.submit(read, out/bank/str(s)/"tree.json") for s in seeds]
            else:
                futures = [pool.submit(collect_tree,plan["config"],s,out/bank/str(s),True) for s in seeds]
            for s,future in zip(seeds,futures):
                row = measure(future.result(),policy,old,schedule,guarded_schedule,option_policy,late_schedule)
                if matched is not None:
                    for method in ("terminal","fixed_1.15"):
                        if row[method]["accuracy"] != matched[str(s)][method]["final_accuracy"]:
                            raise ValueError(f"Counterfactual and direct execution differ at {s}, {method}")
                row["seed"] = s
                rows.append(row)
            if args.verify_execution:
                for method, rule, subdir in (("verified_terminal", policy, "terminal_execution"),
                                             ("verified_option", option_policy, "execution")):
                    tasks = []
                    for seed in seeds:
                        folder = out / subdir / str(seed)
                        if (folder / "decisions.json").exists():
                            task = pool.submit(read, folder / "decisions.json")
                        else:
                            task = pool.submit(run_policy, plan["config"], seed, folder, rule, True)
                        tasks.append(task)
                    for row, task in zip(rows, tasks):
                        actual = task.result()
                        stored = read(out / subdir / str(row["seed"]) / "terminal_selection.json")
                        expected = row[method]
                        if (actual[-1]["test"]["accuracy"] != expected["accuracy"]
                                or actual[-1]["test"]["worst10"] != expected["worst10"]
                                or stored["fork_rounds"] != expected["extra_rounds"]
                                or [r["tau"] for r in actual[-2:]] != expected["actions"]):
                            raise ValueError(f"Direct execution disagrees at seed {row['seed']}, {method}.")
                        expected["trajectory_verified"] = True
            write_json(out/bank/"rows.json",rows)
            methods = [k for k in rows[0] if k!="seed"]
            write_json(out/bank/"comparison.json",table(rows,methods))


if __name__ == "__main__":
    main()
