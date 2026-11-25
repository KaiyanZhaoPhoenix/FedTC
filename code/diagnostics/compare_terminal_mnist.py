"""Separate terminal policy choice from endpoint verification on matched MNIST branches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean

from diagnostics.run_terminal_policy_mnist import (
    ACTIONS, advantages, collect_tree, decide, read, select, write_json,
)


def decision_audit(tree, policy):
    """Compare each prediction to paired private costs along the deployed continuation."""
    first, last = decide(tree, policy)
    first_costs = []
    for leaf in tree["leaves"]:
        continuation = select(policy["last"], leaf["features"], policy["margin"])
        first_costs.append(leaf["costs"][continuation])
    reached = tree["leaves"][first]

    def stage(round_number, head, features, chosen, costs):
        best = min(range(len(costs)), key=costs.__getitem__)
        return dict(round=round_number, selected_tau=ACTIONS[chosen], best_private_tau=ACTIONS[best],
                    selected_is_best=costs[chosen] == costs[best], private_regret=costs[chosen]-costs[best],
                    predicted_gain=advantages(head, features).tolist(),
                    measured_gain=[costs[0]-value for value in costs])

    return [stage(31, policy["first"], tree["features"], first, first_costs),
            stage(36, policy["last"], reached["features"], last, reached["costs"])]


def score_path(tree, proposal, verify):
    """Evaluate one policy path, comparing private costs before reading its test result."""
    base = (0, 0)
    anchor_cost = tree["leaves"][0]["costs"][0]
    candidate_cost = tree["leaves"][proposal[0]]["costs"][proposal[1]]
    selected = base if verify and candidate_cost >= anchor_cost else proposal
    extra = next((10 - 5*i for i, a in enumerate(proposal) if a), 0) if verify else 0
    test = tree["leaves"][selected[0]]["test"][selected[1]]
    return dict(proposed_tau=[ACTIONS[a] for a in proposal], kept_tau=[ACTIONS[a] for a in selected],
                private_gain=anchor_cost-candidate_cost, extra_rounds=extra, **test)


def ablate(tree, policy):
    base = (0, 0)
    learned = decide(tree, policy)
    strong = (2, 2)
    # Same opening window and branch compute as the learned proposal.
    timed = base
    if learned[0] != 0:
        timed = strong
    elif learned[1] != 0:
        timed = (0, 2)

    return dict(seed=tree["seed"], fixed=score_path(tree, base, False), learned=score_path(tree, learned, False),
                learned_verified=score_path(tree, learned, True), schedule=score_path(tree, strong, False),
                schedule_verified=score_path(tree, strong, True), matched_verified=score_path(tree, timed, True))


def summarize(rows):
    methods = [m for m in rows[0] if m != "seed"]
    averages = {m: {k: mean(r[m][k] for r in rows) for k in ("accuracy", "worst10", "extra_rounds")}
                for m in methods}
    diffs = {}
    for control in ("fixed", "learned", "schedule_verified", "matched_verified"):
        diff = [100*(r["learned_verified"]["accuracy"]-r[control]["accuracy"]) for r in rows]
        diffs[control] = dict(mean_pp=mean(diff), wins=sum(d > 0 for d in diff),
                              ties=diff.count(0.), losses=sum(d < 0 for d in diff))
    return dict(means=averages, learned_verified_minus=diffs, per_seed=rows)


def collect_and_score(config, seed, folder, policy):
    tree = collect_tree(config, seed, folder, measure_test=True)
    write_json(folder/"decision_audit.json", decision_audit(tree, policy))
    return ablate(tree, policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--jobs", default=4, type=int)
    parser.add_argument("--data-root", default=Path(__file__).resolve().parents[1]/"dataset", type=Path)
    parser.add_argument("--client-offset", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or args.jobs < 1
            or not 0 <= args.client_offset <= 76 or args.split_seed < 0):
        parser.error("Use distinct nonnegative seeds, positive jobs, and 24 available client IDs.")
    policy = read(args.policy)
    root = args.output.resolve()
    data = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.client_offset:
        mapped = root/"client_data"
        for part in ("train", "test"):
            dest = mapped/"mnist"/part
            dest.mkdir(parents=True)
            for i in range(24):
                source = data/"mnist"/part/f"{args.client_offset+i}.npz"
                (dest/f"{i}.npz").symlink_to(source.resolve(strict=True))
        data = mapped
    config = dict(rounds=40, clients=24, actions=ACTIONS, announced_cohorts=True, validation_fraction=.25,
                  data_root=str(data), split_seed=args.split_seed)
    write_json(root/"protocol.json", dict(seeds=args.seeds, client_offset=args.client_offset,
                                         split_seed=args.split_seed, policy=policy))
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs = [pool.submit(collect_and_score, config, seed, root/str(seed), policy) for seed in args.seeds]
        rows = [job.result() for job in jobs]
    write_json(root/"comparison.json", summarize(rows))


if __name__ == "__main__":
    main()
