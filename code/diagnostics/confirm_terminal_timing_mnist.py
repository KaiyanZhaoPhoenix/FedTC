"""Confirm whether state-dependent timing helps at equal branch compute on fresh seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from diagnostics.audit_terminal_policy_mnist import keep_candidate
from diagnostics.run_terminal_policy_mnist import (
    ACTIONS, BASE, collect_tree, decide, evaluate, fit_head, read, run_policy, select, write_json,
)


def train_timing(trees, cost_per_round):
    strong = ACTIONS.index(1.55)
    if strong != 2 or BASE != 0:
        raise ValueError("Stopping policy expects the reference first and the strong action last.")
    # Costs relative to retaining the anchor. The middle action is a redundant wait,
    # allowing the same deployed advantage scorer to represent the two choices.
    late_features, late_costs, early_gains = [], [], []
    for tree in trees:
        late = tree["leaves"][BASE]
        reference = late["costs"][BASE]
        late_gain = max(0., reference - late["costs"][strong])
        early_gain = max(0., reference - tree["leaves"][strong]["costs"][strong])
        late_features.append(late["features"])
        late_costs.append([0., 0., 5 * cost_per_round - late_gain])
        early_gains.append(early_gain)
    last = fit_head(late_features, late_costs)
    targets = []
    for features, costs, early_gain in zip(late_features, late_costs, early_gains):
        wait = costs[select(last, features, 0.)]
        targets.append([wait, wait, 10 * cost_per_round - early_gain])
    first = fit_head([t["features"] for t in trees], targets)
    return dict(first=first, last=last, margin=0., verify_final=True, timing_only=True,
                cost_per_branch_round=cost_per_round)


def duration(tree, policy):
    # After opening, the strong branch is committed through the terminal checkpoint.
    if select(policy["first"], tree["features"], policy["margin"]) != BASE:
        return 10
    late = tree["leaves"][BASE]
    return 5 if select(policy["last"], late["features"], policy["margin"]) != BASE else 0


def calibrate_timing(root):
    plan = read(root / "plan.json")
    fit = [read(root / "calibration" / str(s) / "tree.json") for s in plan["fit"]]
    tune = [read(root / "calibration" / str(s) / "tree.json") for s in plan["tune"]]
    trials = []
    pairs = {0:(BASE,BASE), 5:(BASE,2), 10:(2,2)}
    for penalty in (0., .0001, .00025, .0005, .001, .002, .005):
        policy = train_timing(fit, penalty)
        gains, rounds = [], []
        for tree in tune:
            d = duration(tree, policy)
            a, b = keep_candidate(tree, pairs[d])
            gains.append(tree["leaves"][BASE]["costs"][BASE] - tree["leaves"][a]["costs"][b])
            rounds.append(d)
        trials.append(dict(penalty=penalty, mean_extra_rounds=float(np.mean(rounds)),
                           private_gain=float(np.mean(gains))))
    eligible = [trial for trial in trials if trial["mean_extra_rounds"] <= 7.5]
    choice = max(eligible, key=lambda r:(r["private_gain"],-r["mean_extra_rounds"]))
    return train_timing(fit, choice["penalty"]), trials


def confirmed_seed(config, policy, seed, folder):
    tree = collect_tree(config, seed, folder / 'branches', measure_test=True)
    start = duration(tree, policy)
    strong = ACTIONS.index(1.55)
    proposals = {0:(BASE, BASE), 5:(BASE, strong), 10:(strong, strong)}
    options = {}
    for cost, proposed in proposals.items():
        a, b = keep_candidate(tree, proposed)
        options[cost] = tree['leaves'][a]['test'][b]
    fixed = {str(ACTIONS[BASE]): options[0]['accuracy']}
    for tau in ACTIONS[1:]:
        rows = evaluate(config, seed, folder / f'fixed_{tau}', f'fixed_{tau}', None)
        fixed[str(tau)] = rows[-1]['test']['accuracy']
        for name in ('manifest.json', 'study_manifest.json'):
            same = read(folder / 'branches' / name)
            other = read(folder / f'fixed_{tau}' / name)
            fields = ('initial_model_sha256', 'private_splits_sha256', 'native_data_sha256') if name == 'manifest.json' else ('upload_schedule_sha256',)
            if any(same[k] != other[k] for k in fields):
                raise ValueError(f'Unmatched {name} for seed {seed}')
    rows = run_policy(config, seed, folder / 'deployed', policy, verify_final=True)
    decision = read(folder / 'deployed' / 'terminal_selection.json')
    if rows[-1]['test'] != options[start] or decision['fork_rounds'] != start:
        raise ValueError(f'Frozen timing policy differs from paired replay at seed {seed}')
    return dict(seed=seed, duration=start, options=options, fixed=fixed,
                verified=decision, final=options[start]['accuracy'])


def summarize(rows):
    durations = [r['duration'] for r in rows]
    counts = {d:durations.count(d) for d in (0,5,10)}
    best_tau = max(rows[0]['fixed'], key=lambda tau:np.mean([r['fixed'][tau] for r in rows]))
    endpoint = np.array([r['final'] for r in rows])
    controls = {
        'fixed_1.15':[r['fixed']['1.15'] for r in rows],
        'strongest_fixed':[r['fixed'][best_tau] for r in rows],
        'always_five':[r['options'][5]['accuracy'] for r in rows],
        'always_ten':[r['options'][10]['accuracy'] for r in rows],
        # Exact expected score over all random permutations of the same fork durations.
        'matched_random_timing':[sum(counts[d]*r['options'][d]['accuracy'] for d in counts)/len(rows)
                                 for r in rows],
    }
    observed = float(endpoint.mean())
    matrix = np.array([[r['options'][d]['accuracy'] for d in (0,5,10)] for r in rows])
    schedule = np.array(durations)//5
    rng = np.random.default_rng(20261124)
    random_means = [matrix[np.arange(len(rows)),rng.permutation(schedule)].mean() for _ in range(10000)]
    comparisons = {}
    for name, control in controls.items():
        delta = 100*(endpoint - control)
        comparisons[name] = dict(control_pct=float(100*np.mean(control)), delta_pp=float(delta.mean()),
                                 wins=int((delta>1e-10).sum()), ties=int((np.abs(delta)<=1e-10).sum()))
    return dict(n=len(rows), final_pct=observed*100, durations=counts, mean_extra_rounds=float(np.mean(durations)),
                strongest_fixed_tau=best_tau, comparisons=comparisons,
                random_timing_one_sided_p=(1+int((np.asarray(random_means)>=observed-1e-12).sum()))/10001)


def compare_frozen_terminal(root, policy, results):
    rows = []
    for result in results:
        tree = read(root / str(result["seed"]) / "branches" / "tree.json")
        proposed = decide(tree, policy)
        a, b = keep_candidate(tree, proposed)
        metrics = tree["leaves"][a]["test"][b]
        rows.append(dict(seed=result["seed"], proposed=proposed, kept=[a,b], accuracy=metrics["accuracy"],
                         worst10=metrics["worst10"], anchor=result["options"][0]["accuracy"],
                         schedule=result["options"][10]["accuracy"],
                         extra_rounds=next((5*(2-i) for i, k in enumerate(proposed) if k != BASE), 0)))
    gain = 100 * np.array([row["accuracy"] - row["anchor"] for row in rows])
    return dict(n=len(rows), final_pct=float(100 * np.mean([row["accuracy"] for row in rows])),
                mean_gain_vs_fixed_pp=float(gain.mean()), wins=int((gain>0).sum()),
                ties=int((gain==0).sum()), losses=int((gain<0).sum()),
                mean_extra=float(np.mean([r["extra_rounds"] for r in rows])),
                mean_vs_schedule_pp=float(100*np.mean([r["accuracy"]-r["schedule"] for r in rows])), rows=rows)


def evaluate_stopping(root, config, policy, original):
    seed = original["seed"]
    folder = root / str(seed)
    tree = read(folder / "branches" / "tree.json")
    start = duration(tree, policy)
    path = folder / "optimal_stopping" / "decisions.json"
    if path.exists():
        rows = read(path)
    else:
        rows = run_policy(config, seed, folder / "optimal_stopping", policy, verify_final=True)
    decision = read(folder / "optimal_stopping" / "terminal_selection.json")
    options = {int(k): v for k, v in original["options"].items()}
    if rows[-1]["test"] != options[start] or decision["fork_rounds"] != start:
        raise ValueError(f"Direct stopping policy differs at seed {seed}")
    return dict(seed=seed, duration=start, options=options, fixed=original["fixed"],
                verified=decision, final=options[start]["accuracy"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--stopping',action='store_true',
                        help='Confirm the learned stopping rule using paired outcomes from the timing study')
    args=parser.parse_args()
    out=args.output.resolve()
    config=read(args.study/'plan.json')['config']
    config['data_root']=str(Path(__file__).resolve().parents[1]/'dataset')
    if args.stopping:
        policy, choices = calibrate_timing(args.study)
        saved = out / "optimal_stopping_policy.json"
        if saved.exists() and policy != read(saved):
            raise ValueError("Stopping rule differs from the frozen artifact.")
        write_json(saved, policy)
        write_json(out / "optimal_stopping_calibration.json", choices)
        originals = read(out / "results.json")
        if [r["seed"] for r in originals] != list(range(66,82)):
            raise ValueError("A complete fresh-seed timing study is required.")
        with ProcessPoolExecutor(args.jobs) as pool:
            jobs = [pool.submit(evaluate_stopping,out,config,policy,row) for row in originals]
            results = [job.result() for job in jobs]
        write_json(out / "optimal_stopping_results.json", results)
        write_json(out / "optimal_stopping_comparison.json", summarize(results))
        return
    out.mkdir(parents=True,exist_ok=False)
    policy=read(args.study/'policy.json')
    policy.update(timing_only=True, verify_final=True)
    seeds=list(range(66,82))
    write_json(out/'policy.json', policy)
    write_json(out/'plan.json',dict(seeds=seeds, reference=1.15, trial=1.55, config=config,
        rule='Frozen round-30/35 controller triggers at most one persistent branch through round 40.',
        criterion='Mean final test gain >=0.10 pp vs strongest fixed and vs matched random timing; one-sided permutation p<0.05 for timing.'))
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs=[pool.submit(confirmed_seed,config,policy,s,out/str(s)) for s in seeds]
        results=[]
        for job in jobs:
            results.append(job.result())
            write_json(out/'results.json', results)
    write_json(out/'comparison.json', summarize(results))
    write_json(out/'frozen_terminal_policy_comparison.json',
               compare_frozen_terminal(out, read(args.study/'policy.json'), results))


if __name__=='__main__':
    main()
