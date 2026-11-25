"""Fit a conservative rollout-improvement policy on existing calibration logs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.run_shadow_rl_mnist import GainModel, REFERENCE, evaluate, summarize_group
from utils.experiment import write_json


class AnchoredValues:
    def __init__(self, model, margin):
        self.model = model
        self.margin = margin

    def values(self, features):
        result = self.model.values(features) - self.margin
        result[self.model.ref] = 0.
        return result


def fit_anchor(groups, actions, horizon):
    """Use true paired continuation gains and keep each calibration seed intact."""
    table = []
    for steps in sorted({2, horizon}):
        for ridge in (4., 12., 48.):
            fits = {}
            for seed in groups:
                train = [row for other, rows in groups.items() if other != seed for row in rows]
                fits[seed] = GainModel(train, actions, regularization=ridge, target_horizon=steps)
            for margin in (.0005, .001, .002, .004, .008):
                lifts = []
                moves = []
                for seed, validation in groups.items():
                    model = AnchoredValues(fits[seed], margin)
                    gain = []
                    switched = []
                    for row in validation:
                        action = int(model.values(np.array(row["features"])).argmax())
                        gain.append(row["gain_long"][action])
                        switched.append(action != model.model.ref)
                    lifts.append(float(np.mean(gain)))
                    moves.append(sum(switched))
                table.append({"horizon": steps, "ridge": ridge, "margin": margin,
                              "mean_gain_vs_reference": float(np.mean(lifts)),
                              "gain_by_seed": lifts, "switches_by_seed": moves})
    selected = max(table, key=lambda item: (item["mean_gain_vs_reference"], item["margin"]))
    rows = [row for group in groups.values() for row in group]
    model = GainModel(rows, actions, regularization=selected["ridge"], target_horizon=selected["horizon"])
    return AnchoredValues(model, selected["margin"]), selected, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=["switched"])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[8, 10, 11])
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    previous = json.loads((args.source/"plan.json").read_text())
    seeds = previous["train_seeds"]
    if (len(seeds) < 2 or len(set(args.eval_seeds)) != len(args.eval_seeds)
            or set(args.eval_seeds) & set(seeds) or args.jobs < 1
            or set(args.scenarios) - set(previous["scenarios"])):
        parser.error("Use distinct held-out seeds and valid source scenarios.")
    config = {key: previous[key] for key in ("actions", "clients", "rounds", "horizon", "stride", "data_root")}
    groups = {seed: [] for seed in seeds}
    hashes = {}
    for scenario in args.scenarios:
        for seed in seeds:
            path = args.source/"calibration"/scenario/str(seed)/"decisions.jsonl"
            hashes[str(path.relative_to(args.source))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != config["rounds"]:
                raise ValueError(f"Calibration is incomplete: {path}.")
            groups[seed].extend(row for row in rows if "gain_long" in row)
    model, selected, grid = fit_anchor(groups, config["actions"], config["horizon"])
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"plan.json", {
        **config, "source": str(args.source.resolve()), "calibration_sha256": hashes,
        "eval_seeds": args.eval_seeds, "scenarios": args.scenarios, "methods": ["learned"],
        "reference_tau": REFERENCE, "fit_controls": selected,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "criterion": "Mean tail test accuracy > best fixed and validation greedy by >=0.10 pp "
                     "and joint wins on at least 2/3 independent seeds.",
    })
    write_json(root/"selection.json", {"selected": selected, "grid": grid})
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        jobs = {(case, seed): pool.submit(evaluate, config, seed, case,
                                         root/"evaluation"/case/str(seed)/"learned", "learned", model)
                for case in args.scenarios for seed in args.eval_seeds}
        for job in jobs.values():
            job.result()
    results = {
        case: {seed: summarize_group(root, case, seed, ["learned"], config["rounds"])
               for seed in args.eval_seeds}
        for case in args.scenarios
    }
    write_json(root/"summary.json", {"results": results})


if __name__ == "__main__":
    main()
