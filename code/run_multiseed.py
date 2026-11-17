#!/usr/bin/env python
"""Repeat the shared comparison protocol over paired seeds and summarize every run."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t

from run_comparison import verify
from utils.experiment import write_json


def statistics(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Statistics require at least two finite seed results.')
    mean, std = float(values.mean()), float(values.std(ddof=1))
    radius = float(student_t.ppf(.975, len(values)-1) * std / np.sqrt(len(values)))
    return {'n': len(values), 'mean': mean, 'sample_std': std,
            'min': float(values.min()), 'max': float(values.max()),
            't95_low': mean-radius, 't95_high': mean+radius}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, algorithms, seeds):
    records, manifests = [], []
    for seed in seeds:
        directory = root / f'seed_{seed}'
        verify(directory, algorithms)
        for algorithm in algorithms:
            path = directory / algorithm
            result = json.loads((path / 'summary.json').read_text())
            manifest = json.loads((path / 'manifest.json').read_text())
            if manifest['args']['seed'] != seed:
                raise RuntimeError('Unexpected seed in manifest.')
            manifests.append(manifest)
            macro_file = path / 'macros.jsonl'
            macros = [json.loads(line) for line in macro_file.read_text().splitlines()] if macro_file.exists() else []
            gates = [m for m in macros if m.get('gate_reason') in ('accepted', 'value_gap')]
            records.append({
                'seed': seed, 'algorithm': algorithm, 'rounds': result['rounds'],
                'final_accuracy_pct': 100*result['final_accuracy'],
                'final_auc': result['final_auc'],
                'controller_train_steps': result.get('controller_train_steps', 0),
                'warmup_macros': sum(m.get('gate_reason') == 'warmup' for m in macros),
                'postwarmup_decisions': len(gates),
                'accepted_macros': sum(m.get('gate_reason') == 'accepted' for m in macros),
                'postwarmup_fallbacks': sum(m.get('gate_reason') == 'value_gap' for m in macros),
                'unique_executed_actions': len({tuple(m['action']) for m in macros}),
                'mean_interval_width': float(np.mean([m['high']-m['low'] for m in macros])) if macros else None,
                'elapsed_seconds': result['elapsed_seconds'],
            })
    for key in ('native_data_sha256', 'private_splits_sha256'):
        if len({m[key] for m in manifests}) != 1:
            raise RuntimeError(f'Data partition changed across seeds: {key}')
    common = [{k: v for k, v in m['args'].items() if k not in ('seed', 'algorithm', 'output')} for m in manifests]
    if any(c != common[0] for c in common[1:]):
        raise RuntimeError('Settings other than seed/method/output changed.')
    summaries = []
    for algorithm in algorithms:
        entries = [r for r in records if r['algorithm'] == algorithm]
        summaries.append({'algorithm': algorithm,
                          **statistics([r['final_accuracy_pct'] for r in entries]),
                          'total_postwarmup_decisions': sum(r['postwarmup_decisions'] for r in entries),
                          'total_accepted_macros': sum(r['accepted_macros'] for r in entries),
                          'total_postwarmup_fallbacks': sum(r['postwarmup_fallbacks'] for r in entries)})
    lookup = {(r['algorithm'], r['seed']): r['final_accuracy_pct'] for r in records}
    comparisons = []
    for first, second in [('FedTC', 'FedAA'), ('FedTCFixed', 'FedAA'), ('FedTC', 'FedTCFixed')]:
        if first not in algorithms or second not in algorithms:
            continue
        differences = np.array([lookup[first, seed]-lookup[second, seed] for seed in seeds])
        comparisons.append({'first': first, 'second': second, 'unit': 'percentage_points',
                            **statistics(differences), 'seed_differences': differences.tolist(),
                            'wins': int((differences > 1e-9).sum()),
                            'ties': int((np.abs(differences) <= 1e-9).sum()),
                            'losses': int((differences < -1e-9).sum())})
    report = {'status': 'complete', 'seeds': seeds, 'run_count': len(records),
              'protocol_verified': True, 'fixed_native_partition': True,
              'fixed_validation_split': True,
              'common_settings': common[0], 'algorithms': summaries,
              'paired_comparisons': comparisons, 'per_seed': records,
              'interval_note': 'Exploratory 95% Student-t intervals over seeds; small-n normal approximation, no multiplicity correction. Paired differences use the same seed. Sample SD uses ddof=1.',
              'scope_note': 'Seeds vary initialization, malicious identities, training/upload randomness and controller RNG. They do not regenerate the native client partition or validation holdout. Short runs do not establish convergence or calibrated critic confidence.'}
    write_json(root / 'aggregate_summary.json', report)
    write_csv(root / 'per_seed.csv', records)
    write_csv(root / 'aggregate.csv', summaries)
    if comparisons:
        write_csv(root / 'paired_comparisons.csv', comparisons)
    lines = [f'Completed {len(records)} runs; seeds={seeds}. Accuracy is percent, SD is across seeds.', '']
    for entry in summaries:
        lines.append(f'{entry["algorithm"]}: {entry["mean"]:.2f} +/- {entry["sample_std"]:.2f}; range [{entry["min"]:.2f}, {entry["max"]:.2f}]')
    lines.append('')
    for entry in comparisons:
        lines.append(f'{entry["first"]} - {entry["second"]}: {entry["mean"]:+.2f} pp; paired t95 [{entry["t95_low"]:+.2f}, {entry["t95_high"]:+.2f}]; W/T/L={entry["wins"]}/{entry["ties"]}/{entry["losses"]}')
    lines += ['', report['scope_note'], report['interval_note']]
    (root / 'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--algorithms', nargs='+', default=['FedAA', 'FedTCFixed', 'FedTC'])
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--output-root', default=f'runs/multiseed_{datetime.now():%Y%m%d_%H%M%S}')
    args, common = parser.parse_known_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Provide at least two distinct nonnegative seeds.')
    if args.jobs < 1 or not args.algorithms or len(set(args.algorithms)) != len(args.algorithms):
        parser.error('jobs must be positive and algorithms must be unique.')
    blocked = {'--seed', '-seed', '--algorithm', '-algo', '--output', '--protocol', '--times', '-t', '--prev', '-pv'}
    if any(arg.split('=')[0] in blocked for arg in common):
        parser.error('Use this runner to specify seeds/algorithms/output; protocol is shared.')
    root = Path(args.output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f'Choose an empty output directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parent
    paths = [project / 'main.py', project / 'run_comparison.py', Path(__file__).resolve()]
    for directory in ('algorithms', 'clients', 'servers', 'utils', 'DDPG'):
        paths.extend((project / directory).rglob('*.py'))
    write_json(root / 'study_manifest.json', {
        'argv': sys.argv, 'python': sys.version, 'seeds': args.seeds,
        'algorithms': args.algorithms, 'jobs': args.jobs, 'common_cli': common,
        'source_sha256': {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},
        'note': 'All declared seeds included; no model/hyperparameter changes during this study. Timing is not comparable under concurrent execution.'})

    def run_seed(seed):
        command = [sys.executable, str(project / 'run_comparison.py'), '--output-root', str(root / f'seed_{seed}'),
                   '--algorithms', *args.algorithms, '--seed', str(seed), *common]
        print(f'Seed {seed}: started', flush=True)
        with (root / f'seed_{seed}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(f'Seed {seed}: {"complete" if result.returncode == 0 else "FAILED"}', flush=True)
        return seed, result.returncode

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(run_seed, seed) for seed in args.seeds]):
            seed, code = future.result()
            if code:
                failed.append({'seed': seed, 'exit_code': code})
    if failed:
        write_json(root / 'failures.json', failed)
        raise SystemExit('Some declared runs failed; no aggregate result produced. Inspect seed logs.')
    summarize(root, args.algorithms, args.seeds)


if __name__ == '__main__':
    main()
