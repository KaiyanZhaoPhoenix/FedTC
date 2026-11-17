#!/usr/bin/env python
"""Run methods through main.py and verify actual comparison provenance."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from utils.experiment import write_json


def verify(root, algorithms):
    manifests, rounds, summaries = [], [], []
    for name in algorithms:
        folder = root / name
        manifests.append(json.loads((folder / 'manifest.json').read_text()))
        rounds.append([json.loads(line) for line in (folder / 'rounds.jsonl').read_text().splitlines()])
        summaries.append(json.loads((folder / 'summary.json').read_text()))
    for key in ('initial_model_sha256', 'native_data_sha256', 'private_splits_sha256', 'malicious_clients_harness_only'):
        if any(m[key] != manifests[0][key] for m in manifests[1:]):
            raise RuntimeError(f'Comparison mismatch: {key}')
    common_args = [{k: v for k, v in m['args'].items() if k not in ('algorithm', 'output')} for m in manifests]
    if any(args != common_args[0] for args in common_args[1:]):
        raise RuntimeError('Comparison mismatch: common training/evaluation configuration.')
    schedules = [[(r['upload_clients'], r['trained_clients']) for r in rows] for rows in rounds]
    if any(s != schedules[0] for s in schedules[1:]):
        raise RuntimeError('Comparison mismatch: training/upload schedules.')
    if any(s['status'] != 'complete' for s in summaries):
        raise RuntimeError('Comparison contains an incomplete run.')
    report = {'protocol_verified': True, 'algorithms': algorithms,
              'checks': ['native data', 'private splits', 'initial parameters', 'common training/evaluation configuration', 'malicious identities', 'training/upload schedules'],
              'results': summaries}
    write_json(root / 'comparison.json', report)
    lines = ['algorithm,rounds,final_accuracy,final_auc,elapsed_seconds,fallback_macros']
    for row in summaries:
        lines.append(','.join(str(row.get(k, '')) for k in ('algorithm', 'rounds', 'final_accuracy', 'final_auc', 'elapsed_seconds', 'fallback_macros')))
    (root / 'comparison.csv').write_text('\n'.join(lines) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--algorithms', nargs='+', default=['FedAA', 'FedTC', 'FedAvg', 'RFA', 'TrimmedMean', 'MultiKrum'])
    p.add_argument('--output-root', default=str(Path(__file__).resolve().parent / 'runs' / f'comparison_{datetime.now():%Y%m%d_%H%M%S}'))
    args, common = p.parse_known_args()
    blocked = {'--algorithm', '-algo', '--output', '--protocol', '--times', '-t', '--prev', '-pv'}
    if any(arg.split('=')[0] in blocked for arg in common):
        p.error('Comparison controls algorithm/output/protocol and runs one seed; pass --seed for the chosen seed.')
    if len(set(args.algorithms)) != len(args.algorithms):
        p.error('Algorithms must be unique.')
    root = Path(args.output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        p.error(f'Output root must be empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    for algorithm in args.algorithms:
        command = [sys.executable, str(Path(__file__).with_name('main.py')), '--algorithm', algorithm,
                   '--protocol', 'shared', '--output', str(root / algorithm), *common]
        print(f'Running {algorithm}; log: {root / (algorithm + ".log")}', flush=True)
        with (root / f'{algorithm}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify(root, args.algorithms)
    print(f'Comparison verified: {root / "comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
