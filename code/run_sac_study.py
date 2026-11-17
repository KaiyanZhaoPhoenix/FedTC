"""Predeclared seed=1 study: tune a constant, honest feedback, then forged feedback."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from run_comparison import verify
from utils.experiment import write_json

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ['FedTCFixed', 'FedTC', 'FedTCSAC', 'FedTCSACGate']
REPORTS = ['none', 'all_max', 'echo', 'bimodal']
GRID = [(lam, tau) for lam in (0., .25, .5) for tau in (1.05, 1.35, 1.8)]


def common(rounds):
    return ['--seed', '1', '--device', 'cpu', '--dataset', 'mnist',
            '--num_clients', '100', '--num_sel_clients', '100', '--aggre_num', '50',
            '--global_rounds', str(rounds), '--local_epochs', '1', '--batch_size', '16',
            '--local_learning_rate', '.1', '--eval_gap', '10', '--eval-clients', '20',
            '--macro-length', '2', '--tc-batch-size', '16', '--warmup-macros', '16',
            '--tc-updates', '4', '--tc-actor-lr', '.0003', '--tc-critic-lr', '.0003',
            '--buffer_size', '512', '--td-mode', 'paper', '--tc-gamma', '.9',
            '--ad_ratio', '.2', '--rho-max', '.2', '--loss-cap', '5',
            '--split-seed', '2026', '--validation-fraction', '.1',
            '--training-cohort', 'all', '--threads', '1',
            '--attack', 'same_value', '--fake-utility', '--audit-validation',
            '--gate-tolerance', '.05', '--exploration-std', '0',
            '--sac-alpha', '.1', '--sac-alpha-lr', '.0003', '--sac-target-entropy', '-2']


def command(folder, algorithm, rounds, report, lam, tau, audit):
    return [sys.executable, str(ROOT/'main.py'), '--algorithm', algorithm, '--output', str(folder),
            *common(rounds), '--report-attack', report, '--fixed-lambda', str(lam),
            '--fixed-tau', str(tau), '--audit-shadow-every', str(audit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=f'runs/sac_study_{datetime.now():%Y%m%d_%H%M%S}')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('jobs must be positive.')
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [ROOT/'main.py', ROOT/'run_comparison.py', Path(__file__).resolve()]
    for directory in ('algorithms', 'clients', 'servers', 'utils', 'DDPG'):
        sources.extend((ROOT/directory).rglob('*.py'))
    plan = {
        'created_at': datetime.now().isoformat(), 'seed': 1, 'jobs': args.jobs,
        'algorithms': ALGORITHMS, 'reports': REPORTS, 'main_rounds': 300, 'tuning_rounds': 80,
        'tuning_grid': GRID, 'common_main_args': common(300), 'common_tuning_args': common(80),
        'tuning_selection': 'Minimum mean all-client private validation capped loss over final 20% of tuning rounds; grid order breaks ties.',
        'tuning_rule': 'Tune once with honest reports; freeze chosen constant across all report attacks. Test scores never select it.',
        'primary_metric': 'Mean honest-client sample-weighted test accuracy over final 20% of rounds (six evaluations).',
        'secondary_metrics': ['final accuracy', 'final honest-client macro accuracy', 'post-hoc bottom20% client accuracy',
                              'executed actions', 'gate rejection', 'one-round paired shadow improvement'],
        'main_runs': 16, 'tuning_runs': 9,
        'design_notes': [
            'One seed only. No confidence intervals or significance claims.',
            'Honest means all validation reports are truthful. Same-value forward poisoning and fake utility remain ON in both stages.',
            'Native MNIST100 partition, same initial model, local SGD, cohorts and held-out splits in all runs.',
            'Original IntervalController source is preserved; paper TD and PopArt are retained as an explicit historical reference.',
            'All learned controllers: H=2, batch16, warmup16, 4 updates/macro, learning rates3e-4. This differs from the previous batch2 short study.',
            'SAC uses midpoint reward, twin soft Q, stochastic tanh Gaussian, auto entropy temperature. Entropy in normalized action coordinates.',
            'SAC gate uses two SEPARATE entropy-free endpoint critics with raw returns, tolerance0.05. SAC input loss fields divide by5; no PopArt.',
            'SAC no-gate trains the same endpoint critics for matched computation; gate execution is their only algorithmic difference.',
            'Endpoint critics bootstrap proposed future SAC actions. They do not certify an always-conservative policy or a safe deployed trajectory.',
            'SAC vs original changes both optimizer and reward/value design; only SAC gate on/off is a controlled gate ablation.',
            'Warmup executes the conservative action in all learned methods; after warmup, no-gate SAC executes its stochastic samples.',
            'Additional shadow checks use honest identities only in isolated audit logs, never as a training signal. They compare one round, not full macro/return.',
            'Validation tuning gives the fixed method 9x80 extra rounds; it is a tuned reference, not an equal-total-compute comparison.',
            'No evaluations of a distinct frozen policy, new datasets, predefined minority groups or convergence guarantees.',
        ],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))},
    }
    write_json(output/'plan.json', plan)
    done, failures = [], []

    def stage(name, entries):
        write_json(output/'status.json', {'status': 'running', 'stage': name, 'completed': done, 'failures': failures})
        def run(entry):
            identifier, cmd = entry
            print('START', identifier, flush=True)
            log_path = output/(identifier.replace('/', '__')+'.log')
            with log_path.open('w') as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            return {'id': identifier, 'exit_code': result.returncode}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, e) for e in entries]
            for future in as_completed(futures):
                result = future.result()
                (done if result['exit_code'] == 0 else failures).append(result)
                write_json(output/'status.json', {'status': 'running', 'stage': name, 'completed': done, 'failures': failures})
                print('DONE' if result['exit_code'] == 0 else 'FAILED', result['id'], flush=True)
        if failures:
            write_json(output/'status.json', {'status': 'failed', 'stage': name, 'completed': done, 'failures': failures})
            raise RuntimeError('Study stage failed; inspect its saved logs.')

    tuning = [(f'tuning/grid_{i}', command(output/'tuning'/f'grid_{i}', 'FedTCFixed', 80, 'none', lam, tau, 0))
              for i, (lam, tau) in enumerate(GRID)]
    write_json(output/'tuning_commands.json', [{'id': i, 'command': c} for i, c in tuning])
    stage('fixed_parameter_validation_tuning', tuning)
    scores = []
    for i, (lam, tau) in enumerate(GRID):
        rows = [json.loads(line) for line in (output/'tuning'/f'grid_{i}'/'validation_audit.jsonl').read_text().splitlines()]
        values = [r['all_client_macro_loss'] for r in rows if r['round'] > .8*80]
        scores.append({'grid': i, 'lambda': lam, 'tau': tau, 'validation_score': sum(values)/len(values)})
    selected = min(scores, key=lambda row: row['validation_score'])
    write_json(output/'fixed_selection.json', {'selected': selected, 'scores': scores, 'used_test_metrics': False})
    print('FIXED SELECTED', selected, flush=True)
    all_entries = {}
    for report in REPORTS:
        all_entries[report] = [(f'main/{report}/{algorithm}', command(output/'main'/report/algorithm,
                               algorithm, 300, report, selected['lambda'], selected['tau'], 25)) for algorithm in ALGORITHMS]
    write_json(output/'main_commands.json', [{'id': i, 'command': c} for group in all_entries.values() for i, c in group])
    stage('honest_feedback', all_entries['none'])
    verify(output/'main'/'none', ALGORITHMS)
    stage('forged_feedback', [e for report in REPORTS[1:] for e in all_entries[report]])
    for report in REPORTS[1:]:
        verify(output/'main'/report, ALGORITHMS)
    for path, digest in plan['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source changed during experiment: {path}')
    write_json(output/'status.json', {'status': 'complete', 'completed': done, 'failures': failures})
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
