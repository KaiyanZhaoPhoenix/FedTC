#!/usr/bin/env python
"""Predeclared single-seed attack and training-budget study, using the shared FedAA pipeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from utils.experiment import write_json

ALGORITHMS = ['FedAA', 'FedTCFixed', 'FedTC']
SCENARIOS = {
    'clean': {'label': '无攻击（攻击者名单保持一致）', 'attack': 'none', 'report_attack': 'none', 'fake_utility': False},
    'same_value': {'label': '同值模型投毒', 'attack': 'same_value', 'report_attack': 'none', 'fake_utility': False},
    'gaussian': {'label': '高斯模型投毒', 'attack': 'gaussian', 'report_attack': 'none', 'fake_utility': False},
    'delta_sign_flip': {'label': '更新反号 ×10', 'attack': 'delta_sign_flip', 'report_attack': 'none', 'fake_utility': False},
    'feedback_only': {'label': '仅 All-Max 反馈造假', 'attack': 'none', 'report_attack': 'all_max', 'fake_utility': False},
    'joint_all_max': {'label': '同值投毒 + 虚假效用 + All-Max', 'attack': 'same_value', 'report_attack': 'all_max', 'fake_utility': True},
    'joint_echo': {'label': '同值投毒 + 虚假效用 + Echo', 'attack': 'same_value', 'report_attack': 'echo', 'fake_utility': True},
    'joint_bimodal': {'label': '同值投毒 + 虚假效用 + Bimodal', 'attack': 'same_value', 'report_attack': 'bimodal', 'fake_utility': True},
}
PROFILES = {
    'small12': {'clients': 12, 'uploads': 8, 'aggre_num': 4, 'evaluators': 10,
                'budgets': [20, 100], 'scenarios': list(SCENARIOS)},
    'full100': {'clients': 100, 'uploads': 100, 'aggre_num': 50, 'evaluators': 20,
                'budgets': [100], 'scenarios': ['clean', 'delta_sign_flip', 'joint_all_max']},
}


def build_plan(project, output, stage, jobs):
    selected_profiles = list(PROFILES) if stage == 'all' else [stage]
    experiments = []
    for profile_name in selected_profiles:
        profile = PROFILES[profile_name]
        for rounds in profile['budgets']:
            for name in profile['scenarios']:
                scenario = SCENARIOS[name]
                identifier = f'{profile_name}_{rounds}r_{name}'
                command = [sys.executable, str(project / 'run_comparison.py'),
                           '--output-root', str(output / identifier), '--algorithms', *ALGORITHMS,
                           '--seed', '1', '--device', 'cpu', '--dataset', 'mnist',
                           '--num_clients', str(profile['clients']), '--num_sel_clients', str(profile['uploads']),
                           '--aggre_num', str(profile['aggre_num']), '--global_rounds', str(rounds),
                           '--local_epochs', '1', '--batch_size', '16', '--local_learning_rate', '0.1',
                           '--eval_gap', '5', '--eval-clients', str(profile['evaluators']),
                           '--macro-length', '2', '--tc-batch-size', '2', '--warmup-macros', '1',
                           '--tc-updates', '4', '--buffer_size', '512', '--td-mode', 'paper',
                           '--ad_ratio', '0.2', '--rho-max', '0.2', '--loss-cap', '5',
                           '--split-seed', '2026', '--validation-fraction', '0.1',
                           '--training-cohort', 'all', '--threads', '1',
                           '--attack', scenario['attack'], '--report-attack', scenario['report_attack']]
                if scenario['fake_utility']:
                    command.append('--fake-utility')
                experiments.append({'id': identifier, 'profile': profile_name, 'rounds': rounds,
                                    'scenario': name, 'command': command})
    source_paths = [project / 'main.py', project / 'run_comparison.py', Path(__file__).resolve()]
    for directory in ('algorithms', 'clients', 'servers', 'utils', 'DDPG'):
        source_paths.extend((project / directory).rglob('*.py'))
    return {'created_at': datetime.now().isoformat(), 'seed': 1, 'jobs': jobs,
            'algorithms': ALGORITHMS, 'profiles': {key: PROFILES[key] for key in selected_profiles},
            'scenarios': SCENARIOS, 'comparisons': len(experiments),
            'planned_model_runs': len(experiments)*len(ALGORITHMS),
            'experiments': experiments,
            'primary_metric': 'Native FedAA honest-client sample-weighted test accuracy at the final round.',
            'secondary_metric': 'Mean of evaluation checkpoints in the last ten rounds; descriptive within-run averaging, not independent replicates.',
            'notes': [
                'One predeclared seed; no inferential confidence intervals or statistical significance claims.',
                'Clean disables all attacks but retains the same designated adversary IDs and evaluation population as attacked conditions.',
                'Each 20/100-round budget is trained independently; total budget changes the controller progress coordinate and terminal target.',
                'FedTCFixed uses (lambda,tau)=(0,1.35); no actor/critic training. It still runs feedback diagnostics.',
                'FedAA keeps its clean server-root reward. Feedback/utility poisoning only affects methods with those interfaces.',
                'Native drop_last=True and its zero-SGD small clients are retained and recorded.',
                'Exploratory controller replay batch=2 and H=2, consistent across this study. This is not the manuscript-scale default.',
                'No method hyperparameter tuning after observing these results; TD remains paper, gate unchanged.',
                'Concurrent CPU execution: runtime is not a controlled overhead benchmark.',
            ],
            'source_sha256': {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source_paths))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=f'runs/single_seed_study_{datetime.now():%Y%m%d_%H%M%S}')
    parser.add_argument('--stage', choices=['all', 'small12', 'full100'], default='all')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('jobs must be positive.')
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parent
    plan = build_plan(project, output, args.stage, args.jobs)
    write_json(output / 'plan.json', plan)
    write_json(output / 'status.json', {'status': 'running', 'completed_comparisons': [], 'failures': []})
    print(f'Plan saved: {plan["planned_model_runs"]} model runs, seed=1, {plan["comparisons"]} comparisons.', flush=True)

    def run(experiment):
        print(f'START {experiment["id"]}', flush=True)
        with (output / f'{experiment["id"]}.log').open('w') as log:
            result = subprocess.run(experiment['command'], stdout=log, stderr=subprocess.STDOUT)
        return {'id': experiment['id'], 'exit_code': result.returncode}

    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, entry) for entry in plan['experiments']]
        for future in as_completed(futures):
            result = future.result()
            (completed if result['exit_code'] == 0 else failures).append(result)
            write_json(output / 'status.json', {'status': 'running', 'completed_comparisons': completed, 'failures': failures})
            print(f'{"DONE" if result["exit_code"] == 0 else "FAILED"} {result["id"]}; {len(completed)+len(failures)}/{plan["comparisons"]}', flush=True)
    write_json(output / 'status.json', {'status': 'complete' if not failures else 'complete_with_failures',
                                       'completed_comparisons': completed, 'failures': failures})
    print(f'Finished. {len(completed)} completed comparisons; {len(failures)} failures. Results: {output}', flush=True)


if __name__ == '__main__':
    main()
