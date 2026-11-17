"""Verify the complete four-way SAC study and export auditable figures/results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_comparison import verify
from run_sac_study import ALGORITHMS, REPORTS, ROOT
from utils.experiment import write_json

LABELS = {'FedTCFixed': 'Tuned fixed', 'FedTC': 'Original interval',
          'FedTCSAC': 'SAC, no gate', 'FedTCSACGate': 'SAC + gate'}
COLORS = {'FedTCFixed': '#444444', 'FedTC': '#0072B2', 'FedTCSAC': '#D55E00', 'FedTCSACGate': '#009E73'}
STYLES = {'FedTCFixed': (':', 's'), 'FedTC': ('--', '^'), 'FedTCSAC': ('-', 'o'), 'FedTCSACGate': ('-.', 'D')}
TITLES = {'none': 'Truthful reports', 'all_max': 'All-Max reports', 'echo': 'Echo reports', 'bimodal': 'Bimodal reports'}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def csv_file(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    plan = json.loads((root/'plan.json').read_text())
    status = json.loads((root/'status.json').read_text())
    assert status['status'] == 'complete' and not status['failures'] and len(status['completed']) == 25
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    selection = json.loads((root/'fixed_selection.json').read_text())
    assert selection['used_test_metrics'] is False
    assert selection['selected'] == min(selection['scores'], key=lambda x: x['validation_score'])
    records, curves, macro_rows, shadow_rows, raw = [], [], [], [], {}
    reference = None
    for report in REPORTS:
        verify(root/'main'/report, ALGORITHMS)
        for algorithm in ALGORITHMS:
            folder = root/'main'/report/algorithm
            manifest = json.loads((folder/'manifest.json').read_text())
            summary = json.loads((folder/'summary.json').read_text())
            rounds, evaluations, macros = [jsonl(folder/name) for name in ('rounds.jsonl', 'evaluations.jsonl', 'macros.jsonl')]
            signature = {k: manifest[k] for k in ('initial_model_sha256', 'native_data_sha256', 'private_splits_sha256', 'malicious_clients_harness_only')}
            signature['schedules'] = [(r['upload_clients'], r['trained_clients']) for r in rounds]
            signature['evaluators'] = [m['evaluators'] for m in macros]
            if reference is None:
                reference = signature
            else:
                assert reference == signature, (report, algorithm, 'provenance')
            a = manifest['args']
            assert a['seed'] == 1 and a['global_rounds'] == 300
            assert a['attack'] == 'same_value' and a['fake_utility'] and a['report_attack'] == report
            assert [r['round'] for r in rounds] == list(range(1, 301))
            assert [r['round'] for r in evaluations] == list(range(10, 301, 10))
            assert len(macros) == 150 and summary['final_accuracy'] == evaluations[-1]['accuracy']
            with h5py.File(folder/'metrics.h5') as f:
                np.testing.assert_array_equal(f['rs_test_acc'][:], [e['accuracy'] for e in evaluations])
            saved = torch.load(folder/'controller.pt', map_location='cpu', weights_only=False)
            if algorithm != 'FedTCFixed':
                assert saved['train_steps'] == 540 and len(saved['replay']) == 150
                assert [entry[1] for entry in saved['replay']] == [m['action'] for m in macros]
                assert all(m['action'] == [0., 1.35] for m in macros[:16])
                assert sum(m['gate_reason'] == 'warmup' for m in macros) == 16
            else:
                assert saved['train_steps'] == 0
                assert all(m['action'] == [selection['selected']['lambda'], selection['selected']['tau']] for m in macros)
            clients = jsonl(folder/'client_accuracy_audit.jsonl')[-1]
            validations = [r for r in jsonl(folder/'validation_audit.jsonl') if r['round'] > 240]
            tail = [100*e['accuracy'] for e in evaluations if e['round'] > 240]
            post = [m for m in macros if m['gate_reason'] != 'warmup'] if algorithm != 'FedTCFixed' else []
            actions = np.array([m['action'] for m in macros])
            relevant_shadow = []
            bad_ids = set(manifest['malicious_clients_harness_only'])
            bad_weights, honest_zero_rates = [], []
            for r in rounds:
                bad_weights.append(sum(w for cid, w in zip(r['upload_clients'], r['weights']) if cid in bad_ids))
                honest_weights = [w for cid, w in zip(r['upload_clients'], r['weights']) if cid not in bad_ids]
                honest_zero_rates.append(sum(w == 0 for w in honest_weights)/len(honest_weights))
                sh = r.get('shadow_harness_only')
                if sh:
                    row = {'report': report, 'algorithm': algorithm, 'round': r['round'], **sh}
                    shadow_rows.append(row)
                    if sh['gate_reason'] != 'warmup':
                        relevant_shadow.append(sh)
            record = {
                'report': report, 'algorithm': algorithm, 'tail20_accuracy_pct': float(np.mean(tail)),
                'final_accuracy_pct': 100*summary['final_accuracy'],
                'macro_client_accuracy_pct': 100*clients['macro_accuracy'],
                'bottom20_client_accuracy_pct': 100*clients['bottom20_client_mean'],
                'tail20_honest_validation_loss': float(np.mean([r['honest_client_macro_loss'] for r in validations])),
                'tail20_all_validation_loss': float(np.mean([r['all_client_macro_loss'] for r in validations])),
                'postwarmup_decisions': len(post), 'postwarmup_fallbacks': sum(m['fallback'] for m in post),
                'candidate_executions': sum(not m['fallback'] for m in post),
                'unique_executed_actions': len(np.unique(actions, axis=0)),
                'lambda_std': float(actions[:, 0].std()), 'tau_std': float(actions[:, 1].std()),
                'controller_train_steps': summary['controller_train_steps'],
                'mean_malicious_weight': float(np.mean(bad_weights)),
                'rounds_with_positive_malicious_weight': sum(w > 0 for w in bad_weights),
                'mean_honest_zero_weight_pct': 100*float(np.mean(honest_zero_rates)),
                'mean_interval_width': float(np.mean([m['high']-m['low'] for m in macros])),
                'shadow_count_postwarmup': len(relevant_shadow),
                'shadow_mean_one_step_improvement': float(np.mean([s['candidate_minus_conservative_improvement'] for s in relevant_shadow])) if relevant_shadow else None,
                'shadow_positive_count': sum(s['candidate_minus_conservative_improvement'] > 1e-6 for s in relevant_shadow),
                'elapsed_seconds': summary['elapsed_seconds'],
            }
            records.append(record)
            for e in evaluations:
                curves.append({'report': report, 'algorithm': algorithm, 'round': e['round'], 'accuracy_pct': 100*e['accuracy']})
            for m in macros:
                macro_rows.append({'report': report, 'algorithm': algorithm, 'macro': m['macro'], 'round_end': m['round_end'],
                                   'lambda': m['action'][0], 'tau': m['action'][1], 'gate_reason': m['gate_reason'],
                                   'interval_low': m['low'], 'interval_high': m['high'],
                                   'midpoint': (m['low']+m['high'])/2,
                                   'honest_sample_mean_harness_only': m['honest_sample_mean_harness_only'],
                                   'sac_alpha': m.get('sac_alpha'), 'gate_margin': m.get('gate_margin')})
            raw[report, algorithm] = {'evaluations': evaluations, 'macros': macros, 'rounds': rounds}

    # Pure gate toggle: identical candidate draws and replay-driven updates until
    # the first difference in executed action (not indefinitely thereafter).
    prefix_checks = {}
    for report in REPORTS:
        ungated = raw[report, 'FedTCSAC']['macros']
        gated = raw[report, 'FedTCSACGate']['macros']
        prefix = 0
        for one, two in zip(ungated, gated):
            assert one['candidate'] == two['candidate']
            if one['action'] != two['action']:
                break
            assert one['reports'] == two['reports']
            prefix += 1
        prefix_checks[report] = prefix
    # A fixed method cannot respond to report attacks: verify exact final tensors.
    fixed = torch.load(root/'main'/'none'/'FedTCFixed'/'model.pt', map_location='cpu', weights_only=False)['model']
    for report in REPORTS[1:]:
        other = torch.load(root/'main'/report/'FedTCFixed'/'model.pt', map_location='cpu', weights_only=False)['model']
        assert all(torch.equal(fixed[k], other[k]) for k in fixed)
    original_gate_equal = {}
    feedback_audit = {}
    honest_warmup = raw['none', 'FedTC']['macros'][:16]
    for report in REPORTS:
        old = torch.load(root/'main'/report/'FedTC'/'model.pt', map_location='cpu', weights_only=False)['model']
        gate = torch.load(root/'main'/report/'FedTCSACGate'/'model.pt', map_location='cpu', weights_only=False)['model']
        original_gate_equal[report] = all(torch.equal(old[k], gate[k]) for k in old)
        warmup = raw[report, 'FedTC']['macros'][:16]
        assert [m['honest_sample_mean_harness_only'] for m in warmup] == [m['honest_sample_mean_harness_only'] for m in honest_warmup]
        feedback_audit[report] = {
            'same_executed_actions_and_true_sample_feedback': True, 'macros': 16,
            'mean_midpoint_minus_honest_sample_mean': float(np.mean([
                (m['low']+m['high'])/2-m['honest_sample_mean_harness_only'] for m in warmup])),
            'scope': 'Warmup paired diagnostic; sample honest mean, not population coverage or independent replicates.'}
    csv_file(root/'results.csv', records)
    csv_file(root/'learning_curves.csv', curves)
    csv_file(root/'controller_history.csv', macro_rows)
    write_json(root/'shadow_checks.json', shadow_rows)
    write_json(root/'results.json', {'selection': selection, 'results': records,
                                    'sac_identical_execution_prefix_macros': prefix_checks,
                                    'fixed_final_models_identical_across_reports': True,
                                    'original_vs_sac_gate_identical_final_models': original_gate_equal,
                                    'warmup_feedback_audit': feedback_audit})

    figure_notes = []
    with plt.rc_context({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained', sharex=True, sharey=True)
        for ax, report in zip(axes.flat, REPORTS):
            for algorithm in ALGORITHMS:
                rows = raw[report, algorithm]['evaluations']
                ls, marker = STYLES[algorithm]
                ax.plot([r['round'] for r in rows], [100*r['accuracy'] for r in rows],
                        label=LABELS[algorithm], color=COLORS[algorithm], linestyle=ls,
                        marker=marker, markevery=4, markersize=3, linewidth=1.5)
            ax.set_title(TITLES[report])
            ax.set_ylim(0, 100)
            ax.grid(alpha=.2)
        for ax in axes[-1]:
            ax.set_xlabel('Federated round')
        for ax in axes[:, 0]:
            ax.set_ylabel('Honest-client test accuracy (%)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
        fig.suptitle('Seed 1 | MNIST, 100 clients | Same-value poisoning + fake utility in every panel\nOnly validation-report attacks change; raw curves, no confidence bands', fontsize=11)
        for extension in ('png', 'pdf'):
            fig.savefig(root/f'accuracy_curves.{extension}', dpi=220, facecolor='white')
        plt.close(fig)
        figure_notes.append({'file': 'accuracy_curves', 'encoding': 'Raw accuracy every 10 rounds, common 0-100% axes; one seed, no error bars.'})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
        for j, algorithm in enumerate(ALGORITHMS):
            values = [next(r for r in records if r['report'] == report and r['algorithm'] == algorithm) for report in REPORTS]
            offset = (j-1.5)*.15
            y = np.arange(4)+offset
            axes[0].scatter([r['tail20_accuracy_pct'] for r in values], y, color=COLORS[algorithm], marker=STYLES[algorithm][1], label=LABELS[algorithm], s=36)
            for x, yi in zip([r['tail20_accuracy_pct'] for r in values], y):
                axes[0].annotate(f'{x:.2f}', (x, yi), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8)
            if algorithm != 'FedTCFixed':
                axes[1].scatter([r['candidate_executions'] for r in values], y, color=COLORS[algorithm], marker=STYLES[algorithm][1], s=36)
        all_values = [r['tail20_accuracy_pct'] for r in records]
        axes[0].set_xlim(min(all_values)-.5, max(all_values)+1.3)
        axes[0].set_xlabel('Last 20% rounds: mean accuracy (%)\nZoomed point axis; descriptive mean of six checkpoints')
        axes[1].set_xlim(-4, 142)
        axes[1].set_xlabel('Candidate executions after warmup (out of 134)\nUngated SAC executes proposals without a gate test')
        for ax in axes:
            ax.set_yticks(range(4), [TITLES[x] for x in REPORTS])
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
        fig.suptitle('Performance and executed adaptation | Seed 1, no statistical uncertainty estimate', fontsize=11)
        for extension in ('png', 'pdf'):
            fig.savefig(root/f'performance_and_execution.{extension}', dpi=220, facecolor='white')
        plt.close(fig)
        figure_notes.append({'file': 'performance_and_execution', 'encoding': 'Points; performance axis zoomed and labeled. No seed-level uncertainty. Candidate executions are not gate accuracy.'})

        fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout='constrained', sharex=True, sharey='row')
        for j, report in enumerate(REPORTS):
            for algorithm in ALGORITHMS:
                rows = raw[report, algorithm]['macros']
                ls, marker = STYLES[algorithm]
                for i in range(2):
                    axes[i, j].plot([m['round_end'] for m in rows], [m['action'][i] for m in rows],
                                    color=COLORS[algorithm], linestyle=ls, label=LABELS[algorithm], linewidth=1., alpha=.9)
            axes[0, j].set_title(TITLES[report])
            axes[1, j].set_xlabel('Federated round')
        axes[0, 0].set_ylabel('Executed lambda')
        axes[1, 0].set_ylabel('Executed tau')
        axes[0, 0].set_ylim(-.02, .52)
        axes[1, 0].set_ylim(1.02, 1.83)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
        fig.suptitle('Executed parameters, including conservative warmup | Seed 1', fontsize=11)
        for extension in ('png', 'pdf'):
            fig.savefig(root/f'executed_actions.{extension}', dpi=220, facecolor='white')
        plt.close(fig)
        figure_notes.append({'file': 'executed_actions', 'encoding': 'Actual executed actions at every macro; same allowed ranges across panels; no smoothing.'})

    write_json(root/'figure_manifest.json', {'source': ['results.csv', 'learning_curves.csv', 'controller_history.csv'],
               'figures': figure_notes, 'seed': 1, 'palette': COLORS, 'uncertainty': 'None: a single paired seed.',
               'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'alt_text': 'Four methods under truthful and three forged-feedback conditions. All panels retain the same model poisoning and fake utility.'})
    (root/'VERIFIED.txt').write_text('25 runs complete; original source hashes unchanged during study; shared data, initialization, upload/training/evaluation cohorts verified; 16 metric files agree with logs; actual executed replay actions verified; SAC pair agrees until execution diverges; fixed final tensors identical across report attacks. One seed only.\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
