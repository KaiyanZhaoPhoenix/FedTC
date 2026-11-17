#!/usr/bin/env python
"""Audit and summarize all predeclared single-seed runs; export raw learning curves."""
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
from PIL import Image

from run_comparison import verify
from utils.experiment import write_json

COLORS = {'FedAA': '#0072B2', 'FedTCFixed': '#444444', 'FedTC': '#D55E00'}
STYLES = {'FedAA': ('--', 's'), 'FedTCFixed': (':', '^'), 'FedTC': ('-', 'o')}
DISPLAY = {'FedAA': 'FedAA', 'FedTCFixed': 'FedTC fixed', 'FedTC': 'FedTC dynamic'}
TITLES = {
    'clean': 'Clean (matched roster)', 'same_value': 'Same-value poisoning',
    'gaussian': 'Gaussian poisoning', 'delta_sign_flip': 'Delta sign flip (-10x)',
    'feedback_only': 'All-Max feedback only', 'joint_all_max': 'Same + utility + All-Max',
    'joint_echo': 'Same + utility + Echo', 'joint_bimodal': 'Same + utility + Bimodal',
}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_and_collect(root, plan):
    status = json.loads((root / 'status.json').read_text())
    if status['status'] != 'complete' or status['failures']:
        raise RuntimeError('Study has not completed all declared runs. No aggregate report generated.')
    project = Path(__file__).resolve().parent
    for relative, expected in plan['source_sha256'].items():
        if hashlib.sha256((project / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed during the study: {relative}')
    records, curves, macros_all, references = [], [], [], {}
    expected_checks = set()
    for experiment in plan['experiments']:
        directory = root / experiment['id']
        verify(directory, plan['algorithms'])
        expected_checks.add(experiment['id'])
        for algorithm in plan['algorithms']:
            folder = directory / algorithm
            summary = json.loads((folder / 'summary.json').read_text())
            manifest = json.loads((folder / 'manifest.json').read_text())
            args = manifest['args']
            setting = plan['scenarios'][experiment['scenario']]
            assert args['seed'] == plan['seed'] == 1
            assert args['global_rounds'] == experiment['rounds'] == summary['rounds']
            assert args['attack'] == setting['attack']
            assert args['report_attack'] == setting['report_attack']
            assert args['fake_utility'] == setting['fake_utility']
            rounds = jsonl(folder / 'rounds.jsonl')
            assert len(rounds) == experiment['rounds']
            assert [r['round'] for r in rounds] == list(range(1, experiment['rounds']+1))
            evaluations = jsonl(folder / 'evaluations.jsonl')
            assert [e['round'] for e in evaluations] == list(range(5, experiment['rounds']+1, 5))
            assert all(e['phase'] == 'global' for e in evaluations)
            assert evaluations[-1]['accuracy'] == summary['final_accuracy']
            with h5py.File(folder / 'metrics.h5') as handle:
                np.testing.assert_array_equal(handle['round'][:], [e['round'] for e in evaluations])
                np.testing.assert_array_equal(handle['rs_test_acc'][:], [e['accuracy'] for e in evaluations])
            common_key = (experiment['profile'], experiment['rounds'])
            signature = {key: manifest[key] for key in ('initial_model_sha256', 'native_data_sha256',
                                                       'private_splits_sha256', 'malicious_clients_harness_only')}
            signature['cohorts'] = [(r['upload_clients'], r['trained_clients']) for r in rounds]
            if common_key in references:
                assert signature == references[common_key], 'Cross-scenario protocol mismatch.'
            else:
                references[common_key] = signature
            macros = jsonl(folder / 'macros.jsonl') if (folder / 'macros.jsonl').exists() else []
            if algorithm.startswith('FedTC'):
                assert len(macros) == experiment['rounds'] // args['macro_length']
            gates = [m for m in macros if m['gate_reason'] in ('accepted', 'value_gap')]
            accepted = [m for m in gates if m['gate_reason'] == 'accepted']
            tail = [100*e['accuracy'] for e in evaluations if e['round'] > experiment['rounds']-10]
            records.append({
                'experiment': experiment['id'], 'profile': experiment['profile'],
                'scenario': experiment['scenario'], 'budget_rounds': experiment['rounds'],
                'seed': 1, 'algorithm': algorithm, 'final_accuracy_pct': 100*summary['final_accuracy'],
                'last10_rounds_mean_accuracy_pct': float(np.mean(tail)),
                'last10_evaluation_points': len(tail), 'final_auc': summary['final_auc'],
                'final_client_accuracy_std': evaluations[-1]['client_std'],
                'controller_train_steps': summary.get('controller_train_steps', 0),
                'warmup_macros': sum(m['gate_reason'] == 'warmup' for m in macros),
                'postwarmup_decisions': len(gates), 'accepted_macros': len(accepted),
                'postwarmup_fallbacks': len(gates)-len(accepted),
                'postwarmup_fallback_pct': 100*(len(gates)-len(accepted))/len(gates) if gates else None,
                'unique_executed_actions': len({tuple(m['action']) for m in macros}),
                'accepted_macro_indices_zero_based': ','.join(str(m['macro']) for m in accepted),
                'mean_interval_width': float(np.mean([m['high']-m['low'] for m in macros])) if macros else None,
                'max_state_iqr_ratio': max(m['raw_next_state'][1] for m in macros) if macros else None,
                'zero_sgd_client_count': len(manifest['zero_sgd_batch_clients']),
                'results_directory': str(folder),
            })
            for evaluation in evaluations:
                curves.append({'experiment': experiment['id'], 'profile': experiment['profile'],
                               'scenario': experiment['scenario'], 'budget_rounds': experiment['rounds'],
                               'algorithm': algorithm, 'round': evaluation['round'],
                               'accuracy_pct': 100*evaluation['accuracy']})
            for macro in macros:
                macros_all.append({'experiment': experiment['id'], 'algorithm': algorithm,
                                   'macro': macro['macro'], 'round_end': macro['round_end'],
                                   'lambda': macro['action'][0], 'tau': macro['action'][1],
                                   'gate_reason': macro['gate_reason'], 'low': macro['low'], 'high': macro['high'],
                                   'interval_width': macro['high']-macro['low']})
    assert expected_checks == {item['id'] for item in status['completed_comparisons']}
    assert len(records) == plan['planned_model_runs']
    for profile, cfg in plan['profiles'].items():
        budgets = sorted(cfg['budgets'])
        for shorter, longer in zip(budgets, budgets[1:]):
            first, second = references[profile, shorter], references[profile, longer]
            assert {k: v for k, v in first.items() if k != 'cohorts'} == {k: v for k, v in second.items() if k != 'cohorts'}
            assert first['cohorts'] == second['cohorts'][:shorter], 'Initial cohort schedule changed with budget.'
    for record in records:
        assert 0 <= record['final_accuracy_pct'] <= 100
    return records, curves, macros_all


def save_figure(fig, stem, metadata, figures):
    for suffix in ('png', 'pdf'):
        target = stem.with_suffix('.'+suffix)
        if target.exists():
            raise FileExistsError(target)
        fig.savefig(target, dpi=250, facecolor='white')
    with Image.open(stem.with_suffix('.png')) as img:
        actual = {'pixels': list(img.size), 'dpi': list(img.info.get('dpi', [])), 'mode': img.mode}
    figures.append({'stem': str(stem), **metadata, **actual})
    plt.close(fig)


def make_figures(root, plan, curves, macros, paired):
    figures = []
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'}):
        for profile, cfg in plan['profiles'].items():
            for budget in cfg['budgets']:
                count = len(cfg['scenarios'])
                rows, cols = (2, 4) if count == 8 else (1, 3)
                fig, axes = plt.subplots(rows, cols, figsize=(13.6, 6.8 if rows == 2 else 3.8),
                                         layout='constrained', sharex=True, sharey=True)
                for ax, scenario in zip(np.asarray(axes).flat, cfg['scenarios']):
                    for algorithm in plan['algorithms']:
                        points = [r for r in curves if r['profile'] == profile and r['budget_rounds'] == budget
                                  and r['scenario'] == scenario and r['algorithm'] == algorithm]
                        style, marker = STYLES[algorithm]
                        ax.plot([p['round'] for p in points], [p['accuracy_pct'] for p in points],
                                color=COLORS[algorithm], linestyle=style, marker=marker,
                                markersize=3.2, markevery=max(1, len(points)//5), linewidth=1.7,
                                label=DISPLAY[algorithm], zorder=3 if algorithm == 'FedTCFixed' else 2)
                    ax.set(title=TITLES[scenario], xlim=(0, budget), ylim=(0, 100))
                    ax.grid(alpha=.18)
                fig.supxlabel('Communication round')
                fig.supylabel('Test accuracy (%)')
                np.asarray(axes).flat[0].legend(fontsize=8, loc='lower right')
                fig.suptitle(f'MNIST | {cfg["clients"]} clients | {budget}-round budget | seed=1 (one run per method)', fontsize=13)
                save_figure(fig, root / f'{profile}_{budget}rounds_accuracy', {
                    'alt_text': f'Unsmoothed accuracy trajectories for all {count} declared scenarios, {cfg["clients"]} clients, {budget} rounds, one seed. All axes use 0 to 100 percent. Overlapping curves are retained.',
                    'uncertainty': 'None: one seed; repeated checkpoints are not independent replicates.',
                    'source': 'learning_curves.csv', 'transform': 'accuracy fraction multiplied by 100; straight lines connect observed checkpoints every five rounds; no smoothing.'}, figures)
        # Expose differences that can be hard to see in overlapping accuracy curves.
        small = [p for p in paired if p['profile'] == 'small12']
        if small:
            fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), layout='constrained', sharey=True)
            for ax, key, title, color in zip(axes,
                    ['dynamic_minus_fedaa_pp', 'dynamic_minus_fixed_pp'],
                    ['Dynamic FedTC - FedAA', 'Dynamic FedTC - fixed FedTC'],
                    [COLORS['FedAA'], COLORS['FedTC']]):
                for budget, marker, offset in zip(plan['profiles']['small12']['budgets'], ['s', 'o'], [-.12, .12]):
                    entries = [next(p for p in small if p['scenario'] == name and p['budget_rounds'] == budget)
                               for name in plan['profiles']['small12']['scenarios']]
                    y = np.arange(len(entries))
                    ax.scatter([p[key] for p in entries], y+offset, edgecolors=color,
                               facecolors='none' if budget == 20 else color,
                               marker=marker, s=35, label=f'{budget} rounds', zorder=3)
                ax.axvline(0, color='#333333', linewidth=1)
                lower = min(0., min(p[key] for p in small))
                upper = max(0., max(p[key] for p in small))
                padding = max(.3, .08*(upper-lower))
                ax.set(title=title, xlim=(lower-padding, upper+padding),
                       yticks=y, yticklabels=[TITLES[p['scenario']] for p in entries])
                ax.grid(axis='x', alpha=.18)
                ax.legend(fontsize=8, loc='best')
            axes[0].invert_yaxis()
            fig.supxlabel('Final accuracy difference (percentage points); positive favors dynamic FedTC')
            fig.suptitle('MNIST, 12 clients, seed=1 | Different x-axis scales | No uncertainty interval', fontsize=12)
            save_figure(fig, root / 'small12_paired_differences', {
                'alt_text': 'Left: dynamic FedTC minus FedAA. Right: dynamic FedTC minus fixed FedTC. Panels explicitly use different x-axis scales. Hollow squares denote 20 rounds and filled circles 100 independently trained rounds for all eight scenarios. Zero denotes equal final accuracy.',
                'uncertainty': 'None: a single paired seed, not statistical significance.',
                'source': 'paired_differences.csv', 'transform': 'Subtract final accuracy percentages within each scenario and training budget.'}, figures)
    write_json(root / 'figure_manifest.json', {'purpose': 'Exploratory experiment report, not submission-ready evidence',
                                               'matplotlib_version': matplotlib.__version__, 'figures': figures})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    plan = json.loads((root / 'plan.json').read_text())
    records, curves, macros = audit_and_collect(root, plan)
    write_csv(root / 'results.csv', records)
    write_csv(root / 'learning_curves.csv', curves)
    write_csv(root / 'controller_history.csv', macros)
    paired = []
    for experiment in plan['experiments']:
        values = {r['algorithm']: r for r in records if r['experiment'] == experiment['id']}
        model_states = {algorithm: torch.load(root / experiment['id'] / algorithm / 'model.pt',
                                              map_location='cpu', weights_only=False)['model']
                        for algorithm in ('FedTC', 'FedTCFixed')}
        same_parameters = all(torch.equal(value, model_states['FedTCFixed'][key])
                              for key, value in model_states['FedTC'].items())
        if values['FedTC']['accepted_macros'] == 0:
            assert same_parameters, 'All-conservative dynamic run unexpectedly differs from fixed.'
        paired.append({'experiment': experiment['id'], 'profile': experiment['profile'],
                       'scenario': experiment['scenario'], 'budget_rounds': experiment['rounds'],
                       **{algo+'_accuracy_pct': values[algo]['final_accuracy_pct'] for algo in plan['algorithms']},
                       'dynamic_minus_fedaa_pp': values['FedTC']['final_accuracy_pct']-values['FedAA']['final_accuracy_pct'],
                       'dynamic_minus_fixed_pp': values['FedTC']['final_accuracy_pct']-values['FedTCFixed']['final_accuracy_pct'],
                       'fixed_minus_fedaa_pp': values['FedTCFixed']['final_accuracy_pct']-values['FedAA']['final_accuracy_pct'],
                       'dynamic_last10_minus_fixed_pp': values['FedTC']['last10_rounds_mean_accuracy_pct']-values['FedTCFixed']['last10_rounds_mean_accuracy_pct'],
                       'dynamic_fixed_final_parameters_identical': same_parameters})
    write_csv(root / 'paired_differences.csv', paired)
    write_json(root / 'summary.json', {'status': 'complete', 'seed': 1, 'verified_model_runs': len(records),
                                     'verified_comparisons': len(paired), 'results': records,
                                     'paired_differences': paired, 'limitations': plan['notes']})
    lines = ['FedTC / 固定参数 FedTC / FedAA 单种子攻击与轮数对照',
             f'已核验：{len(records)} 次模型运行，{len(paired)} 组对照，全部 seed=1。',
             '指标为原 FedAA 诚实客户端测试样本加权准确率（%）；差值单位为百分点。',
             '下面是单次运行的点估计，没有跨种子误差条或显著性结论。', '']
    for profile, cfg in plan['profiles'].items():
        for budget in cfg['budgets']:
            lines += [f'{cfg["clients"]} 客户端，独立训练 {budget} 轮',
                      '场景\tFedAA\t固定 FedTC\t动态 FedTC\t动态－FedAA\t动态－固定']
            for row in paired:
                if row['profile'] == profile and row['budget_rounds'] == budget:
                    lines.append(f'{plan["scenarios"][row["scenario"]]["label"]}\t{row["FedAA_accuracy_pct"]:.2f}\t{row["FedTCFixed_accuracy_pct"]:.2f}\t{row["FedTC_accuracy_pct"]:.2f}\t{row["dynamic_minus_fedaa_pp"]:+.2f}\t{row["dynamic_minus_fixed_pp"]:+.2f}')
            lines.append('')
    lines += ['动态控制器记录（不含预热的门控拒绝比例）', '场景/预算\t候选通过数\t实际判断次数\t拒绝率\t执行动作种类']
    for row in records:
        if row['algorithm'] == 'FedTC':
            lines.append(f'{row["experiment"]}\t{row["accepted_macros"]}\t{row["postwarmup_decisions"]}\t{row["postwarmup_fallback_pct"]:.1f}%\t{row["unique_executed_actions"]}')
    lines += ['', '解释边界：',
              '1. 所有场景固定同一攻击者名单；clean 关闭其攻击，但测试人群不改变。',
              '2. FedAA 使用干净服务器奖励数据，反馈造假不直接作用于 FedAA。',
              '3. 固定 FedTC 为 (lambda,tau)=(0,1.35)，不学习策略。',
              '4. 使用正文 TD 与现有区间/门控；没有根据结果调参或放宽门控。',
              '5. 20/100 轮是独立运行；总轮数会改变控制器的进度坐标，因此不假定前 20 轮相同。',
              '6. 12 客户端仅取原生客户端 0—11；100 客户端使用全量原生划分。两种规模不能当作相同总体。',
              '7. 完整 100 客户端下有 4 个客户端因原生 drop_last=True 而没有 SGD 批次；所有方法一致。',
              '8. 上一轮误启动的多种子数据单独保存，未混入这里的结果。',
              '9. 末 10 轮均值仅包含该窗口内两个评测点；它不是两个独立随机种子。',
              '10. 本次没有测试 ALIE、Min-Max、CIFAR 或少数派保护指标。']
    (root / 'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    make_figures(root, plan, curves, macros, paired)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
