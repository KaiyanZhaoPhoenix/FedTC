"""Read-only verification of published measurements; no data download or model checkpoints required."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def read_text(path):
    path = Path(path)
    if not path.exists():
        path = path.with_name(path.name + '.gz')
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return data.decode('utf-8')


def read_json(path):
    return json.loads(read_text(path))


def read_jsonl(path):
    return [json.loads(line) for line in read_text(path).splitlines()]


def close(one, two, label):
    if not math.isclose(float(one), float(two), rel_tol=0, abs_tol=1e-10):
        raise ValueError(f'Measurement mismatch: {label}: {one} != {two}')


def verify_archive(root):
    root = Path(root).resolve()
    archive = read_json(root/'reproducibility/archive_manifest.json')
    for item in archive['entries']:
        path = root / item['published_path']
        path.resolve().relative_to(root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['published_sha256']:
            raise ValueError(f'Published file hash mismatch: {path}')
    with (root/'results.csv').open(newline='') as stream:
        results = list(csv.DictReader(stream))
    with (root/'learning_curves.csv').open(newline='') as stream:
        curves = list(csv.DictReader(stream))
    sac = 'report' in results[0]
    expected = 16 if sac else 57
    if len(results) != expected:
        raise ValueError(f'Expected {expected} results, got {len(results)}')
    signatures, seen = {}, set()
    raw = root/'reproducibility/raw'
    for row in results:
        algorithm = row['algorithm']
        name = row['report'] if sac else row['experiment']
        key = (name, algorithm)
        if key in seen:
            raise ValueError(f'Duplicate result: {key}')
        seen.add(key)
        directory = raw/'main'/name/algorithm if sac else raw/name/algorithm
        manifest, summary = (read_json(directory/file) for file in ('manifest.json', 'summary.json'))
        evaluations, rounds = (read_jsonl(directory/file) for file in ('evaluations.jsonl', 'rounds.jsonl'))
        if summary['status'] != 'complete' or len(rounds) != summary['rounds']:
            raise ValueError(f'Incomplete run: {key}')
        close(row['final_accuracy_pct'], 100*evaluations[-1]['accuracy'], key)
        close(summary['final_accuracy'], evaluations[-1]['accuracy'], key)
        group_curves = [r for r in curves if r['algorithm']==algorithm and r['report' if sac else 'experiment']==name]
        if [int(r['round']) for r in group_curves] != [r['round'] for r in evaluations]:
            raise ValueError(f'Curve checkpoints differ: {key}')
        for one, two in zip(group_curves, evaluations):
            close(one['accuracy_pct'], 100*two['accuracy'], key)
        signature = {field:manifest[field] for field in ('initial_model_sha256','native_data_sha256','private_splits_sha256','malicious_clients_harness_only')}
        signature['schedules'] = [(r['upload_clients'],r['trained_clients']) for r in rounds]
        comparison = 'all_sac_main' if sac else name
        if comparison in signatures and signature != signatures[comparison]:
            raise ValueError(f'Unmatched comparison provenance: {key}')
        signatures[comparison] = signature
        if sac:
            close(row['tail20_accuracy_pct'],statistics.mean(100*r['accuracy'] for r in evaluations if r['round']>240),key)
        else:
            close(row['last10_rounds_mean_accuracy_pct'],statistics.mean(100*r['accuracy'] for r in evaluations if r['round']>summary['rounds']-10),key)
        if algorithm.startswith('FedTC'):
            macros = read_jsonl(directory/'macros.jsonl')
            if algorithm != 'FedTCFixed':
                post = [m for m in macros if m['gate_reason']!='warmup']
                close(row['postwarmup_decisions'],len(post),key)
                close(row['postwarmup_fallbacks'],sum(m['fallback'] for m in post),key)
                close(row['candidate_executions' if sac else 'accepted_macros'],sum(not m['fallback'] for m in post),key)
            if sac:
                close(row['unique_executed_actions'],len({tuple(m['action']) for m in macros}),key)

    tuning_count = 0
    if sac:
        selection = read_json(root/'reproducibility/fixed_selection.json')
        if selection['used_test_metrics'] or len(selection['scores']) != 9:
            raise ValueError('Unexpected fixed-parameter selection protocol')
        for score in selection['scores']:
            directory = raw/'tuning'/f'grid_{score["grid"]}'
            summary = read_json(directory/'summary.json')
            validation = read_jsonl(directory/'validation_audit.jsonl')
            if summary['status'] != 'complete' or summary['rounds'] != 80:
                raise ValueError('Incomplete tuning run')
            close(score['validation_score'],statistics.mean(r['all_client_macro_loss'] for r in validation if r['round']>64),'fixed validation score')
            tuning_count += 1
        if selection['selected'] != min(selection['scores'],key=lambda x:x['validation_score']):
            raise ValueError('Fixed choice does not follow the recorded validation rule')
    plan = read_json(root/'reproducibility/plan.json')
    source_differences = []
    for name, digest in plan['source_sha256'].items():
        path = ROOT/'code'/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            source_differences.append(name)
    if sac and source_differences:
        raise ValueError(f'SAC study source changed: {source_differences}')
    return {'study':root.name,'archived_files_verified':len(archive['entries']), 'runs_verified':len(results),
            'tuning_runs_verified':tuning_count,
            'checks':['archive file hashes','raw accuracy/curves','post-warmup execution counts','shared data/initialization/training schedules'],
            'historical_source_differences':source_differences,
            'checkpoint_checks':'Not repeated: model and replay checkpoints are not published.',
            'scope':'Single-seed measurements; not an independent training reproduction or a significance test.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archives', nargs='*', type=Path)
    args = p.parse_args()
    archives = args.archives or [ROOT/'results/single_seed_comparison', ROOT/'results/sac_fourway_seed1']
    for root in archives:
        print(json.dumps(verify_archive(root),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
