"""Development, locking and evaluation of the native server-step policy."""
import copy
import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from algorithms.fedtc.step_control import STEPS, TAUS, fit_rules, freeze
from servers.serverFedTCStep import FedTCStep
from utils.experiment import fingerprint, write_json


def protocol(args):
    keys = ('dataset', 'model_name', 'num_classes', 'num_clients', 'num_sel_clients', 'training_cohort',
            'global_rounds', 'macro_length', 'local_steps', 'batch_size', 'local_learning_rate',
            'learning_rate_decay', 'validation_fraction', 'split_seed', 'dirichlet_alpha', 'cohort_schedule',
            'pool_offsets', 'loss_cap', 'tc_gamma')
    config = {key: getattr(args, key) for key in keys}
    if args.local_steps is None:
        config['local_epochs'] = args.local_epochs
    if args.learning_rate_decay:
        config['learning_rate_decay_gamma'] = args.learning_rate_decay_gamma
    return dict(version=1, steps=list(STEPS), taus=list(TAUS), **config)


def pool_digest(root, offset, count):
    return fingerprint({f'{part}/{i}.npz': hashlib.sha256((root / part / f'{i}.npz').read_bytes()).hexdigest()
                        for part in ('train', 'test') for i in range(offset, offset+count)})


def link_pool(source, root, args, offset):
    dest = root / args.dataset
    for part in ('train', 'test'):
        (dest / part).mkdir(parents=True)
        for local in range(args.num_clients):
            (dest / part / f'{local}.npz').symlink_to((source / part / f'{offset+local}.npz').resolve(strict=True))
    if args.dirichlet_alpha is not None:
        (dest / 'config.json').symlink_to((source / 'config.json').resolve(strict=True))


def trajectory(task):
    from main import prepare
    args, stage, index, rules = task
    server = FedTCStep(prepare(args))
    if stage == 'fit':
        result = server.paired_training(index)
    elif stage == 'check':
        _, design, gains = server.probe(server.initial, 0)
        result = dict(status='ok', input_source=getattr(args, 'input_source', 'prepared client partitions'),
                      dataset=args.dataset, model=args.model_name, device=args.device,
                      configured_rounds=args.global_rounds, checked_rounds=1, windows=server.windows,
                      uploads=server.schedule[0], private_gains=gains.tolist(), feature_shape=list(design.shape),
                      costs=server.costs.copy())
        print(json.dumps(result), flush=True)
    else:
        checkpoint = server.output / 'model.pt' if stage == 'deploy' else None
        result = server.evaluate_rules(rules, test=stage != 'tune', checkpoint=checkpoint)
    write_json(server.output / 'costs.json', server.costs)
    write_json(server.output / 'result.json', result)
    print(f'{stage} seed {args.seed} complete', flush=True)
    return result


def paired_summary(rows):
    first = next(iter(rows.values()))
    seeds = [r['seed'] for r in first]
    size = len(seeds)
    if not size or len(set(seeds)) != size:
        raise ValueError('Held statistics require distinct seeds and at least one trajectory')
    identities = [r['identity'] for r in first]
    for group in rows.values():
        if [r['seed'] for r in group] != seeds or [r['identity'] for r in group] != identities:
            raise ValueError('Unmatched paired evaluation')
    boot = np.random.default_rng(241).integers(size, size=(20000, size))
    summary = {}
    for ref, reference in rows.items():
        comparison = summary[ref] = {}
        for name, group in rows.items():
            comparison[name] = {}
            for metric in ('accuracy', 'worst10'):
                values = np.array([r['metrics'][metric] for r in group]) * 100
                diff = values - np.array([r['metrics'][metric] for r in reference]) * 100
                comparison[name][metric] = dict(
                    n=size, mean=float(values.mean()),
                    sample_variance=float(values.var(ddof=1)) if size > 1 else None,
                    diff=float(diff.mean()),
                    ci95=np.percentile(diff[boot].mean(axis=1), [2.5, 97.5]).tolist())
    return summary


def write_seed_report(path, summary):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'n_seeds', 'accuracy_mean_pct', 'accuracy_sample_variance_pp2',
                         'worst10_mean_pct', 'worst10_sample_variance_pp2'])
        for method, comparison in summary.items():
            metrics = comparison[method]
            accuracy, worst10 = metrics['accuracy'], metrics['worst10']
            writer.writerow([method, accuracy['n'], accuracy['mean'], accuracy['sample_variance'],
                             worst10['mean'], worst10['sample_variance']])


def run_experiment(args):
    from main import validate
    validate(args)
    if args.ad_ratio or args.attack != 'none' or args.report_attack != 'none' or args.fake_utility:
        raise ValueError('FedTCStep uses honest feedback and clean uploads: set attack=none and ad_ratio=0')
    if args.prev != 0 or args.times != 1:
        raise ValueError('Use --fit-seeds, --tune-seeds, --held-seeds or deploy --seed for FedTCStep')
    if args.jobs < 1 or (args.jobs != 1 and args.device == 'cuda'):
        raise ValueError('Use positive --jobs; submit one process per GPU with --jobs 1')
    if not 0 <= args.tc_gamma <= 1:
        raise ValueError('tc_gamma must lie in [0,1]')
    if args.pool_offsets is None:
        args.pool_offsets = [0, 0, 0, 0]
    if min(args.pool_offsets) < 0:
        raise ValueError('Pool offsets must be nonnegative')
    banks = args.fit_seeds + args.tune_seeds + args.held_seeds
    if min(banks) < 0 or len(banks) != len(set(banks)):
        raise ValueError('Development, tuning and held seeds must be distinct and nonnegative')
    if args.cohort_schedule == 'grouped' and args.num_sel_clients > args.num_clients // 2:
        raise ValueError('grouped requires K <= floor(N/2); use uniform for larger K')
    args.buffer_size = 1  # shared base allocates a replay buffer; this controller uses paired regression
    if args.output is None:
        from datetime import datetime
        args.output = str(Path('runs') / f'FedTCStep_{args.phase}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    out = Path(args.output).resolve()
    source = Path(args.data_root) / args.dataset
    if args.partition_clients is not None:
        if max(args.pool_offsets) + args.num_clients > args.partition_clients:
            raise ValueError("The requested client pools exceed partition_clients")
        metadata = json.loads((source / 'config.json').read_text())
        count = metadata.get('clients', metadata.get('num_clients'))
        if count != args.partition_clients:
            raise ValueError(f"Requested {args.partition_clients} partitions, metadata lists {count}")
    if args.partition_seed is not None:
        metadata = json.loads((source / 'config.json').read_text())
        if metadata.get('seed') != args.partition_seed:
            raise ValueError('partition_seed differs from the prepared dataset metadata')
    signature = protocol(args)
    held_pool = args.pool_offsets[3]
    if args.phase in ('held', 'deploy'):
        if not args.policy:
            raise ValueError('Specify --policy pointing to policy.json from --phase dev')
        saved = json.loads(Path(args.policy).read_text())
        if saved['protocol'] != signature:
            mismatch = [k for k in signature if signature[k] != saved['protocol'].get(k)]
            raise ValueError(f'Frozen policy protocol mismatch: {mismatch}')
        digest = pool_digest(source, held_pool, args.num_clients)
        if saved['held_files_sha256'] != digest:
            raise ValueError('Held data differs from the frozen partition')
        if args.phase == 'held' and set(args.held_seeds) & set(saved['development_seeds']):
            raise ValueError('Held seeds overlap the frozen policy development seeds')
        rules = saved['rules']
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', vars(args))
    write_json(out / 'protocol.json', signature)
    with TemporaryDirectory() as temp:
        roots = {}
        def run(stage, seeds, offsets, rules=None):
            tasks = []
            for i, seed in enumerate(seeds):
                offset = offsets[i % len(offsets)]
                if offset not in roots:
                    roots[offset] = Path(temp) / str(offset)
                    link_pool(source, roots[offset], args, offset)
                current = copy.copy(args)
                current.source_data_root, current.client_offset = args.data_root, offset
                current.data_root, current.seed = str(roots[offset]), seed
                current.output = str(out / stage / str(seed))
                tasks.append((current, stage, i, rules))
            if args.jobs == 1 or len(tasks) == 1:
                return [trajectory(task) for task in tasks]
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context('spawn')) as workers:
                return list(workers.map(trajectory, tasks))

        if args.phase == 'check':
            result = run('check', [args.seed], [args.pool_offsets[0]])[0]
        elif args.phase == 'dev':
            digest = pool_digest(source, held_pool, args.num_clients)
            groups = run('fit', args.fit_seeds, args.pool_offsets[:2])
            rows = [row for group in groups for row in group]
            write_json(out / 'paired_training.json', rows)
            candidates = fit_rules(rows, math.ceil(args.global_rounds / args.macro_length))
            tune = run('tune', args.tune_seeds, [args.pool_offsets[2]], candidates)
            by_method = {k: [r[k] for r in tune] for k in candidates}
            write_json(out / 'tune.json', by_method)
            locked, selection = freeze(candidates, by_method)
            write_json(out / 'selection.json', selection)
            write_json(out / 'policy.json', dict(protocol=signature, held_files_sha256=digest, rules=locked,
                                                 development_seeds=args.fit_seeds + args.tune_seeds))
            result = selection
        elif args.phase == 'held':
            held = run('held', args.held_seeds, [held_pool], rules)
            rows = {k: [r[k] for r in held] for k in rules}
            write_json(out / 'held.json', rows)
            result = paired_summary(rows)
            write_seed_report(out / 'seed_statistics.csv', result)
        else:
            result = run('deploy', [args.seed], [held_pool], {args.method: rules[args.method]})[0][args.method]
    write_json(out / 'summary.json', result)
    print(f'Results: {out}', flush=True)
    return result
