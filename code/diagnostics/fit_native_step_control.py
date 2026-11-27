"""Fit and evaluate server-step control on native MNIST (MLP) or CIFAR-10 (CifarNet)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dataset.prepare_cifar10 import check_clients
from algorithms.fedtc.step_control import STEPS, select
from servers.step_experiment import paired_summary
from diagnostics.refit_last_mnist import pool_config
from diagnostics.run_last_window_mnist import endpoint_cost
from diagnostics.run_shadow_macro_rl_mnist import open_world
from utils.experiment import fingerprint, set_vector, vector, write_json

HELD=range(16200,16212)


def step_pool(data, folder, offset, dataset):
    config = pool_config(data, folder, offset, 3103, dataset)
    config.update(actions=list(STEPS), action_kind="server_step", classification_cost=True)
    return config


def directions(w,start,i,schedule,taus=(1.15,)):
    uploads=w.collect_uploads(start,i,schedule)
    ends,_,_=w.aggregate_options(uploads,start,taus)
    return [e-start for e in ends]


def commit(w,start,t,a,schedule,first=None):
    taus=(1.15,1.55) if t==7 else (1.15,)
    first=directions(w,start,t*5,schedule,taus) if first is None else first
    ends=[]
    for tau,delta in zip(taus,first):
        model=start+STEPS[a]*delta
        for i in range(t*5+1,(t+1)*5):
            d=directions(w,model,i,schedule,(tau,))[0]
            model=model+STEPS[a]*d
        ends.append(model)
    if len(ends)==2 and endpoint_cost(w,ends[1])<endpoint_cost(w,ends[0]):
        return ends[1]
    return ends[0]


def probe(w,start,t,schedule):
    first=directions(w,start,t*5,schedule,(1.15,1.55) if t==7 else (1.15,))
    # Announced upload slots are public protocol metadata. All clients may be
    # evaluators, including those outside the next training cohort.
    w.classification_cost=True
    initial=w.private_losses(start)
    here=np.array(w.schedule[t*5])
    upcoming=np.array(w.schedule[min(39,(t+1)*5)])
    groups=[np.arange(24),here,upcoming]
    features=[]
    pilot=[]
    for a,step in enumerate(STEPS):
        gains=initial-w.private_losses(start+step*first[0])
        means=[np.average(gains[ids],weights=w.sizes[ids]) for ids in groups]
        identity=np.eye(len(STEPS))[a]
        features.append(np.r_[identity,identity*t/7,means, np.asarray(means)*t/7])
        pilot.append(means[0])
    return first,np.array(features),np.array(pilot)


def rollout(task,checkpoint=None):
    config,s,folder,name,rule,test=task
    torch.set_num_threads(1)
    w,schedule=open_world(config,s,folder)
    stamp=[w.manifest[k] for k in ('initial_model_sha256','native_data_sha256','private_splits_sha256')]+[fingerprint(w.schedule)]
    model=vector(w.global_model)
    choices=[]
    for t in range(8):
        first,features,pilot=probe(w,model,t,schedule)
        a=select(rule,features,pilot,t)
        model=commit(w,model,t,a,schedule,first)
        choices.append(a)
    if test:
        metrics=w.record_test(model)
    else:
        set_vector(w.global_model,model)
        with torch.no_grad():
            metrics={'accuracy':float(w.private_logits().argmax(1).eq(w.private_y).float().mean())}
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        with checkpoint.open('xb') as output:
            torch.save(w.global_model.state_dict(),output)
    return dict(seed=s,identity=stamp,method=name,actions=choices,metrics=metrics)


def shared_evaluation(task):
    """Run equal action prefixes once; SGD, uploads and validation panels are deterministic."""
    config,seed,folder,rules,test=task
    torch.set_num_threads(1)
    w,schedule=open_world(config,seed,folder)
    stamp=[w.manifest[k] for k in ('initial_model_sha256','native_data_sha256','private_splits_sha256')]
    stamp.append(fingerprint(w.schedule))
    result={}
    def visit(model,t,names,history):
        if t==8:
            if test:
                metrics=w.record_test(model)
            else:
                set_vector(w.global_model,model)
                with torch.no_grad():
                    acc=w.private_logits().argmax(1).eq(w.private_y).float().mean()
                metrics={'accuracy':float(acc)}
            for name in names:
                result[name]=dict(seed=seed,identity=stamp,method=name,actions=history,metrics=metrics)
            return
        first,features,pilot=probe(w,model,t,schedule)
        branches={}
        for name in names:
            a=select(rules[name],features,pilot,t)
            branches.setdefault(a,[]).append(name)
        for a,group in branches.items():
            successor=commit(w,model,t,a,schedule,first)
            visit(successor,t+1,group,history+[a])
    visit(vector(w.global_model),0,list(rules),[])
    return result


def dataset_name(value):
    names = {"mnist": "mnist", "cifar10": "Cifar10"}
    try:
        return names[value.lower().replace("-", "")]
    except KeyError:
        raise argparse.ArgumentTypeError("Choose mnist or Cifar10")


def check_source(data, dataset, ids):
    root = data / dataset
    if dataset == "Cifar10":
        check_clients(root, ids)
    else:
        for i in ids:
            for part in ("train", "test"):
                path = root / part / f"{i}.npz"
                if not path.is_file():
                    raise ValueError(f"Missing client partition: {path}")


def held_digest(data, dataset):
    files = {f"{part}/{i}.npz": hashlib.sha256((data / dataset / part / f"{i}.npz").read_bytes()).hexdigest()
             for part in ("train", "test") for i in range(72, 96)}
    return fingerprint(files)


def load_rules(out, requested):
    rules = json.loads((out / "policy.json").read_text())
    path = out / "protocol.json"
    protocol = json.loads(path.read_text()) if path.is_file() else {}
    # The protocol and policies released before CIFAR support used MNIST exclusively.
    datasets = {rule.get("dataset", "mnist") for rule in rules.values()}
    if "dataset" in protocol:
        datasets.add(protocol["dataset"])
    if len(datasets) != 1:
        raise ValueError("The dataset tags in the frozen rules and protocol disagree")
    dataset = datasets.pop()
    if requested is not None and requested != dataset:
        raise ValueError(f"Frozen policy uses {dataset}, requested {requested}. Run --phase dev for {requested} first.")
    return dataset, protocol, rules


def command_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='New directory for dev; existing dev directory for held/deploy')
    p.add_argument('--phase',choices=('check','dev','held','deploy'),required=True)
    p.add_argument('--dataset',type=dataset_name,help='mnist (dev default) or Cifar10; held/deploy infer the frozen dataset')
    p.add_argument('--data-root',type=Path,help='Parent of mnist/ or Cifar10/ (default: code/dataset during dev)')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--seed',type=int)
    p.add_argument('--checkpoint',type=Path,help='Save the deployed model state_dict to a new file')
    p.add_argument('--method',choices=('incumbent','fixed','schedule','direct','immediate','rl'),default='rl')
    return p


def evaluate_archived(args):
    """Read pre-versioned 40-round policy artifacts with their recorded pools."""
    out = args.output.expanduser().resolve()
    dataset, protocol, rules = load_rules(out, args.dataset)
    data = Path(args.data_root or protocol.get('data_root') or Path(__file__).resolve().parents[1] / 'dataset')
    data = data.expanduser().resolve()
    legacy_pool = (args.phase == 'held' and args.data_root is None and 'data_root' not in protocol
                   and (out / 'pools.json').is_file())
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        if legacy_pool:
            config = json.loads((out / 'pools.json').read_text())[3]
        else:
            check_source(data, dataset, range(72, 96))
            if 'held_files_sha256' in protocol and held_digest(data, dataset) != protocol['held_files_sha256']:
                raise ValueError('Held partition differs from the recorded files')
            config = step_pool(data, folder, 72, dataset)
        if args.phase == 'deploy':
            row = rollout((config, args.seed, folder/'trajectory', args.method, rules[args.method], True), args.checkpoint)
            print(json.dumps(row))
            return
        bank = protocol.get('held', HELD)
        tasks = [(config, seed, out/'held'/str(seed), rules, True) for seed in bank]
        with ProcessPoolExecutor(args.jobs) as workers:
            results = list(workers.map(shared_evaluation, tasks))
        rows = {k: [r[k] for r in results] for k in rules}
        write_json(out / 'held.json', rows)
        write_json(out / 'summary.json', paired_summary(rows))


def main():
    """Translate historical command-line flags to the shared training entry."""
    import shutil
    from main import parse_config, run

    parser = command_parser()
    args = parser.parse_args()
    if args.jobs < 1 or (args.seed is not None and args.seed < 0):
        parser.error('Use positive jobs and nonnegative seeds')
    if args.phase != 'check' and args.output is None:
        parser.error('--output is required')
    if args.phase == 'deploy' and args.seed is None:
        parser.error('deploy requires --seed')
    if args.checkpoint and (args.phase != 'deploy' or args.checkpoint.exists()):
        parser.error('--checkpoint requires deploy and a new file path')
    out = args.output.resolve() if args.output else None
    policy = out / 'policy.json' if out else None
    dataset = args.dataset or 'mnist'
    data_root = args.data_root
    if args.phase in ('held', 'deploy'):
        frozen = json.loads(policy.read_text())
        if 'rules' not in frozen:
            # Older unversioned artifacts carry the original fixed 40-round protocol.
            return evaluate_archived(args)
        dataset = args.dataset or frozen['protocol']['dataset']
        if data_root is None:
            data_root = json.loads((out / 'config.json').read_text())['data_root']

    config = Path(__file__).resolve().parents[1] / 'configs/mnist_native_40r.json'
    argv = ['--config', str(config), '--phase', args.phase, '--dataset', dataset,
            '--model', 'auto', '--jobs', str(args.jobs), '--method', args.method]
    if data_root:
        argv += ['--data-root', str(data_root)]
    if args.seed is not None:
        argv += ['--seed', str(args.seed)]
    with TemporaryDirectory() as directory:
        folder = out if args.phase == 'dev' else Path(directory) / 'run'
        argv += ['--output', str(folder)]
        if args.phase in ('held', 'deploy'):
            argv += ['--policy', str(policy)]
        result = run(parse_config(argv))
        if args.phase == 'held':
            for name in ('held.json', 'summary.json'):
                shutil.copyfile(folder / name, out / name)
        if args.checkpoint:
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with (folder / 'deploy' / str(args.seed) / 'model.pt').open('rb') as src, args.checkpoint.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        if args.phase == 'check':
            result.update(dataset=dataset, model='CifarNet' if dataset == 'Cifar10' else 'OneHiddenLayerFc',
                          device='cpu', uploads=len(result['uploads']), steps=STEPS)
        if args.phase in ('check', 'deploy'):
            print(json.dumps(result))


if __name__ == '__main__':
    main()
