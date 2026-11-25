"""Fit and evaluate server-step control on native MNIST (MLP) or CIFAR-10 (CifarNet)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dataset.prepare_cifar10 import check_clients
from diagnostics.paired_feedback_control import RolloutScorer
from diagnostics.refit_last_mnist import pool_config
from diagnostics.run_last_window_mnist import endpoint_cost
from diagnostics.run_shadow_macro_rl_mnist import open_world
from diagnostics.stage_a_world import stream
from utils.experiment import fingerprint, set_vector, vector, write_json

STEPS=(.5,1.,1.5)
FIT=range(16000,16024)
TUNE=range(16100,16108)
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


def fit_seed(task):
    config,s,folder=task
    torch.set_num_threads(1)
    w,schedule=open_world(config,s,folder)
    start=vector(w.global_model)
    rng=stream(s,23)
    selected={(s-16000)%8,(s-16000+4)%8}
    rows=[]
    for t in range(8):
        if t not in selected:
            start=commit(w,start,t,1,schedule)
            continue
        first,features,_=probe(w,start,t,schedule)
        cost0=endpoint_cost(w,start)
        ends=[];targets=[]
        for a in range(3):
            end=commit(w,start,t,a,schedule,first)
            c1=endpoint_cost(w,end)
            c2=endpoint_cost(w,commit(w,end,t+1,1,schedule)) if t<7 else c1
            targets.append([cost0-c1,cost0-c1+.9*(c1-c2)])
            ends.append(end)
        rows.append(dict(seed=s,t=t,x=features.tolist(),labels=targets))
        a=1 if rng.random()<.5 else int(rng.integers(3))
        start=ends[a]
    return rows


def select(rule,features,pilot,t):
    if 'time' in rule: return rule['time'][t]
    if 'arm' in rule: return rule['arm']
    if rule.get('direct'): scores=pilot
    else: scores=RolloutScorer(np.array(rule['scale']),np.array(rule['weight'])).scores(features)
    best=int(scores.argmax())
    return 1 if scores[1]>=scores[best]-1e-12 else best


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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='New directory for dev; existing dev directory for held/deploy')
    p.add_argument('--phase',choices=('check','dev','held','deploy'),required=True)
    p.add_argument('--dataset',type=dataset_name,help='mnist (dev default) or Cifar10; held/deploy infer the frozen dataset')
    p.add_argument('--data-root',type=Path,help='Parent of mnist/ or Cifar10/ (default: code/dataset during dev)')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--seed',type=int)
    p.add_argument('--checkpoint',type=Path,help='Save the deployed model state_dict to a new file')
    p.add_argument('--method',choices=('incumbent','fixed','schedule','direct','immediate','rl'),default='rl')
    args=p.parse_args()
    if args.phase != 'check' and args.output is None:
        p.error('--output is required for dev, held and deploy')
    if args.jobs < 1 or (args.seed is not None and args.seed < 0):
        p.error('--jobs must be positive and --seed must be nonnegative')
    if args.phase == 'deploy' and args.seed is None:
        p.error('--seed is required for deployment')
    if args.checkpoint and (args.phase != 'deploy' or args.checkpoint.exists()):
        p.error('--checkpoint requires deploy and a new file path')
    out = args.output.expanduser().resolve() if args.output else None
    dev = args.phase == 'dev'
    dataset, protocol = args.dataset or 'mnist', {}
    try:
        if args.phase in ('held', 'deploy'):
            dataset, protocol, rules = load_rules(out, args.dataset)
        data = Path(args.data_root or protocol.get('data_root') or Path(__file__).resolve().parents[1] / 'dataset')
        data = data.expanduser().resolve()
        # Legacy held runs can still reuse their saved pool, including its custom source path.
        legacy_pool = (args.phase == 'held' and args.data_root is None and 'data_root' not in protocol
                       and (out / 'pools.json').is_file())
        if not legacy_pool:
            check_source(data, dataset, range(96) if dev or args.phase == 'check' else range(72, 96))
            if 'held_files_sha256' in protocol and held_digest(data, dataset) != protocol['held_files_sha256']:
                raise ValueError("Client files 72–95 differ from the frozen protocol. Supply the original partition.")
    except (ValueError, OSError) as exc:
        p.exit(1, f"{exc}\n")
    if args.phase == 'check':
        with TemporaryDirectory() as f:
            folder = Path(f)
            config = step_pool(data, folder, 0, dataset)
            world, schedule = open_world(config, args.seed or 0, folder / 'check')
            _, features, pilot = probe(world, vector(world.global_model), 0, schedule)
            if not np.isfinite(features).all() or not np.isfinite(pilot).all():
                raise ValueError("Nonfinite private feedback during first-round probe")
            print(json.dumps(dict(dataset=dataset, model=type(world.global_model).__name__,
                                  device=world.args.device, uploads=len(world.schedule[0]),
                                  steps=STEPS, private_gains=pilot.tolist())))
        return
    if args.phase=='deploy':
        with TemporaryDirectory() as f:
            folder=Path(f)
            cfg=step_pool(data,folder,72,dataset)
            rule=rules[args.method]
            row=rollout((cfg,args.seed,folder/'trajectory',args.method,rule,True),args.checkpoint)
            print(json.dumps(row))
        return
    with ExitStack() as stack:
        # Keep client links alive until every worker exits, including on an exception.
        if not dev and not legacy_pool:
            folder = Path(stack.enter_context(TemporaryDirectory()))
            config = step_pool(data, folder, 72, dataset)
        pool = stack.enter_context(ProcessPoolExecutor(args.jobs))
        if dev:
            out.mkdir(parents=True,exist_ok=False)
            write_json(out/'protocol.json',dict(dataset=dataset, data_root=str(data), held_files_sha256=held_digest(data,dataset),
                steps=STEPS,fit=list(FIT),tune=list(TUNE),held=list(HELD),
                client_pools=[0,24,48,72],H=5,gamma=.9,primary='final test accuracy',
                paired='all actions: same five rounds, then five more at native step 1; final window uses common tau check',
                threshold='positive paired 95% lower bound vs incumbent, tuned fixed, time, immediate and direct'))
            cfgs=[step_pool(data,out,offset,dataset) for offset in (0,24,48,72)]
            write_json(out/'pools.json',cfgs)
            tasks=[(cfgs[i%2],s,out/'fit'/str(s)) for i,s in enumerate(FIT)]
            fit=[]
            for seed, group in zip(FIT, pool.map(fit_seed,tasks)):
                fit.extend(group)
                print(f'fit seed {seed} complete', flush=True)
            write_json(out/'paired_training.json',fit)
            x=[r['x'] for r in fit]
            y=np.array([r['labels'] for r in fit])
            rules={'step_'+str(a):dict(arm=a) for a in range(3)}
            rules['direct']=dict(direct=True)
            for j,k in enumerate(('immediate','rl')):
                reg=RolloutScorer.fit(x,y[:,:,j],32)
                rules[k]=dict(scale=reg.scale.tolist(),weight=reg.weight.tolist())
                schedule=[]
                for t in range(8):
                    values=np.mean([r['labels'] for r in fit if r['t']//2==t//2],axis=0)
                    schedule.append(int(values[:,j].argmax()))
                rules['time_'+k]=dict(time=schedule)
            config=cfgs[2];bank=TUNE
        else:
            if legacy_pool:
                config=json.loads((out/'pools.json').read_text())[3]
            bank=protocol.get('held', HELD)
        split='tune' if dev else 'held'
        tasks=[(config,s,out/split/str(s),rules,not dev) for s in bank]
        evaluations=[]
        for seed, result in zip(bank, pool.map(shared_evaluation,tasks)):
            evaluations.append(result)
            print(f'{split} seed {seed} complete', flush=True)
        rows={k:[r[k] for r in evaluations] for k in rules}
        write_json(out/(split+'.json'),rows)
        if dev:
            mean={k:np.mean([z['metrics']['accuracy'] for z in rr]) for k,rr in rows.items()}
            winner=lambda opts:max(opts,key=lambda k:mean[k])
            lock={'incumbent':'step_1','fixed':winner(['step_0','step_1','step_2']),
                  'schedule':winner(['time_immediate','time_rl','step_0','step_1','step_2']),
                  'direct':'direct','immediate':'immediate','rl':'rl'}
            write_json(out/'selection.json',dict(means=mean,lock=lock))
            write_json(out/'policy.json',{k:dict(rules[v], dataset=dataset) for k,v in lock.items()})
            print('locked',mean,lock,flush=True)
        else:
            boot=np.random.default_rng(241).integers(len(bank),size=(20000,len(bank)))
            summary={}
            for ref in rows:
                summary[ref]={}
                for m,group in rows.items():
                    assert [r['identity'] for r in group]==[r['identity'] for r in rows[ref]]
                    summary[ref][m]={}
                    for k in ('accuracy','worst10'):
                        vals=np.array([r['metrics'][k] for r in group])*100
                        d=vals-np.array([r['metrics'][k] for r in rows[ref]])*100
                        summary[ref][m][k]=dict(mean=float(vals.mean()),diff=float(d.mean()),
                            ci95=np.percentile(d[boot].mean(axis=1),[2.5,97.5]).tolist())
            write_json(out/'summary.json',summary)
            print('held',{k:v['accuracy']['mean'] for k,v in summary['incumbent'].items()})


if __name__=='__main__':
    main()
