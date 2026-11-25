"""Test a frozen Stage A policy in the native MNIST 85%-baseline protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.baselines import aggregate
from diagnostics.centered_control import centered
from diagnostics.paired_feedback_control import ACTIONS, Panel, RolloutScorer, design
from diagnostics.refit_last_mnist import pool_config
from diagnostics.run_last_window_mnist import endpoint_cost
from diagnostics.run_shadow_macro_rl_mnist import H, commit, open_world
from diagnostics.stage_a_world import stream
from utils.experiment import fingerprint, set_vector, vector, write_json


def incumbent(world,schedule):
    model=vector(world.global_model)
    for t in range(0,40,H):
        start=world.collect_uploads(model,t,schedule)
        options=[1.15,1.55] if t==35 else [1.15]
        models,_,_=world.aggregate_options(start,model,options)
        ends=[commit(world,m,t,tau,schedule) for m,tau in zip(models,options)]
        keep=int(len(ends)>1 and endpoint_cost(world,ends[1])<endpoint_cost(world,ends[0]))
        model=ends[keep]
    return world.record_test(model)


@torch.no_grad()
def reports(world,panel,param):
    set_vector(world.global_model,param)
    logits=world.global_model(panel.x)
    errors=logits.argmax(1).ne(panel.y).float()
    loss=F.cross_entropy(logits,panel.y,reduction='none').clamp_max(5.)
    return np.array([[float(x.mean()),float(y.mean())] for x,y in
                     zip(errors.split(panel.sizes.tolist()),loss.split(panel.sizes.tolist()))])


def native_run(task):
    config,seed,folder,pack,method=task
    torch.set_num_threads(1)
    world,schedule=open_world(config,seed,folder)
    stamp=[world.manifest[k] for k in ('initial_model_sha256','native_data_sha256','private_splits_sha256')]
    stamp.append(fingerprint(world.schedule))
    if method=='incumbent':
        return dict(seed=seed,identity=stamp,metrics=incumbent(world,schedule),actions=[])
    world.algorithm='FedTCFixed'  # Activate the native per-client training utility.
    model=vector(world.global_model)
    rng=stream(seed,6)
    scorer=RolloutScorer(**{k:np.array(v) for k,v in pack['rl'].items()})
    trace=[]
    for t in range(40):
        set_vector(world.global_model,model)
        world.send_models()
        cohort=[world.clients[i] for i in world.schedule[t]]
        util=world.train_clients(cohort,t,model)
        updates=torch.stack([vector(c.model)-model for c in cohort])
        one=centered(updates,updates.new_tensor(util),(.3,1.05)).delta
        two,_=aggregate('FedAvg',updates,[len(c.train_indices) for c in cohort],4,1)
        if t%H==0:
            selected=np.sort(rng.choice(24,16,replace=False))
            xs,ys=zip(*(world.private[i] for i in selected))
            sizes=torch.tensor([len(y) for y in ys])
            panel=Panel(torch.cat(xs),torch.cat(ys),torch.arange(16).repeat_interleave(sizes),sizes)
            base=reports(world,panel,model)
            gains=np.stack([panel.summarize(base-reports(world,panel,model+scale*torch.lerp(one,two,mix)))
                           for mix,scale in ACTIONS])
            a={'fixed':pack['reference'],'schedule':pack['schedule'][t//H],
               'rl':int(scorer.scores(design(panel.summarize(base),gains,t//H)).argmax())}[method]
            trace.append(a)
        mix,scale=ACTIONS[a]
        model=model+scale*torch.lerp(one,two,mix)
    return dict(seed=seed,identity=stamp,metrics=world.record_test(model),actions=trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    pack=json.loads(args.policy.read_text())
    config=pool_config(Path('dataset').resolve(),out,72,3103)
    seeds=list(range(14600,14608))
    methods=['incumbent','fixed','schedule','rl']
    write_json(out/'protocol.json',dict(seeds=seeds,methods=methods,held_pool=72,
        split_seed=3103,transfer='policy frozen on Stage A; original announced cohort schedule and no contamination'))
    with ProcessPoolExecutor(args.jobs) as pool:
        tasks=[(config,s,out/m/str(s),pack,m) for s in seeds for m in methods]
        result=list(pool.map(native_run,tasks))
    rows={m:result[i::len(methods)] for i,m in enumerate(methods)}
    write_json(out/'held.json',rows)
    boot=np.random.default_rng(241).integers(len(seeds),size=(20000,len(seeds)))
    stats={}
    for m in methods:
        assert [r['identity'] for r in rows[m]]==[r['identity'] for r in rows['incumbent']]
        stats[m]={}
        for key in ('accuracy','worst10'):
            a=np.array([r['metrics'][key] for r in rows[m]])*100
            b=np.array([r['metrics'][key] for r in rows['incumbent']])*100
            stats[m][key]=dict(mean=float(a.mean()),diff=float((a-b).mean()),
                ci95=np.percentile((a-b)[boot].mean(axis=1),[2.5,97.5]).tolist())
    write_json(out/'summary.json',stats)
    print(stats)


if __name__=='__main__':
    main()
