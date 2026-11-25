"""Evaluate validation-based weighting at the original native optimizer step size."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.fedtc.forward import calibrate
from diagnostics.fit_round_feedback import features_rule
from diagnostics.paired_feedback_control import Panel
from diagnostics.refit_last_mnist import pool_config
from diagnostics.round_feedback_mnist import MIXES
from diagnostics.run_shadow_macro_rl_mnist import open_world
from diagnostics.stage_a_world import stream
from diagnostics.transfer_paired_mnist import incumbent, reports
from diagnostics.validated_directions import trust_weights
from utils.experiment import fingerprint, set_vector, vector, write_json


def gradients(world, p, start, updates):
    set_vector(world.global_model,start)
    previous=world.global_model.training
    world.global_model.eval()
    try:
        result=[]
        for x,y in zip(p.x.split(p.sizes.tolist()),p.y.split(p.sizes.tolist())):
            cross=F.cross_entropy(world.global_model(x),y)
            grads=torch.autograd.grad(cross,tuple(world.global_model.parameters()))
            result.append(-updates@torch.cat([g.flatten() for g in grads]))
        ratios=p.sizes.to(updates.dtype)
        return torch.stack(result).T @ (ratios/ratios.sum())
    finally:
        world.global_model.train(previous)


def panel(world, ids):
    xy=[world.private[i] for i in ids]
    xs,ys=zip(*xy)
    sizes=torch.tensor([len(y) for y in ys])
    return Panel(torch.cat(xs),torch.cat(ys),torch.arange(len(ids)).repeat_interleave(sizes),sizes)


def run_one(task, evaluation="test"):
    config,seed,folder,method,rule,save_model=task
    torch.set_num_threads(1)
    w,schedule=open_world(config,seed,folder)
    stamp=[w.manifest[k] for k in ('initial_model_sha256','native_data_sha256','private_splits_sha256')]
    stamp.append(fingerprint(w.schedule))
    if method=='incumbent':
        return dict(seed=seed,identity=stamp,metrics=incumbent(w,schedule),mix=[],method=method)
    model=vector(w.global_model)
    rng=stream(seed,6)
    choices=[]
    for i in range(40):
        if i%5==0:
            clients=rng.choice(24,16,replace=False)
            proposing,checking=panel(w,clients[:8]),panel(w,clients[8:])
        updates=w.collect_uploads(model,i,schedule)
        prior=calibrate(updates,updates.new_zeros(len(updates)),(0.,1.15)).weights
        # Clipped updates start at zero mass and remain excluded.
        g=gradients(w,proposing,model,updates)
        verified=trust_weights(prior,g)
        one,two=updates.T.mv(verified),updates.T.mv(prior)
        models=[model+torch.lerp(one,two,m) for m in MIXES]
        before=reports(w,checking,model) if rule.get('model') is not None else None
        stats=[reports(w,checking,m) for m in models]
        a,_=features_rule(rule,checking,stats,before,i,1.)
        model=models[a]
        choices.append(MIXES[a])
    if evaluation=='test':
        metrics=w.record_test(model)
    else:
        set_vector(w.global_model,model)
        with torch.no_grad():
            metrics={'accuracy':float(w.global_model(w.private_x).argmax(1).eq(w.private_y).float().mean())}
    if save_model:
        torch.save(w.global_model.state_dict(),folder/'global_model.pt')
    return dict(seed=seed,identity=stamp,metrics=metrics,mix=choices,method=method)


def validation_run(task):
    return run_one(task,'validation')


def screen_strength(out,jobs):
    out.mkdir(parents=True,exist_ok=False)
    calibration=pool_config(Path('dataset').resolve(),out,48,3103)
    held=pool_config(Path('dataset').resolve(),out,72,3103)
    seeds=list(range(16400,16406))
    final=list(range(16200,16212))
    write_json(out/'protocol.json',dict(calibration=seeds,held=final,mixes=MIXES,
        selection='final pooled validation accuracy on separate native client pool 48..71'))
    with ProcessPoolExecutor(jobs) as pool:
        tasks=[(calibration,s,out/'calibrate'/str(a)/str(s),'fixed',dict(schedule_mix=[a]*40),False)
               for a in range(3) for s in seeds]
        data=list(pool.map(validation_run,tasks))
        scores=np.array([r['metrics']['accuracy'] for r in data]).reshape(3,-1).mean(1)
        a=int(scores.argmax())
        write_json(out/'locked.json',dict(mix=MIXES[a],scores=scores.tolist()))
        tasks=[(held,s,out/'held'/str(s),'fixed',dict(schedule_mix=[a]*40),False) for s in final]
        rows=list(pool.map(run_one,tasks))
    write_json(out/'held.json',rows)
    print('native fixed mix',MIXES[a],np.mean([r['metrics']['accuracy'] for r in rows]),flush=True)


def independent_clients(out,jobs):
    """One fixed additional replication, 16 new seeds per disjoint client pool."""
    out.mkdir(parents=True,exist_ok=False)
    seeds=list(range(16300,16332))
    pools=(48,72)
    write_json(out/'protocol.json',dict(seeds=seeds,client_offsets=pools,per_pool=16,
        split_seed=3103,methods=['incumbent','direct'],rule={'cost':'error'},stopping='all 32 cases',
        scope='same native optimizer and availability construction; report both pools separately'))
    configs={offset:pool_config(Path('dataset').resolve(),out,offset,3103) for offset in pools}
    methods=('incumbent','direct')
    with ProcessPoolExecutor(jobs) as pool:
        tasks=[(configs[pools[i%2]],seed,out/m/str(seed),m,{'cost':'error'},False)
               for i,seed in enumerate(seeds) for m in methods]
        flat=list(pool.map(run_one,tasks))
    rows={k:flat[i::2] for i,k in enumerate(methods)}
    write_json(out/'held.json',rows)
    summary={}
    for label,index in [('pool48',slice(0,None,2)),('pool72',slice(1,None,2)),('both',slice(None))]:
        group,baseline=rows['direct'][index],rows['incumbent'][index]
        assert [r['identity'] for r in group]==[r['identity'] for r in baseline]
        n=len(group)
        rng=np.random.default_rng(241)
        if label=='both':
            boot=np.c_[2*rng.integers(16,size=(20000,16)),1+2*rng.integers(16,size=(20000,16))]
        else:
            boot=rng.integers(n,size=(20000,n))
        summary[label]={}
        for k in ('accuracy','worst10'):
            vals=np.array([r['metrics'][k] for r in group])*100
            refs=np.array([r['metrics'][k] for r in baseline])*100
            delta=vals-refs
            summary[label][k]=dict(mean=float(vals.mean()),reference=float(refs.mean()),diff=float(delta.mean()),
                ci95=np.percentile(delta[boot].mean(axis=1),[2.5,97.5]).tolist(),wins=int((delta>0).sum()),n=n)
    write_json(out/'summary.json',summary)
    print(summary,flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--phase',choices=('transfer','confirm','deploy','strength','replicate'),default='transfer')
    a.add_argument('--seed',type=int)
    a.add_argument('--policy',type=Path)
    a.add_argument('--baseline',type=Path)
    a.add_argument('--jobs',type=int,default=3)
    args=a.parse_args()
    if args.phase=='replicate':
        independent_clients(args.output,args.jobs)
        return
    if args.phase=='strength':
        screen_strength(args.output,args.jobs)
        return
    if args.phase=='transfer' and (not args.policy or not args.baseline):
        a.error('transfer requires --policy and --baseline')
    if args.phase=='deploy' and args.seed is None:
        a.error('deploy requires --seed')
    pack=json.loads(args.policy.read_text()) if args.policy else dict(
        family='validation',rules={'direct':{'cost':'error'}},selection={'direct':'error'})
    if pack['family']!='validation':
        raise ValueError('Native transfer requires a validation-direction policy')
    baseline=json.loads(args.baseline.read_text())['incumbent'] if args.phase=='transfer' else None
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    config=pool_config(Path('dataset').resolve(),out,72,3103)
    if args.phase=='deploy':
        row=run_one((config,args.seed,out/'training','direct',pack['rules']['direct'],True))
        write_json(out/'result.json',row)
        print(json.dumps(row))
        return
    seeds=[r['seed'] for r in baseline] if baseline else list(range(16200,16212))
    methods=('schedule','direct','immediate','rl') if baseline else ('incumbent','direct')
    write_json(out/'protocol.json',dict(seeds=seeds,held_pool=72,split_seed=3103,step=1.,prior=[0.,1.15],
        methods=methods,policy_selection=pack['selection'],native_incumbent='fixed tau with terminal check',
        confirmation='all policy coefficients frozen on Stage A; same initialization, cohorts and local training as incumbent'))
    with ProcessPoolExecutor(args.jobs) as pool:
        tasks=[(config,s,out/m/str(s),m,pack['rules'].get(m),False) for s in seeds for m in methods]
        result=list(pool.map(run_one,tasks))
    rows={m:result[i::len(methods)] for i,m in enumerate(methods)}
    if baseline is not None:
        rows['incumbent']=baseline
    write_json(out/'held.json',rows)
    boot=np.random.default_rng(241).integers(len(seeds),size=(20000,len(seeds)))
    summary={}
    for ref in rows:
        summary[ref]={}
        for m,records in rows.items():
            assert [r['identity'] for r in records]==[r['identity'] for r in rows[ref]]
            summary[ref][m]={}
            for k in ('accuracy','worst10'):
                vals=np.array([r['metrics'][k] for r in records])*100
                diff=vals-np.array([r['metrics'][k] for r in rows[ref]])*100
                summary[ref][m][k]=dict(mean=float(vals.mean()),diff=float(diff.mean()),
                  ci95=np.percentile(diff[boot].mean(axis=1),[2.5,97.5]).tolist())
    write_json(out/'summary.json',summary)
    print({k:round(v['accuracy']['mean'],2) for k,v in summary['incumbent'].items()})


if __name__=='__main__':
    main()
