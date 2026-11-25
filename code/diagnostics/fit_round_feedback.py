"""Paired round-level policy improvement conditional on a locked server-step schedule."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from algorithms.baselines import aggregate
from diagnostics.paired_feedback_control import ACTIONS, Panel, RolloutScorer
from diagnostics.round_feedback_mnist import COSTS, MIXES, report_score
from diagnostics.validated_directions import options, panels as feedback_panels
from diagnostics.run_centered_control import schedule_id
from diagnostics.run_paired_feedback_control import paired_stats, setup
from diagnostics.stage_a_world import H, WINDOWS, World, stream
from utils.experiment import write_json

DEPTH=10
GAMMA=.9
R=H*WINDOWS
BANK={'fit':range(15100,15136),'tune':range(15500,15516),'held':range(15600,15632)}


def context(panel, before, reports, i, scale):
    gains=np.array([panel.summarize(before-r) for r in reports])
    base=panel.summarize(before)
    ident=np.eye(len(MIXES))
    t=i/(R-1)
    return np.column_stack([ident,ident*t,ident*base[0],ident*np.log2(scale),
                            gains,gains*t,gains*base[0],gains*np.log2(scale)])


def follow(w,model,i,arm):
    w.w=model
    mix,scale=ACTIONS[arm]
    if mix==1:
        updates,_,_=w.uploads(i,i>=20)
        delta,_=aggregate('FedAvg',updates,[len(w.train[c]) for c in w.cohorts[i]],4,1)
        return model+scale*delta.reshape_as(model)
    from diagnostics.paired_feedback_control import directions,proposal
    return proposal(w,model,directions(w,i),arm)


def training(task):
    seed,schedule,family=task
    torch.set_num_threads(1)
    pool=stream(seed,713).choice(48,24,replace=False).tolist()
    w=World('switch',seed,'train','dataset',None,client_ids=pool)
    panels=[feedback_panels(w,t,family) for t in range(WINDOWS)]
    # Independent training label panel, all 24 clients' private validation summaries.
    sizes=torch.bincount(w.val_groups,minlength=24)
    target_panel=Panel(w.vx,w.vy,w.val_groups,sizes)
    behavior=stream(seed,715)
    xs=[]; ys=[]
    for i in range(R):
        arm=schedule[i//H]
        mix,scale=ACTIONS[arm]
        start=w.w
        proposing,p=panels[i//H]
        before=p.reports(start)
        models,reports=options(w,start,i,scale,proposing,p)
        x=context(p,before,reports,i,scale)
        targets=[]
        for model in models:
            first=-target_panel.cost(target_panel.reports(model))
            value=first
            last=first
            for j in range(i+1,min(R,i+DEPTH)):
                model=follow(w,model,j,schedule[j//H])
                utility=-target_panel.cost(target_panel.reports(model))
                value+=GAMMA**(j-i)*(utility-last)
                last=utility
            targets.append((first,value))
        selected=MIXES.index(mix) if behavior.random()<.5 else int(behavior.integers(3))
        w.w=models[selected]
        xs.append(x.tolist())
        ys.append(targets)
    return dict(x=xs,y=ys,clients=pool,seed=seed)


def features_rule(spec,panel,reports,before,i,scale):
    if 'schedule_mix' in spec:
        return spec['schedule_mix'][i],None
    scores=-np.array([report_score(panel,r,spec['cost']) for r in reports])
    if spec.get('model') is not None:
        m=spec['model']
        model=RolloutScorer(np.asarray(m['scale']),np.asarray(m['weight']))
        prediction=model.scores(context(panel,before,reports,i,scale))
        scores=prediction*(1-spec['pilot_fraction'])+spec['pilot_fraction']*(scores-scores.mean())
    best=int(scores.argmax())
    if scores[2]>=scores[best]-1e-12:
        best=2
    return best,scores.tolist()


def evaluate(task):
    seed,split,schedule,name,spec,family=task
    if split=='mixed':
        torch.set_num_threads(1)
        pool=stream(seed,719).choice(np.arange(48,100),24,replace=False).tolist()
        w=World('switch',seed,'held','dataset',None,client_ids=pool)
    else:
        w=setup('switch',seed,'held' if split=='held' else 'calibrate')
    identity=schedule_id(w)
    trace=[]
    for i in range(R):
        if i%H==0:
            proposing,panel=feedback_panels(w,i//H,family)
        scale=ACTIONS[schedule[i//H]][1]
        before=panel.reports(w.w)
        models,reports=options(w,w.w,i,scale,proposing,panel)
        a,score=features_rule(spec,panel,reports,before,i,scale)
        w.w=models[a]
        trace.append(dict(round=i+1,step=scale,mix=MIXES[a],scores=score))
    return dict(seed=seed,method=name,hash=identity,trace=trace,uploads=480,
                metrics=w.metrics('test' if split in ('held','mixed') else 'validation'))


def methods(data, schedule):
    x=np.array([r['x'] for r in data]); y=np.array([r['y'] for r in data])
    xs=x.reshape(-1,*x.shape[-2:]); ys=y.reshape(-1,3,2)
    rules={k:dict(cost=k) for k in COSTS}
    rules['original_schedule']=dict(schedule_mix=[MIXES.index(ACTIONS[a][0]) for a in schedule for _ in range(H)])
    for a in range(3):
        for b in range(3):
            rules[f'time_{a}_{b}']=dict(schedule_mix=[a]*20+[b]*20)
    for j,name in enumerate(('immediate','rl')):
        avg=y[:,:,:,j].mean(0).reshape(WINDOWS,H,3).mean(1)
        rules['time_'+name]=dict(schedule_mix=np.repeat(avg.argmax(axis=1),H).tolist())
        for ridge in (16,256):
            model=RolloutScorer.fit(xs,ys[:,:,j],ridge)
            data_model=dict(scale=model.scale.tolist(),weight=model.weight.tolist())
            for frac in (0.,.5):
                rules[f'{name}_{ridge}_{frac}']=dict(cost='hybrid',pilot_fraction=frac,model=data_model)
    return rules


def run(args):
    p=args.output
    if args.phase=='deploy':
        pack=json.loads((p/'policy.json').read_text())
        row=evaluate((args.seed,'held',pack['schedule'],args.method,pack['rules'][args.method],pack.get('family','aggregate')))
        print(json.dumps(row,indent=2))
        return
    if args.phase=='confirm':
        pack=json.loads((p/'policy.json').read_text())
        rules=dict(pack['rules'])
        family=pack.get('family','aggregate')
        seeds=list(range(15700,15732))
        write_json(p/'confirm_protocol.json',dict(seeds=seeds,ref='schedule',frozen=pack['selection'],
            fixed_pool='two existing disjoint held pools',mixed_pool='24 sampled from clients 48..99'))
        with ProcessPoolExecutor(args.jobs) as pool:
            for split in ('held','mixed'):
                groups={name:list(pool.map(evaluate,[(s,split,pack['schedule'],name,rule,family)
                                             for s in seeds])) for name,rule in rules.items()}
                write_json(p/('confirm_'+split+'.json'),groups)
                write_json(p/('confirm_'+split+'_summary.json'),
                           {ref:paired_stats(groups,ref) for ref in ('schedule','direct','immediate')})
        return
    dev=args.phase=='dev'
    if dev:
        family=args.direction_family
        p.mkdir(parents=True,exist_ok=False)
        sched=json.loads(args.parent_policy.read_text())['schedule']
        write_json(p/'protocol.json',dict(family=family,banks={k:list(v) for k,v in BANK.items()},schedule=sched,
            gamma=GAMMA,continuation_rounds=DEPTH-1,primary='global_acc',fit_client_ids='sample 24 from 0..47',
            group='each seed is paired; held clients 48..95; no further fitting after lock',
            success='positive lower paired 95% bounds vs time-only, direct and equally fitted gamma=0'))
    else:
        pack=json.loads((p/'policy.json').read_text())
        sched=pack['schedule']
        family=pack.get('family','aggregate')
    with ProcessPoolExecutor(args.jobs) as pool:
        if dev:
            data=list(pool.map(training,[(s,sched,family) for s in BANK['fit']]))
            write_json(p/'training.json',data)
            options=methods(data,sched)
            print('measured paired windows complete',flush=True)
        else:
            options=dict(pack['rules'])
            options.setdefault('original',dict(schedule_mix=[MIXES.index(ACTIONS[a][0]) for a in sched for _ in range(H)]))
        split='tune' if dev else 'held'
        rows={k:list(pool.map(evaluate,[(s,split,sched,k,v,family) for s in BANK[split]])) for k,v in options.items()}
    reference='original_schedule' if dev else 'schedule'
    stats=paired_stats(rows,reference)
    write_json(p/(split+'.json'),rows)
    write_json(p/(split+'_summary.json'),stats)
    if dev:
        groups={'schedule':[k for k in options if k.startswith('time_')]+['original_schedule'],
                'direct':list(COSTS),'immediate':[k for k in options if k.startswith('immediate_')],
                'rl':[k for k in options if k.startswith('rl_')]}
        lock={g:max(keys,key=lambda k:stats[k]['global_acc']['mean']) for g,keys in groups.items()}
        lock['original']='original_schedule'
        write_json(p/'policy.json',dict(family=family,schedule=sched,selection=lock,
            rules={k:options[v] for k,v in lock.items()},actions=MIXES,discount=GAMMA,depth=DEPTH))
        print('chosen',[(g,k,round(stats[k]['global_acc']['mean'],2)) for g,k in lock.items()],flush=True)
    else:
        write_json(p/'attribution.json',{k:paired_stats(rows,k) for k in ('direct','immediate')})
        print('held',[(k,round(v['global_acc']['mean'],2)) for k,v in stats.items()],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--phase',choices=('dev','held','deploy','confirm'),required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--parent-policy',type=Path)
    a.add_argument('--direction-family',choices=('aggregate','validation'),default='aggregate')
    a.add_argument('--method',choices=('schedule','direct','immediate','rl'),default='rl')
    a.add_argument('--seed',type=int)
    a.add_argument('--jobs',type=int,default=4)
    args=a.parse_args()
    if args.phase=='dev' and not args.parent_policy: a.error('development requires --parent-policy')
    if args.phase=='deploy' and args.seed is None: a.error('deployment requires --seed')
    run(args)
