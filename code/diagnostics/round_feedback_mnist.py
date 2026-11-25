"""Evaluate paired aggregation directions at the cadence of incoming client cohorts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.paired_feedback_control import ACTIONS, Panel, directions, proposal
from diagnostics.run_centered_control import schedule_id
from diagnostics.run_paired_feedback_control import paired_stats, setup
from diagnostics.stage_a_world import H
from utils.experiment import write_json

# Only development data select the private score. Evaluation freezes that choice.
COSTS={'ce':(0.,1.),'hybrid':(1.,.15),'error':(1.,0.)}
MIXES=(0.,.5,1.)


def report_score(panel, reports, kind):
    weights=COSTS[kind]
    return np.average(reports,axis=0,weights=panel.sizes.numpy()) @ weights


def round_options(w, start, i, scale, panel):
    w.w=start
    delta=directions(w,i)
    models=[proposal(w,start,delta,ACTIONS.index((b,scale))) for b in MIXES]
    stats=[panel.reports(m) for m in models]
    return models,stats


def current_direction(w, start, index, scale, panel, kind, default):
    models,reports=round_options(w,start,index,scale,panel)
    costs=np.array([report_score(panel,r,kind) for r in reports])
    a=int(np.argmin(costs))
    if costs[default]<=costs[a]+1e-12:
        a=default
    return models,a,costs


def run_seed(task):
    seed,split,method,kind,schedule=task
    w=setup('switch',seed,split)
    digest=schedule_id(w)
    trace=[]
    for t,arm in enumerate(schedule):
        mix,scale=ACTIONS[arm]
        panel=Panel.from_world(w,t)
        selected=MIXES.index(mix)
        for i in range(t*H,(t+1)*H):
            models,a,costs=current_direction(w,w.w,i,scale,panel,kind,MIXES.index(mix))
            if method=='round' or (method=='macro' and i%H==0):
                selected=a
            w.w=models[selected]
            trace.append(dict(round=i+1,mix=MIXES[selected],step=scale,private_cost=costs.tolist()))
    return dict(seed=seed,method=method,hash=digest,trace=trace,uploads=480,
                metrics=w.metrics('test' if split=='held' else 'validation'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('dev','held'),required=True)
    p.add_argument('--jobs',type=int,default=3)
    args=p.parse_args()
    schedule=json.loads(args.policy.read_text())['schedule']
    out=args.output
    if args.phase=='dev':
        out.mkdir(parents=True,exist_ok=False)
        bank=range(15300,15316)
        methods={'schedule':('schedule','hybrid'),
                 **{m+'_'+k:(m,k) for m in ('round','macro') for k in COSTS}}
        write_json(out/'protocol.json',dict(tune=list(bank),held=list(range(15400,15432)),
            schedule=schedule,primary='global_acc',methods=methods,checks='paired 95% lower bound above zero vs schedule and macro'))
    else:
        bank=range(15400,15432)
        methods=json.loads((out/'locked.json').read_text())
    with ProcessPoolExecutor(args.jobs) as pool:
        data={name:list(pool.map(run_seed,[(s,'calibrate' if args.phase=='dev' else 'held',m,k,schedule)
                                         for s in bank])) for name,(m,k) in methods.items()}
    stats=paired_stats(data,'schedule')
    write_json(out/(args.phase+'.json'),data)
    write_json(out/(args.phase+'_summary.json'),stats)
    if args.phase=='dev':
        selected={m:max((k for k in methods if k.startswith(m)),key=lambda k:stats[k]['global_acc']['mean']) for m in ('round','macro')}
        write_json(out/'locked.json',{'schedule':methods['schedule'],**{m:methods[k] for m,k in selected.items()}})
    else:
        write_json(out/'vs_macro.json',paired_stats(data,'macro'))
    print(args.phase,{k:round(v['global_acc']['mean'],3) for k,v in stats.items()},flush=True)


if __name__=='__main__':
    main()
