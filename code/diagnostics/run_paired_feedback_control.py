"""Run matched development and locked held-out tests for paired feedback control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.paired_feedback_control import (
    ACTIONS, DEPTH, DISCOUNT, Panel, RolloutScorer, advance, design, paired_labels, probe)
from diagnostics.run_centered_control import make_world, schedule_id
from diagnostics.stage_a_world import ACTIVE, H, WINDOWS, stream
from utils.experiment import write_json

BANKS={'reference':range(14200,14210),'fit':range(14100,14132),
       'tune':range(14300,14312),'held':range(14400,14424)}
PENALTIES=(16,256)
PRIMARY={'switch':'global_acc','longtail':'tail20'}


def setup(scenario, seed, phase):
    head='ce' if scenario=='switch' else 'cosine-balanced-local'
    return make_world(scenario,seed,phase,head)


def fixed_seed(task):
    scenario,seed=task
    w=setup(scenario,seed,'calibrate')
    initial=w.w
    curves=[]
    for a in range(len(ACTIONS)):
        w.w=initial
        for t in range(WINDOWS):
            advance(w,w.w,t,a)
        curves.append(w.metrics('validation'))
    return curves


def train_seed(task):
    scenario,seed,reference=task
    w=setup(scenario,seed,'train')
    panels=[Panel.from_world(w,t) for t in range(WINDOWS)]
    behavior=stream(seed,104)
    rows=[]
    for t,panel in enumerate(panels):
        start=w.w
        first,initial,base,signals=probe(w,start,t,panel)
        labels=paired_labels(w,start,t,first,initial,panels,reference)
        # Half of the training behavior stays near the reference policy.
        arm=reference if behavior.random()<.5 else int(behavior.integers(len(ACTIONS)))
        advance(w,start,t,arm,first)
        rows.append(dict(t=t,base=base.tolist(),signals=signals.tolist(),labels=labels.tolist(),a=arm))
    return rows


def direct_scores(signals):
    return signals[:,0]+.15*signals[:,1]


def choose(scores, reference):
    best=int(np.argmax(scores))
    return reference if scores[reference]>=scores[best]-1e-12 else best


def rollout(task):
    scenario,seed,split,method,ref,model=task
    w=setup(scenario,seed,'held' if split=='held' else 'calibrate')
    schedule=schedule_id(w)
    trace=[]
    for t in range(WINDOWS):
        panel=Panel.from_world(w,t)
        start=w.w
        first,initial,base,signals=probe(w,start,t,panel)
        scores=None
        if method=='fixed':
            a=ref
        elif method=='schedule':
            a=model[t]
        elif method in ('pilot','paired'):
            scores=direct_scores(signals)
            a=choose(scores,ref)
        elif method=='paired_extremes':
            a=model[t]
        else:
            scores=model.scores(design(base,signals,t))
            a=choose(scores,ref)
        proposed=int(a)
        deployed=advance(w,start,t,a,first)
        ends=[(a,panel.cost(panel.reports(deployed)))]
        if method.startswith('paired') and a!=ref:
            other=advance(w,start,t,ref,first)
            ends.append((ref,panel.cost(panel.reports(other))))
            if ends[1][1]<=ends[0][1]:
                a,deployed=ref,other
        w.w=deployed
        trace.append(dict(a=int(a),proposed=proposed,pilot=signals.tolist(),base=base.tolist(),
                          scores=None if scores is None else scores.tolist(),end_costs=ends))
    # Only once the final model is fixed do we access test examples.
    return dict(seed=seed,method=method,hash=schedule,trace=trace,
                metrics=w.metrics('test' if split=='held' else 'validation'),
                uploads=ACTIVE*sum(H+(H-1)*(len(r['end_costs'])-1) for r in trace),
                evaluator_model_scores=16*sum(1+len(ACTIONS)+len(r['end_costs']) for r in trace))


def paired_stats(out, ref):
    result={}
    n=len(out[ref])
    boot=np.random.default_rng(241).integers(n,size=(20000,n))
    for method,rows in out.items():
        assert [(r['seed'],r['hash']) for r in rows]==[(r['seed'],r['hash']) for r in out[ref]]
        result[method]={}
        for k in ('global_acc','tail20'):
            a=np.array([r['metrics'][k] for r in rows])*100
            b=np.array([r['metrics'][k] for r in out[ref]])*100
            d=a-b
            result[method][k]=dict(mean=float(a.mean()),difference=float(d.mean()),
                ci95=np.percentile(d[boot].mean(axis=1),[2.5,97.5]).tolist(),
                wins=int((d>0).sum()),ties=int((d==0).sum()))
        result[method]['uploads']=float(np.mean([r['uploads'] for r in rows]))
    return result


def prepare(pool,args):
    s=args.scenario
    p=args.output
    p.mkdir(parents=True,exist_ok=False)
    write_json(p/'protocol.json',dict(scenario=s,banks={k:list(v) for k,v in BANKS.items()},
               actions=ACTIONS,primary=PRIMARY[s],H=H,depth=DEPTH,gamma=DISCOUNT,
               penalties=PENALTIES,cost='sample-weighted private error + 0.15 clipped CE',
               held_criteria='positive paired 95% lower bound against expanded fixed; RL additionally vs immediate and time-only',
               paired_rule='compare two five-round endpoints on the same clients; first upload shared',
               simulator='Stage A; switch begins after round 20; validation reports honest'))
    curves=list(pool.map(fixed_seed,[(s,seed) for seed in BANKS['reference']]))
    scores=np.mean([[r[PRIMARY[s]] for r in row] for row in curves],axis=0)
    ref=int(np.argmax(scores))
    write_json(p/'reference.json',dict(scores=scores.tolist(),arm=ref,curves=curves))
    print('reference',s,ref,ACTIONS[ref],scores.round(4),flush=True)
    fit=list(pool.map(train_seed,[(s,seed,ref) for seed in BANKS['fit']]))
    write_json(p/'training.json',fit)
    print('paired training branches complete',flush=True)


def candidates(folder):
    ref=json.loads((folder/'reference.json').read_text())['arm']
    fit=json.loads((folder/'training.json').read_text())
    rows=[r for traj in fit for r in traj]
    x=[design(np.array(r['base']),np.array(r['signals']),r['t']) for r in rows]
    y=np.array([r['labels'] for r in rows])
    # A schedule uses exactly the same paired labels as the state policies.
    per_step=np.array([[r['labels'] for r in group] for group in fit]).mean(0)
    time=[choose(step[:,0],ref) for step in per_step]
    time_long=[choose(step[:,1],ref) for step in per_step]
    # A second verified proposal uses the strongest alternative judged on fit labels.
    alternatives=per_step[:,:,0].copy()
    alternatives[:,ref]=-np.inf
    other=alternatives.argmax(axis=1).tolist()
    methods={'old_fixed':('fixed',0,None),'fixed':('fixed',ref,None),'schedule_0':('schedule',ref,time),
             'schedule_9':('schedule',ref,time_long),'pilot':('pilot',ref,None),
             'paired':('paired',ref,None),'paired_schedule':('paired_extremes',ref,other)}
    for j,name in enumerate(('immediate','rl')):
        for penalty in PENALTIES:
            model=RolloutScorer.fit(x,y[:,:,j],penalty)
            methods[name+str(penalty)]=(name,ref,model)
    return methods


def evaluate(pool,args,held):
    s=args.scenario
    p=args.output
    split='held' if held else 'tune'
    bank=BANKS[split]
    methods=candidates(p)
    if held:
        lock=json.loads((p/'locked.json').read_text())
        methods={k:methods[v] for k,v in lock.items()}
    out={}
    for name,(m,ref,model) in methods.items():
        tasks=[(s,seed,split,m,ref,model) for seed in bank]
        out[name]=list(pool.map(rollout,tasks))
    stats=paired_stats(out,'fixed')
    write_json(p/(split+'.json'),out)
    write_json(p/(split+'_summary.json'),stats)
    print(split,s,{k:round(v[PRIMARY[s]]['mean'],3) for k,v in stats.items()},flush=True)
    if held:
        write_json(p/'attribution.json',{k:paired_stats(out,k) for k in ('immediate','schedule','pilot','paired')})
    else:
        choices={'old_fixed':['old_fixed'],'fixed':['fixed'],'schedule':['schedule_0','schedule_9'],
                 'pilot':['pilot'],'immediate':['immediate'+str(i) for i in PENALTIES],
                 'rl':['rl'+str(i) for i in PENALTIES],'paired':['paired','paired_schedule']}
        lock={k:max(v,key=lambda name:stats[name][PRIMARY[s]]['mean']) for k,v in choices.items()}
        write_json(p/'locked.json',lock)
        print('frozen',lock,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',choices=tuple(PRIMARY),required=True)
    p.add_argument('--phase',choices=('dev','held','all'),default='all')
    p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args()
    with ProcessPoolExecutor(args.jobs) as pool:
        if args.phase in ('dev','all'):
            prepare(pool,args)
            evaluate(pool,args,False)
        if args.phase in ('held','all'):
            evaluate(pool,args,True)


if __name__=='__main__':
    main()
