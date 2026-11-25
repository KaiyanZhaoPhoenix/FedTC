"""Validate exact continuations, ranking errors and branch accounting for the three-control study."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from diagnostics.run_three_controls_mnist import (
    Experiment, HELD, PILOT, TAUS, budget_path, budget_pick, fit_models, key, world_for,
)
from diagnostics.run_terminal_policy_mnist import advantages, terminal_cost
from utils.experiment import write_json


def read(path):
    return json.loads(path.read_text())


def tau_audit(root, seed, model, teacher):
    world = world_for(root, root/'tau_audit'/str(seed), 'rotating', seed)
    exp = Experiment(world, 'tau')
    names = ['short', 'long', 'no_direction', 'time']
    endpoints = {}
    paired = []
    paths = {}

    def follow(p, previous, i, methods, history):
        n = exp.node(p, i, previous)
        if i % 10 == 5:
            action = int(advantages(teacher,n['x']).argmax())
            selected = {m:action for m in methods}
        else:
            selected = {}
            for m in methods:
                if m == 'time':
                    selected[m] = model['time'][i//5]
                else:
                    x = np.array(n['x'])[:,:6] if m=='no_direction' else n['x']
                    selected[m] = int(advantages(model[m],x).argmax())
        ends = {}
        if 'long' in methods and i % 10 == 0:
            loss5, loss10 = [], []
            for a in range(3):
                q = exp.end(n,a)
                ends[a] = q
                loss5.append(exp.loss(q))
                successor = exp.node(q,i+5,q-p)
                choice = int(advantages(teacher, successor['x']).argmax())
                loss10.append(exp.loss(exp.end(successor,choice)))
            actions = {m:int(advantages(model[m],n['x']).argmax()) for m in ('short','long')}
            paired.append(dict(index=i,loss5=loss5,loss10=loss10,choices=actions))
        for a in sorted(set(selected.values())):
            q = ends[a] if a in ends else exp.end(n,a)
            sharing = [m for m in methods if selected[m] == a]
            if i == 35:
                test = world.record_test(q)
                for m in sharing:
                    endpoints[m] = test
                    paths[m] = history+[a]
            else:
                follow(q,q-p,i+5,sharing,history+[a])

    follow(exp.start,torch.zeros_like(exp.start),0,names,[])
    out = dict(metrics=endpoints, paths=paths, visited_ranking=paired)
    write_json(root/'tau_audit'/str(seed)/'audit.json',out)
    return out


def budget_execution(root, seed, rule):
    world = world_for(root,root/'budget_execution'/str(seed),'rotating',seed)
    exp = Experiment(world,'tau')
    trace = {}
    upload = exp.upload
    steps = []

    def counted(p,index):
        steps.append(exp.steps[index])
        return upload(p,index)

    exp.upload = counted
    p = exp.start
    prev = torch.zeros_like(p)
    used = ()
    windows = []
    for j in range(8):
        n = exp.node(p, j*5, prev)
        k = key(used)+'|'+str(j)
        trace[k] = n['x'][2]
        spend = budget_pick({'nodes':{k:n['x'][2]}},used,j,'rl',rule)
        q = exp.end(n,0)
        gain = None
        if spend:
            used += (j,)
            trial = exp.end(n,2)
            gain = terminal_cost(world,q) - terminal_cost(world,trial)
            if gain > 0:
                q = trial
        windows.append(dict(index=5*j, spend=spend, verified_gain=gain,
                            retained_tau=TAUS[2] if gain is not None and gain>0 else TAUS[0]))
        prev, p = q-p,q
    result = world.record_test(p)
    assert len(steps)==48 and len(used)==2
    out = dict(slots=used,metrics=result,uploads=12*len(steps),sgd_steps=sum(steps),windows=windows,
               model_upload_bytes=12*len(steps)*p.numel()*p.element_size(),visited_features=trace)
    write_json(root/'budget_execution'/str(seed)/'audit.json',out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('tau','budget'))
    parser.add_argument('root',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    root=args.root.resolve()
    with ProcessPoolExecutor(args.workers) as pool:
        if args.phase=='tau':
            rows=[r for s in PILOT for r in read(root/'tau'/'rotating'/'pilot'/str(s)/'records.json')]
            teacher=fit_models(rows)['short']
            model=read(root/'models.json')['tau']
            jobs=[pool.submit(tau_audit,root,s,model,teacher) for s in HELD]
        else:
            rule=read(root/'budget_model.json')['rl']
            jobs=[pool.submit(budget_execution,root,s,rule) for s in HELD]
        outs=[j.result() for j in jobs]
    write_json(root/(args.phase+'_audit_summary.json'), outs)
