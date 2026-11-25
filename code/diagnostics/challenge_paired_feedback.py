"""Challenge the frozen controller with tuned step schedules and equal-cost branch selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.paired_feedback_control import ACTIONS, Panel, RolloutScorer, advance, design, probe
from diagnostics.run_centered_control import schedule_id
from diagnostics.run_paired_feedback_control import BANKS, PRIMARY, candidates, direct_scores, setup
from diagnostics.run_paired_feedback_control import paired_stats
from diagnostics.stage_a_world import ACTIVE, H, WINDOWS
from utils.experiment import write_json


def measure_schedule(task):
    scenario,seed,schedules=task
    w=setup(scenario,seed,'calibrate')
    root=w.w
    cache={():root}
    scores=[]
    for schedule in schedules:
        for t,arm in enumerate(schedule):
            prefix=tuple(schedule[:t+1])
            if prefix not in cache:
                cache[prefix]=advance(w,cache[prefix[:-1]],t,arm)
        w.w=cache[tuple(schedule)]
        scores.append(w.metrics('validation')[PRIMARY[scenario]])
    return scores


def export(model):
    return dict(scale=model.scale.tolist(),weight=model.weight.tolist())


def lock(pool,p,s):
    original=json.loads((p/'locked.json').read_text())
    trained=candidates(p)
    gamma0=trained[original['immediate']][2]
    gamma9=trained[original['rl']][2]
    tune=json.loads((p/'tune.json').read_text())[original['rl']]
    mode=[int(np.bincount([r['trace'][t]['a'] for r in tune],minlength=len(ACTIONS)).argmax())
          for t in range(WINDOWS)]
    schedules=[trained[original['schedule']][2],mode]
    for mix in (0.,.5,1.):
        for early,late in [(4.,1.),(4.,2.),(8.,1.),(8.,2.)]:
            for switch in (2,4):
                schedules.append([ACTIONS.index((mix,early if t<switch else late)) for t in range(WINDOWS)])
    extra_seeds=range(14500,14510)
    vals=list(pool.map(measure_schedule,[(s,i,schedules) for i in extra_seeds]))
    means=np.mean(vals,axis=0)
    choice=int(np.argmax(means))
    frozen=dict(scenario=s,grid=ACTIONS,reference=trained['fixed'][1],schedule=schedules[choice],
        immediate=export(gamma0),rl=export(gamma9),held=list(BANKS['held']),
        method='three-window discounted rollout under a fixed continuation, paired ridge',
        primary=PRIMARY[s],challenge_arms=['fixed','schedule','rl','guard_schedule','guard_pilot','guard_immediate','guard_rl'],
        guard='compare strongest non-scheduled proposal and schedule on the same five-round sample; ties keep schedule')
    write_json(p/'schedule_screen.json',dict(seeds=list(extra_seeds),schedules=schedules,means=means.tolist()))
    write_json(p/'policy.json',frozen)
    print(s,'schedule locked',schedules[choice],round(100*means[choice],2),flush=True)


def deploy(task):
    pack,seed,method=task
    s=pack['scenario']
    w=setup(s,seed,'held')
    schedule=schedule_id(w)
    model=None
    name=method.removeprefix('guard_')
    if name in ('rl','immediate'):
        model=RolloutScorer(**{k:np.array(v) for k,v in pack[name].items()})
    audit=[]
    for t,planned in enumerate(pack['schedule']):
        start=w.w
        panel=Panel.from_world(w,t)
        first,initial,base,signals=probe(w,start,t,panel)
        scores=direct_scores(signals) if model is None else model.scores(design(base,signals,t))
        branch=method.startswith('guard_')
        if method=='fixed':
            a=pack['reference']
        elif method=='schedule':
            a=planned
        elif method=='guard_schedule':
            a=pack['reference']
            if a==planned:
                # Same schedule-only challenger across all seeds and histories.
                a=(planned+3)%len(ACTIONS)
        else:
            if branch:
                scores[planned]=-np.inf
            a=int(np.argmax(scores))
        proposed=a
        end=advance(w,start,t,a,first)
        candidate_cost=panel.cost(panel.reports(end))
        baseline_cost=None
        if branch:
            other=advance(w,start,t,planned,first)
            baseline_cost=panel.cost(panel.reports(other))
            if baseline_cost<=candidate_cost:
                a,end=planned,other
        w.w=end
        audit.append(dict(planned=planned,proposed=proposed,executed=a,initial_cost=panel.cost(initial),
                          candidate_cost=candidate_cost,schedule_cost=baseline_cost))
    return dict(method=method,seed=seed,hash=schedule,trace=audit,metrics=w.metrics('test'),
                uploads=ACTIVE*WINDOWS*(2*H-1 if method.startswith('guard_') else H))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--phase',choices=('lock','held','deploy'),required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--method',choices=('rl','immediate','schedule','guard_rl','guard_immediate','guard_schedule','guard_pilot'),default='rl')
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    p=args.folder
    if args.phase=='deploy':
        if args.seed is None:
            parser.error('--seed is required for deployment')
        row=deploy((json.loads((p/'policy.json').read_text()),args.seed,args.method))
        print(json.dumps(row,indent=2))
        return
    with ProcessPoolExecutor(args.jobs) as pool:
        if args.phase=='lock':
            lock(pool,p,json.loads((p/'protocol.json').read_text())['scenario'])
            return
        pack=json.loads((p/'policy.json').read_text())
        rows={}
        for m in pack['challenge_arms']:
            rows[m]=list(pool.map(deploy,[(pack,s,m) for s in pack['held']]))
        stats={ref:paired_stats(rows,ref) for ref in ('fixed','schedule','guard_schedule','guard_immediate','guard_pilot')}
        write_json(p/'challenge_held.json',rows)
        write_json(p/'challenge_summary.json',stats)
        print(pack['scenario'],{m:round(v[pack['primary']]['mean'],3) for m,v in stats['fixed'].items()},flush=True)


if __name__=='__main__':
    main()
