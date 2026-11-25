"""Matched clean-head, five-round action-branch, and controller experiments."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.baselines import aggregate
from algorithms.fedtc.forward import calibrate, spearman
from diagnostics.centered_control import centered, Feedback, next_action
from diagnostics.stage_a_world import H, WINDOWS, World

DEV = range(11100, 11106)
HELD = range(11200, 11212)
GRID = [(lam, tau) for lam in (0., .3) for tau in (1.05, 1.65, 2.50)]
HEADS = ['ce', 'balanced-global', 'balanced-local', 'cosine-balanced-local']
START = (.15, 1.65)
FEATURES = 'runs/stage_a_data/cifar100_lt_features.npz'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def make_world(scenario, seed, phase, head):
    torch.set_num_threads(1)
    w = World(scenario, seed, phase, 'dataset', FEATURES, balanced_head=head=='balanced-global',
              prototype_init=scenario=='longtail')
    if head in ('balanced-local', 'cosine-balanced-local'):
        counts = torch.stack([torch.bincount(w.y[ids], minlength=w.classes) for ids in w.train])
        w.training_log_prior = (counts.float() + 1).log()
    w.cosine_head = head in ('cosine-ce', 'cosine-balanced-local')
    return w


def window(w, t, method, action, inject, audit=True):
    before = w.val_loss
    ranks, outside, round_logs = [], [], []
    for i in range(t*H, (t+1)*H):
        uploads, utilities, mask = w.uploads(i, inject and i>=20)
        cohort = w.cohorts[i]
        if method in ('centered', 'legacy'):
            r = (centered if method=='centered' else calibrate)(uploads, utilities, action)
            delta = r.delta
            if audit:
                d = r.deviations
                safe = r.weights > 0
                u = utilities[safe]
                if len(u)>1 and u.max()>u.min() and d[safe].max()>d[safe].min():
                    ranks.append(spearman(u, -d[safe]))
                outside.append(float((d>2).float().mean()))
                round_logs.append(dict(round=i+1, bad_d=d[mask].tolist(),
                    bad_inner=int((d[mask]<=1).sum()), bad_zero=int((~safe[mask]).sum()),
                    bad_mass=float(r.weights[mask].sum()), good_zero=int((~safe[~mask]).sum()),
                    tail_mass=float(r.weights.numpy() @ w.tail_frac[cohort])))
        else:
            delta, _ = aggregate(method, uploads, [len(w.train[c]) for c in cohort], 4, 1)
        w.w += delta.reshape_as(w.w)
        if w.cosine_head:
            w.w[:-1] = F.normalize(w.w[:-1], dim=0)
            w.w[-1].zero_()
    w.val_loss = w.losses()
    reports = (before-w.val_loss)[w.evals[t]]
    ci = w.estimator.estimate(reports)
    fb = Feedback(ci.low, ci.high, ci.trimmed_low, ci.trimmed_high, float(np.median(reports)),
                  float((reports<0).mean()), float(np.mean(outside)) if outside else 0.,
                  float(np.mean(ranks)) if ranks else None)
    return fb, round_logs


def schedule_id(w):
    h = hashlib.sha256()
    for part in (w.w, *w.train, w.cohorts, *w.indices, w.evals):
        h.update(np.asarray(part).tobytes())
    return h.hexdigest()


def trajectory(task):
    scenario, seed, phase, head, method, action, inject = task
    w = make_world(scenario, seed, phase, head)
    selection = 'validation' if phase=='calibrate' else 'test'
    curve = [w.metrics(selection)]
    rows = []
    digest = schedule_id(w)
    previous = None
    reason = 'initial'
    for t in range(WINDOWS):
        live = method=='feedback'
        fb, logs = window(w, t, 'centered' if live else method, action, inject,
                          audit=live or phase=='held')
        curve.append(w.metrics(selection))
        rows.append(dict(end=(t+1)*H, action=action, reason=reason,
                         feedback=asdict(fb), rounds=logs))
        if live:
            action, reason = next_action(action, fb, previous)
        previous = fb
    return dict(seed=seed, head=head, method=method, start=rows[0]['action'],
                schedule=digest, curve=curve, windows=rows)


def branch(task):
    scenario, seed, head = task
    w = make_world(scenario, seed, 'calibrate', head)
    rows = []
    for t in range(WINDOWS):
        if t in (0,4,6):
            weights, loss = w.w.clone(), w.val_loss.copy()
            choices = [('centered', a) for a in GRID] + [('legacy', (0.,1.15)), ('legacy',(0.,1.65))]
            for method, action in choices:
                w.w, w.val_loss = weights.clone(), loss.copy()
                fb, logs = window(w, t, method, action, True)
                rows.append(dict(seed=seed, start=t*H, method=method, action=action,
                                 validation=w.metrics('validation'), feedback=asdict(fb), rounds=logs))
            w.w, w.val_loss = weights, loss
        window(w, t, 'legacy', (0.,1.35), True, audit=False)
    return rows


def mean_final(rows, metric):
    return float(np.mean([r['curve'][-1][metric] for r in rows]))


def compare(results, metric, reference):
    index = np.random.default_rng(1543).integers(0,len(HELD),(20000,len(HELD)))
    base = np.array([r['curve'][-1][metric] for r in results[reference]])
    out = {}
    for name, group in results.items():
        assert all((x['seed'],x['schedule'])==(y['seed'],y['schedule'])
                   for x,y in zip(group,results[reference]))
        vals = np.array([r['curve'][-1][metric] for r in group])
        diff = (vals-base)*100
        out[name] = dict(mean_pct=float(vals.mean()*100), paired_pp=float(diff.mean()),
                         ci95=np.percentile(diff[index].mean(axis=1),[2.5,97.5]).tolist())
    return out


def run(args):
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json', dict(dev=list(DEV), held=list(HELD), grid=GRID, heads=HEADS,
         tail_gate_pct=25, choice='first clean head with mean final validation tail >=25%; else max',
         forks=[0,20,30], static_selection='mean final validation primary',
         primary={'switch':'global_acc','longtail':'tail20'}, initial_action=START,
         switch_after_round=20, control='next_action in diagnostics/centered_control.py',
         baseline_bad_bound=4, krum_count=1, branch_screen_pp=2))
    with ProcessPoolExecutor(args.jobs) as pool:
        def trajectories(tasks):
            return list(pool.map(trajectory,tasks))
        def clean(seeds, phase):
            return {h:trajectories([('longtail',s,phase,h,'FedAvg',None,False) for s in seeds])
                    for h in HEADS}
        development = clean(DEV,'calibrate')
        scores = {h:mean_final(group,'tail20') for h,group in development.items()}
        chosen = next((h for h in HEADS if scores[h]>=.25),max(scores,key=scores.get))
        save(out/'clean_development.json',dict(scores=scores,chosen=chosen,results=development))
        print('Clean development',scores,chosen,flush=True)
        held_clean = clean(HELD,'held')
        save(out/'clean_held.json',held_clean)
        clean_summary = {m:compare(held_clean,m,'ce') for m in ('global_acc','tail20')}
        save(out/'clean_summary.json',clean_summary)
        for scenario in ('switch','longtail'):
            if scenario=='longtail' and scores[chosen]<.25:
                continue
            head = chosen if scenario=='longtail' else 'ce'
            primary = 'global_acc' if scenario=='switch' else 'tail20'
            branches = [r for group in pool.map(branch,[(scenario,s,head) for s in DEV]) for r in group]
            save(out/(scenario+'_forks.json'),branches)
            tuning = [trajectories([(scenario,s,'calibrate',head,'centered',a,True) for s in DEV]) for a in GRID]
            scores_fixed = [mean_final(group,primary) for group in tuning]
            arm = GRID[int(np.argmax(scores_fixed))]
            save(out/(scenario+'_static_selection.json'),dict(scores=scores_fixed,action=arm,curves=[
                [r['curve'] for r in group] for group in tuning]))
            print(scenario,'selected static',arm,flush=True)
            methods = [('static','centered',arm),('feedback','feedback',START),
                       ('same_start','centered',START),('legacy','legacy',(0.,1.35)),
                       ('FedAvg','FedAvg',None),('Krum','MultiKrum',None),('TrimmedMean','TrimmedMean',None)]
            results = {name:trajectories([(scenario,s,'held',head,m,a,True) for s in HELD]) for name,m,a in methods}
            save(out/(scenario+'_held.json'),results)
            summary = {k:compare(results,k,'static') for k in ('global_acc','tail20')}
            save(out/(scenario+'_summary.json'),summary)
            print(scenario,primary,summary[primary],flush=True)


def audit(folder, jobs):
    """Post-hoc clean loss ablation and mechanical checks. No policy refitting."""
    def read(name):
        return json.loads((folder/(name+'.json')).read_text())

    clean = read('clean_held')
    with ProcessPoolExecutor(jobs) as pool:
        tasks = [('longtail',s,'held','cosine-ce','FedAvg',None,False) for s in HELD]
        clean['cosine-ce'] = list(pool.map(trajectory,tasks))
    save(folder/'clean_cosine_ce.json',clean['cosine-ce'])
    stats = {'clean': {m:compare(clean,m,'cosine-balanced-local') for m in ('global_acc','tail20')}}
    for scenario, metric in [('switch','global_acc'),('longtail','tail20')]:
        if not (folder/(scenario+'_held.json')).exists():
            continue
        held = read(scenario+'_held')
        forks = read(scenario+'_forks')
        objective = {}
        reference = read(scenario+'_static_selection')['action']
        for t in (0,20,30):
            rows = [r for r in forks if r['start']==t and r['method']=='centered']
            values = {str(a):float(np.mean([r['validation'][metric] for r in rows
                                             if tuple(r['action'])==a])*100) for a in GRID}
            hindsight = []
            for seed in DEV:
                trial = [r for r in rows if r['seed']==seed]
                fixed = next(r for r in trial if r['action']==reference)
                best = max(r['validation'][metric] for r in trial)
                hindsight.append(100*(best-fixed['validation'][metric]))
            objective[t] = dict(validation_pct=values, span_pp=max(values.values())-min(values.values()),
                                hindsight_gain_over_selected_static_pp=hindsight)
        phases = {}
        for label, section in [('clean',slice(0,4)),('switched',slice(4,8))]:
            windows = [w for r in held['feedback'] for w in r['windows'][section]]
            feedback = [w['feedback'] for w in windows]
            rnd = [q for w in windows for q in w['rounds']]
            ds = [d for q in rnd for d in q['bad_d']]
            phases[label] = dict(windows=len(windows), lower_below_zero=sum(f['low']<0 for f in feedback),
                negative_lower_positive_median=sum(f['low']<0 and f['median']>0 for f in feedback),
                rho_fuse=sum(f['safe_rho'] is not None and f['safe_rho']<-.2 for f in feedback),
                mean_rho=float(np.mean([f['safe_rho'] for f in feedback if f['safe_rho'] is not None])),
                mean_action=np.mean([w['action'] for w in windows],axis=0).tolist(),
                bad_uploads=len(ds),bad_inner=sum(q['bad_inner'] for q in rnd),
                bad_zero=sum(q['bad_zero'] for q in rnd),bad_weight_pct=100*float(np.mean([q['bad_mass'] for q in rnd])))
        stats[scenario] = dict(forks=objective,feedback=phases,
            feedback_vs_identical_start=compare({'feedback':held['feedback'],'same_start':held['same_start']},
                                                   metric,'same_start')['feedback'])
    save(folder/'mechanism_audit.json',stats)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--audit-only', action='store_true', help='Read existing main run, add independent cosine CE control and mechanism summaries')
    args = p.parse_args()
    if not args.audit_only:
        run(args)
    audit(args.output, args.jobs)
