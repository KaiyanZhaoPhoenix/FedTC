"""Matched, held-out MNIST experiments for filter strength, anchor mixing and branch allocation."""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from functools import lru_cache
import argparse
import hashlib
import json
import numpy as np
import torch

from algorithms.fedtc.forward import calibrate
from diagnostics.run_shadow_rl_mnist import ShadowMNIST
from diagnostics.run_terminal_policy_mnist import fit_head, advantages, terminal_cost
from main import parser as training_parser, prepare
from utils.experiment import set_vector, vector, write_json, fingerprint

H = 5
WINDOWS = 8
TAUS = [1.15, 1.25, 1.55]
ENVS = ['rotating', 'uniform', 'flip']
PILOT = list(range(101, 105))
TRAIN = list(range(105, 113))
HELD = list(range(121, 129))


def vec_hash(x):
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def cosine(x, y):
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0, eps=1e-9))


def world_for(root, folder, env, seed):
    held = seed in HELD
    offset = 24 + 24*(seed % 2) if held else 0
    args = training_parser().parse_args([
        '--algorithm','FedTCFixed','--dataset','mnist','--device','cpu','--num_clients','24',
        '--num_sel_clients','12','--aggre_num','12','--training-cohort','sampled','--batch_size','16',
        '--global_rounds','40','--macro-length','5','--local_epochs','1','--validation-fraction','.25',
        '--split-seed',str(3031 + offset),'--seed',str(seed),'--output',str(folder),
        '--data-root',str(root/'data'/str(offset)),
        '--attack','delta_sign_flip' if env=='flip' else 'none','--ad_ratio','.2' if env=='flip' else '0'])
    w = ShadowMNIST(prepare(args))
    if env != 'uniform':
        def mode(c):
            labels = torch.stack([c._full_train_data[i][1] for i in c.train_indices])
            return int(torch.bincount(labels).argmax())
        ids = sorted(range(24),key=lambda i:mode(w.clients[i]))
        rng = np.random.default_rng(np.random.SeedSequence([seed,960]))
        blocks = np.concatenate([rng.permutation(2) for _ in range(4)])
        w.schedule = [rng.permutation(ids[12*b:12*b+12]).tolist() for b in np.repeat(blocks,5)]
    return w


class Experiment:
    def __init__(self, world, family):
        self.w, self.family = world, family
        self.start = vector(world.global_model)
        self.steps = [sum(len(world.clients[i].train_indices)//16 for i in ids) for ids in world.schedule]

    def upload(self, p, index):
        w = self.w
        set_vector(w.global_model, p)
        clients = [w.clients[i] for i in w.schedule[index]]
        w.send_models()
        w.train_clients(clients,index,p)
        return torch.stack([vector(c.model)-p for c in clients])

    def deltas(self, uploads, actions):
        zeros = uploads.new_zeros(len(uploads))
        if self.family == 'alpha':
            base = calibrate(uploads,zeros,(0.,1.15))
            return [torch.lerp(base.delta,base.anchor,a/2) for a in actions]
        return [calibrate(uploads,zeros,(0.,TAUS[a])).delta for a in actions]

    def node(self, p, index, previous):
        w = self.w
        uploads = self.upload(p,index)
        ds = self.deltas(uploads,range(3))
        losses = np.array([w.private_losses(p+d) for d in ds])
        diff = losses[0]-losses
        mask = np.isin(np.arange(24),w.schedule[min(index+H,39)])
        context = [index/40,float(losses[0]@w.sample_weights),
                   np.mean(np.isin(w.schedule[index],w.schedule[min(index+H,39)]))]
        features = []
        for d, gain in zip(ds,diff):
            difference = d-ds[0]
            features.append([*context, gain@w.sample_weights,
                             np.average(gain[mask],weights=w.sample_weights[mask]),
                             np.average(gain[~mask],weights=w.sample_weights[~mask]),
                             float(difference.norm()/(ds[0].norm()+1e-9)),
                             1-cosine(d,ds[0]),cosine(d,previous),cosine(difference,previous)])
        return dict(p=p, first=[p+d for d in ds], x=features,
                    upload=vec_hash(uploads), index=index)

    def end(self, node, action):
        p = node['first'][action]
        for i in range(node['index']+1,node['index']+5):
            p = p + self.deltas(self.upload(p,i),[action])[0]
        return p

    def loss(self,p):
        return float(self.w.private_losses(p)@self.w.sample_weights)


def best(head, features):
    return int(advantages(head,features).argmax())


def make_training(exp, seed, teacher):
    rows = []
    rng = np.random.default_rng(seed+8080)
    p = exp.start
    previous = torch.zeros_like(p)
    for index in range(0,40,5):
        n = exp.node(p,index,previous)
        ends = [exp.end(n,a) for a in range(3)]
        short = [exp.loss(q) for q in ends]
        long = list(short)
        if teacher and index < 35:
            for a,q in enumerate(ends):
                nxt = exp.node(q,index+5,q-p)
                successor = exp.end(nxt,best(teacher,nxt['x']))
                long[a] = exp.loss(successor)
        executed = best(teacher,n['x']) if teacher else int(rng.integers(3))
        rows.append(dict(index=index,features=n['x'],short=short,long=long,executed=executed,upload=n['upload']))
        previous = ends[executed]-p
        p = ends[executed]
    return rows


def fit_models(rows):
    X = [r['features'] for r in rows]
    model = {h:fit_head(X,[r[h] for r in rows]) for h in ('short','long')}
    model['no_direction'] = fit_head([np.array(x)[:,:6] for x in X],[r['long'] for r in rows])
    model['time'] = [int(np.mean([r['long'] for r in rows if r['index']==i],axis=0).argmin())
                     for i in range(0,40,5)]
    angles = np.array([x[2][7] for x in X])
    targets = np.array([r['long'] for r in rows])
    rules = [(float(t),lo,hi) for t in np.quantile(angles,[0,.25,.5,.75,1])
             for lo in range(3) for hi in range(3)]
    model['angle'] = min(rules,key=lambda rule:
        targets[np.arange(len(rows)),np.where(angles<=rule[0],rule[1],rule[2])].mean())
    return model


def choose(method, x, i, model):
    if method.startswith('fixed'):
        return int(method[-1])
    if method=='time':
        return model['time'][i//H]
    if method=='angle':
        t,lo,hi = model['angle']
        return lo if x[2][7]<=t else hi
    if method=='no_direction':
        return best(model[method],np.array(x)[:,:6])
    return best(model[method],x)


def score_policies(exp, model, family):
    methods = ['fixed0','fixed1','fixed2','time','short','long',
               'angle' if family=='alpha' else 'no_direction']
    results = {}
    # Share only identical model/action prefixes across independent policies.
    def visit(p, prev, i, names, history):
        n = exp.node(p,i,prev)
        decisions = {m:choose(m,n['x'],i,model) for m in names}
        for a in sorted(set(decisions.values())):
            q = exp.end(n,a)
            subset = [m for m in names if decisions[m]==a]
            item = dict(action=a,index=i,upload=n['upload'],features=n['x'],private_loss=exp.loss(q))
            records = history+[item]
            if i==35:
                metrics = exp.w.record_test(q)
                for m in subset:
                    results[m] = dict(**metrics,windows=records,uploads=480,sgd_steps=sum(exp.steps))
            else:
                visit(q,q-p,i+H,subset,records)
    visit(exp.start,torch.zeros_like(exp.start),0,methods,[])
    return results


def budget_tree(exp, held):
    nodes = {}
    scores = {}
    # Cache outcomes across budgets when verification returns exactly the same state.
    states = {}
    def visit(p,prev,i,used):
        key = (vec_hash(p),vec_hash(prev),i)
        if i==40:
            if key not in states:
                states[key] = dict(cost=terminal_cost(exp.w,p))
                if held:
                    states[key].update(exp.w.record_test(p))
            scores[used] = states[key]
            return
        if key not in states:
            n = exp.node(p,i,prev)
            ref = exp.end(n,0)
            states[key] = [n['x'][2],ref,None,None,n['upload']]
        x,ref,chosen,gain,upload = states[key]
        nodes[used,i//H] = x
        if len(used)<2:
            if chosen is None:
                # Reconstruct the node only if an earlier path exhausted its budget.
                n = exp.node(p,i,prev)
                trial = exp.end(n,2)
                gain = terminal_cost(exp.w,ref)-terminal_cost(exp.w,trial)
                chosen = trial if gain>0 else ref
                states[key][2:4] = [chosen,gain]
            visit(chosen,chosen-p,i+H,used+(i//H,))
        if 8-i//H > 2-len(used):
            visit(ref,ref-p,i+H,used)
    visit(exp.start,torch.zeros_like(exp.start),0,())
    def path(t): return ','.join(map(str,t))
    return dict(nodes={path(u)+'|'+str(j):x for (u,j),x in nodes.items()},
                leaves={path(u):v for u,v in scores.items()},steps=exp.steps)


def key(used):
    return ','.join(map(str,used))


def budget_x(tree, used, j):
    return tree['nodes'][key(used)+'|'+str(j)]


def budget_pick(tree, used, j, method, rule):
    if len(used)==2:
        return False
    if 8-j==2-len(used):
        return True
    x = budget_x(tree,used,j)
    if method=='threshold':
        return x[6]>=rule
    if method=='rl':
        # Features also encode remaining capacity, prior use and progress.
        v = np.r_[x, (2-len(used))/2, (j-used[-1])/8 if used else 0.]
        center,scale,beta,bias=rule[str(j)]
        return float(((v-center)/scale).clip(-3,3)@beta+bias)>0
    return j in rule


def budget_path(tree, used, start, method, rule):
    for j in range(start,8):
        if budget_pick(tree,used,j,method,rule):
            used += (j,)
    return used


def cost(tree,path):
    return tree['leaves'][key(path)]['cost']


def fit_budget(trees):
    pairs=list(combinations(range(8),2))
    timing=min(pairs,key=lambda pair:np.mean([cost(t,pair) for t in trees]))
    divergence=[x[6] for t in trees for x in t['nodes'].values()]
    cuts=np.quantile(divergence,np.linspace(0,1,21))
    threshold=min(cuts,key=lambda cut:np.mean([cost(t,budget_path(t,(),0,'threshold',cut)) for t in trees]))
    heads={}
    for j in reversed(range(7)):
        features, targets = [], []
        for tree in trees:
            for used in [()]+[(u,) for u in range(j)]:
                if 8-j<=2-len(used):
                    continue
                wait=budget_path(tree,used,j+1,'rl',heads)
                spend=budget_path(tree,used+(j,),j+1,'rl',heads)
                x=budget_x(tree,used,j)
                features.append(np.r_[x,(2-len(used))/2,(j-used[-1])/8 if used else 0.])
                targets.append(cost(tree,wait)-cost(tree,spend))
        v=np.array(features)
        center,scale=v.mean(0),v.std(0).clip(1e-6)
        z=((v-center)/scale).clip(-3,3)
        y=np.array(targets)
        beta=np.linalg.solve(z.T@z+16*np.eye(z.shape[1]),z.T@(y-y.mean()))
        heads[str(j)]=[center.tolist(),scale.tolist(),beta.tolist(),float(y.mean())]
    return dict(time=timing,threshold=float(threshold),rl=heads)


def task(root, family, env, seed, phase, model=None):
    folder = root/family/env/phase/str(seed)
    world=world_for(root,folder,env,seed)
    exp=Experiment(world,'tau' if family=='budget' else family)
    if family=='budget':
        data=budget_tree(exp,phase=='test')
    elif phase=='test':
        data=score_policies(exp,model,family)
    else:
        data=make_training(exp,seed,model)
    write_json(folder/'records.json',data)
    return data


def batch(pool, root, tasks, phase, models):
    jobs={(f,e,s):pool.submit(task,root,f,e,s,phase,models.get(f)) for f,e,s in tasks}
    return {k:v.result() for k,v in jobs.items()}


def run(root, workers):
    root.mkdir(parents=True,exist_ok=False)
    native=Path('dataset/mnist').resolve()
    for offset in (0,24,48):
        for split in ('train','test'):
            target=root/'data'/str(offset)/'mnist'/split
            target.mkdir(parents=True)
            for i in range(24):
                (target/f'{i}.npz').symlink_to(native/split/f'{i+offset}.npz')
    write_json(root/'plan.json',dict(pilot=PILOT,train=TRAIN,test=HELD,
        train_clients=[0,23],test_client_ranges=[[24,47],[48,71]],windows=8,
        primary='paired final round 40 global accuracy; secondary worst 3 of 24 clients',
        budget='exactly two verified five-round trials, shared first upload',
        first='cross-seed frozen short continuation; ten-round endpoint CE vs five-round CE',
        eval_pop='all 24 clients including simulated flip clients, on clean client test data'))
    settings=[('tau','rotating')]+[('alpha',e) for e in ENVS]
    with ProcessPoolExecutor(workers) as pool:
        pilot=batch(pool,root,[(f,e,s) for s in PILOT for f,e in settings],'pilot',{})
        teachers={f:fit_models([r for (family,e,s),rows in pilot.items() if family==f for r in rows])['short']
                  for f in ('tau','alpha')}
        fit=batch(pool,root,[(f,e,s) for s in TRAIN for f,e in settings],'train',teachers)
        models={f:fit_models([r for (family,e,s),rows in fit.items() if family==f for r in rows])
                for f in ('tau','alpha')}
        write_json(root/'models.json',models)
        batch(pool,root,[(f,e,s) for s in HELD for f,e in settings],'test',models)
        bt=batch(pool,root,[('budget','rotating',s) for s in TRAIN],'train',{})
        bm=fit_budget(list(bt.values()))
        write_json(root/'budget_model.json',bm)
        batch(pool,root,[('budget','rotating',s) for s in HELD],'test',{})
    summarize(root)


def summarize(root):
    def read(f): return json.loads(f.read_text())
    metrics={}
    individual={}
    for family,env in [('tau','rotating')]+[('alpha',e) for e in ENVS]+[('budget','rotating')]:
        rows=[]
        for seed in HELD:
            records=read(root/family/env/'test'/str(seed)/'records.json')
            if family=='budget':
                bm=read(root/'budget_model.json')
                values={}
                for m,rule in bm.items():
                    path=budget_path(records,(),0,m,rule)
                    leaf=records['leaves'][key(path)]
                    steps=sum(records['steps'])+sum(sum(records['steps'][5*j+1:5*j+5]) for j in path)
                    values[m]=dict(**leaf,uploads=576,sgd_steps=steps,slots=path)
                leaves=list(records['leaves'].values())
                assert len(leaves)==28
                values['random']={k:float(np.mean([v[k] for v in leaves])) for k in ('accuracy','worst10')}
                values['random'].update(uploads=576,sgd_steps=sum(records['steps'])*1.+sum(
                    sum(records['steps'][5*j+1:5*j+5]) for j in range(8))/4)
                records=values
            rows.append(records)
        name=family+'_'+env
        metrics[name]={m:{k:float(np.mean([r[m][k] for r in rows])) for k in
                         ('accuracy','worst10','uploads','sgd_steps')} for m in rows[0]}
        individual[name]=[{m:{k:v for k,v in stats.items() if k not in ('windows','cost')} for m,stats in row.items()} for row in rows]
    write_json(root/'summary.json',dict(means=metrics,per_seed=individual))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',default=5,type=int)
    args=parser.parse_args()
    run(args.output.resolve(),args.workers)
