"""Matched online CI-bandit vs discounted fitted Q vs validation-selected fixed control."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from diagnostics.stage_a_world import ACTIVE, GRID, H, WINDOWS, World, stream
from utils.experiment import write_json

FIT=list(range(7100,7148))
TUNE=list(range(7200,7206))
HELD=list(range(7300,7316))
METHODS=("static","bandit","rl")


def gain(row):
    # Remove only the fixed sampling penalty; this shifts all finite-horizon
    # returns at a given time by the same constant.
    return .75*row["low"]+.25*row["high"]+.5*row["slack"]


class Critic:
    def __init__(self,experience,discount,normalization):
        self.center,self.scale=normalization
        self.discount=discount
        x=self.design([r["s"] for r in experience])
        after=self.design([r["next"] for r in experience])
        actions=np.array([r["a"] for r in experience])
        reward=np.array([gain(r) for r in experience])
        survives=np.array([r["t"]<WINDOWS-1 for r in experience])
        masks=[actions==a for a in range(len(GRID))]
        transforms=[]
        for mask in masks:
            xi=x[mask]
            penalty=8*np.eye(xi.shape[1])
            penalty[0,0]=1e-6
            transforms.append(np.linalg.solve(xi.T@xi+penalty,xi.T))
        target=reward.copy()
        for _ in range(WINDOWS if discount else 1):
            self.weights=np.stack([tr@target[mask] for mask,tr in zip(masks,transforms)])
            target=reward+discount*survives*(after@self.weights.T).max(axis=1)

    def design(self,states):
        x=(np.asarray(states)-self.center)/self.scale
        x=x.clip(-3,3)
        return np.column_stack((np.ones(len(x)),x,x[:,:5]*x[:,7:8],x[:,7]**2))

    def scores(self,state):
        return (self.design([state])@self.weights.T)[0]


def rollout(config,scenario,phase,seed,method,arm,experience,norm,behavior):
    torch.set_num_threads(1)
    w=World(scenario,seed,phase,config["data"],config["cifar"],config["balanced_head"],config["prototype_init"])
    logged=[]; added=[]; curve=[]
    cohorts_hash=sha256(np.asarray(w.cohorts,dtype='<i8').tobytes()+np.asarray(w.evals,dtype='<i8').tobytes()).hexdigest()
    init_hash=sha256(w.w.numpy().tobytes()).hexdigest()
    initial_test=w.metrics("test") if phase=="held" else None
    for t in range(WINDOWS):
        state=w.observe()
        scores=None
        if phase=="train":
            a=int(behavior[t])
        elif method=="static":
            a=arm
        else:
            scores=Critic(experience+added,.9 if method=="rl" else 0.,norm).scores(state)
            a=int(np.argmax(scores))
        r=w.advance(t,a)
        row=dict(s=state,a=a,next=w.observe(),t=t,low=r['low'],high=r['high'],slack=r['slack'])
        added.append(row)
        logged.append(dict(row,**{k:v for k,v in r.items() if k not in row},scores=None if scores is None else scores.tolist()))
        if phase=="held":
            curve.append(w.metrics("test"))
    if phase=="train":return added
    return dict(seed=seed,method=method,arm=arm,init_hash=init_hash,cohorts_hash=cohorts_hash,
                tail_labels=w.tail.tolist(),actions=[r["a"] for r in logged],log=logged,curve=curve,
                validation=w.metrics("validation"),uploads=ACTIVE*H*WINDOWS,initial_test=initial_test)


def run(root,scenario,config,jobs):
    root.mkdir(parents=True,exist_ok=False)
    primary="global_acc" if scenario=="switch" else "tail20"
    offset=2000 if config["prototype_init"] else 1000 if config["balanced_head"] else 0
    fit,tuning,held=([s+offset for s in bank] for bank in (FIT,TUNE,HELD))
    behavior=np.stack([stream(t,1701).permutation(np.tile(np.arange(6),len(FIT)//6)) for t in range(WINDOWS)],axis=1)
    write_json(root/"protocol.json",dict(scenario=scenario,grid=GRID,macro=H,windows=WINDOWS,fit=fit,tune=tuning,
          held=held,primary=primary,balanced_head=config["balanced_head"],prototype_init=config["prototype_init"],
          global_noninferiority_pp=.5,equivalence_margin_pp=.5,behavior=behavior.tolist(),
          discounted_ridge=8,score="0.75 L + 0.25 U; remove action-independent sampling offset",
          training="shared balanced exploratory trajectories; both adapt after each observed window; reset between held trajectories",
          switch="clean through round 20, fixed ALIE-style z=1 plus top-ranked utility from 4/24 clients from round 21; honest validation reports",
          tail="CIFAR100 train imbalance 10:1, Dirichlet(0.1), fixed classes 80..99 before federated split; official balanced test",
          model="linear softmax head; pooled MNIST pixels or frozen ImageNet SqueezeNet1.1 512-D features",
          monitoring="eight evaluations for each held method; no test responses are used by the policy"))
    with ProcessPoolExecutor(jobs) as pool:
        tasks=[pool.submit(rollout,config,scenario,"train",s,"behavior",None,None,None,b)
               for s,b in zip(fit,behavior)]
        tune=[pool.submit(rollout,config,scenario,"calibrate",s,"static",a,None,None,None) for s in tuning for a in range(6)]
        exp=[row for f in tasks for row in f.result()]
        vals=[f.result() for f in tune]
        static_scores=[np.mean([r["validation"][primary] for r in vals if r["arm"]==a]) for a in range(6)]
        arm=int(np.argmax(static_scores))
        x=np.array([r["s"] for r in exp])
        norm=(x.mean(0),x.std(0).clip(1e-5))
        write_json(root/"development.json",dict(transitions=exp,normalization=[z.tolist() for z in norm],
                   static_scores=static_scores,static_arm=arm))
        print(scenario,"development complete",arm,static_scores,flush=True)
        tasks=[pool.submit(rollout,config,scenario,"held",s,m,arm,exp,norm,None) for s in held for m in METHODS]
        rows=[f.result() for f in tasks]
    for s in held:
        triplet=[r for r in rows if r["seed"]==s]
        assert len({(r["cohorts_hash"],r["init_hash"],r["uploads"],tuple(r["tail_labels"])) for r in triplet})==1
    write_json(root/"held.json",rows)
    summarize(root,rows)
    print(scenario,"complete",flush=True)


def summarize(root,rows):
    rng=np.random.default_rng(3501)
    idx=rng.integers(0,len(HELD),(25000,len(HELD)))
    means={}; pairs={}
    for k in ("global_acc","tail20"):
        values={m:np.array([r["curve"][-1][k]*100 for r in rows if r["method"]==m]) for m in METHODS}
        means[k]={m:float(v.mean()) for m,v in values.items()}
        pairs[k]={}
        for a,b in (("rl","static"),("rl","bandit"),("bandit","static")):
            d=values[a]-values[b]
            pairs[k][a+"-"+b]=dict(pp=float(d.mean()),ci95=np.percentile(d[idx].mean(axis=1),[2.5,97.5]).tolist(),
                                    wins=int((d>0).sum()),ties=int((d==0).sum()),losses=int((d<0).sum()))
    write_json(root/"summary.json",dict(means_pct=means,paired=pairs))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario",choices=("switch","longtail"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,default=Path("dataset"))
    parser.add_argument("--cifar-features",type=Path,default=Path("runs/stage_a_data/cifar100_lt_features.npz"))
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--balanced-head",action="store_true",help="CIFAR floor diagnostic: use train-only class priors in local SGD; fresh seed banks")
    parser.add_argument("--prototype-init",action="store_true",help="Initialize all CIFAR heads with normalized client-training class sums")
    args=parser.parse_args()
    if args.jobs<1:parser.error("Positive worker count required")
    if args.balanced_head and args.scenario!="longtail":parser.error("Class-prior diagnostic requires the long-tail scenario")
    if args.prototype_init and (args.scenario!="longtail" or args.balanced_head):
        parser.error("Prototype initialization is a separate long-tail diagnostic")
    run(args.output,args.scenario,dict(data=str(args.data_root.resolve()),cifar=str(args.cifar_features.resolve()),
        balanced_head=args.balanced_head,prototype_init=args.prototype_init),args.jobs)
