"""Small deterministic federated head-training simulator for the Stage A comparison."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.fedtc.forward import calibrate, macro_state
from algorithms.fedtc.interval import IntervalEstimator

GRID = [(lam,tau) for lam in (0.,.3) for tau in (1.15,1.35,1.65)]
CLIENTS, ACTIVE, EVALUATORS, LOCAL_STEPS, BATCH = 24,12,16,4,32
H, WINDOWS = 5,8


def stream(seed,tag):
    return np.random.default_rng(np.random.SeedSequence([seed,tag]))


def features(x):
    normalized = F.normalize(torch.as_tensor(x,dtype=torch.float32),dim=1)*4
    return F.pad(normalized,(0,1),value=1.)


@lru_cache(maxsize=3)
def digits(root,offset):
    arrays={}
    for split in ("train","test"):
        xx,yy,clients=[],[],[]
        client_ids = range(offset, offset+CLIENTS) if isinstance(offset, (int,np.integer)) else offset
        for c, source_id in enumerate(client_ids):
            with np.load(Path(root)/"mnist"/split/f"{source_id}.npz",allow_pickle=True) as f:
                v=f["data"].item()
            xx.append(F.avg_pool2d(torch.tensor(np.array(v["x"])),2).flatten(1))
            yy.extend(v["y"])
            clients.extend([c]*len(v["y"]))
        arrays[split]=(features(torch.cat(xx)),torch.tensor(np.array(yy),dtype=torch.long),np.array(clients))
    return arrays


@lru_cache(maxsize=1)
def cifar(file):
    with np.load(file) as d:
        center=d["train_x"].mean(0)
        scale=d["train_x"].std(0).clip(.1)
        return {s:(features(((d[s+"_x"]-center)/scale).clip(-4,4)),torch.tensor(d[s+"_y"])) for s in ("train","test")}


def dirichlet(labels,seed):
    rng=stream(seed,1)
    labels=np.asarray(labels)
    for _ in range(100):
        buckets=[[] for _ in range(CLIENTS)]
        for cls in np.unique(labels):
            ids=rng.permutation(np.flatnonzero(labels==cls))
            sizes=rng.multinomial(len(ids),rng.dirichlet(np.full(CLIENTS,.1)))
            for b,part in zip(buckets,np.split(ids,sizes.cumsum()[:-1])):
                b.extend(part)
        if min(map(len,buckets))>=32:
            return [np.asarray(b) for b in buckets]
    raise RuntimeError("Could not draw a usable client partition")


def ce(logits,labels):
    return F.cross_entropy(logits.flatten(0,-2),labels.flatten(),reduction="none").reshape(labels.shape)


class World:
    def __init__(self,scenario,seed,phase,root,cifar_file,balanced_head=False,prototype_init=False,client_ids=None):
        self.scenario=scenario
        self.cosine_head=False
        if scenario=="switch":
            offset={"train":0,"calibrate":24,"held":48+24*(seed%2)}[phase]
            if client_ids is not None and (len(client_ids)!=CLIENTS or len(set(client_ids))!=CLIENTS):
                raise ValueError("A MNIST client pool must contain 24 distinct client IDs")
            data=digits(root,offset if client_ids is None else tuple(client_ids))
            self.x,self.y,ids=data["train"]
            self.tx,self.ty,_=data["test"]
            shards=[np.flatnonzero(ids==c) for c in range(CLIENTS)]
            self.classes=10
        else:
            data=cifar(cifar_file)
            self.x,self.y=data["train"]
            self.tx,self.ty=data["test"]
            shards=dirichlet(self.y,seed)
            self.classes=100
        self.w=torch.randn((self.x.shape[1],self.classes),generator=torch.Generator().manual_seed(seed))*.002
        part=stream(seed,2)
        train,valid=[],[]
        for shard in shards:
            ids=part.permutation(shard)
            n=max(1,min(len(ids)-1,round(.2*len(ids))))
            train.append(ids[n:]);valid.append(ids[:n])
        self.train=train
        self.val=np.concatenate(valid)
        self.val_ids=np.concatenate([np.full(len(v),c) for c,v in enumerate(valid)])
        self.sizes=torch.tensor([len(v) for v in valid])
        counts=np.bincount(self.y[np.concatenate(train)].numpy(),minlength=self.classes)
        self.training_log_prior=torch.tensor(np.log(counts.clip(1)/counts.sum()),dtype=self.w.dtype) if balanced_head else None
        if prototype_init:
            sums=torch.zeros(self.classes,self.x.shape[1]-1)
            for ids in train:
                sums.index_add_(0,self.y[ids],self.x[ids,:-1])
            self.w[:-1]=F.normalize(sums,dim=1).T
            self.w[-1]=0
        self.tail=np.argsort(counts,kind="stable")[:self.classes//5] if scenario=="switch" else np.arange(80,100)
        self.tail_frac=np.array([np.isin(self.y[t],self.tail).mean() for t in train])
        self.adversaries=stream(seed,3).choice(CLIENTS,4,replace=False)
        cohort=stream(seed,4); batches=stream(seed,5); evals=stream(seed,6)
        self.cohorts=[cohort.choice(CLIENTS,ACTIVE,replace=False) for _ in range(H*WINDOWS)]
        self.indices=[torch.tensor(np.stack([batches.choice(train[c],(LOCAL_STEPS+1,BATCH),replace=True)
                                             for c in ids])) for ids in self.cohorts]
        self.evals=[evals.choice(CLIENTS,EVALUATORS,replace=False) for _ in range(WINDOWS)]
        self.estimator=IntervalEstimator(CLIENTS,EVALUATORS,rho_max=.2,loss_cap=5.)
        self.vx,self.vy=self.x[self.val],self.y[self.val]
        self.val_groups=torch.from_numpy(self.val_ids)
        self.val_loss=self.losses()
        self.state=macro_state([],0.,0.,0.).numpy()
        self.last_action=1

    def losses(self):
        loss=ce(self.vx@self.w,self.vy).clamp(max=5)
        return (torch.bincount(self.val_groups,weights=loss,minlength=CLIENTS)/self.sizes).numpy()

    def observe(self):
        return np.r_[self.state,GRID[self.last_action]].tolist()

    def uploads(self,i,contaminated):
        ids=self.indices[i]; cohort=self.cohorts[i]
        xx=self.x[ids]; yy=self.y[ids]
        local=self.w.expand(ACTIVE,-1,-1).clone()
        pre=ce(xx[:,-1]@local,yy[:,-1]).mean(1)
        prior = self.training_log_prior
        if prior is not None and prior.ndim == 2:
            prior = prior[cohort, None, :]
        for k in range(LOCAL_STEPS):
            logits=xx[:,k]@local
            if prior is not None:
                logits=logits+prior
            grad=logits.softmax(-1)
            grad.scatter_add_(2,yy[:,k,:,None],-torch.ones_like(yy[:,k,:,None],dtype=grad.dtype))
            step=(xx[:,k].transpose(1,2)@grad)/BATCH
            if self.cosine_head:
                # Unit class directions, fixed scale from the norm-4 input features.
                direction = local[:, :-1]
                tangent = step[:, :-1] - direction * (step[:, :-1] * direction).sum(1, keepdim=True)
                local[:, :-1] = F.normalize(direction - .35 * tangent, dim=1)
            else:
                local -= .35 * step
        util=(pre-ce(xx[:,-1]@local,yy[:,-1]).mean(1)).clamp_min(0)
        uploads=(local-self.w).flatten(1)
        mask=np.isin(cohort,self.adversaries) & (contaminated)
        if mask.any():
            # Fixed ALIE-style benchmark contamination, z=1, omniscient simulator.
            clean=uploads[~mask]
            uploads[mask]=clean.mean(0)-clean.std(0,unbiased=False)
            util[mask]=util.max()+1
        return uploads,util,mask

    def advance(self,t,a):
        geom=[];mass=[];available=[];bad_mass=[];uf=[];bad_count=0
        before=self.val_loss
        for i in range(t*H,(t+1)*H):
            cohort=self.cohorts[i]
            uploads,util,mask=self.uploads(i,self.scenario=="switch" and i>=20)
            bad_count+=int(mask.sum())
            result=calibrate(uploads,util,GRID[a])
            self.w=self.w+result.delta.reshape_as(self.w)
            geom.append(result.geometry)
            wt=result.weights.numpy()
            mass.append(float(wt@self.tail_frac[cohort]))
            available.append(float(self.tail_frac[cohort].mean()))
            bad_mass.append(float(wt[mask].sum()))
            uf.append(bool(result.fallback))
        self.val_loss=self.losses()
        ci=self.estimator.estimate((before-self.val_loss)[self.evals[t]])
        self.state=macro_state(geom,ci.low,ci.high,(t+1)/WINDOWS).numpy()
        self.last_action=a
        return dict(low=ci.low,high=ci.high,slack=ci.sampling_slack,tail_mass=np.mean(mass),
                    available_tail=np.mean(available),corrupted_weight=np.mean(bad_mass),
                    corrupted_uploads=bad_count,anchor_fallbacks=sum(uf))

    def metrics(self,split):
        x,y=(self.tx,self.ty) if split=="test" else (self.vx,self.vy)
        pred=(x@self.w).argmax(1)
        correct=pred.eq(y)
        n=torch.bincount(y,minlength=self.classes)
        hits=torch.bincount(y[correct],minlength=self.classes)
        present=self.tail[n[self.tail].numpy()>0]
        return dict(global_acc=correct.float().mean().item(),tail20=(hits[present]/n[present]).mean().item())
