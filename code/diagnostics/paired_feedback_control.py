"""Paired validation probes and offline finite-horizon rollout policy improvement."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.baselines import aggregate
from algorithms.fedtc.step_control import RolloutScorer
from diagnostics.centered_control import centered
from diagnostics.stage_a_world import H, WINDOWS

# First knob interpolates uniform-client FedTC calibration and sample weighting.
# Second knob scales the aggregated update on the server.
ACTIONS = [(mix, scale) for scale in (1., 2., 4., 8.) for mix in (0., .5, 1.)]
DISCOUNT = .9
DEPTH = 3


def directions(world, index):
    updates, utility, _ = world.uploads(index, index >= 20)
    calibrated = centered(updates, utility, (.3, 1.05)).delta
    sizes = [len(world.train[c]) for c in world.cohorts[index]]
    average, _ = aggregate('FedAvg', updates, sizes, 4, 1)
    return calibrated.reshape_as(world.w), average.reshape_as(world.w)


def proposal(world, weights, deltas, arm):
    mix, scale = ACTIONS[arm]
    result = weights + scale * torch.lerp(deltas[0], deltas[1], mix)
    if world.cosine_head:
        result[:-1] = F.normalize(result[:-1], dim=0)
        result[-1].zero_()
    return result


def advance(world, start, macro, arm, first=None):
    world.w = start
    for i in range(macro*H, (macro+1)*H):
        delta = first if i==macro*H and first is not None else directions(world,i)
        world.w = proposal(world, world.w, delta, arm)
    return world.w


@dataclass
class Panel:
    x: torch.Tensor
    y: torch.Tensor
    groups: torch.Tensor
    sizes: torch.Tensor

    @classmethod
    def from_world(cls, world, macro, clients=None):
        ids = torch.as_tensor(world.evals[macro] if clients is None else clients)
        selected = torch.isin(world.val_groups, ids)
        groups = world.val_groups[selected]
        _, inverse, sizes = torch.unique(groups, return_inverse=True, return_counts=True)
        return cls(world.vx[selected], world.vy[selected], inverse, sizes)

    def reports(self, model):
        logits = self.x @ model
        # Only per-client summaries are used by the server. No private examples
        # or class labels are exposed to the controller in the simulator API.
        loss = F.cross_entropy(logits,self.y,reduction='none').clamp_max(5.)
        error = logits.argmax(1).ne(self.y).float()
        return torch.stack([torch.bincount(self.groups,weights=v,minlength=len(self.sizes))/self.sizes
                            for v in (error,loss)],dim=1).numpy()

    def summarize(self, reports):
        return np.r_[np.average(reports,axis=0,weights=self.sizes.numpy()),
                     reports.mean(axis=0),np.quantile(reports[:,1],.75)]

    def cost(self, reports):
        err, loss = np.average(reports,axis=0,weights=self.sizes.numpy())
        return float(err + .15*loss)


def probe(world, start, macro, panel):
    world.w = start
    deltas = directions(world,macro*H)
    initial = panel.reports(start)
    base = panel.summarize(initial)
    signals = np.stack([panel.summarize(initial-panel.reports(proposal(world,start,deltas,a)))
                        for a in range(len(ACTIONS))])
    return deltas, initial, base, signals


def design(base, signals, macro):
    # Action-specific intercepts and simple time/loss interactions share the
    # same paired training labels as the validation-gain terms.
    identity = np.eye(len(ACTIONS))
    t = macro/(WINDOWS-1)
    return np.column_stack([identity, identity*t, identity*base[0], signals,
                            signals*t, signals*base[0]])


def paired_labels(world,start,t,first,initial,panels,reference):
    labels=[]
    for arm in range(len(ACTIONS)):
        model=start
        gains=[]
        for j in range(t,min(WINDOWS,t+DEPTH)):
            p=panels[j]
            before=initial if j==t else p.reports(model)
            model=advance(world,model,j,arm if j==t else reference,first if j==t else None)
            gains.append(p.cost(before)-p.cost(p.reports(model)))
        labels.append((gains[0],sum(DISCOUNT**k*g for k,g in enumerate(gains))))
    return np.array(labels)
