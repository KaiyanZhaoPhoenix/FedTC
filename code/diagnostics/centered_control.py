"""Opt-in centered forward calibration and deterministic five-round feedback control."""
from __future__ import annotations
import math
from dataclasses import dataclass, replace

import torch

from algorithms.fedtc.forward import EPS_SCALE, calibrate

SIGMA = 1.5
TAU_RANGE = (1.05, 2.5)


def centered(updates, utilities, action):
    lam, tau = action
    if not (0 <= lam <= .5 and TAU_RANGE[0] <= tau <= TAU_RANGE[1]):
        raise ValueError("Centered calibration requires lambda in [0,.5] and tau in [1.05,2.5]")
    base = calibrate(updates, utilities, (0., 1.35))
    budget = 1 - lam + lam * base.utilities
    penalty = torch.tanh((base.deviations - 1).clamp_min(0) / SIGMA)
    credit = (budget - tau * penalty).clamp_min(0)
    cutoff = 1 + SIGMA * torch.atanh(budget / tau)
    credit[base.deviations >= cutoff] = 0
    weights = credit / credit.sum()
    geom = base.geometry.clone()
    geom[-1] = (credit > 0).float().mean()
    radius = (base.scale + EPS_SCALE) * (1 + SIGMA * math.atanh(1 / tau))
    return replace(base, delta=updates.T.mv(weights), weights=weights, credibility=credit,
                   cutoff_radius=radius, geometry=geom, fallback=False)


@dataclass(frozen=True)
class Feedback:
    low: float
    high: float
    trimmed_low: float
    trimmed_high: float
    median: float
    negative_share: float
    outside: float
    safe_rho: float | None


def next_action(action, current, previous):
    """Consume completed windows. Abstain from changing lambda when all ranks tie."""
    lam, tau = action
    shock = previous is not None and (
        current.outside - previous.outside >= .10 or
        (current.high - current.low) - (previous.high - previous.low) >= .08 or
        previous.trimmed_high - current.trimmed_high >= .06)
    minority_alarm = (current.low < 0 and current.trimmed_low < -.01
                      and current.median > 0 and current.negative_share >= .25)
    reason = "hold"
    if shock:
        tau += .30
        reason = "tighten"
    elif minority_alarm:
        tau -= .30
        reason = "loosen"
    if current.safe_rho is not None:
        target = 0. if current.safe_rho < -.2 else .3
        lam = (lam + target) / 2
    return (lam, min(max(tau, TAU_RANGE[0]), TAU_RANGE[1])), reason
