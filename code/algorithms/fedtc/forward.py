"""Forward layer, corresponding to eq:geom_median through eq:cutoff_radius."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

LAMBDA_MAX = 0.5
TAU_MIN = 1.05
TAU_MAX = 1.80
SIGMA = 4.0
EPS_SCALE = 1e-8
CONSERVATIVE_ACTION = (0.0, 1.35)


def validate_action(action) -> tuple[float, float]:
    lam, tau = map(float, action)
    if not (0 <= lam <= LAMBDA_MAX and TAU_MIN <= tau <= TAU_MAX):
        raise ValueError("Action must satisfy lambda in [0, .5], tau in [1.05, 1.8].")
    return lam, tau


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """One-based average ranks; ties have equal influence, independent of order."""
    if values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("Ranks require a nonempty finite vector.")
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True,
                                      return_counts=True)
    ends = counts.cumsum(0).to(values.dtype)
    return (ends - (counts.to(values.dtype) - 1) / 2)[inverse]


def rank_normalize(values: torch.Tensor) -> torch.Tensor:
    return (average_ranks(values) - 1) / max(len(values) - 1, 1)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    a, b = average_ranks(x), average_ranks(y)
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float((a.dot(b) / denom).clamp(-1, 1)) if denom > 0 else 0.0


def distances_to(points: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    # Avoid another full K x d allocation for ResNet-18 client updates.
    squared = points.new_zeros(len(points))
    for start in range(0, points.shape[1], 262144):
        diff = points[:, start:start + 262144] - center[start:start + 262144]
        squared += (diff * diff).sum(dim=1)
    result = squared.sqrt()
    if not torch.isfinite(result).all():
        raise ValueError("Update magnitudes overflowed the aggregation dtype.")
    return result


@torch.no_grad()
def geometric_median(points: torch.Tensor, iterations: int = 15,
                     eps: float = 1e-6) -> torch.Tensor:
    """Stabilized Weiszfeld with the correct coincident-point subgradient test.

    The manuscript's finite iteration budget approximates the geometric median.
    It does not in general solve the minimization exactly in 15 iterations.
    """
    if (points.ndim != 2 or min(points.shape) == 0 or
            not points.is_floating_point() or not torch.isfinite(points).all()):
        raise ValueError("Updates must be a nonempty finite floating-point K x d tensor.")
    if iterations < 1 or eps <= 0:
        raise ValueError("Weiszfeld iterations and epsilon must be positive.")
    center = points.mean(0)
    for _ in range(iterations):
        dist = distances_to(points, center)
        coincident = dist == 0
        if coincident.all():
            return center
        inverse = torch.where(coincident, torch.zeros_like(dist), dist.clamp_min(eps).reciprocal())
        candidate = points.T.mv(inverse / inverse.sum())
        if coincident.any():
            residual = points.T.mv(inverse) - center * inverse.sum()
            residual_norm = float(residual.norm())
            multiplicity = int(coincident.sum())
            if residual_norm <= multiplicity:
                return center
            mix = multiplicity / residual_norm
            candidate = mix * center + (1 - mix) * candidate
        change = (candidate - center).norm()
        center = candidate
        if change <= eps:
            break
    return center


@dataclass
class CalibrationResult:
    delta: torch.Tensor
    anchor: torch.Tensor
    weights: torch.Tensor
    credibility: torch.Tensor
    distances: torch.Tensor
    deviations: torch.Tensor
    utilities: torch.Tensor
    scale: float
    cutoff_radius: float
    geometry: torch.Tensor
    fallback: bool


@torch.no_grad()
def calibrate(updates: torch.Tensor, utilities: torch.Tensor, action,
              iterations: int = 15, eps: float = 1e-6) -> CalibrationResult:
    lam, tau = validate_action(action)
    anchor = geometric_median(updates, iterations, eps)
    utilities = torch.as_tensor(utilities, dtype=updates.dtype, device=updates.device)
    if utilities.shape != (len(updates),):
        raise ValueError("One scalar utility per client is required.")
    ranked = rank_normalize(utilities)  # Finite out-of-range fake claims remain rank bounded.
    dist = distances_to(updates, anchor)
    scale = torch.quantile(dist, 0.5)
    deviation = dist / (scale + EPS_SCALE)
    credibility = ((1 - lam) + lam * ranked - tau * torch.tanh(deviation / SIGMA)).clamp_min(0)
    radius = SIGMA * (float(scale) + EPS_SCALE) * math.atanh(1 / tau)
    credibility[dist >= radius] = 0  # Exact zero at the analytical boundary, including roundoff.
    fallback = bool(credibility.sum() == 0)
    weights = credibility / credibility.sum() if not fallback else torch.zeros_like(credibility)
    delta = updates.T.mv(weights) if not fallback else anchor.clone()
    iqr = torch.quantile(deviation, 0.75) - torch.quantile(deviation, 0.25)
    geometry = torch.tensor([
        float(scale), float(iqr / (torch.quantile(deviation, 0.5) + EPS_SCALE)),
        float((deviation > 2).float().mean()), spearman(ranked, -deviation),
        float((credibility > 0).float().mean()),
    ], dtype=torch.float32)
    return CalibrationResult(delta, anchor, weights, credibility, dist, deviation,
                             ranked, float(scale), radius, geometry, fallback)


def macro_state(geometries: list[torch.Tensor], low: float, high: float,
                progress: float) -> torch.Tensor:
    """Raw 8D state; interval coordinates are normalized on controller input.

    Completed macro-step geometry is used for the next action (no future data).
    Entries 2--5 average per-round summaries; entry 1 logs the mean raw scale.
    """
    if not geometries:
        geometry = torch.tensor([0., 0., 0., 0., 1.])
    else:
        geometry = torch.stack(geometries).mean(0)
    geometry[0] = torch.log1p(geometry[0])
    return torch.cat((geometry, torch.tensor([low, high - low, progress])))
