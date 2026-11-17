"""Decoupled evaluator confidence interval (eq:trimmed_bounds, eq:ci_interval)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


def hypergeometric_tail(population: int, bad: int, sample: int, bound: int) -> float:
    """Exact P[B > bound], without a SciPy dependency."""
    denominator = math.comb(population, sample)
    return sum(math.comb(bad, k) * math.comb(population - bad, sample - k) / denominator
               for k in range(max(bound + 1, sample - (population - bad), 0),
                              min(sample, bad) + 1))


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    trimmed_low: float
    trimmed_high: float
    sampling_slack: float
    bad_bound: int
    delta_bad: float

    def to_dict(self):
        return asdict(self)


class IntervalEstimator:
    def __init__(self, population: int, sample_size: int, rho_max: float = .2,
                 loss_cap: float = 5., delta_sample: float = .05,
                 delta_bad: float = .05, eta: float | None = None):
        if not (1 <= sample_size <= population and 0 <= rho_max < .5):
            raise ValueError("Require 1 <= evaluator cohort <= population and 0 <= rho_max < .5.")
        if not (0 < delta_sample < 1 and 0 < delta_bad < 1 and math.isfinite(loss_cap) and loss_cap > 0):
            raise ValueError("Invalid loss cap or confidence failure probability.")
        self.population, self.sample_size = population, sample_size
        self.rho_max, self.loss_cap, self.delta_sample = rho_max, loss_cap, delta_sample
        maximum_bad = math.floor(population * rho_max + 1e-12)
        start = math.ceil(rho_max * sample_size - 1e-12)
        if eta is None:
            choices = [b for b in range(start, (sample_size - 1) // 2 + 1)
                       if hypergeometric_tail(population, maximum_bad, sample_size, b) <= delta_bad]
            if not choices:
                raise ValueError("No b < m/2 meets delta_bad. Increase evaluator cohort size or relax delta_bad.")
            self.bad_bound = choices[0]
        else:
            if not math.isfinite(eta) or eta < 0:
                raise ValueError("eta must be finite and nonnegative.")
            self.bad_bound = math.ceil((rho_max + eta) * sample_size - 1e-12)
            if 2 * self.bad_bound >= sample_size:
                raise ValueError("The manuscript requires b < m/2.")
        self.delta_bad = hypergeometric_tail(population, maximum_bad, sample_size, self.bad_bound)
        # Snap eta to b/m - rho, so m(1-rho-eta) = m-b exactly. This also avoids
        # overstating the honest count when the paper's ceil rounds upward.
        self.effective_eta = self.bad_bound / sample_size - rho_max
        self.slack = loss_cap * math.sqrt(2 * math.log(2 / delta_sample) / (sample_size - self.bad_bound))

    def estimate(self, reports) -> ConfidenceInterval:
        values = torch.as_tensor(reports, dtype=torch.float64, device="cpu")
        if values.shape != (self.sample_size,) or torch.isnan(values).any():
            raise ValueError("Evaluator reports must have the configured size and contain no NaN.")
        values = values.clamp(-self.loss_cap, self.loss_cap).sort().values
        b, m = self.bad_bound, self.sample_size
        low, high = float(values[:m-b].mean()), float(values[b:].mean())
        return ConfidenceInterval(low - self.slack, high + self.slack, low, high,
                                  self.slack, b, self.delta_bad)
