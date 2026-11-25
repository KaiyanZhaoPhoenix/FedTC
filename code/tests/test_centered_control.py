import math
import unittest
from dataclasses import replace

import torch

from algorithms.fedtc.forward import EPS_SCALE
from diagnostics.centered_control import Feedback, centered, next_action


class CenteredControlTests(unittest.TestCase):
    def test_zero_median_and_ties(self):
        updates = torch.tensor([[0.], [0.], [0.], [10.], [20.]])
        result = centered(updates, torch.ones(5), (.5, 2.5))
        self.assertFalse(result.fallback)
        torch.testing.assert_close(result.weights, torch.tensor([1/3, 1/3, 1/3, 0., 0.]))
        self.assertTrue(torch.isfinite(result.delta).all())

    def test_inner_credit_and_rank_dependent_cutoff(self):
        updates = torch.tensor([[-5.], [-2.], [-1.], [0.], [1.], [2.], [5.]])
        u = torch.arange(7).float()
        r = centered(updates, u, (.5, 2.5))
        budget = .5 + .5 * r.utilities
        inner = r.deviations <= 1
        torch.testing.assert_close(r.credibility[inner], budget[inner])
        radius = 1 + 1.5 * torch.atanh(budget/2.5)
        self.assertTrue(torch.equal(r.weights > 0, r.deviations < radius))
        self.assertAlmostEqual(r.cutoff_radius/(r.scale+EPS_SCALE), 1+1.5*math.atanh(1/2.5))
        with self.assertRaises(ValueError):
            centered(updates, u, (0., 2.6))

    def test_alert_requires_observed_harm_and_shock_has_priority(self):
        base = Feedback(-3.8, 4., .02, .08, .05, .1, 0., None)
        action = (.15, 1.65)
        self.assertEqual(next_action(action, base, None), (action, 'hold'))
        mixed = replace(base, trimmed_low=-.03, negative_share=.35, safe_rho=-.5)
        out, reason = next_action(action, mixed, base)
        self.assertEqual(reason, 'loosen')
        self.assertAlmostEqual(out[0], .075)
        self.assertAlmostEqual(out[1], 1.35)
        out, reason = next_action((.15, 2.45), replace(mixed, outside=.20), base)
        self.assertEqual((reason, out[1]), ('tighten', 2.5))
