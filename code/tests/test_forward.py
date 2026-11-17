import math
import unittest

import torch

from algorithms.fedtc.forward import (
    EPS_SCALE,
    SIGMA,
    calibrate,
    geometric_median,
    macro_state,
    rank_normalize,
    spearman,
)


class ForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)

    def test_known_geometric_median(self):
        points = torch.tensor([[0., 0.], [2., 0.], [1., math.sqrt(3)]], dtype=torch.float64)
        torch.testing.assert_close(geometric_median(points), points.mean(0), atol=1e-8, rtol=0)

    def test_coincident_points_and_single_client(self):
        for points in (torch.ones(5, 7), torch.zeros(1, 3)):
            result = calibrate(points, torch.ones(len(points)), (.5, 1.8))
            torch.testing.assert_close(result.delta, points[0])
            self.assertFalse(result.fallback)
            self.assertAlmostEqual(float(result.weights.sum()), 1.)

    def test_average_ties_and_fake_utility_rank(self):
        ranked = rank_normalize(torch.tensor([0., 0., 1., 1e6]))
        torch.testing.assert_close(ranked, torch.tensor([1/6, 1/6, 2/3, 1.]))
        self.assertEqual(spearman(torch.ones(5), torch.arange(5.)), 0.)

    def test_geometric_median_resists_outlier(self):
        points = torch.tensor([[-1., 0.], [0., 0.], [1., 0.], [1000., 0.], [0., .01]])
        median = geometric_median(points, iterations=200)
        self.assertLess(float(median.norm()), .1)

    def test_non_collapse_and_convex_bound(self):
        for count in (1, 2, 7, 20):
            updates = torch.randn(count, 31, dtype=torch.float64)
            updates[-1] *= 100
            utilities = torch.rand(count, dtype=torch.float64)
            for lam in (0., .5):
                for tau in (1.05, 1.8):
                    result = calibrate(updates, utilities, (lam, tau))
                    central = result.deviations <= 1
                    self.assertGreaterEqual(int(central.sum()), math.ceil(count / 2))
                    self.assertTrue((result.credibility[central] > .059).all())
                    self.assertAlmostEqual(float(result.weights.sum()), 1.)
                    self.assertLess(float((result.delta - result.anchor).norm()), result.cutoff_radius)
                    self.assertTrue((result.credibility[result.distances >= result.cutoff_radius] == 0).all())

    def test_rotation_and_permutation_equivariance(self):
        x = torch.randn(13, 5, dtype=torch.float64)
        utility = torch.tensor([1.] * 6 + [0.] * 7, dtype=torch.float64)
        rotation = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64)).Q
        order = torch.randperm(13)
        first = calibrate(x, utility, (.2, 1.35))
        second = calibrate(x[order] @ rotation, utility[order], (.2, 1.35))
        torch.testing.assert_close(second.delta, first.delta @ rotation, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(second.weights, first.weights[order], atol=1e-8, rtol=1e-6)

    def test_calibration_matches_scalar_formula(self):
        x, utility = torch.randn(10, 9, dtype=torch.float64), torch.rand(10, dtype=torch.float64)
        result = calibrate(x, utility, (.3, 1.4))
        for i in range(10):
            expected = max(0., .7 + .3 * float(result.utilities[i]) -
                           1.4 * math.tanh(float(result.distances[i]) / (SIGMA * (result.scale + EPS_SCALE))))
            self.assertAlmostEqual(float(result.credibility[i]), expected)

    def test_invalid_values_fail_explicitly(self):
        for action in ((.6, 1.4), (.1, 1.), (float('nan'), 1.4)):
            with self.assertRaises(ValueError):
                calibrate(torch.zeros(2, 3), torch.ones(2), action)
        with self.assertRaises(ValueError):
            calibrate(torch.tensor([[float('nan')]]), torch.ones(1), (0., 1.35))

    def test_state_uses_mean_raw_scale_before_log(self):
        geometries = [torch.tensor([1., 2., .3, .2, .8]), torch.tensor([3., 4., .5, .4, 1.])]
        state = macro_state(geometries, -2., 3., .5)
        self.assertEqual(state.shape, (8,))
        self.assertAlmostEqual(float(state[0]), math.log(3), places=6)
        self.assertEqual(float(state[6]), 5.)
