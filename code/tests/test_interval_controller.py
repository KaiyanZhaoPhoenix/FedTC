import itertools
import math
import unittest

import torch

from algorithms.fedtc.controller import ControllerConfig, IntervalController
from algorithms.fedtc.forward import macro_state
from algorithms.fedtc.interval import IntervalEstimator, hypergeometric_tail


class IntervalTests(unittest.TestCase):
    def test_hypergeometric_against_enumeration(self):
        population, bad, size = 9, 3, 4
        samples = list(itertools.combinations(range(population), size))
        actual = sum(sum(i < bad for i in sample) > 1 for sample in samples) / len(samples)
        self.assertAlmostEqual(hypergeometric_tail(population, bad, size, 1), actual)

    def test_minority_negative_reports_survive_in_lower_bound(self):
        estimator = IntervalEstimator(10, 10, rho_max=.2, loss_cap=5)
        interval = estimator.estimate([-2., -2., .1, .1, .1, .1, .1, .1, 5., 5.])
        self.assertAlmostEqual(interval.trimmed_low, (-4 + .6) / 8)
        self.assertLess(interval.trimmed_low, 0)

    def test_trimmed_bounds_cover_honest_cohort_for_all_extreme_attacks(self):
        estimator = IntervalEstimator(10, 10, rho_max=.2, loss_cap=5)
        honest = [-4., -2., -1., 0., .1, .2, 1., 2.]
        for malicious in itertools.product([-5., -.7, .3, 5.], repeat=2):
            interval = estimator.estimate(honest + list(malicious))
            mean = sum(honest) / len(honest)
            self.assertLessEqual(interval.trimmed_low, mean)
            self.assertGreaterEqual(interval.trimmed_high, mean)

    def test_rounding_uses_honest_integer_count(self):
        estimator = IntervalEstimator(60, 20, rho_max=.2, eta=.111)
        self.assertEqual(estimator.bad_bound, 7)
        self.assertAlmostEqual(estimator.slack, 5 * math.sqrt(2 * math.log(40) / 13))

    def test_clipping_and_invalid_cohort(self):
        estimator = IntervalEstimator(10, 10)
        interval = estimator.estimate([float('inf')] * 5 + [-float('inf')] * 5)
        self.assertTrue(math.isfinite(interval.low))
        with self.assertRaises(ValueError):
            estimator.estimate([float('nan')] * 10)
        with self.assertRaises(ValueError):
            IntervalEstimator(60, 2, rho_max=.4, delta_bad=.001)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.controller = IntervalController(ControllerConfig(batch_size=2, updates_per_macro=1, warmup_macros=1), seed=3)

    def test_popart_preserves_predictions_and_actions_for_same_raw_state(self):
        controller = self.controller
        raw = torch.randn(5, 8)
        raw[:, 6] = raw[:, 6].abs()
        state = controller.normalize_state(raw)
        action = controller.actor(state).detach()
        networks = [controller.low, controller.high, controller.target_low, controller.target_high]
        values = [(network(state, action) * controller.popart.std + controller.popart.mean).detach() for network in networks]
        controller.update_statistics(7.)
        new_state = controller.normalize_state(raw)
        torch.testing.assert_close(controller.actor(new_state), action, atol=2e-6, rtol=2e-6)
        for network, value in zip(networks, values):
            torch.testing.assert_close(network(new_state, action) * controller.popart.std + controller.popart.mean,
                                       value, atol=2e-6, rtol=2e-6)

    def test_gate_and_executed_action_replay(self):
        controller = self.controller
        state = macro_state([], 0., 0., .1)
        controller.train_steps = 1
        with torch.no_grad():
            controller.low.head.weight.zero_()
            controller.low.head.bias.fill_(-10)
            controller.high.head.weight.zero_()
            controller.high.head.bias.fill_(10)
        action, gate = controller.select_action(state, 3)
        self.assertEqual(action, [0., 1.35])
        self.assertTrue(gate['fallback'])
        controller.remember(state, action, -1., 1., state, False)
        self.assertEqual(controller.replay[-1][1], action)
        with torch.no_grad():
            controller.low.head.bias.fill_(20)
        action, gate = controller.select_action(state, 3)
        self.assertFalse(gate['fallback'])
        self.assertTrue(0 <= action[0] <= .5 and 1.05 <= action[1] <= 1.8)

    def test_terminal_td_target_does_not_bootstrap(self):
        controller = self.controller
        state = macro_state([], 0., 0., 1.)
        with torch.no_grad():
            for network in (controller.low, controller.high):
                network.head.weight.zero_()
                network.head.bias.zero_()
            for network in (controller.target_low, controller.target_high):
                network.head.weight.zero_()
                network.head.bias.fill_(10000.)
        for _ in range(2):
            controller.remember(state, [0., 1.35], -2., 3., state, True)
        losses = controller.train()
        self.assertAlmostEqual(losses['critic_loss_low'], 4.)
        self.assertAlmostEqual(losses['critic_loss_high'], 9.)

    def test_saturated_actor_respects_exact_python_action_endpoints(self):
        controller = self.controller
        controller.train_steps = 1
        with torch.no_grad():
            controller.actor.head.weight.zero_()
            controller.actor.head.bias.fill_(-100)
            controller.low.head.weight.zero_()
            controller.low.head.bias.fill_(10)
            controller.high.head.weight.zero_()
            controller.high.head.bias.zero_()
        action, gate = controller.select_action(macro_state([], 0., 0., .5), 5)
        self.assertFalse(gate['fallback'])
        self.assertGreaterEqual(action[1], 1.05)

    def test_learning_updates_actor_and_both_critics(self):
        controller = self.controller
        before = [p.detach().clone() for p in controller.actor.parameters()]
        state = macro_state([], -1., 2., .5)
        for i in range(2):
            controller.remember(state, [.2, 1.5], -1., 2., state, i == 1)
        losses = controller.train()
        self.assertTrue(all(math.isfinite(value) for value in losses.values()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, controller.actor.parameters())))
        self.assertEqual(controller.train_steps, 1)

    def test_checkpoint_restores_replay_rng_and_optimizer(self):
        controller = self.controller
        state = macro_state([], -1., 2., .5)
        for _ in range(3):
            controller.remember(state, [.2, 1.5], -1., 2., state, False)
        controller.update_statistics(.5)
        controller.train()
        from copy import deepcopy
        restored = IntervalController(controller.config, seed=99)
        restored.load_state_dict(deepcopy(controller.state_dict()))
        self.assertEqual(controller.train(), restored.train())
        for a, b in zip(controller.actor.parameters(), restored.actor.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
