import copy
import unittest

import torch

from algorithms.fedtc.sac import GaussianActor, SACConfig, SACController, backup_targets
from algorithms.fedtc.forward import CONSERVATIVE_ACTION
from tests import test_integration as integration


class SACTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make(self, gate=False):
        return SACController(SACConfig(batch_size=2, updates_per_macro=1, warmup_macros=0, use_gate=gate), seed=1)

    def populate(self, controller):
        for i in range(3):
            state = torch.tensor([.4, .5, .1, 0., .9, -.5, 1., i/4])
            controller.remember(state, [.2, 1.4], -.2, .3, state+.01, i == 2)

    def test_soft_entropy_never_enters_gate_targets_and_terminal_no_bootstrap(self):
        x = lambda v: torch.tensor([[float(v)]])
        args = [x(2), x(1), x(3), x(1), x(8), x(-4), x(6), x(10), .5]
        soft1, low1, high1 = backup_targets(*args, .1)
        soft2, low2, high2 = backup_targets(*args, 2.)
        self.assertNotEqual(float(soft1), float(soft2))
        self.assertEqual(float(low1), 4.)
        self.assertEqual(float(high1), 8.)
        torch.testing.assert_close(low1, low2)
        torch.testing.assert_close(high1, high2)
        args[3] = x(0)
        for actual, expected in zip(backup_targets(*args, 2.), [2, 1, 3]):
            self.assertEqual(float(actual), expected)

    def test_gaussian_action_bounds_and_log_jacobian(self):
        actor = GaussianActor(8)
        for p in actor.parameters():
            p.data.zero_()
        action, logp = actor.sample(torch.zeros(1, 8), torch.Generator().manual_seed(1), deterministic=True)
        torch.testing.assert_close(action, torch.tensor([[.25, 1.425]]))
        self.assertAlmostEqual(float(logp), -torch.log(torch.tensor(2*torch.pi)).item(), places=5)
        action, logp = actor.sample(torch.zeros(1000, 8), torch.Generator().manual_seed(9))
        self.assertTrue(bool(((action[:, 0] >= 0) & (action[:, 0] <= .5)).all()))
        self.assertTrue(bool(((action[:, 1] >= 1.05) & (action[:, 1] <= 1.8)).all()))
        self.assertTrue(bool(torch.isfinite(logp).all()))
        self.assertGreater(float(action[:, 0].std()), .05)

    def test_gate_toggle_and_replay_store_actual_executed_action(self):
        a, b = self.make(False), self.make(True)
        for controller in (a, b):
            controller.train_steps = 1
            for net, value in ((controller.low, -10.), (controller.high, 10.)):
                net.head.weight.data.zero_()
                net.head.bias.data.fill_(value)
        state = torch.zeros(8)
        action_a, gate_a = a.select_action(state, 3)
        action_b, gate_b = b.select_action(state, 3)
        self.assertEqual(gate_a['candidate'], gate_b['candidate'])
        self.assertFalse(gate_a['fallback'])
        self.assertTrue(gate_b['fallback'])
        self.assertEqual(action_b, list(CONSERVATIVE_ACTION))
        self.assertNotEqual(action_a, action_b)
        b.remember(state, action_b, -1, 1, state, False)
        self.assertEqual(b.replay[-1][1], action_b)

    def test_gate_variants_train_identically_given_identical_data(self):
        a, b = self.make(False), self.make(True)
        for controller in (a, b):
            self.populate(controller)
        self.assertEqual(a.train(), b.train())
        for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)

    def test_checkpoint_restores_learning_and_action_rng_without_global_rng_use(self):
        a = self.make()
        self.populate(a)
        before = torch.get_rng_state().clone()
        a.train()
        a.select_action(torch.zeros(8), 4)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        b = self.make()
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        self.assertEqual(a.select_action(torch.zeros(8), 4), b.select_action(torch.zeros(8), 4))
        self.assertEqual(a.train(), b.train())
        for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
            torch.testing.assert_close(pa, pb, rtol=0, atol=0)


class SACIntegrationTests(unittest.TestCase):
    setUp = integration.FrameworkTests.setUp
    tearDown = integration.FrameworkTests.tearDown
    args = integration.FrameworkTests.args
    def test_sac_variants_no_root_and_utility_and_audit_logging(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch
        from servers.serverFedTC import FedTC
        for algorithm in ('FedTCSAC', 'FedTCSACGate'):
            server = FedTC(self.args(algorithm, extra=['--audit-validation', '--audit-shadow-every', '1']))
            with patch.object(server, 'test_server', side_effect=AssertionError('Root data used')):
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = server.train()
            self.assertEqual(summary['controller_train_steps'], 2)
            rounds = [json.loads(line) for line in (server.output/'rounds.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['utilities'] is not None for r in rounds))
            self.assertIsNotNone(rounds[0]['shadow_harness_only'])
            self.assertTrue((server.output/'validation_audit.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
