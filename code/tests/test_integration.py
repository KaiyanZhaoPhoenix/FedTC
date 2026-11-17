import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from algorithms.baselines import aggregate
from algorithms.fedtc.controller import ControllerConfig, IntervalController
from clients.clientFedAA import clientFedAA
from clients.clientFedTC import clientFedTC
from clients.clientbase import Client
from main import parser, prepare, ROOT
from run_comparison import verify
from servers.serverbase import Server
from servers.serverFedTC import FedTC
from servers.serverShared import SharedServer
from utils import data_utils
from utils.experiment import attack_model, set_vector, vector
from utils.holdout import split_indices


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        dataset = self.root / 'data' / 'mnist'
        (dataset / 'train').mkdir(parents=True)
        (dataset / 'test').mkdir()
        rng = np.random.default_rng(43)
        for client in range(4):
            for part, size in [('train', 24), ('test', 10)]:
                np.savez_compressed(dataset / part / f'{client}.npz', data={
                    'x': rng.normal(size=(size, 2)).astype('float32'),
                    'y': np.arange(size) % 2})
        # No server_test.npz deliberately: FedTC and baselines must not need it.

    def tearDown(self):
        data_utils.configure_data_root(ROOT / 'dataset')
        self.temp.cleanup()

    def args(self, algorithm='FedTC', output=None, extra=()):
        args = parser().parse_args([
            '--algorithm', algorithm, '--data-root', str(self.root / 'data'),
            '--output', str(self.root / (output or algorithm)),
            '--num_clients', '4', '--num_sel_clients', '3', '--aggre_num', '1',
            '--num_classes', '2', '--global_rounds', '4', '--local_epochs', '1',
            '--batch_size', '4', '--eval-clients', '4', '--rho-max', '0',
            '--macro-length', '2',
            '--ad_ratio', '0', '--attack', 'none', '--buffer_size', '16',
            '--tc-batch-size', '1', '--warmup-macros', '0', '--tc-updates', '1',
            '--evaluate-local', *extra])
        prepare(args)
        torch.manual_seed(args.seed)
        args.model = torch.nn.Linear(2, 2)
        return args

    def test_inherits_upstream_training_and_metrics(self):
        self.assertIs(clientFedTC.train, clientFedAA.train)
        self.assertIs(clientFedTC.test_metrics, Client.test_metrics)
        self.assertIs(SharedServer.evaluate, Server.evaluate)
        self.assertIs(SharedServer.send_models, Server.send_models)

    def test_holdout_is_disjoint_complete_stable(self):
        train, val = split_indices(31, .1, 2026, 3)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train+val), list(range(31)))
        self.assertEqual((train, val), split_indices(31, .1, 2026, 3))
        self.assertNotEqual(val, split_indices(31, .1, 2026, 4)[1])

    def test_sgd_uses_only_training_split_with_native_clipping(self):
        args = self.args()
        client = clientFedTC(args, 0, 24, 10)
        reference = copy.deepcopy(client.model)
        optimizer = torch.optim.SGD(reference.parameters(), lr=args.local_learning_rate)
        for x, y in client.load_train_data():
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(reference(x), y).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 60)
            optimizer.step()
        client.train()
        torch.testing.assert_close(vector(client.model), vector(reference), atol=0, rtol=0)
        train_examples = client.load_train_data().dataset
        for x, _ in train_examples:
            self.assertFalse(any(torch.equal(x, client._full_train_data[i][0]) for i in client.validation_indices))

    def test_local_rng_independent_of_controller_and_first_round_updates_match(self):
        args = self.args('FedAvg')
        first = SharedServer(args)
        second = SharedServer(self.args('FedTCFixed'))
        first.send_models()
        second.send_models()
        first.train_clients(first.clients[:3], 0, vector(first.global_model))
        torch.randn(103)  # Stand-in for controller draws.
        second.train_clients(second.clients[:3], 0, vector(second.global_model))
        for one, two in zip(first.clients, second.clients):
            torch.testing.assert_close(vector(one.model), vector(two.model), atol=0, rtol=0)

    def test_fedtc_never_reads_root_data_and_feedback_never_reads_test(self):
        server = FedTC(self.args())
        with patch.object(server, 'test_server', side_effect=AssertionError('Clean root was used')):
            with contextlib.redirect_stdout(io.StringIO()):
                summary = server.train()
        self.assertEqual(summary['controller_train_steps'], 2)
        self.assertEqual(len(server.rs_test_acc), 4)
        self.assertEqual(len(server.rs_test_auc), 4)
        with h5py.File(server.output / 'metrics.h5') as metrics:
            self.assertEqual(list(metrics['round'][:]), [1, 2, 3, 4])
            self.assertEqual(metrics.attrs['phase'], 'global')
        with patch.object(server.clients[0], 'load_test_data', side_effect=AssertionError('Test leakage')):
            self.assertAlmostEqual(server.clients[0].report_improvement(server.global_model, server.global_model, 5), 0)
        for client in server.clients:
            torch.testing.assert_close(vector(client.model), vector(server.global_model), atol=0, rtol=0)
        macros = [json.loads(line) for line in (server.output / 'macros.jsonl').read_text().splitlines()]
        self.assertEqual(len(macros), 2)
        self.assertEqual([m['round_end'] for m in macros], [2, 4])
        independent_rng = np.random.default_rng(np.random.SeedSequence([1, 202]))
        self.assertEqual([m['evaluators'] for m in macros],
                         [independent_rng.choice(4, 4, replace=False).tolist() for _ in macros])

    def test_comparison_checks_provenance_and_detects_schedule_change(self):
        algorithms = ['FedTC', 'FedAvg']
        for name in algorithms:
            server = FedTC(self.args(name)) if name == 'FedTC' else SharedServer(self.args(name))
            with contextlib.redirect_stdout(io.StringIO()):
                server.train()
        self.assertTrue(verify(self.root, algorithms)['protocol_verified'])
        manifest_path = self.root / 'FedAvg' / 'manifest.json'
        original_manifest = manifest_path.read_text()
        altered = json.loads(original_manifest)
        altered['args']['local_learning_rate'] *= 2
        manifest_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(RuntimeError, 'configuration'):
            verify(self.root, algorithms)
        manifest_path.write_text(original_manifest)
        path = self.root / 'FedAvg' / 'rounds.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['upload_clients'].reverse()
        path.write_text('\n'.join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(RuntimeError, 'schedules'):
            verify(self.root, algorithms)

    def test_sampled_training_leaves_other_clients_untrained(self):
        server = SharedServer(self.args('FedAvg', extra=['--training-cohort', 'sampled']))
        server.send_models()
        before = vector(server.global_model)
        server.train_clients(server.clients[:2], 0, before)
        for client in server.clients[2:]:
            torch.testing.assert_close(vector(client.model), before, rtol=0, atol=0)
        self.assertFalse(torch.equal(vector(server.clients[0].model), before))

    def test_original_server_npz_loading_and_native_parameter_broadcast(self):
        target = self.root / 'data' / 'mnist' / 'test' / 'server_test.npz'
        np.savez_compressed(target, x=np.ones((5, 2)), y=np.arange(5) % 2)
        self.args()
        root_data = data_utils.read_data('mnist', None, False)
        self.assertEqual(len(root_data), 5)
        self.assertEqual(root_data[0][0].dtype, torch.float32)

    def test_upstream_and_delta_sign_flip_are_distinct_and_deterministic(self):
        client = clientFedTC(self.args(), 0, 24, 10)
        start = vector(client.model)
        set_vector(client.model, start + 1)
        attack_model(client, start, 'delta_sign_flip', 1, 0)
        torch.testing.assert_close(vector(client.model), start - 10)
        set_vector(client.model, start + 1)
        attack_model(client, start, 'sign_flip', 1, 0)
        ratio = vector(client.model) / (start + 1)
        self.assertTrue(bool((ratio > 0).all()))
        output = vector(client.model)
        set_vector(client.model, start + 1)
        np.random.randn(100)
        attack_model(client, start, 'sign_flip', 1, 0)
        torch.testing.assert_close(vector(client.model), output, atol=0, rtol=0)


class BaselineTests(unittest.TestCase):
    def test_sample_weighted_fedavg_and_coordinate_trim(self):
        points = torch.tensor([[1., 3.], [2., 2.], [10., -10.]])
        delta, weights = aggregate('FedAvg', points, [1, 2, 1], 0, 1)
        torch.testing.assert_close(delta, torch.tensor([3.75, -.75]))
        delta, weights = aggregate('TrimmedMean', points, [1]*3, 1, 1)
        torch.testing.assert_close(delta, torch.tensor([2., 2.]))
        self.assertIsNone(weights)

    def test_multikrum_rejects_remote_outlier_and_invalid_bound(self):
        points = torch.tensor([[0., 0.], [.1, 0.], [0., .1], [.1, .1], [.05, .05], [100., 100.]])
        delta, weights = aggregate('MultiKrum', points, [1]*6, 1, 3)
        self.assertEqual(float(weights[-1]), 0.)
        self.assertLess(float(delta.norm()), .2)
        with self.assertRaises(ValueError):
            aggregate('MultiKrum', points, [1]*6, 2, 1)

    def test_explicit_consistent_popart_target_includes_mean_compensation(self):
        controller = IntervalController(ControllerConfig(td_mode='consistent', gamma=.5, batch_size=1, updates_per_macro=1))
        controller.popart.mean, controller.popart.std = 1.5, 2.
        for network in (controller.low, controller.high, controller.target_low, controller.target_high):
            for parameter in network.parameters():
                parameter.data.zero_()
        controller.target_low.head.bias.data.fill_(2.)
        controller.target_high.head.bias.data.fill_(3.)
        controller.remember(torch.zeros(8), [0., 1.35], 4., 6., torch.zeros(8), False)
        result = controller.train()
        self.assertAlmostEqual(result['critic_loss_low'], 2.625**2)
        self.assertAlmostEqual(result['critic_loss_high'], 4.125**2)


if __name__ == '__main__':
    unittest.main()
