import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset.prepare_cifar10 import verify_alpha
from main import parse_config, prepare, ROOT, run
from servers.serverFedTCStep import FedTCStep
from servers.step_experiment import paired_summary, write_seed_report
from utils.data_utils import configure_data_root


class HeldStatisticsTests(unittest.TestCase):
    def test_report_uses_final_percentage_per_seed_and_sample_variance(self):
        seeds = [16212, 16213, 16214]
        rows = {
            'rl': [dict(seed=s, identity=[s], metrics=dict(accuracy=a, worst10=w))
                   for s, a, w in zip(seeds, [.46, .43, .49], [.25, .31, .34])],
            'fixed': [dict(seed=s, identity=[s], metrics=dict(accuracy=.45, worst10=.30)) for s in seeds],
        }
        summary = paired_summary(rows)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'seed_statistics.csv'
            write_seed_report(path, summary)
            with path.open(newline='') as stream:
                report = {row['method']: row for row in csv.DictReader(stream)}
        self.assertEqual(int(report['rl']['n_seeds']), 3)
        self.assertAlmostEqual(float(report['rl']['accuracy_mean_pct']), 46.)
        self.assertAlmostEqual(float(report['rl']['accuracy_sample_variance_pp2']), 9.)
        self.assertAlmostEqual(float(report['rl']['worst10_mean_pct']), 30.)
        self.assertAlmostEqual(float(report['rl']['worst10_sample_variance_pp2']), 21.)
        self.assertAlmostEqual(float(report['fixed']['accuracy_sample_variance_pp2']), 0.)
        self.assertAlmostEqual(summary['fixed']['rl']['accuracy']['diff'], 1.)
        one = paired_summary({'rl': rows['rl'][:1]})
        self.assertIsNone(one['rl']['rl']['accuracy']['sample_variance'])


class NativeEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.base = Path(self.temp.name)
        root = self.base / 'Cifar10'
        rng = np.random.default_rng(12)
        for part, n in [('train', 12), ('test', 4)]:
            (root / part).mkdir(parents=True)
            for i in range(4):
                np.savez(root / part / f'{i}.npz', data=dict(
                    x=rng.normal(size=(n, 3, 32, 32)).astype('float32'), y=(np.arange(n)+i) % 10))
        (root / 'config.json').write_text(json.dumps(dict(clients=4, alpha=.5, seed=7, partition='class-wise Dirichlet')))
        cfg = json.loads((ROOT / 'configs/cifar10_resnet_100r.json').read_text())
        cfg.update(device='cpu', data_root=str(self.base), partition_clients=4, partition_seed=7, num_clients=4,
                   num_sel_clients=2, global_rounds=3, macro_length=2, local_steps=5, batch_size=3,
                   pool_offsets=[0]*4, fit_seeds=[1], tune_seeds=[2], held_seeds=[3], learning_rate_decay=True)
        self.config = self.base / 'train.json'
        self.config.write_text(json.dumps(cfg))

    def tearDown(self):
        configure_data_root(ROOT / 'dataset')
        self.temp.cleanup()

    def args(self, phase, *extra):
        return parse_config(['--config', str(self.config), '--output', str(self.base / phase),
                             '--phase', phase, *extra])

    def test_step_budget_and_sibling_isolation_with_batchnorm(self):
        server = FedTCStep(prepare(self.args('check', '--cohort-schedule', 'uniform', '--clients-per-round', '3')))
        self.assertTrue(all(len(set(row)) == 3 for row in server.schedule))
        start = server.initial
        with patch.object(server, 'test', side_effect=AssertionError('test metric access during decision')):
            first, _, _ = server.probe(start, 0)
            self.assertEqual(server.costs['local_optimizer_steps'], 3*5)
            self.assertEqual(server.costs['uploads'], 3)
            end, _ = server.commit(start, 0, 2, first)
            server.commit(start, 0, 0, first)
            replay, _ = server.commit(start, 0, 2, first)
        torch.testing.assert_close(end.params, replay.params, rtol=0, atol=0)
        for a, b in zip(end.buffers, replay.buffers):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertNotEqual([s[1]['last_epoch'] for s in end.optimizers],
                            [s[1]['last_epoch'] for s in start.optimizers])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(end.buffers, start.buffers)))
        self.assertEqual(end.optimizers, replay.optimizers)

    def test_phases_tail_policy_binding_and_checkpoint(self):
        run(self.args('dev'))
        policy = str(self.base / 'dev/policy.json')
        held = run(self.args('held', '--policy', policy))
        result = run(self.args('deploy', '--policy', policy, '--seed', '3'))
        with (self.base / 'held/seed_statistics.csv').open(newline='') as stream:
            report = {row['method']: row for row in csv.DictReader(stream)}
        self.assertEqual(report['rl']['n_seeds'], '1')
        self.assertEqual(report['rl']['accuracy_sample_variance_pp2'], '')
        self.assertEqual(len(result['actions']), 2)
        self.assertEqual(result['taus'][0], 1.15)
        self.assertAlmostEqual(held['incumbent']['rl']['accuracy']['mean'], 100*result['metrics']['accuracy'])
        restored = FedTCStep(prepare(self.args('check', '--seed', '3')))
        state = torch.load(self.base / 'deploy/deploy/3/model.pt', weights_only=True)
        restored.global_model.load_state_dict(state)
        counts = []
        for c in restored.clients:
            c.model.load_state_dict(state)
            counts.append(c.test_metrics())
        self.assertEqual(result['metrics']['accuracy'], sum(v[0] for v in counts)/sum(v[1] for v in counts))
        costs = json.loads((self.base / 'deploy/deploy/3/costs.json').read_text())
        self.assertEqual(costs['uploads'], 6)  # two tau proposals reuse the single terminal upload
        with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
            run(self.args('deploy', '--policy', policy, '--global-rounds', '4'))
        with self.assertRaisesRegex(ValueError, 'alpha'):
            verify_alpha(self.base / 'Cifar10', .1)
