"""CIFAR inputs must reach the native CNN without mixing test samples into training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset import prepare_cifar10 as cifar
from diagnostics import fit_native_step_control as native
from utils.experiment import vector


class CifarStepControlTests(unittest.TestCase):
    def test_prepare_keeps_official_splits_separate(self):
        class Raw:
            def __init__(self, root, train, download):
                n = 400 if train else 80
                self.targets = np.arange(n) % 10
                self.data = np.full((n, 32, 32, 3), 255 if train else 0, dtype=np.uint8)
                # Encode the example ID across two distinct channels.
                self.data[:, 0, 0, 1] = np.arange(n) % 256
                self.data[:, 0, 0, 2] = np.arange(n) // 256

        with tempfile.TemporaryDirectory() as tmp, patch("torchvision.datasets.CIFAR10", Raw), \
                patch.object(cifar, "CLIENTS", 4):
            root = Path(tmp) / "Cifar10"
            cifar.prepare(Path(tmp), root, 3103, False)
            counts = cifar.check_clients(root, range(4))
            for part, n, pixel in (("train", 400, 1), ("test", 80, -1)):
                self.assertEqual(sum(counts[part]), n)
                ids = []
                for i in range(4):
                    with np.load(root / part / f"{i}.npz", allow_pickle=True) as archive:
                        x = archive["data"].item()["x"]
                    self.assertTrue((x[:, 0] == pixel).all())
                    channels = np.rint((x[:, 1:, 0, 0] + 1) * 127.5).astype(int)
                    ids.extend(channels[:, 0] + 256 * channels[:, 1])
                self.assertEqual(sorted(ids), list(range(n)))
            with self.assertRaisesRegex(ValueError, "already exists"):
                cifar.prepare(Path(tmp), root, 3103, False)

    def test_cifar_probe_uses_cnn_and_private_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "data"
            for part, n in (("train", 24), ("test", 3)):
                folder = data / "Cifar10" / part
                folder.mkdir(parents=True)
                for i in range(72, 96):
                    x = np.full((n, 3, 32, 32), (i - 84) / 16, dtype=np.float32)
                    np.savez(folder / f"{i}.npz", data=dict(x=x, y=(i + np.arange(n)) % 10))
            config = native.step_pool(data, base, 72, "Cifar10")
            world, schedule = native.open_world(config, 42, base / "run")
            self.assertEqual(type(world.global_model).__name__, "CifarNet")
            manifest = json.loads((base / "run/study_manifest.json").read_text())
            self.assertEqual(manifest["action_kind"], "server_step")
            self.assertEqual(manifest["actions"], [0.5, 1.0, 1.5])
            self.assertEqual(tuple(world.private_x.shape), (24 * 6, 3, 32, 32))
            with patch.object(world, "record_test", side_effect=AssertionError("test access during decision")):
                directions, features, gains = native.probe(world, vector(world.global_model), 0, schedule)
            self.assertEqual(features.shape, (3, 12))
            self.assertTrue(np.isfinite(gains).all())
            self.assertGreater(float(directions[0].norm()), 0)
            with torch.no_grad():
                torch.testing.assert_close(world.private_logits(), world.global_model(world.private_x))

    def test_frozen_policy_cannot_cross_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "policy.json").write_text(json.dumps({"rl": {"arm": 1}}))
            self.assertEqual(native.load_rules(out, None)[0], "mnist")
            with self.assertRaisesRegex(ValueError, "requested Cifar10"):
                native.load_rules(out, "Cifar10")
            (out / "policy.json").write_text(json.dumps({"rl": {"arm": 1, "dataset": "Cifar10"}}))
            self.assertEqual(native.load_rules(out, None)[0], "Cifar10")
            (out / "protocol.json").write_text(json.dumps({"dataset": "mnist"}))
            with self.assertRaisesRegex(ValueError, "disagree"):
                native.load_rules(out, None)

    def test_hwc_and_missing_client_are_reported_at_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "train/0.npz"):
                cifar.check_clients(root, [0])
            (root / "train").mkdir()
            np.savez(root / "train/0.npz", data=dict(x=np.zeros((21, 32, 32, 3)), y=np.zeros(21)))
            with self.assertRaisesRegex(ValueError, "NCHW"):
                cifar.check_clients(root, [0])
