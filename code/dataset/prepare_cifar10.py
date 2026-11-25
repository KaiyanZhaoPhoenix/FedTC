"""Prepare CIFAR-10 client NPZs from the official Python batches, or check an existing partition."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np


CLIENTS = 100
ALPHA = 0.5


def check_clients(root, ids):
    """Validate the FedAA client format before starting expensive worker processes."""
    counts = {}
    for part in ("train", "test"):
        counts[part] = []
        for i in ids:
            path = root / part / f"{i}.npz"
            if not path.is_file():
                raise ValueError(f"Missing client partition: {path}")
            try:
                with np.load(path, allow_pickle=True) as archive:
                    data = archive["data"].item()
                    x, y = np.asarray(data["x"]), np.asarray(data["y"])
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise ValueError(f"{path}: expected an NPZ containing data={{'x': ..., 'y': ...}}") from exc
            if x.ndim != 4 or x.shape[1:] != (3, 32, 32) or y.shape != (x.shape[0],):
                raise ValueError(f"{path}: expected NCHW (N,3,32,32) and labels (N,), got {x.shape}, {y.shape}")
            if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
                raise ValueError(f"{path}: supply finite, preprocessed floating-point images")
            if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y > 9).any():
                raise ValueError(f"{path}: expected integer labels 0 through 9")
            if len(y) < (2 if part == "train" else 1):
                raise ValueError(f"{path}: too few samples for the private train/validation or test split")
            counts[part].append(len(y))
    return counts


def allocate(labels, proportions, rng):
    labels = np.asarray(labels)
    clients = [[] for _ in range(CLIENTS)]
    for label, probabilities in enumerate(proportions):
        indices = rng.permutation(np.flatnonzero(labels == label))
        sizes = rng.multinomial(len(indices), probabilities)
        for client, chunk in zip(clients, np.split(indices, sizes.cumsum()[:-1])):
            client.extend(chunk)
    return [rng.permutation(client) for client in clients]


def prepare(raw_root, destination, seed, download):
    from torchvision.datasets import CIFAR10

    if destination.exists():
        raise ValueError(f"{destination} already exists; use --verify-only or a different --data-root")
    train = CIFAR10(str(raw_root), train=True, download=download)
    test = CIFAR10(str(raw_root), train=False, download=False)
    train_rng, test_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    # Draw only from training labels until each client can run a batch of 16 after its holdout.
    for _ in range(1000):
        proportions = train_rng.dirichlet(np.full(CLIENTS, ALPHA), size=10)
        train_ids = allocate(train.targets, proportions, train_rng)
        if min(map(len, train_ids)) >= 21:
            break
    else:
        raise ValueError("Could not draw a partition with a full local SGD batch for every client")
    for _ in range(1000):
        test_ids = allocate(test.targets, proportions, test_rng)
        if all(map(len, test_ids)):
            break
    else:
        raise ValueError("Could not draw nonempty client test sets")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        staged = Path(tmp) / "Cifar10"
        for part, source, clients in (("train", train, train_ids), ("test", test, test_ids)):
            (staged / part).mkdir(parents=True)
            labels = np.asarray(source.targets, dtype=np.int64)
            for i, indices in enumerate(clients):
                x = source.data[indices].transpose(0, 3, 1, 2).astype(np.float32)
                x = np.ascontiguousarray(x / 127.5 - 1.0)
                np.savez_compressed(staged / part / f"{i}.npz", data=dict(x=x, y=labels[indices]))
        meta = dict(dataset="Cifar10", clients=CLIENTS, partition="class-wise Dirichlet",
                    alpha=ALPHA, seed=seed, train_source="official train", test_source="official test",
                    normalization="float32 CHW, pixel / 127.5 - 1")
        (staged / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
        staged.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Parent directory of Cifar10/")
    parser.add_argument("--raw-root", type=Path, help="Parent directory of cifar-10-batches-py/")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download official batches")
    parser.add_argument("--verify-only", action="store_true", help="Validate clients 0–95 used by step control")
    parser.add_argument("--seed", type=int, default=3103)
    args = parser.parse_args()
    destination = args.data_root.expanduser().resolve() / "Cifar10"
    if args.verify_only and (args.download or args.raw_root):
        parser.error("--verify-only reads prepared clients; omit --download and --raw-root")
    if not args.verify_only and (args.raw_root is None or args.seed < 0):
        parser.error("Preparation requires --raw-root and a nonnegative --seed")
    try:
        if not args.verify_only:
            prepare(args.raw_root.expanduser().resolve(), destination, args.seed, args.download)
        counts = check_clients(destination, range(96 if args.verify_only else CLIENTS))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{destination}: {len(counts['train'])} clients, {sum(counts['train'])} train, {sum(counts['test'])} test")


if __name__ == "__main__":
    main()
