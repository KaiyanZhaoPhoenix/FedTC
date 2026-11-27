"""Partition official CIFAR-10, MNIST or Fashion-MNIST train/test sets into FedAA client files."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np


CLIENTS = 100
ALPHA = 0.5


def check_clients(root, ids, dataset="Cifar10"):
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
            shape = (3, 32, 32) if dataset == "Cifar10" else (1, 28, 28)
            if x.ndim != 4 or x.shape[1:] != shape or y.shape != (x.shape[0],):
                raise ValueError(f"{path}: expected NCHW with sample shape {shape} and labels (N,), got {x.shape}, {y.shape}")
            if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
                raise ValueError(f"{path}: supply finite, preprocessed floating-point images")
            if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y > 9).any():
                raise ValueError(f"{path}: expected integer labels 0 through 9")
            if len(y) < (2 if part == "train" else 1):
                raise ValueError(f"{path}: too few samples for the private train/validation or test split")
            counts[part].append(len(y))
    return counts


def verify_alpha(root, expected):
    metadata = Path(root) / "config.json"
    if not metadata.is_file():
        raise ValueError(f"Cannot verify Dirichlet alpha: missing {metadata}")
    meta = json.loads(metadata.read_text())
    if meta.get("partition", "dir") not in ("dir", "class-wise Dirichlet") or meta.get("non_iid") is False:
        raise ValueError(f"{metadata} does not describe a Dirichlet partition")
    actual = meta.get("alpha")
    if not np.isfinite(expected) or expected <= 0 or actual is None or not np.isclose(float(actual), expected, rtol=1e-10, atol=0):
        raise ValueError(f"Partition alpha is {actual}, requested {expected} in {metadata}")


def allocate(labels, proportions, rng):
    labels = np.asarray(labels)
    clients = [[] for _ in range(proportions.shape[1])]
    for label, probabilities in enumerate(proportions):
        indices = rng.permutation(np.flatnonzero(labels == label))
        sizes = rng.multinomial(len(indices), probabilities)
        for client, chunk in zip(clients, np.split(indices, sizes.cumsum()[:-1])):
            client.extend(chunk)
    return [rng.permutation(client) for client in clients]


def prepare(raw_root, destination, seed, download, clients=None, alpha=ALPHA, min_train=21, dataset="Cifar10"):
    from torchvision.datasets import CIFAR10, FashionMNIST, MNIST

    if destination.exists():
        raise ValueError(f"{destination} already exists; use --verify-only or a different --data-root")
    clients = CLIENTS if clients is None else clients
    if clients < 1 or not np.isfinite(alpha) or alpha <= 0 or min_train < 2:
        raise ValueError("Require positive clients and alpha, min_train >= 2")
    constructor = {"Cifar10": CIFAR10, "mnist": MNIST, "fashionmnist": FashionMNIST}[dataset]
    train = constructor(str(raw_root), train=True, download=download)
    test = constructor(str(raw_root), train=False, download=False)
    train_rng, test_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    if len(train.targets) < clients * min_train or len(test.targets) < clients:
        raise ValueError("Too few examples for the requested client counts and partition minimum")
    # Condition only on training data when enforcing the minimum local training budget.
    for _ in range(1000):
        proportions = train_rng.dirichlet(np.full(clients, alpha), size=10)
        train_ids = allocate(train.targets, proportions, train_rng)
        if min(map(len, train_ids)) >= min_train:
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
        staged = Path(tmp) / dataset
        for part, source, partitions in (("train", train, train_ids), ("test", test, test_ids)):
            (staged / part).mkdir(parents=True)
            labels = np.asarray(source.targets, dtype=np.int64)
            for i, indices in enumerate(partitions):
                raw = np.asarray(source.data)[indices]
                x = (raw.transpose(0, 3, 1, 2) if dataset == "Cifar10" else raw[:, None]).astype(np.float32)
                x = np.ascontiguousarray(x / 127.5 - 1.0)
                np.savez_compressed(staged / part / f"{i}.npz", data=dict(x=x, y=labels[indices]))
        meta = dict(dataset=dataset, clients=clients, min_train=min_train, partition="class-wise Dirichlet",
                    alpha=alpha, seed=seed, train_source="official train", test_source="official test",
                    normalization="float32 CHW, pixel / 127.5 - 1")
        (staged / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
        staged.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Cifar10", "mnist", "fashionmnist"))
    parser.add_argument("--config", type=Path, help="Read partition size, seed, alpha and SGD batch requirements from the training JSON")
    parser.add_argument("--data-root", required=True, type=Path, help="Parent directory of the dataset partitions")
    parser.add_argument("--raw-root", type=Path, help="Torchvision download root (parent of cifar-10-batches-py for CIFAR)")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download official batches")
    parser.add_argument("--verify-only", action="store_true", help="Validate clients 0–95 used by step control")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-clients", type=int, help="Number of native client partitions")
    parser.add_argument("--dirichlet-alpha", type=float)
    parser.add_argument("--min-train", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()) if args.config else {}
    dataset = args.dataset or cfg.get("dataset", "Cifar10")
    if dataset not in ("Cifar10", "mnist", "fashionmnist"):
        parser.error(f"No official batch preparer for {dataset}; supply native client NPZ files")
    if args.num_clients is None:
        args.num_clients = cfg.get("partition_clients")
    if args.seed is None:
        args.seed = cfg.get("partition_seed", 3103)
    if args.dirichlet_alpha is None:
        args.dirichlet_alpha = cfg.get("dirichlet_alpha")
    if args.min_train is None:
        batch, fraction = cfg.get("batch_size", 16), cfg.get("validation_fraction", .25)
        if batch < 1 or not 0 < fraction < 1:
            parser.error("A positive batch and a validation fraction in (0,1) are required")
        args.min_train = max(2, batch)
        while args.min_train - max(1, round(args.min_train * fraction)) < batch:
            args.min_train += 1
    destination = args.data_root.expanduser().resolve() / dataset
    if args.verify_only and (args.download or args.raw_root):
        parser.error("--verify-only reads prepared clients; omit --download and --raw-root")
    if not args.verify_only and (args.raw_root is None or args.seed < 0):
        parser.error("Preparation requires --raw-root and a nonnegative --seed")
    try:
        if not args.verify_only:
            prepare(args.raw_root.expanduser().resolve(), destination, args.seed, args.download, args.num_clients,
                    ALPHA if args.dirichlet_alpha is None else args.dirichlet_alpha, args.min_train, dataset)
        if args.dirichlet_alpha is not None:
            verify_alpha(destination, args.dirichlet_alpha)
        count = args.num_clients if args.num_clients is not None else (96 if args.verify_only else CLIENTS)
        if count < 1:
            raise ValueError("--num-clients must be positive")
        counts = check_clients(destination, range(count), dataset)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{destination}: {len(counts['train'])} clients, {sum(counts['train'])} train, {sum(counts['test'])} test")


if __name__ == "__main__":
    main()
