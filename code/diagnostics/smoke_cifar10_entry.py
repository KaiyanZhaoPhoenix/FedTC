"""Offline, one-round startup check through main.run with temporary CIFAR-shaped inputs."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from main import parse_config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), help='Override the device from the config')
    cli = parser.parse_args()
    options = ['--config', cli.config, '--output', cli.output, '--phase', 'check']
    if cli.device is not None:
        options.extend(['--device', cli.device])
    args = parse_config(options)
    if args.dataset != 'Cifar10' or args.algorithm != 'FedTCStep':
        parser.error('Offline smoke input is intended for CIFAR-10 FedTCStep')
    if args.batch_size < 1 or not 0 < args.validation_fraction < 1:
        parser.error("Require a positive batch size and validation fraction in (0,1)")
    rng = np.random.default_rng(7)
    n = args.batch_size + 1
    while n - max(1, round(n * args.validation_fraction)) < args.batch_size:
        n += 1
    with TemporaryDirectory() as folder:
        root = Path(folder) / args.dataset
        for part, size in [('train', n), ('test', 2)]:
            (root / part).mkdir(parents=True)
            for i in range(args.num_clients):
                x = rng.uniform(-1, 1, (size, 3, 32, 32)).astype('float32')
                np.savez(root / part / f'{i}.npz', data=dict(x=x, y=(i + np.arange(size)) % args.num_classes))
        args.data_root = folder
        args.pool_offsets = [0] * 4
        args.dirichlet_alpha = args.partition_seed = args.partition_clients = None
        args.input_source = 'synthetic CIFAR-shaped startup fixture'
        print('Synthetic CIFAR-shaped inputs. Checking one upload round.', flush=True)
        run(args)


if __name__ == '__main__':
    main()
