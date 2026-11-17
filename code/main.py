#!/usr/bin/env python
"""FedAA-native entry extended with FedTC and shared baselines."""
import argparse
import copy
from datetime import datetime
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from DDPG.environment import weights_init
from servers.serverFedAA import FedAA
from servers.serverShared import SharedServer
from servers.serverFedTC import FedTC
from utils.data_utils import configure_data_root
from utils.models import OneHiddenLayerFc, TwoHiddenLayerFc, CifarNet, Logistic, ResNet2

ALGORITHMS = ('FedTC', 'FedAA', 'FedAvg', 'RFA', 'TrimmedMean', 'MultiKrum', 'FedTCFixed', 'FedTCSAC', 'FedTCSACGate')
ROOT = Path(__file__).resolve().parent


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false.')


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('-algo', '--algorithm', choices=ALGORITHMS, default='FedTC')
    p.add_argument('--protocol', choices=('shared', 'upstream'), default='shared')
    p.add_argument('-go', '--goal', default='test')
    p.add_argument('-dev', '--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('-did', '--device_id', default='0')
    p.add_argument('-data', '--dataset', default='mnist')
    p.add_argument('--data-root', default=str(ROOT / 'dataset'))
    p.add_argument('-nb', '--num_classes', type=int, default=10)
    p.add_argument('-lbs', '--batch_size', type=int, default=64)
    p.add_argument('-lr', '--local_learning_rate', type=float, default=.1)
    p.add_argument('-ls', '--local_epochs', type=int, default=20)
    p.add_argument('-ld', '--learning_rate_decay', type=boolean, nargs='?', const=True, default=False)
    p.add_argument('-ldg', '--learning_rate_decay_gamma', type=float, default=.99)
    p.add_argument('-gr', '--global_rounds', type=int, default=200)
    p.add_argument('-nc', '--num_clients', type=int, default=100)
    p.add_argument('-ns', '--num_sel_clients', type=int, default=100)
    p.add_argument('-an', '--aggre_num', type=int, default=50)
    p.add_argument('-seed', '--seed', type=int, default=1)
    p.add_argument('-eg', '--eval_gap', type=int, default=1)
    p.add_argument('-al', '--alpha', type=float, default=0.)
    p.add_argument('-pv', '--prev', type=int, default=0)
    p.add_argument('-t', '--times', type=int, default=1)
    p.add_argument('-sfn', '--save_folder_name', default='items')
    p.add_argument('--episodes', type=int, default=0)
    p.add_argument('--actor_lr', type=float, default=.01)
    p.add_argument('--critic_lr', type=float, default=.01)
    p.add_argument('--actor_decay', type=float, default=1e-5)
    p.add_argument('--critic_decay', type=float, default=1e-5)
    p.add_argument('--buffer_size', type=int, default=100000)
    p.add_argument('--ad_ratio', type=float, default=.2)
    p.add_argument('--attack', choices=('none', 'same_value', 'sign_flip', 'gaussian', 'delta_sign_flip'), default='same_value')
    p.add_argument('--training-cohort', choices=('all', 'sampled'), default='all')
    p.add_argument('--validation-fraction', type=float, default=None)
    p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--evaluate-local', action='store_true')
    p.add_argument('--robust-f', type=int, default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--macro-length', type=int, default=5)
    p.add_argument('--eval-clients', type=int, default=20)
    p.add_argument('--rho-max', type=float, default=.2)
    p.add_argument('--loss-cap', type=float, default=5.)
    p.add_argument('--delta-sample', type=float, default=.05)
    p.add_argument('--delta-bad', type=float, default=.05)
    p.add_argument('--report-attack', choices=('none', 'all_max', 'bimodal', 'echo'), default='none')
    p.add_argument('--fake-utility', action='store_true')
    p.add_argument('--td-mode', choices=('paper', 'consistent'), default='paper')
    p.add_argument('--tc-gamma', type=float, default=.9)
    p.add_argument('--tc-actor-lr', type=float, default=.001)
    p.add_argument('--tc-critic-lr', type=float, default=.001)
    p.add_argument('--tc-batch-size', type=int, default=16)
    p.add_argument('--tc-updates', type=int, default=4)
    p.add_argument('--warmup-macros', type=int, default=4)
    p.add_argument('--gate-tolerance', type=float, default=.05)
    p.add_argument('--exploration-std', type=float, default=0.)
    p.add_argument('--fixed-lambda', type=float, default=0.)
    p.add_argument('--fixed-tau', type=float, default=1.35)
    p.add_argument('--sac-alpha', type=float, default=.1)
    p.add_argument('--sac-alpha-lr', type=float, default=.0003)
    p.add_argument('--sac-target-entropy', type=float, default=-2.)
    p.add_argument('--audit-validation', action='store_true')
    p.add_argument('--audit-shadow-every', type=int, default=0)
    return p


def validate(args):
    if not 1 <= args.aggre_num <= args.num_sel_clients <= args.num_clients:
        raise ValueError('Require 1 <= aggre_num <= num_sel_clients <= num_clients.')
    if min(args.batch_size, args.global_rounds, args.local_epochs, args.eval_gap, args.threads, args.buffer_size) < 1:
        raise ValueError('Batch size, rounds, epochs, evaluation gap, threads and replay capacity must be positive.')
    if not 0 <= args.ad_ratio < .5 or not 0 <= args.alpha < 1:
        raise ValueError('Require 0 <= ad_ratio < .5 and 0 <= alpha < 1.')
    if not 0 <= args.prev < args.times or args.seed < 0 or args.split_seed < 0:
        raise ValueError('Invalid repetition interval or random seed.')
    if args.episodes < 0 or args.num_classes < 2 or args.local_learning_rate <= 0:
        raise ValueError('Require episodes >= 0, num_classes >= 2 and local_learning_rate > 0.')
    if args.validation_fraction is None:
        args.validation_fraction = .1 if args.protocol == 'shared' else 0.
    if not 0 <= args.validation_fraction < 1:
        raise ValueError('validation_fraction must be in [0,1).')
    if args.protocol == 'upstream':
        if args.algorithm != 'FedAA' or args.validation_fraction != 0:
            raise ValueError('upstream protocol requires FedAA and validation_fraction=0.')
        if args.attack not in ('same_value', 'sign_flip', 'gaussian'):
            raise ValueError('Upstream protocol supports its original three attack names only.')
        if args.output is not None:
            raise ValueError('Upstream protocol saves to original ./results and ./models; omit --output.')
    elif args.episodes != 0:
        raise ValueError('Shared protocol uses one full run per seed; use --times instead of --episodes.')
    if args.robust_f is None:
        args.robust_f = math.ceil(args.ad_ratio * args.num_sel_clients)
    if args.robust_f < 0:
        raise ValueError('robust_f must be nonnegative.')
    if args.algorithm == 'TrimmedMean' and 2*args.robust_f >= args.num_sel_clients:
        raise ValueError('TrimmedMean requires 2f < num_sel_clients.')
    if args.algorithm == 'MultiKrum' and not (args.num_sel_clients > 2*args.robust_f+2 and args.aggre_num <= args.num_sel_clients-args.robust_f-2):
        raise ValueError('MultiKrum requires n > 2f+2 and aggre_num <= n-f-2.')
    if args.algorithm.startswith('FedTC'):
        from algorithms.fedtc.forward import validate_action
        validate_action([args.fixed_lambda, args.fixed_tau])
        if args.audit_shadow_every < 0:
            raise ValueError('audit_shadow_every must be nonnegative.')
        if args.macro_length < 1 or args.global_rounds % args.macro_length:
            raise ValueError('FedTC rounds must be divisible by macro_length.')
        if args.validation_fraction <= 0 or args.alpha != 0:
            raise ValueError('FedTC requires validation_fraction > 0 and alpha=0.')
        if args.loss_cap < math.log(args.num_classes):
            raise ValueError('Manuscript requires loss_cap >= log(num_classes).')
    args.data_root = str(Path(args.data_root).expanduser().resolve())


def make_model(args):
    if args.dataset == 'mnist':
        model = OneHiddenLayerFc(784, args.num_classes)
    elif args.dataset == 'emnist':
        model = TwoHiddenLayerFc(784, args.num_classes)
    elif args.dataset == 'Cifar10':
        model = CifarNet(num_classes=args.num_classes)
    elif 'synthetic' in args.dataset:
        model = Logistic(60, out_dim=args.num_classes)
    elif args.dataset == 'fashionmnist':
        model = ResNet2(1, args.num_classes)
    else:
        raise ValueError(f'No upstream model mapping for dataset {args.dataset}.')
    return model.to(args.device)


def prepare(args):
    validate(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu.')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    configure_data_root(args.data_root)
    args.model = make_model(args)
    if args.protocol == 'shared':
        args.model.apply(weights_init)
    if args.output is None and args.protocol == 'shared':
        args.output = str(ROOT / 'runs' / f'{args.dataset}_{args.algorithm}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    return args


def run(args):
    results = []
    for repetition in range(args.prev, args.times):
        current = copy.deepcopy(args)
        current.seed = args.seed + repetition
        if args.output is not None and args.times - args.prev > 1:
            current.output = str(Path(args.output) / f'seed_{current.seed}')
        prepare(current)
        if current.protocol == 'upstream':
            server = FedAA(current, repetition)
            server.replay_buffer.device = torch.device(current.device)
        elif current.algorithm.startswith('FedTC'):
            server = FedTC(current, repetition)
        else:
            server = SharedServer(current, repetition)
        results.append(server.train())
        if current.protocol == 'shared':
            print(f'Results: {server.output}', flush=True)
    return results


if __name__ == '__main__':
    run(parser().parse_args())
