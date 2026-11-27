"""A common comparison loop INSIDE FedAA's Server / clientFedAA framework.

Data loading, local SGD, parameter broadcast, accuracy/AUC/loss definitions are
inherited from upstream. Round order is explicit and shared across methods.
"""
import hashlib
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from algorithms.baselines import aggregate
from clients.clientFedTC import clientFedTC
from DDPG.DDPG import DDPG
from servers.serverbase import Server
from servers.serverFedAA import FedAA
from utils.experiment import (append_json, attack_model, fingerprint,
                              model_fingerprint, set_vector, vector, write_json)


class SharedServer(Server):
    def __init__(self, args, times=0):
        super().__init__(args, times)
        self.output = Path(args.output).expanduser().resolve()
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f'Choose a new output directory: {self.output}')
        self.set_clients(clientFedTC)
        too_small = [c.id for c in self.clients if c.train_samples < self.batch_size]
        self.zero_batch_clients = too_small
        if too_small:
            print(f'Native drop_last=True: clients {too_small} have zero SGD batches at batch_size={self.batch_size}.', flush=True)
        if all(c.train_samples < self.batch_size for c in self.clients):
            raise ValueError('All clients have zero batches; reduce batch_size.')
        self.ad_clients = sorted(np.random.RandomState(args.seed).choice(
            self.num_clients, round(self.num_clients * args.ad_ratio), replace=False).tolist())
        self.benign_clients = sorted(set(range(self.num_clients)) - set(self.ad_clients))
        self.cohort_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 101]))
        self.agent = None
        self.pending_fedaa = None
        if self.algorithm == 'FedAA':
            self.agent = DDPG(args)
            self.replay_buffer.device = torch.device(args.device)
        self.evaluation_rounds = []
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = self.make_manifest()
        write_json(self.output / 'manifest.json', self.manifest)

    def make_manifest(self):
        native_files = {}
        root = Path(self.args.data_root) / self.dataset
        for part in ('train', 'test'):
            for client in self.clients:
                path = root / part / f'{client.id}.npz'
                native_files[f'{part}/{client.id}.npz'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if (root / 'config.json').exists():
            native_files['config.json'] = hashlib.sha256((root / 'config.json').read_bytes()).hexdigest()
        splits = {str(c.id): {'train': c.train_indices, 'validation': c.validation_indices} for c in self.clients}
        write_json(self.output / 'private_split_indices.json', splits)
        # Harness provenance only. No identities or private indices enter FedTC's controller.
        return {
            'upstream': 'https://github.com/Gp1g/FedAA',
            'upstream_commit': '1fba884934cbec1b3506812d9d612e6a181b1d79',
            'args': {key: val for key, val in vars(self.args).items() if key != 'model'},
            'initial_model_sha256': model_fingerprint(self.global_model),
            'native_data_sha256': fingerprint(native_files), 'native_files': native_files,
            'server_reward_data_sha256': (hashlib.sha256((root / 'test' / 'server_test.npz').read_bytes()).hexdigest()
                                          if self.algorithm == 'FedAA' else None),
            'private_splits_sha256': fingerprint(splits),
            'zero_sgd_batch_clients': self.zero_batch_clients,
            'malicious_clients_harness_only': self.ad_clients,
            'round_order': 'broadcast -> local SGD -> aggregate -> broadcast -> global evaluation',
            'local_training': 'unmodified clients.clientFedAA.clientFedAA.train',
            'test_metrics': 'inherited Server.evaluate / Client.test_metrics / Client.train_metrics',
            'parameter_buffers': 'upstream parameter-only broadcast and aggregation; buffers stay local',
        }

    def begin_round(self, round_index):
        pass

    def end_round(self, round_index):
        pass

    def train_clients(self, cohort, round_index, start_vector):
        train_ids = set(c.id for c in cohort)
        workers = self.clients if self.args.training_cohort == 'all' else cohort
        utilities = {}
        for client in workers:
            client.round_index = round_index
            needs_utility = self.algorithm.startswith('FedTC') and self.algorithm != 'FedTCStep' and client.id in train_ids
            before = client.bounded_loss(client.model, self.args.loss_cap, validation=False) if needs_utility else 0.
            devices = [torch.cuda.current_device()] if self.device == 'cuda' else []
            # Controller sampling cannot change local minibatches or dropout RNG.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.seed + 100003 * round_index + 97 * client.id)
                client.train()  # Original FedAA method; not a reimplementation.
            if needs_utility:
                after = client.bounded_loss(client.model, self.args.loss_cap, validation=False)
                utilities[client.id] = float(np.clip((before-after) / self.args.loss_cap, 0, 1))
            if client.id in self.ad_clients:
                attack_model(client, start_vector, self.attack, self.seed, round_index)
                if needs_utility and self.args.fake_utility:
                    utilities[client.id] = 1.
        return [utilities.get(c.id, 0.) for c in cohort]

    def aggregate_round(self, cohort, updates, utilities, round_index):
        if self.algorithm == 'FedAA':
            state, selected = FedAA.get_state(self, selected_clients=cohort)
            if self.pending_fedaa is not None:
                previous_state, previous_action, previous_reward = self.pending_fedaa
                self.replay_buffer.add(previous_state, previous_action, state.numpy(), previous_reward, False)
                self.agent.update_parameters(self.replay_buffer, 16)
            action = self.agent.select_action(state)
            self.receive_models_RL(action, selected)
            self.aggregate_parameters()  # Original FedAA weighted model aggregation.
            reward = self.test_server(round_index)  # Only FedAA uses the clean server dataset.
            self.pending_fedaa = (state.numpy().copy(), action.copy(), reward)
            if round_index + 1 == self.global_rounds:
                self.replay_buffer.add(state.numpy(), action, state.numpy(), reward, True)
                self.agent.update_parameters(self.replay_buffer, 16)
                self.pending_fedaa = None
            weights = {c.id: float(w) for c, w in zip(selected, action.reshape(-1))}
            return {'weights': [weights.get(c.id, 0.) for c in cohort],
                    'selected_for_aggregation': [c.id for c in selected], 'server_reward': reward}
        delta, weights = aggregate(self.algorithm, updates, [c.train_samples for c in cohort],
                                   self.args.robust_f, self.aggre_num)
        set_vector(self.global_model, vector(self.global_model) + (1-self.alpha) * delta)
        return {'weights': None if weights is None else weights.cpu().tolist()}

    def evaluate_global(self, round_number):
        # Only global-model metrics enter primary H5. Native local metrics remain available.
        self.send_models()
        for client in self.clients:
            client.round_index = round_number - 1
        self.evaluate(self.benign_clients)
        self.evaluation_rounds.append(round_number)
        row = {'round': round_number, 'phase': 'global', 'accuracy': self.rs_test_acc[-1],
               'auc': self.rs_test_auc[-1], 'client_std': self.rs_test_std[-1],
               'train_loss': self.rs_train_loss[-1]}
        append_json(self.output / 'evaluations.jsonl', row)
        return row

    def train(self):
        started = time.perf_counter()
        for round_index in range(self.global_rounds):
            round_started = time.perf_counter()
            self.begin_round(round_index)
            cohort_ids = self.cohort_rng.choice(self.num_clients, self.num_sel_clients, replace=False).tolist()
            cohort = [self.clients[i] for i in cohort_ids]
            self.send_models()
            start_vector = vector(self.global_model)
            utilities = self.train_clients(cohort, round_index, start_vector)
            updates = torch.stack([vector(c.model) - start_vector for c in cohort])
            if not torch.isfinite(updates).all():
                raise FloatingPointError('Nonfinite uploaded model update.')
            if self.args.evaluate_local and ((round_index + 1) % self.args.eval_gap == 0 or round_index + 1 == self.global_rounds):
                local_acc, local_loss = [], []
                self.evaluate(self.benign_clients, acc=local_acc, loss=local_loss)
                append_json(self.output / 'local_evaluations.jsonl', {
                    'round': round_index + 1, 'phase': 'local',
                    'accuracy': local_acc[0], 'train_loss': local_loss[0]})
            info = self.aggregate_round(cohort, updates, utilities, round_index)
            self.end_round(round_index)
            row = {'round': round_index + 1, 'upload_clients': cohort_ids,
                   'trained_clients': list(range(self.num_clients)) if self.args.training_cohort == 'all' else cohort_ids,
                   'malicious_uploads_harness_only': sum(i in self.ad_clients for i in cohort_ids),
                   'utilities': utilities if self.algorithm.startswith('FedTC') else None, **info}
            if (round_index + 1) % self.args.eval_gap == 0 or round_index + 1 == self.global_rounds:
                row['evaluation'] = self.evaluate_global(round_index + 1)
            row['elapsed_seconds'] = time.perf_counter() - round_started
            append_json(self.output / 'rounds.jsonl', row)
            print(f'{self.algorithm} round {round_index+1}/{self.global_rounds}: {row["elapsed_seconds"]:.2f}s', flush=True)
        self.send_models()
        self.save_results()
        torch.save({'model': self.global_model.state_dict(), 'manifest': self.manifest}, self.output / 'model.pt')
        if self.agent is not None:
            self.agent.save(str(self.output / 'fedaa'))
        self.save_extra()
        summary = {'status': 'complete', 'algorithm': self.algorithm,
                   'rounds': self.global_rounds, 'final_accuracy': self.rs_test_acc[-1],
                   'final_auc': self.rs_test_auc[-1], 'best_accuracy': max(self.rs_test_acc),
                   'elapsed_seconds': time.perf_counter()-started, **self.extra_summary()}
        write_json(self.output / 'summary.json', summary)
        return summary

    def save_results(self):
        with h5py.File(self.output / 'metrics.h5', 'w') as handle:
            for name in ('rs_test_acc', 'rs_test_std', 'rs_test_auc', 'rs_train_loss'):
                handle.create_dataset(name, data=getattr(self, name))
            handle.create_dataset('round', data=self.evaluation_rounds)
            handle.attrs['phase'] = 'global'
        if self.agent is not None:
            write_json(self.output / 'server_rewards.json', self.all_reward)

    def save_extra(self):
        pass

    def extra_summary(self):
        return {}
