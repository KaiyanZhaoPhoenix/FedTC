"""FedAA execution backend for paired trajectories: SGD, state, feedback and endpoint metrics."""
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.fedtc.forward import calibrate
from algorithms.fedtc.step_control import STEPS
from servers.serverShared import SharedServer
from utils.experiment import fingerprint, set_vector, vector, write_json


@dataclass
class State:
    params: torch.Tensor
    buffers: tuple
    optimizers: tuple


class StepExecution(SharedServer):
    def __init__(self, args):
        super().__init__(args)
        if args.local_steps is not None and self.zero_batch_clients:
            raise ValueError(f"Clients with no full SGD batch: {self.zero_batch_clients}. Reduce batch_size.")
        self.horizon = args.macro_length
        self.windows = math.ceil(self.global_rounds / self.horizon)
        self.schedule = self.make_schedule()
        self.initial = State(vector(self.global_model), tuple(b.clone() for b in self.global_model.buffers()),
                             self.optimizer_states())
        self.sizes = np.array([len(c.validation_indices) for c in self.clients])
        private = [c._full_train_data[i] for c in self.clients for i in c.validation_indices]
        self.private_x = torch.stack([x for x, _ in private])
        self.private_y = torch.stack([y for _, y in private])
        self.identity = [self.manifest[key] for key in ('initial_model_sha256', 'native_data_sha256',
                                                         'private_splits_sha256')] + [fingerprint(self.schedule)]
        self.costs = dict(uploads=0, local_optimizer_steps=0, private_model_evaluations=0,
                          private_example_evaluations=0, test_model_evaluations=0)
        self.manifest.update(parameter_buffers="full broadcast; weighted floating buffers, max retained counter",
                             local_training="FedAA SGD with configured step or epoch budget",
                             test_metrics="FedAA client test counts, pooled accuracy and worst-client decile",
                             announced_upload_schedule=self.schedule, identity=self.identity)
        write_json(self.output / 'manifest.json', self.manifest)

    def make_schedule(self):
        if self.args.cohort_schedule == 'uniform':
            return [self.cohort_rng.choice(self.num_clients, self.num_sel_clients, replace=False).tolist()
                    for _ in range(self.global_rounds)]
        def dominant(c):
            labels = torch.stack([c._full_train_data[i][1] for i in c.train_indices])
            return int(torch.bincount(labels).argmax())
        order = sorted(range(self.num_clients), key=lambda i: dominant(self.clients[i]))
        mid = self.num_clients // 2
        if self.num_sel_clients > mid:
            raise ValueError("grouped schedule requires K <= floor(N/2); choose uniform for a larger K")
        halves = (order[:mid], order[mid:])
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 960]))
        blocks = [v for _ in range(math.ceil(self.windows / 2)) for v in rng.permutation(2)]
        return [rng.permutation(halves[blocks[i // self.horizon]])[:self.num_sel_clients].tolist()
                for i in range(self.global_rounds)]

    def optimizer_states(self):
        if not self.args.learning_rate_decay:
            return ()  # FedAA's momentum-free SGD has no persistent state at constant LR.
        return tuple((copy.deepcopy(c.optimizer.state_dict()), copy.deepcopy(c.learning_rate_scheduler.state_dict()))
                     for c in self.clients)

    @torch.no_grad()
    def install(self, state):
        set_vector(self.global_model, state.params)
        for buffer, value in zip(self.global_model.buffers(), state.buffers):
            buffer.copy_(value)

    def directions(self, start, index, taus):
        self.install(start)
        cohort = [self.clients[i] for i in self.schedule[index]]
        workers = self.clients if self.args.training_cohort == 'all' else cohort
        if start.optimizers:
            for c, (opt, scheduler) in zip(self.clients, start.optimizers):
                c.optimizer.load_state_dict(copy.deepcopy(opt))
                c.learning_rate_scheduler.load_state_dict(copy.deepcopy(scheduler))
        params = self.global_model.state_dict()
        for client in workers:
            client.model.load_state_dict(params)
        self.train_clients(cohort, index, start.params)
        updates = torch.stack([vector(c.model) - start.params for c in cohort])
        self.costs['uploads'] += len(cohort)
        self.costs['local_optimizer_steps'] += sum(c.last_train_steps for c in workers)
        buffers = list(zip(*(c.model.buffers() for c in cohort)))
        optimizers = self.optimizer_states()
        result = []
        for tau in taus:
            agg = calibrate(updates, updates.new_zeros(len(cohort)), (0., tau))
            merged = []
            for old, values in zip(start.buffers, buffers):
                if agg.fallback:
                    merged.append(old)
                elif old.is_floating_point():
                    stack = torch.stack(values)
                    shape = (len(values),) + (1,) * old.ndim
                    merged.append((stack * agg.weights.reshape(shape)).sum(0))
                else:
                    merged.append(torch.stack(values)[agg.weights > 0].amax(0))
            # Match the two rounded operations of the original server-step implementation.
            result.append(State((start.params + agg.delta) - start.params, tuple(merged), optimizers))
        return result

    @staticmethod
    def advance(start, direction, arm):
        return State(start.params + STEPS[arm] * direction.params, direction.buffers, direction.optimizers)

    @torch.no_grad()
    def private(self, state):
        self.install(state)
        self.global_model.eval()
        # The simulator batches private forward passes; the policy receives aggregate gains only.
        size = len(self.private_y) if self.dataset == 'mnist' else 128
        logits = torch.cat([self.global_model(x.to(self.device)).cpu() for x in self.private_x.split(size)])
        errors = logits.argmax(1).ne(self.private_y).float()
        loss = F.cross_entropy(logits, self.private_y, reduction='none').clamp(max=self.args.loss_cap)
        sizes = self.sizes.tolist()
        means = np.array([float(x.mean()) for x in (errors + .05 * loss).split(sizes)])
        ce = np.array([float(x.mean()) for x in loss.split(sizes)])
        self.costs['private_model_evaluations'] += 1
        self.costs['private_example_evaluations'] += len(errors)
        return means, float(errors.mean()) + .05 * float(ce @ (self.sizes / self.sizes.sum())), float((1-errors).mean())

    def test(self, state):
        self.install(state)
        for client in self.clients:
            client.model.load_state_dict(self.global_model.state_dict())
        scores = [c.test_metrics() for c in self.clients]
        n = math.ceil(.1 * len(scores))
        self.costs['test_model_evaluations'] += 1
        return dict(accuracy=sum(s[0] for s in scores) / sum(s[1] for s in scores),
                    worst10=float(np.mean(sorted(s[0]/s[1] for s in scores)[:n])), worst_client_count=n)
