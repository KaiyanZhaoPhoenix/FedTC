"""Server-step decisions and paired rollout orchestration, independent of SGD execution."""
import numpy as np
import torch

from algorithms.fedtc.step_control import STEPS, TAUS, features, select
from servers.step_execution import StepExecution


class FedTCStep(StepExecution):
    def taus(self, window):
        return TAUS if window == self.windows - 1 else TAUS[:1]

    def probe(self, start, window):
        index = window * self.horizon
        first = self.directions(start, index, self.taus(window))
        initial = self.private(start)[0]
        gains = [initial - self.private(self.advance(start, first[0], arm))[0] for arm in range(len(STEPS))]
        upcoming = min(index + self.horizon, self.global_rounds - 1)
        design, pilot = features(gains, self.sizes, self.schedule[index], self.schedule[upcoming], window, self.windows)
        return first, design, pilot

    def commit(self, start, window, arm, first=None):
        index = window * self.horizon
        taus = self.taus(window)
        if first is None:
            first = self.directions(start, index, taus)
        ends = []
        for tau, direction in zip(taus, first):
            end = self.advance(start, direction, arm)
            for i in range(index + 1, min(index + self.horizon, self.global_rounds)):
                end = self.advance(end, self.directions(end, i, (tau,))[0], arm)
            ends.append(end)
        chosen = int(len(ends) == 2 and self.private(ends[1])[1] < self.private(ends[0])[1])
        return ends[chosen], taus[chosen]

    def paired_training(self, position):
        sample = {position % self.windows, (position + self.windows // 2) % self.windows}
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 23]))
        start, rows = self.initial, []
        for t in range(self.windows):
            if t not in sample:
                start, _ = self.commit(start, t, 1)
                continue
            first, design, _ = self.probe(start, t)
            c0 = self.private(start)[1]
            ends, labels = [], []
            for arm in range(len(STEPS)):
                end, _ = self.commit(start, t, arm, first)
                c1 = self.private(end)[1]
                c2 = self.private(self.commit(end, t+1, 1)[0])[1] if t+1 < self.windows else c1
                labels.append([c0-c1, c0-c1 + self.args.tc_gamma * (c1-c2)])
                ends.append(end)
            rows.append(dict(seed=self.seed, t=t, x=design.tolist(), labels=labels))
            start = ends[1 if rng.random() < .5 else int(rng.integers(len(STEPS)))]
        return rows

    def evaluate_rules(self, rules, test=False, checkpoint=None):
        # Depth-first traversal shares exactly matching prefixes. Each sibling owns its state.
        rows = {}
        def visit(state, t, names, actions, taus, pilots):
            while t < self.windows:
                first, design, pilot = self.probe(state, t)
                branches = {}
                for name in names:
                    arm = select(rules[name], design, pilot, t)
                    branches.setdefault(arm, []).append(name)
                pilots = pilots + [pilot.tolist()]
                # Single trajectories retain only their current model, including on long runs.
                if len(branches) == 1:
                    arm = next(iter(branches))
                    state, tau = self.commit(state, t, arm, first)
                    t, actions, taus = t+1, actions+[arm], taus+[tau]
                    continue
                for arm, group in branches.items():
                    end, tau = self.commit(state, t, arm, first)
                    visit(end, t+1, group, actions+[arm], taus+[tau], pilots)
                return
            metrics = self.test(state) if test else {'accuracy': self.private(state)[2]}
            if checkpoint is not None:
                self.install(state)
                torch.save(self.global_model.state_dict(), checkpoint)
            for name in names:
                rows[name] = dict(seed=self.seed, method=name, identity=self.identity, actions=actions,
                                  steps=[STEPS[i] for i in actions], taus=taus, private_gains=pilots, metrics=metrics)
        visit(self.initial, 0, list(rules), [], [], [])
        return rows
