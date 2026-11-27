"""Frozen server-step policy and paired finite-horizon regression."""
from dataclasses import dataclass

import numpy as np

STEPS = (.5, 1., 1.5)
TAUS = (1.15, 1.55)


@dataclass
class RolloutScorer:
    scale: np.ndarray
    weight: np.ndarray

    @classmethod
    def fit(cls, contexts, labels, penalty):
        x=np.array(contexts)
        differences=x-x.mean(axis=1,keepdims=True)
        flat=differences.reshape(-1,x.shape[-1])
        scale=np.maximum(flat.std(0),1e-4)
        flat=flat/scale
        y=np.array(labels)
        y=(y-y.mean(axis=1,keepdims=True)).ravel()
        weight=np.linalg.solve(flat.T@flat+penalty*np.eye(flat.shape[1]),flat.T@y)
        return cls(scale,weight)

    def scores(self, x):
        return (x-x.mean(axis=0))/self.scale @ self.weight


def features(gains, sizes, current, upcoming, window, windows):
    denominator = max(1, windows - 1)
    groups = (np.arange(len(sizes)), current, upcoming)
    means = np.array([[np.average(row[ids], weights=sizes[ids]) for ids in groups] for row in gains])
    identity = np.eye(len(STEPS))
    design = np.column_stack((identity, identity * window / denominator, means, means * window / denominator))
    if not np.isfinite(design).all():
        raise FloatingPointError("Nonfinite private validation gain")
    return design, means[:, 0]


def select(rule, design, pilot, window):
    if 'time' in rule:
        return rule['time'][window]
    if 'arm' in rule:
        return rule['arm']
    scores = pilot if rule.get('direct') else RolloutScorer(
        np.asarray(rule['scale']), np.asarray(rule['weight'])).scores(design)
    best = int(scores.argmax())
    return 1 if scores[1] >= scores[best] - 1e-12 else best


def fit_rules(rows, windows):
    labels = np.array([row['labels'] for row in rows])
    times = np.array([row['t'] // 2 for row in rows])
    rules = {f'step_{i}': {'arm': i} for i in range(len(STEPS))}
    rules['direct'] = {'direct': True}
    for j, name in enumerate(('immediate', 'rl')):
        model = RolloutScorer.fit([row['x'] for row in rows], labels[:, :, j], 32)
        rules[name] = dict(scale=model.scale.tolist(), weight=model.weight.tolist())
        schedule = []
        for i in range(windows):
            distance = abs(times - i // 2)
            # Sparse development jobs use the nearest populated time bin.
            schedule.append(int(labels[distance == distance.min(), :, j].mean(axis=0).argmax()))
        rules['time_' + name] = dict(time=schedule)
    return rules


def freeze(rules, rows):
    means = {name: float(np.mean([row['metrics']['accuracy'] for row in group])) for name, group in rows.items()}
    arms = ['step_0', 'step_1', 'step_2']
    lock = dict(incumbent='step_1', fixed=max(arms, key=means.get),
                schedule=max(['time_immediate', 'time_rl'] + arms, key=means.get),
                direct='direct', immediate='immediate', rl='rl')
    return {name: rules[key] for name, key in lock.items()}, dict(means=means, lock=lock)
