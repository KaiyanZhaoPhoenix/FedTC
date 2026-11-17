"""Aggregators used with the SAME FedAA clients, cohorts, and evaluation."""
import torch

from algorithms.fedtc.forward import geometric_median


@torch.no_grad()
def aggregate(name, updates, sizes, bad_bound, krum_count):
    n = len(updates)
    if name == 'FedAvg':
        weights = updates.new_tensor(sizes)
        weights /= weights.sum()
        return updates.T.mv(weights), weights
    if name == 'RFA':
        # Uniform-client geometric median, 15 modified Weiszfeld iterations.
        return geometric_median(updates), None
    if name == 'TrimmedMean':
        if not 0 <= 2 * bad_bound < n:
            raise ValueError('TrimmedMean requires 0 <= 2f < cohort size.')
        ordered = updates.sort(dim=0).values
        return ordered[bad_bound:n-bad_bound].mean(0), None
    if name == 'MultiKrum':
        if not (n > 2 * bad_bound + 2 and 1 <= krum_count <= n - bad_bound - 2):
            raise ValueError('MultiKrum requires n > 2f+2 and 1 <= aggre_num <= n-f-2.')
        distances = torch.cdist(updates, updates).square()
        distances.fill_diagonal_(float('inf'))
        scores = distances.sort(dim=1).values[:, :n-bad_bound-2].sum(1)
        chosen = scores.argsort(stable=True)[:krum_count]
        weights = updates.new_zeros(n)
        weights[chosen] = 1 / krum_count
        return updates.T.mv(weights), weights
    raise ValueError(f'Unknown baseline: {name}')
