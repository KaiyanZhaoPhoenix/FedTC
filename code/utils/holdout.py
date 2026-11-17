"""An algorithm-independent holdout within each upstream training partition."""
import numpy as np


def split_indices(size, fraction, seed, client_id):
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if fraction == 0:
        return list(range(size)), []
    if size < 2:
        raise ValueError(f"Client {client_id} needs at least two original training examples for a holdout.")
    order = np.random.default_rng(np.random.SeedSequence([seed, client_id, 1907])).permutation(size)
    count = max(1, min(size - 1, round(fraction * size)))
    return sorted(order[count:].tolist()), sorted(order[:count].tolist())
