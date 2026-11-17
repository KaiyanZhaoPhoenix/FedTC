"""Deterministic experiment bookkeeping; no alternative data pipeline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def vector(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


@torch.no_grad()
def set_vector(model, values):
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite aggregated model parameters.')
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.copy_(values[offset:offset+count].view_as(parameter))
        offset += count
    if offset != values.numel():
        raise ValueError('Model parameter count differs from update dimension.')


@torch.no_grad()
def attack_model(client, start_vector, attack, seed, round_index):
    """Original FedAA whole-model attacks; explicit delta flip is separate."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_index, client.id, 441]))
    if attack == 'same_value':
        value = float(rng.normal(0, 100))
        for parameter in client.model.parameters():
            parameter.fill_(value)
    elif attack == 'sign_flip':
        # Upstream's name is misleading: it multiplies by a POSITIVE magnitude.
        value = abs(float(rng.normal(0, 10)))
        for parameter in client.model.parameters():
            parameter.mul_(value)
    elif attack == 'gaussian':
        generator = torch.Generator(device='cpu').manual_seed(int(rng.integers(0, 2**31)))
        for parameter in client.model.parameters():
            parameter.copy_(100 * torch.randn(parameter.shape, generator=generator).to(parameter.device))
    elif attack == 'delta_sign_flip':
        set_vector(client.model, start_vector - 10 * (vector(client.model) - start_vector))
    elif attack != 'none':
        raise ValueError(f'Unknown attack: {attack}')


def attack_reports(values, malicious_mask, attack, cap):
    reports = torch.tensor(values, dtype=torch.float64)
    mask = torch.tensor(malicious_mask, dtype=torch.bool)
    indices = torch.where(mask)[0]
    if attack == 'all_max':
        reports[mask] = cap
    elif attack == 'bimodal':
        reports[indices[::2]], reports[indices[1::2]] = cap, -cap
    elif attack == 'echo':
        positives = reports[(~mask) & (reports > 0)]
        reports[mask] = positives.median() if len(positives) else 0
    elif attack != 'none':
        raise ValueError(f'Unknown report attack: {attack}')
    return reports.clamp(-cap, cap)
