"""Run the fixed MNIST last-window check, evaluating only the two final candidates."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.refit_last_mnist import pool_config
from diagnostics.run_shadow_macro_rl_mnist import H, commit, open_world
from utils.experiment import set_vector, vector, write_json


@torch.no_grad()
def endpoint_cost(world, parameters):
    set_vector(world.global_model, parameters)
    logits = world.private_logits()
    error = (logits.argmax(dim=1) != world.private_y).float().mean().item()
    losses = F.cross_entropy(logits, world.private_y, reduction="none").clamp(max=world.args.loss_cap)
    per_client = np.array([chunk.mean().item() for chunk in losses.split(world.sizes.tolist())])
    return error + .05 * float(per_client @ world.sample_weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--client-offset", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.seed, args.split_seed) < 0 or not 0 <= args.client_offset <= 76:
        parser.error("Use nonnegative seeds and an offset between 0 and 76 for the 24-client pool.")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = pool_config(args.data_root.resolve(), out, args.client_offset, args.split_seed)
    world, schedule = open_world(config, args.seed, out / "training")
    anchor, alternative = 1.15, 1.55
    parameters = vector(world.global_model)
    last = config["rounds"] - H
    for index in range(0, last, H):
        uploads = world.collect_uploads(parameters, index, schedule)
        options, _, _ = world.aggregate_options(uploads, parameters, [anchor])
        parameters = commit(world, options[0], index, anchor, schedule)

    uploads = world.collect_uploads(parameters, last, schedule)
    taus = [anchor, alternative]
    options, _, _ = world.aggregate_options(uploads, parameters, taus)
    ends = [commit(world, model, last, tau, schedule) for model, tau in zip(options, taus)]
    costs = [endpoint_cost(world, model) for model in ends]
    choice = int(costs[1] < costs[0])
    set_vector(world.global_model, ends[choice])
    torch.save(world.global_model.state_dict(), out / "global_model.pt")
    write_json(out / "terminal_selection.json", dict(
        candidate_cost=costs[1], anchor_cost=costs[0], kept_candidate=bool(choice),
        selected_tau=taus[choice], fork_rounds=H, final_test=world.record_test(ends[choice]),
    ))


if __name__ == "__main__":
    main()
