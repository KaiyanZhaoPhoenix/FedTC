"""Continue RL optimization on fixed replay; no new FL training or accuracy claims.

Run from the repository root with .venv/bin/python diagnostics/check_rl_duration.py.
All configurations, PopArt statistics, transitions and optimizer states are restored.
Original experiment files are read-only. Gate checks are retrospective diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.fedtc.controller import ControllerConfig, IntervalController
from algorithms.fedtc.forward import CONSERVATIVE_ACTION


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def measure(controller, postwarmup_indices):
    states, actions, lows, highs, next_states, dones = zip(*controller.replay)
    s = controller.normalize_state(torch.stack(states))
    ns = controller.normalize_state(torch.stack(next_states))
    a = torch.tensor(actions, dtype=torch.float32)
    candidate = controller.actor_action(s)
    conservative = torch.tensor(CONSERVATIVE_ACTION).expand(len(s), -1)
    candidate_low = controller.low(s, candidate).flatten()
    reference_high = controller.high(s, conservative).flatten()
    margin = candidate_low - reference_high + controller.config.tolerance
    self_margin = (controller.low(s, conservative).flatten() - reference_high
                   + controller.config.tolerance)
    selected = torch.tensor(postwarmup_indices, dtype=torch.long)

    # Full-replay Bellman residual, using exactly the restored training target.
    # Its target moves during training; this is not held-out prediction error.
    continuation = 1 - torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    next_a = controller.actor_action(ns)
    targets = []
    for rewards, target in ((lows, controller.target_low), (highs, controller.target_high)):
        y = ((torch.tensor(rewards).unsqueeze(1) - controller.popart.mean)
             / controller.popart.std)
        y += controller.config.gamma * continuation * target(ns, next_a)
        if controller.config.td_mode == "consistent":
            y += (controller.config.gamma * continuation * controller.popart.mean
                  / controller.popart.std)
        targets.append(y)
    measured = {
        "train_steps": controller.train_steps,
        "gate_states": len(selected),
        "accepted": int((margin[selected] >= 0).sum()),
        "self_comparison_accepted": int((self_margin[selected] >= 0).sum()),
        "margin_min_normalized": float(margin[selected].min()),
        "margin_max_normalized": float(margin[selected].max()),
        "last_replay_state_candidate": candidate[-1].tolist(),
        "last_replay_state_margin_normalized": float(margin[-1]),
        "last_replay_state_candidate_low_normalized": float(candidate_low[-1]),
        "last_replay_state_reference_high_normalized": float(reference_high[-1]),
        "full_replay_td_mse_low": float(torch.nn.functional.mse_loss(controller.low(s, a), targets[0])),
        "full_replay_td_mse_high": float(torch.nn.functional.mse_loss(controller.high(s, a), targets[1])),
        "candidate_actions_on_postwarmup_states": candidate[selected].tolist(),
        "margins_on_postwarmup_states_normalized": margin[selected].tolist(),
    }
    # Check against the actual deployed gate implementation at each checkpoint.
    for i in postwarmup_indices:
        _, gate = controller.select_action(states[i], controller.config.warmup_macros + 1)
        assert (not gate["fallback"]) == bool(margin[i] >= 0)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/single_seed_attack_horizons_20261117")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/rl_duration_diagnostic_20261117")
    args = parser.parse_args()
    # Do not silently replace a previous diagnostic.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    milestones = [0, 200, 1000, 5000]
    metadata = {
        "purpose": "Optimization-only diagnostic on unchanged saved replay; no new environmental interaction.",
        "additional_gradient_updates": milestones,
        "scenarios": ["clean", "delta_sign_flip", "joint_all_max"],
        "source": str(args.source.resolve()),
        "script_sha256": sha256(Path(__file__)),
        "controller_source_sha256": sha256(ROOT / "algorithms/fedtc/controller.py"),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": 1,
        "new_fl_rounds": 0,
        "new_validation_feedback": 0,
        "popart_updated": False,
        "limits": [
            "Retrospective gate checks are not new executed actions or accuracy measurements.",
            "Fixed single-action replay cannot identify benefits of unseen actions.",
            "This does not test longer training with fresh interactions.",
            "TD residuals use moving targets and training data, not held-out targets.",
        ],
    }
    (args.output / "plan.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    for scenario in metadata["scenarios"]:
        started = time.monotonic()
        folder = args.source / f"full100_100r_{scenario}" / "FedTC"
        path = folder / "controller.pt"
        before = sha256(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        macros = [json.loads(line) for line in (folder / "macros.jsonl").read_text().splitlines()]
        config = ControllerConfig(**saved["config"])
        assert config.exploration_std == 0, "Diagnostic assumes deterministic gate evaluation."
        controller = IntervalController(config, seed=1)
        controller.load_state_dict(saved)
        assert len(macros) == len(controller.replay)
        indices = [i for i, macro in enumerate(macros) if macro["gate_reason"] != "warmup"]
        unique_actions = np.unique([row[1] for row in controller.replay], axis=0)
        source_steps = controller.train_steps
        result = {
            "scenario": scenario,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": before,
            "replay_count": len(controller.replay),
            "unique_executed_actions": unique_actions.tolist(),
            "source_train_steps": source_steps,
            "source_config": saved["config"],
            "frozen_popart": saved["popart"],
            "original_online_accepted": sum(not m["fallback"] for m in macros),
            "original_online_postwarmup_gates": len(indices),
            "minimum_reward_interval_width": min(row[3] - row[2] for row in controller.replay),
            "checkpoints": [],
        }
        for extra in milestones:
            assert extra % config.updates_per_macro == 0
            while controller.train_steps < source_steps + extra:
                controller.train()
            assert controller.train_steps == source_steps + extra
            metrics = {"additional_gradient_updates": extra, **measure(controller, indices)}
            result["checkpoints"].append(metrics)
            print(scenario, "extra", extra, "accepted", f"{metrics['accepted']}/{metrics['gate_states']}",
                  "best margin", round(metrics["margin_max_normalized"], 4), flush=True)
        assert sha256(path) == before
        assert controller.state_dict()["popart"] == saved["popart"]
        assert len(controller.replay) == len(saved["replay"])
        result["source_checkpoint_unchanged"] = True
        result["elapsed_seconds"] = time.monotonic() - started
        (args.output / f"{scenario}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        torch.save(controller.state_dict(), args.output / f"{scenario}_extra5000_controller.pt")
        results.append(result)
    (args.output / "results.json").write_text(json.dumps({"metadata": metadata, "results": results}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
