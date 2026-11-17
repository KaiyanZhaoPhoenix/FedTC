"""FedTC macro controller and feedback, integrated into the FedAA server."""
import copy
import math
from dataclasses import asdict

import numpy as np
import torch

from algorithms.fedtc.controller import ControllerConfig, IntervalController
from algorithms.fedtc.forward import CONSERVATIVE_ACTION, calibrate, macro_state
from algorithms.fedtc.interval import IntervalEstimator
from algorithms.fedtc.sac import SACConfig, SACController
from servers.serverShared import SharedServer
from utils.experiment import append_json, attack_reports, set_vector, vector


class FedTC(SharedServer):
    def __init__(self, args, times=0):
        # Check interval feasibility before allocating clients / creating output.
        self.estimator = IntervalEstimator(args.num_clients, args.eval_clients, args.rho_max,
                                           args.loss_cap, args.delta_sample, args.delta_bad)
        if round(args.num_clients * args.ad_ratio) > math.floor(args.num_clients * args.rho_max + 1e-12):
            raise ValueError('Actual simulated malicious count exceeds rho_max assumption.')
        super().__init__(args, times)
        config = ControllerConfig(td_mode=args.td_mode, gamma=args.tc_gamma,
                                  actor_lr=args.tc_actor_lr, critic_lr=args.tc_critic_lr,
                                  batch_size=args.tc_batch_size, replay_capacity=args.buffer_size,
                                  warmup_macros=args.warmup_macros,
                                  updates_per_macro=args.tc_updates, tolerance=args.gate_tolerance,
                                  exploration_std=args.exploration_std)
        if self.algorithm in ('FedTCSAC', 'FedTCSACGate'):
            sac_config = SACConfig(gamma=args.tc_gamma, actor_lr=args.tc_actor_lr,
                                   critic_lr=args.tc_critic_lr, alpha_lr=args.sac_alpha_lr,
                                   initial_alpha=args.sac_alpha, target_entropy=args.sac_target_entropy,
                                   batch_size=args.tc_batch_size, replay_capacity=args.buffer_size,
                                   warmup_macros=args.warmup_macros, updates_per_macro=args.tc_updates,
                                   tolerance=args.gate_tolerance, state_loss_scale=args.loss_cap,
                                   use_gate=self.algorithm == 'FedTCSACGate')
            self.controller = SACController(sac_config, seed=args.seed)
        else:
            self.controller = IntervalController(config, seed=args.seed)
        self.eval_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 202]))
        self.state = macro_state([], 0., 0., 0.)
        self.fallback_count = self.gate_count = 0
        self.macro_count = args.global_rounds // args.macro_length

    def begin_round(self, round_index):
        if round_index % self.args.macro_length:
            return
        macro = round_index // self.args.macro_length
        if self.algorithm == 'FedTCFixed':
            self.action = [self.args.fixed_lambda, self.args.fixed_tau]
            self.gate = {'fallback': False, 'gate_reason': 'fixed_parameters'}
        else:
            self.action, self.gate = self.controller.select_action(self.state, macro)
            self.fallback_count += int(self.gate['fallback'])
            self.gate_count += 1
        self.snapshot = copy.deepcopy(self.global_model)
        self.geometries = []

    def aggregate_round(self, cohort, updates, utilities, round_index):
        result = calibrate(updates, updates.new_tensor(utilities), self.action)
        shadow = None
        macro = round_index // self.args.macro_length
        if (self.args.audit_shadow_every and round_index % self.args.macro_length == 0
                and macro % self.args.audit_shadow_every == 0 and 'candidate' in self.gate):
            # One-step paired check on identical uploads; NEVER provided to the controller.
            candidate = self.gate['candidate']
            candidate = [min(.5, max(0., candidate[0])), min(1.8, max(1.05, candidate[1]))]
            start = vector(self.global_model)
            models = [copy.deepcopy(self.global_model), copy.deepcopy(self.global_model)]
            losses = []
            for model, action in zip(models, (candidate, CONSERVATIVE_ACTION)):
                alternate = calibrate(updates, updates.new_tensor(utilities), action)
                set_vector(model, start + alternate.delta)
                losses.append([self.clients[i].bounded_loss(model, self.args.loss_cap) for i in self.benign_clients])
            shadow = {'scope': 'one_round_same_uploads_not_macro_return', 'macro': macro,
                      'candidate': candidate, 'candidate_minus_conservative_improvement': float(np.mean(np.array(losses[1])-losses[0])),
                      'candidate_loss': float(np.mean(losses[0])), 'conservative_loss': float(np.mean(losses[1])),
                      'gate_reason': self.gate['gate_reason']}
        set_vector(self.global_model, vector(self.global_model) + result.delta)
        self.geometries.append(result.geometry)
        return {'action': self.action, 'weights': result.weights.cpu().tolist(),
                'credibility': result.credibility.cpu().tolist(), 'deviations': result.deviations.cpu().tolist(),
                'scale': result.scale, 'cutoff_radius': result.cutoff_radius,
                'geometric_median_fallback': result.fallback, 'shadow_harness_only': shadow}

    def end_round(self, round_index):
        if (round_index + 1) % self.args.macro_length:
            return
        macro = round_index // self.args.macro_length
        # Independent uniform sample from ALL clients, after the macro completes.
        ids = self.eval_rng.choice(self.num_clients, self.args.eval_clients, replace=False).tolist()
        values = [self.clients[i].report_improvement(self.snapshot, self.global_model, self.args.loss_cap) for i in ids]
        reports = attack_reports(values, [i in self.ad_clients for i in ids],
                                 self.args.report_attack, self.args.loss_cap)
        interval = self.estimator.estimate(reports)
        self.controller.update_statistics((interval.low + interval.high) / 2)
        next_state = macro_state(self.geometries, interval.low, interval.high, (macro+1) / self.macro_count)
        learning = {}
        if self.algorithm != 'FedTCFixed':
            self.controller.remember(self.state, self.action, interval.low, interval.high, next_state,
                                     macro+1 == self.macro_count)
            learning = self.controller.train()
        self.state = next_state
        append_json(self.output / 'macros.jsonl', {
            'macro': macro, 'round_end': round_index+1, 'action': self.action,
            **self.gate, **interval.to_dict(), 'evaluators': ids, 'reports': reports.tolist(),
            'raw_next_state': self.state.tolist(),
            'popart': asdict(self.controller.popart) if hasattr(self.controller, 'popart') else None,
            'honest_sample_mean_harness_only': float(np.mean([v for i, v in zip(ids, values) if i not in self.ad_clients])) if any(i not in self.ad_clients for i in ids) else None,
            'malicious_evaluators_harness_only': sum(i in self.ad_clients for i in ids), **learning})
        print(f'  FedTC macro {macro+1}: {self.gate["gate_reason"]}, CI=[{interval.low:.4f},{interval.high:.4f}]', flush=True)

    def save_extra(self):
        torch.save(self.controller.state_dict(), self.output / 'controller.pt')

    def evaluate_global(self, round_number):
        row = super().evaluate_global(round_number)
        if self.args.audit_validation:
            # Research harness only. These values do not affect actions or rewards.
            losses = [c.bounded_loss(self.global_model, self.args.loss_cap) for c in self.clients]
            append_json(self.output / 'validation_audit.jsonl', {
                'round': round_number, 'all_client_macro_loss': float(np.mean(losses)),
                'honest_client_macro_loss': float(np.mean([losses[i] for i in self.benign_clients])),
                'per_client_loss': losses})
            if round_number == self.global_rounds:
                accuracy = [self.clients[i].test_metrics() for i in self.benign_clients]
                values = [correct/count for correct, count, _ in accuracy]
                count = max(1, math.ceil(.2*len(values)))
                append_json(self.output / 'client_accuracy_audit.jsonl', {
                    'round': round_number, 'honest_client_ids': self.benign_clients, 'accuracy': values,
                    'macro_accuracy': float(np.mean(values)), 'bottom20_client_mean': float(np.mean(sorted(values)[:count])),
                    'note': 'Post-hoc bottom clients; not a predefined minority group.'})
        return row

    def extra_summary(self):
        return {'macros': self.macro_count, 'fallback_macros': self.fallback_count,
                'gate_decisions': self.gate_count, 'controller_train_steps': self.controller.train_steps,
                'td_mode': self.args.td_mode if self.algorithm not in ('FedTCSAC', 'FedTCSACGate') else 'sac_midpoint_reward_raw_returns',
                'ci_bad_bound': self.estimator.bad_bound,
                'ci_sampling_slack': self.estimator.slack}
