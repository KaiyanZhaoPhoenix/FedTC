# SAC controller and four-way diagnostic

Paths and commands below are relative to `code/`. This document describes the completed implementation; the paired relative-benefit gate is only a proposal in `next_version.md`.

The original `algorithms/fedtc/controller.py` is preserved. `FedTC` still uses its
interval actor/critic and original `paper` TD option. `FedTCFixed` keeps the old
default `(0, 1.35)`, with optional `--fixed-lambda` and `--fixed-tau` overrides.
`FedTCSAC` and `FedTCSACGate` use the new `algorithms/fedtc/sac.py`.

All methods use the native FedAA data, client SGD and shared evaluation protocol.
FedTC variants never read the server root dataset. The new variants also receive
client utility reports; their upload, local-training and evaluation RNG streams
are separate from controller randomness.

## SAC definition

The stochastic actor is a two-layer 64-unit SiLU MLP producing a diagonal Gaussian.
Its samples pass through tanh and an affine map to lambda in `[0, .5]` and tau in
`[1.05, 1.8]`. The tanh Jacobian is included in the log probability. Entropy and its
target `-2` are defined in the normalized action coordinates, before affine scaling.
The initial policy mean is approximately `(.06, 1.35)`.

SAC uses the observed interval midpoint `(L+U)/2` as its scalar reward. Both soft
critics estimate this SAME objective, using the usual minimum target plus entropy.
The temperature starts at .1 and is learned. Targets update at rate .005. There
is no online PopArt transform in this SAC implementation: returns remain in raw
loss-improvement units, and state coordinates containing the prior lower endpoint
and interval width are divided by the fixed loss cap. All other state fields are
unchanged.

Two ADDITIONAL critics fit L and U with entropy-free TD targets. The optional gate
compares the candidate's lower estimate to the conservative action's upper estimate,
with tolerance .05 in raw return units. It does not use the soft Q values. Each
endpoint critic bootstraps a sampled future SAC proposal, so the conservative
comparison describes a conservative first action followed by SAC proposals, not
the return of always using the conservative action. Neither endpoint is a proved
confidence bound on learned policy value. This gate is an explicitly heuristic
diagnostic, not a new safety theorem.

Both SAC variants train all four critics, using identical optimizers and sampling
streams. Only executing or bypassing the gate differs. Each critic's gradient is
clipped independently, so large interval losses cannot rescale the soft-Q gradients.
The replay contains the ACTUALLY EXECUTED action. Warmup uses `(0,1.35)`; after
warmup the ungated variant executes stochastic SAC proposals.

For compatibility, the run summary's legacy `gate_decisions` field counts calls
to the action selector, including warmup and bypassed gates. Use `gate_reason`
and the report's post-warmup counts to distinguish actual gate tests, ungated
executions and initialization. `gate_would_accept` in the ungated run is a shadow
decision on that run's own state distribution, not a second deployed policy.

The current original controller uses a different reward objective and moving
normalization. Comparing it to SAC changes more than just the policy optimizer.
The clean causal gate comparison is `FedTCSAC` versus `FedTCSACGate`.

Algorithm references:
- https://spinningup.openai.com/en/latest/algorithms/sac.html
- https://arxiv.org/abs/1812.05905

## Predeclared study

Run `.venv/bin/python run_sac_study.py --output-root runs/NEW_DIRECTORY --jobs 4`.
It performs these stages in order:

1. Seed 1; 9 fixed choices from lambda `{0,.25,.5}` x tau `{1.05,1.35,1.8}`;
   80 rounds each. Choose by mean all-client private validation capped loss in the
   final 20% of rounds, under truthful reports. No test score selects parameters.
2. Four methods at 300 rounds each, with truthful validation reports.
3. The same four methods at 300 rounds each for All-Max, Echo and Bimodal reports.

All stages keep same-value model poisoning, fake utility, and the same designated
20% adversary IDs ON. Thus "honest feedback" does not mean an attack-free FL task.
The main comparison isolates validation-report attacks. Native MNIST uses all 100
clients, batch16, one local epoch, lr .1, 10% private validation, H=2, m=20, and
evaluation every 10 rounds. All learned controllers use batch16, warmup16, four
updates/macro, gamma .9, learning rates3e-4 and replay capacity512. These settings
are predeclared, not tuned after looking at outcomes. The historical controller
implementation is preserved, but these exploratory training settings differ from
the earlier batch2 experiments.

The primary metric is mean honest-client sample-weighted test accuracy over the
last 20% of training (six checkpoints). Final accuracy, honest-client macro
accuracy, and the mean of the bottom 20% of honest-client accuracies are secondary.
The last metric is a post-hoc distribution summary, not a predefined minority
group or a minority-protection guarantee. Fixed-grid tuning has an additional
720-round budget, explicitly reported rather than hidden as equal total compute.

Every 25 macros, the research harness constructs candidate and conservative models
from exactly the same uploaded updates, evaluates both on honest private validation,
and records their paired ONE-ROUND loss difference. These checks never enter the
controller. They do not establish the sign of an H-round or discounted-return
advantage, and use known honest identities only for offline diagnosis.

Each output directory includes a protocol plan and source hashes before execution,
tuning selection, exact commands, all raw run logs, checkpoints and comparison
audits. One seed supplies exploratory evidence only. No convergence, significance,
general-dataset superiority, or full-trajectory safety claim follows from it.

## Checks

`.venv/bin/python -m unittest discover -s tests -v`

Tests cover terminal masking, entropy exclusion from gate targets, action density
and bounds, identical SAC learning under identical replay with gate toggled,
checkpoint/RNG restoration, actual executed-action storage, and native-framework
integration without server-root access. Original mathematical and integration tests
remain in place.
