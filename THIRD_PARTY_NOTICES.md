# Third-party provenance and license scope

The root `LICENSE` is preserved verbatim from `KaiyanZhaoPhoenix/FedTC` commit `1d74601b018a31e559347b93249b206265c36d9a` (MIT, copyright 2025 Ariatao). Preserving it does not relicense independently sourced third-party material or the manuscript.

## FedAA framework

- Source: https://github.com/Gp1g/FedAA
- Pinned commit: `1fba884934cbec1b3506812d9d612e6a181b1d79`.
- Original README: [FedAA_README.md](docs/upstream/FedAA_README.md).
- Reused and adapted paths: `code/DDPG/`, `code/clients/`, `code/servers/`, `code/utils/`, `code/dataset/`, and `code/upstream_main.py`.
- Exact unchanged component hashes and adaptations: [source_provenance.json](docs/source_provenance.json).
- No license file was present in the inspected upstream snapshot. This repository records that fact; it does not assert that FedAA-derived files are covered by the root MIT license. Consult upstream for its licensing terms before further redistribution.

## Paper assets and template

`paper/icml2026.sty`, `paper/icml2026.bst`, `paper/algorithm.sty`, `paper/algorithmic.sty`, and `paper/fancyhdr.sty` retain their original embedded notices. Manuscript text and figures are research draft materials. Historical figures are catalogued in `paper/content/figs/README.md`; they are not automatically verified experimental evidence.

## Data and dependencies

MNIST data are downloaded separately from the pinned FedAA data snapshot, not bundled here. Its upstream generator uses torchvision MNIST and combines the official train/test pools before repartitioning. External Python dependencies retain their own licenses.
