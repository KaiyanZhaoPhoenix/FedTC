# Runnable implementation

Run commands in this directory. Start with [installation and reproduction](../docs/reproduction.md), [method versions](../docs/method.md), and [SAC definition](../docs/sac.md).

- Entry point: `main.py --help`.
- Exact study data: `dataset/prepare_mnist.py`; no generated data are bundled.
- Tests: `python -m unittest discover -s tests -v`.
- Archived measurements: `python verify_published.py`.
- New six-baseline comparisons: `run_comparison.py`.
- New single-seed scenario suite: `run_scenarios.py`.
- New four-way SAC suite: `run_sac_study.py`.
- Replot a published archive: `report_published.py --help`.
- `upstream_main.py` is the upstream reference, not the default experiment entry.

Original report scripts verify full new-run logs and checkpoints. The published archive omits binary checkpoints and compresses logs; use `verify_published.py` / `report_published.py` for that archive.
