# Repository guidance

- `code/` is the runnable FedAA-based implementation; run Python commands there.
- `paper/` contains the ICML 2026 manuscript; run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` there.
- `docs/` documents versions, limitations and proposals. `docs/next_version.md` is NOT implemented behavior.
- `results/` contains published, single-seed exploratory records. Do not alter measured values or present them as multi-seed confirmation.
- Preserve `FedTC`, `FedTCFixed`, `FedTCSAC`, `FedTCSACGate` and native FedAA baseline semantics unless an explicit methodological change is requested.
- Keep data splits, training/evaluation populations and random streams matched for comparisons. Never use test data or known honest identities as controller input.
- Validate code changes with `python -m unittest discover -s tests -v` from `code/`; validate the archive with `python verify_published.py`.
- Verify manuscript builds, references and page layout; mark placeholders and unverified claims explicitly.
- Keep datasets, checkpoints, environments, caches and TeX intermediates out of Git. Do not commit credentials or local machine paths.
- Retain third-party attribution and template notices. Do not assign a new license to upstream code.
