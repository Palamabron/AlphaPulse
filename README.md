# AlphaPulse

Automated, scriptable experiment runner for the Numerai Tournament — designed to quickly iterate over data → features → model → validation → submission workflows and identify the best-performing pipeline.

***

## What it is

AlphaPulse is a lightweight Python package/repo for running reproducible Numerai experiments end-to-end: fetching/reading data, generating features, training models, evaluating, and optionally producing artifacts needed for submissions.

Typical uses:
- Sweep over feature sets and model hyperparameters.
- Compare validation schemes (e.g., era-based splits) across many runs.
- Keep experiment runs consistent and repeatable (same code path, same environment).

## Installation

```console
pip install alphapulse
```

## Local development setup (uv)

Requirements: Python 3.11+, Git, uv.

```bash
# Install uv (once)
python -m pip install --user uv

# Create/update the virtual environment from uv.lock (recommended)
uv sync --extra dev

# Install Git hooks
uv run pre-commit install

# (Optional) Run all hooks once across the repo
uv run pre-commit run --all-files
```

Notes:
- `uv sync` synchronizes your environment to the lockfile and keeps the virtual environment consistent with it.
- `ruff format --check` verifies formatting without modifying files and exits non-zero if changes would be needed.

## Makefile commands

Common commands (run from repo root):

```bash
make sync
make format
make lint
make types
make test
make deadcode
make nb-format
make nb-lint
make nb-types
make all
```

What each target does:
- `make sync`: Syncs the local virtual environment from `uv.lock` (reproducible dev env).
- `make format`: Auto-fixes lint issues where safe (e.g., imports) and formats Python code in-place.
- `make lint`: Runs lint checks and verifies formatting (no file modifications).
- `make types`: Runs static type checking via mypy.
- `make test`: Runs the test suite via pytest.
- `make deadcode`: Runs dead-code detection (Vulture) on `src/` and `tests/`.
- `make nb-format`: Formats Jupyter notebooks (`.ipynb`) via nbQA + Ruff (in-place).
- `make nb-lint`: Lints Jupyter notebooks via nbQA + Ruff and checks notebook formatting.
- `make nb-types`: Runs mypy checks on notebooks via nbQA.
- `make all`: Runs the typical “CI-style” set of checks (lint + notebooks + types + tests + deadcode).

## Rebuild / update environment

When dependencies change in `pyproject.toml`:

```bash
# Update lockfile if needed
uv lock

# Sync environment to the lockfile
uv sync
```

## Conventional Commits

This repository uses the Conventional Commits specification for commit messages:

`<type>[optional scope]: <description>`

Examples:
- `feat: add experiment runner scaffold`
- `fix: handle empty notebooks directory in nbqa targets`
- `chore: simplify pre-commit configuration`
- `docs: update README`

More information here: https://www.conventionalcommits.org/en/v1.0.0/

## License

`alphapulse` is distributed under the terms of the MIT license.
