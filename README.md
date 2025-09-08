# AlphaPulse

[![PyPI - Version](https://img.shields.io/pypi/v/alphapulse.svg)](https://pypi.org/project/alphapulse)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/alphapulse.svg)](https://pypi.org/project/alphapulse)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install alphapulse
```

## Local development setup

Requirements: Python 3.11+, Git, Hatch, uv.

```bash
# install Hatch and uv (once)
python -m pip install --user hatch uv

# create and enter the dev environment
hatch env create
hatch shell

# install Git hooks and run once across the repo
pre-commit install
pre-commit run --all-files
```

Common checks:

```bash
hatch run lint
hatch run format
hatch run types
hatch run all
```

## Rebuild dev environment

When dependencies change in pyproject.toml, use uv to resync the environment quickly.

```bash
# Fast sync (recommended)
uv sync

# Via Hatch (uses the configured env/installer)
hatch run -e default -- uv sync
```

For a clean rebuild:

```bash
# Remove and recreate the Hatch environment
hatch env remove default
hatch run -e default -- uv sync
```

To add/remove packages and update pyproject.toml automatically:

```bash
# Add dependencies
uv add tenacity loguru requests

# Remove dependencies
uv remove tenacity
```

## License

`alphapulse` is distributed under the terms of the MIT license.
