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

## License

`alphapulse` is distributed under the terms of the MIT license.
