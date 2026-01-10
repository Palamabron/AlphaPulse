.PHONY: sync-core sync-dev format lint types test all nb-format nb-lint nb-types deadcode precommit

sync-core:
	uv sync

sync-dev:
	uv sync --extra dev

format:
	uv run ruff check --select I --fix src tests
	uv run ruff format .

lint:
	uv run ruff check src tests
	uv run ruff format --check .

types:
	uv run mypy --install-types --non-interactive src/alphapulse tests

test:
	uv run pytest

# --- Notebooks (.ipynb) via nbQA ---
# nbQA expects: nbqa <tool> <notebook(s)/dir(s)> -- <tool args>. [web:95]
# Ruff formatter is a subcommand: ruff format ... [web:32]
NB := $(shell find notebooks -type f -name "*.ipynb" 2>/dev/null)

nb-format:
	@if [ -z "$(NB)" ]; then echo "No notebooks to format."; exit 0; else \
		uv run nbqa ruff $(NB) -- check --fix; \
		uv run nbqa ruff $(NB) -- format; \
	fi

nb-lint:
	@if [ -z "$(NB)" ]; then echo "No notebooks to lint."; exit 0; else \
		uv run nbqa ruff $(NB) -- check; \
		uv run nbqa ruff $(NB) -- format --check; \
	fi

nb-types:
	@if [ -z "$(NB)" ]; then echo "No notebooks to type-check."; exit 0; else \
		uv run nbqa mypy $(NB) -- --install-types --non-interactive; \
	fi

# --- Dead code ---
deadcode:
	uv run vulture src --min-confidence 80

all: lint nb-lint types nb-types test deadcode

# Convenience target to run the same things pre-commit will run
precommit: lint nb-lint types deadcode
