PYTHON := uv run
RUFF := $(PYTHON) ruff
MYPY := $(PYTHON) mypy --config-file pyproject.toml

.PHONY: fmt
fmt:
	$(RUFF) check --select I --fix src tests scripts
	$(RUFF) format .

.PHONY: types
types:
	$(MYPY) src/alphapulse tests

.PHONY: lint
lint:
	$(RUFF) check src tests scripts
	$(RUFF) format --check .

.PHONY: test
test:
	$(PYTHON) pytest \
		tests/ \
		-v --tb=short \
		--cov=alphapulse \
		--cov-report=term-missing

# --- Notebooks (.ipynb) via nbQA ---
NB := $(shell find notebooks -type f -name "*.ipynb" 2>/dev/null)

.PHONY: nb-format
nb-format:
	@if [ -z "$(NB)" ]; then echo "No notebooks to format."; exit 0; else \
		uv run nbqa ruff $(NB) -- check --fix; \
		uv run nbqa ruff $(NB) -- format; \
	fi

.PHONY: nb-lint
nb-lint:
	@if [ -z "$(NB)" ]; then echo "No notebooks to lint."; exit 0; else \
		uv run nbqa ruff $(NB) -- check; \
		uv run nbqa ruff $(NB) -- format --check; \
	fi

.PHONY: nb-types
nb-types:
	@if [ -z "$(NB)" ]; then echo "No notebooks to type-check."; exit 0; else \
		uv run nbqa mypy $(NB) -- --install-types --non-interactive; \
	fi

# --- Dead code ---
.PHONY: deadcode
deadcode:
	uv run vulture src --min-confidence 80

.PHONY: eda-lint
eda-lint:
	$(RUFF) check eda --config pyproject.toml \
		--extend-exclude "" \
		--select E,F,W,I,UP
	$(RUFF) format --check eda --config pyproject.toml \
		--extend-exclude ""

.PHONY: check
check: lint types test deadcode

.PHONY: precommit
precommit: lint nb-lint types deadcode
