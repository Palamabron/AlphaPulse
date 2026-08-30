PYTHON := uv run
RUFF := $(PYTHON) ruff
MYPY := $(PYTHON) mypy --config-file pyproject.toml
THESIS_JOBNAME := AlphaPulse_master_thesis
THESIS_BUILD_DIR := $(CURDIR)/master_thesis/build/final
THESIS_OUTPUT_DIR := $(CURDIR)/output/pdf
TIMES_NEW_ROMAN_DIR ?= /home/kuba/.cache/alphapulse-fonts

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
ifneq ($(wildcard notebooks),)
NB := $(shell find notebooks -type f -name "*.ipynb" 2>/dev/null)
else
NB :=
endif

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
		--exclude "" \
		--select E,F,W,I,UP
	$(RUFF) format --check eda --config pyproject.toml \
		--exclude ""

.PHONY: check
check: lint types test deadcode

.PHONY: precommit
precommit: lint nb-lint types deadcode

.PHONY: thesis
thesis:
	@mkdir -p "$(THESIS_BUILD_DIR)" "$(THESIS_OUTPUT_DIR)"
	@for font_face in times.ttf timesbd.ttf timesi.ttf timesbi.ttf; do \
		if [ ! -f "$(TIMES_NEW_ROMAN_DIR)/$$font_face" ]; then \
			echo "Missing Times New Roman face: $(TIMES_NEW_ROMAN_DIR)/$$font_face"; \
			echo "See master_thesis/FONTS.md before building the submission PDF."; \
			exit 1; \
		fi; \
		cp "$(TIMES_NEW_ROMAN_DIR)/$$font_face" "$(CURDIR)/master_thesis/build/$$font_face"; \
	done
	latexmk -cd -xelatex \
		-interaction=nonstopmode \
		-halt-on-error \
		-file-line-error \
		-jobname="$(THESIS_JOBNAME)" \
		-outdir="$(THESIS_BUILD_DIR)" \
		master_thesis/main.tex
	@cp "$(THESIS_BUILD_DIR)/$(THESIS_JOBNAME).pdf" \
		"$(THESIS_OUTPUT_DIR)/$(THESIS_JOBNAME).pdf"
