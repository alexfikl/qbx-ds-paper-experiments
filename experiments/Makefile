PYTHON?=nice python -X dev -O
PYTEST_ADDOPTS?=
DIM?=3

all: help

help: 						## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

format: black isort					## Run all formatting scripts
.PHONY: format

fmt: format
.PHONY: fmt

black:								## Run ruff format over the source code
	ruff format *.py
	@echo -e "\e[1;32mruff format clean!\e[0m"
.PHONY: black

isort:								## Run ruff isort fixes over the source code
	ruff check --fix --select=I *.py
	@echo -e "\e[1;32mruff isort clean!\e[0m"
.PHONY: isort

lint: typos ruff pylint				## Run all linting scripts
.PHONY: lint

typos:								## Run typos over the source code and documentation
	@typos
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

ruff:								## Run ruff checks over the source code
	ruff check *.py
	@echo -e "\e[1;32mruff lint clean!\e[0m"
.PHONY: ruff

pylint:								## Run pylint checks over the source code
	$(PYTHON) -m pylint *.py
	@echo -e "\e[1;32mpylint clean!\e[0m"
.PHONY: pylint

# }}}

# {{{ generate figures

run:			## Run all experiments
	$(PYTHON) ds-run.py --no-visualize
.PHONY: run

visualize:		## Generate paper worthy plots for all cached results
	DS_QBX_NO_TIMESTAMP=1 $(PYTHON) ds-visualize.py --ext pdf --overwrite
	DS_QBX_NO_TIMESTAMP=1 $(PYTHON) ds-geometry.py --ext pdf --overwrite --ambient_dim 2
	DS_QBX_NO_TIMESTAMP=1 $(PYTHON) ds-geometry.py --ext pdf --overwrite --ambient_dim 3
.PHONY: visualize

# }}}
