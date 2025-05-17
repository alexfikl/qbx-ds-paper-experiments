PYTHON?=nice python -X dev -O
MAMBA?=mamba

help: 						## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ lint

format: black isort					## Run all formatting scripts
.PHONY: format

fmt: format
.PHONY: fmt

black:								## Run ruff format over the source code
	ruff format experiments/*.py scripts/*.py
	@echo -e "\e[1;32mruff format clean!\e[0m"
.PHONY: black

isort:								## Run ruff isort fixes over the source code
	ruff check --fix --select=I experiments/*.py scripts/*.py
	@echo -e "\e[1;32mruff isort clean!\e[0m"
.PHONY: isort

lint: typos ruff pylint				## Run all linting scripts
.PHONY: lint

typos:								## Run typos over the source code and documentation
	@typos
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

ruff:								## Run ruff checks over the source code
	ruff check experiments/*.py scripts/*.py
	@echo -e "\e[1;32mruff lint clean!\e[0m"
.PHONY: ruff

pylint:								## Run pylint checks over the source code
	$(PYTHON) -m pylint experiments/*.py scripts/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"
.PHONY: pylint

# }}}

# {{{ run

check-install:				## Check that pytential was correctly installed
	$(PYTHON) -c "import pytential; print(pytential)"
.PHONY: check-install

docker-build:				## Build docker container and run experiments
	docker build -f Dockerfile . -t qbx-direct-solver
.PHONY: docker-build

run:			## Run all experiments
	$(PYTHON) experiments/ds-run.py --no-visualize
.PHONY: run

visualize:		## Generate paper worthy plots for all cached results
	DS_QBX_NO_TIMESTAMP=1 $(PYTHON) experiments/ds-visualize.py --ext pdf
	DS_QBX_NO_TIMESTAMP=1 $(PYTHON) experiments/ds-geometry.py --ext pdf --ambient_dim 2
	DS_QBX_NO_TIMESTAMP=1 $(PYTHON) experiments/ds-geometry.py --ext pdf --ambient_dim 3
.PHONY: visualize

# }}}

# {{{ pin

conda-packages.txt: conda.yml
	$(MAMBA) create \
		-n qbx-ds-experiments \
		--file $< \
		--dry-run --json >| conda-packages.json
	$(PYTHON) scripts/pin-conda-environment.py \
		--outfile $@ \
		conda-packages.json

requirements.txt: pyproject.toml
	uv pip compile --upgrade --universal --python-version '3.10' \
		--extra git \
		--output-file $@ $<

pin: conda-packages.txt requirements.txt	## Generate pinned dependencies
	cat $^
.PHONY: pin

# }}}

clean:						## Remove temporary files
	rm -rf conda-packages.json
.PHONY: clean

purge: clean				## Remove all generated files
	rm -rf conda-packages.txt requirements.txt
.PHONY: purge
