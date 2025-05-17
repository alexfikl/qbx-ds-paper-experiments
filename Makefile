PYTHON?=python -X dev
MAMBA?=mamba

help: 						## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ run

check-install:				## Check that pytential was correctly installed
	$(PYTHON) -c "import pytential; print(pytential)"
.PHONY: check-install

docker-build:				## Build docker container and run experiments
	docker build -f Dockerfile . -t qbx-direct-solver
.PHONY: docker-build

# }}}

# {{{ pin

conda-packages.txt: conda.yml
	$(MAMBA) create \
		-n qbx-ds-experiments \
		--file $< \
		--dry-run --json >| conda-packages.json
	$(PYTHON) ../scripts/pin-conda-environment.py \
		--outfile $@ \
		conda-packages.json

requirements.txt: ../pyproject.toml
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
