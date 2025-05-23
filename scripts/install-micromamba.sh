#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o xtrace

MINIFORGE_VERSION="25.3.0-3"
MINIFORGE_HOME="${HOME}/.miniforge"
MINIFORGE_ARCH="$(uname)-$(uname -m)"

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}"
MINIFORGE_FILE="Miniforge3-${MINIFORGE_VERSION}-${MINIFORGE_ARCH}.sh"
curl -L -O "${MINIFORGE_URL}/${MINIFORGE_FILE}"
rm -rf "${MINIFORGE_HOME}"
bash "${MINIFORGE_FILE}" -b -p "${MINIFORGE_HOME}"

source "${MINIFORGE_HOME}/bin/activate"
mamba create --yes --name paper --file conda-packages.txt

source activate paper

python -m pip install --upgrade pip hatchling editables wheel
python -m pip install -r requirements.txt
mamba clean --all --yes
