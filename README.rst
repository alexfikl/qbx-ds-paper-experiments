Introduction
------------

.. |badge-license| image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://spdx.org/licenses/MIT.html
    :alt: MIT License

.. |badge-zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.15487041.svg
    :target: https://doi.org/10.5281/zenodo.15487041
    :alt: Zenodo repository

|badge-license| |badge-zenodo|

This contains a Docker image and some scripts that are used to generate the
results from the paper in a reproducible manner.

Building Docker Image
---------------------

To build the image, start your docker daemon and just do

.. code:: bash

    docker buildx build -f Dockerfile . -t qbx-ds-experiments

or, equivalently

.. code:: bash

    make docker-build

Reproducing results
-------------------

To run the scripts in the ``experiments`` folder you need to go through the following
steps

.. code:: bash

    # 1. build the container as above
    # 2. enter the container using:
    make docker-run
    # 3. inside the container:
    source "${HOME}/.miniforge/bin/activate" paper
    make run
    make visualize

The first ``make run`` command will run all the experiments and generate a set of
``npz`` files with the results. This can take a very long time and uses all
available CPU cores (through OpenCL and PoCL), so make sure to have the necessary
resources. The results can then be visualized with ``make visualize``.

You can of course also run the scripts individually by calling the respective
file. The ``make run`` command will automatically run the experiments over 2D and
3D and over a single-layer and double-layer potential to generate all the results
from the paper.

Locked dependencies
-------------------

The repository contains a list of hardcoded dependencies at the time of writing
to ensure reproducible results. They can be found in

* ``conda-packages.txt``: base system Conda dependencies.
* ``requirements.txt``: additional Python dependencies.

To update the versions to the latest ones, you can use

.. code:: bash

    make -B pin
