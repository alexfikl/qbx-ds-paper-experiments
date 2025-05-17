Introduction
------------

This contains a `docker` image and some scripts in the `experiments` folder
that are used to generate the results from the paper in a reproducible manner.

Building Docker Image
---------------------

To build the image, start your docker daemon and just do

.. code:: bash

    docker buildx build -f Dockerfile . -t qbx-ds-experiments

or, equivalently

.. code:: bash

    make docker-build

Locked dependencies
-------------------

This contains a list of hardcoded dependencies at the time of writing to
ensure reproducible results. They can be found in

* ``conda-packages.txt``: base system Conda dependencies.
* ``requirements.txt``: additional Python dependencies.

To update the versions to the latest ones, you can use

.. code:: bash

    make -B pin

Reproducing results
-------------------

To run the scripts in the ``experiments`` folder you need to go trough the followin
steps

.. code:: bash

    # enter the container that was built as above
    make docker-run
    # inside the containter:
    source "${HOME}/.miniforge/bin/activate" paper
    make run
    make visualize

The first command ``make run`` will run all the experiments and generate a set of
``npz`` files with the results. These can then be visualized with ``make visualize``.

You can of course also run the scripts individually by calling the respective
file. The ``make run`` command will automatically run the experiments over 2D and
3D and over a single-layer and double-layer potential to generate all the results
from the paper.
