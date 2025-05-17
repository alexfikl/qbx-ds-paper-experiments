Introduction
------------

This contains a `docker` image and some scripts in the `experiments` folder
that are used to generate the results from the paper in a reproducible manner.

Building Docker Image
---------------------

To build the image, start your docker daemon and just do

.. code:: bash

    docker build -f Dockerfile . -t qbx-direct-solver

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

    make pin-deps

Reproducing results
-------------------

TODO
