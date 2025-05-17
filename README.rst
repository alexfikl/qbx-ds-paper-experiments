Introduction
------------

This contains a `docker` image and some scripts in the `experiments` folder
that are used to generate the results from the paper in a reproducible manner.
Other code can be found in the `dsplayground` folder, but that is not pinned in
any way, for more fun!

Building Docker Image
---------------------

To build the image, start your docker daemon and just do

.. code:: bash

    docker build -f Dockerfile . -t qbx-direct-solver

This contains a list of hardcoded dependencies at the time of writing to
ensure reproducible results. To update the versions to the latest ones, do

.. code:: bash

    make pin-deps

Running Scripts
---------------

.. note::

    Work In Progress
