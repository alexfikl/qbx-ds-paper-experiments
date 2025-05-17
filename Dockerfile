FROM debian:bullseye
LABEL maintainer="alexfikl@gmail.com"

RUN apt-get update \
    && apt-get install --no-install-recommends -y make curl ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash paper
USER paper

# install latexrun script
RUN mkdir /home/paper/bin
WORKDIR /home/paper/bin

# install conda environment
RUN mkdir /home/paper/qbx-direct-solver-results
WORKDIR /home/paper/qbx-direct-solver-results

COPY --chown=paper install.sh install.sh
COPY --chown=paper requirements.txt requirements.txt
COPY --chown=paper conda-packages.txt conda-packages.txt
RUN bash install.sh

# copy the scripts for running the experiments
# COPY --chown=paper experiments experiments
COPY --chown=paper Makefile Makefile
