FROM debian:bullseye
LABEL maintainer="alexfikl@gmail.com"

RUN apt-get update \
    && apt-get install --no-install-recommends -y make curl ca-certificates texlive-base texlive-pictures texlive-science \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash paper
USER paper

# install latexrun script
RUN mkdir /home/paper/bin
WORKDIR /home/paper/bin

# install conda environment
RUN mkdir /home/paper/qbx-ds-experiments
WORKDIR /home/paper/qbx-ds-experiments

COPY --chown=paper scripts scripts
COPY --chown=paper requirements.txt requirements.txt
COPY --chown=paper conda-packages.txt conda-packages.txt
RUN bash scripts/install-micromamba.sh

# copy the scripts for running the experiments
COPY --chown=paper experiments experiments
COPY --chown=paper Makefile Makefile

# use Pocl in the docker container
ENV PYOPENCL_CTX=portable
ENV PYOPENCL_TEST=portable
