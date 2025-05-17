#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
import subprocess

import common_ds_tools as ds


log = ds.set_up_logging("ds")
DS_EXPERIMENTS = {
    "ds-backward-accuracy.py",
    "ds-backward-dof-scaling.py",
    "ds-forward-accuracy.py",
    "ds-forward-model.py",
    "ds-geometry.py",
}


def run(
    path: pathlib.Path,
    *,
    ext: str = "pdf",
    suffix: str = "",
    overwrite: bool = False,
    visualize: bool = False,
) -> int:
    if not path.exists():
        log.error("No script at location: '%s'", path)

    # gather arguments
    args = ["python"]
    if not __debug__:
        args += ["-O"]

    args += [str(path), "--suffix", suffix, "--ext", ext]
    if overwrite:
        args += ["--overwrite"]

    if not visualize:
        args += ["--no-visualize"]

    # run
    from itertools import product

    ret = 0
    for ambient_dim, lpot_type in product([2, 3], ["s", "d"]):
        try:
            subprocess.run(
                [*args, "--ambient_dim", str(ambient_dim), "--lpot_type", lpot_type],
                check=True,
            )
        except subprocess.CalledProcessError:
            log.error(
                "Failed to run '%s' with (ambient_dim=%d, lpot_type='%s').",
                path.name,
                ambient_dim,
                lpot_type,
            )
            ret = 1

    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a single or multiple experiments")
    parser.add_argument(
        "paths",
        nargs="*",
        default=None,
        help="Paths to experiments to run",
    )
    ds.add_arguments(parser, filename=False)
    args = parser.parse_args()

    if not args.paths:
        args.paths = list(DS_EXPERIMENTS)

    errno = 0
    for path in args.paths:
        p = pathlib.Path(path).resolve()
        if p.name not in DS_EXPERIMENTS:
            log.error("Path not a known experiment: '%s'", path)
            errno = 1

        log.info("Running '%s'", p)
        ret = run(
            p,
            ext=args.ext,
            suffix=args.suffix,
            overwrite=args.overwrite,
            visualize=not args.no_visualize,
        )
        errno |= ret

        if ret:
            log.error("Experiment failed: '%s'", p)

    raise SystemExit(errno)
