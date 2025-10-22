#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import pathlib
import re
import subprocess

import common_ds_tools as ds


dirname = pathlib.Path(__file__).parent

log = ds.set_up_logging("ds")
RE_ARCHIVE_NAME = re.compile(
    r"(\d{4}-\d{2}-\d{2}-)?(?P<script>.*)-(?P<dim>\d)(?P<lpot>[ds])-?(?P<suffix>.*).npz"
)


def visualize(
    npz: str = ".",
    ext: str = "pdf",
    *,
    overwrite: bool = False,
) -> int:
    npz = pathlib.Path(npz)
    m = RE_ARCHIVE_NAME.match(npz.name)
    if not m:
        log.error("Filename did not match pattern: '%s'.", npz.name)
        return 1

    dim = int(m["dim"])
    lpot_type = m["lpot"]
    suffix = m["suffix"]

    script = dirname / f"{m['script']}.py"
    if not script.exists():
        log.error("Script does not exist: '%s'.", script)
        return 1

    log.info("Calling script '%s'.", script.name)

    # gather arguments
    args = ["python"]
    if not __debug__:
        args += ["-O"]

    args += [str(script), "--filename", str(npz), "--suffix", suffix, "--ext", ext]
    if overwrite:
        args += ["--overwrite"]

    args += ["--ambient_dim", str(dim), "--lpot_type", lpot_type]
    subprocess.run(args, check=True)

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize experiment results from npz files"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories containing npz files",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite any existing plots",
    )
    parser.add_argument(
        "-e",
        "--ext",
        default="pdf",
        help="Extension used for visualization",
    )
    args = parser.parse_args()

    filenames = {}
    for path in args.paths:
        p = pathlib.Path(path).resolve()
        if p.is_dir():
            filenames.update(dict.fromkeys(p.glob("*.npz")))
            continue

        if p.suffix == ".npz":
            filenames[p] = None
        else:
            log.warning("Unsupported file type (expected 'npz'): '%s'", p)

    errno = 0
    for filename in filenames:
        errno |= visualize(filename, ext=args.ext, overwrite=args.overwrite)

    raise SystemExit(errno)

# vim: fdm=marker
