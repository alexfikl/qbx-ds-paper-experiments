#!/usr/bin/env python

from __future__ import annotations

import logging
import pathlib

import rich.logging


log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
This converts the output of `conda create --json` to a simple list of pinned
packages. These can then be used to create a new reproducible (ish) environment
using

    > conda create --yes --name my-env-name --file pinned.txt

Examples:

    > {SCRIPT_PATH.name} --outfile conda.lock.txt conda.json
"""


def convert_conda_json(
    infile: pathlib.Path,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if not infile.exists():
        log.error("File does not exist: '%s'.", infile)
        return 1

    if outfile is None:
        outfile = infile.with_suffix(".txt")

    if not overwrite and outfile.exists():
        log.error("Output file already exists (use '--force'): '%s'.", outfile)

    import json

    with open(infile, encoding="utf-8") as infd:
        packages = json.load(infd)

    with open(outfile, "w", encoding="utf-8") as outfd:
        outfd.write("@EXPLICIT\n")
        for package in packages["actions"]["FETCH"]:
            outfd.write(f"{package['url']}\n")

    return 0


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    raise SystemExit(
        convert_conda_json(
            args.filename,
            outfile=args.outfile,
            overwrite=args.force,
        )
    )
