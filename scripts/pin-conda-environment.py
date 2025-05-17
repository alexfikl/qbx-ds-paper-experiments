#!/usr/bin/env python

from __future__ import annotations

import pathlib


def convert_conda_json(
    infile: pathlib.Path, outfile: pathlib.Path | None = None
) -> int:
    if outfile is None:
        outfile = infile.with_suffix(".txt")

    if not infile.exists():
        print(f"file does not exist: '{infile}'")
        return 1

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

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    args = parser.parse_args()

    raise SystemExit(convert_conda_json(args.filename, outfile=args.outfile))
