#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import common_ds_tools as ds


if TYPE_CHECKING:
    from meshmode.array_context import PyOpenCLArrayContext
    from pytential.collection import GeometryCollection


scriptname = pathlib.Path(__file__)
log = ds.set_up_logging("ds")
ds.set_recommended_matplotlib()


# {{{ run


@dataclass(frozen=True)
class ExperimentResult:
    id_eps: float
    error: float

    proxy_count: int
    proxy_radius_factor: float

    parameters: ds.ExperimentParameters


def run(
    actx: PyOpenCLArrayContext,
    places: GeometryCollection,
    param: ds.ExperimentParameters,
    *,
    rng: np.random.Generator | None = None,
) -> ExperimentResult:
    if rng is None:
        rng = ds.seeded_rng(seed=42)

    # {{{ retrieve discretization

    dd = places.auto_source
    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)

    # }}}

    # {{{ construct skeletonization wrangler

    from pytential import bind, sym

    k = param.helmholtz_k
    if k == 0:
        from sumpy.kernel import LaplaceKernel

        knl = LaplaceKernel(places.ambient_dim)
        kernel_arguments = {}
        context = {}
    else:
        from sumpy.kernel import HelmholtzKernel

        knl = HelmholtzKernel(places.ambient_dim, helmholtz_k_name="k")
        kernel_arguments = {"k": sym.var("k")}
        context = {"k": k}

    sym_u = sym.var("sigma")

    if param.lpot_type == "s":
        sym_op = sym.S(
            knl, sym_u, qbx_forced_limit=-1, kernel_arguments=kernel_arguments
        )
    elif param.lpot_type == "d":
        sym_op = -sym_u / 2 + sym.D(
            knl, sym_u, qbx_forced_limit="avg", kernel_arguments=kernel_arguments
        )
    else:
        raise ValueError(f"unknown layer potential type: '{param.lpot_type}'")

    # }}}

    # {{{ evaluate errors

    from pytential.linalg.hmatrix import build_hmatrix_by_proxy
    from pytools import ProcessTimer

    with ProcessTimer() as p:
        wrangler = build_hmatrix_by_proxy(
            actx,
            places,
            sym_op,
            sym_u,
            domains=[dd],
            context=context,
            id_eps=param.id_eps,
            rng=rng,
            _max_particles_in_box=param.max_particles_in_box,
            _approx_nproxy=param.proxy_approx_count,
            _proxy_radius_factor=param.proxy_radius_factor,
        )
        hmat = wrangler.get_backward()

    log.info("[construction] time: %s", p)

    x_ref = ds.make_uniform_random_array(actx, density_discr, rng=rng)
    b_ref = bind(places, sym_op, auto_where=dd)(actx, sigma=x_ref)

    from meshmode.dof_array import flat_norm

    with ProcessTimer() as p:
        x_hmat = hmat @ b_ref
        error = actx.to_numpy(flat_norm(x_hmat - x_ref) / flat_norm(x_ref))

    log.info("[solve] time: %s", p)

    # }}}

    return ExperimentResult(
        id_eps=param.id_eps,
        error=error,
        proxy_count=wrangler.proxy.nproxy,
        proxy_radius_factor=wrangler.proxy.radius_factor,
        parameters=param,
    )


# }}}


# {{{


def experiment_run(
    actx_factory,
    *,
    suffix: str = "",
    ext: str = "png",
    overwrite: bool = False,
    visualize: bool = False,
    **kwargs,
) -> int:
    actx = actx_factory()
    eoc = ds.EOCRecorder("Accuracy", "id_eps")

    from dataclasses import replace

    ambient_dim = kwargs.pop("ambient_dim", 3)
    if ambient_dim == 2:
        param = ds.ExperimentParameters2(**kwargs)
    else:
        param = ds.ExperimentParametersTorus3(**kwargs)

    filename = ds.make_archive(scriptname, param, suffix=suffix)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    nruns = 3
    id_eps = 10.0 ** (-np.arange(2, 16))
    error = np.empty((nruns, id_eps.size), dtype=id_eps.dtype)
    places = ds.make_geometry_collection(actx, param)

    # NOTE: adding a few more proxy points just to be on the safe side
    proxy_approx_count = param.get_model_proxy_count() + 64
    assert id_eps.shape == proxy_approx_count.shape

    for i in range(id_eps.size):
        # NOTE: ensure each set starts out the same
        rng = ds.seeded_rng(seed=42)
        param_i = replace(
            param,
            id_eps=id_eps[i],
            proxy_approx_count=proxy_approx_count[i],
        )

        # NOTE: run more times to get some nicer statistics
        for irun in range(nruns):
            result = run(actx, places, param_i, rng=rng)
            error[irun, i] = result.error

        eoc.add_data_point(id_eps[i], result.error)
        log.info("id_eps %.2e error %.12e %.12e %.12e", id_eps[i], *error[:, i])

    log.info("\n%s", eoc)

    ds.savez(
        filename,
        id_eps=id_eps,
        error=error,
        param=result.parameters,
        overwrite=overwrite,
    )

    if visualize:
        experiment_visualize(filename, ext=ext, overwrite=overwrite)

    return 0


def experiment_visualize(
    filename: str,
    *,
    ext: str = "pdf",
    strip: bool = False,
    overwrite: bool = False,
) -> int:
    path = pathlib.Path(filename)
    if not path.exists():
        log.error("Filename does not exist: '%s'", filename)
        return 1

    basename = ds.strip_timestamp(path.with_suffix(""), strip=strip)
    data = np.load(filename, allow_pickle=True)
    id_eps = data["id_eps"]
    errors = data["error"]

    ds.visualize_eoc(
        f"{basename}-convergence.{ext}",
        *[
            ds.EOCRecorder.from_array(
                r"$E_{2, \mathrm{solution}}$",
                id_eps,
                error,
                abscissa=r"\epsilon_{\mathrm{id}}",
            )
            for error in errors
        ],
        order=1,
        xlabel=r"$\epsilon_{\mathrm{id}}$",
        ylabel=r"Error",
        first_legend_only=True,
        keep_color=True,
        align_order_to_abscissa=True,
        overwrite=overwrite,
    )

    return 0


# }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    ds.add_arguments(parser)
    args, unknown = parser.parse_known_args()

    if args.filename:
        errno = experiment_visualize(
            args.filename,
            ext=args.ext,
            overwrite=args.overwrite,
        )
    else:
        from meshmode import _acf

        errno = experiment_run(
            _acf,
            suffix=args.suffix,
            ext=args.ext,
            overwrite=args.overwrite,
            visualize=not args.no_visualize,
            **ds.parse_unknown_arguments(unknown),
        )

    raise SystemExit(errno)

# vim: fdm=marker
