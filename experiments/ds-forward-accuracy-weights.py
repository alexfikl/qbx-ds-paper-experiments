#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
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
    error: float

    id_eps: float
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

    # {{{ construct discretization

    dd = places.auto_source
    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)

    log.info("nelements:     %d", density_discr.mesh.nelements)
    log.info("ndofs:         %d", density_discr.ndofs)

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
            _weighted_proxy=param.proxy_weighted,
            _max_particles_in_box=param.max_particles_in_box,
            _approx_nproxy=param.proxy_approx_count,
            _proxy_radius_factor=param.proxy_radius_factor,
        )
        hmat = wrangler.get_forward()

    log.info("[construction] time: %s", p)

    x_ref = ds.make_uniform_random_array(actx, density_discr, rng=rng)
    b_ref = bind(places, sym_op, auto_where=dd)(actx, sigma=x_ref)

    from meshmode.dof_array import flat_norm

    with ProcessTimer() as p:
        b_hmat = hmat @ x_ref
        error = actx.to_numpy(flat_norm(b_hmat - b_ref) / flat_norm(b_ref))

    log.info("[matvec] time: %s", p)

    # }}}

    return ExperimentResult(
        error=error,
        id_eps=param.id_eps,
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
) -> None:
    actx = actx_factory()
    eoc_wo = ds.EOCRecorder("Weightless", "id_eps")
    eoc_wi = ds.EOCRecorder("Weighted", "id_eps")

    from dataclasses import replace

    ambient_dim = kwargs.pop("ambient_dim", 3)
    if ambient_dim == 2:
        param = ds.ExperimentParameters2(**kwargs)
        param = replace(param, starfish_arms=32)
    else:
        param = ds.ExperimentParametersTorus3(**kwargs)

    filename = ds.make_archive(scriptname, param, suffix=suffix)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    id_eps = 10.0 ** (-np.arange(2, 16))
    error = np.empty((2, id_eps.size), dtype=id_eps.dtype)
    places = ds.make_geometry_collection(actx, param)

    for i in range(id_eps.size):
        # run skeletonization *with* target weights
        param_i = replace(param, id_eps=id_eps[i], proxy_weighted=(True, True))
        result = run(actx, places, param_i)
        error[0, i] = result.error

        # run skeletonization *without* target weights
        param_i = replace(param, id_eps=id_eps[i], proxy_weighted=(True, False))
        result = run(actx, places, param_i)
        error[1, i] = result.error

        eoc_wi.add_data_point(id_eps[i], error[0, i])
        eoc_wo.add_data_point(id_eps[i], error[1, i])
        log.info("id_eps %.2e error wi %.12e wo %.12e", id_eps[i], *error[:, i])

    log.info("\n%s\n%s", eoc_wi.pretty_print(), eoc_wo.pretty_print())

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

    eoc_wi = ds.EOCRecorder.from_array("Weighted", id_eps, errors[0])
    eoc_wo = ds.EOCRecorder.from_array("No weights", id_eps, errors[1])

    ds.visualize_eoc(
        f"{basename}-convergence.{ext}",
        eoc_wi,
        eoc_wo,
        order=1,
        xlabel=r"$\epsilon_{\mathrm{id}}$",
        ylabel=r"$E_{2, \mathrm{rel}}(\mathbf{\sigma})$",
        align_order_to_abscissa=True,
        overwrite=overwrite,
    )

    return 0


# }}}clusters


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
