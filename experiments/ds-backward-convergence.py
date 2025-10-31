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
    h_max: float
    error: float
    parameters: ds.ExperimentParameters


def run(
    actx: PyOpenCLArrayContext,
    places: GeometryCollection,
    param: ds.ExperimentParameters,
    *,
    use_fmm: bool = False,
    rng: np.random.Generator | None = None,
) -> ExperimentResult:
    if rng is None:
        rng = ds.seeded_rng(seed=42)

    from pytential import bind, sym

    # {{{ obtain discretization

    dd = places.auto_source
    if use_fmm:
        # FIXME: QBXLayerPotentialSource seems to force STAGE1 in some places,
        # so if we want to do a square operator.. we need to do it on STAGE1
        dd = dd.copy(discr_stage=sym.QBX_SOURCE_STAGE1)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)

    log.info("nelements:     %d", density_discr.mesh.nelements)
    log.info("ndofs:         %d", density_discr.ndofs)

    h_max = bind(places, sym.h_max(places.ambient_dim, dofdesc=dd))(actx)
    return ExperimentResult(h_max=actx.to_numpy(h_max), error=0.0, parameters=param)

    # }}}

    # {{{ construct symbolic operators

    k = param.helmholtz_k
    if k == 0:
        from sumpy.kernel import LaplaceKernel

        knl = LaplaceKernel(places.ambient_dim)
        kernel_arguments = {}
        context = {}
        dtype = np.dtype(np.float64)
    else:
        from sumpy.kernel import HelmholtzKernel

        knl = HelmholtzKernel(places.ambient_dim, helmholtz_k_name="k")
        kernel_arguments = {"k": sym.var("k")}
        context = {"k": k}
        dtype = np.dtype(np.complex128)

    sym_sigma = sym.var("sigma")
    side = -1

    if param.lpot_type == "d":
        # NOTE: this is the interior problem
        sym_op = side * sym_sigma / 2 + sym.D(
            knl, sym_sigma, qbx_forced_limit="avg", kernel_arguments=kernel_arguments
        )
        sym_repr = sym.D(
            # FIXME: qbx_forced_limit needs to be None for "non-self evaluation"
            knl,
            sym_sigma,
            qbx_forced_limit=None,
            kernel_arguments=kernel_arguments,
        )
    else:
        raise ValueError(f"unknown layer potential type: '{param.lpot_type}'")

    sym_op_pot = sym.int_g_vec(
        knl,
        sym_sigma,
        # FIXME: qbx_forced_limit needs to be None for "non-self evaluation"
        qbx_forced_limit=None,
        kernel_arguments=kernel_arguments,
    )

    # }}}

    # {{{ build reference solution

    point_sources = places.get_geometry("point_sources")
    charges = ds.make_uniform_random_array(actx, point_sources, rng=rng)

    b_ref = bind(
        places,
        sym_op_pot,
        auto_where=("point_sources", dd),
    )(actx, sigma=charges, **context)

    x_ref = bind(
        places,
        sym_op_pot,
        auto_where=("point_sources", "point_targets"),
    )(actx, sigma=charges, **context)

    # }}}

    # {{{ solve using hmatrix

    if use_fmm:
        from pytential.linalg.gmres import gmres

        scipy_op = bind(places, sym_op, auto_where=(dd, dd)).scipy_op(
            actx, "sigma", dtype, **context
        )
        gmres_result = gmres(
            scipy_op,
            b_ref,
            tol=param.id_eps,
            progress=True,
            hard_failure=True,
            stall_iterations=50,
            no_progress_factor=1.05,
        )
        x_hmat = gmres_result.solution
    else:
        from pytential.linalg.hmatrix import build_hmatrix_by_proxy
        from pytools import ProcessTimer

        with ProcessTimer() as p:
            wrangler = build_hmatrix_by_proxy(
                actx,
                places,
                sym_op,
                sym_sigma,
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

        with ProcessTimer() as p:
            x_hmat = hmat @ b_ref

        log.info("[solve] time: %s", p)

    h_max = bind(places, sym.h_max(places.ambient_dim, dofdesc=dd))(actx)
    x_hmat = bind(
        places,
        sym_repr,
        auto_where=(dd, "point_targets"),
    )(actx, sigma=x_hmat, **context)
    error = actx.to_numpy(
        actx.np.linalg.norm(x_hmat - x_ref) / actx.np.linalg.norm(x_ref)
    )

    # }}}

    return ExperimentResult(
        h_max=actx.to_numpy(h_max),
        error=error,
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
    eoc = ds.EOCRecorder("Accuracy", "h_max")

    from dataclasses import replace

    ambient_dim = kwargs.pop("ambient_dim", 3)
    kwargs = {
        # NOTE: make sure we do a good job at the approximation
        "id_eps": 1.0e-14,
        **kwargs,
    }

    if ambient_dim == 2:
        kwargs = {
            # NOTE: QBX needs oversampling, but the direct solver does not work
            # on QBX_SOURCE_STAGE2_QUAD, so we just oversample the base discr!
            "mesh_order": 20,
            "target_order": 20,
            "source_ovsmp": 1,
            # NOTE: the default resolutions are too fine to see convergence
            "resolutions": (256, 512, 768, 1024, 1024 + 256),
            # NOTE: make sure we do a good job at the approximation
            "starfish_arms": 8,
            **kwargs,
        }

        param = ds.ExperimentParameters2(**kwargs)
    else:
        kwargs = {
            # NOTE: QBX needs oversampling, but the direct solver does not work
            # on QBX_SOURCE_STAGE2_QUAD, so we just oversample the base discr!
            "mesh_order": 16,
            "target_order": 16,
            "source_ovsmp": 1,
            # NOTE: the default resolutions are too fine to see convergence
            # NOTE: this weird order ensures decreasing h_max
            "resolutions": ((10, 10), (10, 5), (15, 10), (20, 15), (25, 20)),
            **kwargs,
        }

        param = ds.ExperimentParametersTorus3(**kwargs)

    filename = ds.make_archive(scriptname, param, suffix=suffix)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    nruns = len(param.resolutions)
    error = np.empty(nruns)
    h_max = np.empty(nruns)

    for i, resolution in enumerate(param.resolutions):
        # NOTE: ensure each set starts out the same
        rng = ds.seeded_rng(seed=42)
        param_i = replace(param, resolution=resolution)
        if isinstance(resolution, int):
            resolution_s = f"{resolution:5d}"
        else:
            resolution_s = ", ".join(f"{r:3d}" for r in resolution)
            resolution_s = f"({resolution_s})"

        places = ds.make_geometry_collection(actx, param_i)
        result = run(actx, places, param_i, rng=rng)

        h_max[i] = result.h_max
        error[i] = result.error
        eoc.add_data_point(result.h_max, result.error)
        log.info(
            "resolution %s h_max %.2e error %.12e", resolution_s, h_max[i], error[i]
        )

    log.info("\n%s", eoc)

    ds.savez(
        filename,
        h_max=h_max,
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
    h_max = data["h_max"]
    errors = data["error"]

    ds.visualize_eoc(
        f"{basename}-convergence.{ext}",
        *[
            ds.EOCRecorder.from_array(
                r"$E_{2, \text{solution}}$",
                h_max,
                error,
                abscissa=r"h_{\text{max}}",
            )
            for error in errors
        ],
        order=1,
        xlabel=r"$\epsilon_{\text{id}}$",
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
