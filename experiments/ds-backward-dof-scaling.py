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
    proxy_count: int
    proxy_radius_factor: float

    ndofs: int
    ds_build: ds.TimingResult
    ds_solve: ds.TimingResult
    fmm_solve: ds.TimingResult

    parameters: ds.ExperimentParameters


def run(
    actx: PyOpenCLArrayContext,
    places: GeometryCollection,
    param: ds.ExperimentParameters,
    *,
    rng: np.random.Generator | None = None,
    use_fmm: bool = True,
    is_profiling: bool = True,
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

    # {{{ direct solver timings

    from pytential.linalg.hmatrix import build_hmatrix_by_proxy

    def ds_build_hmatrix():
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

        # NOTE: this is likely not required either, because the last step of
        # constructing hmat is converting everything into numpy arrays anyway
        actx.queue.finish()
        return wrangler, hmat

    x_ref = ds.make_uniform_random_array(actx, density_discr, rng=rng)
    b_ref = bind(places, sym_op, auto_where=dd)(actx, sigma=x_ref, **context)

    # NOTE: this is here so that all the necessary values get cached in 'places'
    wrangler, hmat = ds_build_hmatrix()

    def ds_solve():
        result = hmat @ b_ref

        # NOTE: this is not particularly needed, because hmat is all in numpy
        # right now, but just in case there's some lingering work to be done
        actx.queue.finish()
        return result

    # NOTE: ensure the work is done before we start timing
    actx.queue.finish()

    if is_profiling:
        # from pytools import ProcessTimer

        # with ProcessTimer() as p:
        #     ds_build_hmatrix()
        # log.info("BUILD: %s", p)
        # build_time = ds.TimingResult(
        #     name="Build", walltime=p.wall_elapsed, mean=p.wall_elapsed, std=0.0
        # )

        # with ProcessTimer() as p:
        #     ds_solve()
        # log.info("SOLVE: %s", p)
        # solve_time = ds.TimingResult(
        #     name="Solve", walltime=p.wall_elapsed, mean=p.wall_elapsed, std=0.0
        # )

        build_time = ds.TimingResult(name="Build", walltime=0.0, mean=0.0, std=0.0)
        solve_time = ds.TimingResult(name="Solve", walltime=0.0, mean=0.0, std=0.0)
        with ds.profileit("profile-2.cProfile", overwrite=True):
            ds_solve()
    else:
        build_time = ds.timeit(ds_build_hmatrix, name="Build")
        solve_time = ds.timeit(ds_solve, name="Solve")

    # }}}

    # {{{ fmm timings

    from meshmode.dof_array import flat_norm

    x_hmat = hmat @ b_ref
    ds_error_rtol = actx.to_numpy(flat_norm(x_hmat - x_ref) / flat_norm(x_ref))
    log.info("GMRES tolerance: %.12e (id_eps %.12e)", ds_error_rtol, param.id_eps)

    def fmm_gmres_solve() -> None:
        bound_op = bind(places, sym_op, auto_where="fmm")
        scipy_op = bound_op.scipy_op(actx, "sigma", b_ref.entry_dtype, **context)

        from pytential.linalg.gmres import gmres

        result = gmres(
            scipy_op,
            b_ref,
            tol=0.1 * ds_error_rtol,
            progress=True,
        )

        assert result.success

    fmm_solve = (
        ds.timeit(fmm_gmres_solve, name="FMM")
        if use_fmm
        else ds.TimingResult(name="FMM", walltime=0.0, mean=0.0, std=0.0)
    )

    # }}}

    return ExperimentResult(
        id_eps=param.id_eps,
        proxy_count=wrangler.proxy.nproxy,
        proxy_radius_factor=wrangler.proxy.radius_factor,
        ndofs=density_discr.ndofs,
        ds_build=build_time,
        ds_solve=solve_time,
        fmm_solve=fmm_solve,
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
    from dataclasses import replace

    actx = actx_factory()

    ambient_dim = kwargs.pop("ambient_dim", 3)
    if ambient_dim == 2:
        param = ds.ExperimentParameters2(**kwargs)
    else:
        param = ds.ExperimentParametersSphere3(**kwargs)

    filename = ds.make_archive(scriptname, param, suffix=suffix)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    ndofs = []
    build_timings = []
    solve_timings = []
    fmm_timings = []

    eocds = ds.EOCRecorder("DS", "N")
    eocfmm = ds.EOCRecorder("FMM", "N")

    from pytools import ProcessTimer

    with ProcessTimer() as p:
        # for resolution in param.resolutions:
        for resolution in [0.18]:
            rng = ds.seeded_rng(seed=42)

            places = ds.make_geometry_collection(
                actx, replace(param, resolution=resolution)
            )
            result = run(actx, places, param, rng=rng, use_fmm=False)

            ndofs.append(result.ndofs)
            build_timings.append(result.ds_build)
            solve_timings.append(result.ds_solve)
            fmm_timings.append(result.fmm_solve)

            log.info(
                "resolution %s ndofs %6d time build %s solve %s fmm %s",
                resolution,
                result.ndofs,
                result.ds_build,
                result.ds_solve,
                result.fmm_solve,
            )

            eocds.add_data_point(
                result.ndofs, result.ds_build.mean + result.ds_solve.mean
            )
            eocfmm.add_data_point(result.ndofs, result.fmm_solve.mean)

        if len(eocds.history) > 1:
            log.info("\n%s", eocds)
            log.info("\n%s", eocfmm)

        ds.savez(
            filename,
            resolutions=ndofs,
            build_timings=build_timings,
            solve_timings=solve_timings,
            fmm_timings=fmm_timings,
            eocds=eocds,
            eocfmm=eocfmm,
            param=result.parameters,
            overwrite=overwrite,
        )

    log.info("[check_backward_scaling] time: %s", p)

    if visualize:
        experiment_visualize(filename, ext=ext, overwrite=overwrite)

    return 0


def experiment_visualize(
    filename: str,
    *,
    ext: str = "pdf",
    strip: bool = False,
    overwrite: bool = False,
    show_timings: bool = False,
) -> None:
    path = pathlib.Path(filename)
    if not path.exists():
        log.error("Filename does not exist: '%s'", filename)
        return 1

    basename = ds.strip_timestamp(path.with_suffix(""), strip=strip)
    data = np.load(filename, allow_pickle=True)

    ambient_dim = data["param"][()]["ambient_dim"]
    resolutions = data["resolutions"]
    build_timings = data["build_timings"]
    solve_timings = data["solve_timings"]
    # fmm_timings = data["fmm_timings"]

    import matplotlib.pyplot as mp
    from matplotlib import ticker

    if show_timings:
        ds.visualize_timings(
            f"{basename}-build-timing.{ext}",
            resolutions,
            ds.TimingResult.from_array("Build", build_timings),
            xlabel="$n$",
            overwrite=overwrite,
        )

        ds.visualize_timings(
            f"{basename}-solve-timing.{ext}",
            resolutions,
            ds.TimingResult.from_array("Solve", solve_timings),
            xlabel="$n$",
            overwrite=overwrite,
        )

        ds.visualize_timings(
            f"{basename}-timing.{ext}",
            resolutions,
            ds.TimingResult.from_array("DS", build_timings + solve_timings),
            # ds.TimingResult.from_array("FMM", fmm_timings),
            xlabel="$n$",
            overwrite=overwrite,
        )

    from itertools import cycle

    fig = mp.figure(figsize=(8, 7))
    color = cycle(mp.rcParams["axes.prop_cycle"].by_key()["color"])

    with ds.axis(fig, f"{basename}-scaling.{ext}", overwrite=overwrite) as ax1:
        # {{{ plot timings

        lines = []
        labels = []

        (line,) = ax1.loglog(resolutions, build_timings[0, :], "v-", color=next(color))
        lines.append(line)
        labels.append(r"Build $(\leftarrow)$")
        ax1.fill_between(
            resolutions,
            build_timings[1, :] - build_timings[2, :],
            build_timings[1, :] + build_timings[2, :],
            color=line.get_color(),
            alpha=0.25,
        )

        ax2 = ax1.twinx()
        (line,) = ax2.loglog(resolutions, solve_timings[0, :], "^-", color=next(color))
        lines.append(line)
        labels.append(r"Solve $(\rightarrow)$")
        ax2.fill_between(
            resolutions,
            solve_timings[1, :] - solve_timings[2, :],
            solve_timings[1, :] + solve_timings[2, :],
            color=line.get_color(),
            alpha=0.25,
        )

        # }}}

        # {{{ plot expected scaling

        order = 1 if ambient_dim == 2 else 1.5
        # compute polynomial order line
        ymax = np.max(build_timings[1, :])
        ymin = np.min(build_timings[1, :])
        xmax = resolutions[-1]
        xmin = np.exp(np.log(xmax) + np.log(ymin / ymax) / order)
        log.info("O(n^p):     [%g, %g] x [%g, %g]", xmin, xmax, ymin, ymax)

        (line,) = ax1.loglog([xmin, xmax], [ymin, ymax], "k--")
        lines.append(line)
        labels.append("$O(n)$" if ambient_dim == 2 else "$O(n^{1.5})$")

        if ambient_dim == 3:
            from scipy.special import lambertw

            # compute n log n order line
            ymax = np.max(solve_timings[1, :])
            ymin = np.min(solve_timings[1, :])
            xmax = resolutions[-1]
            k = xmax * np.log(xmax) / (ymax / ymin)
            xmin = np.real(k / lambertw(k))
            log.info("O(n log n): [%g, %g] x [%g, %g]", xmin, xmax, ymin, ymax)

            (line,) = ax2.loglog([xmin, xmax], [ymin, ymax], "k:")
            lines.append(line)
            labels.append(r"$O(n \log n)$")

        # }}}

        ax1.set_xlabel("$n$")
        ax1.set_ylabel("Build time (s)")

        fmt = ticker.FuncFormatter(lambda x, _: f"{x:.1f}")
        ax1.yaxis.set_minor_formatter(fmt)
        ax1.yaxis.set_major_formatter(fmt)
        ax1.legend(lines, labels)

        ax2.set_ylabel("Solve time (s)", rotation=-90, labelpad=25)
        fmt = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
        ax2.yaxis.set_minor_formatter(fmt)
        ax2.yaxis.set_major_formatter(fmt)

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
