#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

import common_ds_tools as ds
from meshmode.array_context import PyOpenCLArrayContext
from pytential.collection import GeometryCollection


scriptname = pathlib.Path(__file__)
log = ds.set_up_logging("ds")
ds.set_recommended_matplotlib()


# {{{ run


@dataclass(frozen=True)
class ExperimentResult:
    error: float
    em_model: ds.Model
    em_const: ds.ModelConstant

    parameters: ds.ExperimentParameters

    @property
    def pxy_count(self) -> int:
        return self.em_model.pxy_count


def run_single(
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
        raise ValueError(f"Unknown layer potential type: '{param.lpot_type}'")

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

    em_model = ds.em_parameter_from_skeletonization(
        actx, hmat.skeletons, param.id_eps, param.proxy_radius_factor
    )
    em_const = ds.em_constants_from_skeletonization(actx, hmat.skeletons)
    log.info(
        "id_eps %.12e q %d alpha %.5f (%.5f) rpxy [%.5e, %.5e]",
        em_model.id_eps,
        em_model.pxy_count,
        em_model.pxy_cluster_ratio,
        em_model.pxy_qbx_ratio,
        em_model.pxy_radius_min,
        em_model.pxy_radius_max,
    )
    log.info("c_lr %.12e c_id %.12e c_mp %.12e", *em_const)

    assert em_model.pxy_count == wrangler.proxy.nproxy
    assert abs(em_model.pxy_qbx_ratio - wrangler.proxy.radius_factor) < 1.0e-15

    return ExperimentResult(
        error=error,
        em_model=em_model,
        em_const=em_const,
        parameters=param,
    )


def run(
    actx: PyOpenCLArrayContext,
    places: GeometryCollection,
    param: ds.ExperimentParameters,
    *,
    id_eps_offset: float = 2.0e1,
    pxy_min_count: int = 8,
    pxy_max_count: int = 768,
    pxy_increment: int = 8,
) -> ExperimentResult:
    from dataclasses import replace

    param = replace(param, proxy_approx_count=pxy_min_count)
    log.info(">> initial proxy_approx_count: %d", param.proxy_approx_count)

    increase_count = 0
    prev_error = None
    while True:
        result = run_single(actx, places, param)

        log.info(
            "proxy_approx_count: %d (error %.12e id_eps %.12e [%.12e])",
            result.pxy_count,
            result.error,
            param.id_eps,
            id_eps_offset * param.id_eps,
        )
        log.info("-" * 128)

        # exit if we exceeded the proxy count
        if result.pxy_count > pxy_max_count:
            log.warning(
                "Exceeded max proxy count: %d > %d > %d",
                result.pxy_count,
                pxy_max_count,
                pxy_min_count,
            )
            break

        # exit if we start to increase the error
        if prev_error is not None and result.error > prev_error:
            log.warning(
                "Error is increasing: prev %.12e current %.12e",
                prev_error,
                result.error,
            )

            # NOTE: quit if this happens more than once.. we're merciful here!
            increase_count += 1
            if increase_count >= 2:
                break

        # exit if we reached the target
        # NOTE: multiplying this by an offset because for small id_eps the error
        # would decrease very slowly after some point. This allows the loop to exit
        # earlier, with a reasonable value for the proxy point count. In most
        # cases, the previous check will trigger first though, so we'll see..
        if result.error < id_eps_offset * param.id_eps:
            break

        prev_error = result.error
        param = replace(
            param,
            # NOTE: in 3D, `proxy_approx_count` is different than `result.pxy_count`
            # due to how they're constructed -- pxy_count should always be larger
            proxy_approx_count=result.pxy_count + pxy_increment,
        )

    nproxies = ds.em_estimate_proxy_from_parameters(result.em_model, result.em_const)

    log.info("proxy_count: empirical %4d model %4d", result.pxy_count, nproxies)
    log.info("=" * 128)

    return result


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

    nparam = 7
    nconst = 3
    id_eps = 10.0 ** (-np.arange(2, 16))
    dtype = id_eps.dtype

    # {{{ id_eps vs error: "for each id_eps: find nproxy to match"

    places = ds.make_geometry_collection(actx, param)

    id_empirical_proxies = np.empty(id_eps.shape, dtype=np.int32)
    id_model_error = np.empty(id_eps.size, dtype=dtype)
    id_model_param = np.empty((id_eps.size, nparam), dtype=dtype)
    id_model_const = np.empty((id_eps.size, nconst), dtype=dtype)

    from pytools import ProcessTimer

    with ProcessTimer() as p:
        # NOTE: 3d needs more proxies usually, so we help it out by incrementing
        # it by a larger value so that we get to the desired value sooner
        pxy_increment = 4 ** (places.ambient_dim - 1)
        pxy_min_count = 8

        for i in range(id_eps.size):
            result = run(
                actx,
                places,
                replace(param, id_eps=id_eps[i]),
                pxy_min_count=pxy_min_count,
            )

            id_empirical_proxies[i] = result.pxy_count
            id_model_error[i] = result.error
            id_model_param[i] = result.em_model
            id_model_const[i] = result.em_const

            eoc.add_data_point(id_eps[i], result.error)
            log.info(
                "id_eps %.12e error %.12e nproxy %5d",
                id_eps[i],
                result.error,
                result.pxy_count,
            )
            log.info("=" * 128)

            # NOTE: have the next iteration start from about the same count as
            # it can never realistically require _less_ proxy points
            pxy_min_count = max(result.pxy_count - 2 * pxy_increment, pxy_min_count)

    log.info("\n%s", eoc)
    log.info("[check_forward_model] time: %s", p)

    # }}}

    # {{{ proxy count vs error: "for each (radius, nproxy) compute error"

    pxy_proxy_radius_factor = np.array([1.15, 1.25, 1.5])
    pxy_nproxy = np.linspace(8, id_empirical_proxies[-1], 16, dtype=np.int32)
    size = (pxy_nproxy.size, pxy_proxy_radius_factor.size)
    pxy_empirical_error = np.empty(size, dtype=dtype)
    pxy_model_param = np.empty((*size, nparam), dtype=dtype)
    pxy_model_const = np.empty((*size, nconst), dtype=dtype)

    from itertools import product

    with ProcessTimer() as p:
        for i, j in product(range(size[0]), range(size[1])):
            param_i = replace(
                param,
                id_eps=1.0e-15,
                proxy_approx_count=pxy_nproxy[i],
                proxy_radius_factor=pxy_proxy_radius_factor[j],
            )
            result = run_single(actx, places, param_i)

            pxy_empirical_error[i, j] = result.error
            pxy_model_param[i, j] = result.em_model
            pxy_model_const[i, j] = result.em_const

            model_error = ds.em_estimate_error_from_parameters(
                result.em_model,
                result.em_const,
            )
            log.info(
                "nproxy %4d error %.12e model %.12e",
                pxy_nproxy[i],
                result.error,
                model_error,
            )

    log.info("[check_forward_model] time: %s", p)

    # }}}

    ds.savez(
        filename,
        id_eps=id_eps,
        id_error=eoc,
        id_empirical_proxies=id_empirical_proxies,
        id_model_error=id_model_error,
        id_model_param=id_model_param,
        id_model_const=id_model_const,
        pxy_proxy_radius_factor=pxy_proxy_radius_factor,
        pxy_nproxy=pxy_nproxy,
        pxy_empirical_error=pxy_empirical_error,
        pxy_model_param=pxy_model_param,
        pxy_model_const=pxy_model_const,
        param=param,
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
    show_convergence: bool = True,
    show_model: bool = False,
) -> int:
    path = pathlib.Path(filename)
    if not path.exists():
        log.error("Filename does not exist: '%s'", filename)
        return 1

    basename = ds.strip_timestamp(path.with_suffix(""), strip=strip)
    data = np.load(filename, allow_pickle=True)

    import matplotlib.pyplot as mp

    fig = mp.figure()
    rng = ds.seeded_rng(seed=42)

    param = data["param"][()]
    id_eps = data["id_eps"]
    dtype = id_eps.dtype

    ambient_dim = param["ambient_dim"]

    # {{{ id_eps vs error

    if show_convergence:
        eoc = ds.EOCRecorder.from_array("Error", id_eps, data["id_error"])
        ds.visualize_eoc(
            f"{basename}-convergence.{ext}",
            eoc,
            order=1,
            xlabel=r"$\epsilon_{\mathrm{id}}$",
            ylabel=r"$E(\mathbf{\sigma})$",
            overwrite=overwrite,
        )

    # }}}

    # {{{ id_eps vs proxies

    empirical_proxies = data["id_empirical_proxies"].astype(dtype)
    model_param = data["id_model_param"]
    model_const = data["id_model_const"]

    # fix constants to whole range
    # NOTE: this add some random offset to the error so that we don't have to
    # match it exactly, which seems to choke on some very small values
    eps_offset = 10 if ambient_dim == 2 else 2
    correction = ds.em_constant_fit(
        ds.to_models(model_param),
        ds.to_consts(model_const),
        eps_offset * data["id_model_error"][:, 0],
        rng=rng,
    )
    log.info("Correction: lr %.12e id %.12e mp %.12e", *correction)

    model_const_corrected = correction * model_const
    model_proxies = np.array([
        ds.em_estimate_proxy_from_parameters(
            ds.to_model(model_param[i]),
            ds.to_const(model_const_corrected[i]),
        )
        for i in range(id_eps.size)
    ]).astype(dtype)
    log.info("Model Proxies: %r", model_proxies.astype(np.int32))

    if show_model:
        with ds.axis(fig, f"{basename}-model.{ext}", overwrite=overwrite) as ax:
            ax.plot(empirical_proxies, model_proxies, "o-")
            ax.plot(empirical_proxies, empirical_proxies, "ko--")

            ax.set_xlabel("Empirical")
            ax.set_ylabel("Model")

    with ds.axis(fig, f"{basename}-id-eps.{ext}", overwrite=overwrite) as ax:
        ax.semilogx(id_eps, model_proxies, "o-", label="Model")
        ax.semilogx(id_eps, empirical_proxies, "ko--", label="Empirical")

        if ambient_dim == 2:
            ax.set_ylim([0, 200])
        else:
            ax.set_ylim([0, 600])

        ax.set_xlabel(r"$\epsilon_{\mathrm{id}}$")
        ax.set_ylabel("$q$")
        ax.legend()

    # }}}

    # {{{ proxy vs error

    proxy_radius_factor = data["pxy_proxy_radius_factor"]
    nproxy = data["pxy_nproxy"]
    model_param = data["pxy_model_param"]
    model_const = data["pxy_model_const"]
    empirical_error = data["pxy_empirical_error"]

    # fix constants to whole range
    imin = 3 if ambient_dim == 2 else 0
    # error_offset = 1
    # correction = ds.em_constant_fit(
    #     ds.to_models(model_param[imin:, 2]),
    #     ds.to_consts(model_const[imin:, 2]),
    #     error_offset * empirical_error[imin:, 2],
    # )
    log.info("Correction: lr %.12e id %.12e mp %.12e", *correction)

    model_error = np.empty((nproxy.size, proxy_radius_factor.size))
    for j in range(proxy_radius_factor.size):
        model_const_corrected = correction * model_const[:, j]
        error_offset = 0.25 if ambient_dim == 2 else 1.0
        model_error[:, j] = error_offset * np.array([
            ds.em_estimate_error_from_parameters(
                ds.to_model(model_param[i, j]),
                ds.to_const(model_const_corrected[i]),
            )
            for i in range(nproxy.size)
        ]).astype(dtype)

    with ds.axis(fig, f"{basename}-proxy.{ext}", overwrite=overwrite) as ax:
        markers = ["o", "v", "^", "s", "D"]
        for j in range(proxy_radius_factor.size):
            ax.semilogy(
                nproxy[imin:],
                empirical_error[imin:, j],
                color="k",
                ls="--",
                marker=markers[j],
            )

        for j in range(proxy_radius_factor.size):
            ax.semilogy(
                nproxy[imin:],
                model_error[imin:, j],
                ls="-",
                marker=markers[j],
                label=rf"Model ($\alpha = {proxy_radius_factor[j]:.2f}$)",
            )

        ax.set_xlabel("$q$")
        ax.set_ylabel(r"$E_{2, \mathrm{rel}}(\sigma)$")
        ax.legend()

    # }}}

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
