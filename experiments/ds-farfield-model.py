#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la

import common_ds_tools as ds


if TYPE_CHECKING:
    from meshmode.array_context import PyOpenCLArrayContext
    from pytential.linalg.skeletonization import SkeletonizationResult


scriptname = pathlib.Path(__file__)
log = ds.set_up_logging("ds")
ds.set_recommended_matplotlib()


# {{{ run


@dataclass(frozen=True)
class ExperimentParameters:
    lpot_type: str = "s"
    helmholtz_k: int = 0

    nelements: int = 1024
    mesh_order: int = 4
    target_order: int = 4
    qbx_order: int = 4
    source_ovsmp: int = 1

    id_eps: float = 1.0e-8

    proxy_approx_count: int = 64
    proxy_radius_factor: float = 1.25
    proxy_norm_type: str = "l2"
    proxy_weighted: tuple[bool, bool] = (True, True)
    proxy_remove_source_transforms: bool = False

    max_particles_in_box: int = 128

    @property
    def nclusters(self) -> int:
        return self.target_order * self.nelements // self.max_particles_in_box

    @property
    def fine_order(self) -> int:
        return self.source_ovsmp * self.target_order


@dataclass(frozen=True)
class ExperimentResult:
    cluster_index: int

    id_eps: float
    far_error: float
    near_error: float
    skel_error: float

    proxy_count: int
    proxy_radius_factor: float

    skeleton: SkeletonizationResult
    parameters: ExperimentParameters


def run(
    actx: PyOpenCLArrayContext,
    param: ExperimentParameters,
    *,
    rng: np.random.Generator | None = None,
) -> ExperimentResult:
    rng = ds.seeded_rng(seed=42)

    # {{{ construct discretization

    import meshmode.mesh.generation as mgen

    mesh = mgen.make_curve_mesh(
        mgen.NArmedStarfish(5, 0.25),
        np.linspace(0, 1, param.nelements + 1),
        param.mesh_order,
    )

    import meshmode.discretization.poly_element as mpoly
    from meshmode.discretization import Discretization

    pre_density_discr = Discretization(
        actx, mesh, mpoly.InterpolatoryQuadratureSimplexGroupFactory(param.target_order)
    )

    from pytential.qbx import QBXLayerPotentialSource

    qbx = QBXLayerPotentialSource(
        pre_density_discr,
        fine_order=param.fine_order,
        qbx_order=param.qbx_order,
        fmm_order=False,
    )

    from pytential import GeometryCollection, sym

    dd = sym.DOFDescriptor("ds", discr_stage=sym.QBX_SOURCE_STAGE2)
    places = GeometryCollection(qbx, auto_where=dd)

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)

    log.info("nelements:     %d", density_discr.mesh.nelements)
    log.info("ndofs:         %d", density_discr.ndofs)

    # }}}

    # {{{ construct skeletonization wrangler

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

    from pytential.linalg.skeletonization import make_skeletonization_wrangler

    wrangler = make_skeletonization_wrangler(
        places,
        sym_op,
        sym_u,
        context=context,
        auto_where=dd,
        # NOTE: debug / testing variables
        _weighted_proxy=param.proxy_weighted,
        _remove_source_transforms=param.proxy_remove_source_transforms,
    )

    from pytential.linalg.proxy import QBXProxyGenerator

    proxy = QBXProxyGenerator(
        places,
        approx_nproxy=param.proxy_approx_count,
        radius_factor=param.proxy_radius_factor,
        norm_type=param.proxy_norm_type,
    )

    from pytential.linalg.cluster import partition_by_nodes

    # NOTE: for tree_kind=None on a curve, the indices are next to each other
    indices, _ = partition_by_nodes(
        actx,
        places,
        dofdesc=dd,
        tree_kind=None,
        max_particles_in_box=param.max_particles_in_box,
    )

    srcindex = indices.nclusters // 2
    nearindices = srcindex - 1, srcindex + 1
    farindex = 0
    nearindex = nearindices[-1]

    from pytential.linalg.utils import make_index_list
    from pytools.obj_array import make_obj_array

    eval_src_indices = make_index_list(
        make_obj_array([
            indices.cluster_indices(srcindex),
            indices.cluster_indices(srcindex),
        ])
    )
    eval_tgt_indices = make_index_list(
        make_obj_array([
            indices.cluster_indices(farindex),
            indices.cluster_indices(nearindex),
        ])
    )

    from pytential.linalg.utils import TargetAndSourceClusterList

    tgt_src_index = TargetAndSourceClusterList(indices, indices)
    eval_tgt_src_index = TargetAndSourceClusterList(eval_tgt_indices, eval_src_indices)

    # }}}

    # {{{ evaluate errors

    from pytential.linalg.skeletonization import _skeletonize_block_by_proxy_with_mats
    from pytools import ProcessTimer

    with ProcessTimer() as p:
        skeleton = _skeletonize_block_by_proxy_with_mats(
            actx,
            0,
            0,
            places,
            proxy,
            wrangler,
            tgt_src_index,
            id_eps=param.id_eps,
            rng=rng,
            max_particles_in_box=param.max_particles_in_box,
        )
    log.info("time: %s", p)

    mat = wrangler.evaluate_self(actx, places, eval_tgt_src_index, 0, 0)

    A = ds.cluster_take(mat, eval_tgt_src_index)
    S = ds.skeleton_cluster_take(
        A, skeleton, ((farindex, srcindex), (nearindex, srcindex))
    )

    # far errors
    R = skeleton.R[srcindex]
    L = skeleton.L[farindex]
    far_error = ds.rel_norm(A[0], L @ S[0] @ R, ord=2)

    # near errors
    R = skeleton.R[srcindex]
    L = skeleton.L[nearindex]
    near_error = ds.rel_norm(A[1], L @ S[1] @ R, ord=2)

    # }}}

    return ExperimentResult(
        cluster_index=srcindex,
        id_eps=param.id_eps,
        far_error=far_error,
        near_error=near_error,
        skel_error=near_error,
        proxy_count=proxy.nproxy,
        proxy_radius_factor=proxy.radius_factor,
        skeleton=skeleton,
        parameters=param,
    )


# }}}


# {{{ check_error_vs_id_eps_ideal


def check_error_vs_id_eps_ideal(
    actx_factory,
    *,
    suffix: str = "",
    ext: str = "png",
    overwrite: bool = False,
    visualize: bool = True,
) -> None:
    actx = actx_factory()
    eocf = ds.EOCRecorder("Far", "id_eps")
    eocn = ds.EOCRecorder("Near", "id_eps")

    from dataclasses import replace

    id_eps = 10.0 ** (-np.arange(2, 16))
    param = ExperimentParameters()

    filename = ds.make_archive(scriptname, param, suffix=suffix)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    for i in range(id_eps.size):
        result = run(actx, replace(param, id_eps=id_eps[i]))

        eocf.add_data_point(id_eps[i], result.far_error)
        eocn.add_data_point(id_eps[i], result.near_error)
        log.info(
            "id_eps %.2e far %.12e near %.12e",
            id_eps[i],
            result.far_error,
            result.near_error,
        )

    log.info("\n%s", eocf)
    log.info("\n%s", eocn)

    ds.savez(
        filename,
        id_eps=id_eps,
        far_error=eocf,
        near_error=eocn,
        param=result.parameters,
        overwrite=overwrite,
    )

    if visualize:
        ds.visualize_eoc(
            filename.parent / f"{filename.stem}-convergence.{ext}",
            eocf,
            eocn,
            order=1,
            xlabel=r"\epsilon_{id}",
            overwrite=overwrite,
        )


# }}}


# {{{ check_error_vs_id_eps_no_target_weights


def check_error_vs_id_eps_no_target_weights(
    actx_factory,
    *,
    ext: str = "png",
    suffix: str = "-t0w0",
    overwrite: bool = True,
    visualize: bool = True,
) -> None:
    actx = actx_factory()
    eocf = ds.EOCRecorder("Far", "id_eps")
    eocn = ds.EOCRecorder("Near", "id_eps")

    from dataclasses import replace

    param = ExperimentParameters(proxy_weighted=(True, False))

    id_eps = 10.0 ** (-np.arange(2, 16))
    nranks = np.zeros(id_eps.shape, dtype=np.int32)

    for i in range(id_eps.size):
        result = run(actx, replace(param, id_eps=id_eps[i]))

        k = result.cluster_index
        skel_tgt_src_index = result.skeleton.skel_tgt_src_index
        nranks[i], _ = skel_tgt_src_index.cluster_shape(k, k)

        eocf.add_data_point(id_eps[i], result.far_error)
        eocn.add_data_point(id_eps[i], result.near_error)

        log.info(
            "id_eps %.2e far %.12e near %.12e",
            id_eps[i],
            result.far_error,
            result.near_error,
        )

    basename = ds.make_filename(scriptname, param, suffix=suffix)

    # {{{ proxy matrix singular values

    if visualize:
        import matplotlib.pyplot as mp

        fig = mp.figure()

        with ds.axis(fig, f"{basename}-sigma.{ext}", overwrite=overwrite) as ax:
            for i in range(result.skeleton.nclusters):
                mat = np.hstack(result.skeleton._tgt_eval_result[i])
                sigma = la.svd(mat, compute_uv=False)

                ax.semilogy(sigma[:32], "k-" if i == result.cluster_index else "--")

            ax.set_xlabel("$Cluster$")
            ax.set_ylabel(r"$\kappa$")

    # }}}

    # {{{ proxy matrix ranks and sizes

    if visualize:
        nbrindex = result.skeleton._src_eval_result.nbrindex
        tgt_src_index = result.skeleton.tgt_src_index

        with ds.axis(fig, f"{basename}-ranks.{ext}", overwrite=overwrite) as ax:
            ax.plot(nranks, label="$Rank$")
            ax.axhline(result.proxy_count, ls="--", color="k", label="$Proxy$")
            ax.axhline(
                nbrindex.cluster_shape(k, k)[0], ls="-.", color="k", label="$Neighbor$"
            )
            ax.axhline(
                tgt_src_index.cluster_shape(k, k)[0],
                ls=":",
                color="k",
                label="$Cluster$",
            )

            ax.set_ylabel("$N$")
            ax.legend()

    # }}}

    # {{{ convergence

    log.info("\n%s", eocf)
    log.info("\n%s", eocn)

    ds.savez(
        basename.with_suffix(".npz"),
        id_eps=id_eps,
        far_error=eocf.error,
        near_error=eocn.error,
        param=result.parameters,
        overwrite=overwrite,
    )

    if visualize:
        ds.visualize_eoc(
            f"{basename}-convergence.{ext}",
            eocf,
            eocn,
            order=1,
            xlabel=r"\epsilon_{id}",
            overwrite=overwrite,
        )

    # }}}


# }}}


if __name__ == "__main__":
    import sys

    from meshmode import _acf

    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        check_error_vs_id_eps_ideal(_acf)

# vim: fdm=marker
