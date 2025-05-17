#! /usr/bin/env python3

# SPDX-FileCopyrightText: 2023 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pathlib

import numpy as np

import common_ds_tools as ds
from meshmode.array_context import PyOpenCLArrayContext
from pytential.collection import GeometryCollection


scriptname = pathlib.Path(__file__)
log = ds.set_up_logging("ds")
ds.set_recommended_matplotlib()


# {{{ run


# {{{ run2d


def run2d(
    actx: PyOpenCLArrayContext,
    places: GeometryCollection,
    param: ds.ExperimentParameters,
    *,
    suffix: str = "",
    ext: str = "pdf",
    strip: bool = False,
    overwrite: bool = False,
    visualize: bool = False,
) -> None:
    from arraycontext import flatten

    dd = places.auto_source.to_stage1()
    discr = places.get_discretization(dd.geometry, dd.discr_stage)
    log.info("STAGE1 nelements %6d ndofs %6d", discr.mesh.nelements, discr.ndofs)

    dd = places.auto_source.to_stage2()
    discr = places.get_discretization(dd.geometry, dd.discr_stage)
    log.info("STAGE2 nelements %6d ndofs %6d", discr.mesh.nelements, discr.ndofs)

    dd = places.auto_source
    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)

    x, y = actx.to_numpy(flatten(density_discr.nodes(), actx)).reshape(
        places.ambient_dim, -1
    )

    from pytential.linalg.cluster import partition_by_nodes

    indices, ctree = partition_by_nodes(
        actx,
        places,
        dofdesc=dd,
        tree_kind="adaptive-level-restricted",
        # tree_kind=None,
        max_particles_in_box=param.max_particles_in_box,
    )
    log.info("nlevels %3d nclusters %6d ", ctree.nlevels, indices.nclusters)

    from pytential.linalg.proxy import QBXProxyGenerator

    proxy = QBXProxyGenerator(
        places,
        approx_nproxy=param.proxy_approx_count,
        radius_factor=param.proxy_radius_factor,
        norm_type=param.proxy_norm_type,
    )
    pxy = proxy(actx, dd, indices).to_numpy(actx)

    import matplotlib.pyplot as mp

    fig = mp.figure()
    basename = ds.make_filename(scriptname, param, suffix=suffix)

    filename = ds.strip_timestamp(
        pathlib.Path(f"{basename}-clusters.{ext}"), strip=strip
    )
    with ds.axis(fig, filename, overwrite=overwrite) as ax:
        cluster_sizes = np.sort(np.diff(indices.starts))
        ax.bar(np.arange(indices.nclusters), cluster_sizes, align="edge")
        ax.axhline(int(np.mean(cluster_sizes)), color="k", ls="--")

        ax.set_xlabel(r"Cluster Index (Sorted)")
        ax.set_ylabel(r"$n_i$ (\#DOFs)")
        ax.set_xlim([0, indices.nclusters])
        ax.set_ylim([0, 512])

    filename = ds.strip_timestamp(pathlib.Path(f"{basename}.{ext}"), strip=strip)
    with ds.axis(fig, filename, overwrite=overwrite) as ax:
        ax.set_axis_off()
        icluster = 22

        for i in range(indices.nclusters):
            isrc = indices.cluster_indices(i)

            if i == icluster:
                ax.plot(x[isrc][::5], y[isrc][::5], "ko", ms=5)
            else:
                ax.plot(x[isrc][::5], y[isrc][::5], "o", ms=5)

            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_aspect("equal")

        i = icluster
        c = mp.Circle(pxy.centers[:, i], pxy.radii[i], color="k", alpha=0.1)
        ax.add_artist(c)
        c = mp.Circle(pxy.centers[:, i], pxy.cluster_radii[i], color="k", alpha=0.1)
        ax.add_artist(c)
        ax.plot(pxy.centers[0, i], pxy.centers[1, i], "ro")


# }}}


# {{{ run3d


def postprocess_vtu(
    filename: pathlib.Path,
    icluster: int,
    nclusters: int,
    *,
    field: str = "marker",
    ext: str = "png",
    pxy_radius: float,
    box_radius: float,
    center: np.ndarray,
) -> None:
    from itertools import chain, cycle

    from matplotlib.colors import to_rgb

    try:
        import paraview.simple as pv
    except ImportError:
        import sys

        log.error("This script needs access to the Paraview Python bindings!")
        log.error("If you have Paraview installed, you may need to export")
        log.error("some additional variables")
        log.error(
            "  export "
            'PYTHONPATH="${PARAVIEW_ROOT}/lib/python%d.%d/site-packages:${PYTHONPATH}"',
            sys.version_info[0],
            sys.version_info[1],
        )
        log.error('  export LD_LIBRARY_PATH="${PARAVIEW_ROOT}/lib:${LD_LIBRARY_PATH}"')
        return

    ext = "pdf" if ext == "tikz" else ext
    outfile = str(
        (filename.parent / f"{filename.stem}-{icluster:02d}").with_suffix(f".{ext}")
    )

    pv._DisableFirstRenderCameraReset()
    view = pv.GetActiveViewOrCreate("RenderView")

    # read in file
    surface = pv.XMLUnstructuredGridReader(FileName=[str(filename)])

    # show main geometry
    surface_display = pv.Show(surface, view)
    surface_display.SetRepresentationType("Surface")
    surface_display.EdgeColor = [0, 0, 0]
    surface_display.InterpolateScalarsBeforeMapping = 0

    # set field to color by
    pv.ColorBy(surface_display, ("POINTS", field))

    # get colormap
    colormap = pv.GetColorTransferFunction(field)
    colormap.RescaleTransferFunction(0, nclusters - 1)
    colormap.Discretize = 1
    colormap.NumberOfTableValues = nclusters

    colors = [
        to_rgb(c)
        for c in (
            # NOTE: this is the `gem12` color scheme from MATLAB
            "#0072bd",
            "#d95319",
            "#edb120",
            "#7e2f8e",
            "#77ac30",
            "#4dbeee",
            "#a2142f",
            "#ffd60a",
            "#6582fd",
            "#ff453a",
            "#00a3a3",
            "#cb845d",
        )
    ]

    colormap.RGBPoints = list(
        chain.from_iterable([
            (i, 0, 0, 0) if i == icluster else (i, color[0], color[1], color[2])
            for i, color in zip(range(nclusters), cycle(colors))
        ])
    )
    surface_display.RescaleTransferFunctionToDataRange(True, False)

    # make a sphere
    proxies = pv.Sphere(registrationName="ProxySphere")
    proxies.Radius = pxy_radius
    proxies.Center = center
    proxies.ThetaResolution = proxies.PhiResolution = 32

    proxies_display = pv.Show(proxies, view)
    proxies_display.AmbientColor = [0.0, 0.0, 0.0]
    proxies_display.Opacity = 0.5

    cluster = pv.Sphere(registrationName="ClusterSphere")
    cluster.Radius = box_radius
    cluster.Center = center
    cluster.ThetaResolution = cluster.PhiResolution = 32

    cluster_display = pv.Show(cluster, view)
    cluster_display.DiffuseColor = [0.0, 0.0, 0.0]
    cluster_display.Opacity = 0.15

    # set sizes
    pv.LoadPalette(paletteName="WhiteBackground")
    layout = pv.GetLayout()
    layout.SetSize(1024, 1024)

    # set camera and finish up
    view.CameraPosition = [0.0, 0.0, -45.0]
    view.CameraViewUp = [-0.03, 0.8, 0.6]
    view.OrientationAxesVisibility = 0

    view.Update()
    pv.RenderAllViews()

    pv.ExportView(
        outfile,
        view=view,
        Plottitle="",
        Rasterize3Dgeometry=0,
        Compressoutputfile=0,
    )
    log.info("Saving '%s'.", outfile)

    pv.Delete(view)


def run3d(
    actx: PyOpenCLArrayContext,
    places: GeometryCollection,
    param: ds.ExperimentParameters,
    *,
    suffix: str = "",
    ext: str = "pdf",
    strip: bool = False,
    overwrite: bool = False,
) -> None:
    from arraycontext import unflatten

    # {{{ get geometry

    dd = places.auto_source.to_stage1()
    discr = places.get_discretization(dd.geometry, dd.discr_stage)
    log.info("STAGE1 nelements %6d ndofs %6d", discr.mesh.nelements, discr.ndofs)

    dd = places.auto_source.to_stage2()
    discr = places.get_discretization(dd.geometry, dd.discr_stage)
    log.info("STAGE2 nelements %6d ndofs %6d", discr.mesh.nelements, discr.ndofs)

    dd = places.auto_source
    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)

    # }}}

    # {{{ get clusters

    from pytential.linalg.cluster import partition_by_nodes

    indices, ctree = partition_by_nodes(
        actx,
        places,
        dofdesc=dd,
        tree_kind="adaptive-level-restricted",
        max_particles_in_box=param.max_particles_in_box,
    )
    log.info("nlevels %3d nclusters %6d ", ctree.nlevels, indices.nclusters)

    # }}}

    # {{{ get proxy balls

    from pytential.linalg.proxy import QBXProxyGenerator

    proxy = QBXProxyGenerator(
        places,
        approx_nproxy=param.proxy_approx_count,
        radius_factor=param.proxy_radius_factor,
        norm_type=param.proxy_norm_type,
    )
    pxy = proxy(actx, dd, indices).to_numpy(actx)

    # }}}

    from meshmode.discretization.visualization import make_visualizer

    marker = np.zeros(density_discr.ndofs)
    for i in range(indices.nclusters):
        isrc = indices.cluster_indices(i)
        marker[isrc] = np.sum(pxy.centers[:, i])
        marker[isrc] = i

    template_ary = actx.thaw(density_discr.nodes()[0])
    marker_dev = unflatten(template_ary, actx.from_numpy(marker), actx)

    basename = ds.make_filename(scriptname, param, suffix=suffix)
    vis = make_visualizer(
        actx,
        density_discr,
        vis_order=None,
        element_shrink_factor=None,
    )

    filename = ds.strip_timestamp(pathlib.Path(f"{basename}.vtu"), strip=strip)
    vis.write_vtk_file(filename, [("marker", marker_dev)], overwrite=overwrite)
    # vis.write_vtk_file(filename, [], overwrite=overwrite)
    log.info("Saving '%s'", filename)

    for i in [20]:
        postprocess_vtu(
            filename,
            i,
            indices.nclusters,
            pxy_radius=pxy.radii[i],
            box_radius=pxy.cluster_radii[i],
            center=pxy.centers[:, i],
            ext=ext,
        )

    import matplotlib.pyplot as mp

    fig = mp.figure()
    filename = filename.parent / f"{filename.stem}-clusters.{ext}"

    with ds.axis(fig, filename, overwrite=overwrite) as ax:
        cluster_sizes = np.sort(np.diff(indices.starts))
        ax.bar(np.arange(indices.nclusters), cluster_sizes, align="edge")
        ax.axhline(int(np.mean(cluster_sizes)), color="k", ls="--")

        ax.set_xlabel(r"Cluster Index (Sorted)")
        ax.set_ylabel(r"$n_i$ (\#DOFs)")
        ax.set_xlim([0, indices.nclusters])
        ax.set_ylim([0, 512])


# }}}

# }}}


# {{{


def experiment_visualize(
    actx_factory,
    *,
    suffix: str = "",
    ext: str = "png",
    strip: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> int:
    actx = actx_factory()

    ambient_dim = kwargs.pop("ambient_dim", 3)
    if ambient_dim == 2:
        # NOTE: this is the value used in Figure 4
        # kwargs["max_particles_in_box"] = 512
        param = ds.ExperimentParameters2(**kwargs)
        places = ds.make_geometry_collection(actx, param, force_equidistant=True)
        run2d(
            actx,
            places,
            param,
            suffix=suffix,
            ext=ext,
            overwrite=overwrite,
            strip=strip,
        )
    else:
        param = ds.ExperimentParametersTorus3(**kwargs)
        places = ds.make_geometry_collection(actx, param, force_equidistant=True)
        run3d(
            actx,
            places,
            param,
            suffix=suffix,
            ext=ext,
            overwrite=overwrite,
            strip=strip,
        )

    return 0


# }}}


if __name__ == "__main__":
    import argparse

    from meshmode import _acf

    parser = argparse.ArgumentParser()
    ds.add_arguments(parser)
    args, unknown = parser.parse_known_args()

    errno = experiment_visualize(
        _acf,
        ext=args.ext,
        suffix=args.suffix,
        overwrite=args.overwrite,
        **ds.parse_unknown_arguments(unknown),
    )

    raise SystemExit(errno)

# vim: fdm=marker
