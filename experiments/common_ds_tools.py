# SPDX-FileCopyrightText: 2022 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
import pathlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import matplotlib.pyplot as mp
import numpy as np
import numpy.linalg as la


if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterator, Sequence

    from arraycontext import PyOpenCLArrayContext
    from meshmode.discretization import Discretization
    from meshmode.mesh import Mesh
    from pytential.collection import GeometryCollection
    from pytential.linalg.skeletonization import SkeletonizationResult
    from pytential.linalg.utils import TargetAndSourceClusterList

PathLike = str | pathlib.Path
_DS_QBX_LOGGING_SET_UP = False
_DS_QBX_NO_TIMESTAMP = "DS_QBX_NO_TIMESTAMP" in os.environ


Array: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.inexact]]
RealArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
PointsArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.floating]]
IndexArray: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.integer]]


def set_up_logging(
    name: str,
    *,
    level: int = logging.INFO,
    markup: bool = False,
) -> logging.Logger:
    import rich.logging

    handler = rich.logging.RichHandler(markup=markup, log_time_format="[%X]")

    log = logging.getLogger(name)
    log.propagate = False
    log.setLevel(level)
    log.addHandler(handler)

    global _DS_QBX_LOGGING_SET_UP
    if not _DS_QBX_LOGGING_SET_UP:
        for module in ("boxtree", "sumpy", "pytential", "meshmode"):
            py_logger = logging.getLogger(module)
            py_logger.setLevel(level)
            py_logger.addHandler(handler)

        _DS_QBX_LOGGING_SET_UP = True

    return log


log = set_up_logging(__name__)


def seeded_rng(seed: int = 42) -> np.random.Generator:
    import scipy.linalg.interpolative as sli

    # NOTE: also set global seeds in case any other randomness is lurking around
    sli.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    return np.random.default_rng(seed=seed)


def add_arguments(parser: argparse.ArgumentParser, *, filename: bool = True) -> None:
    if filename:
        parser.add_argument(
            "-f",
            "--filename",
            type=pathlib.Path,
            help="Existing npz archive to visualize",
        )

    parser.add_argument(
        "-s",
        "--suffix",
        default="",
        help="Unique suffix to add to output file",
    )
    parser.add_argument(
        "-e",
        "--ext",
        default="pdf",
        help="Extension used for visualization",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite output files",
    )
    parser.add_argument(
        "-q",
        "--no-visualize",
        action="store_true",
        help="Visualize experiment results",
    )


def parse_unknown_arguments(unknown: Sequence[str]) -> dict[str, Any]:
    key = None
    value = None

    result = {}
    while unknown:
        arg = unknown.pop()
        if arg.startswith("--"):
            key = arg[2:]
        else:
            try:
                value = eval(arg)
            except Exception:
                value = arg

        if key and value:
            result[key] = value
            key = None
            value = None

    return result


# {{{ caching


def dc_dict(obj: Any) -> dict[str, Any]:
    from dataclasses import fields

    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def savez(filename: PathLike, *, overwrite: bool = False, **kwargs: Any) -> None:
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    from dataclasses import is_dataclass

    def _get(value: Any) -> Any:
        if isinstance(value, EOCRecorder):
            return value.error

        if is_dataclass(value):
            return dc_dict(value)

        if isinstance(value, list) and value and isinstance(value[0], TimingResult):
            return np.array([(t.walltime, t.mean, t.std) for t in value]).T

        return value

    np.savez(filename, **{k: _get(v) for k, v in kwargs.items()})


def make_filename(
    script: pathlib.Path, param: ExperimentParameters, suffix: str = ""
) -> pathlib.Path:
    suffix = f"-{suffix}" if suffix else ""
    suffix = f"-{param.ambient_dim}{param.lpot_type}{suffix}"

    if _DS_QBX_NO_TIMESTAMP:
        filename = f"{script.stem}{suffix}"
    else:
        from datetime import datetime

        filename = datetime.now().strftime(f"%Y-%m-%d-{script.stem}{suffix}")

    return script.parent / filename


def make_archive(
    script: pathlib.Path, param: ExperimentParameters, suffix: str = ""
) -> pathlib.Path:
    return make_filename(script, param, suffix).with_suffix(".npz")


def strip_timestamp(filename: pathlib.Path, *, strip: bool = True) -> pathlib.Path:
    if _DS_QBX_NO_TIMESTAMP or strip:
        parts = filename.stem.split("-")
        if parts[0].isdigit():
            parts = parts[3:]

        stem = "-".join(parts)
    else:
        stem = filename.stem

    return (filename.parent / stem).with_suffix(filename.suffix)


# }}}


# {{{ plotting


def set_recommended_matplotlib() -> None:
    # NOTE: since v1.1.0 an import is required to import the styles
    import scienceplots  # noqa: F401

    dirname = pathlib.Path(__file__).parent
    mp.style.use(["science", "ieee"])
    mp.style.use(dirname / "default.mplstyle")


def savefig(
    fig: mp.Figure,
    filename: PathLike,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> None:
    log.info("Saving '%s'", filename)

    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    fig.savefig(filename, **kwargs)


@contextmanager
def axis(
    fig: mp.Figure,
    filename: PathLike,
    *,
    overwrite: bool = False,
) -> Iterator[Any]:
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(filename)

    try:
        yield fig.gca()
    finally:
        savefig(fig, pathlib.Path(filename), overwrite=overwrite)
        fig.clf()


def loglog_scaling_line(
    ax: mp.Axes,
    x: Array,
    y: Array,
    *,
    order: float | tuple[float, float] = 1,
    label: str | Literal[False] | None = None,
    linestyle: str = "--",
) -> None:
    if isinstance(order, (int, float)):
        order = (float(order), 0)

    p, q = order
    if label is None:
        nlabel = "" if p == 0 else ("n" if p == 1 else f"n^{{{p:.1f}}}")
        lognlabel = "" if q == 0 else (r"\log n" if q == 1 else rf"\log^{{{q:.1f}}} n")
        label = f"$O({nlabel}{lognlabel})$"
    elif label is False:
        label = None

    if q == 0:
        ymax = np.max(y)
        ymin = y[0]
        xmax = x[-1]
        xmin = np.exp(np.log(xmax) + np.log(ymin / ymax) / p)
        log.info("%s: [%g, %g] x [%g, %g]", label, xmin, xmax, ymin, ymax)
    else:
        from scipy.special import lambertw

        ymax = np.max(y)
        ymin = y[0]
        xmax = x[-1]
        k = xmax * np.log(xmax) / (ymax / ymin)
        xmin = np.real(k / lambertw(k))
        log.info("%s: [%g, %g] x [%g, %g]", label, xmin, xmax, ymin, ymax)

    ax.loglog([xmin, xmax], [ymin, ymax], "k", linestyle=linestyle, label=label)


# }}}


# {{{ errors


def rel_norm(A: Array, B: Array, ord: Any = 2) -> float:
    """Computer a relative error norm."""
    b_norm = la.norm(B, ord=ord)
    if b_norm < 1.0e-8:
        b_norm = 1

    return la.norm(A - B) / b_norm


def estimate_order_of_convergence(x: RealArray, y: RealArray) -> tuple[float, float]:
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    eps = np.finfo(x.dtype).eps
    c = np.polyfit(np.log10(x + eps), np.log10(y + eps), 1)
    return float(10 ** c[-1]), float(c[-2])


def estimate_gliding_order_of_convergence(
    x: RealArray,
    y: RealArray,
    *,
    gliding_mean: int | None = None,
) -> RealArray:
    assert x.size == y.size
    if x.size <= 1:
        raise RuntimeError("Need at least two values to estimate order.")

    if gliding_mean is None:
        gliding_mean = x.size

    npoints = x.size - gliding_mean + 1
    return np.array(
        [
            estimate_order_of_convergence(
                x[i : i + gliding_mean], y[i : i + gliding_mean] + 1.0e-16
            )
            for i in range(npoints)
        ],
        dtype=x.dtype,
    )


@dataclass(frozen=True)
class EOCRecorder:
    name: str
    abscissa: str = "h"
    history: list[tuple[float, float]] = field(default_factory=list)

    def add_data_point(self, h: Any, error: Any) -> None:
        self.history.append((float(h), float(error)))

    @property
    def error(self) -> RealArray:
        _, error = np.array(self.history).T
        return error

    @property
    def h(self) -> RealArray:
        h, _ = np.array(self.history).T
        return h

    def __str__(self) -> str:
        return self.pretty_print()

    def pretty_print(
        self,
        *,
        abscissa_label: str | None = None,
        error_label: str | None = None,
        gliding_mean: int = 2,
        abscissa_format: str = "%.12e",
        error_format: str = "%.12e",
        eoc_format: str = "%.2f",
        table_type: str = "markdown",
    ) -> str:
        if not abscissa_label:
            abscissa_label = self.abscissa.strip("$")

        if not error_label:
            error_label = self.name.strip("$")

        h, error = np.array(self.history).T
        gm = estimate_gliding_order_of_convergence(h, error, gliding_mean=2)

        from rich.box import ASCII
        from rich.table import Table

        tbl = Table(abscissa_label, error_label, "EOC", box=ASCII)
        for i, (h_i, e_i) in enumerate(self.history):
            eoc = "" if i == 0 else eoc_format % gm[i - 1, 1]
            tbl.add_row(abscissa_format % h_i, error_format % e_i, eoc)

        if len(self.history) > 1:
            gm = estimate_gliding_order_of_convergence(h, error)
            tbl.add_row("Overall", "--", eoc_format % gm[0, 1])

        import io

        from rich.console import Console

        file = io.StringIO()
        console = Console(file=file)
        console.print(tbl)
        return file.getvalue()

    @classmethod
    def from_array(
        cls,
        name: str,
        h: RealArray,
        error: RealArray,
        *,
        abscissa: str = "h",
    ) -> EOCRecorder:
        eoc = cls(name=name, abscissa=abscissa)
        for h_i, e_i in zip(h, error, strict=True):
            eoc.add_data_point(h_i, e_i)

        return eoc


def visualize_eoc(
    filename: PathLike,
    *eocs: EOCRecorder,
    order: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legends: tuple[str, ...] | None = None,
    enable_legend: bool = True,
    keep_color: bool = False,
    first_legend_only: bool = False,
    align_order_to_abscissa: bool = False,
    overwrite: bool = False,
) -> None:
    """
    :arg order: expected order for all the errors recorded in *eocs*.
    :arg abscissa: name for the abscissa.
    """
    filename = pathlib.Path(filename)
    if not overwrite and filename.exists():
        raise FileExistsError(f"output file '{filename}' already exists")

    if not eocs:
        log.warning("No data given ('eocs' is empty), so nothing is plotted.")
        return

    fig = mp.figure()
    ax = fig.gca()

    from pytools import single_valued

    if xlabel is None:
        abscissa = single_valued([eoc.abscissa for eoc in eocs])
        xlabel = abscissa
    else:
        abscissa = xlabel.strip("$")

    if legends is None:
        legends = tuple(eoc.name for eoc in eocs)

    if len(legends) != len(eocs):
        raise ValueError("'legends' does not match given eocs")

    if first_legend_only:
        legends = [legends[0]] + [None] * (len(legends) - 1)
    legends = [(name if name else None) for name in legends]

    # {{{ plot eoc

    max_h = 0.0
    min_e, max_e = np.inf, 0.0

    color = None
    for eoc, name in zip(eocs, legends, strict=True):
        max_h = max(max_h, np.max(eoc.h))
        min_e = min(min_e, np.min(eoc.error))
        max_e = max(max_e, np.max(eoc.error))

        (line,) = ax.loglog(eoc.h, eoc.error, "o-", color=color, label=name)
        if keep_color and color is None:
            color = line.get_color()

    # }}}

    # {{{ plot order

    if order is not None:
        if order == 1:
            olabel = rf"O({abscissa})"
        elif isinstance(order, int):
            olabel = rf"O({abscissa}^{{{order}}})"
        elif isinstance(order, float):
            olabel = rf"O({abscissa}^{{{order:.2f}}})"
        else:
            raise ValueError(f"unsupported 'order': {order}")

        if align_order_to_abscissa:
            ax.loglog(eocs[0].h, eocs[0].h, "k--", label=f"${olabel}$")
        else:
            min_h = np.exp(np.log(max_h) + np.log(min_e / max_e) / order)
            ax.loglog([max_h, min_h], [max_e, min_e], "k--", label=f"${olabel}$")

    # }}}

    ax.grid(True, which="major", linestyle="-", alpha=0.75)
    ax.grid(True, which="minor", linestyle="--", alpha=0.5)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if enable_legend:
        ax.legend()

    savefig(fig, filename, overwrite=overwrite)
    mp.close(fig)


# }}}


# {{{ timings


@dataclass(frozen=True)
class TimingResult:
    name: str
    walltime: float
    mean: float
    std: float

    @classmethod
    def from_array(cls, name: str, timings: RealArray) -> list[TimingResult]:
        return [cls(name, t[0], t[1], t[2]) for t in timings.T]

    def __str__(self) -> str:
        return f"{self.mean:.5f}s ± {self.std:.3f}"


def timeit(
    stmt: Callable[[], Any],
    *,
    name: str = "timing",
    repeat: int = 8,
    number: int = 1,
    skip: int = 1,
) -> TimingResult:
    """Run *stmt* using :func:`timeit.repeat`.

    :returns: a :class:`TimeResult` with statistics about the runs.
    """

    import timeit as _timeit

    r = _timeit.repeat(stmt=stmt, repeat=repeat + 1, number=number)
    r = np.array(r[skip:])

    return TimingResult(
        name=name,
        walltime=np.min(r),
        mean=np.mean(r),
        std=np.std(r, ddof=1),
    )


def visualize_timings(
    filename: PathLike,
    abscissa: RealArray,
    *timings: Sequence[TimingResult],
    xlabel: str | None = None,
    ylabel: str | None = None,
    legends: Sequence[str] | None = None,
    with_stats: bool = True,
    overwrite: bool = False,
) -> None:
    if ylabel is None:
        ylabel = "Time~(s)"

    if legends is None:
        legends = [timing[0].name.replace(" ", "~") for timing in timings]

    if len(legends) != len(timings):
        raise ValueError(f"Got {len(legends)} legends and {len(timings)} timings")

    fig = mp.figure()
    ax = fig.gca()

    if xlabel:
        ax.set_xlabel(f"${xlabel}$")
    ax.set_ylabel(f"${ylabel}$")

    for n, timing in enumerate(timings):
        mean = np.array([t.mean for t in timing])

        ylabel = f"${legends[n]}$" if len(legends) > 1 else None

        if with_stats:
            std = np.array([t.std for t in timing])
            ax.fill_between(abscissa, mean - std, mean + std, alpha=0.25)

        ax.plot(abscissa, mean, label=ylabel)

    if len(legends) > 1:
        ax.legend()

    savefig(fig, filename, overwrite=overwrite)
    mp.close(fig)


@contextmanager
def profileit(
    outfile: PathLike | None = None,
    *,
    overwrite: bool = False,
) -> Iterator[None]:
    if outfile is not None:
        outfile = pathlib.Path(outfile)
        if not overwrite and outfile.exists():
            raise FileExistsError(outfile)

    import cProfile

    p = cProfile.Profile()
    p.enable()

    try:
        yield None
    finally:
        p.disable()
        if outfile is not None:
            p.dump_stats(outfile)


# }}}


# {{{ geometry


@dataclass(frozen=True)
class GeometryParameters:
    ambient_dim: int

    resolution: int
    mesh_order: int
    target_order: int
    qbx_order: int
    source_ovsmp: int

    inner_radius: float
    outer_radius: float

    @property
    def fine_order(self) -> int:
        return self.source_ovsmp * self.target_order

    def make_mesh(self) -> Mesh:
        raise NotImplementedError


def make_uniform_random_array(
    actx: PyOpenCLArrayContext,
    discr: Discretization,
    *,
    rng: np.random.Generator | None = None,
) -> Any:
    if rng is None:
        rng = np.random.default_rng()

    from arraycontext import unflatten
    from meshmode.discretization import Discretization
    from pytential.source import PointPotentialSource

    if isinstance(discr, Discretization):
        template = actx.thaw(discr.nodes()[0])
        return unflatten(
            template, actx.from_numpy(rng.uniform(-1.0, 1.0, size=discr.ndofs)), actx
        )
    elif isinstance(discr, PointPotentialSource):
        # NOTE: these are going to be used as charges and we want them to have
        # zero average charge so that all the operators work nicely.
        result = rng.normal(size=discr.ndofs)
        result[-1] = -np.sum(result[:-1])
        assert np.sum(result) < 1.0e-15

        return actx.from_numpy(result)
    else:
        raise TypeError(f"unknown geometry type: {type(discr)}")


def make_circular_point_group(
    ambient_dim: int,
    npoints: int,
    radius: float,
    center: Array | None = None,
):
    if center is None:
        center = np.array([0.0, 0.0])
    center = np.asarray(center)

    t = np.linspace(0.0, 1.0, npoints, endpoint=False)
    t = 2.0 * np.pi * t**1.5

    result = np.zeros((ambient_dim, npoints))
    result[:2, :] = center[:, None] + radius * np.vstack([np.cos(t), np.sin(t)])

    return result


def make_source_and_target_points(
    actx: PyOpenCLArrayContext,
    *,
    side: int,
    inner_radius: float,
    outer_radius: float,
    ambient_dim: int,
    nsources: int = 10,
    ntargets: int = 20,
) -> tuple[PointsArray, PointsArray]:
    if side == -1:
        test_src_geo_radius = outer_radius
        test_tgt_geo_radius = inner_radius
    elif side == +1:
        test_src_geo_radius = inner_radius
        test_tgt_geo_radius = outer_radius
    else:
        raise ValueError(f"unknown side: {side}")

    from pytential.source import PointPotentialSource

    test_sources = make_circular_point_group(ambient_dim, nsources, test_src_geo_radius)
    point_source = PointPotentialSource(actx.freeze(actx.from_numpy(test_sources)))

    from pytential.target import PointsTarget

    test_targets = make_circular_point_group(ambient_dim, ntargets, test_tgt_geo_radius)
    point_target = PointsTarget(actx.freeze(actx.from_numpy(test_targets)))

    return point_source, point_target


def make_geometry_sphere(
    actx: PyOpenCLArrayContext,
    *,
    target_order: int,
    radius: float,
    center: PointsArray,
) -> Discretization:
    from meshmode.mesh.generation import generate_sphere

    ref_mesh = generate_sphere(1, target_order, uniform_refinement_rounds=1)

    from meshmode.mesh.processing import affine_map

    mesh = affine_map(ref_mesh, A=radius, b=center)

    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
        InterpolatoryQuadratureSimplexGroupFactory,
    )

    return Discretization(
        actx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order)
    )


def make_geometry_collection(
    actx: PyOpenCLArrayContext,
    param: GeometryParameters,
    force_equidistant: bool = False,
) -> GeometryCollection:
    import meshmode.discretization.poly_element as mpoly
    from meshmode.discretization import Discretization

    if force_equidistant:
        factory = mpoly.InterpolatoryEquidistantGroupFactory(param.target_order)
    else:
        factory = mpoly.InterpolatoryQuadratureSimplexGroupFactory(param.target_order)

    mesh = param.make_mesh()
    pre_density_discr = Discretization(actx, mesh, factory)

    from pytential.qbx import QBXLayerPotentialSource

    qbx_fmm = QBXLayerPotentialSource(
        pre_density_discr,
        fine_order=param.fine_order,
        qbx_order=param.qbx_order,
        fmm_backend="sumpy",
        fmm_order=param.qbx_order + 6,
    )

    qbx_ds = QBXLayerPotentialSource(
        pre_density_discr,
        fine_order=param.fine_order,
        qbx_order=param.qbx_order,
        fmm_order=False,
    )

    sources, targets = make_source_and_target_points(
        actx,
        side=-1,
        inner_radius=param.inner_radius,
        outer_radius=param.outer_radius,
        ambient_dim=mesh.ambient_dim,
    )

    from pytential import GeometryCollection, sym

    dd = sym.DOFDescriptor("ds", discr_stage=sym.QBX_SOURCE_STAGE2)
    places = GeometryCollection(
        {
            "ds": qbx_ds,
            "fmm": qbx_fmm,
            "point_sources": sources,
            "point_targets": targets,
        },
        auto_where=dd,
    )

    density_discr = places.get_discretization(dd.geometry, dd.discr_stage)
    log.info("nelements:     %d", density_discr.mesh.nelements)
    log.info("ndofs:         %d", density_discr.ndofs)

    return places


@dataclass(frozen=True, eq=True)
class ExperimentParameters(GeometryParameters):
    lpot_type: str = "d"
    helmholtz_k: int = 0

    mesh_order: int = 4
    target_order: int = 4
    qbx_order: int = 4
    # NOTE: none of the tests use QBX_SOURCE_STAGE2_QUAD
    source_ovsmp: int = 4

    inner_radius: float = 1.0
    outer_radius: float = 1.0

    id_eps: float = 1.0e-8

    proxy_approx_count: int = 64
    proxy_radius_factor: float = 1.15
    proxy_norm_type: str = "l2"
    proxy_weighted: tuple[bool, bool] = (True, True)
    proxy_remove_source_transforms: bool = False

    max_particles_in_box: int = 128

    def make_mesh(self) -> Mesh:
        raise NotImplementedError

    def get_model_proxy_count(self) -> IndexArray:
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class ExperimentParameters2(ExperimentParameters):
    ambient_dim: int = 2

    # NOTE: resolution = number of elements in 2d
    resolution: int = 2048
    resolutions: tuple[int, ...] = field(
        default_factory=lambda: (128, 256, 512, 1024, 2048, 3072, 4096)
    )

    # NOTE: these are ballparked from testing the error model
    proxy_approx_count: int = 192
    # NOTE: this is ballparked to give 5-ish levels and 100-ish cluster sizes
    max_particles_in_box: int = 192

    starfish_arms: int = 16
    starfish_amplitude: float = 0.25

    # NOTE: these are valid for the (arms, amplitude) above
    inner_radius: float = 0.25
    outer_radius: float = 2.0

    def make_mesh(self) -> Mesh:
        from meshmode.mesh.generation import NArmedStarfish, make_curve_mesh

        mesh = make_curve_mesh(
            NArmedStarfish(self.starfish_arms, self.starfish_amplitude),
            np.linspace(0, 1, self.resolution + 1),
            self.mesh_order,
        )

        return mesh

    def get_model_proxy_count(self) -> IndexArray:
        from pytools.persistent_dict import KeyBuilder

        keyb = KeyBuilder()
        param_hash = keyb(self)
        if param_hash not in {"1e24ef0cda3d738a", "1cb070a9892ae75f"}:
            raise RuntimeError(f"These parameters do not match: {param_hash}")

        # NOTE: these are obtained by running `ds-forward-model` and fitting
        # the results using `ds.em_constant_fit` using the same parameters.
        # MAKE SURE TO UPDATE THESE IF ANY OF THE PARAMETERS CHANGE!!!
        if self.lpot_type == "s":
            return np.array([
                1,
                1,
                1,
                9,
                24,
                41,
                58,
                74,
                91,
                107,
                121,
                139,
                150,
                167,
            ])
        elif self.lpot_type == "d":
            return np.array([
                1,
                1,
                1,
                1,
                9,
                25,
                39,
                57,
                73,
                89,
                106,
                121,
                135,
                151,
            ])
        else:
            raise ValueError(f"Unknown 'lpot_type': {self.lpot_type!r}")


@dataclass(frozen=True, eq=True)
class ExperimentParametersTorus3(ExperimentParameters):
    ambient_dim: int = 3

    # NOTE: resolution = (n_major, n_minor)
    resolution: tuple[int, int] | int = 1
    resolutions: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: (
            (20, 10),
            (25, 15),
            (32, 22),
            (42, 32),
            (50, 40),
        )
    )

    # NOTE: these are ballparked from testing the error model
    proxy_approx_count: int = 512
    # NOTE: this is ballparked to give 5-ish levels and 300-ish cluster sizes
    max_particles_in_box: int = 512 + 256

    torus_radius_outer: float = 10.0
    torus_radius_inner: float = 2.0

    inner_radius: float = 10.0
    outer_radius: float = 14.0

    def make_mesh(self) -> Mesh:
        from meshmode.mesh.generation import generate_torus
        from meshmode.mesh.refinement import refine_uniformly

        if isinstance(self.resolution, int):
            mesh = generate_torus(
                self.torus_radius_outer,
                self.torus_radius_inner,
                order=self.mesh_order,
            )
            mesh = refine_uniformly(mesh, self.resolution)
        elif isinstance(self.resolution, tuple) and len(self.resolution) == 2:
            mesh = generate_torus(
                self.torus_radius_outer,
                self.torus_radius_inner,
                self.resolution[0],
                self.resolution[1],
                order=self.mesh_order,
            )
        else:
            raise TypeError(f"Unsupported resolution type: {self.resolution}")

        return mesh

    def get_model_proxy_count(self) -> IndexArray:
        from pytools.persistent_dict import KeyBuilder

        keyb = KeyBuilder()
        param_hash = keyb(self)
        if param_hash not in {"dbac5395a11117a3", "7b73797f949ae77c"}:
            raise RuntimeError(f"These parameters do not match: {param_hash}")

        # NOTE: MAKE SURE TO UPDATE THESE IF ANY OF THE PARAMETERS CHANGE!!!
        if self.lpot_type == "s":
            return np.array([
                0,
                0,
                0,
                0,
                8,
                28,
                57,
                99,
                149,
                214,
                285,
                367,
                465,
                567,
            ])
        elif self.lpot_type == "d":
            return np.array([
                0,
                0,
                0,
                0,
                0,
                0,
                7,
                26,
                55,
                96,
                146,
                210,
                280,
                367,
            ])
        else:
            raise ValueError(f"Unknown 'lpot_type': {self.lpot_type!r}")


@dataclass(frozen=True, eq=True)
class ExperimentParametersSphere3(ExperimentParameters):
    ambient_dim: int = 3

    # NOTE: resolution = gmsh spacing
    resolution: float = 0.25
    resolutions: tuple[float, ...] = field(
        default_factory=lambda: (0.5, 0.4, 0.3, 0.25, 0.21, 0.18, 0.15)
    )

    # NOTE: this will be approximated by the error model?
    proxy_approx_count: int = 128 + 128
    # NOTE: this is hard to get right using the tree from boxtree, i.e. box
    # sizes can vary quite a bit based on geometry; this looks nice enough.
    max_particles_in_box: int = 512 + 256

    sphere_radius: float = 1.5

    def make_mesh(self) -> Mesh:
        from meshmode.mesh.io import ScriptSource, generate_gmsh

        script = ScriptSource(
            f"""
            Mesh.CharacteristicLengthMax = {self.resolution};
            Mesh.HighOrderOptimize = 1;
            Mesh.Algorithm = 1;

            SetFactory("OpenCASCADE");
            Sphere(1) = {{0, 0, 0, {self.sphere_radius}}};
            """,
            "geo",
        )

        mesh = generate_gmsh(
            script,
            order=self.mesh_order,
            dimensions=2,
            force_ambient_dim=3,
            target_unit="MM",
        )

        return mesh

    def get_model_proxy_count(self) -> IndexArray:
        raise NotImplementedError


# }}}


# {{{ cluster handling


def intersect1d(x: RealArray, y: RealArray) -> IndexArray:
    """Find the indices that correspond to the intersection of *x* and *y*."""
    return np.where((x.reshape(1, -1) - y.reshape(-1, 1)) == 0)[1]


def cluster_take(mat: Array, cl: TargetAndSourceClusterList) -> Array:
    """Take and reshape all clusters from *mat*.

    :arg mat: an array of shape ``(n,)`` containing matrix entries for all the
        indices in *cl*.
    :returns: an object :class:`~numpy.ndarray` where each entry corresponds to
        the respective cluster and has the shape of the cluster.
    """
    from pytools.obj_array import make_obj_array

    return make_obj_array([
        cl.flat_cluster_take(mat, i).reshape(cl.cluster_shape(i, i))
        for i in range(cl.nclusters)
    ])


def skeleton_cluster_take(
    mat: Array,
    skeleton: SkeletonizationResult,
    clusters: tuple[tuple[int, int], ...],
) -> Array:
    """Take a subset of the entries in *mat* that correspond to the skeleton.

    :arg mat: the result of :func:`cluster_take`.
    :arg clusters: a tuple of ``(itgt, jsrc)`` cluster indices.
    """

    result = []
    for i, (itgt, jsrc) in enumerate(clusters):
        itgt = intersect1d(
            skeleton.tgt_src_index.targets.cluster_indices(itgt),
            skeleton.skel_tgt_src_index.targets.cluster_indices(itgt),
        )
        jsrc = intersect1d(
            skeleton.tgt_src_index.sources.cluster_indices(jsrc),
            skeleton.skel_tgt_src_index.sources.cluster_indices(jsrc),
        )

        result.append(mat[i][np.ix_(itgt, jsrc)])

    from pytools.obj_array import make_obj_array

    return make_obj_array(result)


# }}}


# {{{ error model


class Model(NamedTuple):
    """Error model parameters."""

    ambient_dim: int
    """Ambient dimension of the problem."""
    id_eps: float
    """Tolerance used in the interpolative decomposition."""

    pxy_count: int
    """Maximum number of proxy points."""
    pxy_cluster_ratio: float
    """The (largest) ratio of proxy radius to cluster radius."""
    pxy_qbx_ratio: float
    """The ratio of proxy radius to cluster radius including the QBX."""
    pxy_radius_max: float
    """Largest proxy radius (maximum over all clusters)."""
    pxy_radius_min: float
    """Smallest proxy radius (minimum over all clusters)."""


def to_model(ary: RealArray) -> Model:
    return Model(int(ary[0]), ary[1], int(ary[2]), ary[3], ary[4], ary[5], ary[6])


def to_models(ary: RealArray) -> tuple[Model, ...]:
    assert ary.ndim == 2
    return tuple(to_model(p) for p in ary)


class ModelConstant(NamedTuple):
    """Constants used in the error model (a maximum over all the clusters)."""

    lr: float
    r"""Constant multiplying the whole error model depending on the norm
    of :math:`(\mathbf{L}, \mathbf{R})`."""

    id: float
    r"""Constant multiplying :math:`\epsilon_{id}`."""
    mp: float
    r"""Constant multiplying :math:`\epsilon_{ms}` and :math:`\epsilon_{mt}`."""

    def scale(self, m: Model) -> tuple[float, float]:
        alpha = m.pxy_qbx_ratio

        if m.ambient_dim == 2:
            c_id = 2.0 * np.pi * self.id * m.pxy_radius_max
            c_mp = self.mp / (2.0 * np.pi) / (alpha - 1)
        elif m.ambient_dim == 3:
            c_id = 4.0 * np.pi * self.id * m.pxy_radius_max**2
            c_mp = self.mp / (4.0 * np.pi * m.pxy_radius_max) / (alpha - 1)
        else:
            raise ValueError(f"Unsupported dimension: {m.ambient_dim}")

        return c_id, c_mp


def to_const(ary: RealArray) -> ModelConstant:
    assert ary.shape == (3,)
    return ModelConstant(*ary)


def to_consts(ary: RealArray) -> tuple[ModelConstant, ...]:
    assert ary.ndim == 2
    return tuple(to_const(c) for c in ary)


def em_parameter_from_skeletonization(
    actx: PyOpenCLArrayContext,
    results: Sequence[SkeletonizationResult],
    id_eps: float,
    proxy_radius_factor: float,
) -> Model:
    r"""Compute the cluster and proxy ratios from the skeletonization results.

    :arg results: a list of skeletonization results for each level of hierarchical
        decomposition; the first entry contains the highest level
        :math:`\ell = N_{levels} - 1`.
    """
    assert len(results) >= 1

    # NOTE: taking the values from the leaf level, which is 0 in our notation
    result = results[0]
    pxy = result._src_eval_result.pxy

    pxy_radii = actx.to_numpy(pxy.radii)
    cluster_radii = actx.to_numpy(pxy.cluster_radii)

    return Model(
        ambient_dim=pxy.places.ambient_dim,
        id_eps=id_eps,
        pxy_count=pxy.pxyindex.cluster_size(0),
        pxy_cluster_ratio=np.max(pxy_radii / cluster_radii),
        pxy_qbx_ratio=proxy_radius_factor,
        pxy_radius_max=np.max(pxy_radii),
        pxy_radius_min=np.min(pxy_radii),
    )


def em_constants_from_skeletonization(
    actx,
    results: Sequence[SkeletonizationResult],
    *,
    ord: Any = 2,
) -> ModelConstant:
    r"""Compute the model constants from the skeletonization results.

    :arg results: a list of skeletonization results for each level of hierarchical
        decomposition; the first entry contains the highest level
        :math:`\ell = N_{levels} - 1`.

    :arg ord: type of the norm used to compute (mostly) matrix norms.
    """
    from arraycontext import flatten
    from pytential import bind, sym

    result = results[0]
    pxy = result._src_eval_result.pxy

    # get geometry
    dofdesc = pxy.dofdesc
    places = pxy.places

    # compute weights and area elements
    waa = bind(
        places,
        sym.weights_and_area_elements(places.ambient_dim, dofdesc=dofdesc),
    )(actx)

    waa = actx.to_numpy(flatten(waa, actx))
    w_far = la.norm(waa, ord=np.inf)

    # get index sets
    nbrindex = result._src_eval_result.nbrindex.targets
    srcindex = result._src_eval_result.nbrindex.sources

    # compute coefficients
    norm_r = 0.0
    norm_l = 0.0
    c_id = 0.0
    c_mp = 0.0

    for i in range(result.nclusters):
        norm_r_i = la.norm(result.R[i], ord=ord)
        norm_r = max(norm_r, norm_r_i)
        norm_l_i = la.norm(result.L[i], ord=ord)
        norm_l = max(norm_r, norm_l_i)

        # Equation 4.4 (left)
        w_near = la.norm(nbrindex.cluster_take(waa, i), ord=np.inf)
        a0 = w_far / w_near
        b0 = 1.0
        c_id = max(a0, b0)

        # Equation 4.4 (right)
        w_i = la.norm(srcindex.cluster_take(waa, i), ord=np.inf)
        a1 = (1.0 + norm_l_i) * w_far
        b1 = (1.0 + norm_r_i) * w_i
        c_mp = max(a1, b1)

    # Equation 4.2: Full error is given by
    #   c_lr * ((1 + c_id 4 pi R / q) id_eps
    #          + c_mp / (4 pi R (alpha - 1)) 1 / alpha^p)
    c_lr = (2.0 + norm_r + norm_l) / 2.0
    return ModelConstant(lr=c_lr, id=c_id, mp=c_mp)


RANK_OFFSET2D = 1
RANK_OFFSET3D = 100


def em_proxy_from_order(ambient_dim: int, order: int, c_rank: int | None = None) -> int:
    if ambient_dim == 2:
        if c_rank is None:
            c_rank = RANK_OFFSET2D

        return (order + 1) // c_rank
    elif ambient_dim == 3:
        if c_rank is None:
            c_rank = RANK_OFFSET3D

        return 2 * order * (order + 1) // c_rank
    else:
        raise ValueError(f"Unsupported dimension: {ambient_dim}")


def em_order_from_proxy(
    ambient_dim: int, nproxy: int, c_rank: int | None = None
) -> int:
    from math import ceil

    if ambient_dim == 2:
        if c_rank is None:
            c_rank = RANK_OFFSET2D

        return ceil(c_rank * nproxy - 1)
    elif ambient_dim == 3:
        if c_rank is None:
            c_rank = RANK_OFFSET3D

        return ceil(0.5 * (np.sqrt(2 * c_rank * nproxy + 1) - 1))
    else:
        raise ValueError(f"Unsupported dimension: {ambient_dim}")


def em_constant_correction(
    m: Model,
    c: ModelConstant,
    target_error: float,
    *,
    c_rank: int | None = None,
    opt: bool = False,
) -> ModelConstant:
    """Fits the constants *mc* in the model *mp* to match the *target_error*
    (as best as possible).

    :arg opt: if *True*, a least squares fit is used to minimize the error.
        Otherwise, a direct approximation is attempted.
    """
    # NOTE: we cannot do better than id_eps on the inside
    target_error = max(target_error, 2.0 * c.lr * m.id_eps)

    def get_estimated_error(x: np.array) -> float:
        return em_estimate_error_from_parameters(
            m, ModelConstant(c.lr, x[0] * c.id, x[1] * c.mp), c_rank=c_rank
        )

    def f(x: np.array) -> float:
        # NOTE: minimize relative error
        return np.array([target_error - get_estimated_error(x)]) / target_error

    if opt:
        import scipy.optimize as so

        tol = 2 * np.finfo(1.0).eps
        result = so.least_squares(
            f,
            np.array([1.0, 1.0]),
            bounds=([target_error / 10, target_error / 10], np.inf),
            ftol=tol,
            xtol=tol,
            gtol=tol,
        )
        x = result.x
    else:
        # NOTE: This is based on Equation 4.2 where we want to find the constants
        # so that each term has the target error or close to it
        alpha = m.pxy_qbx_ratio
        pxy_count = m.pxy_count
        if m.ambient_dim == 2:
            c_id_factor = pxy_count / (2.0 * np.pi * m.pxy_radius_max)
            c_mp_factor = (alpha - 1.0) * (2.0 * np.pi)
        elif m.ambient_dim == 3:
            c_id_factor = pxy_count / (4.0 * np.pi * m.pxy_radius_max)
            c_mp_factor = (alpha - 1.0) * (4.0 * np.pi * m.pxy_radius_max)
        else:
            raise ValueError(f"Unsupported dimension: {m.ambient_dim}")

        # this matches c_id such that the id term == target_error
        c_id = c_id_factor * (target_error / c.lr - m.id_eps)
        id_factor = c_id / c.id

        # this matches c_mp such that the mp term == target_error
        p = em_order_from_proxy(m.ambient_dim, pxy_count, c_rank=c_rank)
        c_mp = c_mp_factor * alpha**p * target_error / c.lr
        mp_factor = c_mp / c.mp

        x = np.array([id_factor, mp_factor])

    log.info(m)
    log.info(c)
    log.info(
        "Result: target %.12e before %.12e after %.12e relative %.12e",
        target_error,
        get_estimated_error(np.array([1.0, 1.0])),
        get_estimated_error(x),
        f(x).item(),
    )

    return ModelConstant(lr=1.0, id=x[0], mp=x[1])


def em_constant_fit(
    ms: tuple[Model, ...],
    cs: tuple[ModelConstant, ...],
    target_error: RealArray,
    *,
    c_rank: int | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[ModelConstant, ...]:
    """Fits the constants *mc* for the models *mp* to the given target errors.

    This is similar to :func:`em_constant_correction`, but it will fit the
    constants over a range of models.
    """
    if rng is None:
        rng = np.random.default_rng()

    # NOTE: we cannot do better than id_eps on the inside
    dtype = target_error.dtype
    target_error = np.array(
        [
            max(target_error[i], 2.0 * cs[i].lr * ms[i].id_eps)
            for i in range(target_error.size)
        ],
        dtype=dtype,
    )

    def get_estimated_error(x: RealArray) -> RealArray:
        return np.array([
            em_estimate_error_from_parameters(
                m_i,
                ModelConstant(c_i.lr, x[0] * c_i.id, x[1] * c_i.mp),
                c_rank=c_rank,
            )
            for m_i, c_i in zip(ms, cs, strict=True)
        ])

    def f(x: RealArray) -> RealArray:
        return abs(get_estimated_error(x) - target_error) / target_error

    import scipy.optimize as so

    tol = 2 * np.finfo(dtype).eps
    xmin = 100 * tol
    min_distance = np.inf
    x = np.array([xmin, xmin])

    for _ in range(128):
        x0 = xmin + (1.0 - xmin) * rng.random(2)
        result = so.least_squares(
            f,
            x0,
            bounds=([xmin, xmin], np.inf),
            ftol=tol,
            xtol=tol,
            gtol=tol,
        )

        distance = abs(result.x[1] - result.x[0])
        if distance < min_distance:
            min_distance = distance
            x = result.x

    log.info("Result: x (%.12e, %.12e) error match %.12e", *x, la.norm(f(x), ord=2))

    if verbose:
        for i in range(target_error.size):
            cstar = ModelConstant(cs[i].lr, x[0] * cs[i].id, x[1] * cs[i].mp)
            log.info("%s", cs[i])
            log.info("%s", cstar)
            log.info(
                "Error %.12e Corrected %.12e Target %.12e (rel %.12e)",
                em_estimate_error_from_parameters(ms[i], cs[i]),
                em_estimate_error_from_parameters(ms[i], cstar),
                target_error[i],
                la.norm(f(result.x), ord=np.inf),
            )
            log.info("")

    return ModelConstant(lr=1.0, id=x[0], mp=x[1])


def em_estimate_error_from_parameters(
    m: Model,
    c: ModelConstant,
    *,
    c_rank: int | None = None,
) -> float:
    """Estimate the error given the parameters.

    :arg m: model parameters (see :func:`em_parameter_from_skeletonization`).
    :arg c: model constants (see :func:`em_constants_from_skeletonization`) .

    :returns: an estimate for the error using the error model.
    """
    p = em_order_from_proxy(m.ambient_dim, m.pxy_count, c_rank=c_rank)

    alpha = m.pxy_qbx_ratio
    c_id, c_mp = c.scale(m)

    # Equation 4.2
    return c.lr * ((1.0 + c_id) * m.id_eps + c_mp / alpha**p)


def em_estimate_proxy_from_parameters(
    m: Model,
    c: ModelConstant,
    *,
    c_rank: int | None = None,
) -> int:
    """Estimate the number of required proxy points to get an error of *id_eps*.

    :arg m: model parameters (see :func:`em_parameter_from_skeletonization`).
    :arg c: model constants (see :func:`em_constants_from_skeletonization`) .

    :returns: an estimated number of proxy points.
    """
    # Corollary 4.3
    alpha = m.pxy_qbx_ratio
    c_id, c_mp = c.scale(m)
    p = -np.log((1.0 + c_id) / c_mp * m.id_eps) / np.log(alpha)

    # NOTE: this can happen for small id_eps. the issue seems to be that the
    # number of proxies bottoms out, i.e. for id_eps > 1.0e-5, 8-16 proxies is
    # enough, so our `em_constant_fit` does a bad job at fitting to that.
    p = max(0, p)

    return max(0, em_proxy_from_order(m.ambient_dim, int(p), c_rank=c_rank))


# }}}
