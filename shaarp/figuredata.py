"""Headless plot-ready figure data from the VALIDATED SHAARP.py public facades.

This is the GUI-plot backing the SHAARP.ml GUI exposes as "Generate Maker Fringes
Plot", "Sample-rotation scan", and "Fresnel Coefficients Plot" -- but as pure,
side-effect-free DATA rather than a widget: each builder returns a `FigureData`
(x array, named y-series, axis labels, title, and the underlying validation
status) that a notebook, REPL, web view, or matplotlib call can render however it
likes. No plotting library is imported at module load; `to_matplotlib(ax)` imports
matplotlib lazily only if the caller wants a quick render.

All numbers come from the validated facades (run_maker_fringes /
run_sample_rotation / run_fresnel_sweep), so the plotted curves are exactly the
live-Wolfram-validated outputs (Maker: ext1+ml1/ml2/ml3/ml5/ml6; SampleRotate
single-layer ext1; Fresnel: power-flux regression).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .api import run_fresnel_sweep, run_maker_fringes, run_sample_rotation


@dataclass(frozen=True)
class FigureData:
    """Plot-ready data: x, named y-series, labels, title, validation provenance."""

    title: str
    x_label: str
    y_label: str
    x: list[float]
    series: dict[str, list[float]]
    validation_status: str
    note: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_matplotlib(self, ax=None):
        """Lazy, optional render. Imports matplotlib only when called."""
        import matplotlib.pyplot as plt  # noqa: PLC0415 (intentional lazy import)

        if ax is None:
            _, ax = plt.subplots()
        for name, y in self.series.items():
            ax.plot(self.x, y, label=name, marker="o", markersize=3)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(self.title)
        ax.legend(fontsize=8)
        return ax


def _real_list(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=complex)
    return [float(np.real(v)) for v in arr.ravel()]


def maker_figure_data(system, theta_grid=None, options=None) -> FigureData:
    """Maker-fringe transmitted-SHG intensity vs incidence angle (parallel + perp).

    Uses ``run_maker_fringes`` (SHAARP.ml unit normalization by default), so the
    plotted ``parallel``/``perpendicular`` series are exactly the live-Mathematica
    ``listMFpara``/``listMFperp`` quantities -- matched to ~1e-15 on the non-singular
    validated cases (see ``tests/test_figuredata.py::MakerPlotMathematicaValidationTests``)."""
    result = run_maker_fringes(system, theta_grid, options)
    n = result.numeric
    return FigureData(
        title="Maker fringes - transmitted 2omega SHG",
        x_label="incidence angle theta (deg)",
        y_label="transmitted SHG intensity (arb.)",
        x=_real_list(n["theta_deg"]),
        series={
            "parallel analyzer": _real_list(n["parallel_intensity"]),
            "perpendicular analyzer": _real_list(n["perpendicular_intensity"]),
        },
        validation_status=result.validation.status,
        note="Plotted intensities are listMFpara/listMFperp (SHAARP.ml unit normalization), "
             "matched to live Mathematica MFList to ~1e-15 on non-singular cases (ext1 + ml1/2/3/5/6).",
        meta={"kind": result.kind},
    )


def sample_rotation_figure_data(system, azimuth_grid, options=None) -> FigureData:
    """SHG intensity vs sample azimuth (reflected + transmitted, parallel + perp)."""
    result = run_sample_rotation(system, azimuth_grid, options)
    n = result.numeric
    return FigureData(
        title="Sample-rotation scan - 2omega SHG vs azimuth",
        x_label="sample azimuth (deg)",
        y_label="SHG intensity (arb.)",
        x=_real_list(n["sample_azimuth_deg"]),
        series={
            "reflected parallel": _real_list(n["reflected_parallel_intensity"]),
            "reflected perpendicular": _real_list(n["reflected_perpendicular_intensity"]),
            "transmitted parallel": _real_list(n["transmitted_parallel_intensity"]),
            "transmitted perpendicular": _real_list(n["transmitted_perpendicular_intensity"]),
        },
        validation_status=result.validation.status,
        note="Single-layer azimuth scan live-Wolfram validated (ext1); multilayer azimuth (ml4) is an open export item.",
        meta={"kind": result.kind},
    )


def fresnel_figure_data(system, theta_grid=None, options=None) -> FigureData:
    """Fresnel coefficient magnitudes |rp|,|rs|,|tp|,|ts| vs incidence angle.

    For a ``MultilayerSystem`` the plot defaults to the SHAARP.ml multilayer
    ``listFresnel`` path (``workflow="gui_multilayer"`` /
    ``transmitted_wave_policy="shaarp_ml_selected"``) -- the exact Rp/Rs/Tp/Ts
    coefficients SHAARP.ml plots, and the path validated against Mathematica's
    ``listFresnel`` values to 1e-10 (see
    ``tests/test_fresnel_sweep_python_comparison.py`` and
    ``tests/test_figuredata.py::FresnelPlotMathematicaValidationTests``). A bare
    ``Material`` falls back to the single-interface workflow. Pass ``options``
    to override the routing explicitly."""
    from .config import MultilayerSystem  # local import: keep module load light

    multilayer = isinstance(system, MultilayerSystem)
    if options is None and multilayer:
        options = {"workflow": "gui_multilayer", "transmitted_wave_policy": "shaarp_ml_selected"}
    result = run_fresnel_sweep(system, theta_grid, options)
    n = result.numeric

    def _mag(values):
        arr = np.asarray(values, dtype=complex)
        return [float(abs(v)) for v in arr.ravel()]

    note = (
        "Multilayer listFresnel (shaarp_ml_selected) -- Mathematica-validated to 1e-10."
        if multilayer
        else "Single-interface Fresnel (bare Material): staged python, see facade validation status."
    )
    return FigureData(
        title="Fresnel coefficients vs incidence angle",
        x_label="incidence angle theta (deg)",
        y_label="coefficient magnitude",
        x=_real_list(n["theta_deg"]),
        series={"|rp|": _mag(n["rp"]), "|rs|": _mag(n["rs"]), "|tp|": _mag(n["tp"]), "|ts|": _mag(n["ts"])},
        validation_status=result.validation.status,
        note=note,
        meta={"kind": result.kind},
    )


@dataclass(frozen=True)
class StackSchematic:
    """Plot-ready layer-stack schematic: per-layer arrays a GUI/notebook can draw
    as a stacked bar / depth diagram. Semi-infinite media (no thickness) are
    flagged and given a nominal display thickness so they render."""

    layer_index: list[int]
    names: list[str]
    materials: list[str]
    thickness_um: list[float]
    shg_active: list[bool]
    is_semi_infinite: list[bool]
    cumulative_depth_um: list[float]
    nominal_semi_infinite_um: float

    def to_matplotlib(self, ax=None):
        import matplotlib.pyplot as plt  # noqa: PLC0415 (lazy)

        if ax is None:
            _, ax = plt.subplots()
        bottom = 0.0
        for i, t in enumerate(self.thickness_um):
            ax.bar(0, t, bottom=bottom, width=0.6,
                   color=("#c62828" if self.shg_active[i] else "#90a4ae"),
                   edgecolor="#263238")
            ax.text(0, bottom + t / 2, f"{self.names[i]} ({self.materials[i]})",
                    ha="center", va="center", fontsize=7)
            bottom += t
        ax.set_ylabel("depth (um; semi-infinite media shown at nominal thickness)")
        ax.set_xticks([])
        ax.set_title("Layer stack schematic")
        return ax


def stack_schematic_data(system, nominal_semi_infinite_um: float = 0.25) -> StackSchematic:
    """Return plottable per-layer stack data from a MultilayerSystem.

    Pure/headless: reads only layer name/material/thickness/shg_active. Top and
    substrate are semi-infinite (thickness_um is None) -> flagged and shown at a
    nominal thickness so a schematic renders without misrepresenting them."""
    layers = list(system.layers)
    idx, names, mats, thick, active, semi, cum = [], [], [], [], [], [], []
    running = 0.0
    for i, layer in enumerate(layers):
        t = layer.thickness_um
        is_semi = t is None
        disp_t = float(nominal_semi_infinite_um if is_semi else t)
        idx.append(i)
        names.append(str(layer.name))
        mats.append(str(layer.material.name))
        thick.append(disp_t)
        active.append(bool(layer.shg_active))
        semi.append(is_semi)
        running += disp_t
        cum.append(running)
    return StackSchematic(
        layer_index=idx,
        names=names,
        materials=mats,
        thickness_um=thick,
        shg_active=active,
        is_semi_infinite=semi,
        cumulative_depth_um=cum,
        nominal_semi_infinite_um=float(nominal_semi_infinite_um),
    )


@dataclass(frozen=True)
class LayerOrientation:
    """Per-layer 3D crystal-orientation geometry for a stack schematic.

    ``crystal_axes_lab`` is the 3x3 orientation (rotation) matrix -- its rows are
    the crystal frame axes expressed in the lab frame, the directions a 3D GUI
    would draw as the crystal axes for that layer. ``optical_class`` is the
    birefringence class from the (real part of the) lab permittivity eigenvalues,
    and for a uniaxial layer ``optic_axis_lab`` is the optic-axis unit vector in
    the lab frame (else ``None``)."""

    layer_index: int
    name: str
    material: str
    point_group: str
    shg_active: bool
    optical_class: str
    crystal_axes_lab: list[list[float]]
    optic_axis_lab: list[float] | None


@dataclass(frozen=True)
class StackOrientationSchematic:
    """Plot-ready per-layer crystal-orientation geometry (the 3D-schematic data
    behind the GUI's 2D/3D stack view; pure/headless, no render)."""

    layers: list[LayerOrientation]


def _orientation_matrix(material: Any) -> "np.ndarray":
    rot = material.orientation.rotation_matrix
    rot = rot() if callable(rot) else rot
    return np.asarray(rot, dtype=float)


def _optical_class_and_axis(eps_lab: Any) -> tuple[str, list[float] | None]:
    """Classify isotropic/uniaxial/biaxial from the lab permittivity and return
    the optic-axis unit vector (lab frame) for the uniaxial case."""
    e = np.real(np.asarray(eps_lab, dtype=complex))
    e = 0.5 * (e + e.T)  # symmetrize for a stable eigenbasis
    vals, vecs = np.linalg.eigh(e)
    scale = max(1.0, float(np.max(np.abs(vals))))
    rounded = np.round(vals / scale, 6)
    unique, counts = np.unique(rounded, return_counts=True)
    if len(unique) == 1:
        return "isotropic", None
    if len(unique) == 3:
        return "biaxial", None
    odd = unique[counts == 1]
    if len(odd) == 1:
        idx = int(np.argmin(np.abs(rounded - odd[0])))
        axis = vecs[:, idx]
        axis = axis / np.linalg.norm(axis)
        return "uniaxial", [round(float(x), 9) for x in axis]
    return "uniaxial", None


def stack_orientation_schematic_data(system) -> StackOrientationSchematic:
    """Per-layer 3D crystal-orientation geometry from a MultilayerSystem.

    Pure/headless: reads each layer's orientation matrix, point group, eps, and
    SHG-active flag. Composes with :func:`stack_schematic_data` (depth) to give a
    complete 2D/3D stack schematic data model a GUI can render."""
    out: list[LayerOrientation] = []
    for i, layer in enumerate(system.layers):
        mat = layer.material
        axes = _orientation_matrix(mat)
        eps_lab = mat.eps_w() if hasattr(mat, "eps_w") else mat.epsilon_omega
        optical_class, optic_axis = _optical_class_and_axis(eps_lab)
        out.append(
            LayerOrientation(
                layer_index=i,
                name=str(layer.name),
                material=str(mat.name),
                point_group=str(mat.structure.point_group),
                shg_active=bool(layer.shg_active),
                optical_class=optical_class,
                crystal_axes_lab=[[round(float(x), 9) for x in row] for row in axes],
                optic_axis_lab=optic_axis,
            )
        )
    return StackOrientationSchematic(layers=out)


__all__ = [
    "FigureData",
    "LayerOrientation",
    "StackOrientationSchematic",
    "stack_orientation_schematic_data",
    "StackSchematic",
    "maker_figure_data",
    "sample_rotation_figure_data",
    "fresnel_figure_data",
    "stack_schematic_data",
]
