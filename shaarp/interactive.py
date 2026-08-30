from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .api import (
    export_result,
    run_fresnel_sweep,
    run_maker_fringes,
    run_ml_numeric,
    run_ml_partial_analytical,
    run_sample_rotation,
    run_si_full_analytical,
    run_si_numeric,
)
from .config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry
from . import presets


def default_interactive_system() -> MultilayerSystem:
    """Return a small default system intended to run quickly in notebooks."""

    return MultilayerSystem(
        wavelength_um=1.064,
        polarimetry=Polarimetry(theta_deg=20.0, phi_deg=0.0, psi_deg=0.0, ellipticity_deg=0.0),
        layers=[
            Layer("air in", presets.air(), shg_active=False),
            Layer("nonlinear film", default_interactive_material(), thickness_um=0.2, shg_active=True),
            Layer("air out", presets.air(), shg_active=False),
        ],
    )


def default_interactive_material() -> Material:
    """Return a nondegenerate anisotropic SI material for quick notebook runs."""

    return Material(
        name="si-compat-demo",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation.from_cubic_miller_surface((1, 2, 3)),
        epsilon_omega=np.diag([1.58**2, 1.76**2, 2.01**2]),
        epsilon_2omega=np.diag([2.65 + 0.05j, 3.05 + 0.04j, 3.55 + 0.03j]),
        d_voigt_pm_v=np.array(
            [
                [0.35 + 0.08j, -0.12j, 0.5, 0.11, 0.04j, -0.27],
                [0.18j, 0.42, -0.09 + 0.04j, 0.31j, 0.63, -0.22],
                [-0.08, 0.17j, 0.41 + 0.09j, -0.34, 0.48j, 0.21],
            ],
            dtype=complex,
        ),
    )


def compute_polarimetry_result(mode: str, *, point_group: str, theta_deg: float, simplify: bool = False):
    """Pure (no-widget) compute for the GUI's validated SHG-polarimetry modes.

    Returns the same ``SHAARPResult`` the rest of the panel displays. ``mode`` is
    ``"si_polarimetry"`` (the SHAARP.si FULL-analytical reflected polarimetry closed form, symbolic
    in the input polarization phi / analyzer / azimuth / theta / indices / d) or ``"ml_polarimetry"``
    (the SHAARP.ml PARTIAL-analytical film polarimetry, symbolic in phi, d, h). These route the
    VALIDATED ``workflow="polarimetry"`` facades -- not the symbolic scaffolds the older
    ``si_analytical`` / ``ml_partial_analytical`` modes use. Factored out so it is unit-testable
    without ipywidgets/IPython.
    """

    if mode == "si_polarimetry":
        return run_si_full_analytical(
            {"point_group": point_group, "incident_theta_rad": float(np.deg2rad(theta_deg))},
            {"workflow": "polarimetry", "simplify": simplify},
        )
    if mode == "ml_polarimetry":
        return run_ml_partial_analytical(
            {"point_group": point_group}, {"workflow": "polarimetry"}
        )
    raise ValueError(f"Unknown polarimetry mode: {mode!r}")


def compute_dextraction_demo() -> dict[str, Any]:
    """Pure (no-widget) d-extraction round-trip for the GUI's 'd-extraction (demo)' mode.

    Simulates a reflected SHG polarimetry scan of an illustrative uniaxial crystal with a known
    six-component d-tensor (the exact validated configuration of
    ``tests/test_polarimetry_d_extraction_api.py``), recovers d with the public
    :func:`shaarp.extract_si_d_voigt`, and reports the recovery quality. Demonstrates the package's
    INVERSE capability (the closed form fulfilling its purpose: fit a scan -> get d) end to end.
    """

    import math

    from .polarimetry_extraction import extract_si_d_voigt
    from .shg import solve_single_interface_shg
    from .tensors import rotate_d_voigt_crystal_to_lab

    eps_w = [2.2**2, 2.2**2, 2.5**2]
    eps_2w = [2.4**2, 2.4**2, 2.7**2]
    positions = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
    true = np.array([0.30, 0.50, 0.60, 0.80, 0.40, 0.20])
    d0 = np.zeros((3, 6))
    for (m, ell), v in zip(positions, true):
        d0[m, ell] = v

    def _rz(az):
        c, s = math.cos(az), math.sin(az)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def measure(theta, az, phi):
        d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d0, _rz(az)), dtype=complex))
        r = solve_single_interface_shg(
            np.diag(eps_w).astype(complex), np.diag(eps_2w).astype(complex), d_rot,
            incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
            incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
        )
        c = np.asarray(r.coefficients)
        return complex(c[0]), complex(c[1])

    res = extract_si_d_voigt(
        eps_omega_principal=eps_w, eps_2omega_principal=eps_2w, d_positions=positions,
        geometries=[(0.35, 0.0), (0.35, 0.6), (0.35, 1.2), (0.75, 0.0), (0.75, 0.6), (0.75, 1.2)],
        phi_values=list(np.linspace(0.25, math.pi - 0.25, 6)), measure=measure,
        method="field", observable="reflected", basis="numeric", mu=1, eps0=1,
    )
    return {
        "kind": "d_extraction_demo",
        "identifiable": bool(res.identifiable),
        "rank": f"{res.rank}/{res.n_components}",
        "relative_residual": float(res.relative_residual),
        "max_abs_error_vs_true": float(np.max(np.abs(np.real(res.values) - true))),
        "recovered_d": {f"d[{m},{ell}]": round(float(np.real(v)), 4) for (m, ell), v in zip(positions, res.values)},
        "true_d": {f"d[{m},{ell}]": float(t) for (m, ell), t in zip(positions, true)},
    }


def compute_polarimetry_curve(
    *,
    theta_deg: float = 45.0,
    sample_azimuth_deg: float = 0.0,
    n_omega: float = 3.4,
    n_2omega: float = 3.7,
    n_phi: int = 181,
) -> dict[str, Any]:
    """Numeric I_s(phi), I_p(phi) over a full input-polarization sweep, evaluated FROM the
    SHAARP.si full-analytical reflected-SHG polarimetry CLOSED FORM for an illustrative
    GaAs(111)-like cubic -43m crystal (one independent constant d14, set to 1). Pure/headless:
    returns plain numpy arrays (phi_deg, intensity_s, intensity_p) so the polarimetry curve is
    unit-testable without rendering; the GUI 'SI polarimetry (plot)' mode renders it. Mirrors the
    validated examples/gaas111_shg_polarimetry.py.
    """

    import math

    import sympy as sp

    from .config import CrystalOrientation
    from .symbolic import solve_si_shg_full_analytical_symbolic
    from .tensors import rotate_d_voigt_crystal_to_lab

    d14 = sp.Symbol("d14")
    phi = sp.Symbol("phi", real=True)
    d = np.zeros((3, 6))
    d[0, 3] = d[1, 4] = d[2, 5] = 1.0  # -43m: d14 = d25 = d36
    ori = CrystalOrientation.from_cubic_miller_surface(
        np.array([1.0, 1.0, 1.0]), in_plane_hint=np.array([1.0, -1.0, 0.0])
    )
    geom = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, np.asarray(ori.rotation_matrix(), dtype=float)), dtype=complex))
    d_lab = sp.Matrix(geom.tolist()) * d14
    az = math.radians(sample_azimuth_deg)
    sol = solve_si_shg_full_analytical_symbolic(
        eps_x_omega=n_omega**2, eps_y_omega=n_omega**2, eps_z_omega=n_omega**2,
        eps_x_2omega=n_2omega**2, eps_y_2omega=n_2omega**2, eps_z_2omega=n_2omega**2,
        d_voigt_lab=d_lab, incident_theta_rad=math.radians(theta_deg), phi_symbol=phi,
        sample_azimuth_symbol=(sp.Float(az) if az else None), omega=1, mu=1, eps0=1, simplify=False,
    )
    e_s = sp.lambdify(phi, sol.reflected_s.subs(d14, 1.0), "numpy")
    e_p = sp.lambdify(phi, sol.reflected_p.subs(d14, 1.0), "numpy")
    phi_deg = np.linspace(0.0, 360.0, int(n_phi))
    rad = np.radians(phi_deg)
    i_s = np.broadcast_to(np.abs(e_s(rad)) ** 2, rad.shape).astype(float)
    i_p = np.broadcast_to(np.abs(e_p(rad)) ** 2, rad.shape).astype(float)
    return {"phi_deg": phi_deg, "intensity_s": i_s, "intensity_p": i_p,
            "theta_deg": float(theta_deg), "sample_azimuth_deg": float(sample_azimuth_deg)}


def build_polarimetry_figure(curve: dict[str, Any] | None = None, **kwargs):
    """Build (do not display) a polar matplotlib Figure of the polarimetry curve from
    :func:`compute_polarimetry_curve`. Returned so it is testable under the Agg backend; the GUI
    displays it. Extra kwargs are forwarded to ``compute_polarimetry_curve`` when ``curve`` is None."""

    import matplotlib.pyplot as plt

    data = curve if curve is not None else compute_polarimetry_curve(**kwargs)
    rad = np.radians(np.asarray(data["phi_deg"]))
    fig, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw={"projection": "polar"})
    ax.plot(rad, data["intensity_p"], color="navy", lw=2, label=r"$I_p^{2\omega}(\varphi)$")
    ax.plot(rad, data["intensity_s"], color="darkorange", lw=2, label=r"$I_s^{2\omega}(\varphi)$")
    ax.set_title(
        f"SI reflected SHG polarimetry (closed form)\n"
        f"GaAs(111)-like -43m, $\\theta_i$={data.get('theta_deg', 45.0):.0f}°",
        fontsize=10,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), fontsize=9)
    fig.tight_layout()
    return fig


def build_dextraction_figure(demo: dict[str, Any] | None = None):
    """Build (do not display) a matplotlib Figure comparing the RECOVERED vs TRUE d-tensor
    components from :func:`compute_dextraction_demo` -- a visual that the inverse works. Returned
    (not shown) so it is testable under the Agg backend; the GUI 'd-extraction (plot)' mode renders
    it. Mirrors the recovered-vs-true bars of the validated examples/d_extraction_demo.py."""

    import matplotlib.pyplot as plt

    d = demo if demo is not None else compute_dextraction_demo()
    labels = list(d["true_d"].keys())
    true_vals = [d["true_d"][k] for k in labels]
    rec_vals = [d["recovered_d"][k] for k in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x - 0.2, true_vals, width=0.4, color="navy", label="true $d$")
    ax.bar(x + 0.2, rec_vals, width=0.4, color="darkorange", alpha=0.85, label="recovered $d$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("$d$ (arb. units)")
    ax.set_title(
        f"d-extraction: recovered vs. true (rank {d['rank']}, rel. residual {d['relative_residual']:.1e})",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def make_interactive_session(system: MultilayerSystem | None = None) -> Any:
    """Build an ipywidgets SHAARP.py control panel for notebook use."""

    try:
        import ipywidgets as widgets
        from IPython.display import JSON, clear_output, display
    except ImportError as exc:
        raise ImportError("The interactive SHAARP session requires ipywidgets and IPython.") from exc

    base_system = system or default_interactive_system()
    base_material = default_interactive_material()
    mode = widgets.Dropdown(
        options=[
            ("SI compat", "si_compat"),
            ("ML numeric", "ml_numeric"),
            ("Fresnel sweep", "fresnel_sweep"),
            ("Maker fringes", "maker_fringes"),
            ("Sample rotation", "sample_rotation"),
            ("SI analytical", "si_analytical"),
            ("ML partial analytical", "ml_partial_analytical"),
            ("SI polarimetry", "si_polarimetry"),
            ("ML polarimetry", "ml_polarimetry"),
            ("SI polarimetry (plot)", "si_polarimetry_plot"),
            ("d-extraction (demo)", "d_extraction"),
            ("d-extraction (plot)", "d_extraction_plot"),
        ],
        description="Mode",
    )
    theta = widgets.FloatSlider(value=float(np.asarray(base_system.polarimetry.theta_deg)), min=-60, max=80, step=1, description="theta")
    phi = widgets.FloatSlider(value=float(np.asarray(base_system.polarimetry.phi_deg)), min=0, max=360, step=5, description="phi")
    psi = widgets.FloatSlider(value=float(np.asarray(base_system.polarimetry.psi_deg)), min=0, max=180, step=5, description="psi")
    ellipticity = widgets.FloatSlider(
        value=float(np.asarray(base_system.polarimetry.ellipticity_deg)),
        min=-90,
        max=90,
        step=5,
        description="ellipticity",
    )
    theta_min = widgets.FloatText(value=-20.0, description="theta min")
    theta_max = widgets.FloatText(value=20.0, description="theta max")
    theta_step = widgets.FloatText(value=10.0, description="theta step")
    azimuth_min = widgets.FloatText(value=0.0, description="az min")
    azimuth_max = widgets.FloatText(value=90.0, description="az max")
    azimuth_step = widgets.FloatText(value=30.0, description="az step")
    transmitted_policy = widgets.ToggleButtons(
        options=[("SHAARP selected", "shaarp_ml_selected"), ("Physical sum", "physical_sum")],
        description="T policy",
    )
    inhom_policy = widgets.Dropdown(
        options=[
            ("solve", "solve"),
            ("minimum norm if ill-conditioned", "minimum_norm_if_ill_conditioned"),
            ("minimum norm", "minimum_norm"),
        ],
        description="inhom",
    )
    validation_refs = widgets.Checkbox(value=True, description="Refs")
    run_button = widgets.Button(description="Run", icon="play")
    export_button = widgets.Button(description="Export JSON", icon="save")
    output = widgets.Output()
    state: dict[str, Any] = {"last_result": None}

    def configured_system() -> MultilayerSystem:
        pol = replace(
            base_system.polarimetry,
            theta_deg=float(theta.value),
            phi_deg=float(phi.value),
            psi_deg=float(psi.value),
            ellipticity_deg=float(ellipticity.value),
        )
        return replace(base_system, polarimetry=pol)

    def angle_grid() -> np.ndarray:
        return _closed_grid(float(theta_min.value), float(theta_max.value), float(theta_step.value))

    def azimuth_grid() -> np.ndarray:
        return _closed_grid(float(azimuth_min.value), float(azimuth_max.value), float(azimuth_step.value))

    def run_clicked(_):
        with output:
            clear_output(wait=True)
            try:
                current = configured_system()
                if mode.value == "si_compat":
                    result = run_si_numeric(
                        base_material,
                        {
                            "workflow": "shaarp_si_compat",
                            "polarimetry": Polarimetry(theta_deg=float(theta.value)),
                            "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
                        },
                    )
                elif mode.value == "ml_numeric":
                    result = run_ml_numeric(
                        current,
                        {
                            "inhomogeneous_solution_policy": inhom_policy.value,
                            "include_validation_summary": bool(validation_refs.value),
                        },
                    )
                elif mode.value == "fresnel_sweep":
                    result = run_fresnel_sweep(
                        current,
                        angle_grid(),
                        {
                            "workflow": "gui_multilayer",
                            "transmitted_wave_policy": transmitted_policy.value,
                            "include_validation_summary": bool(validation_refs.value),
                        },
                    )
                elif mode.value == "maker_fringes":
                    result = run_maker_fringes(
                        current,
                        angle_grid(),
                        {
                            "transmitted_wave_policy": transmitted_policy.value,
                            "inhomogeneous_solution_policy": inhom_policy.value,
                            "include_validation_summary": bool(validation_refs.value),
                        },
                    )
                elif mode.value == "sample_rotation":
                    result = run_sample_rotation(
                        current,
                        azimuth_grid(),
                        {
                            "inhomogeneous_solution_policy": inhom_policy.value,
                            "include_validation_summary": bool(validation_refs.value),
                        },
                    )
                elif mode.value == "si_analytical":
                    result = run_si_full_analytical(
                        {
                            "point_group": base_material.structure.point_group,
                            "incident_theta_rad": np.deg2rad(float(theta.value)),
                        },
                        {"workflow": "isotropic_symbolic_scaffold", "simplify": False},
                    )
                elif mode.value == "ml_partial_analytical":
                    result = run_ml_partial_analytical(
                        {"point_group": current.layers[1].material.structure.point_group},
                        {"workflow": "pnl_symbolic_scaffold", "simplify": True},
                    )
                elif mode.value in ("si_polarimetry", "ml_polarimetry"):
                    # the VALIDATED polarimetry closed forms (not the scaffolds above)
                    result = compute_polarimetry_result(
                        mode.value,
                        point_group=base_material.structure.point_group,
                        theta_deg=float(theta.value),
                    )
                elif mode.value == "si_polarimetry_plot":
                    # visualize the closed-form polarimetry I_s(phi)/I_p(phi) at the chosen incidence
                    import matplotlib.pyplot as plt

                    state["last_result"] = None
                    fig = build_polarimetry_figure(theta_deg=float(theta.value))
                    display(fig)
                    plt.close(fig)
                    return
                elif mode.value == "d_extraction":
                    # inverse capability: simulate a scan and recover d (own dict summary)
                    state["last_result"] = None
                    display(JSON(compute_dextraction_demo(), expanded=True))
                    return
                elif mode.value == "d_extraction_plot":
                    # visualize the inverse: recovered vs true d bars
                    import matplotlib.pyplot as plt

                    state["last_result"] = None
                    fig = build_dextraction_figure()
                    display(fig)
                    plt.close(fig)
                    return
                else:
                    raise ValueError(f"Unsupported mode: {mode.value}")
                state["last_result"] = result
                display(JSON(_result_summary(result), expanded=False))
            except Exception as exc:
                state["last_result"] = None
                print(f"{type(exc).__name__}: {exc}")

    def export_clicked(_):
        with output:
            if state["last_result"] is None:
                print("Run a simulation before exporting.")
                return
            path = export_result(state["last_result"], "outputs/interactive_last_result.json")
            print(f"Exported {path}")

    run_button.on_click(run_clicked)
    export_button.on_click(export_clicked)
    controls = widgets.VBox(
        [
            widgets.HBox([mode, run_button, export_button]),
            widgets.HBox([theta, phi, psi, ellipticity]),
            widgets.HBox([theta_min, theta_max, theta_step]),
            widgets.HBox([azimuth_min, azimuth_max, azimuth_step]),
            widgets.HBox([transmitted_policy, inhom_policy, validation_refs]),
            output,
        ]
    )
    return controls


def _closed_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    if stop < start:
        raise ValueError("Grid stop must be greater than or equal to start.")
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return start + step * np.arange(count, dtype=float)


def _result_summary(result) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "validation_status": result.validation.status,
        "validation_notes": list(result.validation.notes),
        "conventions": {
            "field": result.conventions.field_convention,
            "boundary": result.conventions.tangential_boundary,
            "intensity": result.conventions.intensity_normalization,
            "phase_matching": result.conventions.phase_matching_policy,
            "maker_transmitted": result.conventions.maker_transmitted_policy,
        },
        "numeric_shapes": {key: list(np.asarray(value).shape) for key, value in result.numeric.items()},
        "stage_keys": list(result.stages),
    }
