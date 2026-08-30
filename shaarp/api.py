from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import Material, MultilayerSystem, Polarimetry
from .io import to_jsonable
from .linear import solve_fresnel_coefficient_sweep
from .multilayer import multilayer_shg
from .multilayer_shg_boundary import (
    solve_multilayer_maker_fringes_sweep,
    solve_multilayer_shg_from_system_polarimetry,
    solve_multilayer_shg_polarimetry_sweep,
    solve_multilayer_shg_sample_azimuth_sweep,
)
from .single_interface import single_interface_intensity
from .si_compat import solve_shaarp_si_reflected_shg
from .tensors import rotate_d_voigt_crystal_to_lab, rotate_rank2_crystal_to_lab


@dataclass(frozen=True)
class PhysicsConventions:
    field_convention: str = "exp(i(omega*t - k.r))"
    tangential_boundary: str = "continuity of tangential E and H"
    mode_identity: str = "fast/slow branches are numeric eigenmodes; ordinary/extraordinary labels are only assigned where physically defined"
    intensity_normalization: str = "reported intensities are analyzer field-norm squared unless a result explicitly states Poynting-flux power"
    phase_matching_policy: str = "solve by default; minimum_norm modes are explicit options for singular or ill-conditioned inhomogeneous solves"
    maker_transmitted_policy: str = "shaarp_ml_selected by default; physical_sum is available where exposed by the solver"


@dataclass(frozen=True)
class ValidationStatus:
    status: str
    mathematica_reference: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_reduced_model(self) -> bool:
        """True if this result came from a REDUCED (effective-index / source-sum)
        path rather than the full validated SHAARP solver. Lets callers/notebooks
        query the reduced label programmatically instead of string-matching the
        status text. (The reduced paths in shaarp.multilayer / shaarp.single_interface
        also emit RuntimeWarnings and set model='reduced_effective_index*'.)"""
        return "reduced" in self.status.lower()

    @property
    def is_mathematica_validated(self) -> bool:
        """True if the status asserts a live Mathematica/Wolfram value match and is
        NOT a reduced/unvalidated/mismatch state. Conservative: a positive 'match'
        or 'validated' word with no disqualifier."""
        s = self.status.lower()
        if any(bad in s for bad in ("reduced", "not_mathematica", "not_full", "mismatch", "unresolved", "regression")):
            return False
        return any(good in s for good in ("match", "validated", "agreement"))


@dataclass(frozen=True)
class SHAARPResult:
    kind: str
    numeric: dict[str, Any]
    stages: dict[str, Any] = field(default_factory=dict)
    conventions: PhysicsConventions = field(default_factory=PhysicsConventions)
    validation: ValidationStatus = field(
        default_factory=lambda: ValidationStatus(
            "not_mathematica_validated",
            notes=("This facade result needs a Mathematica reference comparison before equivalence can be claimed.",),
        )
    )
    warnings: tuple[str, ...] = ()
    raw: Any = None

    @property
    def is_reduced_model(self) -> bool:
        """Convenience: did this result come from a reduced (non-validated) path?
        True if the validation says reduced OR any reduced-model RuntimeWarning was
        attached."""
        return self.validation.is_reduced_model or any("reduced model" in w.lower() for w in self.warnings)


@dataclass(frozen=True)
class MathematicaComparisonResult:
    case_id: str
    key: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    shape: tuple[int, ...]


def run_si_numeric(case: Material, options: dict[str, Any] | None = None) -> SHAARPResult:
    """Run a single-interface reflected-SHG calculation (the SHAARP.si numeric path).

    This is the facade the desktop GUI's SHAARP.si *SHG Simulation* calls.

    Args:
        case: The interface :class:`~shaarp.Material` (crystal with its dielectric and ``d`` tensors
            and orientation).
        options: Optional settings dictionary. Recognized keys:

            - ``workflow``: ``"reduced"`` (default) or ``"shaarp_si_compat"`` -- the latter is the
              validated, Mathematica-faithful single-interface workflow (including the
              ``normal_incidence_2omega_branch_policy`` that handles exactly normal incidence).
            - ``polarimetry``: a :class:`~shaarp.Polarimetry` giving the incidence angle(s) and
              input/analyzer polarization.
            - ``incident_index``: refractive index of the incidence medium (default ``1.0``).
            - ``include_validation_summary``: if true, attach the Mathematica stage-comparison summary.

    Returns:
        SHAARPResult: the reflected-SHG numeric data plus stage/validation metadata.

    Example:
        >>> from shaarp import run_si_numeric, Polarimetry, presets
        >>> res = run_si_numeric(
        ... presets.linbo3_1120_xcut(),
        ... {"workflow": "shaarp_si_compat", "polarimetry": Polarimetry(theta_deg=45)},
        ... )
    """

    options = dict(options or {})
    workflow = options.pop("workflow", "reduced")
    if workflow == "shaarp_si_compat":
        return _run_si_numeric_shaarp_si_compat(case, options)
    if workflow != "reduced":
        raise ValueError("run_si_numeric workflow must be 'reduced' or 'shaarp_si_compat'.")
    polarimetry = options.pop("polarimetry", None)
    incident_index = options.pop("incident_index", 1.0)
    include_validation_summary = bool(options.pop("include_validation_summary", False))
    if options:
        raise ValueError(f"Unsupported run_si_numeric options: {sorted(options)}")

    raw = single_interface_intensity(case, polarimetry=polarimetry, incident_index=incident_index)
    stages = {"model": raw.model}
    if include_validation_summary:
        stages["mathematica_stage_summary"] = _safe_validation_artifacts(_load_si_stage_comparison_summary)
    return SHAARPResult(
        kind="si_numeric",
        numeric={
            "phi_deg": raw.phi_deg,
            "psi_deg": raw.psi_deg,
            "e_2omega": raw.e_2omega,
            "intensity": raw.intensity,
        },
        stages=stages,
        validation=ValidationStatus(
            "reduced_model_not_full_shaarp_si",
            notes=tuple(raw.warnings)
            or ("Legacy reduced SI entry point; use staged Mathematica references before claiming SHAARP.si equivalence.",),
        ),
        warnings=tuple(raw.warnings),
        raw=raw,
    )


def _run_si_numeric_shaarp_si_compat(case: Material, options: dict[str, Any]) -> SHAARPResult:
    polarimetry = options.pop("polarimetry", None) or Polarimetry()
    incident_polarization = options.pop("incident_polarization", "s")
    branch_policy = options.pop("normal_incidence_2omega_branch_policy", "python_exact_zero")
    mu = options.pop("mu", 1.0)
    eps0 = options.pop("eps0", 1.0)
    if options:
        raise ValueError(f"Unsupported run_si_numeric shaarp_si_compat options: {sorted(options)}")
    theta = np.asarray(polarimetry.theta_deg, dtype=float)
    if theta.shape != ():
        raise ValueError("workflow='shaarp_si_compat' currently expects scalar Polarimetry.theta_deg.")
    # FRAME BRIDGE (closes the "compat rotated-material m_y violation").
    # Two conventions meet here, and the old code crossed them for any non-symmetric orientation:
    # * Material.orientation.rotation_matrix() is the PORT convention — ROWS = crystal Z axes in
    # lab — consumed by the upward-propagating (SHAARP.ml-frame) solvers as eps_lab = M.T eps M.
    # * solve_shaarp_si_reflected_shg is the SHAARP.si-V1.04-faithful stack: internally it
    # propagates DOWNWARD (u = (sin t, 0, -cos t)) and its ``orientation_matrix`` input is
    # V1.04's ``a`` matrix — COLUMNS = crystal Z axes in lab (eps_lab = a eps a.T). Its 36-case
    # stage validation (incl. 5 general tilted orientations, 30/36 cases) fed live-Mathematica
    # ``a`` matrices, so THAT is the validated input convention.
    # Passing rows-M directly (old code) consumed eps at the TRANSPOSED orientation while d was
    # rotated with rows-M — eps and d then described two DIFFERENT crystals, which is what broke
    # the m_y mirror on the tilted TaAs (112) (nonzero s-out for s-in; the generic solver gives
    # ~1e-36). The exact intensity-level bridge from the port's upward frame to the si downward
    # frame is the 180-deg lab azimuth (global z-mirror ≡ 180° azimuth up to a global d sign,
    # unobservable in |E|²): rows_si = M · diag(-1,-1,1), a = rows_si.T, and d rotated with the
    # SAME rows_si. For si case-study materials this hands the V1.04-faithful stack exactly the
    # verbatim V1.04 orientation state. Fields are frame-representations (per-channel magnitudes
    # and intensities are frame-invariant); fence: tests/test_api.py::TestCompatFrameBridge.
    orientation_rows = np.asarray(case.orientation.rotation_matrix(), dtype=float)
    rows_si_frame = orientation_rows @ np.diag([-1.0, -1.0, 1.0])
    d_lab = rotate_d_voigt_crystal_to_lab(case.d_voigt(), rows_si_frame)
    raw = solve_shaarp_si_reflected_shg(
        epsilon_omega_crystal=case.eps_w(),
        epsilon_2omega_crystal=case.eps_2w(),
        orientation_matrix=rows_si_frame.T,
        d_voigt_lab=d_lab,
        theta_in_rad=float(np.deg2rad(theta.item())),
        incident_polarization=incident_polarization,
        mu=mu,
        eps0=eps0,
        normal_incidence_2omega_branch_policy=branch_policy,
    )
    return SHAARPResult(
        kind="si_numeric_shaarp_si_compat",
        numeric={
            "reflected_er2w": raw.reflected_total.electric,
            "reflected_intensity": float(np.vdot(raw.reflected_total.electric, raw.reflected_total.electric).real),
            "boundary_residual_norm": float(np.linalg.norm(raw.boundary_residual)),
        },
        stages={
            "linear_omega": raw.linear_omega,
            "sources": raw.sources,
            "inhomogeneous": raw.inhomogeneous,
            "homogeneous_2omega": raw.homogeneous_2omega,
            "normal_incidence_2omega_branch_policy": branch_policy,
        },
        validation=ValidationStatus(
            "mathematica_stage_validated_for_exported_36_case_si_reflected_er2w",
            "benchmarks/mathematica_reference/shaarp_si_two_omega_boundary_stage_reference_v1.json",
            notes=(
                "Opt-in SHAARP.si compatibility workflow; 36 exported reflected ER2w stage cases match with shaarp_reference_like branch policy.",
                "This is stage-reference validation, not a full notebook execution claim.",
                "SCOPE: the 36-case reference includes 30 tilted+anisotropic "
                "oblique cases, so the rotated-material scope IS validated — as V1.04-numeric "
                "faithfulness. The V1.04 numeric five-wave branch itself, however, builds its two "
                "transmitted-omega wavevectors with the branch index magnitudes EXCHANGED for "
                "birefringent crystals at oblique incidence (the reference's own Vkt data satisfy "
                "kx_e*kx_o = sin^2(theta_i) instead of kx_e = kx_o = sin(theta_i); deviations reach "
                "~24% of k_par at 55 deg), so tangential phase matching is violated and channel "
                "amplitudes deviate from the textbook-exact generic solver at O(dn/n) (e.g. ~2.5% in "
                "|r_s| for x-cut LiNbO3, O(2x) SHG channels for strongly birefringent TaAs). The "
                "published-paper pipeline (closed-form polarimetry + fitting notebooks) does not use "
                "this branch and is unaffected. For physics use the generic/closed-form routes; this "
                "workflow is the V1.04-numeric-faithfulness reference.",
            ),
        ),
        raw=raw,
    )


def run_ml_numeric(case: MultilayerSystem, options: dict[str, Any] | None = None) -> SHAARPResult:
    """Run the multilayer SHG polarimetry calculation (the SHAARP.ml numeric path).

    The facade behind the desktop GUI's SHAARP.ml *SHG Simulation*.

    Args:
        case: the :class:`~shaarp.MultilayerSystem` -- an air / film(s) / substrate stack with
            per-layer materials, thicknesses, and SHG-active flags, plus its :class:`~shaarp.Polarimetry`.
        options: optional settings. Recognized keys include ``include_validation_summary`` (attach the
            Mathematica validation artifacts) and ``legacy_reduced`` (use the older reduced model).

    Returns:
        SHAARPResult: the multilayer reflected/transmitted SHG data with stage/validation metadata.

    Example:
        >>> from shaarp import run_ml_numeric, default_interactive_system
        >>> res = run_ml_numeric(default_interactive_system())
    """

    options = dict(options or {})
    legacy_reduced = bool(options.pop("legacy_reduced", False))
    include_validation_summary = bool(options.pop("include_validation_summary", False))
    if legacy_reduced:
        if options:
            raise ValueError(f"Unsupported legacy run_ml_numeric options: {sorted(options)}")
        raw = multilayer_shg(case)
        stages = {"layer_sources": raw.layer_sources, "model": raw.model}
        if include_validation_summary:
            stages["mathematica_validation_artifacts"] = _safe_validation_artifacts(_load_ml_numeric_validation_artifacts)
        return SHAARPResult(
            kind="ml_numeric_reduced",
            numeric={
                "phi_deg": raw.phi_deg,
                "reflected_intensity": raw.reflected_intensity,
                "transmitted_intensity": raw.transmitted_intensity,
            },
            stages=stages,
            validation=ValidationStatus("reduced_model_not_full_shaarp_ml", notes=tuple(raw.warnings)),
            warnings=tuple(raw.warnings),
            raw=raw,
        )

    # Multiple-reflection assumption for the POLAR-PLOT polarimetry (same validated mapping as the
    # Maker/SampleRotate paths: 0 = FMR, 1 = Jerphagnon-Kurtz, 2 = Herman-Hayden; JK/HH agreement
    # vs live SHAARP.ml pinned in tests/test_jkhh_*). Added (T1 sensitivity gate found
    # the GUI's SHG Simulation silently dropped the Assumptions choice).
    mrassumption = int(options.pop("mrassumption", 0))
    if mrassumption not in (0, 1, 2):
        raise ValueError("run_ml_numeric option mrassumption must be 0 (FMR), 1 (JK) or 2 (HH)")
    single_pass_omega = False
    single_pass_omega_writeback = True
    single_pass_2omega = False
    if mrassumption == 2:  # Herman-Hayden
        single_pass_omega = True
    elif mrassumption == 1:  # Jerphagnon-Kurtz
        single_pass_omega = True
        single_pass_omega_writeback = False
        single_pass_2omega = True
    raw = solve_multilayer_shg_from_system_polarimetry(
        case,
        condition_threshold=options.pop("condition_threshold", 1e12),
        inhomogeneous_source_policy=options.pop("inhomogeneous_source_policy", "all"),
        inhomogeneous_solution_policy=options.pop("inhomogeneous_solution_policy", "solve"),
        single_pass_omega=single_pass_omega,
        single_pass_omega_writeback=single_pass_omega_writeback,
        single_pass_2omega=single_pass_2omega,
    )
    if options:
        raise ValueError(f"Unsupported run_ml_numeric options: {sorted(options)}")
    amplitude, intensity = _reflected_analyzer_from_case(raw, case)
    stages = {
        "fundamental": raw.fundamental,
        "layer_sources": raw.layer_sources,
        "layer_inhomogeneous_fields": raw.layer_inhomogeneous_fields,
        "shg": raw.shg,
    }
    if include_validation_summary:
        stages["mathematica_validation_artifacts"] = _safe_validation_artifacts(_load_ml_numeric_validation_artifacts)
    return SHAARPResult(
        kind="ml_numeric",
        numeric={
            "reflected_analyzer_amplitude": amplitude,
            "reflected_intensity": intensity,
            "fundamental_coefficients": raw.fundamental.coefficients,
            "shg_coefficients": raw.shg.coefficients,
            "fundamental_residual_norm": float(np.linalg.norm(raw.fundamental.residual)),
            "shg_residual_norm": float(np.linalg.norm(raw.shg.residual)),
        },
        stages=stages,
        validation=ValidationStatus(
            "staged_python_not_fully_mathematica_validated",
            notes=("The multilayer solver is Mathematica-value-validated END-TO-END for the exported case families (Maker depth 4-9 layers, JK/HH, sample rotation, point-group d; see docs/validation.md); this per-run status stays conservative because an arbitrary user configuration outside those families is not individually validated.",),
        ),
        raw=raw,
    )


def run_fresnel_sweep(case: MultilayerSystem | Material, angle_grid: Any = None, options: dict[str, Any] | None = None) -> SHAARPResult:
    """Linear Fresnel reflection/transmission **power** coefficients vs incidence angle.

    The facade behind the desktop GUI's SHAARP.ml *Fresnel Coefficients*.

    Args:
        case: the :class:`~shaarp.MultilayerSystem` (or a single :class:`~shaarp.Material`) to sweep.
        angle_grid: incidence angles in degrees (array). Defaults to an internal grid.
        options: optional settings, e.g. ``workflow="gui_multilayer"`` for the GUI's multilayer path.

    Returns:
        SHAARPResult: arrays ``rp, rs, tp, ts`` (power coefficients) vs ``theta_deg`` in ``.numeric``.
        For a lossless stack ``R + T = 1`` per polarization, and ``p``/``s`` are degenerate at normal
        incidence.
    """

    options = dict(options or {})
    include_validation_summary = bool(options.pop("include_validation_summary", False))
    workflow = options.pop("workflow", "interface")
    if workflow == "gui_multilayer":
        transmitted_wave_policy = options.pop("transmitted_wave_policy", "shaarp_ml_selected")
        if options:
            raise ValueError(f"Unsupported run_fresnel_sweep gui_multilayer options: {sorted(options)}")
        if not isinstance(case, MultilayerSystem):
            raise TypeError("workflow='gui_multilayer' requires a MultilayerSystem.")
        return _run_gui_multilayer_fresnel_sweep(
            case,
            angle_grid,
            transmitted_wave_policy=transmitted_wave_policy,
            include_validation_summary=include_validation_summary,
        )
    if workflow != "interface":
        raise ValueError("run_fresnel_sweep workflow must be 'interface' or 'gui_multilayer'.")
    if isinstance(case, MultilayerSystem):
        case.validate()
        top = case.layers[0].material
        target = case.layers[1].material
        incident_index = complex(np.sqrt(np.trace(top.eps_w()) / 3))
        epsilon_lab = rotate_rank2_crystal_to_lab(target.eps_w(), target.orientation.rotation_matrix())
    elif isinstance(case, Material):
        incident_index = options.pop("incident_index", 1.0)
        epsilon_lab = rotate_rank2_crystal_to_lab(case.eps_w(), case.orientation.rotation_matrix())
    else:
        raise TypeError("case must be a Material or MultilayerSystem.")

    raw = solve_fresnel_coefficient_sweep(
        epsilon_lab,
        incident_index=incident_index,
        theta_deg=angle_grid,
        theta_step_deg=options.pop("theta_step_deg", 1.0),
        theta_min_deg=options.pop("theta_min_deg", 0.0),
        theta_max_deg=options.pop("theta_max_deg", 89.0),
        omega=options.pop("omega", 1.0),
    )
    if options:
        raise ValueError(f"Unsupported run_fresnel_sweep options: {sorted(options)}")
    stages = {"p_results": raw.p_results, "s_results": raw.s_results}
    if include_validation_summary:
        stages["mathematica_validation_artifacts"] = _safe_validation_artifacts(_load_fresnel_validation_artifacts)
    return SHAARPResult(
        kind="fresnel_sweep",
        numeric={"theta_deg": raw.theta_deg, "rp": raw.rp, "rs": raw.rs, "tp": raw.tp, "ts": raw.ts},
        stages=stages,
        validation=ValidationStatus(
            "staged_python_not_fully_mathematica_validated",
            notes=("This legacy helper computes power ratios from Poynting flux and is NOT the listFresnel-validated path; the Mathematica-listFresnel-validated GUI Fresnel sweep (selected-wave policy, ~1e-10) is the Fresnel Coefficients functionality of compute_ml_gui_result / the merged GUI.",),
        ),
        raw=raw,
    )


def run_maker_fringes(case: MultilayerSystem, angle_grid: Any = None, options: dict[str, Any] | None = None) -> SHAARPResult:
    """Run a SHAARP.ml-style Maker-fringe transmitted-SHG angle sweep.

    The facade behind the desktop GUI's SHAARP.ml *Maker Fringes*.

    Args:
        case: the :class:`~shaarp.MultilayerSystem` (thin-film stack).
        angle_grid: incidence-angle grid in degrees. Defaults to an internal dense grid.
        options: optional settings. Common keys: ``mrassumption`` (0 = Full multiple reflections,
            1 = Jerphagnon-Kurtz, 2 = Herman-Hayden), ``inhomogeneous_source_policy`` (the FMR
            forward/backward/standing sub-mode), and ``include_validation_summary``.

    Returns:
        SHAARPResult: ``parallel_intensity`` / ``perpendicular_intensity`` vs incidence angle.

    Note:
        Defaults to the SHAARP.ml unit normalization ``mu=1, eps0=1`` (overridable via
    ``options``) so the returned ``parallel_intensity`` / ``perpendicular_intensity``
    are the same ``listMFpara`` / ``listMFperp`` quantities the original Mathematica
    package reports (matched to ~1e-15 on the validated cases). Passing physical
    ``mu``/``eps0`` instead rescales the output and is no longer the Mathematica-
    validated convention.
    """

    options = dict(options or {})
    transmitted_wave_policy = options.pop("transmitted_wave_policy", "shaarp_ml_selected")
    include_validation_summary = bool(options.pop("include_validation_summary", False))
    raw = solve_multilayer_maker_fringes_sweep(
        case,
        theta_deg=angle_grid,
        theta_min_deg=options.pop("theta_min_deg", -45.0),
        theta_max_deg=options.pop("theta_max_deg", 45.0),
        theta_step_deg=options.pop("theta_step_deg", 1.0),
        # the SHAARP.ml Assumptions panel: 0 = Full (FMR), 1 = JK (forward-only), 2 = HH
        mrassumption=int(options.pop("mrassumption", 0)),
        condition_threshold=options.pop("condition_threshold", 1e12),
        inhomogeneous_source_policy=options.pop("inhomogeneous_source_policy", "forward_only"),
        inhomogeneous_solution_policy=options.pop("inhomogeneous_solution_policy", "solve"),
        transmitted_wave_policy=transmitted_wave_policy,
        mu=options.pop("mu", 1.0),
        eps0=options.pop("eps0", 1.0),
    )
    if options:
        raise ValueError(f"Unsupported run_maker_fringes options: {sorted(options)}")
    stages = {"results": raw.results}
    if include_validation_summary:
        stages["mathematica_validation_artifacts"] = _safe_validation_artifacts(_load_maker_validation_artifacts)
    status = (
        "maker_outputs_nonsingular_mathematica_validated_with_phase_matching_diagnostic_and_transmitted_sum_caveat"
        if transmitted_wave_policy == "shaarp_ml_selected"
        else "physically_motivated_not_mathematica_gui_default"
    )
    return SHAARPResult(
        kind="maker_fringes",
        numeric={
            "theta_deg": raw.theta_deg,
            "parallel_amplitude": raw.parallel_amplitude,
            "perpendicular_amplitude": raw.perpendicular_amplitude,
            "parallel_intensity": raw.parallel_intensity,
            "perpendicular_intensity": raw.perpendicular_intensity,
        },
        stages=stages,
        conventions=PhysicsConventions(maker_transmitted_policy=transmitted_wave_policy),
        validation=ValidationStatus(
            status,
            "benchmarks/mathematica_reference/maker_fringes_reference_v1.json",
            notes=(
                "Nonsingular Maker cases passed strict Mathematica comparison for the SHAARP-selected transmitted-wave policy.",
                "Live-Wolfram multilayer end-to-end validation now spans single-film (ext1, 12 incidence angles), "
                "4-layer with isotropic interlayer (ml1, 6/6), 5-layer with a non-degenerate anisotropic interlayer "
                "(ml2, 4/4), and two coupled active layers (ml3, 4/4) -- see "
                "stages['mathematica_validation_artifacts']['multilayer_breadth_live_wolfram_summaries'] "
                "(pass include_validation_summary=True).",
                "The isotropic/quasi-isotropic degenerate-mode bug (identical degenerate transverse eigenvectors -> "
                "rank-deficient multilayer matrix) was fixed 2026-05-31; multilayer stacks now agree to roundoff.",
                "The phase-matching-like case remains a documented singular solveInhom discrepancy.",
                "The physical transmitted-wave sum is exposed separately and is not the SHAARP.ml GUI default.",
            ),
        ),
        raw=raw,
    )


def run_sample_rotation(case: MultilayerSystem, azimuth_grid: Any, options: dict[str, Any] | None = None) -> SHAARPResult:
    """Run a physical sample-azimuth sweep separate from polarizer angle."""

    options = dict(options or {})
    include_validation_summary = bool(options.pop("include_validation_summary", False))
    raw = solve_multilayer_shg_sample_azimuth_sweep(
        case,
        sample_azimuth_deg=azimuth_grid,
        rotate_top=options.pop("rotate_top", False),
        rotate_substrate=options.pop("rotate_substrate", False),
        condition_threshold=options.pop("condition_threshold", 1e12),
        inhomogeneous_source_policy=options.pop("inhomogeneous_source_policy", "forward_only"),
        inhomogeneous_solution_policy=options.pop("inhomogeneous_solution_policy", "solve"),
    )
    if options:
        raise ValueError(f"Unsupported run_sample_rotation options: {sorted(options)}")
    stages = {"results": raw.results}
    if include_validation_summary:
        stages["validation_artifacts"] = _load_sample_rotation_validation_artifacts()
    return SHAARPResult(
        kind="sample_rotation",
        numeric={
            "sample_azimuth_deg": raw.sample_azimuth_deg,
            "analyzer_amplitude": raw.analyzer_amplitude,
            "intensity": raw.intensity,
            "reflected_parallel_amplitude": raw.reflected_parallel_amplitude,
            "reflected_perpendicular_amplitude": raw.reflected_perpendicular_amplitude,
            "transmitted_parallel_amplitude": raw.transmitted_parallel_amplitude,
            "transmitted_perpendicular_amplitude": raw.transmitted_perpendicular_amplitude,
            "reflected_parallel_intensity": raw.reflected_parallel_intensity,
            "reflected_perpendicular_intensity": raw.reflected_perpendicular_intensity,
            "transmitted_parallel_intensity": raw.transmitted_parallel_intensity,
            "transmitted_perpendicular_intensity": raw.transmitted_perpendicular_intensity,
        },
        stages=stages,
        validation=ValidationStatus(
            "mathematica_nonsingular_match_with_phase_matching_diagnostic",
            notes=(
                "Mathematica SampleRotate nonsingular references match; phase-matching-like case "
                "remains a singular solveInhom diagnostic, so full SampleRotate parity is not claimed.",
                "An extended live-Wolfram azimuth grid (ext1, 12 nonsingular cases across all 4 orientations) "
                "matches to roundoff -- see stages['validation_artifacts']['extended_azimuth_grid_ext1'] "
                "(pass include_validation_summary=True).",
            ),
        ),
        raw=raw,
    )


def run_si_full_analytical(case: Any, options: dict[str, Any] | None = None) -> SHAARPResult:
    """Closed-form (symbolic) single-interface reflected SHG -- the SHAARP.si analytical path.

    The facade behind the desktop GUI's SHAARP.si *Partial / Full Analytical Expression*. With
    ``options={"workflow": "polarimetry"}`` it returns the validated full-analytical reflected-SHG
    **polarimetry** closed form (symbolic in the input polarization ``phi`` and the ``d_ij`` tensor),
    whose expression strings are rendered by :func:`~shaarp.analytical_expression_text`.

    Args:
        case: a dict describing the case, e.g. ``{"point_group": "3m", "incident_theta_rad": ...}``.
        options: workflow selection and flags, e.g. ``{"workflow": "polarimetry", "simplify": False}``.
            (Other workflows expose the isotropic SymPy assembly and are labeled scaffolds.)

    Returns:
        SHAARPResult: ``kind="..._polarimetry"`` with the closed-form expressions in ``.stages``.
    """

    options = dict(options or {})
    workflow = options.pop("workflow", "isotropic_symbolic_scaffold")
    simplify = bool(options.pop("simplify", False))
    # LAYERED (opt-in): emit the derivation as a chain of NAMED intermediates -- k^(T,o,w),
    # theta^(T,e,w), E^(T,o,w)_Li, C^(T,ee,2w)_Li -- exactly as SHAARP.si prints it, with the
    # final amplitudes written in those names. Also what keeps a BIAXIAL eps tractable.
    # DEFAULT FALSE so this facade stays a FAITHFUL WRAPPER of solve_si_shg_full_analytical_symbolic
    # (whose own default is the flattened form): the facade must not silently return a different
    # expression than the solver it exposes, and existing callers keep the directly-evaluable form.
    # The desktop GUI opts in explicitly for its "Full Analytical" mode.
    layered = bool(options.pop("layered", False))
    if workflow not in {"isotropic_symbolic_scaffold", "polarimetry"}:
        raise ValueError(
            "run_si_full_analytical workflow must be 'isotropic_symbolic_scaffold' or 'polarimetry'."
        )
    if options:
        raise ValueError(f"Unsupported run_si_full_analytical options: {sorted(options)}")

    if workflow == "polarimetry":
        return _run_si_full_analytical_polarimetry(case, simplify, layered=layered)

    from .symbolic import d_voigt_symbolic, solve_isotropic_single_interface_shg_symbolic

    sp = _sympy_module()
    params = dict(case or {})
    point_group = params.pop("point_group", "1")
    incident_index_omega = params.pop("incident_index_omega", 1.0 + 0.01j)
    transmitted_index_omega = params.pop("transmitted_index_omega", 1.55 + 0.04j)
    incident_index_2omega = params.pop("incident_index_2omega", 1.04 + 0.02j)
    transmitted_index_2omega = params.pop("transmitted_index_2omega", 1.78 + 0.05j)
    incident_theta_rad = params.pop("incident_theta_rad", 0.31)
    incident_polarization = params.pop("incident_polarization", "p")
    d_voigt_lab = params.pop("d_voigt_lab", d_voigt_symbolic(point_group))
    epsilon_2omega_lab = params.pop(
        "epsilon_2omega_lab",
        sp.eye(3) * transmitted_index_2omega**2,
    )
    if params:
        raise ValueError(f"Unsupported run_si_full_analytical case keys: {sorted(params)}")

    raw = solve_isotropic_single_interface_shg_symbolic(
        transmitted_epsilon_2omega=epsilon_2omega_lab,
        d_voigt_lab=d_voigt_lab,
        incident_index_omega=incident_index_omega,
        transmitted_index_omega=transmitted_index_omega,
        incident_index_2omega=incident_index_2omega,
        transmitted_index_2omega=transmitted_index_2omega,
        incident_theta_rad=incident_theta_rad,
        incident_polarization=incident_polarization,
        simplify=simplify,
    )
    return SHAARPResult(
        kind="si_full_analytical_isotropic_scaffold",
        numeric={},
        stages={
            "workflow": workflow,
            "point_group": point_group,
            "linear_coefficients": _symbolic_to_strings(raw.linear_omega.coefficients),
            "shg_boundary_coefficients": _symbolic_to_strings(raw.shg.boundary.coefficients),
            "boundary_residual": _symbolic_to_strings(raw.shg.boundary.boundary_residual),
            "source_polarizations": {
                "ee": _symbolic_to_strings(raw.shg.sources.ee.polarization),
                "oo": _symbolic_to_strings(raw.shg.sources.oo.polarization),
                "eo": _symbolic_to_strings(raw.shg.sources.eo.polarization),
            },
            "symbol_metadata": {
                "d_voigt_prefix": "d",
                "distinct_frequency_parameters": [
                    "incident_index_omega",
                    "transmitted_index_omega",
                    "incident_index_2omega",
                    "transmitted_index_2omega",
                ],
                "orientation_layer_subscripts": "not present in this legacy isotropic scaffold",
            },
        },
        validation=ValidationStatus(
            "si_isotropic_full_analytical_published_validated__general_facade_not_full_mathematica_validated",
            notes=(
                "ISOTROPIC (e.g. GaAs class) full-analytical chain: FORM-VALIDATED at EVERY stage against the "
                "PUBLISHED supplementary (npj Comput Mater s41524-022-00930-4, Suppl. Note 5, eq 9-32) for "
                "GaAs(111) -- omega-boundary fundamental fields (eq 29-32), nonlinear polarization P_NL "
                "(eq 20-28, all 9 ee/oo/eo components), solveInhom C_Li (eq 11-19), and the 2omega-boundary "
                "final signal E_p/E_s^2w (eq 9-10). Verified in BOTH the real and the COMPLEX (absorbing "
                "n-tilde) domain, with the physically-correct decaying branch. See "
                "tests/test_published_gaas111_pnl_supplementary.py + tests/test_published_gaas111_stages_supplementary.py.",
                "Building blocks are ALSO live-Wolfram value-validated (doldExp 22 SHG classes; dnewExp "
                "crystal->lab; computePNL/partial-source vs 36-case si_pnl_stage; solveInhom vs 20 live cases).",
                "SCOPE CAVEAT (honest): the GENERAL ANISOTROPIC (biaxial/uniaxial) reflected-SHG assembly still "
                "uses NUMERIC principal-axis eigenmodes for the eigenvalue stage, so a closed-form symbolic-"
                "equivalence claim is made ONLY for the isotropic class here, not for arbitrary anisotropic d/eps.",
            ),
        ),
        warnings=(
            "Isotropic full-analytical chain is published-validated (real+complex); general anisotropic "
            "eigenmode stage remains numeric.",
        ),
    )


def run_ml_partial_analytical(case: Any, options: dict[str, Any] | None = None) -> SHAARPResult:
    """Closed-form (symbolic) multilayer thin-film SHG -- the SHAARP.ml partial-analytical path.

    The facade behind the desktop GUI's SHAARP.ml *Partial Analytical Expressions*. With
    ``options={"workflow": "polarimetry"}`` it returns the validated single-film SHG **polarimetry**
    closed form, symbolic in the input polarization ``phi``, the ``d_ij`` tensor, and the film
    thickness ``h`` (the remaining quantities numeric). Render with
    :func:`~shaarp.analytical_expression_text`.

    Args:
        case: a dict describing the film, e.g. ``{"point_group": "3m"}`` (plus optional indices /
            thickness / angle symbols).
        options: workflow selection, e.g. ``{"workflow": "polarimetry"}``.

    Returns:
        SHAARPResult: ``kind="ml_partial_analytical_polarimetry"`` with the closed-form expressions in
        ``.stages``. For a centrosymmetric/isotropic film the GUI returns a graceful
        "SHG symmetry-forbidden" result instead (see :func:`~shaarp.compute_ml_gui_result`).
    """

    options = dict(options or {})
    workflow = options.pop("workflow", "pnl_symbolic_scaffold")
    simplify = bool(options.pop("simplify", True))
    if workflow not in {"pnl_symbolic_scaffold", "polarimetry"}:
        raise ValueError(
            "run_ml_partial_analytical workflow must be 'pnl_symbolic_scaffold' or 'polarimetry'."
        )
    if options:
        raise ValueError(f"Unsupported run_ml_partial_analytical options: {sorted(options)}")

    if workflow == "polarimetry":
        return _run_ml_partial_analytical_polarimetry(case)

    from .symbolic import d_voigt_symbolic, partial_pnl_expressions

    sp = _sympy_module()
    params = dict(case or {})
    point_group = params.pop("point_group", "1")
    prefix = params.pop("prefix", "d")
    h_symbol = params.pop("h_symbol", "h")
    e_fast = params.pop("e_fast", sp.Matrix(sp.symbols("e_ex e_ey e_ez")))
    e_slow = params.pop("e_slow", sp.Matrix(sp.symbols("e_ox e_oy e_oz")))
    if params:
        raise ValueError(f"Unsupported run_ml_partial_analytical case keys: {sorted(params)}")

    d_voigt = d_voigt_symbolic(point_group, prefix=prefix)
    sources = partial_pnl_expressions(point_group, e_fast, e_slow, prefix=prefix)
    if simplify:
        sources = {label: value.applyfunc(sp.simplify) for label, value in sources.items()}
    return SHAARPResult(
        kind="ml_partial_analytical_pnl_scaffold",
        numeric={},
        stages={
            "workflow": workflow,
            "point_group": point_group,
            "d_voigt": _symbolic_to_strings(d_voigt),
            "partial_sources": {label: _symbolic_to_strings(value) for label, value in sources.items()},
            "symbol_metadata": {
                "d_voigt_prefix": prefix,
                "h": str(sp.Symbol(h_symbol)),
                "fast_wave_symbols": _symbolic_to_strings(e_fast),
                "slow_wave_symbols": _symbolic_to_strings(e_slow),
                "point_group_tensor_source_reference": (
                    "benchmarks/mathematica_reference/point_group_tensor_patterns_from_si_source_v1.json"
                ),
                "ml_partial_pnl_symbolic_reference": (
                    "benchmarks/mathematica_reference/wolfram_live_symbolic_ml_partial_pnl_reference_v1.json"
                ),
            },
        },
        validation=ValidationStatus(
            "python_symbolic_pnl_scaffold_not_full_mathematica_validated",
            notes=(
                "This exposes GUI-style dij partial PNL expressions and a symbolic layer-thickness placeholder.",
                "It is not the full SHAARP.ml partial analytical multilayer expression benchmark; the VALIDATED closed-form route is run_ml_partial_analytical(workflow=polarimetry) -- symbolic in phi/d/h and validated against the numeric solver.",
                "Provenance: the partial-PNL source expressions ARE live-Wolfram value-validated -- "
                "symbolic computePNL P2o/P2eo/Peoo match the 36-case si_pnl_stage live reference to ~1e-16 "
                "(benchmarks/mathematica_reference/symbolic_pnl_stage_live_comparison_summary_v1.json). The "
                "multilayer symbolic phase-factor assembly on top is not benchmarked (use the validated polarimetry route above).",
            ),
        ),
        warnings=(
            "Scoped partial analytical scaffold; multilayer symbolic phase factors and Mathematica references remain queued.",
        ),
    )


def _run_si_full_analytical_polarimetry(case: Any, simplify: bool, *, layered: bool = True) -> SHAARPResult:
    """SHAARP.si FULL-analytical SHG POLARIMETRY through the public API: the reflected SHG
    as a closed form in the input polarization phi (and analyzer psi, sample azimuth, theta,
    indices, d_ijk) -- the expression fit to a polarimetry scan to extract d.

    case keys (all optional): point_group ('-43m' default), d_voigt_lab (symbolic Matrix;
    default from point_group), incident_theta_rad, eps_omega_principal=(ex,ey,ez),
    eps_2omega_principal, phi_symbol, analyzer_symbol, sample_azimuth_symbol.
    """

    from .symbolic import d_voigt_symbolic, solve_si_shg_full_analytical_symbolic

    sp = _sympy_module()
    params = dict(case or {})
    point_group = params.pop("point_group", "-43m")
    d_voigt_lab = params.pop("d_voigt_lab", d_voigt_symbolic(point_group))
    incident_theta_rad = params.pop("incident_theta_rad", 0.5)
    eps_omega = params.pop("eps_omega_principal", (2.2**2, 2.2**2, 2.5**2))
    eps_2omega = params.pop("eps_2omega_principal", (2.4**2, 2.4**2, 2.7**2))
    phi_symbol = params.pop("phi_symbol", sp.Symbol("phi", real=True))
    analyzer_symbol = params.pop("analyzer_symbol", sp.Symbol("psi", real=True))
    sample_azimuth_symbol = params.pop("sample_azimuth_symbol", None)
    if params:
        raise ValueError(f"Unsupported run_si_full_analytical case keys: {sorted(params)}")

    capture: dict = {}
    sol = solve_si_shg_full_analytical_symbolic(
        eps_x_omega=eps_omega[0], eps_y_omega=eps_omega[1], eps_z_omega=eps_omega[2],
        eps_x_2omega=eps_2omega[0], eps_y_2omega=eps_2omega[1], eps_z_2omega=eps_2omega[2],
        d_voigt_lab=d_voigt_lab, incident_theta_rad=incident_theta_rad,
        phi_symbol=phi_symbol, analyzer_symbol=analyzer_symbol,
        sample_azimuth_symbol=sample_azimuth_symbol, simplify=simplify, layered=layered,
        capture_stages=capture,
    )
    # Full published derivation, step by step (eq 29-32 -> 20-28 -> 11-19 -> 9-10): the GUI shows
    # these as collapsible stages ABOVE the final reflected amplitudes, matching how the original
    # notebook builds up the closed form. Each block is the ACTUAL object used in THIS solve.
    deriv: dict[str, str] = {}
    try:
        em = capture.get("omega_eigenmodes")
        if em is not None:
            deriv["deriv_1_omega"] = "\n".join(
                f"E^(T,w)_{'ab'[i]} = {sp.Matrix(em[i].electric).T.tolist()[0]}"
                for i in range(len(em)))
        src = capture.get("sources")
        if src is not None:
            deriv["deriv_2_pnl"] = "\n".join(
                f"P_{name} = {sp.Matrix(getattr(src, name).polarization).T.tolist()[0]}"
                for name in ("ee", "oo", "eo"))
        inh = capture.get("inhomogeneous")
        if inh is not None:
            deriv["deriv_3_inhom"] = "\n".join(
                f"E^(inh)_{name} = {sp.Matrix(getattr(inh, name).electric).T.tolist()[0]}"
                for name in ("ee", "oo", "eo"))
        # The LAYERED chain: every named intermediate with its own defining equation, in the order
        # SHAARP.si derives them (k -> theta -> E at omega, the 2omega bases, then the bound C_Li).
        # The final amplitudes below are written in exactly these names -- his printed structure.
        defs = capture.get("layered_definitions")
        if defs:
            deriv["deriv_0_definitions"] = "\n".join(f"{nm} = {ex}" for nm, ex in defs)
    except Exception:
        deriv = {}  # a capture hiccup must never break the (validated) final expressions
    return SHAARPResult(
        kind="si_full_analytical_polarimetry",
        numeric={},
        stages={
            "workflow": "polarimetry",
            "point_group": point_group,
            **deriv,
            "reflected_s_2omega": str(sol.reflected_s),
            "reflected_p_2omega": str(sol.reflected_p),
            "analyzed_intensity": str(sol.intensity),
            "symbols": {
                "input_polarization": str(phi_symbol),
                "analyzer": str(analyzer_symbol),
                "sample_azimuth": str(sample_azimuth_symbol),
            },
        },
        validation=ValidationStatus(
            "si_full_analytical_polarimetry_form_validated_published_gaas111_and_numeric_jones__not_full_mathematica_validated",
            notes=(
                "The reflected SHG POLARIMETRY (closed form in input polarization phi with d_ijk as the "
                "symbolic coefficient -- the d-extraction expression), symbolic in phi, analyzer psi, sample "
                "azimuth, incidence theta, refractive indices, and d_ijk.",
                "FORM-VALIDATED: vs the PUBLISHED GaAs(111) full closed form (npj Comput Mater "
                "s41524-022-00930-4, Suppl. Note 5, eq 9-32) over a phi sweep (ratio 1.000 to the 5-decimal "
                "floor, both channels, incl. the eo cross term); s/p endpoints vs numeric "
                "solve_single_interface_shg; and the C3v 3-fold sample-azimuth symmetry. For the ANISOTROPIC "
                "(uniaxial/biaxial) classes the full phi-curve (cross term incl.) is validated vs the numeric "
                "arbitrary-Jones solve_single_interface_shg(incident_jones=...) over a phi sweep (~1e-12..1e-16, "
                "real + complex absorbing). See tests/test_si_shg_polarimetry_symbolic.py.",
                "SCOPE (honest): this is a form/numeric-equivalence validation of the closed-form polarimetry, "
                "not a full end-to-end Mathematica SHAARP.si replacement claim.",
            ),
        ),
        warnings=(
            "Closed-form polarimetry is form-validated (published GaAs(111) + numeric Jones incl. anisotropic); "
            "not a full end-to-end Mathematica .si validation.",
        ),
    )


# Above this many expression-tree NODES an expression is not rendered to a string (see
# _render_or_summarize). Measured, not guessed -- see that function's docstring.
_RENDER_NODE_LIMIT = 200_000


def _expr_node_count(expr: Any, limit: int) -> int:
    """Nodes in the expression tree, counting no further than ``limit``.

    Bounded on purpose: the guard must never become the cost it is guarding against, so this stops
    as soon as the limit is passed instead of walking a multi-million-node tree to completion."""
    sp = _sympy_module()
    n = 0
    for _ in sp.preorder_traversal(expr):
        n += 1
        if n > limit:
            break
    return n


def _render_or_summarize(expr: Any) -> str:
    """``str(expr)``, unless the expression is so large that PRINTING it is the bottleneck.

    WHY (profiled). The ML partial-analytical panel froze the GUI for ~8 minutes on the
    shipped case ML | GaAs (111) 800 nm: 471 s per Update, reproduced independently, and 1269.7 s
    for that material's sweep group against 79-92 s for every other ML material. Profiling the
    click showed the cost is NOT the physics -- of 1089 s inside compute_ml_gui_result, **1065 s
    (98%) was sympy string printing** (sstr / _print_Add / _print_Mul / as_ordered_terms, including
    2.1M Expr.__complex__ calls while ordering terms). The solve itself is ~24 s.

    Root cause of the SIZE: GaAs at 800 nm is ABSORBING (Im eps(2w) = 17.6, since 400 nm is deep in
    its absorption), so every coefficient is complex and the tree explodes. Measured directly: the
    reflected amplitude is an Add with only **11 top-level terms**, but it renders to **21,167,857
    characters (21 MB)** in 461 s -- the depth, not the top-level width, is what costs. (An earlier
    guard keyed on len(expr.args) therefore never fired: 11 <= any sane term threshold. The 691k
    _print_Add calls in the profile are RECURSIVE, not top-level terms.) The obvious alternative
    explanation, a dense rotated d, was tested and REFUTED: GaAs(111) has 8/18 nonzero lab d
    components, identical to LiNbO3 z-cut, which renders ~20x faster.

    So the guard is on RENDERING only, and keys on TREE SIZE. The solve still runs and the returned
    object is unchanged -- this never alters what is computed, only what is turned into display
    text. A 21 MB string is not usable output anyway: no one reads it, no text widget wants it, and
    producing it is precisely what freezes the window."""
    try:
        n_nodes = _expr_node_count(expr, _RENDER_NODE_LIMIT)
    except Exception:
        return str(expr)
    if n_nodes <= _RENDER_NODE_LIMIT:
        return str(expr)
    return (f"[expression too large to display: over {_RENDER_NODE_LIMIT:,} expression-tree nodes. "
            f"Rendering it is the cost, not the solve -- this happens for ABSORBING materials "
            f"(complex eps), where every coefficient is complex. The result object is complete and "
            f"unchanged; use the numeric workflows to work with it.]")


def _run_ml_partial_analytical_polarimetry(case: Any) -> SHAARPResult:
    """SHAARP.ml PARTIAL-analytical SHG POLARIMETRY through the public API: the reflected SHG
    of a single nonlinear film as a closed form in the input polarization phi, the SHG
    coefficient d_ijk, and the thickness h -- with eigenmodes/angles/indices NUMERIC (that
    split is what makes SHAARP.ml "partial"). The d-extraction expression for a film.

    case keys (all optional): point_group ('-43m'), d_voigt_symbolic (3x6 Matrix; default
    from point_group), incident_theta_rad, film_index_omega, substrate_index_omega,
    film_index_2omega, substrate_index_2omega, thickness_symbol, phi_symbol.
    """

    import numpy as np

    from .multilayer_basis import build_multilayer_2omega_basis, build_multilayer_omega_basis
    from .multilayer_shg_symbolic import solve_multilayer_shg_symbolic_polarimetry
    from .symbolic import d_voigt_symbolic, rotate_d_voigt_symbolic
    from .waves import EPS0, MU0

    sp = _sympy_module()
    params = dict(case or {})
    point_group = params.pop("point_group", "-43m")
    d_voigt = params.pop("d_voigt_symbolic", d_voigt_symbolic(point_group))
    incident_theta_rad = params.pop("incident_theta_rad", 0.5)
    nf = params.pop("film_index_omega", 2.1)
    ns = params.pop("substrate_index_omega", 1.5)
    nf2 = params.pop("film_index_2omega", 2.25)
    ns2 = params.pop("substrate_index_2omega", 1.55)
    # full 3x3 lab-frame tensors override the scalar-index shortcuts, so callers (the GUI) can pass
    # the REAL stack's epsilon incl. uniaxial films (release-audit l). Defaults
    # reproduce the original scalar construction byte-for-byte.
    film_eps_w = params.pop("film_epsilon_omega_lab", None)
    film_eps_2 = params.pop("film_epsilon_2omega_lab", None)
    sub_eps_w = params.pop("substrate_epsilon_omega_lab", None)
    sub_eps_2 = params.pop("substrate_epsilon_2omega_lab", None)
    h_symbol = params.pop("thickness_symbol", sp.Symbol("h", positive=True))
    phi_symbol = params.pop("phi_symbol", sp.Symbol("phi", real=True))
    # the ambient (incident-medium) indices — 1.0 = the historical hardwired air,
    # so the defaults reproduce every prior result exactly.
    amb_w_idx = complex(params.pop("ambient_index_omega", 1.0))
    amb_2_idx = complex(params.pop("ambient_index_2omega", 1.0))
    # N-layer keys. The singular film_* / d_voigt_symbolic / thickness_symbol keys above
    # remain the single-film spelling of the same thing, so every legacy caller is unaffected.
    layer_eps_w = params.pop("layer_epsilon_omega_lab", None)
    layer_eps_2 = params.pop("layer_epsilon_2omega_lab", None)
    layer_d_sym = params.pop("layer_d_voigt_symbolic", None)
    thickness_symbols = params.pop("thickness_symbols", None)
    # a SYMBOLIC sample azimuth. The SHG is linear in every d component and
    # `solve_multilayer_shg_symbolic_polarimetry` carries each one as the scalar weight of a
    # precomputed unit-d source column, so a d that is a trig polynomial in psi_s yields a closed
    # form in psi_s at ZERO extra linear solves. Physical only where eps is invariant under the
    # rotation (rotating eps turns a biaxial into a rotated biaxial, whose k_z solve is the full
    # Booker quartic) -- the caller must gate on that; see shaarp_gui.azimuth_closed_form_feasible.
    sample_azimuth_symbol = params.pop("sample_azimuth_symbol", None)
    ellipticity_rad = params.pop("ellipticity_rad", 0)
    if params:
        raise ValueError(f"Unsupported run_ml_partial_analytical case keys: {sorted(params)}")

    if layer_eps_w is not None:
        epsW = [np.asarray(e, dtype=complex) for e in layer_eps_w]
    else:
        epsW = [np.asarray(film_eps_w, dtype=complex) if film_eps_w is not None
                else np.diag([nf**2] * 3).astype(complex)]
    if layer_eps_2 is not None:
        eps2 = [np.asarray(e, dtype=complex) for e in layer_eps_2]
    else:
        eps2 = [np.asarray(film_eps_2, dtype=complex) if film_eps_2 is not None
                else np.diag([nf2**2] * 3).astype(complex)]
    d_layers = ([sp.Matrix(d) for d in layer_d_sym] if layer_d_sym is not None
                else [sp.Matrix(d_voigt)] * len(epsW))
    if sample_azimuth_symbol is not None:
        # Rz(+a), NOT its transpose: `CrystalOrientation.with_lab_azimuth_deg` (the numeric
        # sample-rotation sweep) composes as rotate_d(d_lab0, Rz(a) @ A0), verified to 1.7e-16
        # (the transpose is off by O(1)). NOTE the SI closed form
        # (symbolic.solve_si_shg_full_analytical_symbolic) uses the MIRROR convention -- do not
        # copy it here; that mismatch is filed for review.
        _c, _s = sp.cos(sample_azimuth_symbol), sp.sin(sample_azimuth_symbol)
        _rz = sp.Matrix([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]])
        d_layers = [rotate_d_voigt_symbolic(d, _rz, simplify=False) for d in d_layers]
    h_entries = list(thickness_symbols) if thickness_symbols is not None else [h_symbol] * len(epsW)
    if not (len(epsW) == len(eps2) == len(d_layers) == len(h_entries)):
        raise ValueError("layer_epsilon_*, layer_d_voigt_symbolic and thickness_symbols must "
                         "all have one entry per interior layer.")
    sub_w_mat = (np.asarray(sub_eps_w, dtype=complex) if sub_eps_w is not None
                 else np.diag([ns**2] * 3).astype(complex))
    sub_2_mat = (np.asarray(sub_eps_2, dtype=complex) if sub_eps_2 is not None
                 else np.diag([ns2**2] * 3).astype(complex))
    common = dict(
        incident_index=amb_w_idx, incident_theta_rad=incident_theta_rad,
        layer_epsilon_lab=epsW, substrate_epsilon_lab=sub_w_mat, omega=1.0,
    )
    bs = build_multilayer_omega_basis(incident_polarization="s", **common)
    bp = build_multilayer_omega_basis(incident_polarization="p", **common)
    b2 = build_multilayer_2omega_basis(
        top_index_2omega=amb_2_idx, tangential_index_omega=amb_w_idx,
        incident_theta_rad=incident_theta_rad,
        layer_epsilon_2omega_lab=eps2, substrate_epsilon_2omega_lab=sub_2_mat, omega_2=2.0,
    )
    sol = solve_multilayer_shg_symbolic_polarimetry(
        omega_basis_s=bs, omega_basis_p=bp, twoomega_basis=b2,
        layer_d_voigt_symbolic=d_layers, layer_epsilon_2omega_lab=eps2,
        thickness_symbols=h_entries, phi_symbol=phi_symbol, ellipticity=ellipticity_rad,
        # the SHAARP.ml unit normalization, as EVERY other path uses and as the
        # validated fence (tests/test_ml_shg_polarimetry_symbolic.py) compares under. Passing SI
        # MU0/EPS0 here made the SHG inhomogeneous operator `curl_curl(k) - w^2 mu eps0 eps`
        # numerically singular -- its two terms differ by ~1e16 in SI -- measured condition
        # 7.2e14..7.7e17 with all 30 solves flagging ill_conditioned and falling back silently.
        # The visible symptom was PA breaking the crystal's OWN point-group symmetry (MoS2 -6m2 at
        # 120 deg: 1.1e-2; quartz 32: 1.3e0). With the normalization the operator conditions at
        # 9.68 -- the same value the numeric path reports -- and the symmetry holds to 2.7e-15.
        mu=1.0, eps0=1.0,
    )
    return SHAARPResult(
        kind="ml_partial_analytical_polarimetry",
        numeric={},
        stages={
            "workflow": "polarimetry",
            "point_group": point_group,
            "reflected_s_2omega": _render_or_summarize(sol.reflected_s),
            "reflected_p_2omega": _render_or_summarize(sol.reflected_p),
            "symbols": {
                "input_polarization": str(phi_symbol),
                "thickness": ", ".join(str(e) for e in h_entries),
                **({"sample_azimuth": str(sample_azimuth_symbol)}
                   if sample_azimuth_symbol is not None else {}),
            },
            "layers": len(epsW),
        },
        validation=ValidationStatus(
            "ml_partial_analytical_polarimetry_form_validated_numeric_jones__not_full_mathematica_validated",
            notes=(
                "The reflected SHG POLARIMETRY of a single nonlinear film as a closed form in the input "
                "polarization phi, the SHG coefficient d_ijk, and the thickness h -- the d-extraction "
                "expression. PARTIAL by construction: eigenmodes, refraction angles, and refractive indices "
                "are NUMERIC; only phi, d_ijk, and h are symbolic (per SHAARP.ml's partial-analytical contract).",
                "FORM-VALIDATED: substituting numeric (phi, d, h) reproduces the independent numeric "
                "arbitrary-Jones multilayer workflow solve_multilayer_shg_from_tensors_jones over a (phi, h) "
                "sweep to ~1e-18 (real) / ~1e-10 (complex absorbing). See "
                "tests/test_ml_shg_polarimetry_symbolic.py.",
                "SCOPE (honest): a form/numeric-equivalence validation of the closed-form film polarimetry, "
                "not a full end-to-end Mathematica SHAARP.ml multilayer replacement claim.",
            ),
        ),
        warnings=(
            "Closed-form film polarimetry is form-validated vs the numeric Jones workflow; not a full "
            "end-to-end Mathematica .ml validation.",
        ),
    )


def _sympy_module() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "Analytical facades require SymPy. Install with `python -m pip install sympy`."
        ) from exc
    return sp


def _symbolic_to_strings(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _symbolic_to_strings(value.tolist())
    if isinstance(value, dict):
        return {key: _symbolic_to_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_symbolic_to_strings(item) for item in value]
    return str(value)


def compare_with_mathematica(reference: dict[str, Any], python_result: SHAARPResult, tolerance: dict[str, float] | None = None) -> list[Any]:
    """Compare selected Mathematica output keys against a `SHAARPResult`.

    Keys may be plain numeric-result keys such as ``"intensity"`` or dotted
    paths rooted at ``numeric``/``stages``/``validation``/``conventions`` such
    as ``"stages.normal_incidence_2omega_branch_policy"``.
    """

    tolerance = dict(tolerance or {})
    atol = float(tolerance.get("atol", 1e-8))
    rtol = float(tolerance.get("rtol", 1e-8))
    keys = reference.get("keys") or list(python_result.numeric)
    results = []
    for key in keys:
        mathematica_value = _resolve_reference_path(reference, key)
        python_value = _resolve_result_path(python_result, key)
        results.append(_compare_value(python_result.kind, key, mathematica_value, python_value, atol=atol, rtol=rtol))
    return results


def _compare_value(
    case_id: str,
    key: str,
    mathematica_value: Any,
    python_value: Any,
    *,
    atol: float,
    rtol: float,
) -> MathematicaComparisonResult:
    if _is_bool_like(mathematica_value) or _is_bool_like(python_value):
        passed = _is_bool_like(mathematica_value) and _is_bool_like(python_value) and bool(mathematica_value) == bool(python_value)
        error = 0.0 if passed else float("inf")
        return MathematicaComparisonResult(case_id, key, passed, error, error, ())
    if isinstance(mathematica_value, str) or isinstance(python_value, str):
        passed = str(mathematica_value) == str(python_value)
        error = 0.0 if passed else float("inf")
        return MathematicaComparisonResult(case_id, key, passed, error, error, ())
    if _contains_metadata_value(mathematica_value) or _contains_metadata_value(python_value):
        passed = to_jsonable(mathematica_value) == to_jsonable(python_value)
        error = 0.0 if passed else float("inf")
        return MathematicaComparisonResult(case_id, key, passed, error, error, ())
    return _compare_numeric_value(case_id, key, mathematica_value, python_value, atol=atol, rtol=rtol)


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _contains_metadata_value(value: Any) -> bool:
    if value is None or isinstance(value, str) or _is_bool_like(value):
        return True
    if isinstance(value, dict):
        if {"real", "imag"} <= set(value) or {"re", "im"} <= set(value):
            return False
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_metadata_value(item) for item in value)
    return False


def _resolve_reference_path(reference: dict[str, Any], key: str) -> Any:
    if key in reference:
        return reference[key]
    if "." not in key:
        raise ValueError(f"Mathematica reference is missing key {key!r}.")
    return _resolve_mapping_path(reference, key, missing_message=f"Mathematica reference is missing key {key!r}.")


def _resolve_result_path(result: SHAARPResult, key: str) -> Any:
    if "." not in key:
        if key not in result.numeric:
            raise ValueError(f"Python result is missing numeric key {key!r}.")
        return result.numeric[key]
    root, rest = key.split(".", 1)
    if root == "numeric":
        return _resolve_mapping_path(result.numeric, rest, missing_message=f"Python result is missing path {key!r}.")
    if root == "stages":
        return _resolve_mapping_path(result.stages, rest, missing_message=f"Python result is missing path {key!r}.")
    if root == "validation":
        return _resolve_mapping_path(result.validation, rest, missing_message=f"Python result is missing path {key!r}.")
    if root == "conventions":
        return _resolve_mapping_path(result.conventions, rest, missing_message=f"Python result is missing path {key!r}.")
    raise ValueError(f"Python result path must start with 'numeric', 'stages', 'validation', or 'conventions': {key!r}.")


def _resolve_mapping_path(mapping: Any, path: str, *, missing_message: str) -> Any:
    current = mapping
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if not isinstance(current, dict) and hasattr(current, part):
            current = getattr(current, part)
            continue
        raise ValueError(_missing_path_message(missing_message, current))
    return current


def _missing_path_message(message: str, current: Any) -> str:
    if isinstance(current, dict):
        return f"{message} available keys: {sorted(current)}."
    keys = [key for key in dir(current) if not key.startswith("_")]
    if keys:
        return f"{message} available keys: {sorted(keys)}."
    return message


def export_result(result: SHAARPResult, path: str | Path, format: str = "json") -> Path:
    """Export a high-level result as JSON or a flat CSV of numeric arrays."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        path.write_text(json.dumps(to_jsonable(result), indent=2), encoding="utf-8")
        return path
    if format == "csv":
        _export_numeric_csv(result.numeric, path)
        return path
    raise ValueError("format must be 'json' or 'csv'.")


def _reflected_analyzer_from_case(raw, case: MultilayerSystem) -> tuple[complex, float]:
    from .multilayer_shg_boundary import analyze_reflected_2omega_polarimetry

    return analyze_reflected_2omega_polarimetry(raw.shg, case)


def _run_gui_multilayer_fresnel_sweep(
    case: MultilayerSystem,
    angle_grid: Any,
    *,
    transmitted_wave_policy: str,
    include_validation_summary: bool,
) -> SHAARPResult:
    if transmitted_wave_policy not in {"shaarp_ml_selected", "physical_sum"}:
        raise ValueError("transmitted_wave_policy must be 'shaarp_ml_selected' or 'physical_sum'.")
    theta_deg = np.asarray(angle_grid if angle_grid is not None else [case.polarimetry.theta_deg], dtype=float)
    if theta_deg.ndim != 1:
        raise ValueError("angle_grid must be a one-dimensional sequence for workflow='gui_multilayer'.")
    if np.any(theta_deg <= -90.0) or np.any(theta_deg >= 90.0):
        raise ValueError("GUI multilayer Fresnel incidence angles must satisfy -90 < theta_deg < 90.")

    # rp/tp share P-incidence (s,p)=(0,1); rs/ts share S-incidence (1,0). Solve the
    # fundamental once per (angle, incident polarization) -- reading reflected AND
    # transmitted from each solve -- and build the angle-independent setup once. This
    # makes a fine, continuous incidence-angle Fresnel sweep affordable.
    from .multilayer_shg_boundary import _system_setup

    base_setup = _system_setup(case)
    # p- and s-incidence at the same angle share every anisotropic mode solve;
    # this cache makes the second pass reuse the first's bases (byte-identical, ~2x faster).
    basis_cache: dict = {}
    rp, tp = _gui_fresnel_pair(case, theta_deg, (0.0j, 1.0 + 0.0j), component="p", transmitted_wave_policy=transmitted_wave_policy, base_setup=base_setup, basis_cache=basis_cache)
    rs, ts = _gui_fresnel_pair(case, theta_deg, (1.0 + 0.0j, 0.0j), component="s", transmitted_wave_policy=transmitted_wave_policy, base_setup=base_setup, basis_cache=basis_cache)

    stages: dict[str, Any] = {
        "workflow": "gui_multilayer",
        "transmitted_wave_policy": transmitted_wave_policy,
        "listFresnel_labels": ["Rp", "Rs", "Tp", "Ts"],
    }
    if include_validation_summary:
        stages["mathematica_validation_artifacts"] = _safe_validation_artifacts(_load_fresnel_validation_artifacts)
    status = (
        "gui_fresnel_list_selected_policy_mathematica_validated_with_transmitted_sum_caveat"
        if transmitted_wave_policy == "shaarp_ml_selected"
        else "physically_motivated_not_mathematica_gui_default"
    )
    return SHAARPResult(
        kind="fresnel_sweep",
        numeric={"theta_deg": theta_deg, "rp": rp, "rs": rs, "tp": tp, "ts": ts},
        stages=stages,
        conventions=PhysicsConventions(maker_transmitted_policy=transmitted_wave_policy),
        validation=ValidationStatus(
            status,
            "benchmarks/mathematica_reference/fresnel_sweep_reference_v1.json",
            notes=(
                "workflow='gui_multilayer' mirrors SHAARP.ml GUI listFresnel curves.",
                "Transmitted-wave policy controls SHAARP-compatible selected wave versus physical total transmitted field.",
            ),
        ),
    )


def _gui_fresnel_pair(
    case: MultilayerSystem,
    theta_deg: np.ndarray,
    incident_jones_sp: tuple[complex, complex],
    *,
    component: str,
    transmitted_wave_policy: str,
    base_setup: dict | None = None,
    basis_cache: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflected AND transmitted Fresnel POWER curves for ONE incident polarization.

    The linear fundamental is solved ONCE per angle and BOTH the reflected
    (``solution.top_unknown``) and transmitted (``solution.substrate_unknown``)
    fields are read from that single solve -- so the GUI's four Fresnel curves
    (Rp/Rs/Tp/Ts) cost two solves per angle, not four. The angle-independent
    ``_system_setup`` is built once (``base_setup``) and only
    ``incident_theta_rad`` is overridden per angle (byte-identical to recomputing
    it). ``component`` selects 'p' or 's'. Returns ``(reflected_power,
    transmitted_power)`` over ``theta_deg``."""
    from .multilayer_shg_boundary import _system_setup

    if base_setup is None:
        base_setup = _system_setup(case)
    reflected: list[float] = []
    transmitted: list[float] = []
    for theta in theta_deg:
        point_pol = replace(case.polarimetry, theta_deg=float(theta))
        setup = dict(base_setup)
        setup["incident_theta_rad"] = float(point_pol.radians()[0])
        try:
            solution = _solve_gui_fresnel_fundamental(case, incident_jones_sp, setup=setup, basis_cache=basis_cache)
        except np.linalg.LinAlgError:
            # ISOLATED removable singularity of the boundary solve (same class as the Maker-fringe
            # eigen-degeneracies; e.g. the Pt-film metal at exactly 85.0 deg, exposed when the
            # Fresnel range was fixed to the original's 0-90). Mask the point; interpolated below.
            reflected.append(np.nan)
            transmitted.append(np.nan)
            continue
        s_r, p_r = _sum_wave_frame_jones_sp(solution.top_unknown)
        reflected.append(float(abs(p_r if component == "p" else s_r) ** 2))
        if transmitted_wave_policy == "shaarp_ml_selected":
            t_waves = _select_fundamental_transmitted_wave(solution.substrate_unknown)
        else:
            t_waves = solution.substrate_unknown
        s_t, p_t = _sum_wave_frame_jones_sp(t_waves)
        transmitted.append(float(abs(p_t if component == "p" else s_t) ** 2))
    r_arr = np.asarray(reflected, dtype=float)
    t_arr = np.asarray(transmitted, dtype=float)
    th_arr = np.asarray(list(theta_deg), dtype=float)
    for arr in (r_arr, t_arr):
        bad = ~np.isfinite(arr)
        if bad.any() and (~bad).sum() >= 2:
            arr[bad] = np.interp(th_arr[bad], th_arr[~bad], arr[~bad])
    return r_arr, t_arr


def _select_fundamental_transmitted_wave(waves):
    """Same precise rule as the Maker transmitted-wave selection (_transmitted_waves_for_maker_
    policy): keep slot [0] -- and ONLY when slot [0] is EMPTY while another slot carries the field
    (the DEGENERATE-substrate basis-swap signature, e.g. an air substrate), follow the field.
    Reading slot [0] unconditionally made the GUI Fresnel tp/ts curves swap with exact-0 holes at
    scattered angles (caught by the continuity lens, -- third instance of the class)."""
    if not waves:
        return []
    norms = [float(np.linalg.norm(np.asarray(w.electric))) for w in waves]
    best = max(norms)
    if best > 0.0 and norms[0] <= 1e-9 * best:
        return [waves[int(np.argmax(norms))]]
    return [waves[0]]


def _solve_gui_fresnel_fundamental(case: MultilayerSystem, incident_jones_sp: tuple[complex, complex], *, setup: dict | None = None, basis_cache: dict | None = None):
    from .multilayer_basis import build_multilayer_omega_basis_jones
    from .multilayer_boundary import solve_multilayer_boundary
    from .multilayer_shg_boundary import _system_setup

    # `setup` (optional): precomputed _system_setup with `incident_theta_rad` already
    # set for this angle. Only that entry is angle-dependent, so a sweep builds the
    # setup once and overrides it per angle (byte-identical, far cheaper).
    if setup is None:
        setup = _system_setup(case)
    # `basis_cache` (optional, keyed by angle) lets the SECOND incident
    # polarization at an angle reuse the first's anisotropic bases -- the brentq-heavy half of the
    # Fresnel sweep, and independent of polarization. Byte-identical; halves the sweep.
    _key = None if basis_cache is None else float(setup["incident_theta_rad"])
    basis = build_multilayer_omega_basis_jones(
        incident_index=setup["top_index_omega"],
        incident_theta_rad=setup["incident_theta_rad"],
        incident_jones_sp=incident_jones_sp,
        layer_epsilon_lab=setup["layer_epsilon_omega_lab"],
        substrate_epsilon_lab=setup["substrate_epsilon_omega_lab"],
        omega=setup["omega"],
        layer_shaarp_basis=setup["layer_shaarp_basis_omega"],
        substrate_shaarp_basis=setup["substrate_shaarp_basis_omega"],
        reuse=(basis_cache.get(_key) if basis_cache is not None else None),
    )
    if basis_cache is not None and _key not in basis_cache:
        basis_cache[_key] = basis
    return solve_multilayer_boundary(
        top_known=basis.incident,
        top_unknown_basis=basis.reflected_basis,
        layer_unknown_basis=basis.layer_basis,
        substrate_unknown_basis=basis.substrate_basis,
        thicknesses=setup["thicknesses"],
        mu=1.0,
    )


def _sum_wave_frame_jones_sp(waves) -> tuple[complex, complex]:
    from .multilayer_shg_boundary import _sum_wave_frame_jones_sp as sum_wave_frame_jones_sp

    return sum_wave_frame_jones_sp(waves)


def _load_si_stage_comparison_summary() -> dict[str, Any]:
    return _load_project_json("benchmarks/mathematica_reference/shaarp_si_stage_comparison_summary_v1.json")


def _load_maker_validation_artifacts() -> dict[str, Any]:
    report_paths = [
        "benchmarks/mathematica_reference/discrepancy_reports/maker_transmitted_wave_selection_potential_bug.json",
        "benchmarks/mathematica_reference/discrepancy_reports/maker_phase_matching_solve_inhom_singular.json",
    ]
    return {
        "status": (
            "maker_outputs_nonsingular_mathematica_validated_with_phase_matching_diagnostic_and_transmitted_sum_caveat"
        ),
        "mathematica_reference": "benchmarks/mathematica_reference/maker_fringes_reference_v1.json",
        "mathematica_stage_diagnostics": "benchmarks/mathematica_reference/maker_stage_diagnostics_v1.json",
        "fine_grid_summary": _load_project_json(
            "benchmarks/mathematica_reference/maker_fine_comparison_summary_v1.json"
        ),
        "fringe_resolution_summary": _load_project_json(
            "benchmarks/mathematica_reference/maker_fine_fringe_resolution_comparison_summary_v1.json"
        ),
        "phase_matching_local_summary": _load_project_json(
            "benchmarks/mathematica_reference/maker_fine_phase_matching_local_comparison_summary_v1.json"
        ),
        "high_oblique_stage_summary": _load_project_json(
            "benchmarks/mathematica_reference/maker_high_oblique_12_stage_comparison_summary_v1.json"
        ),
        "high_oblique_solvefresneln_summary": _load_project_json(
            "benchmarks/mathematica_reference/maker_high_oblique_12_solvefresneln_system_summary_v1.json"
        ),
        "high_oblique_linear_stage_summary": _load_project_json(
            "benchmarks/mathematica_reference/maker_high_oblique_80_linear_stage_summary_v1.json"
        ),
        "multilayer_breadth_live_wolfram_summaries": _multilayer_breadth_summaries(),
        "discrepancy_reports": [_load_project_json(path) for path in report_paths],
    }


def _load_fresnel_validation_artifacts() -> dict[str, Any]:
    solve_fresnel = _load_project_json("benchmarks/mathematica_reference/solve_fresnel_reference_v1.json")
    solve_fresnel_n = _load_project_json("benchmarks/mathematica_reference/solve_fresnel_n_reference_v1.json")
    gui_fresnel = _load_project_json("benchmarks/mathematica_reference/fresnel_sweep_reference_v1.json")
    return {
        "status": "gui_fresnel_list_selected_policy_mathematica_validated_with_transmitted_sum_caveat",
        "gui_fresnel_list_agreement_claimed": True,
        "gui_fresnel_reference": {
            "reference_path": "benchmarks/mathematica_reference/fresnel_sweep_reference_v1.json",
            "status": gui_fresnel["status"],
            "case_count": gui_fresnel["case_count"],
            "suite_ids": [suite["suite_id"] for suite in gui_fresnel["suites"]],
            "comparison_policy": (
                "SHAARP-compatible selected transmitted wave matches Mathematica; physical transmitted sum is a documented caveat."
            ),
        },
        "discrepancy_reports": [
            _load_project_json(
                "benchmarks/mathematica_reference/discrepancy_reports/fresnel_transmitted_wave_selection_potential_bug.json"
            )
        ],
        "component_references": {
            "solve_fresnel": {
                "reference_path": "benchmarks/mathematica_reference/solve_fresnel_reference_v1.json",
                "status": solve_fresnel["status"],
                "case_count": solve_fresnel["case_count"],
                "unknown_order": solve_fresnel["unknown_order"],
            },
            "solve_fresnel_n": {
                "reference_path": "benchmarks/mathematica_reference/solve_fresnel_n_reference_v1.json",
                "status": solve_fresnel_n["status"],
                "case_count": solve_fresnel_n["case_count"],
                "suite_ids": [suite["suite_id"] for suite in solve_fresnel_n["suites"]],
            },
        },
    }


def _load_ml_numeric_validation_artifacts() -> dict[str, Any]:
    system = _load_project_json("benchmarks/multilayer_system_benchmarks_v1.json")
    polarimetry = _load_project_json("benchmarks/multilayer_polarimetry_benchmarks_v1.json")
    boundary = _load_project_json("benchmarks/multilayer_boundary_system_benchmarks_v1.json")
    solve_fresnel_n = _load_project_json("benchmarks/mathematica_reference/solve_fresnel_n_reference_v1.json")
    return {
        "status": "python_regression_with_component_references_not_full_ml_agreement",
        "full_shaarp_ml_agreement_claimed": False,
        "python_regression_benchmarks": {
            "multilayer_system": {
                "reference_path": "benchmarks/multilayer_system_benchmarks_v1.json",
                "status": system["status"],
                "case_count": system["case_count"],
            },
            "multilayer_polarimetry": {
                "reference_path": "benchmarks/multilayer_polarimetry_benchmarks_v1.json",
                "status": polarimetry["status"],
                "case_count": polarimetry["case_count"],
            },
            "multilayer_boundary_system": {
                "reference_path": "benchmarks/multilayer_boundary_system_benchmarks_v1.json",
                "status": boundary["status"],
                "case_count": boundary["case_count"],
            },
        },
        "component_mathematica_references": {
            "solve_fresnel_n": {
                "reference_path": "benchmarks/mathematica_reference/solve_fresnel_n_reference_v1.json",
                "status": solve_fresnel_n["status"],
                "case_count": solve_fresnel_n["case_count"],
                "suite_ids": [suite["suite_id"] for suite in solve_fresnel_n["suites"]],
            },
        },
        "gui_mathematica_workflows": {
            "maker_fringes": _load_maker_validation_artifacts(),
            "fresnel_sweep": _load_fresnel_validation_artifacts(),
            "sample_rotation": _load_sample_rotation_validation_artifacts(),
        },
    }


def _load_sample_rotation_validation_artifacts() -> dict[str, Any]:
    polarimetry = _load_project_json("benchmarks/multilayer_polarimetry_benchmarks_v1.json")
    sample_rotation = _load_project_json("benchmarks/sample_rotation_benchmarks_v1.json")
    sample_rotation_reference = _load_project_json("benchmarks/mathematica_reference/sample_rotation_reference_v1.json")
    sample_rotation_summary = _load_project_json(
        "benchmarks/mathematica_reference/sample_rotation_comparison_summary_v1.json"
    )
    azimuths = sorted({float(case["sample_azimuth_deg"]) for case in polarimetry["cases"]})
    return {
        "status": "mathematica_sample_rotation_nonsingular_match_with_phase_matching_diagnostic",
        "mathematica_sample_rotation_agreement_claimed": sample_rotation_summary[
            "full_sample_rotation_agreement_claimed"
        ],
        "python_sample_rotation_benchmark": {
            "reference_path": "benchmarks/sample_rotation_benchmarks_v1.json",
            "status": sample_rotation["status"],
            "case_count": sample_rotation["case_count"],
        },
        "mathematica_reference": {
            "reference_path": "benchmarks/mathematica_reference/sample_rotation_reference_v1.json",
            "status": sample_rotation_reference["status"],
            "case_count": sample_rotation_reference["case_count"],
        },
        "comparison_summary": {
            "reference_path": "benchmarks/mathematica_reference/sample_rotation_comparison_summary_v1.json",
            "status": sample_rotation_summary["status"],
            "nonsingular_case_count": sample_rotation_summary["nonsingular_case_count"],
            "nonsingular_fail_count": sample_rotation_summary["nonsingular_fail_count"],
            "diagnostic_fail_count": sample_rotation_summary["diagnostic_fail_count"],
            "full_sample_rotation_agreement_claimed": sample_rotation_summary[
                "full_sample_rotation_agreement_claimed"
            ],
        },
        "python_regression_benchmark": {
            "reference_path": "benchmarks/multilayer_polarimetry_benchmarks_v1.json",
            "status": polarimetry["status"],
            "case_count": polarimetry["case_count"],
            "sample_azimuth_deg": azimuths,
        },
        "extended_azimuth_grid_ext1": _sample_rotation_ext1_summary(),
    }


def _sample_rotation_ext1_summary() -> dict[str, Any] | None:
    """Compact roll-up of the live-Wolfram extended SampleRotate azimuth grid
    (ext1: 12 nonsingular cases across all 4 orientations), validated.
    Existence-guarded so the facade degrades gracefully if the summary is absent."""
    payload = _load_project_json_optional(
        "benchmarks/mathematica_reference/sample_rotation_comparison_summary_ext1.json"
    )
    if payload is None:
        return None
    return {
        "reference_path": "benchmarks/mathematica_reference/sample_rotation_comparison_summary_ext1.json",
        "status": payload.get("status"),
        "full_sample_rotation_agreement_claimed": payload.get("full_sample_rotation_agreement_claimed"),
        "nonsingular_case_count": payload.get("nonsingular_case_count"),
        "nonsingular_fail_count": payload.get("nonsingular_fail_count"),
    }


def _load_project_json(relative_path: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_project_json_optional(relative_path: str) -> dict[str, Any] | None:
    """Like _load_project_json but returns None if the file is absent/unreadable.

    Used for validation-artifact summaries that may not all be present in every
    checkout, so the public facade advertises whatever evidence exists without
    crashing on a missing summary."""
    try:
        return _load_project_json(relative_path)
    except (OSError, json.JSONDecodeError):
        return None


def _safe_validation_artifacts(loader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Build the validation-metadata dict via ``loader``, but degrade GRACEFULLY if the
    referenced benchmark JSON files are absent or malformed.

    Validation artifacts are METADATA ONLY (status strings, case counts, reference paths) --
    they are NOT part of the physics compute. A frozen/packaged build that did not bundle the
    top-level ``benchmarks/`` directory must NOT crash an actual SHG computation just because
    this provenance metadata can't be loaded. (Root cause of the exe FileNotFoundError:
    the GUI requests include_validation_summary=True, which loaded benchmarks/*.json with the
    non-optional loader; PyInstaller bundles shaarp/ data but not top-level benchmarks/.) The
    result's own ``validation.status`` is set independently and still displays."""
    try:
        return loader()
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {"status": "validation_metadata_unavailable_in_this_build"}


def _multilayer_breadth_summaries() -> dict[str, Any]:
    """Compact roll-up of the live-Wolfram multilayer end-to-end validations
    (single-film ext1 + ml1/ml2/ml3 breadth), so run_maker_fringes /
    run_ml_numeric advertise the CURRENT validated scope, not just the original
    single-film v1 reference. Each entry is existence-guarded."""
    specs = {
        "single_film_incidence_ext1": "benchmarks/mathematica_reference/maker_fringes_comparison_summary_ext1.json",
        "multilayer_4layer_ml1": "benchmarks/mathematica_reference/maker_fringes_comparison_summary_ml1.json",
        "multilayer_5layer_anisotropic_ml2": "benchmarks/mathematica_reference/maker_fringes_comparison_summary_ml2.json",
        "multilayer_two_active_layer_ml3": "benchmarks/mathematica_reference/maker_fringes_comparison_summary_ml3.json",
    }
    out: dict[str, Any] = {}
    for key, path in specs.items():
        payload = _load_project_json_optional(path)
        if payload is None:
            continue
        out[key] = {
            "reference_path": path,
            "status": payload.get("status"),
            "full_shaarp_ml_agreement_claimed": payload.get("full_shaarp_ml_agreement_claimed"),
            "nonsingular_case_count": payload.get("nonsingular_case_count"),
            "nonsingular_fail_count": payload.get("nonsingular_fail_count"),
        }
    return out


def _export_numeric_csv(numeric: dict[str, Any], path: Path) -> None:
    arrays = {key: np.ravel(np.asarray(value)) for key, value in numeric.items() if np.asarray(value).shape != ()}
    scalars = {key: np.asarray(value).item() for key, value in numeric.items() if np.asarray(value).shape == ()}
    if not arrays:
        arrays = {key: np.asarray([value]) for key, value in scalars.items()}
        scalars = {}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("CSV export requires numeric arrays with the same flattened length.")
    headers = list(arrays)
    scalar_headers = list(scalars)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers + scalar_headers)
        for idx in range(next(iter(lengths))):
            row = [arrays[key][idx] for key in headers] + [scalars[key] for key in scalar_headers]
            writer.writerow([_csv_cell(value) for value in row])


def _csv_cell(value: Any) -> Any:
    if isinstance(value, complex) or np.iscomplexobj(value):
        item = complex(value)
        return f"{item.real:.17g}{item.imag:+.17g}j"
    if isinstance(value, np.generic):
        return value.item()
    return value


def _compare_numeric_value(
    case_id: str,
    key: str,
    mathematica_value: Any,
    python_value: Any,
    *,
    atol: float,
    rtol: float,
) -> MathematicaComparisonResult:
    mma = _numeric_array(mathematica_value)
    py = _numeric_array(to_jsonable(python_value))
    if mma.shape != py.shape:
        return MathematicaComparisonResult(case_id, key, False, float("inf"), float("inf"), tuple(py.shape))
    delta = np.abs(py - mma)
    scale = np.maximum(np.abs(mma), np.finfo(float).tiny)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    return MathematicaComparisonResult(
        case_id,
        key,
        bool(np.allclose(py, mma, atol=atol, rtol=rtol)),
        max_abs,
        max_rel,
        tuple(py.shape),
    )


def _numeric_array(value: Any) -> np.ndarray:
    return np.asarray(_convert_numeric(value), dtype=complex)


def _convert_numeric(value: Any) -> Any:
    if isinstance(value, dict):
        if {"real", "imag"} <= set(value):
            return complex(float(value["real"]), float(value["imag"]))
        if {"re", "im"} <= set(value):
            return complex(float(value["re"]), float(value["im"]))
        raise ValueError(f"Unsupported numeric object keys: {sorted(value)}")
    if isinstance(value, list):
        return [_convert_numeric(item) for item in value]
    if isinstance(value, (int, float, complex)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"Unsupported numeric value: {value!r}")
