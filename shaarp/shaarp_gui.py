"""Merged SHAARP.si + SHAARP.ml interactive GUI (one panel, navigate between the two).

The original SHAARP ships two separate Mathematica GUIs -- SHAARP.si (single interface) and
SHAARP.ml (multilayer). This module reproduces both as ONE ipywidgets panel with a top-level Tab
to switch between them (``make_shaarp_gui()``), each tab faithfully exposing its original's
Functionality modes and driving the VALIDATED Python solver facades (run_si_*/run_ml_*).

Design for testability: the per-tab "given the control values, produce the result" logic lives in
PURE, no-widget functions (``compute_si_gui_result`` / ``compute_ml_gui_result``) so the GUI logic
is unit-tested headlessly; the widgets are a thin wiring layer on top.

All build phases are COMPLETE (P1 navigation -> P4b polish; see): functionality
modes on the validated backends, constrained material/tensor entry, Miller-index orientation,
.ml assumptions + system presets, Fresnel/Maker outputs, 2D/3D schematics, material preset
slots, data export, and copyable closed-form analytical expressions.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from matplotlib.figure import Figure  # bare Figure (NOT pyplot): builders never register in the
# global pyplot figure registry, so figures don't accumulate -> no OOM across the suite / many Updates.

from .api import (
    SHAARPResult,
    run_fresnel_sweep,
    run_maker_fringes,
    run_ml_numeric,
    run_ml_partial_analytical,
    run_si_full_analytical,
    run_si_numeric,
)
from .config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry
from .interactive import default_interactive_material, default_interactive_system

# The Functionality modes each original GUI exposes (User Guide is a static help page, handled in
# the widget layer, so the compute cores cover the simulation/analytical modes).
SI_FUNCTIONALITIES = ("SHG Simulation", "Partial Analytical", "Full Analytical")
ML_FUNCTIONALITIES = ("SHG Simulation", "Maker Fringes", "Fresnel Coefficients", "Partial Analytical")

# Functionality dropdown labels (what the user sees), mapped below to the canonical compute modes
# above. the dropdown now lists COMPUTE MODES ONLY -- the view-only entries
# the original GUI carried here (User Guide / Set Material Properties / 2D|3D Schematics) were pure
# redundancy: the Guide is on the Help menu + startup tab, the schematic is a persistent banner, and
# the crystal-axes view moved next to the orientation inputs where it is actually used. Removing them
# stops the dropdown from mixing "run physics" with "show a static picture".
# the ML list is exactly FOUR compute modes. "RA Scan (sample azimuth)" is
# gone -- rotational anisotropy is SHG Simulation with the sample turning and the polarizer/analyzer
# held fixed, so it is a TOGGLE in Polarimetry Settings, exactly as the original .ml GUI has it
# (samplerotationcontrol, "Rotate Sample"/"Fix Sample", forcing RotatePolarizer=RotateAnalyzer=False).
SI_FUNCTIONALITY_DISPLAY = (
    "SHG Simulation", "Partial Analytical Expression", "Full Analytical Expression",
)
ML_FUNCTIONALITY_DISPLAY = (
    "SHG Simulation", "Maker Fringes", "Fresnel Coefficients", "Partial Analytical Expressions",
)
FUNCTIONALITY_CANON = {
    "SHG Simulation": "SHG Simulation",
    "Maker Fringes": "Maker Fringes",
    "Fresnel Coefficients": "Fresnel Coefficients",
    "Partial Analytical Expression": "Partial Analytical",
    "Partial Analytical Expressions": "Partial Analytical",
    "Full Analytical Expression": "Full Analytical",
    # legacy canonical labels accepted verbatim too
    "Partial Analytical": "Partial Analytical",
    "Full Analytical": "Full Analytical",
    # view-only (no physics compute) -> None
    "User Guide": None,
    "Set Material Properties": None,
    "2D Schematics": None,
    "3D Schematics": None,
}

# The SHAARP.ml Assumptions panel: which multiple-reflection treatment the multilayer solve uses.
# Keys are the ORIGINAL GUI's faithful labels; values are the validated mrassumption codes.
ML_ASSUMPTIONS = {
    "Full Multiple Reflections (FMR)": 0,
    "Jerphagnon & Kurtz Assumption (No MR)": 1,
    "Herman & Hayden Assumption (MR only for 2ω Homo Waves)": 2,
}

# FMR sub-options (the original's winhAssumption 0/1/2): whether to include backward waves and,
# if so, standing waves. Maps to the multilayer solver's inhomogeneous_source_policy. ALL THREE are
# validated against live SHAARP.ml: forward_only by the 739-angle Maker grid, and forward_backward
# (winh=1) / all (winh=2) by tests/test_fmr_backward_reference.py (quartz+Au docs, agreement ~1e-9).
# Keys are the original GUI's faithful labels ("Assumptions for 2omega Inhomogeneous Waves").
FMR_SUBMODES = {
    "Forward waves only": "forward_only",
    "Forward + Backward waves": "forward_backward",
    "Forward + Backward + Standing waves": "all",
}

# Back-compat aliases: the earlier short labels are still accepted by compute_ml_gui_result (and by
# the maker examples/tests), so renaming the GUI labels to the faithful originals breaks nothing.
_ASSUMPTION_ALIASES = {
    "Full (FMR)": "Full Multiple Reflections (FMR)",
    "JK": "Jerphagnon & Kurtz Assumption (No MR)",
    "HH": "Herman & Hayden Assumption (MR only for 2ω Homo Waves)",
}
_FMR_SUBMODE_ALIASES = {
    "No backward waves": "Forward waves only",
    "Backward waves": "Forward + Backward waves",
    "Backward + standing waves": "Forward + Backward + Standing waves",
}

# .ml multilayer-system presets — the PAPER's demonstrated heterostructures at their published
# wavelengths (npj Comput. Mater. 10, 64 (2024); case-study fidelity audit). Factories,
# lazily resolved.
def _quartz_au_docs_system():
    # Fig 4: Z-cut quartz (121.2 um) + 13.9 nm backside Au at 800 nm — the docs' canonical Maker
    # example, VALIDATED vs live SHAARP.ml to 3.6e-9 (tests/test_quartz_au_docs_reference.py);
    # lazy import keeps shaarp's module load light. The case study lives inside the package, so
    # this preset works from a pip install with no checkout.
    from shaarp.quartz_au_docs_case import build_quartz_au_system

    return build_quartz_au_system()


def _fig6_zno_pt_al2o3_system():
    # Fig 6: ZnO (159 nm) // Pt (200 nm) // Al2O3 substrate at 1550 nm; only ZnO is SHG-active.
    # Same stack as benchmarks.paper_cases.ml_fig6_system, kept inline so shaarp/ never imports
    # benchmarks at runtime.
    from .casestudy_materials import build_casestudy_material
    from .config import Layer, MultilayerSystem

    lam = 1.55
    air = build_casestudy_material("Air", wavelength_um=lam)
    zno = build_casestudy_material("ZnO (001)", wavelength_um=lam)
    pt = build_casestudy_material("Pt (111) (1550 nm)", wavelength_um=lam)
    al2o3 = build_casestudy_material("Al2O3 (0001) (1550 nm)", wavelength_um=lam)
    return MultilayerSystem(wavelength_um=lam, layers=[
        Layer(name="air", material=air, thickness_um=None, shg_active=False),
        Layer(name="ZnO", material=zno, thickness_um=0.159, shg_active=True),
        Layer(name="Pt", material=pt, thickness_um=0.200, shg_active=False),
        Layer(name="Al₂O₃ substrate", material=al2o3, thickness_um=None, shg_active=False),
    ])


def _fig7_lno_quartz_system():
    # Fig 7 case (2): x-cut LiNbO3 (50 um) // z-cut quartz (35 um) two-crystal interference stack
    # at 1550 nm, theta_i = 0.5 deg. Same stack as benchmarks.paper_cases.ml_fig7_case(2), kept
    # inline for the same no-benchmarks-import reason.
    from .casestudy_materials import build_casestudy_material
    from .config import Layer, MultilayerSystem, Polarimetry

    lam = 1.55
    air = build_casestudy_material("Air", wavelength_um=lam)
    lno = build_casestudy_material("LiNbO3 x-cut (1550 nm)", wavelength_um=lam)
    quartz = build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=lam)
    return MultilayerSystem(
        wavelength_um=lam,
        layers=[
            Layer(name="air", material=air, thickness_um=None, shg_active=False),
            Layer(name="LiNbO₃ (21̄1̄0)", material=lno, thickness_um=50.0, shg_active=True),
            Layer(name="quartz (001)", material=quartz, thickness_um=35.0, shg_active=True),
            Layer(name="air out", material=air, thickness_um=None, shg_active=False),
        ],
        polarimetry=Polarimetry(theta_deg=0.5, phi_deg=0.0),
    )


ML_SYSTEM_PRESETS = {
    "Quartz + Au (Fig 4, 800 nm)": _quartz_au_docs_system,
    "ZnO / Pt / Al2O3 (Fig 6, 1550 nm)": _fig6_zno_pt_al2o3_system,
    "LiNbO3 / Quartz (Fig 7, 1550 nm)": _fig7_lno_quartz_system,
}

# Back-compat: the pre-audit preset names (older sessions, notebooks, user scripts) still resolve.
_ML_PRESET_ALIASES = {
    "Quartz + Au (docs, validated)": "Quartz + Au (Fig 4, 800 nm)",
}


_EPS_THETA_DEG = 1.0e-3


def _desingularize_theta_deg(theta_deg: float) -> float:
    """Nudge exactly-degenerate incidence angles off the singular points of the anisotropic eigenmode
    index. At normal incidence (theta_i == 0) and grazing (|theta_i| == 90) the transverse wavevector
    is zero, so the eigenmode index formula (si_compat: sqrt((v +/- sqrt(v^2 - w)) / w)) divides by
    zero -> NaN -> singular boundary matrix (NonInvertibleMatrixError). Shifting by 1e-3 deg gives the
    continuous normal/grazing-incidence LIMIT to ~1e-6 while keeping the solve well-conditioned.
    (Found: the LiNbO3 case study at theta_i = 0 crashed the SI SHG Update.)"""
    t = float(theta_deg)
    if abs(t) < _EPS_THETA_DEG:
        return _EPS_THETA_DEG
    if abs(abs(t) - 90.0) < _EPS_THETA_DEG:
        return (90.0 - _EPS_THETA_DEG) if t >= 0.0 else -(90.0 - _EPS_THETA_DEG)
    return t


# cache for the fully-symbolic SI closed forms (theta/index-independent; keyed by point group +
# iso/uniaxial pattern) -- see compute_si_gui_result
_FULL_SYMBOLIC_CACHE: dict = {}


def _epsilon_lab_of(material, *, omega: bool) -> np.ndarray:
    """Lab-frame 3x3 dielectric tensor of a Material (crystal tensor rotated by its orientation)."""
    from .tensors import rotate_rank2_crystal_to_lab

    tensor = material.eps_w() if omega else material.eps_2w()
    return np.asarray(rotate_rank2_crystal_to_lab(tensor, material.orientation.rotation_matrix()),
                      dtype=complex)


def compute_si_gui_result(
    functionality: str,
    *,
    point_group: str = "-43m",
    theta_deg: float = 45.0,
    material=None,
):
    """Pure (no-widget) compute for the SHAARP.si tab. Returns the SHAARPResult the panel displays.

    ``functionality`` is one of :data:`SI_FUNCTIONALITIES`. "SHG Simulation" runs the validated
    numeric single-interface compat workflow; "Partial Analytical" / "Full Analytical" run the
    validated full-analytical reflected-SHG polarimetry closed form (symbolic in phi / d_ij).
    """

    if functionality not in SI_FUNCTIONALITIES:
        raise ValueError(f"SHAARP.si functionality must be one of {SI_FUNCTIONALITIES}, got {functionality!r}")
    mat = material if material is not None else default_interactive_material()
    if functionality == "SHG Simulation":
        return run_si_numeric(
            mat,
            {
                "workflow": "shaarp_si_compat",
                # Do NOT desingularize here: the shaarp_si_compat workflow has its OWN
                # normal_incidence_2omega_branch_policy that handles EXACTLY theta_i=0; nudging to a
                # tiny epsilon instead lands in the ill-conditioned near-normal zone where its angle
                # root-solve fails to converge. (Only the closed-form eigenmode paths need the nudge.)
                "polarimetry": Polarimetry(theta_deg=float(theta_deg)),
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )
    # Partial / Full analytical -> the validated closed-form polarimetry (symbolic in phi, d_ij).
    # MATERIAL-FIRST point group: when a case-study material is
    # selected, its OWN crystal class must drive the closed form. Previously the panel's point-group
    # combo (default -43m) won here, so e.g. LiNbO3 + "Full Analytical" silently returned the -43m
    # d14 closed form -- the wrong crystal class, rendered without any error. The combo is only the
    # fallback for custom (material-less) input.
    pg_effective = point_group
    if material is not None:
        mat_pg = getattr(getattr(material, "structure", None), "point_group", None)
        if mat_pg:
            pg_effective = mat_pg
    if pg_effective not in SHG_POINT_GROUPS:
        # Centrosymmetric / isotropic case material (Air & Au = ∞∞m, Pt & Blank-linear = m3m,
        # Al₂O₃ = 6/mmm): SHG is symmetry-forbidden and no closed form exists -- return the same
        # graceful zero result the ML partial-analytical branch uses instead of crashing.
        note = f"0   (SHG symmetry-forbidden: centrosymmetric/isotropic point group {pg_effective})"
        return SHAARPResult(
            kind="si_full_analytical_polarimetry",
            numeric={},
            stages={"workflow": "polarimetry", "point_group": pg_effective,
                    "reflected_p_2omega": note, "reflected_s_2omega": note},
        )

    # --- assemble the closed-form case FROM THE ACTUAL INPUTS (release-audit /) ---
    # Previously both analytical modes ran with the runner's GENERIC default indices (2.2/2.5/...),
    # so the numeric prefactors matched neither the panel's ε nor the selected material, and "Full"
    # was indistinguishable from "Partial". Now:
    # * Partial Analytical = symbolic (φ, ψ, d_ij); θ and the PANEL/material ε substituted.
    # * Full Analytical = additionally symbolic in θ_i and the refractive indices -- for
    # identity-oriented isotropic/uniaxial classes (the tractable closed-form scope; a fully
    # symbolic biaxial/rotated form runs into minutes-to-hours of CAS time, per the pinned
    # practical boundary). Outside that scope it falls back to the substituted form and SAYS SO.
    import sympy as sp
    from .symbolic import d_voigt_symbolic

    case: dict = {"point_group": pg_effective}
    extra_symbols: dict[str, str] = {}
    eps_w_t = eps_2w_t = None
    rot_identity = True
    pure_rz = True
    rot_fingerprint = b"identity"
    if material is not None:
        # LAB-frame tensors (crystal eps rotated by the orientation) -- for x-/y-cut and Miller-
        # oriented materials the crystal-frame diagonal is NOT the lab diagonal
        eps_w_m = _epsilon_lab_of(material, omega=True)
        eps_2w_m = _epsilon_lab_of(material, omega=False)
        off = max(float(np.max(np.abs(eps_w_m - np.diag(np.diag(eps_w_m))))),
                  float(np.max(np.abs(eps_2w_m - np.diag(np.diag(eps_2w_m))))))
        if off > 1e-9 * max(1.0, float(np.max(np.abs(eps_w_m)))):
            # Graceful, informative result instead of an error dialog (same pattern as the
            # symmetry-forbidden case): the principal-aligned closed form genuinely does not
            # apply to an orientation with off-diagonal lab-frame ε (e.g. TaAs (112)).
            note = ("(no closed form: this material/orientation has OFF-DIAGONAL lab-frame ε, "
                    "outside the principal-aligned analytical scope — use 'SHG Simulation' for "
                    "the numeric polarimetry instead)")
            return SHAARPResult(
                kind="si_full_analytical_polarimetry",
                numeric={},
                stages={"workflow": "polarimetry", "point_group": pg_effective,
                        "reflected_p_2omega": note, "reflected_s_2omega": note,
                        "symbols": {"note": "closed form unavailable for this orientation"}},
            )
        # COMPLEX principal values, degeneracy-snapped -- the SAME
        # construction as the plotted curve's (si_figure_kwargs_from_material), so the analytical
        # expression and the polar plot share one truth (referee gate A enforces their agreement).
        eps_w_t = _snap_degenerate_triple(complex(eps_w_m[i, i]) for i in range(3))
        eps_2w_t = _snap_degenerate_triple(complex(eps_2w_m[i, i]) for i in range(3))
        case["eps_omega_principal"] = eps_w_t
        case["eps_2omega_principal"] = eps_2w_t
        rot = np.asarray(material.orientation.rotation_matrix(), dtype=float)
        rot_identity = bool(np.max(np.abs(rot - np.eye(3))) < 1e-9)
        # A PURE rotation about L3 keeps uniaxial/isotropic eps principal-aligned and keeps the
        # Rz-rotated symbolic d CAS-tractable (~1 min, cached) -- so it remains eligible for the
        # fully-symbolic Full mode (covers the flagship z-cut LiNbO3, whose crystal-physics
        # convention is an Rz rotation). General rotations (x-cut, (111), (112)) are not.
        pure_rz = bool(abs(rot[2, 2] - 1.0) < 1e-9
                       and float(np.max(np.abs([rot[0, 2], rot[1, 2], rot[2, 0], rot[2, 1]]))) < 1e-9)
        rot_fingerprint = rot.round(12).tobytes()
        if not rot_identity:
            from .symbolic import rotate_d_voigt_symbolic
            # CONVENTION (verified numerically): rotate_d_voigt_symbolic(d, A) equals
            # the validated numeric rotate_d_voigt_crystal_to_lab(d, A.T) -- its argument is the
            # TRANSPOSE of orientation.rotation_matrix() (existing callers pass rot_z_T). Agreement
            # with the numeric path checked to 3e-14 on GaAs(111).
            # EXACT rotation entries (FA-1: "everything are symbolic"): the
            # orientation is geometry, so its entries belong in the closed form as exact radicals
            # (1/sqrt(3), sqrt(2)/2, ...) the way his Mathematica prints them -- NOT as
            # 0.5773502691896258. nsimplify recovers the radical; anything it cannot recognise
            # stays a float (rational=False keeps e.g. sqrt(6)/6 exact rather than a fraction).
            rot_exact = sp.Matrix(rot.T.tolist()).applyfunc(
                lambda v: sp.nsimplify(v, rational=False, tolerance=1e-12))
            case["d_voigt_lab"] = rotate_d_voigt_symbolic(d_voigt_symbolic(pg_effective), rot_exact)
            extra_symbols["orientation"] = "d rotated crystal→lab for this material's orientation"

    # SCOPE (FA-1, -- : "you are not actually showing full analytical … in my full
    # analytical Mathematica, everything are symbolic"). The mode had been substituting numbers for
    # 7 of the 8 palette cases. The lab eps is already guaranteed DIAGONAL here (a non-diagonal one
    # returned the "no closed form" result above) and the orientation only rotates d, which is
    # rotated SYMBOLICALLY -- so every case that reaches this point is fully symbolic, biaxial
    # included. Measured per-case (subprocess probe, symbolic indices + theta):
    # expanded emitter: LiNbO3 MTI x-cut 9.9 s, LiOsO3 2.6 s, KTP (100) TIMEOUT >420 s, LBO ditto
    # LAYERED emitter: all four 0.2-0.6 s (KTP 0.4 s) -- naming the eigenmodes instead of
    # substituting their radicals removes the Booker-quartic explosion.
    # The eps PATTERN below now only chooses SYMBOL NAMES (n_o/n_e vs n_x/n_y/n_z), not scope.
    def _eps_pattern(t):
        """Classify a lab-frame principal triple: 'isotropic' | 'axis||z|x|y' (uniaxial) | 'biaxial'."""
        if t is None:
            return "isotropic"
        eq = lambda u, v: abs(u - v) <= 1e-9 * max(abs(u), 1.0)  # noqa: E731
        xy, yz, xz = eq(t[0], t[1]), eq(t[1], t[2]), eq(t[0], t[2])
        if xy and yz:
            return "isotropic"
        if xy:
            return "axis||z"     # ordinary in x,y -> optic axis along z (the classic z-cut)
        if yz:
            return "axis||x"     # ordinary in y,z -> optic axis along x (e.g. LiNbO3 MTI x-cut)
        if xz:
            return "axis||y"     # ordinary in x,z -> optic axis along y (e.g. LiOsO3)
        return "biaxial"

    def _sym_triple(pattern, n_o, n_e, n_x=None, n_y=None, n_z=None):
        """The symbolic principal triple for a pattern -- the ordinary index sits on the two
        degenerate axes and the extraordinary one on the optic axis. BIAXIAL gets three distinct
        principal indices (no o/e split exists for it)."""
        return {"isotropic": (n_o**2, n_o**2, n_o**2),
                "axis||z": (n_o**2, n_o**2, n_e**2),
                "axis||x": (n_e**2, n_o**2, n_o**2),
                "axis||y": (n_o**2, n_e**2, n_o**2),
                "biaxial": (n_x**2, n_y**2, n_z**2)}[pattern]

    pat_w, pat_2 = _eps_pattern(eps_w_t), _eps_pattern(eps_2w_t)
    want_full = functionality == "Full Analytical"
    # EVERY principal-aligned lab eps is now fully symbolic, biaxial included: the layered
    # named-intermediate emitter never substitutes an eigenmode radical inline, so the Booker
    # quartic no longer explodes (measured on this palette: KTP (100) 420 s TIMEOUT -> 0.4 s,
    # LBO likewise). A NON-diagonal lab eps (a rotated crystal, e.g. TaAs (112)) still has no
    # principal-aligned closed form and returned the substituted result above.
    full_ok = want_full
    if full_ok:
        # The fully-symbolic form depends ONLY on (point group, iso/uniaxial pattern, orientation)
        # -- theta and all indices are symbols -- so cache it: the CAS solve costs seconds-to-a-
        # minute (~10 s identity 3m, ~57 s Rz-rotated 3m) and would otherwise be paid again on
        # every Update click / sweep cell.
        cache_key = (pg_effective, pat_w, pat_2, rot_fingerprint)
        cached = _FULL_SYMBOLIC_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # fully symbolic θ_i and indices; the ordinary/extraordinary symbols are placed on the axes
        # the LAB eps pattern actually names (isotropic collapses n_e onto n_o).
        case["incident_theta_rad"] = sp.Symbol("theta_i", real=True, positive=True)
        n_w = sp.Symbol("n_omega", positive=True)
        n_we = sp.Symbol("n_omega_e", positive=True)
        n_2 = sp.Symbol("n_2omega", positive=True)
        n_2e = sp.Symbol("n_2omega_e", positive=True)
        bx_w = [sp.Symbol(f"n_omega_{a}", positive=True) for a in "xyz"]
        bx_2 = [sp.Symbol(f"n_2omega_{a}", positive=True) for a in "xyz"]
        case["eps_omega_principal"] = _sym_triple(pat_w, n_w, n_we, *bx_w)
        case["eps_2omega_principal"] = _sym_triple(pat_2, n_2, n_2e, *bx_2)
        iso_w, iso_2 = pat_w == "isotropic", pat_2 == "isotropic"
        bi_w, bi_2 = pat_w == "biaxial", pat_2 == "biaxial"
        extra_symbols["incidence"] = "theta_i (symbolic)"
        if bi_w or bi_2:
            extra_symbols["indices"] = ("n_omega_x/y/z, n_2omega_x/y/z (symbolic)")
            extra_symbols["optic class"] = "biaxial, principal axes along the lab frame"
        elif iso_w and iso_2:
            extra_symbols["indices"] = "n_omega, n_2omega (symbolic)"
        else:
            extra_symbols["indices"] = "n_omega/n_omega_e, n_2omega/n_2omega_e (symbolic)"
            extra_symbols["optic axis"] = f"uniaxial, {pat_w if not iso_w else pat_2} (lab frame)"
    else:
        case["incident_theta_rad"] = float(np.deg2rad(_desingularize_theta_deg(theta_deg)))
        extra_symbols["incidence"] = f"theta_i = {float(theta_deg):g} deg (substituted)"
        extra_symbols["indices"] = ("from the selected material/panel ε (substituted)"
                                    if eps_w_t is not None else "runner defaults (substituted)")
        # NOTE (FA-1 Stage 2): this branch is now "Partial Analytical" ONLY. Every Full Analytical
        # case that reaches here is fully symbolic, so there is no longer a "Full Analytical fell
        # back to substituted numbers" state to explain. The one genuinely unavailable case -- a
        # NON-DIAGONAL lab ε (a rotated crystal such as TaAs (112)) -- returned the explicit
        # "no closed form" result far above, before any of this. That is the right answer there:
        # forcing the principal-aligned treatment on TaAs (max |off-diagonal| = 3.2) disagrees with
        # the validated numeric solver by 6e+00 relative, i.e. it would be WRONG, not just missing.

    # LAYERED belongs to FULL ANALYTICAL only -- that is the mode that reproduces SHAARP.si's
    # printed derivation (named intermediates + their defining equations). PARTIAL ANALYTICAL's
    # contract is the opposite: one FLATTENED expression in phi and d_ij, with theta and the
    # indices substituted, that a user can evaluate or fit directly. Layering that one would make
    # the curve expression unevaluatable on its own (caught by the expression-matches-plot gate).
    result = run_si_full_analytical(
        case, {"workflow": "polarimetry", "simplify": False, "layered": bool(full_ok)})
    try:  # surface the substitution/symbol provenance in the '# symbols:' head line
        result.stages.setdefault("symbols", {}).update(extra_symbols)
    except Exception:
        pass
    if full_ok:
        _FULL_SYMBOLIC_CACHE[cache_key] = result
    return result


def default_ml_system():
    """A REAL-crystal default multilayer system for the .ml tab: air / thin LiNbO3 z-cut film /
    substrate -- a physical Maker-fringe sample (unlike the contrived point-group-1 demo material,
    whose artificial complex tensors produce spurious isolated singularities). Thin film -> the
    Maker fringes stay in a readable range."""

    from dataclasses import replace as _replace

    from .layer_presets import build_stack_preset

    sysm = build_stack_preset("single_film_linbo3")
    layers = list(sysm.layers)
    layers[1] = _replace(layers[1], thickness_um=10.0)  # thick enough to show several Maker fringes
    return _replace(sysm, layers=layers)


def resolve_ml_system_preset(system_preset: str | None):
    """Resolve a .ml case-study preset name (a key of :data:`ML_SYSTEM_PRESETS`) to a system."""
    if system_preset is None or system_preset == "LiNbO3 film (10 um)":
        # None (and the retired pre-audit preset name) -> the API's default single-film system
        return default_ml_system()
    system_preset = _ML_PRESET_ALIASES.get(system_preset, system_preset)
    if system_preset not in ML_SYSTEM_PRESETS:
        raise ValueError(f"system_preset must be one of {tuple(ML_SYSTEM_PRESETS)}, got {system_preset!r}")
    return ML_SYSTEM_PRESETS[system_preset]()


class PresetStore:
    """The original GUIs' 'Material Properties Preset Values' panel: 4 save/recall slots with a
    user label each (session-scoped, like the original -- presets do not persist across sessions).
    Pure/headless-testable: stores plain snapshot dicts of the material-entry values."""

    def __init__(self, n_slots: int = 4):
        self._slots: list[dict | None] = [None] * int(n_slots)
        self._labels: list[str] = [""] * int(n_slots)

    @property
    def n_slots(self) -> int:
        return len(self._slots)

    def save(self, slot: int, snapshot: dict, label: str = "") -> None:
        self._check(slot)
        self._slots[slot] = dict(snapshot)
        self._labels[slot] = str(label)

    def recall(self, slot: int) -> dict | None:
        self._check(slot)
        snap = self._slots[slot]
        return None if snap is None else dict(snap)

    def label(self, slot: int) -> str:
        self._check(slot)
        return self._labels[slot]

    def filled(self, slot: int) -> bool:
        self._check(slot)
        return self._slots[slot] is not None

    def clear(self) -> None:
        self._slots = [None] * len(self._slots)
        self._labels = [""] * len(self._slots)

    def _check(self, slot: int) -> None:
        if not (0 <= int(slot) < len(self._slots)):
            raise ValueError(f"slot must be in [0, {len(self._slots) - 1}], got {slot}")


def export_result_payload(result) -> dict:
    """JSON-serializable payload of a SHAARPResult's numeric data (the original GUI's 'Copy'
    button analog: it copied the plotted lists, e.g. listMFpara). Complex arrays are exported as
    {re, im} pairs; the kind and validation status ride along for provenance."""

    def _ser(v):
        a = np.asarray(v)
        if np.iscomplexobj(a):
            return {"re": np.real(a).tolist(), "im": np.imag(a).tolist()}
        return a.tolist()

    return {
        "kind": str(result.kind),
        "validation_status": str(result.validation.status),
        "numeric": {k: _ser(v) for k, v in result.numeric.items()},
    }


ANALYTICAL_EXPRESSION_KEYS = ("reflected_s_2omega", "reflected_p_2omega", "analyzed_intensity")

# Friendly labels for the per-expression analytical output blocks (the original's separate copyable
# I(phi,psi) expressions). Ordered p then s then analyzed, reflected then transmitted.
# NOTE the E-vs-I distinction (release-audit): the reflected/transmitted stage
# expressions are FIELD AMPLITUDES (signed, linear in d_ij) -- labeling them I (intensity) was
# physically wrong on its face (an intensity cannot be negative). Only "analyzed_intensity" is a
# true intensity (|...|^2).
_ANALYTICAL_ITEM_LABELS = {
    "reflected_p_2omega": "E_p^{2ω}  —  reflected p (parallel) SHG field amplitude, closed form",
    "reflected_s_2omega": "E_s^{2ω}  —  reflected s (perpendicular) SHG field amplitude, closed form",
    "analyzed_intensity": "I^{2ω}(φ,ψ) = |E·â(ψ)|²  —  analyzed SHG intensity (the polarimetry fit expression)",
    "transmitted_p_2omega": "E_p^{T,2ω}  —  transmitted p SHG field amplitude, closed form",
    "transmitted_s_2omega": "E_s^{T,2ω}  —  transmitted s SHG field amplitude, closed form",
}
_ANALYTICAL_ITEM_ORDER = (
    "reflected_p_2omega", "reflected_s_2omega", "analyzed_intensity",
    "transmitted_p_2omega", "transmitted_s_2omega",
)

# the "where:" block that defines the layered chain's named intermediates -- a multi-line
# "name = expression" listing, NOT a single sympifiable expression, so exporters treat it apart
_ANALYTICAL_DEFINITIONS_LABEL = "where — named intermediates (each defined below, in derivation order)"


def analytical_expression_items(result) -> list[tuple[str, str]]:
    """Ordered ``(label, expression-text)`` blocks for the analytical output panel -- the original
    GUI's separate, individually-copyable closed-form expressions (the I(phi,psi) families) plus a
    leading 'symbols' note (point group + which variables are symbolic). Raises ValueError for results
    that carry no analytical expressions (e.g. numeric runs). Pure/headless-testable."""

    stages = getattr(result, "stages", None) or {}
    exprs = [(_ANALYTICAL_ITEM_LABELS.get(k, k), str(stages[k]))
             for k in _ANALYTICAL_ITEM_ORDER if k in stages and stages[k] not in (None, "None")]
    if not exprs:
        raise ValueError(f"result kind {getattr(result, 'kind', '?')!r} carries no analytical expressions")
    items: list[tuple[str, str]] = []
    head = []
    pg = stages.get("point_group")
    if pg:
        head.append(f"point group {pg}")
    symbols = stages.get("symbols") or {}
    described = ", ".join(f"{k} = {v}" for k, v in symbols.items() if v not in (None, "None"))
    if described:
        head.append(described)
    if head:
        items.append(("symbols", "; ".join(head)))
    items.extend(exprs)
    # The LAYERED chain's definitions travel WITH the expressions (FA-1 Stage 2). Without them a
    # copied amplitude is unusable: it is written in named intermediates (CT_ee_2w_L1, kT_o_w_z...)
    # and the d_ij live inside those definitions, so the headline expression alone can no longer be
    # fitted to extract d -- which is the whole point of this output.
    definitions = str(stages.get("deriv_0_definitions", "")).strip()
    if definitions:
        items.append((_ANALYTICAL_DEFINITIONS_LABEL, definitions))
    return items


def analytical_expression_text(result) -> str:
    """Plain-text rendering of an analytical result's closed-form expressions -- the originals'
    copyable analytical output (SHAARP.si's headline feature: the polarimetry closed form an
    experimentalist fits to extract d_ijk). Works for both the SI full-analytical and the ML
    partial-analytical polarimetry results (their expression strings live in result.stages).
    Raises ValueError for results that carry no analytical expressions (e.g. numeric runs).
    Pure/headless-testable."""

    items = analytical_expression_items(result)  # raises ValueError for numeric results
    parts = [f"# {getattr(result, 'kind', '')}"]
    for label, text in items:
        if label == "symbols":
            parts.append(f"# symbols: {text}")
        else:
            parts.append(f"{'='*60}\n{label}\n{'='*60}\n{text}")
    return "\n\n".join(parts)


_ANALYTICAL_ITEM_LABELS_HTML = {
    "reflected_p_2omega": "E<sub>p</sub><sup>2ω</sup>&nbsp; — &nbsp;reflected p (parallel) SHG field amplitude, closed form",
    "reflected_s_2omega": "E<sub>s</sub><sup>2ω</sup>&nbsp; — &nbsp;reflected s (perpendicular) SHG field amplitude, closed form",
    "analyzed_intensity": "I<sup>2ω</sup>(φ,ψ) = |E·â(ψ)|<sup>2</sup>&nbsp; — &nbsp;analyzed SHG intensity (the polarimetry fit expression)",
    "transmitted_p_2omega": "E<sub>p</sub><sup>T,2ω</sup>&nbsp; — &nbsp;transmitted p SHG field amplitude, closed form",
    "transmitted_s_2omega": "E<sub>s</sub><sup>T,2ω</sup>&nbsp; — &nbsp;transmitted s SHG field amplitude, closed form",
}

# raw sympy token -> typeset HTML, longest-first so n_2omega_e wins over n_2omega over n_omega
_MATH_TOKEN_HTML = (
    ("n_2omega_e", "n<sub>2ω,e</sub>"),
    ("n_2omega", "n<sub>2ω</sub>"),
    ("n_omega_e", "n<sub>ω,e</sub>"),
    ("n_omega", "n<sub>ω</sub>"),
    ("theta_i", "θ<sub>i</sub>"),
    ("theta", "θ"),
    ("phi", "φ"),
    ("psi", "ψ"),
    ("omega", "ω"),
)


def math_display_html(expr_text: str) -> str:
    """Typeset one raw sympy expression string as inline HTML with REAL super/subscripts --
    the notation the original ♯SHAARP prints (release-audit the panel showed
    ``n_omega**2``/``theta_i``/``d14`` where the original typesets n_ω², θᵢ, d₁₄). DISPLAY layer
    only: Copy and the .txt export keep the machine-readable sympy text unchanged.

    Conversions: known symbol names -> Greek + <sub>; ``d14``-style Voigt components ->
    d<sub>14</sub>; ``**n`` and ``**(expr)`` -> <sup>; ``sqrt(`` -> ``√(``; the imaginary unit
    ``I`` -> ``i``; remaining ``*`` multiplication -> ``·``. Pure/headless-testable."""

    import html as _html
    import re as _re

    s = _html.escape(expr_text)
    # DISPLAY-precision floats (~6 significant digits, like the original's typeset output);
    # the raw/Copy/export layer keeps sympy's full precision untouched.
    s = _re.sub(r"\d+\.\d+(?:[eE][+-]?\d+)?",
                lambda m: f"{float(m.group(0)):.6g}", s)
    # LAYERED-chain intermediates (FA-1 Stage 2): the sympify-safe identifiers kT_o_w_z /
    # ET_e_2w_L1 / CT_ee_2w_L3 / thetaT_o_w are typeset back into SHAARP.si's printed notation
    # k^(T,o,ω)_z etc. Done BEFORE the generic token pass so 'theta'/'omega' inside them are not
    # rewritten piecemeal. Display-only: Copy/export keep the machine-readable names.
    def _intermediate(m):
        head, tag, sub = m.group(1), m.group(2).replace("_", ","), m.group(3)
        head = {"theta": "θ"}.get(head, head)
        tag = tag.replace("2w", "2ω").replace(",w", ",ω")
        sub = f"<sub>{sub.lstrip('_')}</sub>" if sub else ""
        return f"{head}<sup>({tag})</sup>{sub}"

    s = _re.sub(r"\b(k|E|C|theta)((?:T|R)_(?:o|e|s|p|ee|oo|eo)_2?w)((?:_(?:x|y|z|L[123]))?)\b",
                _intermediate, s)
    for tok, rep in _MATH_TOKEN_HTML:
        s = _re.sub(rf"\b{tok}\b", rep, s)
    s = _re.sub(r"\bd(\d\d)\b", r"d<sub>\1</sub>", s)
    s = _re.sub(r"\bI\b", "i", s)
    s = s.replace("sqrt(", "√(")
    s = _re.sub(r"\*\*(\d+(?:\.\d+)?)", r"<sup>\1</sup>", s)          # x**2 -> x²
    s = _re.sub(r"\*\*\(([^()]*)\)", r"<sup>(\1)</sup>", s)           # x**(a/b) -> x^(a/b)
    s = s.replace("*", "·")
    return s


# sympy symbol name -> Mathematica-safe name (underscores are PATTERN syntax in Wolfram
# Language, so theta_i etc. must be renamed; the \[...] forms paste as the original
# notebook's Greek symbols). Longest-first so n_2omega_e wins over n_2omega.
_MATHEMATICA_SYMBOL_NAMES = (
    ("n_2omega_e", "ne2\\[Omega]"),
    ("n_2omega", "n2\\[Omega]"),
    ("n_omega_e", "ne\\[Omega]"),
    ("n_omega", "n\\[Omega]"),
    ("theta_i", "\\[Theta]i"),
    ("phi", "\\[CurlyPhi]"),
    ("psi", "\\[Psi]"),
)


def analytical_expression_mathematica(result) -> str:
    """Wolfram-Language rendering of an analytical result's closed forms (the author's request,
    'I need mathematica format for full analytical and partial analytical').

    Each expression string is re-parsed with sympy and printed via
    :func:`sympy.printing.mathematica.mathematica_code` (``Sqrt[..]``, ``Exp[..]``, ``^``
    powers), with symbols renamed to Mathematica-safe names (``theta_i`` -> ``\\[Theta]i``
    etc. -- underscores are Blank patterns in WL and must not appear in symbol names).
    Numeric fidelity of the conversion is wolframscript-verified.
    Raises ValueError for results that carry no analytical expressions."""

    import sympy as sp
    from sympy.printing.mathematica import mathematica_code

    items = analytical_expression_items(result)  # raises ValueError for numeric results
    renames = {old: sp.Symbol(new) for old, new in _MATHEMATICA_SYMBOL_NAMES}

    def _wl(expr):
        """Rename every symbol to a Wolfram-safe name: the explicit map first, then a general
        underscore strip. ANY residual underscore would make the symbol a Blank PATTERN in WL
        (``CT_ee_2w_L1`` parses as ``CT`` matching a blank), silently producing a wrong-but-valid
        expression -- so the layered chain's named intermediates must be sanitized too."""
        expr = expr.subs({sp.Symbol(old): new_sym for old, new_sym in renames.items()})
        leftover = {s: sp.Symbol(str(s).replace("_", "")) for s in expr.free_symbols
                    if "_" in str(s)}
        return expr.subs(leftover) if leftover else expr

    # In Wolfram the named intermediates must be ASSIGNED BEFORE the expressions that use them,
    # so the "where:" block is hoisted here (the plain-text rendering keeps the reader-friendly
    # "expression ... where: ..." order instead).
    items = sorted(items, key=lambda it: 0 if it[0] == "symbols"
                   else (1 if it[0] == _ANALYTICAL_DEFINITIONS_LABEL else 2))
    parts = [f"(* {getattr(result, 'kind', '')} *)"]
    for label, text in items:
        if label == "symbols":
            parts.append(f"(* symbols: {text} *)")
            continue
        if label == _ANALYTICAL_DEFINITIONS_LABEL:
            # emit the chain as real WL assignments so the exported expression is self-contained
            lines = []
            for line in text.splitlines():
                name, _, rhs = line.partition(" = ")
                if not rhs:
                    continue
                lines.append(f"{name.replace('_', '')} = {mathematica_code(_wl(sp.sympify(rhs)))};")
            parts.append("(* " + label + " *)\n" + "\n".join(lines))
            continue
        parts.append(f"(* {label} *)\n{mathematica_code(_wl(sp.sympify(text)))}")
    return "\n\n".join(parts)


# ordered step-by-step derivation blocks: the published eq 29-32 -> 20-28 -> 11-19 chain
# that BUILDS UP to the final reflected amplitude. Display-only (multi-line, not single
# sympifiable expressions), so kept OUT of the Copy/Mathematica/export path.
_ANALYTICAL_DERIVATION_STEPS = (
    ("deriv_0_definitions",
     "Step 0 — Named intermediates (the definition chain)",
     "Every quantity the final amplitude is written in, each with its own defining equation, in "
     "the order ♯SHAARP.si derives them: the transmitted ω eigenmodes (k, θ, E), the 2ω "
     "homogeneous bases, then the bound-field coefficients C<sub>Li</sub>."),
    ("deriv_1_omega",
     "Step 1 — Transmitted fundamental (ω) fields",
     "The two transmitted eigenmodes at ω from the linear Fresnel boundary (eq 29–32) — the "
     "fields that drive the nonlinear source."),
    ("deriv_2_pnl",
     "Step 2 — Nonlinear polarization P_NL (ee, oo, eo)",
     "The 2ω source P = ε₀ d : E<sub>ω</sub>E<sub>ω</sub> for each mode pairing (eq 20–28)."),
    ("deriv_3_inhom",
     "Step 3 — Inhomogeneous (bound) 2ω fields",
     "The particular 2ω fields radiated by P<sub>NL</sub>, from solveInhom (eq 11–19)."),
)


def analytical_derivation_items(result) -> list[tuple[str, str, str]]:
    """Ordered ``(heading, subtitle, typeset-HTML-body)`` derivation steps for the full-analytical
    panel (show it step by step' request). These are the intermediate stages that
    build up to the final reflected amplitude, each the ACTUAL object used in the solve and each
    validated against the published supplementary (tasks #37-40). Empty list if the result carries
    no captured derivation (numeric runs, or a fallback with no stages). Display-only: the raw text
    is multi-line, so Copy/Mathematica/export continue to use the single final expressions."""

    def _typeset_line(line: str) -> str:
        # each captured block line is "LHS = [row-vector]"; typeset the LHS label with real
        # sub/superscripts and the RHS vector through the shared math typesetter.
        if " = " in line:
            lhs, rhs = line.split(" = ", 1)
            lhs = (lhs.replace("E^(T,w)_a", "<b>E</b><sub>a</sub><sup>T,ω</sup>")
                      .replace("E^(T,w)_b", "<b>E</b><sub>b</sub><sup>T,ω</sup>")
                      .replace("P_ee", "<b>P</b><sub>ee</sub>").replace("P_oo", "<b>P</b><sub>oo</sub>")
                      .replace("P_eo", "<b>P</b><sub>eo</sub>")
                      .replace("E^(inh)_ee", "<b>E</b><sub>ee</sub><sup>inh</sup>")
                      .replace("E^(inh)_oo", "<b>E</b><sub>oo</sub><sup>inh</sup>")
                      .replace("E^(inh)_eo", "<b>E</b><sub>eo</sub><sup>inh</sup>"))
            if __import__("re").fullmatch(r"[A-Za-z][A-Za-z0-9_]*", lhs):
                lhs = math_display_html(lhs)  # a layered intermediate: kT_o_w_z -> k^(T,o,ω)_z
            return lhs + " = " + math_display_html(rhs)
        return math_display_html(line)

    stages = getattr(result, "stages", None) or {}
    items: list[tuple[str, str, str]] = []
    for key, heading, subtitle in _ANALYTICAL_DERIVATION_STEPS:
        text = stages.get(key)
        if not text or text in ("None",):
            continue
        body = "<br>".join(_typeset_line(line) for line in str(text).splitlines() if line.strip())
        items.append((heading, subtitle, body))
    return items


def analytical_expression_html(result) -> str:
    """HTML rendering of an analytical result's closed forms with proper super/subscripts
    -- what the desktop panel DISPLAYS; :func:`analytical_expression_text` remains the copy/
    export (machine-readable) layer. Same structure and same ValueError contract."""

    import html as _html

    items = analytical_expression_items(result)  # raises ValueError for numeric results
    math_style = "font-family: 'Cambria Math','STIX Two Math','Times New Roman',serif; font-size: 11pt;"
    parts = [f"<div style='color:#777; font-size:8pt;'># {_html.escape(str(getattr(result, 'kind', '')))}</div>"]
    for label, text in items:
        if label == "symbols":
            # typeset the symbol NAMES here too (follow-up: raw n_omega/theta_i in this line
            # read as "still incorrect"); the .txt export keeps the raw names for Copy users.
            parts.append("<div style='color:#777; font-size:8pt;'># symbols: "
                         f"{math_display_html(text)}</div>")
            continue
        # map the plain label back to its stage key to pick the typeset header
        stage_key = next((k for k, v in _ANALYTICAL_ITEM_LABELS.items() if v == label), None)
        head = _ANALYTICAL_ITEM_LABELS_HTML.get(stage_key, _html.escape(label))
        parts.append(
            f"<div style='font-weight:bold; margin-top:14px;'>{head}</div>"
            f"<div style='white-space:pre; margin-bottom:6px; {math_style}'>{math_display_html(text)}</div>")
    return "<html><body>" + "\n".join(parts) + "</body></html>"


ORIENTATION_MODES = ("z-cut (identity)", "Miller (hkl + in-plane uvw)", "Crystal Physics Directions (Z1,Z2,Z3)")


def build_orientation(
    structure: CrystalStructure,
    mode: str = "z-cut (identity)",
    *,
    surface_hkl=(0, 0, 1),
    in_plane_uvw=(1, 0, 0),
    z_axes=None,
) -> CrystalOrientation:
    """The original GUIs' crystal-orientation entry. ``z-cut (identity)`` keeps the crystal-physics
    axes aligned with the lab axes (the prior behavior). ``Miller (hkl + in-plane uvw)`` is the
    faithful SHAARP ``hklConvert`` mode: ``surface_hkl`` is the surface plane (reciprocal space,
    normal -> lab L3) and ``in_plane_uvw`` is the direct-lattice direction perpendicular to the
    plane of incidence (-> lab L2); it must lie IN the surface plane (validated, raises if not).
    Lattice-aware via ``structure`` (a, b, c, alpha, beta, gamma). Pure/headless-testable."""

    if mode not in ORIENTATION_MODES:
        raise ValueError(f"orientation mode must be one of {ORIENTATION_MODES}, got {mode!r}")
    if mode == "z-cut (identity)":
        return CrystalOrientation()
    if mode == "Crystal Physics Directions (Z1,Z2,Z3)":
        # the original's direct-entry mode: rows Z1/Z2/Z3 are the crystal-physics axes in the lab
        # frame (L1, L2, L3). Validate orthonormality so a malformed entry is rejected, not
        # silently producing a non-rotation.
        z = np.asarray(z_axes if z_axes is not None else np.eye(3), dtype=float)
        if z.shape != (3, 3):
            raise ValueError("z_axes must be a 3x3 matrix of Z1/Z2/Z3 rows in the lab frame.")
        if float(np.max(np.abs(z @ z.T - np.eye(3)))) > 1e-6:
            raise ValueError("Z1/Z2/Z3 must be mutually orthonormal (an orthonormal lab-frame triad).")
        return CrystalOrientation(z)
    return CrystalOrientation.from_shaarp_miller_surface(structure, surface_hkl, in_plane_uvw)


def build_custom_si_material(
    point_group: str,
    *,
    n_omega: float = 2.0,
    n_2omega: float = 2.2,
    n_omega_e: float | None = None,
    n_2omega_e: float | None = None,
    d_free: dict[tuple[int, int], complex] | None = None,
    eps_omega_full=None,
    eps_2omega_full=None,
    d_full=None,
    lattice: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
    orientation_mode: str = "z-cut (identity)",
    surface_hkl=(0, 0, 1),
    in_plane_uvw=(1, 0, 0),
    z_axes=None,
    name: str = "custom",
) -> Material:
    """Build a CUSTOM single-interface material from GUI-entry values: ordinary (and optionally
    extraordinary -> uniaxial-z) refractive indices at omega/2omega plus the point group's FREE
    d components (the symmetry relations are applied automatically by build_constrained_d_voigt,
    exactly like the original GUI's constrained SHG-tensor entry). ``lattice`` = (a, b, c,
    alpha_deg, beta_deg, gamma_deg); ``orientation_mode``/``surface_hkl``/``in_plane_uvw`` select
    the crystal orientation (see :func:`build_orientation`). NOTE: eps and d are entered in the
    CRYSTAL-PHYSICS frame and rotated to the lab by the orientation (the package's Material
    semantics). Pure/headless-testable."""

    # SHG d tensor (crystal frame): a full 3x6 complex matrix overrides the symmetry-constrained
    # free-component entry (the original GUI's full-matrix d input); otherwise build from free d_ij.
    if d_full is not None:
        d_crystal = np.asarray(d_full, dtype=complex).reshape(3, 6)
    else:
        free = point_group_free_components(point_group)
        values = {(r, c): (d_free.get((r, c), 1.0) if d_free else 1.0) for (r, c, _n) in free}
        d_crystal = build_constrained_d_voigt(point_group, values)
    ne_w = n_omega if n_omega_e is None else n_omega_e
    ne_2 = n_2omega if n_2omega_e is None else n_2omega_e
    a, b, c, al, be, ga = (float(x) for x in lattice)
    structure = CrystalStructure(point_group=point_group, a=a, b=b, c=c,
                                 alpha_deg=al, beta_deg=be, gamma_deg=ga)
    orientation = build_orientation(structure, orientation_mode,
                                    surface_hkl=surface_hkl, in_plane_uvw=in_plane_uvw, z_axes=z_axes)
    # full 3x3 complex dielectric tensors override the (ordinary/extraordinary) scalar-index entry
    eps_w = (np.asarray(eps_omega_full, dtype=complex).reshape(3, 3) if eps_omega_full is not None
             else np.diag([n_omega**2, n_omega**2, ne_w**2]).astype(complex))
    eps_2w = (np.asarray(eps_2omega_full, dtype=complex).reshape(3, 3) if eps_2omega_full is not None
              else np.diag([n_2omega**2, n_2omega**2, ne_2**2]).astype(complex))
    return Material(
        name=name, structure=structure, orientation=orientation,
        epsilon_omega=eps_w, epsilon_2omega=eps_2w, d_voigt_pm_v=d_crystal,
    )


def build_custom_ml_system(
    point_group: str,
    *,
    film_n_omega: float = 2.2,
    film_n_2omega: float = 2.3,
    film_n_omega_e: float | None = None,
    film_n_2omega_e: float | None = None,
    ambient_n_omega: float = 1.0,
    ambient_n_2omega: float = 1.0,
    substrate_n_omega: float = 1.45,
    substrate_n_2omega: float = 1.46,
    thickness_um: float = 1.0,
    wavelength_um: float = 1.064,
    d_free: dict[tuple[int, int], complex] | None = None,
    film_eps_omega_full=None,
    film_eps_2omega_full=None,
    film_d_full=None,
    lattice: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
    orientation_mode: str = "z-cut (identity)",
    surface_hkl=(0, 0, 1),
    in_plane_uvw=(1, 0, 0),
    z_axes=None,
) -> MultilayerSystem:
    """Build a CUSTOM air / film / substrate multilayer from GUI-entry values (the original .ml
    GUI's per-layer material entry, in its most common shape): film indices + the point group's
    free d components (symmetry-constrained), film thickness, isotropic substrate indices, and
    the wavelength. Pure/headless-testable."""

    from . import presets

    film = build_custom_si_material(
        point_group, n_omega=film_n_omega, n_2omega=film_n_2omega,
        n_omega_e=film_n_omega_e, n_2omega_e=film_n_2omega_e, d_free=d_free, name="custom film",
        eps_omega_full=film_eps_omega_full, eps_2omega_full=film_eps_2omega_full, d_full=film_d_full,
        lattice=lattice, orientation_mode=orientation_mode,
        surface_hkl=surface_hkl, in_plane_uvw=in_plane_uvw, z_axes=z_axes,
    )
    substrate = Material(
        name="custom substrate", structure=CrystalStructure(point_group="∞∞m"), orientation=CrystalOrientation(),
        epsilon_omega=(np.eye(3) * substrate_n_omega**2).astype(complex),
        epsilon_2omega=(np.eye(3) * substrate_n_2omega**2).astype(complex),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    # a non-air ambient is an isotropic-n medium exactly like the
    # substrate; the exact-1.0 default keeps the presets.air() object (equal-results contract).
    if ambient_n_omega == 1.0 and ambient_n_2omega == 1.0:
        ambient = presets.air()
    else:
        ambient = Material(
            name="ambient", structure=CrystalStructure(point_group="∞∞m"), orientation=CrystalOrientation(),
            epsilon_omega=(np.eye(3) * ambient_n_omega**2).astype(complex),
            epsilon_2omega=(np.eye(3) * ambient_n_2omega**2).astype(complex),
            d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
        )
    return MultilayerSystem(
        wavelength_um=float(wavelength_um),
        polarimetry=Polarimetry(theta_deg=0.0, phi_deg=0.0, psi_deg=0.0),
        layers=[
            Layer("air in", ambient, shg_active=False),
            Layer("custom film", film, thickness_um=float(thickness_um), shg_active=True),
            Layer("custom substrate", substrate, shg_active=False),
        ],
    )


def compute_ml_gui_result(
    functionality: str,
    *,
    point_group: str = "-43m",
    theta_deg: float = 20.0,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 45.0,
    theta_step_deg: float = 5.0,
    assumption: str = "Full Multiple Reflections (FMR)",
    fmr_submode: str = "Forward waves only",
    system_preset: str | None = None,
    system=None,
    analytical_d_known: dict | None = None,
    analytical_d_symbolic: bool | None = None,
    analytical_h_value: float | None = None,
    sample_rotation: bool = False,
    sample_rotation_step_deg: float = 10.0,
    sample_rotation_ccw: bool = True,
    sample_rotate_polarizer: bool = False,
    sample_rotate_analyzer: bool = False,
    sample_analyzer_offset_deg: float = 0.0,
    fixed_phi_deg: float | None = None,
    analyzer_psi_deg: float | None = None,
    ellipticity_deg: float | None = None,
    fresnel_min_deg: float | None = None,
    fresnel_max_deg: float | None = None,
    fresnel_step_deg: float | None = None,
):
    """Pure (no-widget) compute for the SHAARP.ml tab. Returns the SHAARPResult the panel displays.

    ``functionality`` is one of :data:`ML_FUNCTIONALITIES`. "SHG Simulation" runs the validated
    numeric multilayer workflow; "Maker Fringes" runs the validated transmitted-SHG vs incidence-
    angle sweep; "Fresnel Coefficients" runs the linear reflection/transmission sweep; "Partial
    Analytical" runs the validated film polarimetry closed form (symbolic in phi, d, h).

    ``assumption`` is the SHAARP.ml Assumptions panel choice (data:`ML_ASSUMPTIONS`: Full (FMR) /
    JK / HH) and applies to the Maker sweep. ``system_preset`` picks a case-study system
    (data:`ML_SYSTEM_PRESETS`); an explicit ``system`` argument overrides it.
    """

    if functionality not in ML_FUNCTIONALITIES:
        raise ValueError(f"SHAARP.ml functionality must be one of {ML_FUNCTIONALITIES}, got {functionality!r}")
    # accept the legacy short labels (Full (FMR)/JK/HH, No backward waves/...) transparently
    assumption = _ASSUMPTION_ALIASES.get(assumption, assumption)
    fmr_submode = _FMR_SUBMODE_ALIASES.get(fmr_submode, fmr_submode)
    if assumption not in ML_ASSUMPTIONS:
        raise ValueError(f"assumption must be one of {tuple(ML_ASSUMPTIONS)}, got {assumption!r}")
    sys_ = system if system is not None else resolve_ml_system_preset(system_preset)
    # None means "keep the system's polarimetry value" -- every pre-F70 headless call stays
    # byte-identical; the GUI always passes its panel values explicitly.
    _pol0 = sys_.polarimetry
    _phi_v = float(_pol0.phi_deg) if fixed_phi_deg is None else float(fixed_phi_deg)
    _psi_v = float(_pol0.psi_deg) if analyzer_psi_deg is None else float(analyzer_psi_deg)
    _ell_v = float(_pol0.ellipticity_deg) if ellipticity_deg is None else float(ellipticity_deg)
    if functionality == "SHG Simulation":
        if sample_rotation:
            # the rotate/fix x 3 sweep lives inside SHG Simulation (mirrors the original's
            # `If[Functionality == "SHG Simulation", If[samplerotationcontrol, ...]]`).
            return ml_sample_rotation_result(
                sys_, theta_deg=float(theta_deg), fixed_phi_deg=_phi_v, analyzer_psi_deg=_psi_v,
                ellipticity_deg=_ell_v, step_deg=float(sample_rotation_step_deg),
                ccw=bool(sample_rotation_ccw), rotate_polarizer=bool(sample_rotate_polarizer),
                rotate_analyzer=bool(sample_rotate_analyzer),
                analyzer_offset_deg=float(sample_analyzer_offset_deg))
        pol = replace(sys_.polarimetry, theta_deg=_desingularize_theta_deg(float(theta_deg)))
        # Apply the Assumptions panel to the POLAR PLOTS too (original .ml docs: assumptions apply
        # to "the polar plots, Fresnel coefficients and/or Maker fringes"). Found by the T1
        # input-sensitivity gate: this branch silently dropped `assumption`.
        opts = {"include_validation_summary": True, "mrassumption": ML_ASSUMPTIONS[assumption]}
        if ML_ASSUMPTIONS[assumption] == 0:  # FMR sub-mode (backward / standing waves)
            opts["inhomogeneous_source_policy"] = FMR_SUBMODES[fmr_submode]
        return run_ml_numeric(replace(sys_, polarimetry=pol), opts)
    if functionality == "Maker Fringes":
        grid = _closed_grid(float(theta_min_deg), float(theta_max_deg), float(theta_step_deg))
        opts = {"include_validation_summary": True, "mrassumption": ML_ASSUMPTIONS[assumption]}
        # FMR sub-mode (the original's winhAssumption 0/1/2): backward / standing waves.
        if ML_ASSUMPTIONS[assumption] == 0:  # FMR
            opts["inhomogeneous_source_policy"] = FMR_SUBMODES[fmr_submode]
        # the Maker sweep needs the INPUT polarization (phi, ellipticity)
        # and the DETECTION polarization (analyzer psi; the perpendicular channel is psi + 90)
        # from Polarimetry Settings. The sweep consumes system.polarimetry per point, so thread
        # the panel values there; None-defaults keep every pre-F70 headless call byte-identical.
        _mk_pol = replace(sys_.polarimetry, phi_deg=_phi_v, psi_deg=_psi_v,
                          ellipticity_deg=_ell_v)
        return run_maker_fringes(replace(sys_, polarimetry=_mk_pol), grid, opts)
    if functionality == "Fresnel Coefficients":
        # the Fresnel scan range is controlled SEPARATELY from the Maker
        # Fringes range (own min/max/step, default step 0.1 deg). The original fixed the range at
        # 0-90 and exposed only the step -- the author's directive supersedes; None-defaults reproduce
        # the original convention (0-89.9, theta_step_deg) so pre-F70 calls are byte-identical.
        # (89.9 upper bound: the solver's open interval excludes 90.)
        _fr_lo = 0.0 if fresnel_min_deg is None else float(fresnel_min_deg)
        _fr_hi = 89.9 if fresnel_max_deg is None else min(float(fresnel_max_deg), 89.9)
        _fr_st = float(theta_step_deg) if fresnel_step_deg is None else float(fresnel_step_deg)
        grid = _closed_grid(_fr_lo, _fr_hi, _fr_st)
        return run_fresnel_sweep(sys_, grid, {"workflow": "gui_multilayer"})
    # Partial analytical -> the validated ML film polarimetry closed form.
    pg = sys_.layers[1].material.structure.point_group if len(sys_.layers) > 1 else point_group
    # with N interior layers the closed form is symmetry-forbidden only when EVERY one of
    # them is passive (before, a centrosymmetric FIRST film short-circuited the whole stack).
    _interior_pgs = [L.material.structure.point_group
                     for L in (sys_.layers[1:-1] if len(sys_.layers) > 2 else sys_.layers[1:])]
    _interior_active = [p for p in _interior_pgs if p in SHG_POINT_GROUPS]
    if _interior_pgs and not _interior_active:
        pg = _interior_pgs[0]
    if pg not in SHG_POINT_GROUPS and not _interior_active:
        # Centrosymmetric / isotropic film point group (Air & Au coating = ∞∞m, Al₂O₃ = 6/mmm,
        # Pt & Blank-linear = m3m): SHG is symmetry-forbidden, so the partial-analytical closed form is
        # identically zero. The symbolic d-tensor builder has no pattern for these groups (and the
        # closed form rejects an all-zero d), so return a graceful zero result the panel renders as a
        # clear note -- instead of crashing. (The numeric SHG/Maker/Fresnel modes handle these films.)
        note = f"0   (SHG symmetry-forbidden: centrosymmetric/isotropic point group {pg})"
        return SHAARPResult(
            kind="ml_partial_analytical_polarimetry",
            numeric={},
            stages={"workflow": "polarimetry", "point_group": pg,
                    "reflected_p_2omega": note, "reflected_s_2omega": note},
        )
    # Build the closed form FROM THE ACTUAL STACK (release-audit l): previously only
    # the point group was passed, so the expression used the runner's GENERIC defaults -- theta =
    # 0.5 rad (28.6 deg) regardless of the user's incidence angle, film n = 2.1/2.25 and substrate
    # 1.5/1.55 regardless of the configured layers, and no orientation rotation of d.
    case: dict = {"point_group": pg,
                  "incident_theta_rad": float(np.deg2rad(_desingularize_theta_deg(theta_deg)))}
    ml_extra: dict[str, str] = {"incidence": f"theta_i = {float(theta_deg):g} deg (substituted)"}
    interiors = list(sys_.layers[1:-1]) if len(sys_.layers) > 2 else list(sys_.layers[1:])
    first_active_row: int | None = None  # user-visible row of the first SHG-active layer
    if interiors:
        import sympy as sp
        from .symbolic import d_voigt_symbolic, rotate_d_voigt_symbolic

        # EVERY interior layer reaches the closed form. Before this, the
        # case was built from layers[1] and layers[-1] only, so a 4-layer stack silently
        # computed as a 1-film system -- the panel showed three films while the expression
        # described one, with no warning (the engine's own single-film guard never fired
        # because the API always handed it a one-element list).
        case["layer_epsilon_omega_lab"] = [_epsilon_lab_of(L.material, omega=True) for L in interiors]
        case["layer_epsilon_2omega_lab"] = [_epsilon_lab_of(L.material, omega=False) for L in interiors]
        # Per-layer d: symbolic where the layer is SHG-active, zeros where it is not. Symbol
        # names are SUFFIXED PER LAYER when more than one layer is active -- the original does
        # the same (`SHAARP.ml.nb:7256-7292` uses d11m1, d11m2, ...); without it two layers of
        # the same point group would silently share one d33.
        from .layer_stack import interior_layer_number

        active = [bool(L.shg_active) and L.material.structure.point_group in SHG_POINT_GROUPS
                  for L in interiors]
        # F60 per-layer 'analytical dij' (SHAARP.ml.nb:7256-7292): a FLAGGED layer keeps its d
        # symbolic; an unflagged one has its entered numbers substituted. BACK-COMPAT: when no
        # layer is flagged at all (presets, headless callers, and the historical default where
        # the closed form was always the d-extraction expression), every layer stays symbolic.
        any_d_flag = any(bool(getattr(L, "analytic_d", False)) for L in interiors)
        # ``analytical_d_symbolic=False`` is the caller SAYING "nothing is symbolic" -- without
        # it, clearing the last per-layer flag would fall through to the legacy all-symbolic
        # default and the checkbox could never turn d numeric (the same trap the thickness flag
        # hit). The GUI passes it whenever no layer is flagged; headless callers leave it None
        # and keep the historical d-extraction expression.
        d_list = []
        for idx, (L, is_active) in enumerate(zip(interiors, active), start=1):
            if not is_active:
                d_list.append(sp.zeros(3, 6))
                continue
            if ((any_d_flag and not bool(getattr(L, "analytic_d", False)))
                    or (not any_d_flag and analytical_d_symbolic is False)):
                d_list.append(sp.Matrix(np.asarray(L.material.d_voigt(), dtype=complex).tolist()))
                continue
            pg_l = L.material.structure.point_group
            d_sym = sp.Matrix(d_voigt_symbolic(pg_l))
            # ALWAYS suffix, with the USER-VISIBLE row number (the original suffixes
            # unconditionally too, `SHAARP.ml.nb:7256-7292`). Before F61 the suffix appeared only
            # when 2+ layers were SHG-active and counted interiors, so ONE layer could be spelled
            # `d11`, `d11m1` or `d11m2` depending on unrelated layers. `d_voigt_symbolic`'s
            # `prefix` REPLACES the leading "d", so rename the symbols instead.
            _row = interior_layer_number(idx - 1)
            d_sym = d_sym.subs({sym: sp.Symbol(f"{sym.name}m{_row}")
                                for sym in sorted(d_sym.free_symbols, key=str)})
            rot = np.asarray(L.material.orientation.rotation_matrix(), dtype=float)
            if np.max(np.abs(rot - np.eye(3))) >= 1e-9:
                # rotate_d_voigt_symbolic takes the TRANSPOSE of rotation_matrix() (SI branch).
                # EXACT rotation entries, the same nsimplify step the SI
                # branch uses (FA-1). Raw floats (0.5773502691896258 ...) cannot cancel in the
                # CAS, so the (111)/(112) cuts produced 0.8-1.2 MB closed forms and 45-230 s
                # solves -- the "perf regression" and the tests.test_si_normal_incidence
                # gate-budget hang (GaAs(111) 1064 nm alone > 150 s). Axis-aligned cuts are
                # untouched (their entries are exactly 0/±1 either way).
                rot_exact = sp.Matrix(rot.T.tolist()).applyfunc(
                    lambda v: sp.nsimplify(v, rational=False, tolerance=1e-12))
                d_sym = rotate_d_voigt_symbolic(d_sym, rot_exact)
                ml_extra.setdefault("orientation", "d rotated crystal→lab per layer")
            d_list.append(sp.Matrix(d_sym))
        case["layer_d_voigt_symbolic"] = d_list
        first_active_row = next((interior_layer_number(k) for k, a in enumerate(active) if a), None)
        # keep the familiar single-film provenance key alongside the per-layer list
        _first_active = next((L for L, a in zip(interiors, active) if a), interiors[0])
        ml_extra["film"] = f"{_first_active.name} ε (substituted)"
        ml_extra["layers"] = ", ".join(
            f"{i}: {L.name}{'' if a else ' (passive)'}"
            f"{' [d symbolic]' if (a and ((not any_d_flag and analytical_d_symbolic is not False) or getattr(L, 'analytic_d', False))) else ''}"
            for i, (L, a) in ((interior_layer_number(k), pair)
                              for k, pair in enumerate(zip(interiors, active))))
        # F60 per-layer 'analytical h' (SHAARP.ml.nb:5135-5146 stores h1, h2, ... per layer;
        # setup.nb:11011-11017 consumes them as a list, mixing numeric and symbolic freely).
        # one spelling, numbered by the USER-VISIBLE row, and OFF strictly means
        # substituted. The pre-F61 code both (a) dropped the subscript for a single-interior
        # stack and (b) carried a "stack default" branch that made the FIRST interior layer
        # symbolic (bare `h`) whenever no flag was set -- so an explicitly unflagged layer came
        # out symbolic. That is the same back-compat trap that made `analytic_d` unable to turn
        # itself off: a "no flags => legacy default" rule overrides the user's OFF.
        entries, notes = [], []
        for idx, L in enumerate(interiors, start=1):
            row = interior_layer_number(idx - 1)
            if bool(getattr(L, "analytic_h", False)):
                entries.append(sp.Symbol(f"h{row}", positive=True))
                notes.append(f"h{row} (symbolic)")
            else:
                entries.append(sp.Float(float(L.thickness_um or 0.0)))
                notes.append(f"h{row} = {float(L.thickness_um or 0.0):g} um (substituted)")
        case["thickness_symbols"] = entries
        ml_extra["thickness"] = ", ".join(notes)
    if len(sys_.layers) > 2:
        sub = sys_.layers[-1].material
        case["substrate_epsilon_omega_lab"] = _epsilon_lab_of(sub, omega=True)
        case["substrate_epsilon_2omega_lab"] = _epsilon_lab_of(sub, omega=False)
    # the AMBIENT's true indices reach the closed form (previously hardwired n0 = 1
    # even when the stack's top layer was not air). Same isotropy contract as the numeric solver.
    amb = sys_.layers[0].material
    amb_w = np.asarray(amb.eps_w(), dtype=complex)
    amb_2 = np.asarray(amb.eps_2w(), dtype=complex)
    for _lbl, _e in (("omega", amb_w), ("2omega", amb_2)):
        if np.max(np.abs(_e - _e[0, 0] * np.eye(3))) > 1e-9:
            raise ValueError(f"top epsilon_{_lbl} must be isotropic for the partial-analytical "
                             "workflow (the incident medium is the stack's first layer)")
    def _scalar_index(eps00):
        n = complex(np.sqrt(eps00))
        # real media collapse to float so the symbolic expressions render exactly as before
        return float(n.real) if abs(n.imag) < 1e-12 else n

    case["ambient_index_omega"] = _scalar_index(amb_w[0, 0])
    case["ambient_index_2omega"] = _scalar_index(amb_2[0, 0])
    # 'analytical dij' KNOWN/UNKNOWN MIXING (fidelity FB2, original .ml: "In the case where certain
    # components of the SHG tensor is known ... the partial analytical expression only involves the
    # unknown symbolic components"): substitute the user-entered numeric components into the
    # symbolic pattern; dependent components inherit through the symmetry relations automatically.
    if analytical_d_known:
        import sympy as sp
        from .symbolic import d_voigt_symbolic
        base = case.get("d_voigt_symbolic")
        if base is None:
            base = d_voigt_symbolic(pg)
        import re as _re
        subs = {}
        # substitute EXACTLY the names given. A key that already carries its layer suffix
        # (d11m2) is layer-scoped by construction. A BARE key comes from a legacy/headless caller
        # or the SI tab; map it to the first SHG-active layer's spelling as well, so single-film
        # callers keep working -- but never spray it across every layer (that let a number
        # entered for one layer rewrite the others).
        _first_row = first_active_row
        for name, val in analytical_d_known.items():
            v = complex(val)
            known = (sp.Float(v.real) if v.imag == 0.0
                     else sp.Float(v.real) + sp.I * sp.Float(v.imag))
            nm = str(name)
            subs[sp.Symbol(nm)] = known
            if _first_row is not None and not _re.search(r"m\d+$", nm):
                subs[sp.Symbol(f"{nm}m{_first_row}")] = known
        case["d_voigt_symbolic"] = sp.Matrix(base).subs(subs)
        # the per-layer LIST is what the API consumes -- substituting only the singular
        # key silently dropped the known/unknown mixing (FB2) the moment the case went N-layer.
        if case.get("layer_d_voigt_symbolic") is not None:
            case["layer_d_voigt_symbolic"] = [sp.Matrix(d).subs(subs)
                                              for d in case["layer_d_voigt_symbolic"]]
        ml_extra["known d"] = ", ".join(f"{k}={complex(v).real:g}" if complex(v).imag == 0
                                        else f"{k}={complex(v)}" for k, v in analytical_d_known.items())
    # 'analytical h' toggle (FB2): OFF -> the panel thickness is SUBSTITUTED into the closed form
    # (the original's default when a layer is not marked 'analytical h'); ON -> symbolic h (default).
    if analytical_h_value is not None and not case.get("thickness_symbols"):
        # legacy/headless single-film callers only. When the per-layer list exists it ALWAYS
        # wins in api.py, so setting the singular key here would be dead -- but its old
        # ml_extra side effect still overwrote the per-layer notes, letting the provenance
        # claim "substituted" while h2 remained in the expression.
        import sympy as sp
        case["thickness_symbol"] = sp.Float(float(analytical_h_value))
        ml_extra["thickness"] = f"h = {float(analytical_h_value):g} um (substituted)"
    if sample_rotation:
        # SAMPLE ROTATION under Partial Analytical -> a CLOSED FORM in the azimuth. Feasible
        # only where the rotation leaves eps alone (see azimuth_closed_form_feasible); otherwise
        # fall back to the numeric sweep and say why, rather than emitting a wrong expression.
        _ok, _reason = azimuth_closed_form_feasible(sys_)
        if _ok and not ANALYTIC_AZIMUTH_ENABLED:
            _ok, _reason = False, (
                "the closed form in the sample azimuth is NOT YET VERIFIED: substituting psi into "
                "it disagrees by ~32% with re-solving the numerically-rotated sample, which is the "
                "path validated against Mathematica. Every declared solver input agrees to <=1.9e-13 "
                "between the two routes, so the cause is not yet identified. Ran the numeric "
                "azimuth sweep instead (see ANALYTIC_AZIMUTH_ENABLED)")
        if not _ok:
            _res = ml_sample_rotation_result(
                sys_, theta_deg=theta_deg, fixed_phi_deg=_phi_v,
                analyzer_psi_deg=_psi_v, ellipticity_deg=_ell_v,
                step_deg=sample_rotation_step_deg, ccw=sample_rotation_ccw,
                rotate_polarizer=bool(sample_rotate_polarizer),
                rotate_analyzer=bool(sample_rotate_analyzer),
                analyzer_offset_deg=float(sample_analyzer_offset_deg))
            _res.stages["analytic_azimuth_fallback_reason"] = _reason
            return _res
        import math as _math

        import sympy as _sp
        _psi_s = _sp.Symbol("psi_s", real=True)
        # the solver's +azimuth reads CW from the beam side, so a CCW sweep is the negated angle
        case["sample_azimuth_symbol"] = (-_psi_s if sample_rotation_ccw else _psi_s)
        # under the sample-rotation pin the polarizer and analyzer are FIXED: substitute phi so the
        # closed form is a function of the azimuth alone (and so it matches the numeric sweep)
        case["phi_symbol"] = _sp.Float(_math.radians(_phi_v))
        case["ellipticity_rad"] = _math.radians(_ell_v)
        ml_extra["sample azimuth"] = (
            f"psi_s (radians, {'CCW' if sample_rotation_ccw else 'CW'} looking at the sample from "
            f"the beam side); eps invariance {_reason.split('to ')[-1]}")
        ml_extra["input polarization"] = f"phi = {_phi_v:g} deg (substituted)"
    result = run_ml_partial_analytical(case, {"workflow": "polarimetry"})
    try:
        result.stages.setdefault("symbols", {}).update(ml_extra)
    except Exception:
        pass
    return result


# the SHG-active point groups are the original's "Noncentrosymmetric ->" popup
# (SHAARP.ml.nb:5191; the .si popup lists the same 23): the 20 crystallographic classes plus the
# three SHG-active Curie groups (inf, infm, in), in the original's order. The single source of
# truth (lists, aliases, activity, crystal-system lattice rules) is shaarp/point_groups.py; this
# name is kept for every existing membership check and the ipywidgets dropdown.
from .point_groups import SHG_ACTIVE_GROUPS as SHG_POINT_GROUPS  # noqa: E402


def point_group_free_components(point_group: str) -> list[tuple[int, int, str]]:
    """Return the INDEPENDENT (free) Voigt d-components for a point group as
    ``(row, col, name)`` tuples -- the cells an experimentalist actually enters; the rest of the
    3x6 tensor follows by the point group's symmetry relations (e.g. 3m's d16 = -d22, d24 = d15).
    Derived from the package's authoritative symbolic pattern ``d_voigt_symbolic``."""

    from .symbolic import d_voigt_symbolic

    d = d_voigt_symbolic(point_group)
    seen: set[str] = set()
    free: list[tuple[int, int, str]] = []
    for r in range(3):
        for c in range(6):
            entry = d[r, c]
            # a bare +symbol (coefficient +1) at its first occurrence = an independent component;
            # zeros, negated copies (-d22), and repeats are dependent.
            if getattr(entry, "is_Symbol", False):
                name = str(entry)
                if name not in seen:
                    seen.add(name)
                    free.append((r, c, name))
    return free


def build_constrained_d_voigt(point_group: str, free_values: dict[tuple[int, int], complex]) -> np.ndarray:
    """Build the full symmetry-constrained 3x6 numeric Voigt d-tensor for a point group from the
    INDEPENDENT component values (keyed by their ``(row, col)`` primary position, as returned by
    :func:`point_group_free_components`). The point group's relations (sign-flips, equalities,
    zeros) are applied automatically via the symbolic pattern -- so a user enters only the free
    d_ij and gets a physically valid tensor, exactly as the original GUI does."""

    from .symbolic import _sympy, d_voigt_symbolic

    sp = _sympy()
    d_sym = d_voigt_symbolic(point_group)
    subs = {}
    for (r, c, name) in point_group_free_components(point_group):
        val = free_values.get((r, c), 0.0)
        subs[sp.Symbol(name)] = complex(val)
    d_num = d_sym.subs(subs)
    return np.array([[complex(d_num[r, c]) for c in range(6)] for r in range(3)], dtype=complex)


def _constrained_or_full_d_voigt(point_group: str, d_free) -> np.ndarray:
    """Numeric 3x6 Voigt d for the polarimetry closed form.

    For a crystallographic point group in the symbolic set, apply its symmetry constraints to the free
    components (the normal path). For a point group OUTSIDE that set -- the isotropic / centrosymmetric
    Curie groups carried by some case-study materials (Air & Au coating = ∞∞m, Al₂O₃ = 6/mmm,
    Pt & Blank-linear = m3m) -- there is no symbolic d-pattern; those groups are centrosymmetric or
    isotropic, so SHG is symmetry-forbidden (d ≈ 0). Build the matrix DIRECTLY from the provided
    component map instead of crashing, yielding the physically correct (near-zero) SHG response."""

    from .point_groups import is_known_point_group, is_shg_active
    if is_known_point_group(point_group) and not is_shg_active(point_group):
        return np.zeros((3, 6), dtype=complex)  # SHG-inactive group -> d = 0 by symmetry
    try:
        free = point_group_free_components(point_group)
    except (ValueError, KeyError):
        m = np.zeros((3, 6), dtype=complex)
        for (r, c), v in (d_free or {}).items():
            m[r, c] = complex(v)
        return m
    values = {(r, c): (d_free.get((r, c), 1.0) if d_free else 1.0) for (r, c, _n) in free}
    return build_constrained_d_voigt(point_group, values)


def _snap_degenerate_triple(vals, rel_tol: float = 1e-9) -> tuple:
    """Collapse float-noise splits between near-equal principal eps values to EXACT equality.

    A mathematically isotropic (or uniaxial) eps rotated numerically carries ~1e-15 relative
    noise between its principal values. Downstream, the symbolic eigenmode builder tests EXACT
    equality to pick the degeneracy-safe branch; missing it routes the biaxial D-formula into
    divisions by that noise (T3/GaAs(111) polarimetry NaN at most incident
    angles, garbage magnitudes at the rest). Snapping within rel_tol restores the exact
    degeneracy the physics has. Purely-real values are returned as floats so the untouched
    real-eps paths stay byte-identical."""

    v = [complex(x) for x in vals]
    scale = max(abs(x) for x in v) or 1.0
    groups: list[list[int]] = []
    for i, x in enumerate(v):
        for grp in groups:
            if abs(x - v[grp[0]]) <= rel_tol * scale:
                grp.append(i)
                break
        else:
            groups.append([i])
    out = list(v)
    for grp in groups:
        if len(grp) > 1:
            mean = sum(v[i] for i in grp) / len(grp)
            for i in grp:
                out[i] = mean
    return tuple(float(x.real) if x.imag == 0.0 else x for x in out)


def si_polarimetry_curve(
    point_group: str,
    *,
    theta_deg: float = 45.0,
    n_omega: float = 2.0,
    n_2omega: float = 2.2,
    n_omega_e: float | None = None,
    n_2omega_e: float | None = None,
    d_free: dict[tuple[int, int], complex] | None = None,
    n_phi: int = 181,
    ellipticity_deg: float = 0.0,
    analyzer_deg: float | None = None,
    fixed_phi_deg: float | None = None,
    corotating_offset_deg: float | None = None,
    eps_omega_principal: tuple | None = None,
    eps_2omega_principal: tuple | None = None,
    d_voigt_lab_full=None,
    eps_omega_lab_full=None,
    eps_2omega_lab_full=None,
    incident_index_omega: float = 1.0,
    incident_index_2omega: float = 1.0,
) -> dict[str, Any]:
    """Numeric reflected-SHG polarimetry I_s(phi), I_p(phi) for ANY point group, evaluated from the
    SHAARP.si full-analytical CLOSED FORM with that point group's symmetry-constrained d-tensor
    (z-cut: crystal frame = lab frame). ``d_free`` maps the independent ``(row,col)`` components to
    values (default: unit 1.0 each). ``n_omega``/``n_2omega`` are the ordinary indices; passing
    ``n_omega_e``/``n_2omega_e`` makes the crystal UNIAXIAL with the optic axis along the surface
    normal (eps_z from the extraordinary index; default None -> isotropic, the prior behavior).

    Polarimetry-settings parity with the original GUI: ``ellipticity_deg`` is the incident
    ellipticity Delta-delta (the incident Jones becomes (sin phi * e^{i*ellipticity}, cos phi));
    ``analyzer_deg`` selects a FIXED analyzer at psi (the returned ``intensity_analyzed`` =
    |sin psi * E_s + cos psi * E_p|^2). With ``analyzer_deg=None`` the two standard rotating-analyzer
    channels (parallel ``intensity_p``, perpendicular ``intensity_s``) are returned.
    Pure/headless -> plain arrays so the GUI plot is testable."""

    import math

    import sympy as sp

    from .symbolic import solve_si_shg_full_analytical_symbolic

    try:  # NonInvertibleMatrixError moved modules across sympy versions
        from sympy.matrices.exceptions import NonInvertibleMatrixError
    except ImportError:  # pragma: no cover - older sympy
        from sympy.matrices.common import NonInvertibleMatrixError

    # Orientation-aware overrides (release-audit): a caller with an obliquely-
    # oriented material passes the LAB-frame principal eps triples and the numerically-rotated
    # full 3x6 lab d, making the plotted polarimetry EXACT for x-/y-cut and Miller-oriented
    # crystals (previously the curve always assumed z-cut: crystal frame = lab frame).
    if d_voigt_lab_full is not None:
        # keep d COMPLEX (T3/: the old np.real() here silently applied the paper's
        # eps_R-class approximation to any material with a complex tensor)
        d_lab = np.asarray(d_voigt_lab_full, dtype=complex)
    else:
        d_lab = _constrained_or_full_d_voigt(point_group, d_free)
    if eps_omega_lab_full is None:
        # ABSORBING principal-aligned crystals (complex eps entered at z-cut/identity — e.g. the
        # TaAs or LiOsO3 tensors kept after a case flips to Custom): the full symbolic closed form
        # explodes combinatorially in sympy on complex uniaxial radicals (GUI hang).
        # Route them through the SAME validated numeric per-phi solver as general orientations —
        # the reference the closed form itself is fenced against to 1e-12..1e-16.
        _ne_w = n_omega if n_omega_e is None else n_omega_e
        _ne_2 = n_2omega if n_2omega_e is None else n_2omega_e
        _w3 = (eps_omega_principal if eps_omega_principal is not None
               else (n_omega ** 2, n_omega ** 2, _ne_w ** 2))
        _23 = (eps_2omega_principal if eps_2omega_principal is not None
               else (n_2omega ** 2, n_2omega ** 2, _ne_2 ** 2))
        # a NON-AIR incident medium routes through the same numeric branch — the symbolic
        # closed form derives with n0 = 1 baked into its tangential wavevector (kx = w*sin(theta));
        # the numeric solver takes the true incident indices first-class.
        if (any(abs(complex(v).imag) > 1e-12 for v in (*tuple(_w3), *tuple(_23)))
                or incident_index_omega != 1.0 or incident_index_2omega != 1.0):
            eps_omega_lab_full = np.diag(np.asarray(tuple(_w3), dtype=complex))
            eps_2omega_lab_full = np.diag(np.asarray(tuple(_23), dtype=complex))
    if eps_omega_lab_full is not None:
        # GENERAL-orientation branch: a lab-frame eps that is NOT
        # principal-aligned (e.g. the TaAs (112) tilt of a uniaxial crystal) has no closed-form
        # route, and this case previously fell back SILENTLY to the crystal-frame z-cut
        # approximation. Evaluate the curve per-phi with the validated numeric arbitrary-Jones
        # solver instead -- the very reference the closed form itself was validated against to
        # 1e-12..1e-16 (tests/test_si_shg_polarimetry_symbolic.py). Same field conventions as
        # that test's _numeric helper: the GENERIC solver's coefficients[0] = reflected s
        # amplitude, [1] = reflected p (NOTE: the compat workflow's coefficient order is the
        # OPPOSITE -- do not conflate); incident Jones (J_s, J_p) = (sin(phi) e^{i ell}, cos(phi)).
        from .shg import solve_single_interface_shg

        ew_l = np.asarray(eps_omega_lab_full, dtype=complex).reshape(3, 3)
        e2_l = np.asarray(eps_2omega_lab_full if eps_2omega_lab_full is not None
                          else eps_omega_lab_full, dtype=complex).reshape(3, 3)
        th_rad = math.radians(_desingularize_theta_deg(theta_deg))
        ell_phase = complex(math.cos(math.radians(ellipticity_deg)),
                            math.sin(math.radians(ellipticity_deg)))
        _pair_cache: dict[float, tuple[complex, complex]] = {}

        def _sp_pair(phi_val: float) -> tuple[complex, complex]:
            if phi_val not in _pair_cache:
                r = solve_single_interface_shg(
                    ew_l, e2_l, d_lab,
                    incident_index_omega=float(incident_index_omega),
                    incident_index_2omega=float(incident_index_2omega),
                    incident_theta_rad=th_rad,
                    incident_jones=(math.sin(phi_val) * ell_phase, math.cos(phi_val)),
                    omega=1.0, mu=1.0, eps0=1.0)
                co = np.asarray(r.coefficients)
                _pair_cache[phi_val] = (complex(co[0]), complex(co[1]))  # (E_s, E_p)
            return _pair_cache[phi_val]

        def _num_channel(idx):
            def _f(rads):
                if np.ndim(rads) == 0:
                    return _sp_pair(float(rads))[idx]
                flat = [_sp_pair(float(v))[idx] for v in np.ravel(rads)]
                return np.asarray(flat, dtype=complex).reshape(np.shape(rads))
            return _f

        sol = None  # no symbolic solution object (transmitted best-effort below is skipped)
        e_s = _num_channel(0)
        e_p = _num_channel(1)
    else:
        ne_w = n_omega if n_omega_e is None else n_omega_e
        ne_2 = n_2omega if n_2omega_e is None else n_2omega_e
        eps_w3 = (_snap_degenerate_triple(eps_omega_principal) if eps_omega_principal is not None
                  else _snap_degenerate_triple((n_omega**2, n_omega**2, ne_w**2)))
        eps_23 = (_snap_degenerate_triple(eps_2omega_principal) if eps_2omega_principal is not None
                  else _snap_degenerate_triple((n_2omega**2, n_2omega**2, ne_2**2)))
        phi = sp.Symbol("phi", real=True)
        try:
            sol = solve_si_shg_full_analytical_symbolic(
                eps_x_omega=eps_w3[0], eps_y_omega=eps_w3[1], eps_z_omega=eps_w3[2],
                eps_x_2omega=eps_23[0], eps_y_2omega=eps_23[1], eps_z_2omega=eps_23[2],
                d_voigt_lab=sp.Matrix(d_lab.tolist()), incident_theta_rad=math.radians(_desingularize_theta_deg(theta_deg)),
                phi_symbol=phi, ellipticity=math.radians(ellipticity_deg), omega=1, mu=1, eps0=1, simplify=False,
            )
            e_s = sp.lambdify(phi, sol.reflected_s, "numpy")
            e_p = sp.lambdify(phi, sol.reflected_p, "numpy")
        except (NonInvertibleMatrixError, np.linalg.LinAlgError, ZeroDivisionError):
            # Degenerate interface: the "crystal" is optically identical to the ambient -- the vacuum
            # "Air" sample (n=1, d=0) has no index contrast, hence no boundary to solve, no reflection,
            # and no SHG. Return a flat zero curve (the physically correct "no SHG") instead of crashing.
            sol = None
            e_s = e_p = (lambda _r: 0.0)
    # 'Fix Polarizer' mode: hold the incident polarization at phi0 and sweep the ANALYZER psi instead
    # -> the polar plot traces the rotating-analyzer intensity I(psi) = |sin psi E_s(phi0) + cos psi E_p(phi0)|^2.
    if fixed_phi_deg is not None:
        es0 = complex(e_s(math.radians(float(fixed_phi_deg))))
        ep0 = complex(e_p(math.radians(float(fixed_phi_deg))))
        psi_deg = np.linspace(0.0, 360.0, int(n_phi))
        psi = np.radians(psi_deg)
        ia = (np.abs(np.sin(psi) * es0 + np.cos(psi) * ep0) ** 2).astype(float)
        return {"phi_deg": psi_deg, "intensity_analyzed": ia, "fixed_phi_deg": float(fixed_phi_deg),
                "point_group": point_group, "theta_deg": float(theta_deg),
                "ellipticity_deg": float(ellipticity_deg)}
    phi_deg = np.linspace(0.0, 360.0, int(n_phi))
    rad = np.radians(phi_deg)
    es = np.broadcast_to(e_s(rad), rad.shape)
    ep = np.broadcast_to(e_p(rad), rad.shape)
    if corotating_offset_deg is not None:
        # The ORIGINAL's rotating-polarizer/rotating-analyzer mode: the analyzer TRACKS the
        # polarizer at psi = phi + offset ("the analyzer-polarizer offset angle may be entered",
        # .si/.ml docs, fidelity FB1). Parallel channel at the offset, perpendicular at +90 deg.
        off = math.radians(float(corotating_offset_deg))
        i_par = (np.abs(np.sin(rad + off) * es + np.cos(rad + off) * ep) ** 2).astype(float)
        i_per = (np.abs(np.cos(rad + off) * es - np.sin(rad + off) * ep) ** 2).astype(float)
        return {"phi_deg": phi_deg, "intensity_parallel": i_par, "intensity_perpendicular": i_per,
                "corotating_offset_deg": float(corotating_offset_deg),
                "point_group": point_group, "theta_deg": float(theta_deg),
                "ellipticity_deg": float(ellipticity_deg)}
    i_s = (np.abs(es) ** 2).astype(float)
    i_p = (np.abs(ep) ** 2).astype(float)
    out = {"phi_deg": phi_deg, "intensity_s": i_s, "intensity_p": i_p,
           "point_group": point_group, "theta_deg": float(theta_deg),
           "ellipticity_deg": float(ellipticity_deg)}
    # TRANSMITTED-channel polarimetry too (the original's output panel shows BOTH reflected and
    # transmitted I_p/I_s polar plots). transmitted_field is [E_p, E_s, E_z] in the lab frame
    # (the transmitted-channel observable, validated vs numeric ~1e-15).
    try:
        e_t = sol.transmitted_field
        e_tp = sp.lambdify(phi, e_t[0], "numpy")
        e_ts = sp.lambdify(phi, e_t[1], "numpy")
        out["intensity_p_trans"] = (np.abs(np.broadcast_to(e_tp(rad), rad.shape)) ** 2).astype(float)
        out["intensity_s_trans"] = (np.abs(np.broadcast_to(e_ts(rad), rad.shape)) ** 2).astype(float)
    except Exception:
        pass  # reflected channels always present; transmitted is best-effort
    if analyzer_deg is not None:
        psi = math.radians(analyzer_deg)
        out["intensity_analyzed"] = (np.abs(math.sin(psi) * es + math.cos(psi) * ep) ** 2).astype(float)
        # perpendicular (crossed) analyzer at psi+90 -> the fixed-analyzer view gets the SAME
        # parallel+perpendicular pair as the co-rotating view, so both render identically.
        psi_perp = psi + math.pi / 2.0
        out["intensity_analyzed_perp"] = (
            np.abs(math.sin(psi_perp) * es + math.cos(psi_perp) * ep) ** 2).astype(float)
        out["analyzer_deg"] = float(analyzer_deg)
    return out


def si_figure_kwargs_from_material(material) -> dict:
    """Derive the build_si_polarimetry_figure inputs (point group, indices, d components) from a
    Material, so the SI polar plot reflects the selected case-study material rather than the scalar
    entry fields. ORIENTATION-AWARE (release-audit): for obliquely-oriented crystals
    (x-/y-cut, Miller-oriented like GaAs(111)) the LAB-frame principal eps triples and the rotated
    full lab d are passed, so the plotted polarimetry is exact -- previously the curve silently used
    the z-cut (crystal frame = lab) approximation for those materials."""

    pg = material.structure.point_group
    ew = np.asarray(material.epsilon_omega)
    e2 = np.asarray(material.epsilon_2omega)
    n_o = float(np.sqrt(ew[0, 0]).real)
    n_o_2 = float(np.sqrt(e2[0, 0]).real)
    n_e = float(np.sqrt(ew[2, 2]).real)
    n_e_2 = float(np.sqrt(e2[2, 2]).real)
    d = np.asarray(material.d_voigt_pm_v)
    try:
        d_free = {(r, c): complex(d[r, c]) for (r, c, _n) in point_group_free_components(pg)}
    except (ValueError, KeyError):
        # point group outside the symbolic crystallographic set (isotropic/centrosymmetric Curie
        # groups: Air & Au = ∞∞m, Al₂O₃ = 6/mmm, Pt & Blank-linear = m3m). Pass the material's ACTUAL
        # nonzero components so the figure path (which falls back to a direct d build) stays exact.
        d_free = {(r, c): complex(d[r, c]) for r in range(3) for c in range(6) if abs(d[r, c]) > 0}
    kwargs = {"point_group": pg, "n_omega": n_o, "n_2omega": n_o_2,
              "n_omega_e": n_e, "n_2omega_e": n_e_2, "d_free": d_free}
    rot = np.asarray(material.orientation.rotation_matrix(), dtype=float)
    from .tensors import rotate_d_voigt_crystal_to_lab

    # ALWAYS pass the material's full lab-frame d (identity orientation: lab = crystal frame).
    # The d_free pull above keeps only the point group's IDEAL pattern components -- a registry
    # tensor with residual components outside that pattern (e.g. the MoS2 case study) was
    # silently truncated (registry-exactness gate, post-mortem).
    kwargs["d_voigt_lab_full"] = np.asarray(
        rotate_d_voigt_crystal_to_lab(d, rot), dtype=complex)
    eps_w_lab = _epsilon_lab_of(material, omega=True)
    eps_2w_lab = _epsilon_lab_of(material, omega=False)
    off = max(float(np.max(np.abs(eps_w_lab - np.diag(np.diag(eps_w_lab))))),
              float(np.max(np.abs(eps_2w_lab - np.diag(np.diag(eps_2w_lab))))))
    if off < 1e-9 * max(1.0, float(np.max(np.abs(eps_w_lab)))):  # principal-aligned in the lab
        # COMPLEX principal values, degeneracy-snapped: the old
        # float(np.real(...)) dropped absorption entirely, and the unsnapped rotation noise
        # between equal principal values sent the symbolic solver down the biaxial D-formula
        # (division by ~1e-16 differences -> NaN curves for GaAs(111), the 2022 paper's case).
        # APPLIED TO IDENTITY ORIENTATIONS TOO (post-mortem): the identity
        # branch used to keep only real n_o/n_e scalars from eps_xx/eps_zz -- silently dropping
        # Im(eps) for absorbing materials (MoS2, Au: 75%/250% off) and eps_yy for identity-
        # oriented biaxial crystals (LBO: 9%). The registry-exactness gate caught all three.
        kwargs["eps_omega_principal"] = _snap_degenerate_triple(
            complex(eps_w_lab[i, i]) for i in range(3))
        kwargs["eps_2omega_principal"] = _snap_degenerate_triple(
            complex(eps_2w_lab[i, i]) for i in range(3))
    else:
        # GENERAL orientation: the lab eps is NOT diagonal --
        # e.g. the TaAs (112) tilt of a uniaxial crystal -- so no closed-form route exists.
        # Previously this case fell through SILENTLY and the figure plotted the crystal-frame
        # z-cut approximation. Pass the full LAB tensors so si_polarimetry_curve evaluates the
        # curve with the validated numeric arbitrary-Jones solver instead (exact).
        kwargs["eps_omega_lab_full"] = eps_w_lab.astype(complex)
        kwargs["eps_2omega_lab_full"] = eps_2w_lab.astype(complex)
    return kwargs


def si_effective_index_curve(
    point_group: str = "",
    *,
    n_omega: float = 2.0,
    n_2omega: float = 2.2,
    n_omega_e: float | None = None,
    n_2omega_e: float | None = None,
    n_theta: int = 181,
    incident_index: float = 1.0,
    eps_omega_lab_full=None,
    **_ignored,
) -> dict[str, Any]:
    """Effective refractive index n(theta_i) of the two transmitted eigenmodes (the ordinary/
    extraordinary indices, with Snell's law) vs incident angle -- the original SHAARP.si output
    panel's 'Effective refractive index' line plot. Wraps the validated anisotropic Snell solver
    :func:`shaarp.anisotropic.solve_snell_modes`; for the z-cut figure the crystal frame is the lab
    frame, so eps_lab = diag(n_o^2, n_o^2, n_e^2) at the fundamental omega. A GENERAL orientation
     passes the full lab-frame ``eps_omega_lab_full`` instead, so the tile shows the actual
    tilted-crystal mode indices. Pure -> plain arrays. NO new physics -- the indices come straight
    from the validated solver."""

    import math

    from .anisotropic import solve_snell_modes

    ne_w = n_omega if n_omega_e is None else n_omega_e
    eps_lab = (np.asarray(eps_omega_lab_full, dtype=complex).reshape(3, 3)
               if eps_omega_lab_full is not None
               else np.diag([n_omega ** 2, n_omega ** 2, ne_w ** 2]).astype(complex))
    theta_deg = np.linspace(0.0, 89.0, int(n_theta))
    n1 = np.empty(theta_deg.shape, dtype=float)
    n2 = np.empty(theta_deg.shape, dtype=float)
    for k, th in enumerate(theta_deg):
        modes = solve_snell_modes(eps_lab, complex(incident_index), math.radians(_desingularize_theta_deg(float(th))))
        a = float(np.real(modes.fast.refractive_index))
        b = float(np.real(modes.slow.refractive_index))
        n1[k], n2[k] = (a, b) if a <= b else (b, a)  # sort -> two continuous curves (no label swap)
    return {"theta_deg": theta_deg, "n_ordinary": n1, "n_extraordinary": n2}


def build_si_polarimetry_figure(point_group: str, *, theta_deg: float = 45.0, incident_index: float = 1.0,
                                precomputed_curve: dict | None = None, **kwargs):
    """The SHAARP.si SHG-Simulation output panel: the FOUR analysis tiles the original shows beneath
    the optical-setup schematics -- (top) the two reflected SHG polar plots I_p^{2omega}(phi,psi) and
    I_s^{2omega}(phi,psi+pi/2); (bottom-left) the Effective Refractive Index n(theta_i) line plot;
    (bottom-right) the incident-polarization Ellipticity locus (E_p vs E_s). With a FIXED analyzer set
    it falls back to a single analyzed-channel polar plot. Returned, not shown -> Agg-testable.

    ``precomputed_curve``: a dict already returned by :func:`si_polarimetry_curve` for the SAME
    (point_group, theta_deg, kwargs) -- skips the internal recompute so the caller can keep the
    plotted data for export (the export previously carried only the compat stage
    numerics, computed at the workflow's canonical polarization, NOT the plotted curves)."""

    import math

    import matplotlib.pyplot as plt

    c = (precomputed_curve if precomputed_curve is not None
         else si_polarimetry_curve(point_group, theta_deg=theta_deg, **kwargs))
    rad = np.radians(np.asarray(c["phi_deg"]))
    ell = c.get("ellipticity_deg", 0.0)
    sub = f"$\\theta_i$={theta_deg:.0f}°" + (f", $\\Delta\\delta$={ell:.0f}°" if ell else "")

    def _finish_polar(ax, values):
        """Intensity is non-negative, so the radial axis must start at the CENTER (rmin=0), never the
        negative band matplotlib auto-picks for all-zero data (which draws a misleading 'signal'
        circle). For a genuinely zero channel -- a symmetry-forbidden/centrosymmetric or vacuum
        material -- collapse to the center and label it, so 'no SHG' reads as no SHG."""
        vmax = float(np.max(np.abs(np.asarray(values, dtype=float)))) if len(values) else 0.0
        if vmax <= 1e-12:
            ax.set_ylim(0.0, 1.0)
            ax.set_yticklabels([])
            ax.text(0.5, 0.5, "SHG ≈ 0\n(symmetry-forbidden)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="0.35")
        else:
            ax.set_rmin(0.0)
    def _four_tile(chan_a, title_a, chan_b, title_b, suptitle):
        """The SHAARP.si SHG-Simulation multi-tile output: the two reflected-SHG polar plots (the two
        analyzer channels) on top, effective refractive index n(theta_i) and the incident-polarization
        ellipticity locus below. BOTH the co-rotating ("Rotate Analyzer") and fixed-psi ("Fixed analyzer")
        modes render through this same helper, so their layout is identical."""
        fig = Figure(figsize=(7.8, 7.6), layout="constrained")  # constrained -> re-flows on resize, no pyplot leak
        ax_a = fig.add_subplot(2, 2, 1, projection="polar")
        ax_a.plot(rad, chan_a, color="navy", lw=2)
        _finish_polar(ax_a, chan_a)
        ax_a.set_title(title_a, fontsize=9)
        ax_b = fig.add_subplot(2, 2, 2, projection="polar")
        ax_b.plot(rad, chan_b, color="darkorange", lw=2)
        _finish_polar(ax_b, chan_b)
        ax_b.set_title(title_b, fontsize=9)
        # (bottom-left) Effective refractive index n(theta_i): ordinary / extraordinary, Snell's law
        # exact mode indices -- full lab eps for general orientations; the (possibly complex /
        # biaxial) principal triple for principal-aligned materials; scalar n's only as last resort.
        _tile_eps = kwargs.get("eps_omega_lab_full")
        if _tile_eps is None and kwargs.get("eps_omega_principal") is not None:
            _tile_eps = np.diag(np.asarray(list(kwargs["eps_omega_principal"]), dtype=complex))
        idx = si_effective_index_curve(
            point_group, n_omega=kwargs.get("n_omega", 2.0), n_2omega=kwargs.get("n_2omega", 2.2),
            n_omega_e=kwargs.get("n_omega_e"), n_2omega_e=kwargs.get("n_2omega_e"),
            incident_index=incident_index, eps_omega_lab_full=_tile_eps)
        ax_n = fig.add_subplot(2, 2, 3)
        ax_n.plot(idx["theta_deg"], idx["n_ordinary"], color="navy", lw=1.8, label=r"$n^{1}$")
        ax_n.plot(idx["theta_deg"], idx["n_extraordinary"], color="darkorange", ls="--", lw=1.8, label=r"$n^{2}$")
        ax_n.set_xlabel(r"incident angle $\theta_i$ (deg)")
        ax_n.set_ylabel("refractive index")
        ax_n.set_title("Effective refractive index", fontsize=9)
        ax_n.legend(fontsize=8)
        ax_n.grid(alpha=0.3)
        # (bottom-right) incident-polarization Ellipticity locus E_p vs E_s (linear when Delta-delta=0)
        ax_e = fig.add_subplot(2, 2, 4)
        phi0 = math.radians(45.0)
        dd = math.radians(ell)
        t = np.linspace(0.0, 2.0 * np.pi, 240)
        ax_e.plot(math.cos(phi0) * np.cos(t), math.sin(phi0) * np.cos(t + dd), color="crimson", lw=2)
        ax_e.axhline(0.0, color="k", lw=0.5)
        ax_e.axvline(0.0, color="k", lw=0.5)
        ax_e.set_aspect("equal", adjustable="box")
        ax_e.set_xlabel(r"$E_p$")
        ax_e.set_ylabel(r"$E_s$")
        ax_e.set_title("Ellipticity (incident wave)", fontsize=9)
        ax_e.grid(alpha=0.3)
        fig.suptitle(suptitle, fontsize=10)
        return fig

    if "intensity_parallel" in c:  # Rotating analyzer: TRUE co-rotating psi = phi + offset (parallel +
        # perpendicular channels). Rui: "Rotate Analyzer" must co-rotate at every offset incl. 0
        # (the old offset!=0 gate let offset-0 fall through to the fixed p/s channels -> it rendered a
        # FIXED analyzer). Same 4-tile layout as the fixed-analyzer view below.
        off = c.get("corotating_offset_deg", 0.0)
        off_txt = f"{off:g}"
        ta = r"$I_\parallel^{2\omega}(\varphi,\ \psi{=}\varphi{+}" + off_txt + r"°)$  reflected"
        tb = r"$I_\perp^{2\omega}(\varphi,\ \psi{=}\varphi{+}" + off_txt + r"°{+}90°)$  reflected"
        st = ("SHAARP.si reflected SHG polarimetry — rotating (co-rotating) analyzer"
              + (f", offset {off:g}°" if off else "") + f"\npoint group {point_group}, {sub}")
        return _four_tile(c["intensity_parallel"], ta, c["intensity_perpendicular"], tb, st)

    if "intensity_analyzed" in c and c.get("fixed_phi_deg") is not None:
        # Fix Polarizer: incident polarization held at phi0, sweep the ANALYZER psi -> single I(psi) polar.
        fig = Figure(figsize=(5.6, 5.6)); ax = fig.add_subplot(111, projection="polar")
        lbl = rf"$I^{{2\omega}}(\psi),\ \varphi={c['fixed_phi_deg']:.0f}°$"
        ax.plot(rad, c["intensity_analyzed"], color="crimson", lw=2, label=lbl)
        _finish_polar(ax, c["intensity_analyzed"])
        ax.set_title(f"SHAARP.si reflected SHG — fixed polarizer φ={c['fixed_phi_deg']:.0f}°\n"
                     f"point group {point_group}, {sub}", fontsize=10)
        ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10), fontsize=9)
        fig.tight_layout()
        return fig

    if "intensity_analyzed" in c:  # Fixed analyzer psi: SAME 4-tile layout as Rotating
        # -> parallel channel at the fixed psi, perpendicular (crossed) at psi+90.
        psi0 = float(c.get("analyzer_deg", 0.0))
        ta = rf"$I^{{2\omega}}(\varphi,\ \psi={psi0:g}°)$  reflected"
        tb = rf"$I^{{2\omega}}(\varphi,\ \psi={psi0 + 90:g}°)$  reflected"
        chan_b = c.get("intensity_analyzed_perp", c.get("intensity_s"))
        st = (f"SHAARP.si reflected SHG polarimetry — fixed analyzer ψ={psi0:g}°"
              f"\npoint group {point_group}, {sub}")
        return _four_tile(c["intensity_analyzed"], ta, chan_b, tb, st)

    # bare default (no analyzer mode selected): the fixed p (psi=0) / s (psi=90) channels -- kept for
    # backward-compat callers/tests that request neither co-rotating nor a fixed psi.
    return _four_tile(
        c["intensity_p"], r"$I_p^{2\omega}(\varphi,\psi)$  reflected",
        c["intensity_s"], r"$I_s^{2\omega}(\varphi,\psi+\frac{\pi}{2})$  reflected",
        f"SHAARP.si reflected SHG polarimetry — point group {point_group}, {sub}")


def ml_polarimetry_curve(
    system,
    *,
    n_phi: int = 145,
    ellipticity_deg: float = 0.0,
    theta_deg: float | None = None,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    mrassumption: int = 0,
    fixed_phi_deg: float | None = None,
    corotating_offset_deg: float | None = None,
    analyzer_deg: float | None = None,
) -> dict[str, Any]:
    """Numeric MULTILAYER reflected + transmitted SHG polarimetry I_p(phi), I_s(phi) for a multilayer
    system -- the four-channel data behind the original SHAARP.ml SHG-Simulation output's FOUR polar
    plots (reflected I_p/I_s and transmitted I_p/I_s vs incident polarization phi).

    ``analyzer_deg`` (found by the causality audit): a FIXED analyzer at psi --
    previously the ML tab's "Fix Analyzer" mode was a DEAD CONTROL (the SI tab honored it, the
    ML branch never read it). Adds ``intensity_analyzed`` (reflected) and
    ``intensity_analyzed_trans`` = |sin psi * E_s + cos psi * E_p|^2 from the same validated per-phi
    Jones amplitudes, matching the SI channel convention.

    Computed by sweeping the incident polarization phi over the SAME validated point solve the GUI's
    numeric workflow uses (func:`solve_multilayer_shg_from_system_polarimetry`) and reading the
    reflected (s, p) and SHAARP.ml-selected transmitted (s, p) 2omega Jones amplitudes per phi
    (I = |.|^2). Pure/headless -> plain arrays so the polar figure is Agg-testable. NO new physics --
    every value comes from the validated boundary solve."""

    import math
    from dataclasses import replace

    from .multilayer_shg_boundary import (
        reflected_2omega_jones_sp,
        shaarp_ml_selected_transmitted_2omega_jones_sp,
        solve_multilayer_shg_from_system_polarimetry,
        transmitted_2omega_jones_sp,
    )

    phi_deg = np.linspace(0.0, 360.0, int(n_phi))
    th = _desingularize_theta_deg(float(system.polarimetry.theta_deg if theta_deg is None else theta_deg))
    # Map the Assumptions-panel selection to the validated single-pass flags (mirrors run_maker_fringes
    # exactly): 0=FMR (full MR), 1=JK (single-pass omega no-writeback + single-pass 2omega), 2=HH
    # (single-pass omega with writeback). So the 4 polar plots HONOR the chosen assumption, not just FMR.
    sp = {"single_pass_omega": False, "single_pass_omega_writeback": True, "single_pass_2omega": False}
    if mrassumption == 2:
        sp["single_pass_omega"] = True
    elif mrassumption == 1:
        sp["single_pass_omega"] = True
        sp["single_pass_omega_writeback"] = False
        sp["single_pass_2omega"] = True
    # 'Fix Polarizer' mode: one solve at phi0, sweep the ANALYZER psi -> reflected + transmitted I(psi)
    if fixed_phi_deg is not None:
        pol = replace(system.polarimetry, phi_deg=float(fixed_phi_deg), theta_deg=th,
                      ellipticity_deg=float(ellipticity_deg))
        res = solve_multilayer_shg_from_system_polarimetry(
            replace(system, polarimetry=pol), condition_threshold=condition_threshold,
            inhomogeneous_source_policy=inhomogeneous_source_policy, **sp)
        r_s, r_p = reflected_2omega_jones_sp(res.shg)
        try:
            t_s, t_p = transmitted_2omega_jones_sp(res.shg)
        except Exception:
            try:
                t_s, t_p = shaarp_ml_selected_transmitted_2omega_jones_sp(res.shg)
            except Exception:
                t_s, t_p = 0j, 0j
        psi = np.radians(phi_deg)
        i_refl = (np.abs(np.sin(psi) * r_s + np.cos(psi) * r_p) ** 2).astype(float)
        i_trans = (np.abs(np.sin(psi) * t_s + np.cos(psi) * t_p) ** 2).astype(float)
        return {"phi_deg": phi_deg, "intensity_reflected": i_refl, "intensity_transmitted": i_trans,
                "fixed_phi_deg": float(fixed_phi_deg), "theta_deg": th,
                "ellipticity_deg": float(ellipticity_deg)}
    i_p = np.empty(phi_deg.shape, dtype=float)
    i_s = np.empty(phi_deg.shape, dtype=float)
    i_pt = np.empty(phi_deg.shape, dtype=float)
    i_st = np.empty(phi_deg.shape, dtype=float)
    _amps = {k: np.zeros(phi_deg.shape, dtype=complex) for k in ("r_s", "r_p", "t_s", "t_p")}
    for k, ph in enumerate(phi_deg):
        pol = replace(system.polarimetry, phi_deg=float(ph), theta_deg=th,
                      ellipticity_deg=float(ellipticity_deg))
        res = solve_multilayer_shg_from_system_polarimetry(
            replace(system, polarimetry=pol),
            condition_threshold=condition_threshold,
            inhomogeneous_source_policy=inhomogeneous_source_policy,
            **sp,
        )
        r_s, r_p = reflected_2omega_jones_sp(res.shg)
        # transmitted polar uses the FULL substrate 2omega field projected to (s, p) -- the original
        # output's transmitted I_p/I_s polar plots (NOT the single Maker-selected wave). Fall back to
        # the Maker-selected wave only if the substrate does not carry exactly two waves.
        try:
            t_s, t_p = transmitted_2omega_jones_sp(res.shg)
        except Exception:
            try:
                t_s, t_p = shaarp_ml_selected_transmitted_2omega_jones_sp(res.shg)
            except Exception:
                t_s, t_p = 0j, 0j
        if corotating_offset_deg is not None:
            # co-rotating analyzer (FB1): psi = phi + offset; parallel channel projects the
            # (s, p) Jones onto (sin psi, cos psi), perpendicular onto (cos psi, -sin psi).
            psi = math.radians(float(ph) + float(corotating_offset_deg))
            i_p[k] = float(abs(math.sin(psi) * r_s + math.cos(psi) * r_p) ** 2)   # parallel
            i_s[k] = float(abs(math.cos(psi) * r_s - math.sin(psi) * r_p) ** 2)   # perpendicular
            i_pt[k] = float(abs(math.sin(psi) * t_s + math.cos(psi) * t_p) ** 2)
            i_st[k] = float(abs(math.cos(psi) * t_s - math.sin(psi) * t_p) ** 2)
        else:
            i_p[k] = float(abs(r_p) ** 2)
            i_s[k] = float(abs(r_s) ** 2)
            i_pt[k] = float(abs(t_p) ** 2)
            i_st[k] = float(abs(t_s) ** 2)
        _amps["r_s"][k], _amps["r_p"][k] = complex(r_s), complex(r_p)
        _amps["t_s"][k], _amps["t_p"][k] = complex(t_s), complex(t_p)
    out = {"phi_deg": phi_deg, "intensity_p": i_p, "intensity_s": i_s,
           "intensity_p_trans": i_pt, "intensity_s_trans": i_st,
           "theta_deg": th, "ellipticity_deg": float(ellipticity_deg)}
    if corotating_offset_deg is not None:
        out["corotating_offset_deg"] = float(corotating_offset_deg)
    if analyzer_deg is not None and corotating_offset_deg is None:
        # fixed analyzer: project the stored per-phi Jones amplitudes onto psi. The loop
        # keeps the raw amplitudes for this projection in the rotating branch below.
        psi = math.radians(float(analyzer_deg))
        psi_p = psi + math.pi / 2.0  # crossed analyzer -> the fixed view gets the SAME parallel+perp
        # 2x2 layout as the co-rotating view, so both render identically.
        out["intensity_analyzed"] = (np.abs(np.sin(psi) * _amps["r_s"]
                                            + np.cos(psi) * _amps["r_p"]) ** 2).astype(float)
        out["intensity_analyzed_perp"] = (np.abs(np.sin(psi_p) * _amps["r_s"]
                                                 + np.cos(psi_p) * _amps["r_p"]) ** 2).astype(float)
        out["intensity_analyzed_trans"] = (np.abs(np.sin(psi) * _amps["t_s"]
                                                  + np.cos(psi) * _amps["t_p"]) ** 2).astype(float)
        out["intensity_analyzed_trans_perp"] = (np.abs(np.sin(psi_p) * _amps["t_s"]
                                                       + np.cos(psi_p) * _amps["t_p"]) ** 2).astype(float)
        out["analyzer_deg"] = float(analyzer_deg)
    return out


def ml_beam_ellipses(system, *, theta_deg: float, phi_deg: float = 0.0,
                     ellipticity_deg: float = 0.0) -> dict[str, tuple[complex, complex]]:
    """Complex (s, p) Jones amplitudes of the INCIDENT, REFLECTED and TRANSMITTED fundamental
    beams for a multilayer at one geometry -- the data behind the original .ml output's
    "ellipticity of the incident, reflected and transmitted beams" figure (fidelity FB6).
    Reflection/transmission from the SAME validated linear solve the GUI Fresnel path uses."""

    import math
    from dataclasses import replace

    from .api import (
        _select_fundamental_transmitted_wave,
        _solve_gui_fresnel_fundamental,
        _sum_wave_frame_jones_sp,
    )
    from .multilayer_shg_boundary import _system_setup

    phi = math.radians(float(phi_deg))
    delta = math.radians(float(ellipticity_deg))
    j_in = (math.sin(phi) * complex(math.cos(delta), math.sin(delta)), math.cos(phi))
    case = replace(system, polarimetry=replace(system.polarimetry,
                                               theta_deg=_desingularize_theta_deg(float(theta_deg)),
                                               phi_deg=float(phi_deg)))
    setup = _system_setup(case)
    setup["incident_theta_rad"] = math.radians(_desingularize_theta_deg(float(theta_deg)))
    sol = _solve_gui_fresnel_fundamental(case, j_in, setup=setup)
    j_refl = _sum_wave_frame_jones_sp(sol.top_unknown)
    j_trans = _sum_wave_frame_jones_sp(_select_fundamental_transmitted_wave(sol.substrate_unknown))
    return {"incident": (complex(j_in[0]), complex(j_in[1])),
            "reflected": (complex(j_refl[0]), complex(j_refl[1])),
            "transmitted": (complex(j_trans[0]), complex(j_trans[1]))}


# the closed form in the sample azimuth. It was held OFF through F69-F71 because
# substituting psi into it disagreed with the validated route by ~32%. That was NOT this code:
# `_run_ml_partial_analytical_polarimetry` was passing SI mu/eps0 into the symbolic solve, which
# made the SHG inhomogeneous operator numerically singular (condition 7.7e17) and corrupted the
# very reference the closed form was being judged against -- see and
# tests/test_pa_crystal_symmetry.py. With F72's one-line units fix the closed form reproduces the
# numerically-rotated route to 3.9e-13 across the azimuth, so it is ON.
# Fenced by tests/test_ml_analytic_sample_rotation.py (Tier-1 substitute-psi, both directions).
ANALYTIC_AZIMUTH_ENABLED = True


def azimuth_closed_form_feasible(system, tol: float = 1e-9):
    """(ok, reason) — may the sample azimuth stay SYMBOLIC for this stack?

    A sample rotation turns BOTH eps and d. The closed form can only carry the azimuth through d:
    rotating eps would turn a biaxial crystal into a general ROTATED biaxial whose k_z solve is the
    full Booker quartic, with no closed form. So the analytic route is physical exactly when every
    rotating layer's lab eps is invariant under a rotation about the surface normal.

    This is a RUNTIME check, deliberately not a point-group lookup: the palette's "Quartz x-cut"
    carries an isotropic eps and IS invariant despite its cut. The measured split is bimodal --
    invariant materials land at <= 4.2e-16, dependent ones at >= 4.5e-2 -- so 1e-9 is unambiguous.
    Only the INTERIOR layers are checked: run_sample_rotation pins rotate_top/rotate_substrate
    False, so the half-spaces are unrotated on both routes."""
    import numpy as _np

    worst, worst_msg = 0.0, ""
    interiors = list(system.layers[1:-1]) if len(system.layers) > 2 else list(system.layers)
    for k, layer in enumerate(interiors, start=1):
        for omega in (True, False):
            eps = _epsilon_lab_of(layer.material, omega=omega)
            scale = max(1.0, float(_np.max(_np.abs(eps))))
            for ang in (37.0, 113.0):
                a = _np.deg2rad(ang)
                c, s = _np.cos(a), _np.sin(a)
                rz = _np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                res = float(_np.max(_np.abs(rz @ eps @ rz.T - eps))) / scale
                if res > worst:
                    worst = res
                    worst_msg = (f"layer {k} ({getattr(layer, 'name', '?')}) "
                                 f"eps({'w' if omega else '2w'}) is not invariant under rotation "
                                 f"about the surface normal (residual {res:.1e} > {tol:.0e})")
    if worst > tol:
        return False, (worst_msg + ": a closed form in the sample azimuth would need the "
                       "rotated-biaxial quartic. Ran the numeric azimuth sweep instead.")
    return True, f"eps is invariant under the azimuthal rotation to {worst:.1e}"


def ml_sample_rotation_grid(step_deg: float, ccw: bool = True):
    """(user grid, solver grid) for a sample-rotation sweep.

    The user grid is the original's ``Range[0, 360, step]`` -- inclusive, ascending, and what the
    polar plot's angular axis shows. The SOLVER grid carries the sign: the port's internal +azimuth
    is a right-hand rotation about the surface normal that points INTO the sample
    (meth:`CrystalOrientation.with_lab_azimuth_deg`), which reads CLOCKWISE to someone looking at
    the sample from the beam side -- so a counter-clockwise sweep (the GUI default)
    is the negated grid."""
    user = np.arange(0.0, 360.0 + 1e-9, float(step_deg))
    return user, (-user if ccw else user.copy())


def ml_sample_rotation_result(system, *, theta_deg: float, fixed_phi_deg: float = 0.0,
                              analyzer_psi_deg: float = 0.0, ellipticity_deg: float = 0.0,
                              step_deg: float = 10.0, ccw: bool = True,
                              rotate_polarizer: bool = False, rotate_analyzer: bool = False,
                              analyzer_offset_deg: float = 0.0):
    """Sample-rotation sweep with independent rotate/fix per element (generalized).

    Rui: polarizer, analyzer, and sample each carry their own rotate/fix choice, ANY
    combination legal, every rotating element following ONE common scan angle t (the user grid)
    through its rotation cos(t)/sin(t). Here the sample rotates (the sweep exists only then):

    - polarizer: phi(t) = t when ``rotate_polarizer``, else the fixed ``fixed_phi_deg``;
    - analyzer: psi(t) = t + offset when ``rotate_analyzer`` (offset applies only when the
      polarizer rotates too, mirroring the original's offset gate), else ``analyzer_psi_deg``;
    - sample: azimuth(t) = -t (CCW user sense) or +t (CW) into the solver, as in F69.

    The solver broadcasts per-point phi/psi arrays natively
    (func:`solve_multilayer_shg_sample_azimuth_sweep`), so no new machinery is needed. The
    original .ml pins phi and psi fixed during SampleRotate; the rotate flags are the author's
    extension, default False = the behavior byte-for-byte."""
    from dataclasses import replace

    from .api import run_sample_rotation

    user, solver = ml_sample_rotation_grid(step_deg, ccw)
    phi_val = user.copy() if rotate_polarizer else float(fixed_phi_deg)
    _offset = float(analyzer_offset_deg) if (rotate_polarizer and rotate_analyzer) else 0.0
    psi_val = (user + _offset) if rotate_analyzer else float(analyzer_psi_deg)
    pol = replace(system.polarimetry,
                  theta_deg=_desingularize_theta_deg(float(theta_deg)),
                  phi_deg=phi_val,
                  psi_deg=psi_val,
                  ellipticity_deg=float(ellipticity_deg))
    ra_sys = replace(system, polarimetry=pol)
    res = run_sample_rotation(ra_sys, solver)
    res.numeric["sample_azimuth_deg_user"] = user
    res.stages["sample_rotation"] = {
        "system": ra_sys, "azimuth_deg_user": user, "azimuth_deg_solver": solver,
        "step_deg": float(step_deg), "direction": "CCW" if ccw else "CW",
        "sign_to_solver": -1 if ccw else 1, "theta_deg": float(theta_deg),
        "fixed_phi_deg": float(fixed_phi_deg), "analyzer_psi_deg": float(analyzer_psi_deg),
        "ellipticity_deg": float(ellipticity_deg),
        "rotate_polarizer": bool(rotate_polarizer), "rotate_analyzer": bool(rotate_analyzer),
        "analyzer_offset_deg": _offset,
    }
    return res


def build_ra_scan_figure(result, *, title_suffix: str | None = None, azimuth_deg=None):
    """Rotational-anisotropy (RA) polar figure — 2ω SHG intensity vs SAMPLE azimuth, from the
    validated :func:`shaarp.run_sample_rotation` result. Two polar tiles (reflected | transmitted),
    each with the parallel + perpendicular analyzer channels; azimuth is the polar angle, so the
    crystal's rotational symmetry (e.g. MoS₂ six-fold) reads directly off the lobe count. Agg-testable."""

    n = result.numeric
    # the GUI passes the USER-sense grid (its CW/CCW choice); headless callers keep the
    # solver grid, so every pre-F69 figure is byte-identical.
    az = np.radians(np.asarray(n["sample_azimuth_deg"] if azimuth_deg is None else azimuth_deg,
                               dtype=float))

    def _r(v):  # intensities are real; take the real part defensively (complex-typed arrays exist)
        return np.real(np.asarray(v)).astype(float)

    fig = Figure(figsize=(7.8, 4.4), layout="constrained")
    axes = fig.subplots(1, 2, subplot_kw={"projection": "polar"})
    for ax, title, kpar, kper in (
        (axes[0], r"$I^{R,2\omega}$  reflected",
         "reflected_parallel_intensity", "reflected_perpendicular_intensity"),
        (axes[1], r"$I^{T,2\omega}$  transmitted",
         "transmitted_parallel_intensity", "transmitted_perpendicular_intensity"),
    ):
        ax.plot(az, _r(n[kpar]), color="navy", lw=2, label=r"$\parallel$ (p)")
        ax.plot(az, _r(n[kper]), color="darkorange", lw=2, label=r"$\perp$ (s)")
        ax.set_title(title, fontsize=9)
        ax.set_theta_zero_location("E")
        ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.12), fontsize=7, frameon=False)
    head = "SHAARP.ml rotational anisotropy — 2ω SHG vs sample azimuth"
    if title_suffix:
        head += f"\n{title_suffix}"
    fig.suptitle(head, fontsize=10)
    return fig


def build_ml_polarimetry_figure(curve, *, point_group: str = "", assumption_label: str | None = None,
                                ellipses: dict[str, tuple[complex, complex]] | None = None):
    """2x2 polar matplotlib Figure of the MULTILAYER SHG polarimetry: reflected I_p/I_s and
    transmitted I_p/I_s vs incident polarization phi (the original SHAARP.ml SHG-Simulation output's
    four polar plots). Takes the dict from :func:`ml_polarimetry_curve`. Agg-testable."""

    import matplotlib.pyplot as plt

    rad = np.radians(np.asarray(curve["phi_deg"]))
    th = curve.get("theta_deg")
    if curve.get("fixed_phi_deg") is not None:  # Fix Polarizer: reflected + transmitted I(psi)
        fig = Figure(figsize=(7.8, 4.4), layout="constrained"); axes = fig.subplots(1, 2, subplot_kw={"projection": "polar"})
        for ax, key, color, title in (
            (axes[0], "intensity_reflected", "navy", r"$I^{R,2\omega}(\psi)$  reflected"),
            (axes[1], "intensity_transmitted", "crimson", r"$I^{T,2\omega}(\psi)$  transmitted"),
        ):
            ax.plot(rad, np.asarray(curve[key], dtype=float), color=color, lw=2)
            ax.set_title(title, fontsize=9)
        head = f"SHAARP.ml SHG — fixed polarizer φ={curve['fixed_phi_deg']:.0f}°"
        if point_group:
            head += f" — point group {point_group}"
        if assumption_label:
            head += f"\nAssumption Used: {assumption_label}"
        fig.suptitle(head, fontsize=10)
        return fig
    # Which four channels + titles + heading go in the 2x2 (or 2x3-with-ellipses) grid. Fixed-analyzer
    # ψ now renders through the SAME grid as Rotating: parallel = ψ, perpendicular =
    # ψ+90 (the old 1x2 early-return made its layout differ from Rotating's 4-polar).
    psi_fix = curve.get("analyzer_deg")
    co = curve.get("corotating_offset_deg")
    if psi_fix is not None:  # Fixed analyzer psi -> channels at psi and psi+90
        p, pp = f"{float(psi_fix):g}", f"{float(psi_fix) + 90:g}"
        chan_keys = ("intensity_analyzed", "intensity_analyzed_perp",
                     "intensity_analyzed_trans", "intensity_analyzed_trans_perp")
        tile_titles = (rf"$I^{{R,2\omega}}(\varphi,\ \psi={p}°)$  reflected",
                       rf"$I^{{R,2\omega}}(\varphi,\ \psi={pp}°)$  reflected",
                       rf"$I^{{T,2\omega}}(\varphi,\ \psi={p}°)$  transmitted",
                       rf"$I^{{T,2\omega}}(\varphi,\ \psi={pp}°)$  transmitted")
        mode_head = f"SHAARP.ml SHG — fixed analyzer ψ={float(psi_fix):g}°"
    elif co is not None:  # co-rotating analyzer labels (FB1): psi tracks phi at the offset
        chan_keys = ("intensity_p", "intensity_s", "intensity_p_trans", "intensity_s_trans")
        t_par = r"$I^{2\omega}(\varphi,\ \psi{=}\varphi{+}$" + f"{co:g}" + r"$°)$"
        t_per = r"$I^{2\omega}(\varphi,\ \psi{=}\varphi{+}$" + f"{co:g}" + r"$°{+}90°)$"
        tile_titles = (t_par + "  reflected", t_per + "  reflected",
                       t_par + "  transmitted", t_per + "  transmitted")
        mode_head = "SHAARP.ml SHG polarimetry — rotating (co-rotating) analyzer" + (
            f", offset {co:g}°" if co else "")
    else:
        chan_keys = ("intensity_p", "intensity_s", "intensity_p_trans", "intensity_s_trans")
        tile_titles = (r"$I_p^{R,2\omega}(\varphi,\psi)$  reflected",
                       r"$I_s^{R,2\omega}(\varphi,\psi+\frac{\pi}{2})$  reflected",
                       r"$I_p^{T,2\omega}(\varphi,\psi)$  transmitted",
                       r"$I_s^{T,2\omega}(\varphi,\psi+\frac{\pi}{2})$  transmitted")
        mode_head = "SHAARP.ml SHG polarimetry"
    if ellipses:
        # 2x3 layout: the four polar tiles + the original's incident/reflected/transmitted
        # beam-ellipticity tile (fidelity FB6)
        fig = Figure(figsize=(10.6, 7.4), layout="constrained")
        gs = fig.add_gridspec(2, 3)
        axes = np.empty((2, 2), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0], projection="polar")
        axes[0, 1] = fig.add_subplot(gs[0, 1], projection="polar")
        axes[1, 0] = fig.add_subplot(gs[1, 0], projection="polar")
        axes[1, 1] = fig.add_subplot(gs[1, 1], projection="polar")
        ax_e = fig.add_subplot(gs[:, 2])
        t = np.linspace(0.0, 2.0 * np.pi, 241)
        colors = {"incident": "black", "reflected": "crimson", "transmitted": "navy"}
        for name, (js, jp) in ellipses.items():
            e_s = np.real(complex(js) * np.exp(-1j * t))
            e_p = np.real(complex(jp) * np.exp(-1j * t))
            scale = max(float(np.max(np.hypot(e_s, e_p))), 1e-30)
            ax_e.plot(e_s / scale, e_p / scale, color=colors.get(name, "gray"), lw=1.8,
                      label=name)
        # FIXED symmetric limits + equal aspect (report 'ellipticity deformed'):
        # with linear polarization (ellipticity 0) the 'ellipse' is a degenerate LINE; without
        # fixed limits matplotlib autoscaled the flat axis to float noise (~1e-4) and equal
        # aspect then squeezed the box into a hyper-thin strip. The curves are unit-normalized,
        # so +/-1.1 always contains them and the tile stays a square.
        ax_e.set_aspect("equal", adjustable="box")
        ax_e.set_xlim(-1.1, 1.1)
        ax_e.set_ylim(-1.1, 1.1)
        ax_e.set_xlabel(r"$E_s$ (norm.)", fontsize=9)
        ax_e.set_ylabel(r"$E_p$ (norm.)", fontsize=9)
        ax_e.set_title("Beam ellipticity (ω): incident / reflected / transmitted\n"
                       "(linear polarization draws as a straight line)", fontsize=8.5)
        ax_e.legend(fontsize=8, loc="upper right")
        ax_e.grid(True, alpha=0.3)
    else:
        fig = Figure(figsize=(7.4, 7.6), layout="constrained")
        axes = fig.subplots(2, 2, subplot_kw={"projection": "polar"})
    colors4 = ("navy", "crimson", "navy", "crimson")  # reflected ∥/⊥, transmitted ∥/⊥
    for ax, key, title, color in zip(
        (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]), chan_keys, tile_titles, colors4):
        ax.plot(rad, np.asarray(curve[key], dtype=float), color=color, lw=2)
        ax.set_title(title, fontsize=9)
    head = mode_head
    if point_group:
        head += f" — point group {point_group}"
    if th is not None:
        head += f", $\\theta_i$={float(th):.0f}°"
    if assumption_label:
        head += f"\nAssumption Used: {assumption_label}"
    fig.suptitle(head, fontsize=10)
    return fig


def mask_maker_singularities(intensity: np.ndarray) -> tuple[np.ndarray, int]:
    """Mask ISOLATED numerical singularities in a Maker-fringe intensity scan so the curve is
    continuous (the physical Maker fringe is smooth). At isolated incidence angles an eigenmode
    degeneracy makes the field solve singular and the solver returns a sentinel (typically an exact
    0, occasionally a huge spike); such a point departs from BOTH neighbors by orders of magnitude
    while the surrounding curve is smooth -- the signature of a singularity, NOT a resolved fringe
    feature (a real peak/null is spread over several points). Those points are set to NaN (so a plot
    leaves a gap rather than drawing a spurious drop/spike) and the count returned, in keeping with
    the package principle that singularities are FLAGGED, never silently hidden. Returns
    ``(masked_intensity, n_masked)``."""

    y = np.asarray(intensity, dtype=float).copy()
    n = len(y)
    if n < 5:
        return y, 0
    absy = np.abs(y)
    scale = float(np.nanmax(absy)) or 1.0
    zthr = 1e-12 * scale                     # FLOAT-EXACT zero = the solver's singularity sentinel
    pos = absy[absy > 1e-9 * scale]
    med = float(np.median(pos)) if pos.size else 0.0
    if med <= 0:                             # a genuinely-zero / all-noise curve: nothing to flag
        return y, 0
    big = 0.3 * med                          # neighbors this large flanking a 0 => an isolated HOLE
    out = y.copy()
    masked = 0

    def _nearest_finite(k, step):
        j = k + step
        while 0 <= j < n and absy[j] <= zthr:
            j += step
        return j if 0 <= j < n else None

    for k in range(n):
        # (1) SENTINEL zero: an exact 0 whose nearest FINITE neighbors on BOTH sides are large (a
        # hole punched in a high region) AND the zero-run is short (isolated). A PHYSICAL Maker
        # minimum reaches 0 with SMALL neighbors (the curve dips smoothly), so it is NOT flagged --
        # the robust discriminator (sentinel = hole in a high region; minimum = smooth dip to 0).
        if absy[k] <= zthr:
            li, ri = _nearest_finite(k, -1), _nearest_finite(k, +1)
            if li is not None and ri is not None and (ri - li - 1) <= 3 \
                    and min(absy[li], absy[ri]) > big:
                out[k] = np.nan
                masked += 1
            continue
        # (2) isolated spike: a single point >> its finite neighbors (degenerate solve blowing up).
        if 0 < k < n - 1:
            a, b = absy[k - 1], absy[k + 1]
            if max(a, b) > 1e-6 * scale and absy[k] > 5.0 * max(a, b) and absy[k] > 0.3 * med:
                out[k] = np.nan
                masked += 1
    return out, masked


def count_discontinuities(y: np.ndarray) -> int:
    """Number of isolated discontinuities (sentinel holes / spikes) in a curve -- a CONTINUITY
    CHECK for validation, NOT a masker. Per the author's standing lens: "every plot should be
    continuous; a discontinuity means something is off -- investigate." So this is used to ASSERT a
    plotted SHG-vs-angle curve is continuous (count == 0), prompting a root-cause fix when it isn't
    -- the package does NOT mask discontinuities to hide them. (The Maker transmitted-channel
    discontinuity that motivated this was traced to an eigenmode-ordering bug and FIXED at the
    source -- _transmitted_waves_for_maker_policy selects the dominant transmitted wave.)"""

    _masked, n = mask_maker_singularities(np.asarray(y, dtype=float))
    return int(n)


def build_schematic_3d_figure(layers, theta_deg: float = 20.0):
    """3D schematic of the sample stack (the original GUIs' 3D orientation aid): translucent layer
    slabs stacked along lab L3, the surface normal, the plane of incidence (the L1-L3 plane), and
    the incident/reflected ray pair at ``theta_deg``. ``layers`` is the same ``(name,
    thickness_um)`` list as :func:`build_schematic_figure`. Pure -> Agg-testable; illustrative
    geometry (a schematic, not a physics output)."""

    import math

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = Figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False  # draw order wins: slabs first, rays/labels on top
    names = [str(n) for n, _t in layers]
    colors = ["#9ecae1", "#fdd0a2", "#a1d99b", "#d4b9da", "#fcbba1", "#cccccc"]
    # display thickness: semi-infinite media get a fixed slab, finite layers a readable minimum
    th_disp = [0.8 if t is None else max(0.45, min(1.6, float(t))) for _n, t in layers]
    z_top = 0.0
    half = 1.4  # slab half-width in L1/L2
    for i, ((name, _t), dz) in enumerate(zip(layers, th_disp)):
        if i == 0:
            z0, z1 = 0.0, dz  # the incidence medium sits ABOVE the first interface (rays inside)
        else:
            z0, z1 = z_top - dz, z_top
        c = colors[i % len(colors)]
        x0, x1, y0, y1 = -half, half, -half, half
        corners = {
            "top": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            "bottom": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            "front": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            "back": [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            "left": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            "right": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        }
        ax.add_collection3d(Poly3DCollection(
            list(corners.values()), facecolors=c, edgecolors="#666666",
            linewidths=0.5, alpha=(0.18 if i == 0 else 0.45), zorder=1 + i,
        ))
        ax.text(half * 1.08, half * 1.08, (z0 + z1) / 2, name, fontsize=8, va="center")
        if i != 0:
            z_top = z0
    # rays in the L1-L3 plane of incidence (above the first interface at z=0)
    th = math.radians(float(theta_deg))
    ray_len = 1.2
    dx, dz = math.sin(th) * ray_len, math.cos(th) * ray_len
    ax.plot([-dx, 0.0], [0.0, 0.0], [dz, 0.0], color="crimson", lw=2.5, zorder=50)
    ax.plot([0.0, dx], [0.0, 0.0], [0.0, dz], color="crimson", lw=2.5, linestyle="--", zorder=50)
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, ray_len * 1.15], color="black", lw=1, linestyle=":", zorder=50)
    # label placement: words extend AWAY from each other (ha right/left) with a minimum spread --
    # centered labels overlapped under the 3D projection's x-foreshortening at any angle (release
    # audit), not just at theta=0.
    lx = max(dx, 0.55)
    ax.text(-lx, 0, dz * 1.08, "incident", color="crimson", fontsize=8, ha="right", zorder=60)
    ax.text(lx, 0, dz * 1.08, "reflected", color="crimson", fontsize=8, ha="left", zorder=60)
    ax.set_xlabel("L1")
    ax.set_ylabel("L2")
    ax.set_zlabel("L3")
    ax.set_title(f"sample stack (3D), theta = {float(theta_deg):g} deg", fontsize=10)
    below_depth = sum(th_disp[1:]) if len(th_disp) > 1 else 0.6
    ax.set_xlim(-half * 1.3, half * 1.3)
    ax.set_ylim(-half * 1.3, half * 1.3)
    ax.set_zlim(-below_depth * 1.05, max(ray_len * 1.25, th_disp[0] * 1.2))
    ax.view_init(elev=18, azim=-60)
    # Shrink the 3D box INSIDE its axes (zoom<1) so the box corners, the layer-name labels (placed
    # just outside the box) and the L1/L2/L3 axis labels all stay within the canvas at ANY pane size
    # -- a deep multi-layer stack on a small/near-square canvas was overflowing and getting clipped
    # (the "geometry plot is clipped after Update" bug). The axes still fills the figure; only the
    # drawn box is inset, leaving label room. Small top/right margins keep the title + layer names.
    try:
        ax.set_box_aspect(None, zoom=0.78)
    except TypeError:  # older matplotlib without the zoom kwarg
        pass
    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.0, top=0.93)
    return fig


def build_orientation_axes_figure(orientation, *, title: str = "") -> "Figure":
    """The original .ml Set-Material view's KEY orientation visualization (fidelity FB4): the
    crystal-physics axes Z1 Z2 Z3 drawn against the lab axes L1 L2 L3 for the selected layer.
    ``orientation`` is a CrystalOrientation; its rotation matrix rows are the Z axes expressed in
    the lab frame. Pure -> Agg-testable."""

    from matplotlib.lines import Line2D

    fig = Figure(figsize=(3.4, 2.8))  # compact (the Qt canvas caps at 340x280 px)
    ax = fig.add_subplot(111, projection="3d")
    rot = np.asarray(orientation.rotation_matrix(), dtype=float)

    def _perp(v):
        # a unit vector perpendicular to v -- used to nudge the label sideways so a crystal axis
        # that COINCIDES with a lab axis (the z-cut/identity case) still has readable, separated
        # labels instead of Z_i printed on top of L_i.
        v = np.asarray(v, dtype=float)
        ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        p = np.cross(v, ref)
        n = np.linalg.norm(p)
        return p / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    # lab triad (black solid) and crystal-physics triad (crimson dashed), unit arrows from origin.
    # The crystal arrows are drawn slightly SHORTER (0.9) than the lab arrows (1.0) so that when they
    # coincide, both arrowheads stay visible (the black tip pokes past the crimson one) rather than
    # one hiding the other; labels are pushed to OPPOSITE sides of the shared axis so they never stack.
    for k, lab in enumerate(("$L_1$", "$L_2$", "$L_3$")):
        v = np.eye(3)[k]
        ax.quiver(0, 0, 0, *v, color="black", lw=1.6, arrow_length_ratio=0.11)
        # D10: at the shipped 340x280 canvas 0.13 data units is a few pixels, so
        # for the DEFAULT z-cut identity orientation "Z3" printed on top of "L3". Push the two
        # triads' labels further apart -- they are only coincident when the crystal IS the lab.
        ax.text(*(v * 1.20 + _perp(v) * 0.30), lab, color="black", fontsize=10,
                ha="center", va="center")
    for k, lab in enumerate(("$Z_1$", "$Z_2$", "$Z_3$")):
        v = rot[k]
        ax.quiver(0, 0, 0, *(v * 0.9), color="crimson", lw=2.2, arrow_length_ratio=0.12,
                  linestyle=(0, (4, 2)))
        ax.text(*(v * 1.20 - _perp(v) * 0.30), lab, color="crimson", fontsize=10,
                ha="center", va="center", fontweight="bold")
    lim = 1.32
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)  # fill the box, shed the internal padding
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title or "Crystal-physics axes $Z_i$ vs lab axes $L_i$", fontsize=9, pad=2)
    # legend makes solid-black vs dashed-crimson unambiguous even when the two triads overlap.
    # OUTSIDE the axes (at upper-left it overlapped the up-pointing L3/Z3
    # arrows). F66 the GUI walkthrough: at the compact 340 px width the old RIGHT gutter clipped
    # "crystal $Z_i$" and top=1.0 clipped the title to a stray sliver -- the legend now runs
    # ALONG THE BOTTOM (figure-level, two columns: no side gutter, still clear of the arrows)
    # and the title gets real headroom.
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.10)
    fig.legend(handles=[Line2D([0], [0], color="black", lw=1.6, label="lab $L_i$"),
                        Line2D([0], [0], color="crimson", lw=2.2, linestyle=(0, (4, 2)),
                               label="crystal $Z_i$")],
               loc="lower center", ncol=2, fontsize=8, frameon=False,
               handlelength=1.6, columnspacing=1.2, borderaxespad=0.1)
    ax.view_init(elev=18, azim=-60)
    return fig


def build_orientation_axes_figure_2d(orientation, *, title: str = "") -> "Figure":
    """The 2D companion to :func:`build_orientation_axes_figure`, looking DOWN the lab L3
    (the surface normal) -- the top view, which reads an in-plane azimuth far better than a 3D
    triad does. Out-of-plane tilt is not hidden: an axis tipped out of the plane projects SHORT,
    and its L3 component is printed beside the label. Same legend labels and the same six axis
    texts as the 3D view, so either mode satisfies the readability fence. Pure -> Agg-testable."""

    import math

    from matplotlib.lines import Line2D

    fig = Figure(figsize=(3.4, 2.8))
    ax = fig.add_subplot(111)
    rot = np.asarray(orientation.rotation_matrix(), dtype=float)

    def _perp2(vx, vy):
        n = math.hypot(vx, vy)
        return (-vy / n, vx / n) if n > 1e-9 else (0.0, 1.0)

    # lab triad (black solid): L1 -> +x, L2 -> +y, L3 -> out of the page (drawn as a dot marker)
    for k, lab in enumerate(("$L_1$", "$L_2$", "$L_3$")):
        v = np.eye(3)[k]
        if k == 2:                      # the normal points at the viewer -- circled dot, not an arrow
            ax.plot([0], [0], marker="o", ms=9, mfc="none", mec="black", mew=1.4, zorder=3)
            ax.plot([0], [0], marker=".", ms=3, color="black", zorder=4)
            ax.text(0.13, -0.13, lab, color="black", fontsize=10, ha="left", va="top")
            continue
        ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6))
        px, py = _perp2(v[0], v[1])
        ax.text(v[0] * 1.14 + px * 0.10, v[1] * 1.14 + py * 0.10, lab, color="black", fontsize=10,
                ha="center", va="center")

    # crystal triad (crimson dashed), projected onto the L1-L2 plane, drawn slightly shorter so a
    # coincident axis still shows the black tip beyond the crimson one (same trick as the 3D view)
    _placed: list[tuple[float, float]] = []
    _bracketed = False
    for k, lab in enumerate(("$Z_1$", "$Z_2$", "$Z_3$")):
        v = rot[k]
        x, y = float(v[0]) * 0.9, float(v[1]) * 0.9
        if math.hypot(x, y) < 0.06:     # (near-)normal axis: mark it, do not draw a stub arrow
            ax.plot([0], [0], marker="o", ms=13, mfc="none", mec="crimson", mew=2.0,
                    linestyle=(0, (4, 2)), zorder=3)
            ax.text(-0.13, -0.13, lab, color="crimson", fontsize=10, ha="right", va="top",
                    fontweight="bold")
            continue
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.2,
                                    linestyle=(0, (4, 2))))
        px, py = _perp2(x, y)
        out = float(v[2])
        txt = lab if abs(out) < 0.05 else rf"{lab} ({out:+.2f})"
        _bracketed = _bracketed or abs(out) >= 0.05
        # place the label at a MINIMUM radius along its own direction: an axis tilted far out of
        # plane projects SHORT, and a label pinned to that short arrow lands on top of its
        # neighbour's near the origin.
        r = math.hypot(x, y)
        # D9: clear the ARROW TIP too, not just other labels -- a wide multi-character
        # label centred only ~0.13 beyond a short projected arrow was overprinted by its own head.
        rl = max(r + 0.34, 0.62)
        # ...and step it further out while it would sit on a label already placed. Two crystal axes
        # can PROJECT into nearly the same direction even when they are orthogonal in 3D, so a
        # per-axis offset alone does not separate them.
        for _ in range(4):
            lx, ly = ((x / r * rl, y / r * rl) if r > 1e-9 else (0.0, rl))
            lx, ly = lx - px * 0.17, ly - py * 0.17
            if all(math.hypot(lx - qx, ly - qy) > 0.34 for qx, qy in _placed):
                break
            rl += 0.30
        _placed.append((lx, ly))
        ax.text(lx, ly, txt, color="crimson", fontsize=9,
                ha="center", va="center", fontweight="bold")

    # says what the bracketed number on a crystal label means -- without it the projection reads
    # as if the out-of-plane tilt had simply been dropped
    if _bracketed:      # D8: only when a bracket was actually drawn -- the identity
        # orientation prints none, yet the caption claimed otherwise
        ax.text(0.0, -1.58, r"( ) = out-of-plane component along $L_3$", fontsize=7,
                color="0.35", ha="center", va="top")
    lim = 1.62
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.92, lim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title or "Crystal-physics axes $Z_i$ vs lab axes $L_i$", fontsize=9, pad=2)
    # same bottom legend as the 3D view -- and real margins, because a 2D canvas is graded by the
    # layout harness's edge-ink rule rather than the 3D aspect rule.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.14)
    fig.legend(handles=[Line2D([0], [0], color="black", lw=1.6, label="lab $L_i$"),
                        Line2D([0], [0], color="crimson", lw=2.2, linestyle=(0, (4, 2)),
                               label="crystal $Z_i$")],
               loc="lower center", ncol=2, fontsize=8, frameon=False,
               handlelength=1.6, columnspacing=1.2, borderaxespad=0.1)
    return fig


def schematic_indices_for(system):
    """Per-layer ``(n_omega, n_2omega)`` for :func:`build_schematic_figure`'s Snell bending.

    Uses the ORDINARY (in-plane) index sqrt(eps_xx) of each layer's lab dielectric tensor -- the
    schematic draws one ray per medium, so it needs one index per medium; birefringent splitting is
    not what this picture is for. Returns None if anything is unavailable, which makes the figure
    fall back to its illustrative constant rather than fail."""
    try:
        out = []
        for layer in system.layers:
            pair = []
            for omega in (True, False):
                eps = _epsilon_lab_of(layer.material, omega=omega)
                n = float(np.sqrt(complex(np.asarray(eps)[0, 0]).real))
                pair.append(n if n > 0.05 else 1.0)
            out.append((pair[0], pair[1]))
        return out
    except Exception:
        return None


def build_schematic_figure(layers, theta_deg: float = 20.0, *, wavelength_um: float | None = None,
                           assumption: str | None = None, fmr_submode: str | None = None,
                           indices=None):
    """2D schematic of the optical setup (the original GUIs' orientation aid): the layer stack as
    horizontal bands (names + thicknesses; semi-infinite media open-ended), the surface normal, and
    the ray geometry -- incident/reflected fundamental (red) at ``theta_deg`` from the normal and
    the reflected/transmitted SHG (blue dashed). ``layers`` is a list of ``(name, thickness_um)``
    tuples with ``None`` thickness for semi-infinite media (e.g. from a MultilayerSystem's layers,
    or ``[("air", None), (material, None)]`` for the single-interface tab). Pure -> Agg-testable;
    the ray angles are illustrative (a schematic, not a physics output).

    ``indices``: optional per-layer ``(n_omega, n_2omega)`` used for the DRAWN refraction, so
    the ray bends by real Snell angles rather than an illustrative constant. ``None`` entries and a
    missing list fall back to n = 1 for the top medium and 1.5/1.55 elsewhere, which reproduces the
    pre-F74 drawing for callers that do not pass indices.

    ``assumption`` (ML, fidelity FB5): draw the ORIGINAL .ml SHG-view multiple-reflection wave
    ladder inside the first film -- red omega rays, blue 2-omega rays, orange inhomogeneous waves
    reflecting the FMR sub-assumption; Jerphagnon-Kurtz = single pass; Herman-Hayden = single-pass
    omega with the 2-omega ladder."""

    import math

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layers = [(str(n), (None if t is None else float(t))) for n, t in layers]
    n_layers = len(layers)
    if n_layers < 2:
        raise ValueError("A schematic needs at least two media (top + substrate).")

    # display heights: semi-infinite media fixed; finite layers min-height with the true thickness labeled
    heights = [1.2 if t is None else max(0.55, min(1.6, 0.55 + 0.25 * math.log10(max(t, 1e-4) / 1e-4))) for _n, t in layers]
    total = sum(heights)
    fig = Figure(figsize=(7.2, 4.6)); ax = fig.subplots()
    colors = ["#eef4fb", "#cfe2f3", "#f9e3c8", "#dcead5", "#e8d8ee", "#f3f3f3"]
    y_top = 0.0
    interfaces = []
    for i, ((name, t), h) in enumerate(zip(layers, heights)):
        y0 = y_top - h
        ax.add_patch(Rectangle((0.0, y0), 10.0, h, facecolor=colors[i % len(colors)], edgecolor="0.4", lw=0.8))
        label = name if t is None else f"{name}   (h = {t:g} µm)"
        ax.text(0.25, y0 + h / 2, label, va="center", fontsize=9)
        interfaces.append(y_top)
        y_top = y0
    # ---- ONE incidence point drives every ray -------------------------------------------
    # Pre-F74 the fan sat at x0 = 6.0 and the internal ladder at a hardcoded xm = 1.4, so the zigzag
    # was drawn on the far side of the figure from the ray that produced it. Everything below is
    # anchored to (x0, y_surf): the ladder marches away from the incidence point, and the
    # transmitted rays continue from where the ladder leaves the film.
    x0 = 3.2                    # left enough for the incident ray + label, right enough for the zigzag
    X_MAX = 9.3                 # keep ink off the frame (layout harness edge-ink gate)
    y_surf = interfaces[1] if len(interfaces) > 1 else 0.0
    # Draw the TRUE incidence angle (theta_i = 0 -> vertical/normal incidence, not a tilted ray); cap
    # only at the very top for drawability. The earlier max(5,..) floor made normal incidence look
    # oblique, which is physically wrong.
    th = math.radians(min(float(theta_deg), 89.0))
    L = 0.85 * heights[0]  # rays live inside the top medium band
    dx, dy = math.sin(th) * L, math.cos(th) * L

    def _n_pair(i):
        """(n_omega, n_2omega) for layer i -- real values when the caller supplies them."""
        if indices is not None and i < len(indices) and indices[i]:
            nw, n2 = indices[i]
            nw = float(nw) if nw else 1.0
            return nw, (float(n2) if n2 else nw)
        return (1.0, 1.0) if i == 0 else (1.5, 1.55)

    n_top_w, n_top_2w = _n_pair(0)

    def _theta_in(i, two=False):
        """Snell angle inside layer i. The 2w in-plane momentum is inherited from the pump, so it
        refracts against n(2w) with the SAME n_top*sin(theta_i) -- which is why the transmitted
        omega and 2w split apart inside a dispersive film but stay collinear in the ambient."""
        n = max(1e-6, _n_pair(i)[1 if two else 0])
        return math.asin(min(0.999, (n_top_2w if two else n_top_w) * math.sin(th) / n))

    # Minimum lateral separation so the incident and reflected beams stay DISTINCT at/near normal
    # incidence (where dx -> 0 and they would otherwise coincide on the normal). Shifting incident left
    # and reflected right by the SAME gap leaves each ray VECTOR equal to (-/+dx, dy) -- i.e. exactly
    # the true angle -- so this never tilts the rays; it only slides the parallel pair apart when the
    # natural fan (dx) is too small to read. At theta=0 -> two vertical arrows; at oblique theta they
    # meet at the surface point.
    # D1: this legibility slide must stay SMALL. It shifts the incident and
    # reflected rays to opposite sides of x0 so they remain distinct where the true geometry would
    # stack them on the normal -- but at 0.5 it put their surface points ~72 px apart at theta = 0,
    # which is the DEFAULT angle, so the beam appeared to vanish at one point and reappear at
    # another. 0.16 keeps them legible while reading as one point; the explicit marker below makes
    # the shared incidence point unambiguous.
    gap = max(0.0, 0.16 - dx)
    y_bottom = y_surf - sum(heights[1:])
    ax.plot([x0, x0], [y_surf + 0.9 * heights[0], y_bottom * 0.98 + y_surf * 0.02], color="0.4", ls=":", lw=1)  # normal
    ax.plot([x0], [y_surf], marker="o", ms=4.5, color="0.25", zorder=6)   # D1: THE incidence point
    ax.annotate("", xy=(x0 - gap, y_surf), xytext=(x0 - gap - dx, y_surf + dy),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2))           # incident omega (down to surface)
    ax.annotate("", xy=(x0 + gap + dx, y_surf + dy), xytext=(x0 + gap, y_surf),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2))           # reflected omega (specular, up)
    # Reflected 2w is COLLINEAR with the reflected omega -- both leave into the SAME medium (air) at
    # the specular angle = theta_i. (The 2w nonlinear source carries twice the in-plane wavevector,
    # 2*k_x^omega, and the air dispersion doubles |k| too, so sin(theta_2w,refl) = sin(theta_i).)
    # Draw it as a parallel dashed beam offset perpendicular to the ray so both are visible at the SAME
    # angle. (Only the TRANSMITTED omega/2w split apart, via crystal dispersion.)
    ox, oy = 0.22 * dy, -0.22 * dx  # offset perpendicular to the reflected ray (horizontal at normal)
    # D4: at grazing incidence this perpendicular slide pushed the ENTIRE "2w refl"
    # arrow BELOW the surface -- a reflected wave drawn inside the film it reflects from. Clamp the
    # offset so both endpoints stay in the incident medium.
    if oy < 0.0:
        oy = max(oy, -0.35 * min(dy, 0.92 * dy))
    ax.annotate("", xy=(x0 + gap + ox + 0.92 * dx, y_surf + oy + 0.92 * dy), xytext=(x0 + gap + ox, y_surf + oy),
                arrowprops=dict(arrowstyle="-|>", color="navy", lw=1.6, linestyle="--"))  # reflected 2w (collinear)

    def _descend(x_start, y_start, first_layer, *, two, color, lw, ls, alpha=1.0):
        """Refract a ray DOWN through every remaining band, one arrow per layer, each ending on the
        interface it actually reaches. Pre-F74 this was a single straight arrow to
        y_surf - 0.88*sum(heights[1:]), which sliced through every interface at an angle that was
        no refraction of anything."""
        x, y = float(x_start), float(y_start)
        for j in range(first_layer, n_layers):
            h = heights[j] * (0.6 if j == n_layers - 1 else 1.0)   # stop short in the open substrate
            run = h * math.tan(_theta_in(j, two))
            if x + run > X_MAX:                                     # never run into the frame
                run = max(0.0, X_MAX - x)
            ax.annotate("", xy=(x + run, y - h), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                        alpha=alpha))
            x, y = x + run, y - h
        return x, y

    # ---- multiple-reflection wave ladder inside the first FILM (original .ml SHG view, FB5) ----
    finite = [i for i, (_n, tt) in enumerate(layers) if tt is not None]
    ladder_end_x, ladder_end_y, ladder_layer = x0, y_surf, 1
    if assumption is not None and finite:
        fi = finite[0]
        f_top = interfaces[fi]
        f_bot = f_top - heights[fi]
        fh = heights[fi]
        # the ladder slant FOLLOWS
        # the incidence angle through Snell, so at theta_i = 0 every pass is VERTICAL and the passes
        # render as parallel vertical arrows separated by a legibility offset -- the offset NEVER
        # tilts a ray (both endpoints move together), it only slides stacked passes apart.
        seg = fh * math.tan(_theta_in(fi))          # TRUE horizontal run per pass (0 at normal incidence)
        seg2 = fh * math.tan(_theta_in(fi, True))   # the 2w pass runs at ITS refracted angle
        off = max(0.0, 0.30 - seg)                  # legibility spacing when seg is tiny

        def _ladder(x_start, n_arrows, color, ls="-", step=None):
            step = seg if step is None else step
            x = x_start
            for k in range(n_arrows):
                y0, y1 = (f_top, f_bot) if k % 2 == 0 else (f_bot, f_top)
                ax.annotate("", xy=(x + step, y1), xytext=(x, y0),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, linestyle=ls))
                x += step + off
            return x

        a = assumption.lower()
        jk = "jerphagnon" in a or "no mr" in a
        hh = "herman" in a
        # AUTO-SIMPLIFY: never draw more passes than there is room for -- a thin layer or a
        # steep angle would otherwise stack overlapping arrows into an unreadable smear.
        room = max(1, int((X_MAX - x0) // max(seg + off, 0.35)))
        n_w = 1 if (jk or hh) else min(3, room)
        n_2w = 1 if jk else min(3, room)
        # Lateral offset of the 2w ladder. At OBLIQUE incidence the two ladders separate on their
        # own (they step at different refracted angles), so a small nudge suffices. Near NORMAL
        # incidence every pass is vertical and a small nudge INTERLEAVES 2w between the w passes --
        # so there the 2w ladder is placed clear of the whole w ladder instead.
        D2W = 0.38 if seg >= 0.35 else (n_w * (seg + off) + 0.30)
        x_end = _ladder(x0, n_w, "crimson")
        # label each ladder beside its OWN first pass, not on the normal (which sits at x0)

        # the 2w ladder rides the SAME incidence point, offset slightly so both stay visible; it
        # diverges naturally because it steps at its own refracted angle (dispersion).
        _ladder(x0 + D2W, n_2w, "navy", "--", step=seg2)

        # inhomogeneous (bound) waves in ORANGE alongside the omega passes -- FMR only, per
        # sub-assumption. Forward bound wave parallels pass 1 (down), the backward bound wave
        # parallels pass 2 (up) -- BOTH keep the ladder's +x in-plane run (in-plane momentum is
        # conserved; the pre- backward arrow ran up-LEFT, which no physical wave does).
        if not jk and not hh:
            inh_dx = 0.16
            ax.annotate("", xy=(x0 + inh_dx + 0.82 * seg, f_top - 0.82 * fh),
                        xytext=(x0 + inh_dx + 0.18 * seg, f_top - 0.18 * fh),
                        arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=1.8, alpha=0.9))
            sub = (fmr_submode or "").lower()
            if "backward" in sub or "standing" in sub or sub in ("forward_backward", "all"):
                p2x = x0 + seg + off
                ax.annotate("", xy=(p2x + inh_dx + 0.82 * seg, f_bot + 0.82 * fh),
                            xytext=(p2x + inh_dx + 0.18 * seg, f_bot + 0.18 * fh),
                            arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=1.8, alpha=0.9))

        # the transmitted rays continue from where the LAST DOWNWARD pass LANDS. D2:
        # this used the ladder loop's post-increment x (already advanced by seg + off for the next
        # pass), so the transmitted pair started to the right of the exit point and the join at the
        # film/substrate interface visibly broke wherever `off` is nonzero -- i.e. near normal
        # incidence, the default. Passes alternate down/up from k = 0, so the last DOWN pass is the
        # largest even k, and it lands at x_start + k*(seg + off) + seg.
        _last_down = (n_w - 1) if (n_w - 1) % 2 == 0 else max(n_w - 2, 0)
        ladder_end_x = x0 + _last_down * (seg + off) + seg
        ladder_end_y, ladder_layer = f_bot, fi + 1

    # transmitted omega -- de-emphasised (thin, pale): it is the pump leaving, not the SHG signal
    # the user measures, but drawing it completes the wave bookkeeping (have good label").
    tw_x, tw_y = _descend(ladder_end_x, ladder_end_y, ladder_layer, two=False,
                          color="crimson", lw=1.0, ls=(0, (5, 3)), alpha=0.45)
    # transmitted 2w -- the signal ray, full weight
    t2_x, t2_y = _descend(ladder_end_x + 0.16, ladder_end_y, ladder_layer, two=True,
                          color="navy", lw=1.6, ls="--")

    # labels: anchored to each ray's far end. "omega refl" sits ABOVE its arrowhead and "2omega refl"
    # BELOW its own (shorter) arrowhead -- a fixed VERTICAL stagger that holds at every angle. (The
    # previous horizontal-only offsets scaled with the ray geometry and collapsed into an unreadable
    # jumble near normal incidence and at compressed panel sizes -- , .)
    ax.text(x0 - gap - dx - 0.15, y_surf + dy + 0.06, r"$\omega$ in", color="crimson", fontsize=9,
            ha="right", va="bottom")
    ax.text(x0 + gap + dx + 0.12, y_surf + dy + 0.05, r"$\omega$ refl", color="crimson", fontsize=9,
            ha="left", va="bottom")
    ax.text(x0 + gap + ox + 0.92 * dx + 0.12, y_surf + oy + 0.92 * dy - 0.10, r"$2\omega$ refl",
            color="navy", fontsize=9, ha="left", va="top")
    # stagger the two transmitted labels vertically -- they landed on top of each other when
    # the pump and signal rays exit close together (near-equal n(w), n(2w)).
    _t2x = t2_x + 0.14
    ax.text(min(_t2x, X_MAX + 0.55), t2_y + 0.20, r"$2\omega$ trans", color="navy", fontsize=9,
            ha=("left" if _t2x < 8.2 else "right"), va="bottom")
    # D6: the clamp pinned the label's START, so a left-aligned label at X_MAX still ran past the
    # stack. Flip to right-aligned once it would not fit.
    _twx = tw_x + 0.14
    ax.text(min(_twx, X_MAX + 0.55), tw_y - 0.20, r"$\omega$ trans (pump)", color="crimson",
            fontsize=8, alpha=0.7, ha=("left" if _twx < 7.4 else "right"), va="top")
    # incidence-angle readout in the EMPTY right end of the ambient band. It used to sit just
    # left of the normal, which is exactly where the incident ray passes at oblique angles.
    ax.text(X_MAX, y_surf + 0.55 * heights[0], rf"$\theta_i={float(theta_deg):g}°$", fontsize=9,
            ha="right", va="center")
    if assumption is not None and finite:
        a = assumption.lower()
        jk = "jerphagnon" in a or "no mr" in a
        hh = "herman" in a
        short = ("Jerphagnon–Kurtz (no MR)" if jk else
                 "Herman–Hayden (MR for 2ω)" if hh else
                 "Full multiple reflections" + (f" — {fmr_submode}" if fmr_submode else ""))
        # the caption used to sit at x = 0.25 -- the SAME column as every layer name -- one
        # band below the film, so it landed on top of the next layer's label whenever that band was
        # thin (screenshot: it collided with "Au coating"). Put it in reserved space BELOW the
        # whole stack, where nothing else is drawn.
        ax.text(0.25, -total - 0.40, short, fontsize=8, color="0.25", style="italic", va="top")
        # D5: a colour key instead of glyphs pinned beside the rays. Three tags competing for the
        # few millimetres around the incidence point label each other's arrows -- at normal
        # incidence they collapse onto one column entirely.
        # fixed columns -- a width estimated from len(text) under-counted and ran the entries
        # together in the render
        for _kx, _txt, _c in ((0.25, "— ω fundamental", "crimson"),
                              (2.60, "-- 2ω SHG", "navy"),
                              (4.65, "— bound source", "darkorange")):
            ax.text(_kx, -total - 0.16, _txt, fontsize=7.5, color=_c, va="top", ha="left")
    title = "Optical setup schematic"
    if wavelength_um:
        title += rf"   ($\lambda$ = {wavelength_um:g} µm)"
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, 10)
    ax.set_ylim(-total - 0.72, 0.15)   # reserved strip for the colour key + caption
    ax.axis("off")
    fig.tight_layout()
    return fig


def build_fresnel_figure(fresnel_result):
    """Line matplotlib Figure of the multilayer Fresnel TRANSMITTANCE/REFLECTANCE (power) R_p, R_s,
    T_p, T_s vs incidence angle -- the original .ml GUI's 'Generate Fresnel Coefficients Plot' output
    (titled "Fresnel Transmittance/Reflectance", y in [0, 1]; R + T = 1 for a lossless interface).
    The backend (run_fresnel_sweep workflow='gui_multilayer') already returns POWER (|r|^2 and the
    transmittance), so the curves are plotted directly. Agg-testable."""

    import matplotlib.pyplot as plt

    num = fresnel_result.numeric
    theta = np.asarray(num["theta_deg"], dtype=float)
    fig = Figure(figsize=(7.0, 4.4), layout="constrained"); ax = fig.subplots()
    # colors match the original output (R_p blue, R_s orange, T_p green, T_s red)
    for key, color, label in (
        ("rp", "tab:blue", r"$R_p$"),
        ("rs", "tab:orange", r"$R_s$"),
        ("tp", "tab:green", r"$T_p$"),
        ("ts", "tab:red", r"$T_s$"),
    ):
        ax.plot(theta, np.asarray(num[key], dtype=float), color=color, lw=1.8, label=label)
    ax.set_xlabel(r"Incident Angle, $\theta_i$ (deg)")
    ax.set_ylabel("Reflectance / Transmittance (power)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Fresnel Transmittance/Reflectance", fontsize=11)
    ax.legend(fontsize=9, ncols=2)
    ax.grid(alpha=0.3)
    return fig


def build_maker_figure(maker_result, *, assumption_label: str | None = None):
    """Line matplotlib Figure of a SHAARP.ml Maker-fringe sweep (the iconic .ml output): transmitted
    SHG intensity vs incidence angle for the parallel and perpendicular analyzer channels, plotted
    RAW (the channels are natively continuous; see the body comment). ``assumption_label`` adds the
    original's "Assumption Used: ..." subtitle. Takes the SHAARPResult from run_maker_fringes /
    compute_ml_gui_result('Maker Fringes'). Agg-testable."""

    import matplotlib.pyplot as plt

    # The transmitted channels are now natively continuous (the eigenmode-ordering bug that put
    # spurious exact-0 sentinels in the per-channel split is fixed at the source -- the Maker sweep
    # selects the transmitted wave by dominant amplitude, matching live SHAARP.ml). So plot the RAW
    # solver output directly; no masking/interpolation (which would be papering over a bug).
    num = maker_result.numeric
    theta = np.asarray(num["theta_deg"], dtype=float)
    i_par = np.asarray(num["parallel_intensity"], dtype=float)
    i_perp = np.asarray(num["perpendicular_intensity"], dtype=float)
    # constrained_layout (NOT tight_layout): it re-flows on every draw, so when the Qt canvas resizes
    # the figure (e.g. to a short 838x247 viewport on a laptop) the title + x-axis label stay INSIDE
    # the canvas instead of being clipped. tight_layout is computed once at the build figsize and does
    # not adapt to the canvas resize -> it clipped the title/x-axis at short sizes (caught by the
    # layout-integrity harness).
    fig = Figure(figsize=(7.0, 4.2), layout="constrained"); ax = fig.subplots()
    ax.plot(theta, i_par, "-", color="navy", lw=1.8, label=r"$I_\parallel^{2\omega}(\theta_i)$")
    ax.plot(theta, i_perp, "-", color="crimson", lw=1.8, label=r"$I_\perp^{2\omega}(\theta_i)$")
    ax.set_xlabel(r"Incident Angle, $\theta_i$ (deg)")
    ax.set_ylabel(r"$I^{2\omega,T}(\theta_i,\varphi,\psi)$ (a.u.)")
    title = "Maker Fringes Figures"
    if assumption_label:
        title += f"\nAssumption Used: {assumption_label}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return fig


def _closed_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    if stop < start:
        raise ValueError("Grid stop must be >= start.")
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return start + step * np.arange(count, dtype=float)


def _result_summary(result) -> dict[str, Any]:
    return {
        "kind": result.kind,
        "validation_status": result.validation.status,
        "stage_keys": list(result.stages),
        "numeric_shapes": {k: list(np.asarray(v).shape) for k, v in result.numeric.items()},
    }


def make_shaarp_gui(*, system=None, material=None) -> Any:
    """Build the merged SHAARP.si + SHAARP.ml ipywidgets panel: a top-level Tab to NAVIGATE between
    the single-interface (.si) and multilayer (.ml) interfaces, each with its Functionality modes
    wired to the validated solver facades. Requires ipywidgets + IPython (notebook use)."""

    try:
        import ipywidgets as widgets
        from IPython.display import JSON, clear_output, display
    except ImportError as exc:
        raise ImportError("The merged SHAARP GUI requires ipywidgets and IPython.") from exc

    base_system = system or default_ml_system()
    base_material = material or default_interactive_material()

    def _interface_tab(which: str) -> Any:
        funcs = SI_FUNCTIONALITIES if which == "si" else ML_FUNCTIONALITIES
        functionality = widgets.Dropdown(options=funcs, description="Functionality")
        pg_default = (base_material.structure.point_group if which == "si"
                      else base_system.layers[1].material.structure.point_group)
        if pg_default not in SHG_POINT_GROUPS:
            pg_default = "1"
        point_group = widgets.Dropdown(options=SHG_POINT_GROUPS, value=pg_default, description="Point group")
        pg_info = widgets.HTML()

        def _update_pg_info(*_a):
            names = ", ".join(n for _r, _c, n in point_group_free_components(point_group.value)) or "(none)"
            pg_info.value = f"<span style='color:#555'><i>independent d components:</i> {names}</span>"

        _update_pg_info()
        point_group.observe(_update_pg_info, names="value")
        theta = widgets.FloatSlider(value=45.0 if which == "si" else 20.0, min=0, max=80, step=1, description="theta")
        th_min = widgets.FloatText(value=0.0, description="theta min")
        th_max = widgets.FloatText(value=45.0, description="theta max")
        th_step = widgets.FloatText(value=5.0, description="theta step")

        # Material entry (P2b): refractive indices + the point group's FREE d components (the
        # symmetry relations are applied automatically; fields rebuild when the point group
        # changes -- the faithful constrained SHG-tensor entry of the original GUIs).
        n_w = widgets.FloatText(value=2.0, description="n_w (o)")
        n_2w = widgets.FloatText(value=2.2, description="n_2w (o)")
        ne_w = widgets.FloatText(value=2.0, description="n_w (e)")
        ne_2w = widgets.FloatText(value=2.2, description="n_2w (e)")
        sub_n_w = widgets.FloatText(value=1.45, description="sub n_w")
        sub_n_2w = widgets.FloatText(value=1.46, description="sub n_2w")
        thickness = widgets.FloatText(value=1.0, description="h (um)")
        wavelength = widgets.FloatText(value=1.064, description="lambda (um)")
        # Crystal-orientation entry (P4b): z-cut identity, or the faithful SHAARP hklConvert
        # Miller mode (surface plane hkl -> lab L3, in-plane direct-lattice uvw -> lab L2),
        # lattice-aware via (a, b, c, alpha, beta, gamma).
        orient_mode = widgets.Dropdown(options=ORIENTATION_MODES, description="Orientation")
        lat_a = widgets.FloatText(value=1.0, description="a (A)", layout=widgets.Layout(width="140px"))
        lat_b = widgets.FloatText(value=1.0, description="b (A)", layout=widgets.Layout(width="140px"))
        lat_c = widgets.FloatText(value=1.0, description="c (A)", layout=widgets.Layout(width="140px"))
        lat_al = widgets.FloatText(value=90.0, description="alpha", layout=widgets.Layout(width="140px"))
        lat_be = widgets.FloatText(value=90.0, description="beta", layout=widgets.Layout(width="140px"))
        lat_ga = widgets.FloatText(value=90.0, description="gamma", layout=widgets.Layout(width="140px"))
        hkl_h = widgets.IntText(value=0, description="h", layout=widgets.Layout(width="120px"))
        hkl_k = widgets.IntText(value=0, description="k", layout=widgets.Layout(width="120px"))
        hkl_l = widgets.IntText(value=1, description="l", layout=widgets.Layout(width="120px"))
        uvw_u = widgets.IntText(value=1, description="u", layout=widgets.Layout(width="120px"))
        uvw_v = widgets.IntText(value=0, description="v", layout=widgets.Layout(width="120px"))
        uvw_w = widgets.IntText(value=0, description="w", layout=widgets.Layout(width="120px"))
        lat_row = widgets.HBox([lat_a, lat_b, lat_c, lat_al, lat_be, lat_ga])
        miller_row = widgets.HBox([hkl_h, hkl_k, hkl_l, uvw_u, uvw_v, uvw_w])

        def _update_orient_visibility(*_a):
            show = "" if orient_mode.value != "z-cut (identity)" else "none"
            lat_row.layout.display = show
            miller_row.layout.display = show

        _update_orient_visibility()
        orient_mode.observe(_update_orient_visibility, names="value")

        def _lattice():
            return (float(lat_a.value), float(lat_b.value), float(lat_c.value),
                    float(lat_al.value), float(lat_be.value), float(lat_ga.value))

        def _hkl():
            return (int(hkl_h.value), int(hkl_k.value), int(hkl_l.value))

        def _uvw():
            return (int(uvw_u.value), int(uvw_v.value), int(uvw_w.value))

        d_fields_box = widgets.HBox([])
        d_field_map: dict[tuple[int, int], Any] = {}

        def _rebuild_d_fields(*_a):
            d_field_map.clear()
            fields = []
            for (r, c, name) in point_group_free_components(point_group.value):
                w = widgets.FloatText(value=1.0, description=name, layout=widgets.Layout(width="140px"))
                d_field_map[(r, c)] = w
                fields.append(w)
            d_fields_box.children = tuple(fields)

        _rebuild_d_fields()
        point_group.observe(_rebuild_d_fields, names="value")

        def _d_free():
            return {pos: complex(w.value) for pos, w in d_field_map.items()}

        # .ml-only: the SHAARP.ml Assumptions panel + the case-study system presets.
        assumption = widgets.Dropdown(options=tuple(ML_ASSUMPTIONS), description="Assumption")
        system_preset = widgets.Dropdown(
            options=tuple(ML_SYSTEM_PRESETS) + ("Custom film (use fields)",), description="System"
        )
        run_button = widgets.Button(description="Run", icon="play", button_style="primary")
        out = widgets.Output()
        # copyable closed-form display (the originals' analytical-output copy analog): filled
        # after an analytical run; the user selects-all + copies from the Textarea.
        expr_area = widgets.Textarea(
            value="", description="Closed form",
            layout=widgets.Layout(width="95%", height="160px", display="none"),
        )
        state: dict[str, Any] = {"last_result": None}

        # Material-properties preset slots (the originals' Preset 1-4 panel) + data export (the
        # originals' Copy-button analog: writes the last result's numeric lists as JSON).
        preset_store = PresetStore()
        preset_slot = widgets.Dropdown(options=[1, 2, 3, 4], description="Preset")
        preset_label = widgets.Text(value="", description="Label", layout=widgets.Layout(width="200px"))
        save_btn = widgets.Button(description="Save", icon="save")
        recall_btn = widgets.Button(description="Recall")
        clear_btn = widgets.Button(description="Clear all")
        export_btn = widgets.Button(description="Export data", icon="download")

        def _snapshot() -> dict:
            return {
                "point_group": point_group.value, "n_w": float(n_w.value), "n_2w": float(n_2w.value),
                "ne_w": float(ne_w.value), "ne_2w": float(ne_2w.value),
                "sub_n_w": float(sub_n_w.value), "sub_n_2w": float(sub_n_2w.value),
                "thickness": float(thickness.value), "wavelength": float(wavelength.value),
                "d_free": {f"{r},{c}": complex(w.value).real for (r, c), w in d_field_map.items()},
                "orientation_mode": orient_mode.value,
                "lattice": list(_lattice()), "surface_hkl": list(_hkl()), "in_plane_uvw": list(_uvw()),
            }

        def _restore(snap: dict) -> None:
            point_group.value = snap["point_group"]  # triggers the d-field rebuild
            for name, w in (("n_w", n_w), ("n_2w", n_2w), ("ne_w", ne_w), ("ne_2w", ne_2w),
                            ("sub_n_w", sub_n_w), ("sub_n_2w", sub_n_2w),
                            ("thickness", thickness), ("wavelength", wavelength)):
                w.value = float(snap[name])
            for key, value in snap.get("d_free", {}).items():
                r, c = (int(x) for x in key.split(","))
                if (r, c) in d_field_map:
                    d_field_map[(r, c)].value = float(value)
            orient_mode.value = snap.get("orientation_mode", "z-cut (identity)")
            lat = snap.get("lattice", [1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
            for w, v in zip((lat_a, lat_b, lat_c, lat_al, lat_be, lat_ga), lat):
                w.value = float(v)
            for w, v in zip((hkl_h, hkl_k, hkl_l), snap.get("surface_hkl", [0, 0, 1])):
                w.value = int(v)
            for w, v in zip((uvw_u, uvw_v, uvw_w), snap.get("in_plane_uvw", [1, 0, 0])):
                w.value = int(v)

        def on_save(_):
            preset_store.save(int(preset_slot.value) - 1, _snapshot(), preset_label.value)
            save_btn.button_style = "success"

        def on_recall(_):
            snap = preset_store.recall(int(preset_slot.value) - 1)
            if snap is not None:
                _restore(snap)
                preset_label.value = preset_store.label(int(preset_slot.value) - 1)

        def on_clear(_):
            preset_store.clear()
            save_btn.button_style = ""

        def on_export(_):
            import json as _json

            with out:
                if state["last_result"] is None:
                    print("Nothing to export yet -- press Run first.")
                    return
                payload = export_result_payload(state["last_result"])
                fname = f"shaarp_gui_export_{which}_{payload['kind']}.json"
                with open(fname, "w", encoding="utf-8") as fh:
                    _json.dump(payload, fh, indent=1)
                print(f"exported -> {fname}")
                try:  # analytical runs also export their closed-form expressions as text
                    expr = analytical_expression_text(state["last_result"])
                except ValueError:
                    pass
                else:
                    ename = f"shaarp_gui_expr_{which}_{payload['kind']}.txt"
                    with open(ename, "w", encoding="utf-8") as fh:
                        fh.write(expr)
                    print(f"expressions -> {ename}")

        save_btn.on_click(on_save)
        recall_btn.on_click(on_recall)
        clear_btn.on_click(on_clear)
        export_btn.on_click(on_export)

        def on_run(_):
            import matplotlib.pyplot as plt

            # reset up front so a failed run never leaves a STALE closed form on display
            expr_area.value = ""
            expr_area.layout.display = "none"
            with out:
                clear_output(wait=True)
                try:
                    if which == "si":
                        custom = build_custom_si_material(
                            point_group.value, n_omega=float(n_w.value), n_2omega=float(n_2w.value),
                            n_omega_e=float(ne_w.value), n_2omega_e=float(ne_2w.value), d_free=_d_free(),
                            lattice=_lattice(), orientation_mode=orient_mode.value,
                            surface_hkl=_hkl(), in_plane_uvw=_uvw(),
                        )
                        si_layers = [("air", None), (f"crystal ({point_group.value})", None)]
                        sketch = build_schematic_figure(si_layers, theta_deg=float(theta.value))
                        display(sketch)
                        plt.close(sketch)
                        sketch3d = build_schematic_3d_figure(si_layers, theta_deg=float(theta.value))
                        display(sketch3d)
                        plt.close(sketch3d)
                        result = compute_si_gui_result(
                            functionality.value, point_group=point_group.value,
                            theta_deg=float(theta.value), material=custom,
                        )
                        # the iconic .si output: the reflected-SHG polar polarimetry of the entered material
                        fig = build_si_polarimetry_figure(
                            point_group.value, theta_deg=float(theta.value),
                            n_omega=float(n_w.value), n_2omega=float(n_2w.value),
                            n_omega_e=float(ne_w.value), n_2omega_e=float(ne_2w.value), d_free=_d_free(),
                        )
                        display(fig)
                        plt.close(fig)
                    else:
                        if system is not None:
                            sys_arg, preset_arg = system, None  # explicit caller system wins
                        elif system_preset.value == "Custom film (use fields)":
                            sys_arg, preset_arg = build_custom_ml_system(
                                point_group.value, film_n_omega=float(n_w.value),
                                film_n_2omega=float(n_2w.value), film_n_omega_e=float(ne_w.value),
                                film_n_2omega_e=float(ne_2w.value),
                                substrate_n_omega=float(sub_n_w.value), substrate_n_2omega=float(sub_n_2w.value),
                                thickness_um=float(thickness.value), wavelength_um=float(wavelength.value),
                                d_free=_d_free(), lattice=_lattice(), orientation_mode=orient_mode.value,
                                surface_hkl=_hkl(), in_plane_uvw=_uvw(),
                            ), None
                        else:
                            sys_arg, preset_arg = None, system_preset.value
                        sketch_sys = sys_arg if sys_arg is not None else resolve_ml_system_preset(preset_arg)
                        ml_layers = [(L.name, L.thickness_um) for L in sketch_sys.layers]
                        sketch = build_schematic_figure(
                            ml_layers, theta_deg=float(theta.value),
                            wavelength_um=float(sketch_sys.wavelength_um),
                        )
                        display(sketch)
                        plt.close(sketch)
                        sketch3d = build_schematic_3d_figure(ml_layers, theta_deg=float(theta.value))
                        display(sketch3d)
                        plt.close(sketch3d)
                        result = compute_ml_gui_result(
                            functionality.value, point_group=point_group.value,
                            theta_deg=float(theta.value), theta_min_deg=float(th_min.value),
                            theta_max_deg=float(th_max.value), theta_step_deg=float(th_step.value),
                            assumption=assumption.value, system_preset=preset_arg, system=sys_arg,
                        )
                        if functionality.value == "Maker Fringes":  # the iconic .ml output
                            fig = build_maker_figure(result)
                            display(fig)
                            plt.close(fig)
                        elif functionality.value == "Fresnel Coefficients":
                            fig = build_fresnel_figure(result)
                            display(fig)
                            plt.close(fig)
                    state["last_result"] = result
                    try:
                        expr_area.value = analytical_expression_text(result)
                        expr_area.layout.display = ""
                    except ValueError:
                        expr_area.value = ""
                        expr_area.layout.display = "none"
                    display(JSON(_result_summary(result), expanded=False))
                except Exception as exc:  # surface errors in-panel, like the original GUI
                    print(f"{type(exc).__name__}: {exc}")

        run_button.on_click(on_run)
        # control rows: material entry (both tabs); .ml adds angle range + assumptions/presets + film fields.
        mat_row = widgets.HBox([n_w, n_2w, ne_w, ne_2w])
        ml_film = widgets.HBox([sub_n_w, sub_n_2w, thickness, wavelength]) if which == "ml" else widgets.HBox([])
        ml_range = widgets.HBox([th_min, th_max, th_step]) if which == "ml" else widgets.HBox([])
        ml_opts = widgets.HBox([assumption, system_preset]) if which == "ml" else widgets.HBox([])
        preset_row = widgets.HBox([preset_slot, preset_label, save_btn, recall_btn, clear_btn, export_btn])
        return widgets.VBox([
            widgets.HBox([functionality, run_button]),
            widgets.HBox([point_group, theta]),
            pg_info,
            mat_row,
            d_fields_box,
            widgets.HBox([orient_mode]),
            lat_row,
            miller_row,
            ml_film,
            ml_opts,
            ml_range,
            preset_row,
            expr_area,
            out,
        ])

    tabs = widgets.Tab(children=[_interface_tab("si"), _interface_tab("ml")])
    tabs.set_title(0, "SHAARP.si (single interface)")
    tabs.set_title(1, "SHAARP.ml (multilayer)")
    return tabs
