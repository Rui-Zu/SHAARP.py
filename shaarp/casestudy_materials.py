"""Case Study materials — the original SHAARP.ml setup.nb material set, with wavelength dispersion.

The optical data lives in the package-data file ``casestudy_dispersion.json``, produced by a live
Mathematica evaluation of each original setup.nb material function across a wavelength grid
(see benchmarks/generate_casestudy_registry.py). Every dielectric value is the original Mathematica
output (crystal-physics frame); the SHG d tensor is the original's (wavelength-independent) value;
the orientation is the original's computed Z-axes-in-lab matrix (materials whose orientation was
authored in the SHAARP.si downward-propagation frame get a documented lab-azimuth correction at
build time — see ``_SI_FRAME_AZIMUTH_DEG``).

``build_casestudy_material(name, wavelength_um=...)`` interpolates the dielectric tensors at the
requested wavelength, reproducing the original GUI behaviour that the case-study dielectric tensors
update with the fundamental wavelength. With ``wavelength_um=None`` the material's native/standard
wavelength is used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry

_SUBSCRIPT = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_SUPERSCRIPT = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def _clean_label(s: str) -> str:
    """Convert a raw Mathematica box-language label to clean Unicode for display.

    The setup.nb material names were exported with Mathematica box markup -- e.g. "LiNbO3" (3 as a
    subscript) is stored as ``SubscriptBox[LiNbO, 3]``,
    where the - private-use codepoints are Mathematica's \\(\\), \\*, \\" box delimiters.
    Left raw, those private-use chars render as blank 'tofu' boxes in matplotlib/Qt (and read as
    garbage in the output panel). This strips the box delimiters and folds
    SubscriptBox[base, sub] / SuperscriptBox[base, sup] into real Unicode sub/superscripts, so the
    label reads "LiNbO3" with a true subscript 3.
    """
    if not s:
        return s
    s = "".join(c for c in s if not (0xe000 <= ord(c) <= 0xf8ff))  # drop MM private-use box delimiters
    s = re.sub(r"SubscriptBox\[\s*([^\[\],]+?)\s*,\s*([^\[\]]+?)\s*\]",
               lambda m: m.group(1) + m.group(2).translate(_SUBSCRIPT), s)
    s = re.sub(r"SuperscriptBox\[\s*([^\[\],]+?)\s*,\s*([^\[\]]+?)\s*\]",
               lambda m: m.group(1) + m.group(2).translate(_SUPERSCRIPT), s)
    return " ".join(s.split())  # tidy any leftover whitespace runs


@lru_cache(maxsize=1)
def _data() -> dict:
    # importlib.resources keeps this working both from source and inside the frozen exe.
    try:
        from importlib.resources import files

        text = (files("shaarp") / "casestudy_dispersion.json").read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        from pathlib import Path

        text = (Path(__file__).with_name("casestudy_dispersion.json")).read_text(encoding="utf-8")
    return json.loads(text)


CASE_STUDY_ORDER = list(_data()["order"])

# ---------------------------------------------------------------------------------------------
# GUI curation (case-study fidelity audit): the combos present EXACTLY the examples
# the ORIGINAL packages present, with the provenance wavelength in the display label. Display
# labels map to registry keys here so the numeric fences keep their unchanged anchors.
# 'LiNbO3 z-cut (1064 nm)' is setup.nb DEAD CODE (defined but never exposed in the original
# ♯SHAARP.ml palette) — it stays registry-only for the numerical fences and is hidden from the GUI.
# ---------------------------------------------------------------------------------------------

# The original ♯SHAARP.ml palette: exactly its 16 buttons, in the ORIGINAL palette order
# (SHAARP.ml.nb "Case Studies" panel). Wavelengths per the papers/preset provenance:
# Fig 5/6 family = 1550 nm; Fig 3 X-cut quartz = 1064 nm; Fig 4/8 family = 800 nm.
# Materials with MULTIPLE cut/wavelength variants are grouped under a master material header
# (a disabled row) with the variants indented beneath it (author's direction, —
# "a material master title followed by each wavelength down below"); single-variant materials
# stay as one flat row. Child rows keep a compact material tag so every combo text stays
# globally unique (session round-trip + sweep select by text).
GUI_ML_GROUPS: list[tuple[str | None, list[tuple[str, str]]]] = [
    # (master header or None, [(display label, registry key), ...])
    (None, [("Blank linear", "Blank linear")]),
    (None, [("Blank nonlinear", "Blank nonlinear")]),
    (None, [("BaTiO3 (800 nm)", "BaTiO3")]),
    (None, [("Air", "Air")]),
    ("LiNbO3", [
        ("    LiNbO3 x-cut · 1550 nm", "LiNbO3 x-cut (1550 nm)"),
        ("    LiNbO3 z-cut · 1550 nm", "LiNbO3 z-cut (1550 nm)"),
    ]),
    ("KTP", [
        ("    KTP x-cut · 1550 nm", "KTP x-cut"),
        ("    KTP y-cut · 1550 nm", "KTP y-cut"),
    ]),
    (None, [("ZnO (001) (1550 nm)", "ZnO (001)")]),
    (None, [("Pt (111) (1550 nm)", "Pt (111) (1550 nm)")]),
    (None, [("Al2O3 (0001) (1550 nm)", "Al2O3 (0001) (1550 nm)")]),
    (None, [("GaAs (111) (800 nm)", "GaAs (111) (800 nm)")]),
    ("Quartz", [
        ("    Quartz x-cut · 1064 nm", "Quartz x-cut (1064 nm)"),
        ("    Quartz z-cut · 800 nm", "Quartz z-cut (800 nm)"),
    ]),
    (None, [("Au coating (800 nm)", "Au coating (800 nm)")]),
    (None, [("MoS2 (800 nm)", "MoS2")]),
]

# Flat (label, key) view — for contexts that need self-contained rows (the per-layer material
# picker) the child labels are used WITHOUT the hierarchy indent.
GUI_ML_CASES: list[tuple[str, str]] = [
    (label.strip(), key) for _hdr, entries in GUI_ML_GROUPS for (label, key) in entries]

# registry key -> stripped GUI display label (for upgrading stored specs/sessions that carry keys)
CASE_LABEL_BY_KEY: dict[str, str] = {key: label for label, key in GUI_ML_CASES}

# The original ♯SHAARP.si palette groups (V1.03 ≡ V1.04). The .si tool is single-wavelength by
# construction (no λ input — λ enters only through the indices); the DOI cases are the paper's
# 800 nm measurements, labeled on the group header per the author's direction.
GUI_SI_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("—  Cases in DOI (♯SHAARP.si — all 800 nm)  —", [
        ("GaAs (111)", "GaAs (111) (800 nm)"),
        ("LiNbO3 (11-20) MTI X-cut", "LiNbO3 (11-20) MTI X-cut"),
        ("KTP (100)", "KTP (100)"),
        ("TaAs (112)", "TaAs (112)"),
    ]),
    ("—  Complex SHG Coefficients  —", [
        ("GaAs (111) @1064 nm", "GaAs (111) (1064 nm)"),
    ]),
    ("—  Deep UV NLO  —", [
        ("LiB3O5 (LBO)", "LiB3O5 (LBO)"),
        ("KBBF", "KBBF"),
    ]),
    ("—  Polar Metals  —", [
        ("LiOsO3", "LiOsO3"),
    ]),
]


def gui_dispersive_group() -> tuple[str, list[tuple[str, str]]]:
    """The dispersive variants, as a (header, [(label, label)]) group for the material combos.

    Their labels ARE their keys -- they are resolved by shaarp.dispersion, not by this registry --
    and each carries its wavelength range, because that is what decides whether the material can
    answer the sweep being asked of it."""
    from .dispersion import dispersive_material_names

    names = dispersive_material_names()
    return ("—  Dispersive (published index data)  —",
            [(n, n) for n in names])


_CASE_LABEL_TO_KEY: dict[str, str] = {label: key for label, key in GUI_ML_CASES}
for _hdr, _entries in GUI_ML_GROUPS:
    _CASE_LABEL_TO_KEY.update({label: key for label, key in _entries})  # indented child forms
for _hdr, _entries in GUI_SI_GROUPS:
    _CASE_LABEL_TO_KEY.update({label: key for label, key in _entries})


def resolve_case_label(label: str) -> str:
    """Registry key for a GUI display label (labels may carry the provenance wavelength and the
    hierarchy indent). Unmapped strings pass through unchanged, so raw registry keys — e.g. from
    an older saved session — keep resolving."""
    text = str(label)
    return _CASE_LABEL_TO_KEY.get(text) or _CASE_LABEL_TO_KEY.get(text.strip(), text)


def _interp_tensor(grid, re_blocks, im_blocks, lam):
    """Linear-interpolate a 3x3 complex tensor (component-wise, re/im separately) at ``lam``.
    numpy.interp clamps to the grid ends outside range (a documented, safe GUI-level behaviour)."""

    grid = np.asarray(grid, dtype=float)
    re = np.asarray(re_blocks, dtype=float)  # [nλ, 3, 3]
    im = np.asarray(im_blocks, dtype=float)
    out = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            out[i, j] = complex(np.interp(lam, grid, re[:, i, j]),
                                np.interp(lam, grid, im[:, i, j]))
    return out


def casestudy_native_wavelength(display_name: str) -> float:
    return float(_data()["materials"][display_name]["native_lambda_um"])


def casestudy_lambda_range(display_name: str) -> tuple[float, float] | None:
    """(min, max) of the material's exported dispersion grid in um, or None if unknown.

    Outside this range the interpolation CLAMPS to the grid-end tensors (np.interp behaviour,
    documented in :func:`_interp_tensor`). The GUI uses this to WARN instead of clamping
    silently (residual risk R2, closed)."""

    m = _data()["materials"].get(display_name)
    if not m:
        return None
    g = [float(x) for x in (m.get("grid_um") or [])]
    return (min(g), max(g)) if g else None


# A tabulated grid says where the data EXISTS, not where it is still physical. The exported
# Sellmeier models run into their own ultraviolet pole at the HALF wavelength well inside the
# 0.40-2.00 um grid: at lambda = 0.44 um KTP's eps(2w) reaches 7.0e3 and LiNbO3 (1550 nm) goes
# to -0.33 with a zero imaginary part. A transparent medium cannot have a lossless negative
# permittivity, and a principal value far above the material's own eps(w) is a pole rather than
# dispersion -- so both symptoms are screened.
#
# Only eps(2w) is screened, never eps(w). In an ABSORBING medium eps(w) legitimately dwarfs
# eps(2w) -- MoS2 sits at eps(2w) = 0.25 with a far larger eps(w) -- so the same ratio rule
# applied to eps(w) rejects physically fine data. That costs nothing on today's registry, where
# every absorbing material is constant-by-source and returns before reaching this screen; it
# starts to matter the moment a dispersive absorbing material is added. Fenced directly on the
# predicate in tests/test_casestudy_dispersion_quality.py rather than through the registry.
_POLE_EPS_RATIO = 3.0


def _point_is_physical(eps_w: np.ndarray, eps_2w: np.ndarray) -> bool:
    diag_w = [eps_w[i, i] for i in range(3)]
    diag_2w = [eps_2w[i, i] for i in range(3)]
    if any(v.real < 0 and abs(v.imag) <= 1e-12 for v in diag_2w):
        return False
    reference = max(abs(v.real) for v in diag_w) or 1.0
    return max(abs(v.real) for v in diag_2w) <= _POLE_EPS_RATIO * reference


@dataclass(frozen=True)
class SpectralSupport:
    """What a case-study material's shipped optical data can support across a wavelength RANGE.

    ``kind`` is one of:

    ``"tabulated"``    a multi-point grid whose tensors genuinely vary with wavelength, so a
                       spectrum computed from it is meaningful.
    ``"constant"``     a multi-point grid whose tensors are identical at every point -- the
                       original setup.nb defined the material with a constant index -- so a
                       spectrum computed from it is FLAT.
    ``"single_point"`` one tabulated wavelength (the .si palette cases, whose tensors are fixed
                       values in the original notebook), so the interpolation returns that one
                       tensor at every wavelength and a spectrum is likewise flat.

    ``tabulated_um`` is the grid span, i.e. what :func:`casestudy_lambda_range` reports.
    ``usable_um`` is the contiguous sub-interval of that span, anchored at the material's native
    wavelength, where eps(2w) is still physical (see :func:`_point_is_physical`). The two differ
    only for materials whose exported model runs into its ultraviolet pole inside its own grid.
    """

    name: str
    kind: str
    tabulated_um: tuple[float, float] | None
    usable_um: tuple[float, float] | None
    reason: str

    @property
    def varies(self) -> bool:
        """True when a wavelength sweep over this material produces a non-flat spectrum."""
        return self.kind == "tabulated"


@lru_cache(maxsize=None)
def casestudy_spectral_support(display_name: str) -> SpectralSupport:
    """Describe what a wavelength RANGE over ``display_name`` can and cannot deliver.

    Derived from the shipped data on every call rather than hand-maintained per material, so it
    stays true if the registry is ever re-exported."""

    m = _data()["materials"].get(display_name)
    if not m:
        return SpectralSupport(display_name, "single_point", None, None,
                               f"{display_name!r} is not a Case Study material")
    grid = np.asarray(m["grid_um"], dtype=float)
    span = (float(grid.min()), float(grid.max()))
    eps_w = np.asarray(m["epsW_re"], dtype=float) + 1j * np.asarray(m["epsW_im"], dtype=float)
    eps_2w = np.asarray(m["eps2W_re"], dtype=float) + 1j * np.asarray(m["eps2W_im"], dtype=float)

    if grid.size == 1:
        return SpectralSupport(
            display_name, "single_point", span, span,
            # WHAT the data is, not what a sweep does with it: that depends on the path (flat on a
            # single interface, a thickness sweep on a stack) and the spectral warning says both.
            f"{display_name} carries one tabulated wavelength ({grid[0]:g} um); its tensors are "
            "fixed values in the original notebook")

    varies = bool(np.ptp(eps_w.real, axis=0).max() > 1e-12
                  or np.ptp(eps_w.imag, axis=0).max() > 1e-12
                  or np.ptp(eps_2w.real, axis=0).max() > 1e-12
                  or np.ptp(eps_2w.imag, axis=0).max() > 1e-12)
    if not varies:
        return SpectralSupport(
            display_name, "constant", span, span,
            f"{display_name} carries a grid across {span[0]:g}-{span[1]:g} um but its tensors are "
            "the same at every point (the original defines it with a constant index)")

    # the usable window is the CONTIGUOUS run of physical points containing the native
    # wavelength -- not min/max over all good points, which would span straight across an
    # interior pole and re-admit exactly the values being screened out.
    good = [_point_is_physical(eps_w[i], eps_2w[i]) for i in range(grid.size)]
    anchor = int(np.argmin(np.abs(grid - float(m["native_lambda_um"]))))
    if not good[anchor]:
        return SpectralSupport(display_name, "tabulated", span, None,
                               f"{display_name} has no physical eps(2w) at its own native "
                               "wavelength; its exported dispersion needs regenerating")
    lo = hi = anchor
    while lo > 0 and good[lo - 1]:
        lo -= 1
    while hi < grid.size - 1 and good[hi + 1]:
        hi += 1
    usable = (float(grid[lo]), float(grid[hi]))
    if usable == span:
        return SpectralSupport(display_name, "tabulated", span, usable,
                               f"{display_name} is dispersive across {span[0]:g}-{span[1]:g} um")
    return SpectralSupport(
        display_name, "tabulated", span, usable,
        f"{display_name} is tabulated across {span[0]:g}-{span[1]:g} um but its eps(2w) is only "
        f"physical over {usable[0]:g}-{usable[1]:g} um; outside that its index model runs into "
        "its ultraviolet pole at half the wavelength")


def casestudy_usable_lambda_range(display_name: str) -> tuple[float, float] | None:
    """(min, max) um over which this material's eps(2w) is still physical, or None.

    Narrower than :func:`casestudy_lambda_range` for materials whose exported Sellmeier model
    poles inside its own grid. This -- not the grid span -- is the range a wavelength sweep may
    use."""

    return casestudy_spectral_support(display_name).usable_um


# SI-frame azimuth corrections (earlier comments here said ",
# ", which is the wrong session number for this work) — applied on top of the verbatim registry
# orientation when the material is built. ROOT CAUSE (proven by live V1.04 replication):
# SHAARP.si V1.04 propagates its transmitted waves DOWNWARD (mode direction
# u = (sin θ, 0, −cos θ), extracted from SHAARP_V1.04.nb), while this port — like SHAARP.ml
# setup.nb's `solveSnell` (u = (sin θ, 0, +cos θ)), the oracle the port was validated against —
# propagates UPWARD. Consuming an si-frame Z-axes-in-lab matrix in the upward frame z-mirrors
# the geometry. That is observable ONLY when the anisotropy axis is tilted OBLIQUELY in the
# plane of incidence (every other registry case is principal-aligned or in-plane → invariant);
# for intensity observables the exact compensation is a 180° lab azimuth
# (rows → rows·diag(−1,−1,1); the leftover global d sign is unobservable in |E|²).
# TaAs (112): live V1.04 run (its own hklConvert (112)/[1,-1,0] state + downward convention)
# gives n_eo/k_eo = 4.3006/2.1684 (θi=0.5°) → 4.3512/2.0541 (80°) = the published si-2022
# Fig 7(b) rising-n/falling-k trend, and equals this port's values WITH the flip to machine
# precision; without it the port produces the mirrored (falling-n/rising-k) curve. The polar
# channels are m_y-symmetric and azimuth-insensitive (Ip/Is@45° = 74.6 fence unaffected).
# Fence: tests/test_paper_cases.py::TestSiFig7TaAsEffectiveIndex.
_SI_FRAME_AZIMUTH_DEG: dict[str, float] = {"TaAs (112)": 180.0}


def casestudy_miller_label(material_name: str) -> str:
    """Miller-index surface label for a case-study material name (e.g. ``"(001)"``), or ``""``.

    The original .ml Set-Material figure annotates each layer with its surface (hkl); the exported
    registry keeps that under ``miller``. Keyed by the material's DISPLAY name (which is also what
    ``Material.name`` carries for case-study materials), so GUI layer labels can look it up without
    widening the frozen ``Material`` dataclass."""
    m = _data()["materials"].get(material_name)
    if not m:
        return ""
    mil = m.get("miller")
    if not mil:
        return ""
    try:
        h, k, l = (int(round(float(x))) for x in mil)
        return f"({h}{k}{l})"
    except Exception:
        return ""


def build_casestudy_material(display_name: str, *, wavelength_um: float | None = None) -> Material:
    """Construct a Material from the Mathematica-exported Case Study dispersion data.

    The dielectric tensors are interpolated at ``wavelength_um`` (default: the material's native
    wavelength); eps and d are in the CRYSTAL-PHYSICS frame and the stored orientation (the
    original's Z-axes-in-lab matrix) rotates them to the lab, matching Material semantics. Returns
    a FRESH object each call."""

    mats = _data()["materials"]
    if display_name not in mats:
        raise ValueError(f"unknown Case Study material {display_name!r}; choose from {CASE_STUDY_ORDER}")
    m = mats[display_name]
    lam = float(m["native_lambda_um"]) if wavelength_um is None else float(wavelength_um)
    a, b, c, al, be, ga = m["lattice"]
    structure = CrystalStructure(point_group=m["point_group"], a=a, b=b, c=c,
                                 alpha_deg=al, beta_deg=be, gamma_deg=ga)
    orientation = CrystalOrientation(np.array(m["orientation"], dtype=float))
    si_azimuth = _SI_FRAME_AZIMUTH_DEG.get(display_name)
    if si_azimuth:
        orientation = orientation.with_lab_azimuth_deg(si_azimuth)
    eps_w = _interp_tensor(m["grid_um"], m["epsW_re"], m["epsW_im"], lam)
    eps_2w = _interp_tensor(m["grid_um"], m["eps2W_re"], m["eps2W_im"], lam)
    d = np.array(m["d_re"], dtype=complex) + 1j * np.array(m["d_im"], dtype=complex)
    return Material(name=_clean_label(m["name"]), structure=structure, orientation=orientation,
                    epsilon_omega=eps_w, epsilon_2omega=eps_2w, d_voigt_pm_v=d)


def build_casestudy_ml_system(display_name: str, *, thickness_um: float = 1.0,
                              wavelength_um: float | None = None,
                              ambient_n_omega: float = 1.0, ambient_n_2omega: float = 1.0,
                              substrate_n_omega: float = 1.45, substrate_n_2omega: float = 1.46) -> MultilayerSystem:
    """Build an ambient / case-study-film / isotropic-substrate multilayer with the chosen
    original material as the active SHG film, its dielectric tensors taken at ``wavelength_um``.
    Both half-spaces are isotropic media: the ambient defaults to air exactly."""

    from . import presets
    # ONE resolution seam for the film, so a dispersive variant reaches this builder the same way
    # it reaches the layer editor. Resolving here with build_casestudy_material instead made every
    # dispersive ML Update fail with "unknown Case Study material", because the registry is only
    # half of what the material combos offer. Imported lazily: layer_stack imports this module.
    from .layer_stack import material_for_label

    if wavelength_um is None:
        lam = casestudy_native_wavelength(display_name)
    else:
        lam = float(wavelength_um)
    film = material_for_label(display_name, lam)

    def _iso(name, nw, n2w):
        return Material(
            name=name, structure=CrystalStructure(point_group="∞∞m"), orientation=CrystalOrientation(),
            epsilon_omega=(np.eye(3) * nw**2).astype(complex),
            epsilon_2omega=(np.eye(3) * n2w**2).astype(complex),
            d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
        )

    substrate = _iso("substrate", substrate_n_omega, substrate_n_2omega)
    ambient = (presets.air() if ambient_n_omega == 1.0 and ambient_n_2omega == 1.0
               else _iso("ambient", ambient_n_omega, ambient_n_2omega))
    return MultilayerSystem(
        wavelength_um=lam,
        polarimetry=Polarimetry(theta_deg=0.0, phi_deg=0.0, psi_deg=0.0),
        layers=[
            Layer("air in", ambient, shg_active=False),
            Layer(film.name, film, thickness_um=float(thickness_um), shg_active=True),
            Layer("substrate", substrate, shg_active=False),
        ],
    )
