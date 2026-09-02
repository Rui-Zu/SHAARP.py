"""Arbitrary N-layer stack model for the multilayer (.ml) GUI -- the original's Layer Selection.

The original SHAARP.ml GUI lets the user set the Number of Layers and edit each layer's material
and thickness. Whether a layer is SHG-active is NOT a switch: the original offers two point-group
popups per layer, "Noncentrosymmetric ->" and "Centrosymmetric ->" (SHAARP.ml.nb:5191 / :5630),
and activity is the list the group came from (F63, -- the port's former "SHG active"
checkbox was a port invention). This pure, headless-testable model holds that per-layer state and
assembles a validated :class:`MultilayerSystem`. The widget layer (desktop_app) is a thin editor
on top of this model.

Each layer spec is a dict: ``{"material": <case-study display name or 'air'>, "thickness_um": float,
"analytic_h": bool, "analytic_d": bool}`` (+ ``"custom"`` / ``"iso_n"`` / ``"name"`` /
``"analytic_d_known"`` where relevant). SHG activity is DERIVED at build time by
:func:`spec_shg_active` from the layer's point group; a legacy ``"shg_active"`` key in an old
session is ignored. The first and last layers are the ambient/substrate half-spaces (no
thickness); interior layers carry a thickness. Materials come from the Mathematica-exported
Case Study registry (so the optical constants are the original's), taken at the system wavelength.
"""

from __future__ import annotations

from .casestudy_materials import GUI_ML_CASES, build_casestudy_material, resolve_case_label
from .config import Layer, MultilayerSystem, Polarimetry

# the layer material choices: the ambient "air", a plain ISOTROPIC-n entry (the half-space media —
# "remove the substrate group and blend both top and substrate to the layer
# editor ... as isotropic layers" — its n(w)/n(2w) live in spec["iso_n"] and are edited via the
# Layer group's n spins), every ORIGINAL-palette Case Study material (display labels carrying the
# provenance wavelength; case-study fidelity audit), and a per-layer CUSTOM option (the
# original "Set Material Properties" per-layer crystal entry -- a layer whose full point group /
# lattice / orientation / refractive indices / d tensor come from the GUI entry panels, captured
# per layer in spec["custom"]).
CUSTOM_LAYER_CHOICE = "Custom (fields)"
ISOTROPIC_LAYER_CHOICE = "isotropic n (set below)"
LAYER_MATERIAL_CHOICES = ["air", ISOTROPIC_LAYER_CHOICE,
                          *(label for label, _key in GUI_ML_CASES), CUSTOM_LAYER_CHOICE]


def _material_from_custom_spec(cspec: dict, wavelength_um: float):
    """Build a per-layer custom Material from a snapshot of the GUI crystal-entry panels (point group,
    indices OR full complex eps, lattice, orientation, symmetry-constrained d OR full d)."""
    from .shaarp_gui import build_custom_si_material  # lazy import (avoids any import cycle)

    d_free = {tuple(int(x) for x in k.split(",")): complex(v)
              for k, v in (cspec.get("d_free") or {}).items()}
    return build_custom_si_material(
        cspec.get("point_group", "1"),
        n_omega=float(cspec.get("n_w", 2.0)), n_2omega=float(cspec.get("n_2w", 2.2)),
        n_omega_e=cspec.get("ne_w"), n_2omega_e=cspec.get("ne_2w"),
        d_free=d_free,
        eps_omega_full=cspec.get("eps_omega_full"), eps_2omega_full=cspec.get("eps_2omega_full"),
        d_full=cspec.get("d_full"),
        lattice=tuple(cspec.get("lattice", (1.0, 1.0, 1.0, 90.0, 90.0, 90.0))),
        orientation_mode=cspec.get("orientation_mode", "z-cut (identity)"),
        surface_hkl=tuple(cspec.get("surface_hkl", (0, 0, 1))),
        in_plane_uvw=tuple(cspec.get("in_plane_uvw", (1, 0, 0))),
        z_axes=cspec.get("z_axes"),
        name="custom layer",
    )


def default_layer_spec(material: str = "air", thickness_um: float = 1.0, shg_active=None,
                       analytic_h: bool = False, analytic_d: bool = False) -> dict:
    """One layer spec. ``analytic_h`` / ``analytic_d`` are the per-layer symbolic flags --
    the original's per-layer "analytical h" / "analytical dij" buttons
    (``SHAARP.ml.nb:5135-5146``, ``:7256-7292``), which default to OFF (a layer is numeric
    until you press its button).

    ``shg_active`` is accepted for source compatibility but IGNORED: activity is decided by
    the layer's point group at build time (func:`spec_shg_active`), never stored on the spec."""
    return {"material": material, "thickness_um": float(thickness_um),
            "analytic_h": bool(analytic_h), "analytic_d": bool(analytic_d)}


def default_stack() -> list[dict]:
    """A 3-layer air / LiNbO3-film / air starting stack (a physical Maker sample)."""
    return [
        default_layer_spec("air", 0.0, False),
        # analytic_h=True on the film: the pre-F60 GUI had ONE stack-wide "analytical h"
        # checkbox that shipped ON, so this preserves the default output (symbolic in h)
        # while the flag is now per-layer.
        default_layer_spec("LiNbO3 z-cut · 1550 nm", 10.0, True, analytic_h=True,
                           analytic_d=True),
        default_layer_spec("air", 0.0, False),
    ]


def material_for_label(label: str, wavelength_um: float):
    """ONE resolution seam for every material label the GUI can show.

    ``"air"`` -> the air preset; a built-in palette label/key -> the registry material through the
    IDENTICAL ``build_casestudy_material(key, wavelength_um=...)`` call as (the validated
    numerics are untouched); a name from the user's "My Materials" store -> that saved material;
    anything else -> ``ValueError``."""
    from . import presets
    from .casestudy_materials import CASE_STUDY_ORDER

    name = str(label)
    if name.strip() == "air":
        return presets.air()
    key = resolve_case_label(name)
    if key in CASE_STUDY_ORDER:
        return build_casestudy_material(key, wavelength_um=wavelength_um)
    from .user_materials import build_user_material, is_user_material  # lazy: no import cycle

    if is_user_material(name):
        return build_user_material(name.strip(), wavelength_um)
    raise ValueError(f"unknown material {label!r}")


def _layer_material(name: str, wavelength_um: float):
    return material_for_label(name, wavelength_um)


def layer_material_choices() -> list[str]:
    """The ML layer-material combo rows: the pinned palette list plus, when the user's store is
    non-empty, a disabled section header and the user's materials, just before the Custom entry."""
    from .user_materials import USER_SECTION_HEADER, list_names

    names = list_names()
    if not names:
        return list(LAYER_MATERIAL_CHOICES)
    base = list(LAYER_MATERIAL_CHOICES)
    base.remove(CUSTOM_LAYER_CHOICE)
    return base + [USER_SECTION_HEADER, *names, CUSTOM_LAYER_CHOICE]


def layer_number(stack_index: int) -> int:
    """The USER-VISIBLE layer number of a stack position.

    ONE numbering, everywhere: the layer editor, the schematic, the provenance line and every
    SYMBOL the closed form carries. F60 made every medium a numbered layer (row = index + 1);
    before F61 the closed form used an INTERIOR-only counter, so the first film -- row 2 -- was
    described as ``h1`` while the panel called it layer 2 (the panel is editing
    layer 2 but you used h1"). A symbol's number must be the number on screen."""
    return int(stack_index) + 1


def interior_layer_number(interior_position: int) -> int:
    """User-visible number of the ``interior_position``-th interior layer (0-based).

    Interiors start at stack index 1, so interior 0 is row 2."""
    return layer_number(int(interior_position) + 1)


def layer_role_label(index: int, total: int, name: str | None = None) -> str:
    """The row label for medium ``index`` of a ``total``-medium stack.

    EVERY medium is a numbered layer, 1..total (for air/quartz/Au/air
    "I would consider call it 4 layers"). This DEVIATES from the released .ml deliberately:
    that GUI numbers only the interior films (`SHAARP.ml.nb:3707-3712`, `matindex` over
    `Range[2, materialnumber+1]`) BECAUSE it hardcodes air at both ends (`:666-716`,
    `:425-439`) — the half-spaces are not editable there, so they are not entries. Since
    F58 made both half-spaces user-settable media, they are first-class rows here and
    are counted and numbered like any other layer; they carry the original's own
    description of them, "semi-infinite" (`SHAARP.ml.nb:5097-5104`: "thickness of first and
    last material is infinite").

    The rule survives unchanged: ONE number per row. The bug that started it read
    "2: layer 1" (an all-media prefix beside an interior-counting role string), with the
    tensor-panel title compounding it into "layer 2: layer 1"."""
    n = index + 1
    if name:  # a user/preset name replaces the generic role; the number always leads
        return f"{n}: {name}"
    if index == 0:
        return f"{n}: ambient (semi-infinite)"
    if index == total - 1:
        return f"{n}: substrate (semi-infinite)"
    return f"{n}: film"  # ONE number per row — never "2: layer 1"


def interior_layer_count(stack: list[dict]) -> int:
    """The ORIGINAL's "Number of Layers": the INTERIOR layers only.

    The released Mathematica .ml GUI numbers the interior media 1..materialnumber
    (`SHAARP.ml.nb:3707-3712`, `matindex in Range[2, materialnumber+1]`) and treats the two
    half-spaces as unnumbered ambient/exit media it fixes to air; its stack is
    `materialnumber + 2` media long. This port keeps the SAME counting convention while letting
    the user set the half-space media (the extension)."""
    return max(len(stack) - 2, 0)


def set_interior_layer_count(stack: list[dict], n_interior: int) -> list[dict]:
    """Grow/shrink to exactly ``n_interior`` INTERIOR layers, keeping both half-spaces.

    The GUI spin speaks the original's language (Fig 4 = 2 layers: quartz + Au); the stack model
    still stores ambient + interiors + substrate, so this is the one place the two counts meet."""
    return set_layer_count(stack, int(n_interior) + 2)


def set_layer_count(stack: list[dict], n: int) -> list[dict]:
    """Return a new stack with exactly ``n`` TOTAL media (>=2, i.e. counting the two half-spaces):
    grow by inserting interior film layers before the substrate, shrink by removing interior
    layers (never the two half-spaces).

    NOTE: the GUI's "Number of Layers" spin counts INTERIOR layers, matching the original —
    use :func:`set_interior_layer_count` / :func:`interior_layer_count` for that view."""

    if n < 2:
        raise ValueError("a stack needs at least 2 layers (ambient + substrate)")
    s = [dict(layer) for layer in stack]
    if len(s) < 2:
        s = default_stack()
    while len(s) < n:
        s.insert(len(s) - 1, default_layer_spec("LiNbO3 z-cut · 1550 nm", 1.0, True))
    while len(s) > n:
        if len(s) <= 2:
            break
        s.pop(len(s) - 2)
    return s


# --- the layer editor is the SINGLE stack/thickness source for every ML mode -------------
#
# Rui: "you already have layer thickness definition in layer definition, why would
# you need another film thickness definition" — the standalone film-thickness spin is gone; the
# simple 3-layer modes (Custom film / Film: <case>) and the named presets all express their stack
# THROUGH this model, so what the editor shows is always what the compute uses.

# Named presets may carry materials whose .name is NOT a palette label (the Fig-4 docs case builds
# "Z-cut quartz (docs)" / "Au coating (docs)" from transcribed setup.nb constants). At the preset
# wavelength (0.8 um, a registry grid node) the palette materials are numerically identical (Au
# EXACTLY; quartz to 9+ decimals — same Sellmeier), so the display mapping below is faithful; the
# pure mapping-equivalence fence in tests/test_stack_single_source.py pins this against registry
# drift. The compute path for an UNEDITED preset never touches this mapping (the factory system is
# used directly), so the validated preset references cannot be perturbed by it.
PRESET_MATERIAL_LABELS = {
    "Air": "air",
    "air": "air",
    "Z-cut quartz (docs)": "Quartz z-cut (800 nm)",
    "Au coating (docs)": "Au coating (800 nm)",
}

def isotropic_layer_spec(n_w: float = 1.0, n_2w: float = 1.0, *, name: str | None = None) -> dict:
    """A half-space (or interior) layer that is just an isotropic index pair — the form of
    the ambient and substrate media. ``spec["iso_n"] = [n(w), n(2w)]``."""
    spec = {"material": ISOTROPIC_LAYER_CHOICE, "thickness_um": 0.0,
            "iso_n": [float(n_w), float(n_2w)]}
    if name:
        spec["name"] = str(name)
    return spec


def _material_from_iso(n_w: float, n_2w: float, name: str = "isotropic medium"):
    """The exact isotropic-Material construction the simple-mode builders use for their substrate
    (eps = n^2 * I, point group 1, zero d) — shared so the editor path and the builder path can
    never drift apart numerically."""
    import numpy as np

    from .config import CrystalOrientation, CrystalStructure, Material

    return Material(
        name=name, structure=CrystalStructure(point_group="∞∞m"),
        orientation=CrystalOrientation(),
        epsilon_omega=(np.eye(3) * float(n_w) ** 2).astype(complex),
        epsilon_2omega=(np.eye(3) * float(n_2w) ** 2).astype(complex),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )


def stack_halfspace_n(stack: list[dict], which: str) -> tuple[float, float] | None:
    """The (n(w), n(2w)) pair of a half-space row when it IS a plain isotropic medium — 'air'
    reads as (1, 1); a case-study / custom crystal row returns None (the caller must treat the
    half-space as a full material, e.g. Fig-6's anisotropic Al2O3 substrate)."""
    spec = stack[0 if which == "top" else -1]
    mat = spec.get("material")
    if mat == "air":
        return (1.0, 1.0)
    if mat == ISOTROPIC_LAYER_CHOICE:
        n = spec.get("iso_n") or [1.0, 1.0]
        return (float(n[0]), float(n[1]))
    return None


def set_halfspace_n(stack: list[dict], which: str, n_w: float, n_2w: float) -> None:
    """Turn a half-space row into (or update it as) a plain isotropic medium in place."""
    i = 0 if which == "top" else len(stack) - 1
    keep_name = stack[i].get("name")
    stack[i] = isotropic_layer_spec(n_w, n_2w, name=keep_name)


# ---- session serialization (R15 closure): layer specs are plain dicts except for
# the COMPLEX numbers inside per-layer custom snapshots (eps/d entries), which json cannot
# carry. encode/decode walk the spec tree tagging complex values; round-trip is exact. ----

def _enc(v):
    if isinstance(v, complex):
        return {"__complex__": [v.real, v.imag]}
    if isinstance(v, dict):
        return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    return v


def _dec(v):
    if isinstance(v, dict):
        if set(v.keys()) == {"__complex__"}:
            re_, im_ = v["__complex__"]
            return complex(float(re_), float(im_))
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def encode_stack(stack: list[dict]) -> list:
    """JSON-safe form of a layer-spec list (complex values tagged)."""
    return _enc(list(stack))


def decode_stack(data: list) -> list[dict]:
    """Inverse of :func:`encode_stack`."""
    out = _dec(list(data))
    if not isinstance(out, list) or not all(isinstance(s, dict) and "material" in s for s in out):
        raise ValueError("not an encoded layer stack")
    return out


def custom_spec_snapshot_from_material(mat) -> dict:
    """A faithful ``spec["custom"]`` snapshot of an arbitrary Material, consumable by
    :func:`_material_from_custom_spec`. CRITICAL frame note: build_custom_si_material treats eps
    and d as CRYSTAL-frame inputs and applies the orientation itself, so this snapshots the raw
    crystal-frame dataclass fields (NOT any lab-frame accessor — that would double-rotate)."""
    st = mat.structure
    return {
        "point_group": str(st.point_group),
        "lattice": [float(st.a), float(st.b), float(st.c),
                    float(st.alpha_deg), float(st.beta_deg), float(st.gamma_deg)],
        "orientation_mode": "Crystal Physics Directions (Z1,Z2,Z3)",
        "z_axes": [[float(x) for x in row] for row in mat.orientation.z_axes_in_lab],
        "eps_omega_full": [[complex(v) for v in row] for row in mat.epsilon_omega],
        "eps_2omega_full": [[complex(v) for v in row] for row in mat.epsilon_2omega],
        "d_full": [[complex(v) for v in row] for row in mat.d_voigt_pm_v],
    }


def stack_from_system(system) -> list[dict]:
    """Layer-editor specs mirroring a resolved MultilayerSystem (a named preset's REAL stack).
    Palette-labelled where the material is (or aliases to) a palette entry; otherwise a faithful
    per-layer custom snapshot — so the editor always shows, and a flipped copy always computes,
    the true stack."""
    specs: list[dict] = []
    for L in system.layers:
        name = str(getattr(L.material, "name", "") or "")
        label = PRESET_MATERIAL_LABELS.get(name)
        if label is None and name in {lbl for lbl, _k in GUI_ML_CASES}:
            label = name  # already a palette display label (Fig 6/7 presets build from the registry)
        spec = {
            "material": label if label is not None else CUSTOM_LAYER_CHOICE,
            "thickness_um": float(L.thickness_um or 0.0),
            "analytic_h": bool(getattr(L, "analytic_h", False)),
            "analytic_d": bool(getattr(L, "analytic_d", False)),
        }
        if label is None:
            spec["custom"] = custom_spec_snapshot_from_material(L.material)
        if getattr(L, "name", ""):
            spec["name"] = str(L.name)
        specs.append(spec)
    return specs


def simple_film_stack(film_material: str, thickness_um: float,
                      top_n: tuple[float, float] = (1.0, 1.0),
                      bottom_n: tuple[float, float] = (1.45, 1.46)) -> list[dict]:
    """The 3-layer template behind the simple modes: isotropic ambient / film / isotropic
    substrate. The film row's thickness IS the mode's film thickness, and the two half-space
    rows' n(w)/n(2w) ARE the mode's ambient and substrate media (F56 — the old Substrate group
    relocated into the editor; defaults reproduce it exactly: air on top, 1.45/1.46 below)."""
    return [
        isotropic_layer_spec(*top_n, name="ambient"),
        default_layer_spec(film_material, float(thickness_um), True, analytic_h=True,
                           analytic_d=True),
        isotropic_layer_spec(*bottom_n, name="substrate"),
    ]


def stack_film_thickness_um(stack: list[dict]) -> float:
    """Thickness of the first finite interior layer — the 'film' the simple-mode builders take."""
    for i in range(1, max(len(stack) - 1, 1)):
        return float(stack[i]["thickness_um"])
    return 1.0


def spec_point_group(spec: dict, material=None) -> str:
    """The point group a spec's layer carries: the Custom snapshot's, else the built material's."""
    if spec.get("material") == CUSTOM_LAYER_CHOICE and spec.get("custom"):
        return str(spec["custom"].get("point_group") or "")
    if material is not None:
        return str(getattr(getattr(material, "structure", None), "point_group", "") or "")
    return ""


def spec_shg_active(spec: dict, index: int, total: int, material=None) -> bool:
    """SHG activity is DECIDED BY THE POINT GROUP (the original's two popups), never by a
    switch. A layer is a nonlinear source iff it is an interior layer, not an isotropic/air
    medium, and its point group is in the original's "Noncentrosymmetric ->" list. A legacy
    ``"shg_active"`` spec key is ignored. Note the validated factory presets and benchmarks pass
    ``Layer(shg_active=...)`` explicitly (placeholder point group "1" on isotropic media with an
    explicit False) -- that headless path is untouched; this rule applies to editor stacks."""
    from .point_groups import is_shg_active

    if index == 0 or index == total - 1:
        return False
    if spec.get("material") in ("air", ISOTROPIC_LAYER_CHOICE):
        return False
    return is_shg_active(spec_point_group(spec, material))


def build_system_from_stack(stack: list[dict], *, wavelength_um: float = 1.064,
                            theta_deg: float = 0.0) -> MultilayerSystem:
    """Assemble a validated MultilayerSystem from the per-layer specs at the given wavelength.
    Interior layers carry their thickness; the two half-spaces (first/last) are thickness-less."""

    if len(stack) < 2:
        raise ValueError("stack needs at least 2 layers")
    layers = []
    last = len(stack) - 1
    for i, spec in enumerate(stack):
        if spec.get("material") == ISOTROPIC_LAYER_CHOICE:
            n = spec.get("iso_n") or [1.0, 1.0]
            role = "ambient" if i == 0 else "substrate" if i == last else f"layer {i + 1}"
            mat = _material_from_iso(float(n[0]), float(n[1]), name=f"isotropic {role}")
        elif spec.get("material") == CUSTOM_LAYER_CHOICE and spec.get("custom"):
            mat = _material_from_custom_spec(spec["custom"], wavelength_um)  # per-layer custom crystal
        else:
            mat = _layer_material(spec["material"], wavelength_um)
        is_halfspace = i == 0 or i == last
        # user-assigned layer name wins (original .ml: "each layer can be assigned a name"); the
        # role: material auto-label stays the fallback.
        nm = (str(spec["name"]).strip() if spec.get("name")
              else ("ambient" if i == 0 else "substrate" if i == last else f"layer {i + 1}")
              + f": {spec['material']}")
        if is_halfspace:
            # a semi-infinite medium can be neither an SHG source nor symbolic in
            # thickness -- guaranteed in the MODEL, not just by hiding the checkboxes.
            layers.append(Layer(nm, mat, shg_active=False, analytic_h=False, analytic_d=False))
        else:
            active = spec_shg_active(spec, i, len(stack), mat)  # the point group decides
            layers.append(Layer(nm, mat, thickness_um=float(spec["thickness_um"]),
                                 shg_active=active,
                                 analytic_h=bool(spec.get("analytic_h", False)),
                                 analytic_d=bool(spec.get("analytic_d", False)) and active))
    return MultilayerSystem(
        wavelength_um=float(wavelength_um),
        polarimetry=Polarimetry(theta_deg=float(theta_deg), phi_deg=0.0, psi_deg=0.0),
        layers=layers,
    )
