"""- "My Materials": the user's own materials in a local JSON store.

Rui: the session-scoped preset slots were not what users need. A user SAVES the material currently
entered in the crystal panels (point group, lattice, orientation, eps(w)/eps(2w), d, n-spins -- the
very spec the per-layer "Custom (fields)" path already consumes) under a NAME; it then appears as a
selectable material on both tabs like a palette entry, can be UPDATED / RENAMED / DELETED, and the
built-in case studies are never touched: this store holds user entries only, and a name that
collides with a built-in label is refused.

Pure module (no Qt): the GUI and the layer-stack seam call it; tests point it at a temp file via
the ``SHAARP_USER_MATERIALS`` environment variable so the suite never writes the real store.

Store: ``~/.shaarp/user_materials.json`` (the repo's existing per-user folder; tempdir fallback),
written atomically. Schema::

    {"kind": "shaarp_user_materials", "version": 1,
     "materials": {"<name>": {"saved": "YYYY-MM-DD HH:MM:SS",
                              "wavelength_um": 0.8,
                              "spec": {... custom spec, complex values tagged ...}}}}

A saved material is SINGLE-wavelength (its tensors were entered at ``wavelength_um``); the GUI
applies the rule (selecting it sets the wavelength spin to that value).
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time

KIND = "shaarp_user_materials"
VERSION = 1
ENV_OVERRIDE = "SHAARP_USER_MATERIALS"
USER_SECTION_HEADER = "—  My Materials  —"
_SPEC_DROP = ("thickness", "wavelength")  # layer / global properties, never part of a material


# --- location -------------------------------------------------------------------------------------
def store_path() -> str:
    env = os.environ.get(ENV_OVERRIDE)
    if env:
        return env
    d = os.path.join(os.path.expanduser("~"), ".shaarp")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = tempfile.gettempdir()
    return os.path.join(d, "user_materials.json")


# --- read / write ---------------------------------------------------------------------------------
def load() -> dict:
    """``{name: entry}`` with complex values decoded; a missing or corrupt file is ``{}``."""
    from .layer_stack import _dec

    path = store_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("kind") != KIND:
        return {}
    mats = payload.get("materials")
    if not isinstance(mats, dict):
        return {}
    out = {}
    for name, entry in mats.items():
        if isinstance(entry, dict) and isinstance(entry.get("spec"), dict):
            out[str(name)] = {
                "saved": str(entry.get("saved", "")),
                "wavelength_um": float(entry.get("wavelength_um", 1.064)),
                "spec": _dec(entry["spec"]),
            }
    return out


def _write(mats: dict) -> None:
    from .layer_stack import _enc

    path = store_path()
    payload = {"kind": KIND, "version": VERSION,
               "materials": {name: {"saved": e["saved"], "wavelength_um": float(e["wavelength_um"]),
                                    "spec": _enc(e["spec"])}
                             for name, e in mats.items()}}
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def list_names() -> list[str]:
    return sorted(load().keys(), key=str.casefold)


def get(name: str) -> dict | None:
    return load().get(str(name).strip())


def is_user_material(label) -> bool:
    if label is None:
        return False
    return str(label).strip() in load()


# --- names ----------------------------------------------------------------------------------------
def reserved_names() -> set[str]:
    """Every label the combos already use (case-folded): built-in keys and display labels, group
    headers, the ML system presets and the structural choices. A user material may not shadow
    any of them -- combos select by text, so a near-duplicate would be a trap."""
    from .casestudy_materials import CASE_STUDY_ORDER, GUI_ML_GROUPS, GUI_SI_GROUPS
    from .layer_stack import CUSTOM_LAYER_CHOICE, ISOTROPIC_LAYER_CHOICE

    names: set[str] = set()
    names.update(CASE_STUDY_ORDER)
    for hdr, entries in GUI_ML_GROUPS:
        if hdr:
            names.add(hdr)
        for label, key in entries:
            names.add(label.strip())
            names.add(key)
    for hdr, entries in GUI_SI_GROUPS:
        names.add(hdr)
        for label, key in entries:
            names.add(label.strip())
            names.add(key)
    try:
        from .shaarp_gui import ML_SYSTEM_PRESETS
        names.update(ML_SYSTEM_PRESETS.keys())
    except Exception:  # pragma: no cover - the GUI module is optional for the pure store
        pass
    names.update({"air", "Air", "Custom (use fields)", "Custom film (use fields)",
                  "N-layer stack (editor)", CUSTOM_LAYER_CHOICE, ISOTROPIC_LAYER_CHOICE,
                  USER_SECTION_HEADER})
    return {n.casefold() for n in names}


def validate_name(name) -> str:
    text = "" if name is None else str(name).strip()
    if not text:
        raise ValueError("material name must not be empty")
    if text.startswith("—") or text.startswith("-  ") or text != str(name).strip() or "\n" in text:
        raise ValueError("material name must not look like a section header")
    if text.casefold() in reserved_names():
        raise ValueError(f"'{text}' is a built-in material or a reserved name; choose another")
    return text


# --- mutations ------------------------------------------------------------------------------------
def _clean_spec(spec: dict) -> dict:
    out = dict(spec or {})
    for k in _SPEC_DROP:
        out.pop(k, None)
    return out


def save(name, spec: dict, wavelength_um: float) -> str:
    """Create or OVERWRITE (= update) the entry ``name``. Returns the canonical name."""
    text = validate_name(name)
    mats = load()
    mats[text] = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "wavelength_um": float(wavelength_um), "spec": _clean_spec(spec)}
    _write(mats)
    return text


def rename(old, new) -> str:
    mats = load()
    old_t = str(old).strip()
    if old_t not in mats:
        raise KeyError(f"no user material named {old_t!r}")
    new_t = validate_name(new)
    if new_t != old_t and new_t in mats:
        raise ValueError(f"a user material named '{new_t}' already exists")
    entry = mats.pop(old_t)
    mats[new_t] = entry
    _write(mats)
    return new_t


def delete(name) -> bool:
    mats = load()
    text = str(name).strip()
    if text not in mats:
        return False
    mats.pop(text)
    _write(mats)
    return True


# --- building -------------------------------------------------------------------------------------
def build_user_material(name, wavelength_um: float | None = None):
    """The saved spec as a :class:`Material` (named after the entry)."""
    from .layer_stack import _material_from_custom_spec

    entry = get(name)
    if entry is None:
        raise ValueError(f"unknown user material {name!r}")
    lam = float(entry["wavelength_um"] if wavelength_um is None else wavelength_um)
    mat = _material_from_custom_spec(entry["spec"], lam)
    try:
        return dataclasses.replace(mat, name=str(name).strip())
    except Exception:  # pragma: no cover - non-dataclass Material
        return mat


__all__ = [
    "KIND", "VERSION", "ENV_OVERRIDE", "USER_SECTION_HEADER", "store_path", "load", "list_names",
    "get", "is_user_material", "reserved_names", "validate_name", "save", "rename", "delete",
    "build_user_material",
]
