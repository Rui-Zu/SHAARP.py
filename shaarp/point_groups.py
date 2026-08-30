"""Point groups, SHG activity and crystal-system lattice rules -- ONE source of truth.

The original ♯SHAARP.ml GUI offers every layer TWO popups (``SHAARP.ml.nb:5191`` and ``:5630``):
``Noncentrosymmetric ->`` (23 entries, 20 crystallographic classes + the three SHG-active Curie
groups) and ``Centrosymmetric ->`` (16 entries: the 11 centrosymmetric classes, 432 -- filed there
because its Kleinman-symmetric d vanishes -- and the four centrosymmetric/isotropic Curie groups).
Each entry is ``{label, 3x6 pattern, flag}``; flag 0 groups carry an all-zero d and still run the
whole pipeline with P_NL = 0. There is NO per-layer "SHG active" switch in the original -- activity
IS the list the group came from. ♯SHAARP.si (V1.03/V1.04) offers only the 23 active labels.

This module carries the lists in the original's order and spelling, the alias table for every
spelling the notebooks and the port use, the SHG-activity rule, and the crystal-system lattice
constraints the GUI locks (the GUI must "ensure crystal symmetry consistency and lattice
consistency"). No Qt, no sympy: ``config.py``/``symbolic.py``/the GUI import from here.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- the original's two popups, verbatim order --------------------------------------------------
SHG_ACTIVE_GROUPS: tuple[str, ...] = (
    "1", "2", "m", "mm2", "222", "3", "32", "3m", "4", "6", "-4", "4mm", "6mm",
    "422", "622", "-42m", "-6", "-6m2", "-43m", "23", "∞", "∞m", "∞2",
)
SHG_INACTIVE_GROUPS: tuple[str, ...] = (
    "-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m3", "m3m",
    "432", "∞/m", "∞/mm", "∞∞", "∞∞m",
)
ALL_POINT_GROUPS: tuple[str, ...] = SHG_ACTIVE_GROUPS + SHG_INACTIVE_GROUPS
POINT_GROUP_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Noncentrosymmetric (SHG-active)", SHG_ACTIVE_GROUPS),
    ("Centrosymmetric (SHG-inactive)", SHG_INACTIVE_GROUPS),
)

# --- crystal systems ------------------------------------------------------------------------------
_SYSTEM_OF: dict[str, str] = {}
for _pg in ("1", "-1"):
    _SYSTEM_OF[_pg] = "triclinic"
for _pg in ("2", "m", "2/m"):
    _SYSTEM_OF[_pg] = "monoclinic"
for _pg in ("222", "mm2", "mmm"):
    _SYSTEM_OF[_pg] = "orthorhombic"
for _pg in ("4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm"):
    _SYSTEM_OF[_pg] = "tetragonal"
for _pg in ("3", "-3", "32", "3m", "-3m"):
    _SYSTEM_OF[_pg] = "trigonal"
for _pg in ("6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm"):
    _SYSTEM_OF[_pg] = "hexagonal"
for _pg in ("23", "m3", "432", "-43m", "m3m"):
    _SYSTEM_OF[_pg] = "cubic"
for _pg in ("∞", "∞m", "∞2", "∞/m", "∞/mm"):
    _SYSTEM_OF[_pg] = "curie_axial"      # locks like hexagonal (one unique axis)
for _pg in ("∞∞", "∞∞m"):
    _SYSTEM_OF[_pg] = "curie_isotropic"  # locks like cubic
assert set(_SYSTEM_OF) == set(ALL_POINT_GROUPS)

# --- aliases: every spelling the notebooks / the port / a user may type -> canonical label ----------
_BAR = "\u0304"  # combining macron, e.g. "4̄"


def _overscript_forms(n: str) -> tuple[str, ...]:
    """Mathematica spellings of an n-bar: Overscript[n,_] and OverscriptBox, spaces removed."""
    return (
        f"overscript[{n},_]",
        f"\\!\\(\\*overscriptbox[\\({n}\\),\\(_\\)]\\)",
        f"overscriptbox[\\({n}\\),\\(_\\)]",
        f"{n}{_BAR}",
        f"{n}bar",
    )


_ALIASES: dict[str, str] = {}


def _alias(canonical: str, *spellings: str) -> None:
    for s in spellings:
        _ALIASES[s.lower().replace(" ", "")] = canonical


for _n, _canon in (("1", "-1"), ("3", "-3"), ("4", "-4"), ("6", "-6")):
    _alias(_canon, *_overscript_forms(_n))
for _n, _suffix, _canon in (("4", "2m", "-42m"), ("4", "3m", "-43m"), ("6", "m2", "-6m2"),
                            ("6", "2m", "-6m2"), ("3", "m", "-3m")):
    _alias(_canon, *(f"{f}{_suffix}" for f in _overscript_forms(_n)))
_alias("-6m2", "-62m", "6barm2", "6bar2m")          # the .si GUI popup spells it 6̄2m
_alias("-43m", "43m", "td")
_alias("-42m", "4bar2m", "d2d")
_alias("m3", "m-3", "m3bar", "m" + "3" + _BAR, "th")
_alias("m3m", "m-3m", "m3barm", "m" + "3" + _BAR + "m", "oh")
_alias("-3m", "3barm", "d3d")
_alias("432", "o")
_alias("23", "t")
_alias("3m", "c3v")
_alias("4mm", "c4v")
_alias("6mm", "c6v")
_alias("mm2", "c2v")
_alias("222", "d2")
_alias("-1", "ci", "1bar")
_alias("2/m", "c2h")
_alias("mmm", "d2h")
_alias("6/mmm", "d6h")
_alias("4/mmm", "d4h")
_alias("∞", "inf", "infinity", "infinite", "\\[infinity]")
_alias("∞m", "infm", "infinitym", "\\[infinity]m", "∞mm", "infmm")
_alias("∞2", "inf2", "infinity2", "\\[infinity]2", "∞22", "inf22")
_alias("∞/m", "inf/m", "infinity/m", "\\[infinity]/m")
_alias("∞/mm", "inf/mm", "infinity/mm", "\\[infinity]/mm", "∞/mmm", "inf/mmm")
_alias("∞∞", "infinf", "infinityinfinity", "\\[infinity]\\[infinity]")
_alias("∞∞m", "infinfm", "infinityinfinitym", "\\[infinity]\\[infinity]m")
for _pg in ALL_POINT_GROUPS:
    _alias(_pg, _pg)


def canonical_point_group(label: str) -> str:
    """The canonical label (original spelling) for any known alias; unknown input is returned
    stripped, unchanged, so callers can report it."""
    if label is None:
        return ""
    key = str(label).strip().lower().replace(" ", "")
    return _ALIASES.get(key, str(label).strip())


def is_known_point_group(label: str) -> bool:
    return canonical_point_group(label) in _SYSTEM_OF


def is_shg_active(label: str) -> bool:
    """True for the original's ``Noncentrosymmetric`` list (flag 1). Unknown labels are NOT
    active -- a material we cannot classify must never become a nonlinear source by accident."""
    return canonical_point_group(label) in SHG_ACTIVE_GROUPS


def crystal_system(label: str) -> str:
    pg = canonical_point_group(label)
    try:
        return _SYSTEM_OF[pg]
    except KeyError:
        raise ValueError(f"Unknown point group {label!r}") from None


def symbolic_key(label: str) -> str:
    """The key ``shaarp.symbolic.d_voigt_symbolic`` branches on (ASCII for the Curie groups)."""
    pg = canonical_point_group(label)
    return {"∞": "inf", "∞m": "infm", "∞2": "inf2"}.get(pg, pg)


# --- crystal-system lattice constraints (locking rule) ----------------------------------------
@dataclass(frozen=True)
class LatticeLock:
    """Which of (a, b, c, alpha, beta, gamma) are fixed by the crystal system.

    ``b_equals_a`` / ``c_equals_a`` tie lengths to ``a``; an angle value pins that angle; ``None``
    leaves it free. Locked cells are coerced by :func:`apply_lattice_constraints` and the GUI
    greys them out."""

    b_equals_a: bool = False
    c_equals_a: bool = False
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None

    @property
    def locked_indices(self) -> tuple[int, ...]:
        out = []
        if self.b_equals_a:
            out.append(1)
        if self.c_equals_a:
            out.append(2)
        for k, v in ((3, self.alpha), (4, self.beta), (5, self.gamma)):
            if v is not None:
                out.append(k)
        return tuple(out)

    def describe(self) -> str:
        parts = []
        if self.b_equals_a and self.c_equals_a:
            parts.append("a = b = c")
        elif self.b_equals_a:
            parts.append("b = a")
        for name, v in (("α", self.alpha), ("β", self.beta), ("γ", self.gamma)):
            if v is not None:
                parts.append(f"{name} = {v:g}°")
        return ", ".join(parts) if parts else "free"


_LOCKS: dict[str, LatticeLock] = {
    "triclinic": LatticeLock(),
    "monoclinic": LatticeLock(alpha=90.0, gamma=90.0),               # b-unique (config.py z2 = b)
    "orthorhombic": LatticeLock(alpha=90.0, beta=90.0, gamma=90.0),
    "tetragonal": LatticeLock(b_equals_a=True, alpha=90.0, beta=90.0, gamma=90.0),
    "trigonal": LatticeLock(b_equals_a=True, alpha=90.0, beta=90.0, gamma=120.0),   # hexagonal setting
    "hexagonal": LatticeLock(b_equals_a=True, alpha=90.0, beta=90.0, gamma=120.0),
    "cubic": LatticeLock(b_equals_a=True, c_equals_a=True, alpha=90.0, beta=90.0, gamma=90.0),
    "curie_axial": LatticeLock(b_equals_a=True, alpha=90.0, beta=90.0, gamma=120.0),
    "curie_isotropic": LatticeLock(b_equals_a=True, c_equals_a=True, alpha=90.0, beta=90.0, gamma=90.0),
}


def lattice_constraints(label: str) -> LatticeLock:
    return _LOCKS[crystal_system(label)]


def apply_lattice_constraints(label: str, six) -> tuple[float, float, float, float, float, float]:
    """Coerce ``(a, b, c, alpha, beta, gamma)`` to the point group's crystal system (pure)."""
    a, b, c, al, be, ga = (float(x) for x in six)
    lock = lattice_constraints(label)
    if lock.b_equals_a:
        b = a
    if lock.c_equals_a:
        c = a
    if lock.alpha is not None:
        al = lock.alpha
    if lock.beta is not None:
        be = lock.beta
    if lock.gamma is not None:
        ga = lock.gamma
    return (a, b, c, al, be, ga)


__all__ = [
    "SHG_ACTIVE_GROUPS", "SHG_INACTIVE_GROUPS", "ALL_POINT_GROUPS", "POINT_GROUP_SECTIONS",
    "canonical_point_group", "is_known_point_group", "is_shg_active", "crystal_system",
    "symbolic_key", "LatticeLock", "lattice_constraints", "apply_lattice_constraints",
]
