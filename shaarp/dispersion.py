"""Refractive-index tables, so a material can carry its own dispersion.

WHY THIS EXISTS. The exported case-study registry stores a permittivity grid per material, but
most of those grids are constant: the original notebook defined the material with a fixed index, so
a wavelength sweep over it returns a flat line. Every single-interface palette case is in that
position. This module lets a material carry a real n(lambda), k(lambda) table instead -- the form
an index is normally published and tabulated in -- so a spectrum computed from it means something.

THE TABLE. A CSV with a header row, wavelengths ascending, one row per wavelength:

    wavelength_um, nx, ny, nz, kx, ky, kz

The three principal axes are the crystal-physics axes, in the same order as the permittivity
tensors elsewhere. Two shorthands expand into that same shape, because most published data comes
in one of them:

    wavelength_um, n, k                     an isotropic medium
    wavelength_um, no, ne, ko, ke           a uniaxial crystal (ordinary, extraordinary)

The extinction columns are optional and default to zero, which is the right default for a
transparent crystal inside its window.

THE HALF-WAVELENGTH RULE. The permittivity at the second harmonic is the table evaluated at
lambda/2, so a table has to cover ``[lambda_min / 2, lambda_max]`` to support a sweep over
``[lambda_min, lambda_max]`` -- it reaches the blue end of its table twice as fast as you would
expect. Past either end the index is held CONSTANT at the nearest tabulated value and a warning
says so. Announced, that is a defensible approximation; the thing to avoid is the SILENT version,
which is how the exported registry came to report physical-looking numbers outside the range its
own model was valid over.

WHAT A TABLE DOES NOT CARRY. The SHG tensor, the point group, the lattice and the orientation come
from a template material -- a table is linear-optical data. ``d`` is held constant across a sweep;
see the technical reference for what that assumption costs.
"""
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Material

# Accepted spellings for the wavelength column, lower-cased and stripped.
_WAVELENGTH_KEYS = ("wavelength_um", "wavelength", "lambda_um", "lambda", "wl_um", "wl")

# Each layout maps its index columns (and matching extinction columns) onto the three principal
# axes. The uniaxial form repeats the ordinary index on x and y, which is what "ordinary" means.
_LAYOUTS = (
    (("nx", "ny", "nz"), ("kx", "ky", "kz"), (0, 1, 2)),
    (("no", "ne"), ("ko", "ke"), (0, 0, 1)),
    (("n",), ("k",), (0, 0, 0)),
)


@dataclass(frozen=True)
class IndexTable:
    """Tabulated principal n and k against wavelength, in micrometres."""

    name: str
    wavelength_um: np.ndarray          # [m], ascending
    n: np.ndarray                      # [m, 3] principal refractive indices
    k: np.ndarray                      # [m, 3] principal extinction coefficients
    source: str = ""

    @property
    def range_um(self) -> tuple[float, float]:
        return (float(self.wavelength_um[0]), float(self.wavelength_um[-1]))

    def covers(self, lambda_min_um: float, lambda_max_um: float) -> bool:
        """Can this table support a sweep over ``[lambda_min_um, lambda_max_um]``?

        The second harmonic is read at half the fundamental, so the table has to reach down to
        ``lambda_min_um / 2``."""
        low, high = self.range_um
        return (low <= float(lambda_min_um) / 2.0 + 1e-12
                and float(lambda_max_um) <= high + 1e-12)

    def principal(self, lambda_um: float) -> np.ndarray:
        """Complex principal indices ``n + i k`` at one wavelength, as a length-3 array.

        Outside the tabulated range the index is held CONSTANT at the nearest tabulated end and a
        warning says so. That is an approximation, not a measurement -- the material's real index
        keeps moving out there -- but it is a defensible one as long as it is announced, and it
        lets a sweep run past a table's edge instead of stopping dead. Note that a sweep reads the
        table at HALF the fundamental as well, so it reaches the low end twice as fast as you
        would expect."""
        lam = float(lambda_um)
        low, high = self.range_um
        if not (low - 1e-12 <= lam <= high + 1e-12):
            warnings.warn(
                f"{self.name} has index data over {low:g}-{high:g} um; at {lam:g} um the index is "
                "held constant at the nearest tabulated end rather than extrapolated. Remember a "
                "sweep also reads the table at half the fundamental.",
                RuntimeWarning, stacklevel=3)
        out = np.empty(3, dtype=complex)
        for axis in range(3):
            real = float(np.interp(lam, self.wavelength_um, self.n[:, axis]))
            imag = float(np.interp(lam, self.wavelength_um, self.k[:, axis]))
            out[axis] = complex(real, imag)
        return out

    def epsilon(self, lambda_um: float) -> np.ndarray:
        """Diagonal relative permittivity at one wavelength: ``eps = (n + i k)**2``.

        The sign convention matches the rest of the package, where an absorbing medium carries a
        POSITIVE imaginary permittivity."""
        return np.diag(self.principal(lambda_um) ** 2)


def _normalise(field: str) -> str:
    return field.strip().lower().lstrip("﻿")


def load_index_table(path, *, name: str | None = None, source: str = "") -> IndexTable:
    """Read an index table from CSV. See the module docstring for the accepted column layouts."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    rows = [r for r in rows if r and not str(r[0]).strip().startswith("#")]
    if len(rows) < 2:
        raise ValueError(f"{path} needs a header row and at least one data row.")
    header = [_normalise(c) for c in rows[0]]
    index_of = {field: position for position, field in enumerate(header)}

    wavelength_key = next((key for key in _WAVELENGTH_KEYS if key in index_of), None)
    if wavelength_key is None:
        raise ValueError(
            f"{path} has no wavelength column; expected one of {list(_WAVELENGTH_KEYS)} "
            f"but found {header}.")

    for n_fields, k_fields, axis_map in _LAYOUTS:
        if all(field in index_of for field in n_fields):
            break
    else:
        raise ValueError(
            f"{path} has no recognised index columns. Use 'nx, ny, nz' for a biaxial crystal, "
            f"'no, ne' for a uniaxial one, or 'n' for an isotropic medium; found {header}.")

    wavelengths: list[float] = []
    n_values: list[list[float]] = []
    k_values: list[list[float]] = []
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) < len(header):
            row = list(row) + [""] * (len(header) - len(row))
        try:
            wavelengths.append(float(row[index_of[wavelength_key]]))
            raw_n = [float(row[index_of[f]]) for f in n_fields]
            raw_k = [float(row[index_of[f]]) if f in index_of and str(row[index_of[f]]).strip()
                     else 0.0 for f in k_fields]
        except ValueError as exc:
            raise ValueError(f"{path} line {line_number}: {exc}") from exc
        n_values.append([raw_n[axis] for axis in axis_map])
        k_values.append([raw_k[axis] for axis in axis_map])

    order = np.argsort(np.asarray(wavelengths, dtype=float))
    wavelength_um = np.asarray(wavelengths, dtype=float)[order]
    if np.any(np.diff(wavelength_um) <= 0):
        raise ValueError(f"{path} has repeated wavelengths; each row needs its own.")
    table = IndexTable(name=name or path.stem,
                       wavelength_um=wavelength_um,
                       n=np.asarray(n_values, dtype=float)[order],
                       k=np.asarray(k_values, dtype=float)[order],
                       source=source)
    if np.any(table.n <= 0):
        raise ValueError(f"{path} has a non-positive refractive index.")
    if np.any(table.k < 0):
        raise ValueError(f"{path} has a negative extinction coefficient.")
    return table


def table_spectrum(table: IndexTable, template: Material, *, name: str | None = None):
    """Build a wavelength -> Material factory from an index table.

    The table supplies the LINEAR optics only. Everything else -- point group, lattice,
    orientation and the SHG tensor -- comes from ``template``, so a table can give an existing
    case-study material real dispersion without changing anything else about it.

    The permittivity at the fundamental is the table at ``lambda``; at the second harmonic it is
    the table at ``lambda / 2``."""

    label = name or f"{template.name} ({table.name})"

    def _at(lambda_um: float) -> Material:
        lam = float(lambda_um)
        return Material(
            name=label,
            structure=template.structure,
            orientation=template.orientation,
            epsilon_omega=table.epsilon(lam),
            epsilon_2omega=table.epsilon(lam / 2.0),
            d_voigt_pm_v=np.asarray(template.d_voigt_pm_v, dtype=complex),
        )

    _at.material_names = ()            # type: ignore[attr-defined]
    _at.index_table = table            # type: ignore[attr-defined]
    return _at


# ---------------------------------------------------------------------------------------------
# The shipped tables. Generated by benchmarks/generate_dispersion_tables.py from the
# refractiveindex.info database's YAML sources (CC0), each carrying the primary literature it
# came from. Regenerate with that script; do not hand-edit the CSVs.
# ---------------------------------------------------------------------------------------------

_DATA_DIR = "dispersion_data"


def _data_path(name: str) -> Path:
    try:
        from importlib.resources import files

        return Path(str(files("shaarp") / _DATA_DIR / name))
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        return Path(__file__).with_name(_DATA_DIR) / name


def _registry() -> dict:
    import json

    path = _data_path("index.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("materials", {})


def range_label(low_um: float, high_um: float) -> str:
    """The wavelength range as it appears in a material's display name."""
    return f"{low_um:.2f}-{high_um:.2f} um"


def shipped_tables() -> dict:
    """slug -> metadata for every dispersive material that ships with the package."""
    return dict(_registry())


def dispersive_material_names() -> list[str]:
    """Display names for the dispersive variants, each CARRYING ITS WAVELENGTH RANGE.

    The range belongs in the name because it is the first thing that decides whether a material
    can answer the sweep you are about to run -- and because a table's usable span is a property
    of the measurement it came from, not of the crystal."""
    out = []
    for meta in _registry().values():
        low, high = meta["range_um"]
        out.append(f"{meta['display']} {range_label(low, high)}")
    return out


def _slug_for_display(display_name: str) -> str | None:
    for slug, meta in _registry().items():
        low, high = meta["range_um"]
        if display_name in (meta["display"], f"{meta['display']} {range_label(low, high)}"):
            return slug
    return _slug_for_display_exact(display_name)


def _slug_for_display_exact(display_name: str) -> str | None:
    return display_name if display_name in _registry() else None


def resolve_dispersive_name(name: str) -> str | None:
    """The full display name (range included) for any name a dispersive material answers to:
    the full name, the short name the documentation's table prints ("LiNbO3 (dispersive)"), or
    the table's slug. ``None`` when ``name`` is not a dispersive material.

    Every entry point resolves through this one function. The full-name-only check the others
    used to make turned the short name -- the one a reader copies out of the docs -- into an
    "unknown material" error whose list of choices left the dispersive materials out entirely."""
    # subscript digits too: the documentation's table sets formulae as LiNbO₃ and LiB₃O₅, and a name
    # copied from it used to fail for exactly those two of the five
    plain = str(name).strip().translate(_SUBSCRIPT_DIGITS)
    slug = _slug_for_display(plain)
    if slug is None:
        return None
    meta = _registry()[slug]
    low, high = meta["range_um"]
    return f"{meta['display']} {range_label(low, high)}"


_SUBSCRIPT_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def dispersive_twin(registry_name: str) -> tuple[str, str] | None:
    """(full display name, template) of the dispersive variant of this Case Study material's
    crystal, or ``None`` when that crystal has no index table.

    Built on exactly this material ("KTP (100)") is the best match; the same crystal in another cut
    or at another wavelength ("KTP x-cut", "GaAs (111) (800 nm)") is still the crystal a user wants
    a spectrum of, so it is offered too -- and the template is returned so the advice can say which
    cut the variant is. Quartz and a custom material have none, and the old advice that sent users
    to "the same crystal in the Dispersive group" for those pointed at nothing."""
    crystal = str(registry_name).split()[0] if str(registry_name).split() else ""
    same_crystal = None
    for meta in _registry().values():
        low, high = meta["range_um"]
        full = f"{meta['display']} {range_label(low, high)}"
        if meta.get("template") == registry_name:
            return full, meta["template"]
        if crystal and str(meta.get("display", "")).split()[0] == crystal and same_crystal is None:
            same_crystal = (full, meta.get("template", ""))
    return same_crystal


def load_shipped_table(slug_or_display: str) -> IndexTable:
    """Load one of the shipped index tables by slug or by display name."""
    slug = _slug_for_display(slug_or_display)
    if slug is None:
        raise ValueError(f"unknown dispersive material {slug_or_display!r}; choose from "
                         f"{sorted(_registry())}")
    meta = _registry()[slug]
    low, high = meta["range_um"]
    return load_index_table(_data_path(f"{slug}.csv"),
                            name=f"{meta['display']} {range_label(low, high)}",
                            source=meta["reference"])


def dispersive_spectrum(slug_or_display: str):
    """A wavelength -> Material factory for one of the shipped dispersive materials.

    The linear optics come from the published index table; the point group, lattice, orientation
    and SHG tensor come from the case-study material the table is paired with, so this is the same
    crystal the palette already offers, given a real index curve."""
    from .casestudy_materials import build_casestudy_material

    slug = _slug_for_display(slug_or_display)
    if slug is None:
        raise ValueError(f"unknown dispersive material {slug_or_display!r}")
    meta = _registry()[slug]
    table = load_shipped_table(slug)
    template = build_casestudy_material(meta["template"])
    return table_spectrum(table, template, name=table.name)
