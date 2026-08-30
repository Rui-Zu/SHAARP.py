"""Headless validation of user-supplied material properties (gui_set_material_properties).

A GUI "Set Material Properties" form needs to reject physically invalid input
*before* it reaches the solver. This module provides pure, side-effect-free
checks (no display, no plotting) that return human-readable issue lists a form
can surface inline. Two severities:

  * ``issues`` -- HARD physical/symmetry violations (the material is invalid):
      - a dielectric tensor that is not 3x3, not finite, not symmetric
        (reciprocity, eps_ij == eps_ji), or -- by default -- whose real part is
        not positive-definite (a passive transparent dielectric);
      - an SHG d-tensor that is not 3x6 or not finite;
      - a non-zero d-tensor declared under a centrosymmetric / non-SHG-active
        point group (second-harmonic generation is symmetry-forbidden there in
        *every* axis setting).

  * ``diagnostics`` -- SOFT, informational notes that do not by themselves make
    the material invalid:
      - an all-zero d-tensor under an SHG-active point group (likely unset);
      - a non-zero-Voigt-entry count that differs from the point group's
        structural fingerprint (func:`shaarp.symbolic.d_tensor_structure`).

The d-tensor checks are deliberately **convention-robust**: they rely only on
facts that hold in any standard axis setting (centrosymmetry forbids SHG; the
non-zero-entry *count* is a setting-invariant fingerprint). They intentionally do
NOT assert an exact zero-pattern match, because that is setting-dependent -- e.g.
LiNbO3 (3m) carries the d22-family in row 1 (mirror perpendicular x1) while the
symbolic tensor uses row 2 (mirror perpendicular x2). The dielectric and
centrosymmetry checks, by contrast, are exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .symbolic import d_tensor_structure


@dataclass(frozen=True)
class MaterialValidationReport:
    """Result of :func:`check_material_properties`.

    ``issues`` are hard violations (material is invalid); ``diagnostics`` are
    soft, informational notes. ``ok`` is True iff there are no hard issues."""

    issues: list[str]
    diagnostics: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues


def dielectric_tensor_issues(
    eps,
    name: str = "eps",
    *,
    atol: float = 1e-8,
    require_positive_definite: bool = True,
) -> list[str]:
    """Hard issues for a single dielectric tensor (3x3, finite, symmetric, and
    -- by default -- positive-definite real part)."""
    e = np.asarray(eps, dtype=complex)
    if e.shape != (3, 3):
        return [f"{name}: must be a 3x3 tensor, got shape {e.shape}"]
    if not np.all(np.isfinite(e.view(float))):
        # Stop here: symmetry / eigenvalue checks are meaningless on non-finite data.
        return [f"{name}: contains non-finite (NaN/Inf) entries"]
    issues: list[str] = []
    asym = float(np.max(np.abs(e - e.T)))
    if asym > atol:
        issues.append(f"{name}: not symmetric (reciprocity requires eps_ij == eps_ji), max|eps - eps^T| = {asym:.2e}")
    if require_positive_definite:
        real_sym = 0.5 * (e.real + e.real.T)
        min_eig = float(np.min(np.linalg.eigvalsh(real_sym)))
        if min_eig <= 0:
            issues.append(f"{name}: real part is not positive-definite (passive dielectric expected), min eigenvalue = {min_eig:.3g}")
    return issues


def d_tensor_issues_and_diagnostics(
    d_voigt,
    point_group: str,
    *,
    atol: float = 1e-9,
) -> tuple[list[str], list[str]]:
    """Return ``(issues, diagnostics)`` for an SHG d-tensor against its point group.

    Convention-robust: hard issues are basic shape/finiteness and the
    centrosymmetry rule; the non-zero-entry count is a soft diagnostic only."""
    issues: list[str] = []
    diagnostics: list[str] = []
    d = np.asarray(d_voigt, dtype=complex)
    if d.shape != (3, 6):
        return [f"d-tensor: must be a 3x6 Voigt matrix, got shape {d.shape}"], diagnostics
    if not np.all(np.isfinite(d.view(float))):
        return ["d-tensor: contains non-finite (NaN/Inf) entries"], diagnostics

    nonzero_count = int(np.sum(np.abs(d) > atol))

    from .point_groups import is_known_point_group, is_shg_active
    if is_known_point_group(point_group) and not is_shg_active(point_group):
        struct = None  # a known SHG-inactive group (d = 0 by symmetry)
    else:
        try:
            struct = d_tensor_structure(point_group)
        except ValueError:
            struct = None

    if struct is None:
        # point group is not a recognized SHG-active (non-centrosymmetric) class
        if nonzero_count > 0:
            issues.append(
                f"d-tensor is non-zero but point group '{point_group}' is not a recognized "
                f"non-centrosymmetric (SHG-active) crystal class; second-harmonic generation "
                f"requires a non-centrosymmetric point group"
            )
        # all-zero d under a centrosymmetric group is consistent -> no issue
        return issues, diagnostics

    expected_nonzero = len(struct.nonzero_entries)
    if nonzero_count == 0:
        diagnostics.append(
            f"point group '{point_group}' is SHG-active ({struct.independent_count} independent "
            f"constant(s)) but the d-tensor is all zero (unset?)"
        )
    elif nonzero_count != expected_nonzero:
        diagnostics.append(
            f"d-tensor has {nonzero_count} non-zero Voigt entries; point group '{point_group}' has "
            f"{expected_nonzero} in the standard setting -- check the declared point group or the axis "
            f"setting (the exact pattern is setting-dependent)"
        )
    return issues, diagnostics


def check_material_properties(
    material,
    *,
    require_positive_definite: bool = True,
    eps_atol: float = 1e-8,
    d_atol: float = 1e-9,
) -> MaterialValidationReport:
    """Validate a :class:`shaarp.config.Material`'s properties headlessly.

    Checks both dielectric tensors (omega and 2-omega) and the SHG d-tensor
    against the declared crystal point group. Returns a
    :class:`MaterialValidationReport` (hard ``issues`` + soft ``diagnostics``)."""
    issues: list[str] = []
    diagnostics: list[str] = []

    issues += dielectric_tensor_issues(
        material.eps_w(), "epsilon_omega", atol=eps_atol, require_positive_definite=require_positive_definite
    )
    issues += dielectric_tensor_issues(
        material.eps_2w(), "epsilon_2omega", atol=eps_atol, require_positive_definite=require_positive_definite
    )

    point_group = material.structure.point_group
    d_issues, d_diags = d_tensor_issues_and_diagnostics(material.d_voigt(), point_group, atol=d_atol)
    issues += d_issues
    diagnostics += d_diags

    return MaterialValidationReport(issues=issues, diagnostics=diagnostics)


def validate_material_properties(material, **kwargs) -> None:
    """Raise ``ValueError`` if the material has any hard validation issue.

    Soft diagnostics do not raise. Convenience wrapper over
    :func:`check_material_properties`."""
    report = check_material_properties(material, **kwargs)
    if not report.ok:
        raise ValueError("Invalid material properties: " + "; ".join(report.issues))


__all__ = [
    "MaterialValidationReport",
    "dielectric_tensor_issues",
    "d_tensor_issues_and_diagnostics",
    "check_material_properties",
    "validate_material_properties",
]
