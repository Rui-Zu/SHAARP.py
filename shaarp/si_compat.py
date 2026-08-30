from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import root

from .anisotropic import shaarp_ordered_generalized_eigensystem
from .boundary import boundary_residual, solve_boundary_continuity, solve_fresnel_boundary
from .nonlinear import (
    SingleInterfaceInhomogeneousFields,
    SingleInterfaceSources,
    single_interface_sources,
    solve_single_interface_inhomogeneous_fields,
)
from .waves import Wave


@dataclass(frozen=True)
class ShaarpSIActiveAngleRoots:
    """Active SHAARP.si angle roots and six-digit-style branch labels."""

    theta_ord: complex
    theta_ext: complex
    theta_ord_2omega: complex
    theta_ext_2omega: complex
    n_ord_label: complex
    n_ext_label: complex
    n_ord_2omega_label: complex
    n_ext_2omega_label: complex
    ext_factor: complex
    ext_2omega_factor: complex


@dataclass(frozen=True)
class ShaarpSIOmegaBranchIndices:
    """One-based active SHAARP.si omega eigenvector indices."""

    ext_index: int
    ord_index: int
    ext_post_factor_label: complex
    ord_post_factor_label: complex


@dataclass(frozen=True)
class ShaarpSI2OmegaBranchIndices:
    """One-based active SHAARP.si 2omega eigenvector indices."""

    ext_index: int
    ord_index: int
    ext_post_factor_label: complex
    ord_post_factor_label: complex


@dataclass(frozen=True)
class ShaarpSIHomogeneous2OmegaResult:
    """Active SHAARP.si homogeneous 2omega basis waves."""

    ext: Wave
    ord: Wave
    branch_indices: ShaarpSI2OmegaBranchIndices


@dataclass(frozen=True)
class ShaarpSILinearOmegaResult:
    """Self-contained active SHAARP.si omega linear-interface result."""

    incident: Wave
    reflected_p: Wave
    reflected_s: Wave
    transmitted_ext: Wave
    transmitted_ord: Wave
    coefficients: np.ndarray
    roots: ShaarpSIActiveAngleRoots
    branch_indices: ShaarpSIOmegaBranchIndices


@dataclass(frozen=True)
class ShaarpSIReflectedSHGResult:
    """Active SHAARP.si reflected SHG compatibility result."""

    linear_omega: ShaarpSILinearOmegaResult
    homogeneous_2omega: ShaarpSIHomogeneous2OmegaResult
    sources: SingleInterfaceSources
    inhomogeneous: SingleInterfaceInhomogeneousFields
    reflected_p: Wave
    reflected_s: Wave
    transmitted_ext_2omega: Wave
    transmitted_ord_2omega: Wave
    coefficients: np.ndarray
    boundary_residual: np.ndarray

    @property
    def reflected_total(self) -> Wave:
        return Wave(
            k=self.reflected_p.k,
            electric=self.reflected_p.electric + self.reflected_s.electric,
            omega=self.reflected_p.omega,
        )


def solve_shaarp_si_active_angle_roots(
    *,
    epsilon_omega_crystal: np.ndarray,
    epsilon_2omega_crystal: np.ndarray,
    orientation_matrix: np.ndarray,
    theta_in_rad: float,
) -> ShaarpSIActiveAngleRoots:
    """Solve the active SHAARP.si V/W angle-root equations.

    This ports the numerical path in SHAARP.si that solves ``fo``, ``fe``,
    ``fo2w``, and ``fe2w`` from invariants of
    ``Transpose[Cstandard].L.Cstandard``. It is intentionally separate from
    the general anisotropic Snell solver because this path carries SHAARP.si's
    ordinary/extraordinary branch-factor convention.
    """

    omega_context = _angle_context(epsilon_omega_crystal, orientation_matrix)
    two_omega_context = _angle_context(epsilon_2omega_crystal, orientation_matrix)
    # Isotropic gate (third degenerate flavor): at exact isotropy the V/W root landscape is
    # degenerate (V^2 - W ~ 0 everywhere) and the iterative solve can stall at some angles --
    # but the root is then EXACTLY Snell's angle for both branches (verified: the iterative
    # solve, where it does converge at isotropy, returns arcsin(sin th / n) to 1e-12). Gated at
    # 1e-9 relative anisotropy, far below any validated birefringent config.
    n_omega_iso = _isotropic_index_or_none(epsilon_omega_crystal, orientation_matrix)
    n_2omega_iso = _isotropic_index_or_none(epsilon_2omega_crystal, orientation_matrix)
    if n_omega_iso is not None:
        theta_ord = theta_ext = complex(np.emath.arcsin(np.sin(theta_in_rad) / n_omega_iso))
    else:
        theta_ord = _solve_angle_root(omega_context, theta_in_rad, branch_sign=1)
        theta_ext = _solve_angle_root(omega_context, theta_in_rad, branch_sign=-1)
    if n_2omega_iso is not None:
        theta_ord_2omega = theta_ext_2omega = complex(
            np.emath.arcsin(np.sin(theta_in_rad) / n_2omega_iso)
        )
    else:
        theta_ord_2omega = _solve_angle_root(two_omega_context, theta_in_rad, branch_sign=1)
        theta_ext_2omega = _solve_angle_root(two_omega_context, theta_in_rad, branch_sign=-1)

    omega_labels = _indices_and_factor(omega_context, theta_ext, theta_ord)
    two_omega_labels = _indices_and_factor(two_omega_context, theta_ext_2omega, theta_ord_2omega)
    return ShaarpSIActiveAngleRoots(
        theta_ord=_real_if_close(theta_ord),
        theta_ext=_real_if_close(theta_ext),
        theta_ord_2omega=_real_if_close(theta_ord_2omega),
        theta_ext_2omega=_real_if_close(theta_ext_2omega),
        n_ord_label=omega_labels["ordinary_index"],
        n_ext_label=omega_labels["extraordinary_index"],
        n_ord_2omega_label=two_omega_labels["ordinary_index"],
        n_ext_2omega_label=two_omega_labels["extraordinary_index"],
        ext_factor=complex(omega_labels["ext_factor"]),
        ext_2omega_factor=complex(two_omega_labels["ext_factor"]),
    )


def solve_shaarp_si_reflected_shg(
    *,
    epsilon_omega_crystal: np.ndarray,
    epsilon_2omega_crystal: np.ndarray,
    orientation_matrix: np.ndarray,
    d_voigt_lab: np.ndarray,
    theta_in_rad: float,
    incident_polarization: str,
    mu: float = 1.0,
    eps0: float = 1.0,
    normal_incidence_2omega_branch_policy: str = "python_exact_zero",
) -> ShaarpSIReflectedSHGResult:
    """Solve the active SHAARP.si reflected SHG path for one interface."""

    orientation = np.asarray(orientation_matrix, dtype=complex)
    epsilon_2omega_lab = orientation @ np.asarray(epsilon_2omega_crystal, dtype=complex) @ orientation.T
    linear = solve_shaarp_si_linear_omega(
        epsilon_omega_crystal=epsilon_omega_crystal,
        epsilon_2omega_crystal=epsilon_2omega_crystal,
        orientation_matrix=orientation,
        theta_in_rad=theta_in_rad,
        incident_polarization=incident_polarization,
    )
    roots_for_2omega = _apply_normal_incidence_2omega_branch_policy(
        linear.roots,
        theta_in_rad=theta_in_rad,
        policy=normal_incidence_2omega_branch_policy,
    )
    homogeneous_2omega = solve_shaarp_si_homogeneous_2omega(
        epsilon_2omega_crystal=epsilon_2omega_crystal,
        orientation_matrix=orientation,
        roots=roots_for_2omega,
    )
    sources = single_interface_sources(
        d_voigt_lab,
        linear.transmitted_ext,
        linear.transmitted_ord,
    )
    inhomogeneous = solve_single_interface_inhomogeneous_fields(
        epsilon_2omega_lab,
        sources,
        mu=mu,
        eps0=eps0,
    )
    inhomogeneous_waves = [
        Wave(k=field.k, electric=field.electric, omega=field.omega)
        for field in (inhomogeneous.ee, inhomogeneous.oo, inhomogeneous.eo)
    ]
    reflected, transmitted, coefficients = solve_boundary_continuity(
        top_known=[],
        bottom_known=inhomogeneous_waves,
        top_unknown_basis=_si_reflected_2omega_basis(theta_in_rad),
        bottom_unknown_basis=[homogeneous_2omega.ext, homogeneous_2omega.ord],
        mu=mu,
    )
    residual = boundary_residual(reflected, transmitted + inhomogeneous_waves, mu=mu)
    return ShaarpSIReflectedSHGResult(
        linear_omega=linear,
        homogeneous_2omega=homogeneous_2omega,
        sources=sources,
        inhomogeneous=inhomogeneous,
        reflected_p=reflected[0],
        reflected_s=reflected[1],
        transmitted_ext_2omega=transmitted[0],
        transmitted_ord_2omega=transmitted[1],
        coefficients=coefficients,
        boundary_residual=residual,
    )


def _apply_normal_incidence_2omega_branch_policy(
    roots: ShaarpSIActiveAngleRoots,
    *,
    theta_in_rad: float,
    policy: str,
) -> ShaarpSIActiveAngleRoots:
    if policy == "python_exact_zero":
        return roots
    if policy != "shaarp_reference_like":
        raise ValueError(
            "normal_incidence_2omega_branch_policy must be "
            "'python_exact_zero' or 'shaarp_reference_like'."
        )
    if abs(theta_in_rad) > 1e-12:
        return roots
    if abs(roots.theta_ext_2omega) > 1e-12 or abs(roots.theta_ord_2omega) > 1e-12:
        return roots
    index_gap = abs(roots.n_ext_2omega_label - roots.n_ord_2omega_label)
    if index_gap < 1e-2:
        return replace(roots, ext_2omega_factor=-roots.ext_2omega_factor)
    return roots


def solve_shaarp_si_linear_omega(
    *,
    epsilon_omega_crystal: np.ndarray,
    epsilon_2omega_crystal: np.ndarray,
    orientation_matrix: np.ndarray,
    theta_in_rad: float,
    incident_polarization: str,
) -> ShaarpSILinearOmegaResult:
    """Solve SHAARP.si's active omega linear interface path.

    This follows the compatibility behavior observed in the Mathematica SI
    notebook: active V/W angle roots, post-factor labels for eigenvector
    selection, pre-factor labels for the bootstrapped wave-vector constants,
    and the top-side coordinate convention used by the SI Fresnel block.
    """

    if incident_polarization not in {"s", "p"}:
        raise ValueError("incident_polarization must be 's' or 'p'.")
    epsilon_crystal = np.asarray(epsilon_omega_crystal, dtype=complex)
    orientation = np.asarray(orientation_matrix, dtype=complex)
    epsilon_lab = orientation @ epsilon_crystal @ orientation.T
    roots = solve_shaarp_si_active_angle_roots(
        epsilon_omega_crystal=epsilon_crystal,
        epsilon_2omega_crystal=epsilon_2omega_crystal,
        orientation_matrix=orientation,
        theta_in_rad=theta_in_rad,
    )
    branch_indices = select_shaarp_si_omega_branch_indices(
        epsilon_omega_crystal=epsilon_crystal,
        orientation_matrix=orientation,
        roots=roots,
    )
    incident, reflected_basis = _si_top_waves(theta_in_rad, incident_polarization)
    ext_basis = _omega_wave_from_roots(
        epsilon_lab,
        theta=roots.theta_ext,
        refractive_index=roots.n_ext_label,
        eigen_index=branch_indices.ext_index,
    )
    ord_basis = _omega_wave_from_roots(
        epsilon_lab,
        theta=roots.theta_ord,
        refractive_index=roots.n_ord_label,
        eigen_index=branch_indices.ord_index,
    )
    ord_basis = _ensure_degenerate_pair_spans(ext_basis, ord_basis)
    reflected, transmitted, coefficients = solve_fresnel_boundary(
        [incident],
        reflected_basis,
        [ext_basis, ord_basis],
        mu=1.0,
    )
    return ShaarpSILinearOmegaResult(
        incident=incident,
        reflected_p=reflected[0],
        reflected_s=reflected[1],
        transmitted_ext=transmitted[0],
        transmitted_ord=transmitted[1],
        coefficients=coefficients,
        roots=roots,
        branch_indices=branch_indices,
    )


def _ensure_degenerate_pair_spans(ext: Wave, ord_wave: Wave) -> Wave:
    """Isotropic-degenerate guard (second flavor): if ext/ord carry the SAME k and PARALLEL
    polarizations (the eigensystem returned a duplicated eigenvector for the 2D degenerate
    transverse eigenspace), rebuild the ordinary's polarization as the orthogonal transverse
    direction k x E -- exact physics ONLY at isotropy, where every transverse vector at that k
    is an eigenmode. Non-degenerate pairs are returned unchanged (gates stay byte-identical)."""

    e_ext = np.asarray(ext.electric, dtype=complex)
    e_ord = np.asarray(ord_wave.electric, dtype=complex)
    k_ext = np.asarray(ext.k, dtype=complex)
    k_ord = np.asarray(ord_wave.k, dtype=complex)
    n_ext = np.linalg.norm(e_ext)
    n_ord = np.linalg.norm(e_ord)
    if n_ext == 0 or n_ord == 0:
        return ord_wave
    # 1e-6 relative: the active-angle root solve leaves ~1e-7 noise on exactly-degenerate roots,
    # while real birefringence in any validated config splits k by >=1e-2 relative -- 4 orders of
    # margin each way. (1e-8 missed the solver noise floor: |dk| ~ 5.6e-8 at isotropy.)
    k_scale = max(np.linalg.norm(k_ext), 1.0)
    if np.linalg.norm(k_ext - k_ord) > 1e-6 * k_scale:
        return ord_wave
    if np.linalg.norm(np.cross(e_ext / n_ext, e_ord / n_ord)) > 1e-6:
        return ord_wave
    transverse = np.cross(k_ext, e_ext / n_ext)
    norm = np.linalg.norm(transverse)
    if norm == 0:
        return ord_wave
    return replace(ord_wave, electric=(transverse / norm).astype(complex))


def solve_shaarp_si_homogeneous_2omega(
    *,
    epsilon_2omega_crystal: np.ndarray,
    orientation_matrix: np.ndarray,
    roots: ShaarpSIActiveAngleRoots,
) -> ShaarpSIHomogeneous2OmegaResult:
    """Return active SHAARP.si homogeneous 2omega ext/ord basis waves."""

    epsilon_crystal = np.asarray(epsilon_2omega_crystal, dtype=complex)
    orientation = np.asarray(orientation_matrix, dtype=complex)
    epsilon_lab = orientation @ epsilon_crystal @ orientation.T
    branch_indices = select_shaarp_si_2omega_branch_indices(
        epsilon_2omega_crystal=epsilon_crystal,
        orientation_matrix=orientation,
        roots=roots,
    )
    ext = _omega_wave_from_roots(
        epsilon_lab,
        theta=roots.theta_ext_2omega,
        refractive_index=roots.n_ext_2omega_label,
        eigen_index=branch_indices.ext_index,
        omega=2.0,
    )
    ord_wave = _omega_wave_from_roots(
        epsilon_lab,
        theta=roots.theta_ord_2omega,
        refractive_index=roots.n_ord_2omega_label,
        eigen_index=branch_indices.ord_index,
        omega=2.0,
    )
    return ShaarpSIHomogeneous2OmegaResult(
        ext=ext,
        ord=_ensure_degenerate_pair_spans(ext, ord_wave),
        branch_indices=branch_indices,
    )


def select_shaarp_si_omega_branch_indices(
    *,
    epsilon_omega_crystal: np.ndarray,
    orientation_matrix: np.ndarray,
    roots: ShaarpSIActiveAngleRoots,
    match_atol: float = 5e-6,
) -> ShaarpSIOmegaBranchIndices:
    """Select active SHAARP.si omega eigenvector indices after branch-factor flip."""

    epsilon_crystal = np.asarray(epsilon_omega_crystal, dtype=complex)
    orientation = np.asarray(orientation_matrix, dtype=complex)
    context = _angle_context(epsilon_crystal, orientation)
    epsilon_lab = orientation @ epsilon_crystal @ orientation.T
    ext_label = _post_factor_index(context, roots.theta_ext, branch_sign=1, ext_factor=roots.ext_factor)
    ord_label = _post_factor_index(context, roots.theta_ord, branch_sign=-1, ext_factor=roots.ext_factor)
    ext_values, _ = shaarp_ordered_generalized_eigensystem(
        np.array([np.sin(roots.theta_ext), 0.0, -np.cos(roots.theta_ext)], dtype=complex),
        epsilon_lab,
    )
    ord_values, _ = shaarp_ordered_generalized_eigensystem(
        np.array([np.sin(roots.theta_ord), 0.0, -np.cos(roots.theta_ord)], dtype=complex),
        epsilon_lab,
    )
    ext_index = _match_refractive_index_label(ext_values, ext_label, match_atol=match_atol)
    ord_index = _match_refractive_index_label(ord_values, ord_label, match_atol=match_atol)
    if ord_index == ext_index:
        # Degenerate (isotropic) crystal: both post-factor labels match the SAME eigenvalue, so
        # first-match selection returns one eigenvector twice -> two identical basis waves -> a
        # singular boundary matrix. The degenerate transverse eigenspace is 2D; pick the OTHER
        # eigenvector whose eigenvalue also matches the label, so the basis spans the space.
        # Collision-guarded: non-degenerate selections are untouched (gates stay byte-identical).
        ord_index = _match_refractive_index_label(
            ord_values, ord_label, match_atol=match_atol, exclude_index=ext_index
        )
    return ShaarpSIOmegaBranchIndices(
        ext_index=ext_index,
        ord_index=ord_index,
        ext_post_factor_label=ext_label,
        ord_post_factor_label=ord_label,
    )


def select_shaarp_si_2omega_branch_indices(
    *,
    epsilon_2omega_crystal: np.ndarray,
    orientation_matrix: np.ndarray,
    roots: ShaarpSIActiveAngleRoots,
    match_atol: float = 5e-6,
) -> ShaarpSI2OmegaBranchIndices:
    """Select active SHAARP.si 2omega eigenvector indices after branch-factor flip."""

    epsilon_crystal = np.asarray(epsilon_2omega_crystal, dtype=complex)
    orientation = np.asarray(orientation_matrix, dtype=complex)
    context = _angle_context(epsilon_crystal, orientation)
    epsilon_lab = orientation @ epsilon_crystal @ orientation.T
    ext_label = _post_factor_index(context, roots.theta_ext_2omega, branch_sign=1, ext_factor=roots.ext_2omega_factor)
    ord_label = _post_factor_index(context, roots.theta_ord_2omega, branch_sign=-1, ext_factor=roots.ext_2omega_factor)
    ext_values, _ = shaarp_ordered_generalized_eigensystem(
        np.array([np.sin(roots.theta_ext_2omega), 0.0, -np.cos(roots.theta_ext_2omega)], dtype=complex),
        epsilon_lab,
    )
    ord_values, _ = shaarp_ordered_generalized_eigensystem(
        np.array([np.sin(roots.theta_ord_2omega), 0.0, -np.cos(roots.theta_ord_2omega)], dtype=complex),
        epsilon_lab,
    )
    ext_index = _match_refractive_index_label(ext_values, ext_label, match_atol=match_atol)
    ord_index = _match_refractive_index_label(ord_values, ord_label, match_atol=match_atol)
    if ord_index == ext_index:
        # same degenerate-eigenspace guard as the omega selector (isotropic crystals)
        ord_index = _match_refractive_index_label(
            ord_values, ord_label, match_atol=match_atol, exclude_index=ext_index
        )
    return ShaarpSI2OmegaBranchIndices(
        ext_index=ext_index,
        ord_index=ord_index,
        ext_post_factor_label=ext_label,
        ord_post_factor_label=ord_label,
    )


def _omega_wave_from_roots(
    epsilon_lab: np.ndarray,
    *,
    theta: complex,
    refractive_index: complex,
    eigen_index: int,
    omega: float = 1.0,
) -> Wave:
    direction = np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=complex)
    _, vectors = shaarp_ordered_generalized_eigensystem(direction, epsilon_lab)
    return Wave(
        k=omega * refractive_index * direction,
        electric=np.asarray(vectors[eigen_index - 1], dtype=complex),
        omega=omega,
    )


def _si_top_waves(theta_in: float, incident_polarization: str) -> tuple[Wave, list[Wave]]:
    rotation = _rhtilt(theta_in)
    inverse_rotation = np.linalg.inv(rotation)
    if incident_polarization == "p":
        incident_e = rotation @ np.array([1.0, 0.0, 0.0], dtype=complex)
    elif incident_polarization == "s":
        incident_e = rotation @ np.array([0.0, 1.0, 0.0], dtype=complex)
    else:
        raise ValueError(f"Unsupported incident polarization: {incident_polarization!r}")
    incident = Wave(
        k=rotation @ np.array([0.0, 0.0, -1.0], dtype=complex),
        electric=incident_e,
        omega=1.0,
    )
    reflected_p = Wave(
        k=inverse_rotation @ np.array([0.0, 0.0, 1.0], dtype=complex),
        electric=inverse_rotation @ np.array([-1.0, 0.0, 0.0], dtype=complex),
        omega=1.0,
    )
    reflected_s = Wave(
        k=inverse_rotation @ np.array([0.0, 0.0, 1.0], dtype=complex),
        electric=inverse_rotation @ np.array([0.0, 1.0, 0.0], dtype=complex),
        omega=1.0,
    )
    return incident, [reflected_p, reflected_s]


def _si_reflected_2omega_basis(theta_in: float) -> list[Wave]:
    inverse_rotation = np.linalg.inv(_rhtilt(theta_in))
    reflected_k = 2.0 * (inverse_rotation @ np.array([0.0, 0.0, 1.0], dtype=complex))
    reflected_p = Wave(
        k=reflected_k,
        electric=inverse_rotation @ np.array([-1.0, 0.0, 0.0], dtype=complex),
        omega=2.0,
    )
    reflected_s = Wave(
        k=reflected_k,
        electric=inverse_rotation @ np.array([0.0, 1.0, 0.0], dtype=complex),
        omega=2.0,
    )
    return [reflected_p, reflected_s]


def _rhtilt(theta_in: float) -> np.ndarray:
    c = np.cos(theta_in)
    s = np.sin(theta_in)
    return np.array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=complex,
    )


def _isotropic_index_or_none(
    epsilon_crystal: np.ndarray, orientation_matrix: np.ndarray, *, rtol: float = 1e-9
) -> complex | None:
    """Return the scalar refractive index if the LAB-frame epsilon is isotropic within rtol
    (relative), else None. Used to gate the exact-Snell degenerate shortcut."""

    epsilon = np.asarray(epsilon_crystal, dtype=complex)
    orientation = np.asarray(orientation_matrix, dtype=complex)
    epsilon_lab = orientation @ epsilon @ orientation.T
    n_squared = complex(np.trace(epsilon_lab) / 3.0)
    if np.linalg.norm(epsilon_lab - n_squared * np.eye(3)) > rtol * max(abs(n_squared), 1.0):
        return None
    return complex(np.sqrt(n_squared))


def _angle_context(epsilon_crystal: np.ndarray, orientation_matrix: np.ndarray) -> np.ndarray:
    epsilon = np.asarray(epsilon_crystal, dtype=complex)
    orientation = np.asarray(orientation_matrix, dtype=complex)
    epsilon_sqrt = np.asarray(sqrtm(epsilon), dtype=complex)
    principal_rotation = _principal_axis_rotation_like_shaarp_si(epsilon, epsilon_sqrt)
    refractive_index_principal = _chop_small(principal_rotation @ epsilon_sqrt @ principal_rotation.T)
    q_matrix = principal_rotation @ np.linalg.inv(orientation)
    return np.linalg.inv(refractive_index_principal @ q_matrix)


def _principal_axis_rotation_like_shaarp_si(epsilon_crystal: np.ndarray, epsilon_sqrt: np.ndarray) -> np.ndarray:
    inv = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=complex)
    _, eigenvectors = np.linalg.eig(np.real(epsilon_sqrt))
    rotation = inv @ eigenvectors.T
    reconstructed = rotation @ epsilon_crystal @ rotation.T
    if np.allclose(_six_digit(reconstructed), _six_digit(epsilon_crystal), atol=5e-6, rtol=0.0):
        return np.eye(3, dtype=complex)
    return rotation


def _solve_angle_root(context: np.ndarray, theta_in: float, *, branch_sign: int) -> complex:
    def residual(parts: np.ndarray) -> np.ndarray:
        theta = parts[0] + 1j * parts[1]
        value = _angle_root_residual(context, theta_in, theta, branch_sign=branch_sign)
        return np.array([np.real(value), np.imag(value)], dtype=float)

    messages: list[str] = []
    for guess in _angle_initial_guesses(theta_in):
        solution = root(residual, np.array([np.real(guess), np.imag(guess)], dtype=float), method="hybr", tol=1e-12)
        if not solution.success:
            messages.append(str(solution.message))
            continue
        theta = complex(solution.x[0], solution.x[1])
        check = residual(solution.x)
        if np.linalg.norm(check) <= 1e-10:
            return theta
        messages.append(f"residual too large: {check}")
    detail = "; ".join(dict.fromkeys(messages))
    raise RuntimeError(f"SHAARP.si angle root solve failed for branch_sign={branch_sign}: {detail}")


def _angle_initial_guesses(theta_in: float) -> list[complex]:
    guesses = [
        complex(pi / 4, 0.0),
        complex(theta_in if abs(theta_in) > 1e-12 else 0.0, 0.0),
        complex(0.0, 0.0),
        complex(0.1, 0.0),
        complex(0.3, 0.0),
        complex(0.6, 0.0),
        complex(pi / 4, 0.05),
        complex(theta_in if abs(theta_in) > 1e-12 else 0.0, 0.05),
    ]
    out: list[complex] = []
    seen: set[tuple[float, float]] = set()
    for guess in guesses:
        key = (round(float(np.real(guess)), 14), round(float(np.imag(guess)), 14))
        if key not in seen:
            seen.add(key)
            out.append(guess)
    return out


def _angle_root_residual(
    context: np.ndarray,
    theta_in: float,
    theta_out: complex,
    *,
    branch_sign: int,
) -> complex:
    v_value, w_value = _vw(context, theta_out)
    branch_index = np.sqrt((v_value + branch_sign * np.sqrt(v_value**2 - w_value)) / w_value)
    return np.sin(theta_in) - branch_index * np.sin(theta_out)


def _indices_and_factor(context: np.ndarray, theta_ext: complex, theta_ord: complex) -> dict[str, complex]:
    ext_factor = 1
    v_ext, w_ext = _vw(context, theta_ext)
    v_ord, w_ord = _vw(context, theta_ord)
    extraordinary_index = np.sqrt((v_ext + ext_factor * np.sqrt(v_ext**2 - w_ext)) / w_ext)
    ordinary_index = np.sqrt((v_ord - ext_factor * np.sqrt(v_ord**2 - w_ord)) / w_ord)
    if abs(extraordinary_index) > abs(ordinary_index) and abs(theta_ext) > abs(theta_ord):
        ext_factor *= -1
    if abs(extraordinary_index) < abs(ordinary_index) and abs(theta_ext) < abs(theta_ord):
        ext_factor *= -1
    return {
        "extraordinary_index": extraordinary_index,
        "ordinary_index": ordinary_index,
        "ext_factor": complex(ext_factor),
    }


def _post_factor_index(context: np.ndarray, theta: complex, *, branch_sign: int, ext_factor: complex) -> complex:
    v_value, w_value = _vw(context, theta)
    return np.sqrt((v_value + branch_sign * ext_factor * np.sqrt(v_value**2 - w_value)) / w_value)


def _match_refractive_index_label(
    values: np.ndarray, label: complex, *, match_atol: float, exclude_index: int | None = None
) -> int:
    candidates: list[tuple[float, int]] = []
    for index, value in enumerate(values, start=1):
        if abs(value) <= 1e-12:
            continue
        if exclude_index is not None and index == exclude_index:
            continue
        refractive_index = np.sqrt(1 / value)
        candidates.append((abs(refractive_index - label), index))
    if not candidates:
        raise RuntimeError("No nonzero generalized eigenvalues available for SHAARP.si branch index selection.")
    best_error, best_index = min(candidates, key=lambda item: item[0])
    if best_error > match_atol:
        raise RuntimeError(
            "No SHAARP.si branch index matched the post-factor refractive-index label "
            f"within {match_atol}: best_error={best_error}, label={label!r}"
        )
    return best_index


def _vw(context: np.ndarray, theta: complex) -> tuple[complex, complex]:
    matrix = context.T @ _l_matrix(theta) @ context
    v_value = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2
    w_value = (
        matrix[0, 0] * matrix[1, 1]
        + matrix[0, 0] * matrix[2, 2]
        + matrix[1, 1] * matrix[2, 2]
        - matrix[0, 2] * matrix[2, 0]
        - matrix[0, 1] * matrix[1, 0]
        - matrix[1, 2] * matrix[2, 1]
    )
    return v_value, w_value


def _l_matrix(theta: complex) -> np.ndarray:
    ux = np.sin(theta)
    uy = 0.0
    uz = -np.cos(theta)
    return np.array(
        [
            [uy**2 + uz**2, -ux * uy, -ux * uz],
            [-ux * uy, ux**2 + uz**2, -uy * uz],
            [-uz * ux, -uz * uy, ux**2 + uy**2],
        ],
        dtype=complex,
    )


def _six_digit(value: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(value, dtype=complex), 6)


def _chop_small(value: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    result = np.array(value, dtype=complex, copy=True)
    result[np.abs(result) < tol] = 0.0
    return result


def _real_if_close(value: complex, *, tol: float = 1e-12) -> complex | float:
    if abs(np.imag(value)) < tol:
        return float(np.real(value))
    return value
