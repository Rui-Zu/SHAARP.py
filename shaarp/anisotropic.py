from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi

import numpy as np
from scipy.linalg import eig
from scipy.linalg import sqrtm
from scipy.optimize import brentq
from scipy.optimize import root


@dataclass(frozen=True)
class AnisotropicMode:
    """One homogeneous wave mode for a chosen propagation direction."""

    label: str
    refractive_index: complex
    theta_rad: complex
    direction: np.ndarray
    electric_direction: np.ndarray
    eta: complex


@dataclass(frozen=True)
class SnellModes:
    """Fast and slow modes returned by the SHAARP-style Snell solve."""

    fast: AnisotropicMode
    slow: AnisotropicMode


@dataclass(frozen=True)
class OrdinaryExtraordinaryModes:
    """Uniaxial ordinary/extraordinary identity for a pair of Snell modes."""

    ordinary: AnisotropicMode
    extraordinary: AnisotropicMode
    ordinary_branch: str
    extraordinary_branch: str
    optic_axis_lab: np.ndarray
    ordinary_alignment: float
    extraordinary_alignment: float


@dataclass(frozen=True)
class BranchTrackedModes:
    """Continuity-tracked identities for two anisotropic eigenmodes."""

    branch_1: AnisotropicMode
    branch_2: AnisotropicMode
    branch_1_source_label: str
    branch_2_source_label: str
    branch_1_alignment: float
    branch_2_alignment: float
    ambiguous: bool


@dataclass(frozen=True)
class ShaarpEigenBasis:
    """SHAARP.ml standard-eigenproblem basis metadata for solveSnell."""

    epsilon_crystal: np.ndarray
    orientation_rows: np.ndarray
    point_group: str
    epsilon_lab: np.ndarray
    epsilon_principal_frame: np.ndarray
    qp: np.ndarray
    c_transform: np.ndarray


def transverse_projector(direction: np.ndarray) -> np.ndarray:
    """Return L = I - u u^T, matching the projector built in solveSnell."""

    u = np.asarray(direction, dtype=complex)
    norm = np.sqrt(np.dot(u, u))
    if norm == 0:
        raise ValueError("Propagation direction must be non-zero.")
    u = u / norm
    return np.eye(3, dtype=complex) - np.outer(u, u)


def make_shaarp_eigen_basis(
    epsilon_crystal: np.ndarray,
    orientation_rows: np.ndarray,
    point_group: str,
) -> ShaarpEigenBasis:
    """Build the transformed basis used by SHAARP.ml when `eigenStandard=True`."""

    eps_c = np.asarray(epsilon_crystal, dtype=complex)
    qc = np.asarray(orientation_rows, dtype=float)
    if eps_c.shape != (3, 3):
        raise ValueError("epsilon_crystal must be a 3x3 dielectric tensor.")
    if qc.shape != (3, 3):
        raise ValueError("orientation_rows must be a 3x3 matrix.")

    eps_lab = qc.T @ eps_c @ qc
    qp = _qcp_to_qp(point_group, eps_c, qc)
    eps_p = qp.T @ eps_lab @ qp
    r_transform = sqrtm(eps_p) @ qp.T
    c_transform = np.linalg.inv(r_transform)
    return ShaarpEigenBasis(
        epsilon_crystal=eps_c,
        orientation_rows=qc,
        point_group=str(point_group),
        epsilon_lab=eps_lab,
        epsilon_principal_frame=eps_p,
        qp=qp,
        c_transform=c_transform,
    )


def modes_for_direction(
    epsilon_lab: np.ndarray,
    direction: np.ndarray,
    *,
    shaarp_basis: ShaarpEigenBasis | None = None,
) -> tuple[AnisotropicMode, AnisotropicMode]:
    """Solve L E = eta epsilon E for the two transverse optical modes.

    The Mathematica `solveSnell` function computes `eta = 1/n^2` from the
    transverse projector and dielectric tensor, then uses the corresponding
    eigenvectors as electric-field directions. This function ports that
    numerical step for a fixed propagation direction.
    """

    eps = np.asarray(epsilon_lab, dtype=complex)
    if eps.shape != (3, 3):
        raise ValueError("epsilon_lab must be a 3x3 dielectric tensor.")
    if np.max(np.abs(eps - eps.T)) > 1e-8:
        raise ValueError("epsilon_lab must be symmetric for this solver stage.")

    if shaarp_basis is not None:
        eps = np.asarray(shaarp_basis.epsilon_lab, dtype=complex)

    u = _normalize_direction(direction)
    projector = transverse_projector(u)
    if shaarp_basis is None:
        eigvals, eigvecs = np.linalg.eig(np.linalg.solve(eps, projector))
        raw_vectors = eigvecs.T
    else:
        matrix = shaarp_basis.c_transform.T @ projector @ shaarp_basis.c_transform
        eigvals, eigvecs = np.linalg.eig(matrix)
        raw_vectors = np.array([vec @ shaarp_basis.c_transform.T for vec in eigvecs.T], dtype=complex)

    candidates: list[tuple[complex, complex, np.ndarray]] = []
    for eta, vec in zip(eigvals, raw_vectors):
        if abs(eta) < 1e-10:
            continue
        n = np.sqrt(1 / eta)
        if np.real(n) < 0:
            n = -n
            vec = -vec
        if shaarp_basis is None:
            e = _normalize_phase(vec)
        else:
            e = _normalize_without_phase(vec, zero_tol=1e-6)
        candidates.append((n, eta, e))

    if len(candidates) < 2:
        raise RuntimeError("Could not identify two transverse optical modes.")

    candidates.sort(key=lambda item: (np.real(item[0]), np.imag(item[0])))
    if np.max(np.abs(np.imag(u))) < 1e-12:
        theta = float(np.arctan2(np.real(u[0]), np.real(u[2])))
    else:
        theta = np.arctan(u[0] / u[2])
    modes = []
    for label, (n, eta, e) in zip(("fast", "slow"), candidates[:2]):
        modes.append(
            AnisotropicMode(
                label=label,
                refractive_index=n,
                theta_rad=theta,
                direction=np.asarray(u, dtype=complex),
                electric_direction=e,
                eta=eta,
            )
        )
    return modes[0], modes[1]


def shaarp_ordered_generalized_eigensystem(
    direction: np.ndarray,
    epsilon_lab: np.ndarray,
    *,
    zero_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return SHAARP.si-style ordered eigenpairs for ``L E = eta eps E``.

    Mathematica's active SHAARP.si path calls ``Eigensystem[{LNum, elabNum}]``
    and then selects modes by the six-digit ``Sqrt[1/eta]`` label. SciPy's
    generalized eigensolver returns the same finite eigenspaces but not the
    same order: the near-zero longitudinal eigenvalue can appear between or
    before the two transverse eigenvalues. This helper standardizes the order
    observed in the Mathematica branch metadata: finite eigenvalues sorted by
    decreasing real part, then the zero/longitudinal value.
    """

    eps = np.asarray(epsilon_lab, dtype=complex)
    if eps.shape != (3, 3):
        raise ValueError("epsilon_lab must be a 3x3 dielectric tensor.")
    projector = transverse_projector(np.asarray(direction, dtype=complex))
    values, vectors_by_column = eig(projector, eps)
    vectors = vectors_by_column.T
    finite_indices = [index for index, value in enumerate(values) if abs(value) > zero_tol]
    zero_indices = [index for index, value in enumerate(values) if abs(value) <= zero_tol]
    finite_indices.sort(key=lambda index: (np.real(values[index]), np.imag(values[index])), reverse=True)
    zero_indices.sort(key=lambda index: np.linalg.norm(vectors[index]))
    ordered_indices = finite_indices + zero_indices
    ordered_values = np.asarray(values[ordered_indices], dtype=complex)
    ordered_values[np.abs(ordered_values) <= zero_tol] = 0.0
    ordered_vectors = np.asarray([_normalize_without_phase(vectors[index], zero_tol=zero_tol) for index in ordered_indices])
    return ordered_values, ordered_vectors


def solve_snell_modes(
    epsilon_lab: np.ndarray,
    incident_index: complex,
    incident_theta_rad: float,
    *,
    reflected: bool = False,
    shaarp_basis: ShaarpEigenBasis | None = None,
) -> SnellModes:
    """Solve SHAARP-style Snell modes for a nonmagnetic anisotropic medium.

    This ports the first numerical part of Mathematica `solveSnell`: for each
    trial propagation angle, compute fast/slow indices from `L E = eta eps E`,
    then solve `n(theta) sin(theta) = n0 sin(theta_i)`.

    Current verified scope: transmitted and reflected modes. Real/lossless
    media use a bracketed real root; complex absorbing media use a complex root
    solve with the same initial-guess pattern as Mathematica `solveSnell`.
    """

    eps = np.asarray(shaarp_basis.epsilon_lab if shaarp_basis is not None else epsilon_lab, dtype=complex)
    complex_media = np.max(np.abs(np.imag(eps))) > 1e-10 or abs(np.imag(incident_index)) > 1e-10
    n0 = complex(incident_index)
    theta_i = float(incident_theta_rad)
    target = n0 * np.sin(theta_i)
    if abs(target) < 1e-14:
        theta_fast = theta_slow = pi if reflected else 0.0
    elif complex_media:
        theta_guess = (pi - theta_i + 0.1j) if reflected else (theta_i + 0.1j)
        theta_fast = _solve_complex_theta_for_mode(
            eps,
            target,
            mode_index=0,
            guess=theta_guess,
            reflected=reflected,
            shaarp_basis=shaarp_basis,
        )
        theta_slow = _solve_complex_theta_for_mode(
            eps,
            target,
            mode_index=1,
            guess=theta_guess,
            reflected=reflected,
            shaarp_basis=shaarp_basis,
        )
    else:
        theta_fast = _solve_theta_for_mode(
            eps,
            float(np.real(target)),
            mode_index=0,
            reflected=reflected,
            shaarp_basis=shaarp_basis,
        )
        theta_slow = _solve_theta_for_mode(
            eps,
            float(np.real(target)),
            mode_index=1,
            reflected=reflected,
            shaarp_basis=shaarp_basis,
        )

    # Degenerate/isotropic guard (verified root cause of the ml1
    # multilayer discrepancy). `theta_fast` and `theta_slow` are solved
    # INDEPENDENTLY; for an isotropic/quasi-isotropic layer they coincide but the
    # two separate `_mode_at_theta` calls each pick an ARBITRARY basis of the
    # degenerate transverse plane, which can collapse fast and slow onto the SAME
    # electric direction -> two identical multilayer boundary-matrix columns ->
    # rank-deficient fundamental system (verified rank 11/12, cond 5.7e17 at
    # theta=60). Build BOTH modes from a SINGLE call at the shared angle so the
    # pair is internally consistent, and if that pair is still (anti)parallel,
    # assign the fixed, angle-stable s/p pair in the x-z plane of incidence
    # (s = y-hat; p = u x s) -- mirroring Mathematica solveSnell's "Iso or quasi
    # Iso" branch. Strict no-op for genuinely distinct anisotropic modes.
    if abs(complex(theta_fast) - complex(theta_slow)) < 1e-9:
        pair = modes_for_direction(
            eps,
            np.array([np.sin(theta_fast), 0.0, np.cos(theta_fast)], dtype=complex),
            shaarp_basis=shaarp_basis,
        )
        fast = replace(pair[0], theta_rad=theta_fast)
        slow = replace(pair[1], theta_rad=theta_fast)
        e_f = np.asarray(fast.electric_direction, dtype=complex)
        e_s = np.asarray(slow.electric_direction, dtype=complex)
        overlap = abs(
            np.vdot(e_f / (np.linalg.norm(e_f) + 1e-30), e_s / (np.linalg.norm(e_s) + 1e-30))
        )
        if overlap > 0.999:
            u = np.array([np.sin(theta_fast), 0.0, np.cos(theta_fast)], dtype=complex)
            s_hat = np.array([0.0, 1.0, 0.0], dtype=complex)
            p_vec = np.cross(u, s_hat)
            p_norm = np.linalg.norm(p_vec)
            if p_norm > 1e-9:
                fast = replace(fast, electric_direction=_normalize_phase(s_hat))
                slow = replace(slow, electric_direction=_normalize_phase(p_vec / p_norm))
        return SnellModes(fast=fast, slow=slow)

    fast = _mode_at_theta(eps, theta_fast, 0, shaarp_basis=shaarp_basis)
    slow = _mode_at_theta(eps, theta_slow, 1, shaarp_basis=shaarp_basis)
    return SnellModes(fast=fast, slow=slow)


def identify_uniaxial_modes(
    epsilon_lab: np.ndarray,
    modes: SnellModes,
    *,
    optic_axis_lab: np.ndarray | None = None,
    degeneracy_tol: float = 1e-6,
    min_alignment: float = 0.9,
) -> OrdinaryExtraordinaryModes:
    """Classify a uniaxial pair as ordinary and extraordinary.

    SHAARP's nonlinear sources use explicit ordinary/extraordinary identities
    (`P2o`, `P2eo`, `Peoo`). Sorting modes by refractive index is not enough:
    in positive uniaxial media the ordinary wave may be the fast branch, while
    in negative uniaxial media it may be the slow branch.

    This helper is deliberately scoped to uniaxial media. If the optic axis
    cannot be supplied or inferred from a repeated dielectric eigenvalue, or
    if the propagation direction is too close to the optic axis, it raises
    `ValueError` instead of guessing.
    """

    eps = np.asarray(epsilon_lab, dtype=complex)
    if eps.shape != (3, 3):
        raise ValueError("epsilon_lab must be a 3x3 dielectric tensor.")
    if optic_axis_lab is None:
        optic_axis = _infer_uniaxial_optic_axis(eps, degeneracy_tol=degeneracy_tol)
    else:
        optic_axis = _normalize_vector(np.asarray(optic_axis_lab, dtype=complex))

    fast_score = _ordinary_alignment(modes.fast, optic_axis)
    slow_score = _ordinary_alignment(modes.slow, optic_axis)
    if max(fast_score, slow_score) < min_alignment:
        raise ValueError("Could not confidently identify the ordinary wave.")

    if fast_score >= slow_score:
        ordinary = modes.fast
        extraordinary = modes.slow
        ordinary_branch = "fast"
        extraordinary_branch = "slow"
        ordinary_alignment = fast_score
        extraordinary_alignment = float(np.sqrt(max(0.0, 1.0 - slow_score**2)))
    else:
        ordinary = modes.slow
        extraordinary = modes.fast
        ordinary_branch = "slow"
        extraordinary_branch = "fast"
        ordinary_alignment = slow_score
        extraordinary_alignment = float(np.sqrt(max(0.0, 1.0 - fast_score**2)))

    return OrdinaryExtraordinaryModes(
        ordinary=ordinary,
        extraordinary=extraordinary,
        ordinary_branch=ordinary_branch,
        extraordinary_branch=extraordinary_branch,
        optic_axis_lab=optic_axis,
        ordinary_alignment=float(ordinary_alignment),
        extraordinary_alignment=float(extraordinary_alignment),
    )


def track_mode_branches(
    mode_pairs: list[SnellModes] | tuple[SnellModes, ...],
    *,
    ambiguity_tol: float = 1e-6,
) -> list[BranchTrackedModes]:
    """Track two anisotropic branches by polarization continuity.

    This is the safer bookkeeping layer for biaxial media. Biaxial crystals do
    not generally have globally meaningful ordinary/extraordinary identities,
    so later calculations should not force `o/e` labels onto them. For angle or
    orientation sweeps, this helper keeps branch 1 and branch 2 continuous by
    choosing the assignment with the larger electric-vector overlap relative to
    the previous point.
    """

    if len(mode_pairs) == 0:
        return []

    first = mode_pairs[0]
    tracked = [
        BranchTrackedModes(
            branch_1=first.fast,
            branch_2=first.slow,
            branch_1_source_label=first.fast.label,
            branch_2_source_label=first.slow.label,
            branch_1_alignment=1.0,
            branch_2_alignment=1.0,
            ambiguous=False,
        )
    ]
    prev_1 = first.fast
    prev_2 = first.slow

    for pair in mode_pairs[1:]:
        keep_1 = _mode_overlap(prev_1, pair.fast)
        keep_2 = _mode_overlap(prev_2, pair.slow)
        swap_1 = _mode_overlap(prev_1, pair.slow)
        swap_2 = _mode_overlap(prev_2, pair.fast)
        keep_score = keep_1 + keep_2
        swap_score = swap_1 + swap_2
        ambiguous = abs(keep_score - swap_score) <= ambiguity_tol

        if swap_score > keep_score:
            branch_1 = pair.slow
            branch_2 = pair.fast
            branch_1_source_label = pair.slow.label
            branch_2_source_label = pair.fast.label
            branch_1_alignment = swap_1
            branch_2_alignment = swap_2
        else:
            branch_1 = pair.fast
            branch_2 = pair.slow
            branch_1_source_label = pair.fast.label
            branch_2_source_label = pair.slow.label
            branch_1_alignment = keep_1
            branch_2_alignment = keep_2

        tracked.append(
            BranchTrackedModes(
                branch_1=branch_1,
                branch_2=branch_2,
                branch_1_source_label=branch_1_source_label,
                branch_2_source_label=branch_2_source_label,
                branch_1_alignment=float(branch_1_alignment),
                branch_2_alignment=float(branch_2_alignment),
                ambiguous=bool(ambiguous),
            )
        )
        prev_1 = branch_1
        prev_2 = branch_2

    return tracked


def _mode_at_theta(
    epsilon_lab: np.ndarray,
    theta: complex,
    mode_index: int,
    *,
    shaarp_basis: ShaarpEigenBasis | None = None,
) -> AnisotropicMode:
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=complex)
    mode = modes_for_direction(epsilon_lab, direction, shaarp_basis=shaarp_basis)[mode_index]
    return replace(mode, theta_rad=theta)


def _solve_theta_for_mode(
    epsilon_lab: np.ndarray,
    target: float,
    mode_index: int,
    *,
    reflected: bool = False,
    shaarp_basis: ShaarpEigenBasis | None = None,
) -> float:
    if reflected and target < 0:
        low = -pi + 1e-9
        high = -pi / 2 - 1e-9
    elif reflected:
        low = pi / 2 + 1e-9
        high = pi - 1e-9
    elif target < 0:
        low = -pi / 2 + 1e-9
        high = 0.0
    else:
        low = 0.0
        high = pi / 2 - 1e-9

    def residual(theta: float) -> float:
        mode = _mode_at_theta(epsilon_lab, theta, mode_index, shaarp_basis=shaarp_basis)
        return float(np.real(mode.refractive_index) * np.sin(theta) - target)

    f0 = residual(low)
    f1 = residual(high)
    if abs(f0) < 1e-14:
        return low
    if abs(f1) < 1e-14:
        return high
    if f0 * f1 > 0:
        raise ValueError(
            "No real Snell root found. This may be total internal "
            "reflection or an unsupported absorbing/strongly anisotropic case."
        )
    return float(brentq(residual, low, high, xtol=1e-12, rtol=1e-12, maxiter=100))


def _solve_complex_theta_for_mode(
    epsilon_lab: np.ndarray,
    target: complex,
    mode_index: int,
    guess: complex,
    *,
    reflected: bool,
    shaarp_basis: ShaarpEigenBasis | None = None,
) -> complex:
    def residual(parts: np.ndarray) -> np.ndarray:
        theta = parts[0] + 1j * parts[1]
        # The root finder freely explores trial thetas, including ones with a LARGE imaginary part
        # (e.g. near-grazing theta_i ~ 89 deg in a high-index film). There, sin(theta)/cos(theta) blow
        # up and sin^2+cos^2 -- though identically 1 -- is computed as a sum of two huge opposite-sign
        # numbers, so catastrophic cancellation rounds the direction norm to 0 and _normalize_direction
        # RAISES. Let that (or any non-finite) escape and it kills the whole solve. Instead return a
        # large residual so the solver simply steps back. Converged roots have a well-defined unit
        # direction, so this does NOT change any validated result -- it only tames bad trial iterates.
        try:
            mode = _mode_at_theta(epsilon_lab, theta, mode_index, shaarp_basis=shaarp_basis)
            value = mode.refractive_index * np.sin(theta) - target
            out = np.array([np.real(value), np.imag(value)], dtype=float)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return np.array([1.0e6, 1.0e6])
        return out if np.all(np.isfinite(out)) else np.array([1.0e6, 1.0e6])

    messages = []
    for candidate in _complex_theta_initial_guesses(guess, reflected=reflected):
        sol = root(residual, np.array([np.real(candidate), np.imag(candidate)], dtype=float), method="hybr", tol=1e-12)
        if not sol.success:
            messages.append(str(sol.message))
            continue
        theta = complex(sol.x[0], sol.x[1])
        check = residual(sol.x)
        if np.linalg.norm(check) > 1e-9:
            messages.append(f"residual too large: {check}")
            continue
        mode = _mode_at_theta(epsilon_lab, theta, mode_index, shaarp_basis=shaarp_basis)
        kz = mode.refractive_index * np.cos(theta)
        if reflected and np.real(kz) >= -1e-9:
            messages.append(f"root is not on reflected branch: theta={theta}, kz={kz}")
            continue
        if not reflected and np.real(kz) <= 1e-9:
            messages.append(f"root is not on transmitted branch: theta={theta}, kz={kz}")
            continue
        if abs(np.imag(theta)) < 1e-12:
            theta = float(np.real(theta))
        return theta
    detail = "; ".join(dict.fromkeys(messages))
    raise RuntimeError(f"Complex Snell root solve failed for mode {mode_index}: {detail}")


def _complex_theta_initial_guesses(guess: complex, *, reflected: bool) -> list[complex]:
    """Return deterministic nearby guesses for absorbing anisotropic Snell roots.

    SciPy's hybrid solver can fail from one otherwise reasonable complex
    starting point even when a tiny real-part perturbation converges to the same
    residual-valid root. These guesses keep the search local to the intended
    transmitted/reflected branch and do not change the Snell equation.
    """

    real = float(np.real(guess))
    imag = float(np.imag(guess))
    sign = 1.0 if real >= 0 else -1.0
    candidates = [
        complex(real, imag),
        complex(round(real, 2), imag),
        complex(real, 0.0),
        complex(round(real, 2), 0.0),
        complex(real - sign * 0.05, imag),
        complex(real + sign * 0.05, imag),
        complex(real * 0.75, imag),
        complex(real * 1.25, imag),
        complex(real - sign * 0.05, 0.0),
        complex(real + sign * 0.05, 0.0),
    ]
    if reflected:
        candidates.extend(
            [
                complex(0.60 * pi, imag),
                complex(0.75 * pi, imag),
                complex(0.85 * pi, imag),
                complex(0.95 * pi, imag),
                complex(0.60 * pi, 0.0),
                complex(0.75 * pi, 0.0),
                complex(0.85 * pi, 0.0),
                complex(0.95 * pi, 0.0),
            ]
        )
    else:
        candidates.extend(
            [
                complex(0.05 * pi, imag),
                complex(0.15 * pi, imag),
                complex(0.30 * pi, imag),
                complex(0.45 * pi, imag),
                complex(0.05 * pi, 0.0),
                complex(0.15 * pi, 0.0),
                complex(0.30 * pi, 0.0),
                complex(0.45 * pi, 0.0),
            ]
        )
    out = []
    seen = set()
    for candidate in candidates:
        key = (round(float(np.real(candidate)), 14), round(float(np.imag(candidate)), 14))
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _normalize_phase(vec: np.ndarray, *, zero_tol: float = 1e-12) -> np.ndarray:
    out = np.asarray(vec, dtype=complex)
    norm = np.linalg.norm(out)
    if norm == 0:
        raise ValueError("Eigenvector norm is zero.")
    out = out / norm
    pivot = int(np.argmax(np.abs(out)))
    phase = np.angle(out[pivot])
    out = out * np.exp(-1j * phase)
    out = _chop_complex_components(out, zero_tol)
    return out


def _normalize_without_phase(vec: np.ndarray, *, zero_tol: float) -> np.ndarray:
    out = np.asarray(vec, dtype=complex)
    norm = np.linalg.norm(out)
    if norm == 0:
        raise ValueError("Eigenvector norm is zero.")
    out = out / norm
    return _chop_complex_components(out, zero_tol)


def _qcp_to_qp(point_group: str, epsilon_crystal: np.ndarray, orientation_rows: np.ndarray) -> np.ndarray:
    low_symmetry_groups = {"1", "2", "m", "mm2"}
    if str(point_group).strip() in low_symmetry_groups:
        inv = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        sqrt_real = np.real(sqrtm(epsilon_crystal))
        eigvals, eigvecs = np.linalg.eig(sqrt_real)
        order = np.argsort(np.real(eigvals))[::-1]
        eigenvectors = eigvecs[:, order].T
        crystal_to_principal = inv @ eigenvectors
        if np.allclose(
            crystal_to_principal @ epsilon_crystal @ crystal_to_principal.T,
            epsilon_crystal,
            rtol=0.0,
            atol=5e-6,
        ):
            crystal_to_principal = np.eye(3)
    else:
        crystal_to_principal = np.eye(3)
    return np.asarray(crystal_to_principal @ np.linalg.inv(orientation_rows), dtype=complex)


def _chop_complex_components(values: np.ndarray, tol: float) -> np.ndarray:
    out = np.asarray(values, dtype=complex).copy()
    real = np.real(out)
    imag = np.imag(out)
    real[np.abs(real) < tol] = 0.0
    imag[np.abs(imag) < tol] = 0.0
    return real + 1j * imag


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    u = np.asarray(direction, dtype=complex)
    norm = np.sqrt(np.dot(u, u))
    if abs(norm) == 0:
        raise ValueError("Propagation direction must be non-zero.")
    return u / norm


def _infer_uniaxial_optic_axis(epsilon_lab: np.ndarray, *, degeneracy_tol: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(epsilon_lab)
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    best_pair = min(
        pairs,
        key=lambda item: abs(eigvals[item[0]] - eigvals[item[1]])
        / max(abs(eigvals[item[0]]), abs(eigvals[item[1]]), 1.0),
    )
    i, j, distinct = best_pair
    repeated_distance = abs(eigvals[i] - eigvals[j]) / max(abs(eigvals[i]), abs(eigvals[j]), 1.0)
    distinct_distance = min(
        abs(eigvals[distinct] - eigvals[i]) / max(abs(eigvals[distinct]), abs(eigvals[i]), 1.0),
        abs(eigvals[distinct] - eigvals[j]) / max(abs(eigvals[distinct]), abs(eigvals[j]), 1.0),
    )
    if repeated_distance > degeneracy_tol or distinct_distance <= degeneracy_tol:
        raise ValueError("epsilon_lab is not detectably uniaxial.")
    return _normalize_vector(eigvecs[:, distinct])


def _ordinary_alignment(mode: AnisotropicMode, optic_axis: np.ndarray) -> float:
    ordinary_direction = np.cross(optic_axis, mode.direction)
    norm = np.linalg.norm(ordinary_direction)
    if norm < 1e-10:
        raise ValueError("Ordinary/extraordinary identity is degenerate for propagation along the optic axis.")
    ordinary_direction = ordinary_direction / norm
    electric = _normalize_vector(mode.electric_direction)
    return float(abs(np.vdot(ordinary_direction, electric)))


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    out = np.asarray(vector, dtype=complex)
    norm = np.linalg.norm(out)
    if norm == 0:
        raise ValueError("Vector norm is zero.")
    out = out / norm
    pivot = int(np.argmax(np.abs(out)))
    phase = np.angle(out[pivot])
    out = out * np.exp(-1j * phase)
    out[np.abs(out) < 1e-12] = 0
    return out


def _mode_overlap(left: AnisotropicMode, right: AnisotropicMode) -> float:
    left_e = _normalize_vector(left.electric_direction)
    right_e = _normalize_vector(right.electric_direction)
    return float(abs(np.vdot(left_e, right_e)))
