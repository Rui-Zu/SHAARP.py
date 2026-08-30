from __future__ import annotations

from dataclasses import replace

import numpy as np

from .waves import MU0, Wave, tangential_eh


def solve_fresnel_boundary(
    incident: list[Wave],
    reflected_basis: list[Wave],
    transmitted_basis: list[Wave],
    *,
    mu: float = MU0,
) -> tuple[list[Wave], list[Wave], np.ndarray]:
    """Solve tangential E/H continuity, porting Mathematica `solveFresnel`.

    Mathematica constructs:

    - EMtop = sum(E of incident and reflected waves)
    - EMbot = sum(E of transmitted waves)
    - HMtop/HMbot using H = k x E / (omega mu)
    - equations for tangential L1/L2 components of E and H

    Here reflected/transmitted inputs are basis waves with unit amplitudes. The
    returned waves have solved amplitudes applied.
    """

    reflected, transmitted, coeffs = solve_boundary_continuity(
        top_known=incident,
        bottom_known=[],
        top_unknown_basis=reflected_basis,
        bottom_unknown_basis=transmitted_basis,
        mu=mu,
    )
    return reflected, transmitted, coeffs


def solve_boundary_continuity(
    *,
    top_known: list[Wave],
    bottom_known: list[Wave],
    top_unknown_basis: list[Wave],
    bottom_unknown_basis: list[Wave],
    mu: float = MU0,
) -> tuple[list[Wave], list[Wave], np.ndarray]:
    """Solve tangential E/H continuity with known waves on both sides.

    This is the boundary-condition kernel needed for SHG stages where the
    inhomogeneous source field is already known. It solves:

        top_known + top_unknown == bottom_known + bottom_unknown

    for exactly four unknown amplitudes.
    """

    bases = list(top_unknown_basis) + list(bottom_unknown_basis)
    if len(bases) != 4:
        raise ValueError("This boundary solve expects exactly 4 unknown basis waves.")

    known_mismatch = _tangential_sum(top_known, mu=mu) - _tangential_sum(bottom_known, mu=mu)
    rhs = -known_mismatch
    matrix = np.column_stack(
        [_signed_tangential_column(w, idx, len(top_unknown_basis), mu) for idx, w in enumerate(bases)]
    )
    coeffs = np.linalg.solve(matrix, rhs)

    top_unknown = [scale_wave(w, coeffs[i]) for i, w in enumerate(top_unknown_basis)]
    bottom_unknown = [
        scale_wave(w, coeffs[len(top_unknown_basis) + i])
        for i, w in enumerate(bottom_unknown_basis)
    ]
    return top_unknown, bottom_unknown, coeffs


def scale_wave(wave: Wave, amplitude: complex) -> Wave:
    return replace(wave, electric=wave.electric * amplitude)


def _tangential_sum(waves: list[Wave], *, mu: float) -> np.ndarray:
    if not waves:
        return np.zeros(4, dtype=complex)
    e = np.sum([w.electric for w in waves], axis=0)
    h = np.sum([w.magnetic(mu=mu) for w in waves], axis=0)
    return np.array(tangential_eh(e, h), dtype=complex)  # canonical SHAARP [Ex,Ey,Hx,Hy]


def _signed_tangential_column(wave: Wave, index: int, n_reflected: int, mu: float) -> np.ndarray:
    # Top-side unknowns enter with + sign and bottom-side unknowns with - sign:
    # top_known + top_unknown - bottom_known - bottom_unknown = 0.
    sign = 1.0 if index < n_reflected else -1.0
    h = wave.magnetic(mu=mu)
    return sign * np.array([wave.electric[0], wave.electric[1], h[0], h[1]], dtype=complex)


def boundary_residual(top_waves: list[Wave], bottom_waves: list[Wave], *, mu: float = MU0) -> np.ndarray:
    """Return tangential top-minus-bottom E/H mismatch."""

    return _tangential_sum(top_waves, mu=mu) - _tangential_sum(bottom_waves, mu=mu)


def isotropic_fresnel_coefficients(n1: complex, n2: complex, theta_i_rad: float) -> dict[str, complex]:
    """Closed-form isotropic Fresnel amplitudes for verification."""

    sin_t = n1 / n2 * np.sin(theta_i_rad)
    cos_i = np.cos(theta_i_rad)
    cos_t = np.sqrt(1 - sin_t**2 + 0j)
    rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    ts = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    tp = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    return {"rs": rs, "ts": ts, "rp": rp, "tp": tp, "theta_t": np.arcsin(sin_t)}
