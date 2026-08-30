from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MU0 = 4 * np.pi * 1e-7
EPS0 = 8.8541878128e-12


def tangential_eh(e, h):
    """Canonical SHAARP tangential-continuity ordering ``[Ex, Ey, Hx, Hy]``.

    SHAARP's ``solveFresnelN`` builds every interface equation as the two
    tangential E-field components (``EE[[1;;2]]`` = x, y) FOLLOWED BY the two
    tangential H-field components (``getH[[1;;2]]``). This single helper is the
    one place the ordering is defined, so the numeric single-interface solver
    (``boundary.py``), the numeric multilayer solver (``multilayer_boundary.py``),
    and the symbolic single-interface/​multilayer solvers (``symbolic.py``,
    ``multilayer_boundary_symbolic.py``) all use IDENTICAL tangential bookkeeping
    -- SI and ML, numeric and symbolic, converged onto one convention.

    ``e`` and ``h`` are any length>=2 indexables (numpy arrays or sympy
    matrices); a plain 4-list is returned for the caller to wrap.
    """

    return [e[0], e[1], h[0], h[1]]


@dataclass(frozen=True)
class Wave:
    """Plane-wave fields used by SHAARP-style boundary solvers."""

    k: np.ndarray
    electric: np.ndarray
    omega: float = 1.0

    def magnetic(self, mu: float = MU0) -> np.ndarray:
        """Return H = k x E / (omega mu), matching Mathematica getH."""

        return np.cross(self.k, self.electric) / (self.omega * mu)


def make_isotropic_wave(
    *,
    n: complex,
    theta_rad: complex,
    polarization: str,
    amplitude: complex = 1.0,
    omega: float = 1.0,
    direction_sign_z: float = 1.0,
) -> Wave:
    """Create an s or p polarized isotropic plane wave in the L1-L3 plane.

    `direction_sign_z=-1` is a reflected/backward wave. Units are normalized
    with c0=1 for coefficient tests; common scale cancels in boundary solves.
    """

    if polarization not in {"s", "p"}:
        raise ValueError("polarization must be 's' or 'p'.")
    k_dir = np.array(
        [np.sin(theta_rad), 0.0, direction_sign_z * np.cos(theta_rad)],
        dtype=complex,
    )
    k = n * omega * k_dir
    if polarization == "s":
        electric_dir = np.array([0.0, 1.0, 0.0], dtype=complex)
    else:
        electric_dir = np.array(
            [direction_sign_z * np.cos(theta_rad), 0.0, -np.sin(theta_rad)],
            dtype=complex,
        )
    return Wave(k=k, electric=amplitude * electric_dir, omega=omega)
