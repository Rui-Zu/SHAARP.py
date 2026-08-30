from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def voigt_to_chi2(d_voigt: np.ndarray) -> np.ndarray:
    """Expand SHAARP's 3x6 Voigt SHG tensor to d_ijk."""

    d = np.asarray(d_voigt, dtype=complex)
    if d.shape != (3, 6):
        raise ValueError("d_voigt must have shape (3, 6).")
    out = np.zeros((3, 3, 3), dtype=complex)
    pairs = {
        0: [(0, 0)],
        1: [(1, 1)],
        2: [(2, 2)],
        3: [(1, 2), (2, 1)],
        4: [(0, 2), (2, 0)],
        5: [(0, 1), (1, 0)],
    }
    for i in range(3):
        for col, js in pairs.items():
            for j, k in js:
                out[i, j, k] = d[i, col]
    return out


def chi2_to_voigt(chi2: np.ndarray) -> np.ndarray:
    """Collapse a j/k-symmetric rank-3 SHG tensor to SHAARP 3x6 Voigt form."""

    chi = np.asarray(chi2, dtype=complex)
    if chi.shape != (3, 3, 3):
        raise ValueError("chi2 must have shape (3, 3, 3).")
    return np.column_stack(
        [
            chi[:, 0, 0],
            chi[:, 1, 1],
            chi[:, 2, 2],
            0.5 * (chi[:, 1, 2] + chi[:, 2, 1]),
            0.5 * (chi[:, 0, 2] + chi[:, 2, 0]),
            0.5 * (chi[:, 0, 1] + chi[:, 1, 0]),
        ]
    )


def rotate_rank2_crystal_to_lab(tensor: np.ndarray, z_axes_in_lab: np.ndarray) -> np.ndarray:
    r = _real_rotation_matrix(z_axes_in_lab)
    return r.T @ np.asarray(tensor, dtype=complex) @ r


def rotate_rank3_crystal_to_lab(tensor: np.ndarray, z_axes_in_lab: np.ndarray) -> np.ndarray:
    r = _real_rotation_matrix(z_axes_in_lab)
    return np.einsum("ai,bj,ck,abc->ijk", r, r, r, np.asarray(tensor, dtype=complex))


def shaarp_qc_matrix(z_axes_in_lab: np.ndarray) -> np.ndarray:
    """Return SHAARP.ml's `QC` matrix for stored orientation axes.

    `hklConvert` stores `{o1, o2, o3}` as crystal physics axes in lab
    coordinates. Mathematica then builds `QC` with rows `Lab_i. o_j`, so for
    the Python row-wise orientation convention this is `axes.T`.
    """

    return _real_rotation_matrix(z_axes_in_lab).T


def shaarp_principal_axis_transform(point_group: str | list[str], epsilon_crystal: np.ndarray) -> np.ndarray:
    """Return the SHAARP.ml `aCryToPrin` transform used inside `QC2QP`."""

    if isinstance(point_group, (list, tuple)):
        pg = str(point_group[0]) if point_group else ""
    else:
        pg = str(point_group)
    epsilon = np.asarray(epsilon_crystal, dtype=complex)
    if epsilon.shape != (3, 3):
        raise ValueError("epsilon_crystal must be a 3x3 matrix.")

    if pg in {"1", "2", "m", "mm2"}:
        principal_matrix = np.real(sqrtm(epsilon))
        if np.allclose(principal_matrix, principal_matrix.T, atol=1e-12, rtol=1e-12):
            eigvals, eigvecs = np.linalg.eigh(principal_matrix)
        else:
            eigvals, eigvecs = np.linalg.eig(principal_matrix)
            order = np.argsort(eigvals)
            eigvecs = eigvecs[:, order]
        transform = eigvecs.T
        if np.allclose(transform @ epsilon @ transform.T, epsilon, atol=5e-7, rtol=5e-7):
            return np.eye(3, dtype=complex)
        return transform.astype(complex)
    return np.eye(3, dtype=complex)


def shaarp_qp_matrix(point_group: str | list[str], epsilon_crystal: np.ndarray, z_axes_in_lab: np.ndarray) -> np.ndarray:
    """Return SHAARP.ml's `QPomega`/`QP2omega` matrix for an orientation."""

    axes = _real_rotation_matrix(z_axes_in_lab)
    return shaarp_principal_axis_transform(point_group, epsilon_crystal) @ np.linalg.inv(axes)


def rotate_d_voigt_crystal_to_lab(d_voigt: np.ndarray, z_axes_in_lab: np.ndarray) -> np.ndarray:
    """Rotate SHAARP Voigt `dC` into lab coordinates as Mathematica `dL`."""

    return chi2_to_voigt(rotate_rank3_crystal_to_lab(voigt_to_chi2(d_voigt), z_axes_in_lab))


def principal_index(epsilon: np.ndarray) -> complex:
    """Effective scalar index used by the current semi-analytical solvers."""

    eig = np.linalg.eigvals(np.asarray(epsilon, dtype=complex))
    return np.sqrt(np.mean(eig))


def impose_point_group(d: np.ndarray, point_group: str) -> np.ndarray:
    """Apply common SHG Voigt constraints.

    This helper covers the presets included here and leaves unsupported groups
    unchanged so users can still enter a fully custom tensor.
    """

    out = np.asarray(d, dtype=complex).copy()
    pg = point_group.strip().lower().replace(" ", "")
    if pg in {"-43m", "43m", "td"}:
        value = out[0, 3] or out[1, 4] or out[2, 5]
        out[:] = 0
        out[0, 3] = out[1, 4] = out[2, 5] = value
    elif pg in {"3m", "c3v"}:
        d15 = out[0, 4]
        d22 = out[1, 1]
        d31 = out[2, 0]
        d33 = out[2, 2]
        out[:] = 0
        out[0, 4] = out[1, 3] = d15
        out[0, 0] = -d22
        out[0, 1] = d22
        out[1, 5] = -d22
        out[2, 0] = out[2, 1] = d31
        out[2, 2] = d33
    elif pg in {"6mm", "c6v", "4mm", "c4v"}:
        d15 = out[0, 4]
        d31 = out[2, 0]
        d33 = out[2, 2]
        out[:] = 0
        out[0, 4] = out[1, 3] = d15
        out[2, 0] = out[2, 1] = d31
        out[2, 2] = d33
    return out


def _real_rotation_matrix(z_axes_in_lab: np.ndarray) -> np.ndarray:
    raw = np.asarray(z_axes_in_lab)
    if np.iscomplexobj(raw) and np.max(np.abs(np.imag(raw))) > 0:
        raise ValueError("z_axes_in_lab must be real; complex spatial rotations are not supported.")
    r = np.asarray(z_axes_in_lab, dtype=float)
    if r.shape != (3, 3):
        raise ValueError("z_axes_in_lab must be a 3x3 matrix.")
    return r
