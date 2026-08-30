from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import Material, Polarimetry
from .tensors import (
    principal_index,
    rotate_rank2_crystal_to_lab,
    rotate_rank3_crystal_to_lab,
    voigt_to_chi2,
)


@dataclass(frozen=True)
class SingleInterfaceResult:
    phi_deg: np.ndarray
    psi_deg: np.ndarray
    e_2omega: np.ndarray
    intensity: np.ndarray
    model: str
    warnings: tuple[str, ...] = ()

    def plot(self, ax=None):
        ax = ax or plt.subplot(projection="polar")
        phi = np.deg2rad(np.asarray(self.phi_deg, dtype=float))
        ax.plot(phi, np.real(self.intensity))
        ax.set_title("SHAARP.si reflected SHG intensity")
        return ax

    def save_csv(self, path: str | Path) -> None:
        """Save phi, psi, complex field, and intensity columns."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack(
            [
                np.ravel(self.phi_deg),
                np.ravel(self.psi_deg),
                np.ravel(np.real(self.e_2omega)),
                np.ravel(np.imag(self.e_2omega)),
                np.ravel(np.real(self.intensity)),
            ]
        )
        np.savetxt(
            path,
            data,
            delimiter=",",
            header="phi_deg,psi_deg,e_2omega_real,e_2omega_imag,intensity",
            comments="",
        )


def single_interface_intensity(
    material: Material,
    polarimetry: Polarimetry | None = None,
    incident_index: complex = 1.0,
) -> SingleInterfaceResult:
    """Reduced numerical reflected SHG response for a single interface.

    This is not a full port of the Mathematica SHAARP.si solver. The notebooks
    solve anisotropic eigenmodes and 2omega boundary conditions; this function
    keeps the same configuration objects but uses an effective-index Fresnel
    model, tensor contraction P_i=d_ijk E_j E_k, and analyzer projection.
    """

    pol = polarimetry or Polarimetry()
    model_warnings = _reduced_model_warnings(material)
    for message in model_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    theta, phi, psi, delta = np.broadcast_arrays(*pol.radians())
    r = material.orientation.rotation_matrix()
    eps_w_lab = rotate_rank2_crystal_to_lab(material.eps_w(), r)
    eps_2w_lab = rotate_rank2_crystal_to_lab(material.eps_2w(), r)
    d_lab = rotate_rank3_crystal_to_lab(voigt_to_chi2(material.d_voigt()), r)

    n_w = principal_index(eps_w_lab)
    n_2w = principal_index(eps_2w_lab)
    t_s_w, t_p_w = fresnel_transmission(incident_index, n_w, theta)
    _, t_p_2w = fresnel_transmission(n_2w, incident_index, theta)
    t_s_2w, _ = fresnel_transmission(n_2w, incident_index, theta)

    e_p = np.cos(phi) * t_p_w
    e_s = np.sin(phi) * np.exp(1j * delta) * t_s_w
    e_lab = np.stack(
        [
            e_p * np.cos(theta),
            e_s,
            -e_p * np.sin(theta),
        ],
        axis=0,
    )
    p_lab = np.einsum("ijk,j...,k...->i...", d_lab, e_lab, e_lab)

    analyzer_p = np.stack([np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=0)
    analyzer_s = np.stack([np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)], axis=0)
    e_p_out = np.einsum("i...,i...->...", analyzer_p, p_lab) * t_p_2w
    e_s_out = np.einsum("i...,i...->...", analyzer_s, p_lab) * t_s_2w
    e_out = np.cos(psi) * e_p_out + np.sin(psi) * e_s_out
    intensity = np.abs(e_out) ** 2

    return SingleInterfaceResult(
        phi_deg=np.rad2deg(phi),
        psi_deg=np.rad2deg(psi),
        e_2omega=e_out,
        intensity=intensity,
        model="reduced_effective_index",
        warnings=tuple(model_warnings),
    )


def fresnel_transmission(n1: complex, n2: complex, theta_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sin_t = n1 / n2 * np.sin(theta_i)
    cos_i = np.cos(theta_i)
    cos_t = np.sqrt(1 - sin_t**2 + 0j)
    t_s = 2 * n1 * cos_i / (n1 * cos_i + n2 * cos_t)
    t_p = 2 * n1 * cos_i / (n2 * cos_i + n1 * cos_t)
    return t_s, t_p


def _reduced_model_warnings(material: Material) -> list[str]:
    messages: list[str] = []
    eps_w = material.eps_w()
    eps_2w = material.eps_2w()
    if not _is_scalar_isotropic(eps_w) or not _is_scalar_isotropic(eps_2w):
        messages.append(
            "Reduced model warning: anisotropic dielectric tensors are collapsed "
            "to one effective refractive index. This is not the full SHAARP "
            "anisotropic eigenmode and boundary-condition solver."
        )
    if not np.allclose(material.orientation.rotation_matrix(), np.eye(3), atol=1e-12):
        messages.append(
            "Reduced model warning: crystal orientation is used for tensor "
            "rotation, but not for solving anisotropic propagation directions."
        )
    return messages


def _is_scalar_isotropic(epsilon: np.ndarray) -> bool:
    arr = np.asarray(epsilon, dtype=complex)
    return np.allclose(arr, np.eye(3) * arr[0, 0]) and np.allclose(
        arr - np.diag(np.diag(arr)), 0
    )
