from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import MultilayerSystem, Polarimetry
from .single_interface import single_interface_intensity
from .tensors import principal_index


@dataclass(frozen=True)
class MultilayerResult:
    phi_deg: np.ndarray
    reflected_intensity: np.ndarray
    transmitted_intensity: np.ndarray
    layer_sources: list[np.ndarray]
    model: str
    warnings: tuple[str, ...] = ()

    def plot(self, ax=None):
        ax = ax or plt.subplot(projection="polar")
        phi = np.deg2rad(np.asarray(self.phi_deg, dtype=float))
        ax.plot(phi, np.real(self.reflected_intensity), label="R")
        ax.plot(phi, np.real(self.transmitted_intensity), label="T")
        ax.set_title("SHAARP.ml SHG intensity")
        ax.legend()
        return ax

    def save_csv(self, path: str | Path) -> None:
        """Save phi, reflected intensity, and transmitted intensity columns."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack(
            [
                np.ravel(self.phi_deg),
                np.ravel(np.real(self.reflected_intensity)),
                np.ravel(np.real(self.transmitted_intensity)),
            ]
        )
        np.savetxt(
            path,
            data,
            delimiter=",",
            header="phi_deg,reflected_intensity,transmitted_intensity",
            comments="",
        )


def multilayer_shg(system: MultilayerSystem) -> MultilayerResult:
    """Reduced multilayer SHG response preserving SHAARP.ml configuration.

    This is not a full port of the Mathematica SHAARP.ml solver. The notebook
    solves forward/backward anisotropic waves and multilayer boundary-condition
    equations. This function only coherently sums internal nonlinear layer
    source terms using scalar effective refractive indices.
    """

    system.validate()
    if system.include_multiple_reflections:
        raise NotImplementedError(
            "Full multiple reflection is not implemented in this reduced Python "
            "model. Set include_multiple_reflections=False or port the "
            "SHAARP.ml boundary-condition matrix solver first."
        )
    if system.include_backward_shg:
        raise NotImplementedError(
            "Backward SHG waves are not implemented in this reduced Python model."
        )

    pol = system.polarimetry
    model_warnings = [
        "Reduced model warning: this is a coherent source-sum model, not the "
        "full SHAARP.ml forward/backward anisotropic multilayer solver."
    ]
    for message in model_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    theta, phi, _, _ = np.broadcast_arrays(*pol.radians())
    k0_2w = 4 * np.pi / system.wavelength_um

    sources: list[np.ndarray] = []
    reflected = np.zeros_like(phi, dtype=complex)
    transmitted = np.zeros_like(phi, dtype=complex)
    optical_depth_to_front = 0.0j
    total_optical_depth = _total_optical_depth(system)

    for layer in system.layers[1:-1]:
        n2 = principal_index(layer.material.eps_2w())
        thickness = layer.thickness_um or 0.0
        if layer.shg_active:
            local_pol = Polarimetry(
                theta_deg=pol.theta_deg,
                phi_deg=pol.phi_deg,
                psi_deg=pol.psi_deg,
                ellipticity_deg=pol.ellipticity_deg,
                polarizer_mode=pol.polarizer_mode,
                analyzer_mode=pol.analyzer_mode,
            )
            si = single_interface_intensity(layer.material, local_pol)
            source = si.e_2omega * thickness
        else:
            source = np.zeros_like(phi, dtype=complex)
        phase_front = np.exp(1j * k0_2w * optical_depth_to_front / np.cos(theta))
        phase_back = np.exp(1j * k0_2w * (total_optical_depth - optical_depth_to_front) / np.cos(theta))
        reflected += source * phase_front
        transmitted += source * phase_back
        sources.append(source)
        optical_depth_to_front += n2 * thickness

    return MultilayerResult(
        phi_deg=np.rad2deg(phi),
        reflected_intensity=np.abs(reflected) ** 2,
        transmitted_intensity=np.abs(transmitted) ** 2,
        layer_sources=sources,
        model="reduced_effective_index_layer_sum",
        warnings=tuple(model_warnings),
    )


def _total_optical_depth(system: MultilayerSystem) -> complex:
    total = 0.0j
    for layer in system.layers[1:-1]:
        total += principal_index(layer.material.eps_2w()) * (layer.thickness_um or 0.0)
    return total
