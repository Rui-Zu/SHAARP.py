from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anisotropic import ShaarpEigenBasis, solve_snell_modes
from .linear import mode_to_wave
from .waves import Wave, make_isotropic_wave


@dataclass(frozen=True)
class HomogeneousMultilayerBasis:
    """Homogeneous wave bases for the current multilayer boundary kernel."""

    incident: list[Wave]
    reflected_basis: list[Wave]
    layer_basis: list[list[Wave]]
    substrate_basis: list[Wave]
    tangential_index_target: complex


def make_anisotropic_layer_basis(
    epsilon_lab: np.ndarray,
    *,
    tangential_index: complex,
    incident_theta_rad: float,
    omega: float = 1.0,
    shaarp_basis: ShaarpEigenBasis | None = None,
) -> list[Wave]:
    """Return one layer basis ordered `[F1, F2, B1, B2]`.

    `tangential_index * sin(incident_theta_rad)` is the dimensionless Snell
    target. At 2 omega this should normally be the omega incident index, not
    necessarily the top-medium 2 omega index, so the homogeneous waves match
    the nonlinear source tangential wavevector.
    """

    forward = solve_snell_modes(
        epsilon_lab,
        incident_index=tangential_index,
        incident_theta_rad=incident_theta_rad,
        reflected=False,
        shaarp_basis=shaarp_basis,
    )
    backward = solve_snell_modes(
        epsilon_lab,
        incident_index=tangential_index,
        incident_theta_rad=incident_theta_rad,
        reflected=True,
        shaarp_basis=shaarp_basis,
    )
    return [
        mode_to_wave(forward.fast, omega=omega),
        mode_to_wave(forward.slow, omega=omega),
        mode_to_wave(backward.fast, omega=omega),
        mode_to_wave(backward.slow, omega=omega),
    ]


def make_anisotropic_substrate_basis(
    epsilon_lab: np.ndarray,
    *,
    tangential_index: complex,
    incident_theta_rad: float,
    omega: float = 1.0,
    shaarp_basis: ShaarpEigenBasis | None = None,
) -> list[Wave]:
    """Return the two forward substrate modes for the boundary kernel."""

    modes = solve_snell_modes(
        epsilon_lab,
        incident_index=tangential_index,
        incident_theta_rad=incident_theta_rad,
        reflected=False,
        shaarp_basis=shaarp_basis,
    )
    return [
        mode_to_wave(modes.fast, omega=omega),
        mode_to_wave(modes.slow, omega=omega),
    ]


def build_multilayer_omega_basis(
    *,
    incident_index: complex,
    incident_theta_rad: float,
    incident_polarization: str,
    layer_epsilon_lab: list[np.ndarray],
    substrate_epsilon_lab: np.ndarray,
    omega: float = 1.0,
    layer_shaarp_basis: list[ShaarpEigenBasis | None] | None = None,
    substrate_shaarp_basis: ShaarpEigenBasis | None = None,
) -> HomogeneousMultilayerBasis:
    """Build the omega basis set for an isotropic top medium."""

    if incident_polarization not in {"s", "p"}:
        raise ValueError("incident_polarization must be 's' or 'p'.")
    incident = [
        make_isotropic_wave(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization=incident_polarization,
            omega=omega,
        )
    ]
    reflected_basis = [
        make_isotropic_wave(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="s",
            direction_sign_z=-1,
            omega=omega,
        ),
        make_isotropic_wave(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="p",
            direction_sign_z=-1,
            omega=omega,
        ),
    ]
    layer_shaarp_basis = _basis_list_or_none(layer_shaarp_basis, len(layer_epsilon_lab))
    layer_basis = []
    for epsilon_lab, basis in zip(layer_epsilon_lab, layer_shaarp_basis):
        layer_basis.append(
            make_anisotropic_layer_basis(
                epsilon_lab,
                tangential_index=incident_index,
                incident_theta_rad=incident_theta_rad,
                omega=omega,
                shaarp_basis=basis,
            )
        )
    substrate_basis = make_anisotropic_substrate_basis(
        substrate_epsilon_lab,
        tangential_index=incident_index,
        incident_theta_rad=incident_theta_rad,
        omega=omega,
        shaarp_basis=substrate_shaarp_basis,
    )
    return HomogeneousMultilayerBasis(
        incident=incident,
        reflected_basis=reflected_basis,
        layer_basis=layer_basis,
        substrate_basis=substrate_basis,
        tangential_index_target=incident_index * np.sin(incident_theta_rad),
    )


def build_multilayer_omega_basis_jones(
    *,
    incident_index: complex,
    incident_theta_rad: float,
    incident_jones_sp: tuple[complex, complex],
    layer_epsilon_lab: list[np.ndarray],
    substrate_epsilon_lab: np.ndarray,
    omega: float = 1.0,
    layer_shaarp_basis: list[ShaarpEigenBasis | None] | None = None,
    substrate_shaarp_basis: ShaarpEigenBasis | None = None,
    reuse: "HomogeneousMultilayerBasis | None" = None,
) -> HomogeneousMultilayerBasis:
    """Build the omega basis for arbitrary incident Jones amplitudes.

    `incident_jones_sp` is ordered `(s_amplitude, p_amplitude)`.

    `reuse` (perf): a basis previously built at the SAME angle for a DIFFERENT incident
    polarization. Only `incident` depends on `incident_jones_sp`; the reflected/layer/substrate
    bases and the tangential index depend solely on (eps, incident_index, incident_theta_rad,
    omega), and the layer/substrate ones carry the expensive anisotropic Snell-mode root-finds.
    Passing `reuse` rebuilds only `incident` and is byte-identical to a full rebuild -- the reused
    fields are exactly what this call would recompute from identical inputs. Used by the GUI
    Fresnel sweep, which solves p- and s-incidence at every angle.
    """

    s_amplitude, p_amplitude = incident_jones_sp
    incident = []
    if abs(s_amplitude) > 0:
        incident.append(
            make_isotropic_wave(
                n=incident_index,
                theta_rad=incident_theta_rad,
                polarization="s",
                amplitude=s_amplitude,
                omega=omega,
            )
        )
    if abs(p_amplitude) > 0:
        incident.append(
            make_isotropic_wave(
                n=incident_index,
                theta_rad=incident_theta_rad,
                polarization="p",
                amplitude=p_amplitude,
                omega=omega,
            )
        )
    if not incident:
        raise ValueError("incident_jones_sp must contain at least one non-zero amplitude.")
    if reuse is not None:
        return HomogeneousMultilayerBasis(
            incident=incident,
            reflected_basis=reuse.reflected_basis,
            layer_basis=reuse.layer_basis,
            substrate_basis=reuse.substrate_basis,
            tangential_index_target=reuse.tangential_index_target,
        )
    reflected_basis = [
        make_isotropic_wave(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="s",
            direction_sign_z=-1,
            omega=omega,
        ),
        make_isotropic_wave(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="p",
            direction_sign_z=-1,
            omega=omega,
        ),
    ]
    layer_shaarp_basis = _basis_list_or_none(layer_shaarp_basis, len(layer_epsilon_lab))
    layer_basis = []
    for epsilon_lab, basis in zip(layer_epsilon_lab, layer_shaarp_basis):
        layer_basis.append(
            make_anisotropic_layer_basis(
                epsilon_lab,
                tangential_index=incident_index,
                incident_theta_rad=incident_theta_rad,
                omega=omega,
                shaarp_basis=basis,
            )
        )
    substrate_basis = make_anisotropic_substrate_basis(
        substrate_epsilon_lab,
        tangential_index=incident_index,
        incident_theta_rad=incident_theta_rad,
        omega=omega,
        shaarp_basis=substrate_shaarp_basis,
    )
    return HomogeneousMultilayerBasis(
        incident=incident,
        reflected_basis=reflected_basis,
        layer_basis=layer_basis,
        substrate_basis=substrate_basis,
        tangential_index_target=incident_index * np.sin(incident_theta_rad),
    )


def build_multilayer_2omega_basis(
    *,
    top_index_2omega: complex,
    tangential_index_omega: complex,
    incident_theta_rad: float,
    layer_epsilon_2omega_lab: list[np.ndarray],
    substrate_epsilon_2omega_lab: np.ndarray,
    omega_2: float,
    layer_shaarp_basis_2omega: list[ShaarpEigenBasis | None] | None = None,
    substrate_shaarp_basis_2omega: ShaarpEigenBasis | None = None,
) -> HomogeneousMultilayerBasis:
    """Build the homogeneous 2 omega bases matching the SHG source kx."""

    theta_top_2omega = np.arcsin(tangential_index_omega * np.sin(incident_theta_rad) / top_index_2omega + 0j)
    reflected_basis = [
        make_isotropic_wave(
            n=top_index_2omega,
            theta_rad=theta_top_2omega,
            polarization="s",
            direction_sign_z=-1,
            omega=omega_2,
        ),
        make_isotropic_wave(
            n=top_index_2omega,
            theta_rad=theta_top_2omega,
            polarization="p",
            direction_sign_z=-1,
            omega=omega_2,
        ),
    ]
    layer_shaarp_basis_2omega = _basis_list_or_none(layer_shaarp_basis_2omega, len(layer_epsilon_2omega_lab))
    layer_basis = []
    for epsilon_lab, basis in zip(layer_epsilon_2omega_lab, layer_shaarp_basis_2omega):
        layer_basis.append(
            make_anisotropic_layer_basis(
                epsilon_lab,
                tangential_index=tangential_index_omega,
                incident_theta_rad=incident_theta_rad,
                omega=omega_2,
                shaarp_basis=basis,
            )
        )
    substrate_basis = make_anisotropic_substrate_basis(
        substrate_epsilon_2omega_lab,
        tangential_index=tangential_index_omega,
        incident_theta_rad=incident_theta_rad,
        omega=omega_2,
        shaarp_basis=substrate_shaarp_basis_2omega,
    )
    return HomogeneousMultilayerBasis(
        incident=[],
        reflected_basis=reflected_basis,
        layer_basis=layer_basis,
        substrate_basis=substrate_basis,
        tangential_index_target=tangential_index_omega * np.sin(incident_theta_rad),
    )


def _basis_list_or_none(
    bases: list[ShaarpEigenBasis | None] | None,
    expected_len: int,
) -> list[ShaarpEigenBasis | None]:
    if bases is None:
        return [None] * expected_len
    if len(bases) != expected_len:
        raise ValueError("layer_shaarp_basis length must match layer_epsilon_lab length.")
    return bases
