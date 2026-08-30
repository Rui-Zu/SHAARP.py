from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anisotropic import SnellModes, identify_uniaxial_modes, solve_snell_modes
from .boundary import boundary_residual, solve_boundary_continuity
from .linear import LinearInterfaceResult, mode_to_wave, solve_linear_interface
from .nonlinear import (
    InhomogeneousField,
    SingleInterfaceInhomogeneousFields,
    SingleInterfaceSources,
    single_interface_sources,
    solve_single_interface_inhomogeneous_fields,
)
from .waves import EPS0, MU0, Wave, make_isotropic_wave


@dataclass(frozen=True)
class SingleInterfaceSHGResult:
    """Single-interface 2 omega boundary solve building block."""

    linear_omega: LinearInterfaceResult
    transmitted_2omega_modes: SnellModes
    sources: SingleInterfaceSources
    inhomogeneous: SingleInterfaceInhomogeneousFields
    reflected_s_2omega: Wave
    reflected_p_2omega: Wave
    transmitted_fast_2omega: Wave
    transmitted_slow_2omega: Wave
    coefficients: np.ndarray
    boundary_residual: np.ndarray
    source_mode_basis: str
    source_mode_order: tuple[str, str]

    @property
    def reflected_total_2omega(self) -> Wave:
        return Wave(
            k=self.reflected_s_2omega.k,
            electric=self.reflected_s_2omega.electric + self.reflected_p_2omega.electric,
            omega=self.reflected_s_2omega.omega,
        )

    @property
    def transmitted_total_2omega(self) -> np.ndarray:
        return (
            self.transmitted_fast_2omega.electric
            + self.transmitted_slow_2omega.electric
            + self.inhomogeneous.ee.electric
            + self.inhomogeneous.oo.electric
            + self.inhomogeneous.eo.electric
        )


def solve_single_interface_shg(
    epsilon_omega_lab: np.ndarray,
    epsilon_2omega_lab: np.ndarray,
    d_voigt_lab: np.ndarray,
    *,
    incident_index_omega: complex,
    incident_index_2omega: complex,
    incident_theta_rad: float,
    incident_polarization: str = "p",
    incident_jones: tuple[complex, complex] | None = None,
    omega: float = 1.0,
    mu: float = MU0,
    eps0: float = EPS0,
) -> SingleInterfaceSHGResult:
    """Port the numeric structure of Mathematica `f1NL` for one interface.

    Transmitted homogeneous anisotropic modes may be real/lossless or complex
    absorbing. The reflected SH basis is still isotropic on the incident side.
    This generic lab-frame function does not claim SHAARP.si equivalence (no lab-only o/e split exists for biaxial crystals at oblique incidence); the SHAARP.si-equivalent validated entry point is the compat workflow via run_si_numeric (workflow=shaarp_si_compat).

    INPUT POLARIZATION: pass ``incident_jones=(J_s, J_p)`` for an arbitrary input
    polarization (e.g. ``(sin(phi), cos(phi))`` for a polarimetry scan). The linear
    Fresnel solve is linear in the incident field, so the driving extraordinary/ordinary
    fundamental fields combine as ``J_s * (from s) + J_p * (from p)`` -> the reflected SHG
    is the bilinear polarimetry response (it is the numeric reference for the closed-form
    ``solve_si_shg_full_analytical_symbolic``). When ``incident_jones`` is None, the
    discrete ``incident_polarization`` ('s'/'p') path is used unchanged.
    """

    if incident_jones is None:
        linear = solve_linear_interface(
            epsilon_omega_lab,
            incident_index=incident_index_omega,
            incident_theta_rad=incident_theta_rad,
            incident_polarization=incident_polarization,
            omega=omega,
        )
    else:
        linear = solve_linear_interface(
            epsilon_omega_lab, incident_index=incident_index_omega,
            incident_theta_rad=incident_theta_rad, incident_polarization="p", omega=omega,
        )
        linear_s = solve_linear_interface(
            epsilon_omega_lab, incident_index=incident_index_omega,
            incident_theta_rad=incident_theta_rad, incident_polarization="s", omega=omega,
        )
    omega_2 = 2 * omega
    target = incident_index_omega * np.sin(incident_theta_rad)
    reflected_basis = _reflected_2omega_basis(
        incident_index_2omega=incident_index_2omega,
        tangential_target=target,
        omega_2=omega_2,
    )
    transmitted_modes = solve_snell_modes(
        epsilon_2omega_lab,
        incident_index=incident_index_omega,
        incident_theta_rad=incident_theta_rad,
        reflected=False,
    )
    transmitted_basis = [
        mode_to_wave(transmitted_modes.fast, omega=omega_2),
        mode_to_wave(transmitted_modes.slow, omega=omega_2),
    ]
    if incident_jones is None:
        extraordinary_omega, ordinary_omega, source_mode_basis, source_mode_order = _source_waves_for_nonlinearity(
            epsilon_omega_lab,
            linear,
        )
    else:
        j_s, j_p = incident_jones
        e_p, o_p, source_mode_basis, source_mode_order = _source_waves_for_nonlinearity(epsilon_omega_lab, linear)
        e_s, o_s, _, _ = _source_waves_for_nonlinearity(epsilon_omega_lab, linear_s)
        # linear in the incident field: combine the e/o driving fields by the Jones weights
        # (same transmitted wavevector for s and p, so the fields add)
        extraordinary_omega = Wave(
            k=e_p.k, electric=j_s * np.asarray(e_s.electric) + j_p * np.asarray(e_p.electric), omega=omega,
        )
        ordinary_omega = Wave(
            k=o_p.k, electric=j_s * np.asarray(o_s.electric) + j_p * np.asarray(o_p.electric), omega=omega,
        )
    sources = single_interface_sources(
        d_voigt_lab,
        extraordinary_omega,
        ordinary_omega,
    )
    inhomogeneous = solve_single_interface_inhomogeneous_fields(
        epsilon_2omega_lab,
        sources,
        mu=mu,
        eps0=eps0,
    )
    inhomogeneous_waves = [_inhomogeneous_to_wave(field) for field in (inhomogeneous.ee, inhomogeneous.oo, inhomogeneous.eo)]
    reflected, transmitted, coeffs = solve_boundary_continuity(
        top_known=[],
        bottom_known=inhomogeneous_waves,
        top_unknown_basis=reflected_basis,
        bottom_unknown_basis=transmitted_basis,
        mu=mu,
    )
    residual = boundary_residual(reflected, transmitted + inhomogeneous_waves, mu=mu)
    return SingleInterfaceSHGResult(
        linear_omega=linear,
        transmitted_2omega_modes=transmitted_modes,
        sources=sources,
        inhomogeneous=inhomogeneous,
        reflected_s_2omega=reflected[0],
        reflected_p_2omega=reflected[1],
        transmitted_fast_2omega=transmitted[0],
        transmitted_slow_2omega=transmitted[1],
        coefficients=coeffs,
        boundary_residual=residual,
        source_mode_basis=source_mode_basis,
        source_mode_order=source_mode_order,
    )


def _source_waves_for_nonlinearity(
    epsilon_omega_lab: np.ndarray,
    linear: LinearInterfaceResult,
) -> tuple[Wave, Wave, str, tuple[str, str]]:
    """Return omega waves as `(extraordinary, ordinary)` for SHAARP PNL terms."""

    try:
        identified = identify_uniaxial_modes(epsilon_omega_lab, linear.transmitted_modes)
    except ValueError:
        return (
            linear.transmitted_fast,
            linear.transmitted_slow,
            "fast_slow_unclassified",
            ("fast", "slow"),
        )
    return (
        _linear_wave_for_branch(linear, identified.extraordinary_branch),
        _linear_wave_for_branch(linear, identified.ordinary_branch),
        "uniaxial_ordinary_extraordinary",
        (identified.extraordinary_branch, identified.ordinary_branch),
    )


def _linear_wave_for_branch(linear: LinearInterfaceResult, branch: str) -> Wave:
    if branch == "fast":
        return linear.transmitted_fast
    if branch == "slow":
        return linear.transmitted_slow
    raise ValueError(f"Unknown mode branch: {branch}")


def _reflected_2omega_basis(
    *,
    incident_index_2omega: complex,
    tangential_target: complex,
    omega_2: float,
) -> list[Wave]:
    sin_theta = tangential_target / incident_index_2omega
    theta = np.arcsin(sin_theta)
    return [
        make_isotropic_wave(
            n=incident_index_2omega,
            theta_rad=theta,
            polarization="s",
            omega=omega_2,
            direction_sign_z=-1.0,
        ),
        make_isotropic_wave(
            n=incident_index_2omega,
            theta_rad=theta,
            polarization="p",
            omega=omega_2,
            direction_sign_z=-1.0,
        ),
    ]


def _inhomogeneous_to_wave(field: InhomogeneousField) -> Wave:
    return Wave(k=field.k, electric=field.electric, omega=field.omega)
