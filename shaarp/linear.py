from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .anisotropic import BranchTrackedModes, SnellModes, solve_snell_modes, track_mode_branches
from .boundary import solve_fresnel_boundary
from .waves import Wave, make_isotropic_wave


@dataclass(frozen=True)
class LinearInterfaceResult:
    """Linear interface amplitudes for isotropic incident medium to anisotropic medium."""

    incident: Wave
    reflected_s: Wave
    reflected_p: Wave
    transmitted_fast: Wave
    transmitted_slow: Wave
    coefficients: np.ndarray
    transmitted_modes: SnellModes

    @property
    def r_s(self) -> complex:
        return self.coefficients[0]

    @property
    def r_p(self) -> complex:
        return self.coefficients[1]

    @property
    def t_fast(self) -> complex:
        return self.coefficients[2]

    @property
    def t_slow(self) -> complex:
        return self.coefficients[3]


@dataclass(frozen=True)
class BranchTrackedLinearInterfaceResult:
    """One linear-interface result with continuity-tracked transmitted branches."""

    linear: LinearInterfaceResult
    transmitted_branches: BranchTrackedModes
    transmitted_branch_1: Wave
    transmitted_branch_2: Wave


@dataclass(frozen=True)
class FresnelCoefficientSweepResult:
    """SHAARP.ml-style Fresnel coefficient sweep over incidence angle.

    `rp`, `rs`, `tp`, and `ts` are power reflectance/transmittance values from
    normal Poynting-flux ratios. `shaarp_ml_copy_lists` preserves the GUI Copy
    button order: Rp, Rs, Tp, Ts, each as `{theta_deg, value}` rows.
    """

    theta_deg: np.ndarray
    rp: np.ndarray
    rs: np.ndarray
    tp: np.ndarray
    ts: np.ndarray
    p_results: tuple[LinearInterfaceResult, ...]
    s_results: tuple[LinearInterfaceResult, ...]

    @property
    def list_fresnel(self) -> list[np.ndarray]:
        return self.shaarp_ml_copy_lists()

    def shaarp_ml_copy_lists(self) -> list[np.ndarray]:
        return [
            np.column_stack([self.theta_deg, self.rp]),
            np.column_stack([self.theta_deg, self.rs]),
            np.column_stack([self.theta_deg, self.tp]),
            np.column_stack([self.theta_deg, self.ts]),
        ]


def solve_linear_interface(
    epsilon_transmitted_lab: np.ndarray,
    *,
    incident_index: complex,
    incident_theta_rad: float,
    incident_polarization: str,
    omega: float = 1.0,
) -> LinearInterfaceResult:
    """Solve the linear omega interface stage for a SHAARP-style material.

    This connects the ported `solveSnell` and `solveFresnel` pieces:

    1. transmitted fast/slow mode directions from anisotropic Snell solve
    2. reflected s/p basis waves in the incident isotropic medium
    3. transmitted fast/slow electric-direction basis waves
    4. tangential E/H boundary solve for four unknown amplitudes

    Current verified scope is real, lossless transmitted dielectric tensors.
    """

    if incident_polarization not in {"s", "p"}:
        raise ValueError("incident_polarization must be 's' or 'p'.")

    modes = solve_snell_modes(
        epsilon_transmitted_lab,
        incident_index=incident_index,
        incident_theta_rad=incident_theta_rad,
        reflected=False,
    )
    incident = make_isotropic_wave(
        n=incident_index,
        theta_rad=incident_theta_rad,
        polarization=incident_polarization,
        amplitude=1.0,
        omega=omega,
    )
    reflected_basis = [
        make_isotropic_wave(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="s",
            amplitude=1.0,
            omega=omega,
            direction_sign_z=-1.0,
        ),
        make_isotropic_wave(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="p",
            amplitude=1.0,
            omega=omega,
            direction_sign_z=-1.0,
        ),
    ]
    transmitted_basis = [
        mode_to_wave(modes.fast, omega=omega),
        mode_to_wave(modes.slow, omega=omega),
    ]
    reflected, transmitted, coeffs = solve_fresnel_boundary(
        [incident],
        reflected_basis,
        transmitted_basis,
    )
    return LinearInterfaceResult(
        incident=incident,
        reflected_s=reflected[0],
        reflected_p=reflected[1],
        transmitted_fast=transmitted[0],
        transmitted_slow=transmitted[1],
        coefficients=coeffs,
        transmitted_modes=modes,
    )


def solve_linear_interface_sweep(
    epsilon_transmitted_lab: np.ndarray,
    *,
    incident_index: complex,
    incident_theta_rad: Sequence[float],
    incident_polarization: str,
    omega: float = 1.0,
    ambiguity_tol: float = 1e-6,
) -> list[BranchTrackedLinearInterfaceResult]:
    """Solve a linear angle sweep and track transmitted eigenmode branches.

    This is intended for biaxial or otherwise non-uniaxial sweeps where the
    per-angle fast/slow sort may swap identities. It does not assign
    ordinary/extraordinary labels; it only attaches continuous branch 1/branch 2
    bookkeeping to the existing single-angle results.
    """

    results = [
        solve_linear_interface(
            epsilon_transmitted_lab,
            incident_index=incident_index,
            incident_theta_rad=theta,
            incident_polarization=incident_polarization,
            omega=omega,
        )
        for theta in incident_theta_rad
    ]
    tracked = track_mode_branches([result.transmitted_modes for result in results], ambiguity_tol=ambiguity_tol)
    return [
        BranchTrackedLinearInterfaceResult(
            linear=result,
            transmitted_branches=branches,
            transmitted_branch_1=_transmitted_wave_for_label(result, branches.branch_1_source_label),
            transmitted_branch_2=_transmitted_wave_for_label(result, branches.branch_2_source_label),
        )
        for result, branches in zip(results, tracked)
    ]


def solve_fresnel_coefficient_sweep(
    epsilon_transmitted_lab: np.ndarray,
    *,
    incident_index: complex,
    theta_deg: Sequence[float] | None = None,
    theta_step_deg: float = 1.0,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 89.0,
    omega: float = 1.0,
) -> FresnelCoefficientSweepResult:
    """Generate a SHAARP.ml FresnelList-style angle sweep.

    SHAARP.ml builds `FresnelList = Fresnel[mats, theta, mrassumption] /@ Range[0, 89, step]`
    and exposes GUI Copy buttons in the order Rp, Rs, Tp, Ts. This helper keeps
    that output shape as first-class Python data. It uses the staged
    single-interface omega solver and computes power ratios from normal
    Poynting flux; it is a legacy helper, not the Mathematica-validated path.
    The listFresnel-validated sweep (selected-wave policy, ~1e-10 vs live
    SHAARP.ml) is the merged GUI Fresnel Coefficients functionality.
    """

    theta_values_deg = _theta_grid(theta_deg, theta_step_deg, theta_min_deg, theta_max_deg)
    p_results = tuple(
        solve_linear_interface(
            epsilon_transmitted_lab,
            incident_index=incident_index,
            incident_theta_rad=np.deg2rad(theta),
            incident_polarization="p",
            omega=omega,
        )
        for theta in theta_values_deg
    )
    s_results = tuple(
        solve_linear_interface(
            epsilon_transmitted_lab,
            incident_index=incident_index,
            incident_theta_rad=np.deg2rad(theta),
            incident_polarization="s",
            omega=omega,
        )
        for theta in theta_values_deg
    )
    rp = np.array([_reflected_power_ratio(result) for result in p_results], dtype=float)
    rs = np.array([_reflected_power_ratio(result) for result in s_results], dtype=float)
    tp = np.array([_transmitted_power_ratio(result) for result in p_results], dtype=float)
    ts = np.array([_transmitted_power_ratio(result) for result in s_results], dtype=float)
    return FresnelCoefficientSweepResult(
        theta_deg=theta_values_deg,
        rp=rp,
        rs=rs,
        tp=tp,
        ts=ts,
        p_results=p_results,
        s_results=s_results,
    )


def mode_to_wave(mode, *, omega: float = 1.0) -> Wave:
    k = mode.refractive_index * omega * mode.direction
    return Wave(k=np.asarray(k, dtype=complex), electric=mode.electric_direction, omega=omega)


def _transmitted_wave_for_label(result: LinearInterfaceResult, label: str) -> Wave:
    if label == "fast":
        return result.transmitted_fast
    if label == "slow":
        return result.transmitted_slow
    raise ValueError(f"Unknown transmitted branch source label: {label}")


def _theta_grid(
    theta_deg: Sequence[float] | None,
    theta_step_deg: float,
    theta_min_deg: float,
    theta_max_deg: float,
) -> np.ndarray:
    if theta_deg is not None:
        values = np.asarray(theta_deg, dtype=float)
    else:
        if theta_step_deg <= 0:
            raise ValueError("theta_step_deg must be positive.")
        if theta_max_deg < theta_min_deg:
            raise ValueError("theta_max_deg must be greater than or equal to theta_min_deg.")
        count = int(np.floor((theta_max_deg - theta_min_deg) / theta_step_deg + 1e-12)) + 1
        values = theta_min_deg + theta_step_deg * np.arange(count, dtype=float)
    if values.ndim != 1:
        raise ValueError("theta_deg must be a one-dimensional sequence.")
    if np.any(values < 0) or np.any(values >= 90):
        raise ValueError("Fresnel sweep incidence angles must satisfy 0 <= theta_deg < 90.")
    return values


def _normal_poynting_flux(wave: Wave) -> float:
    return float(np.real(np.cross(wave.electric, np.conjugate(wave.magnetic()))[2]) / 2.0)


def _reflected_power_ratio(result: LinearInterfaceResult) -> float:
    incident_flux = _normal_poynting_flux(result.incident)
    reflected_flux = _normal_poynting_flux(result.reflected_s) + _normal_poynting_flux(result.reflected_p)
    if incident_flux <= 0:
        raise ValueError("Incident normal Poynting flux must be positive.")
    return max(0.0, -reflected_flux / incident_flux)


def _transmitted_power_ratio(result: LinearInterfaceResult) -> float:
    incident_flux = _normal_poynting_flux(result.incident)
    transmitted_flux = _normal_poynting_flux(result.transmitted_fast) + _normal_poynting_flux(result.transmitted_slow)
    if incident_flux <= 0:
        raise ValueError("Incident normal Poynting flux must be positive.")
    return max(0.0, transmitted_flux / incident_flux)
