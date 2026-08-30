from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .config import MultilayerSystem, with_sample_azimuth_deg
from .anisotropic import ShaarpEigenBasis, make_shaarp_eigen_basis
from .multilayer_basis import build_multilayer_2omega_basis, build_multilayer_omega_basis, build_multilayer_omega_basis_jones
from .multilayer_boundary import (
    MultilayerBoundarySolution,
    solve_multilayer_boundary,
    solve_multilayer_boundary_single_pass,
)
from .nonlinear import (
    InhomogeneousField,
    MultilayerInhomogeneousFields,
    MultilayerSources,
    multilayer_sources,
    solve_multilayer_inhomogeneous_fields,
)
from .tensors import chi2_to_voigt, rotate_rank2_crystal_to_lab, rotate_rank3_crystal_to_lab, voigt_to_chi2
from .waves import EPS0, MU0, Wave


@dataclass(frozen=True)
class MultilayerSHGBoundaryResult:
    """2 omega multilayer boundary solve with known inhomogeneous layer waves."""

    layer_known: list[list[Wave]]
    boundary: MultilayerBoundarySolution

    @property
    def reflected_2omega(self) -> list[Wave]:
        return self.boundary.top_unknown

    @property
    def layer_homogeneous_2omega(self) -> list[list[Wave]]:
        return self.boundary.layer_unknowns

    @property
    def substrate_2omega(self) -> list[Wave]:
        return self.boundary.substrate_unknown

    @property
    def coefficients(self) -> np.ndarray:
        return self.boundary.coefficients

    @property
    def residual(self) -> np.ndarray:
        return self.boundary.residual


@dataclass(frozen=True)
class MultilayerSHGWorkflowResult:
    """Omega boundary solve plus SHAARP.ml ten-source 2 omega boundary solve."""

    fundamental: MultilayerBoundarySolution
    layer_sources: list[MultilayerSources]
    layer_inhomogeneous_fields: list[MultilayerInhomogeneousFields]
    shg: MultilayerSHGBoundaryResult


@dataclass(frozen=True)
class MultilayerSHGSystemSweepResult:
    """Angle sweep of the system-level staged multilayer SHG workflow."""

    theta_deg: np.ndarray
    results: list[MultilayerSHGWorkflowResult]
    fundamental_residual_norm: np.ndarray
    shg_residual_norm: np.ndarray
    reflected_2omega_field_norm2: np.ndarray


@dataclass(frozen=True)
class MultilayerSHGPolarimetrySweepResult:
    """Broadcast sweep over staged multilayer polarimetry settings."""

    theta_deg: np.ndarray
    phi_deg: np.ndarray
    psi_deg: np.ndarray
    ellipticity_deg: np.ndarray
    shape: tuple[int, ...]
    results: list[MultilayerSHGWorkflowResult]
    analyzer_amplitude: np.ndarray
    intensity: np.ndarray
    fundamental_residual_norm: np.ndarray
    shg_residual_norm: np.ndarray


@dataclass(frozen=True)
class MultilayerSHGSampleAzimuthSweepResult:
    """Broadcast sweep over staged multilayer sample azimuth settings."""

    sample_azimuth_deg: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    psi_deg: np.ndarray
    ellipticity_deg: np.ndarray
    shape: tuple[int, ...]
    results: list[MultilayerSHGWorkflowResult]
    analyzer_amplitude: np.ndarray
    intensity: np.ndarray
    reflected_parallel_amplitude: np.ndarray
    reflected_perpendicular_amplitude: np.ndarray
    transmitted_parallel_amplitude: np.ndarray
    transmitted_perpendicular_amplitude: np.ndarray
    reflected_parallel_intensity: np.ndarray
    reflected_perpendicular_intensity: np.ndarray
    transmitted_parallel_intensity: np.ndarray
    transmitted_perpendicular_intensity: np.ndarray
    fundamental_residual_norm: np.ndarray
    shg_residual_norm: np.ndarray


@dataclass(frozen=True)
class MultilayerMakerFringesSweepResult:
    """SHAARP.ml-style Maker fringes transmitted 2omega angle scan.

    `list_mf_para` and `list_mf_perp` preserve the GUI Copy payload shape:
    two columns `{theta_deg, intensity}` for analyzer angles `psi` and
    `psi + 90 deg`. The analyzer convention is staged and must still be
    checked against Mathematica `MFList` values.
    """

    theta_deg: np.ndarray
    parallel_amplitude: np.ndarray
    perpendicular_amplitude: np.ndarray
    parallel_intensity: np.ndarray
    perpendicular_intensity: np.ndarray
    selected_transmitted_wave_count: int
    results: list[MultilayerSHGWorkflowResult]
    fundamental_residual_norm: np.ndarray
    shg_residual_norm: np.ndarray

    @property
    def list_mf_para(self) -> np.ndarray:
        return np.column_stack([self.theta_deg, self.parallel_intensity])

    @property
    def list_mf_perp(self) -> np.ndarray:
        return np.column_stack([self.theta_deg, self.perpendicular_intensity])

    def shaarp_ml_copy_lists(self) -> list[np.ndarray]:
        return [self.list_mf_para, self.list_mf_perp]


def _warn_if_units_break_dispersion(mu: float, eps0: float) -> None:
    """The multilayer bases are built with the c=1 convention (omega = 2*pi/lambda_um, k = n*omega),
    so the inhomogeneous operator curl_curl(k) - omega^2*mu*eps0*eps is dispersion-CONSISTENT only
    when mu*eps0 == 1 (natural units -- the convention of every live-Mathematica validation).
    Physical SI constants make the eps term ~1e17 smaller than curl_curl, leaving a near-singular
    operator whose per-angle branchy solves produced the 'telegraph' artifact (field hopping
    between exact E/n fractions; root-caused). Warn loudly rather than fail, so
    exploratory use stays possible but never silent."""

    try:
        product = float(abs(complex(mu) * complex(eps0)))
    except (TypeError, ValueError):
        return
    if not (0.99 <= product <= 1.01):
        import warnings

        warnings.warn(
            "Multilayer SHG with mu*eps0 != 1 is dispersion-INCONSISTENT: the layer bases use "
            "k = n*omega with omega = 2*pi/lambda (c=1), so curl_curl(k) = omega^2*mu*eps0*(eps E + P) "
            f"is physical only at natural units (mu = eps0 = 1). Got mu*eps0 = {mu * eps0:.3e}; "
            "results will be numerically unstable (telegraph artifacts). Use mu=1, eps0=1 -- the "
            "live-Mathematica-validated convention.",
            RuntimeWarning,
            stacklevel=3,
        )


def solve_multilayer_shg_boundary(
    *,
    reflected_basis_2omega: list[Wave],
    layer_homogeneous_basis_2omega: list[list[Wave]],
    substrate_basis_2omega: list[Wave],
    layer_inhomogeneous_2omega: list[list[Wave]],
    thicknesses: list[float],
    mu: float = 1.0,
    single_pass: bool = False,
) -> MultilayerSHGBoundaryResult:
    """Solve the SH boundary system for known inhomogeneous layer sources.

    This is the next building block toward the full SHAARP.ml nonlinear
    workflow. It assumes the inhomogeneous 2 omega source waves inside each
    layer have already been computed, then solves for reflected, homogeneous
    in-layer, and substrate 2 omega amplitudes.
    """

    if len(layer_homogeneous_basis_2omega) != len(layer_inhomogeneous_2omega):
        raise ValueError("layer_homogeneous_basis_2omega and layer_inhomogeneous_2omega must have the same length.")
    if single_pass:
        # JK: SHAARP f4NL sequential single-pass 2 omega SHG solve (no writeback at 2 omega).
        boundary = solve_multilayer_boundary_single_pass(
            top_known=[],
            layer_known=layer_inhomogeneous_2omega,
            top_unknown_basis=reflected_basis_2omega,
            layer_unknown_basis=layer_homogeneous_basis_2omega,
            substrate_unknown_basis=substrate_basis_2omega,
            thicknesses=thicknesses,
            mu=mu,
            writeback=False,
        )
    else:
        boundary = solve_multilayer_boundary(
            top_known=[],
            layer_known=layer_inhomogeneous_2omega,
            top_unknown_basis=reflected_basis_2omega,
            layer_unknown_basis=layer_homogeneous_basis_2omega,
            substrate_unknown_basis=substrate_basis_2omega,
            thicknesses=thicknesses,
            mu=mu,
        )
    return MultilayerSHGBoundaryResult(
        layer_known=layer_inhomogeneous_2omega,
        boundary=boundary,
    )


def solve_multilayer_shg_from_fundamental(
    *,
    incident_omega: list[Wave],
    reflected_basis_omega: list[Wave],
    layer_basis_omega: list[list[Wave]],
    substrate_basis_omega: list[Wave],
    layer_d_voigt_lab: list[np.ndarray],
    layer_epsilon_2omega_lab: list[np.ndarray],
    reflected_basis_2omega: list[Wave],
    layer_homogeneous_basis_2omega: list[list[Wave]],
    substrate_basis_2omega: list[Wave],
    thicknesses: list[float],
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
) -> MultilayerSHGWorkflowResult:
    """Run the current automatic SHAARP.ml-style numeric building-block chain.

    Each omega layer basis must be ordered `[F1, F2, B1, B2]`. The solved
    layer amplitudes are used to construct the ten nonlinear source waves for
    that layer before the 2 omega boundary system is solved.
    """

    layer_count = len(layer_basis_omega)
    if not (
        len(layer_d_voigt_lab)
        == len(layer_epsilon_2omega_lab)
        == len(layer_homogeneous_basis_2omega)
        == len(thicknesses)
        == layer_count
    ):
        raise ValueError("Layer omega basis, nonlinear tensors, 2omega epsilon, 2omega basis, and thicknesses must align.")
    if any(len(layer) != 4 for layer in layer_basis_omega):
        raise ValueError("Each omega layer basis must contain four waves ordered [F1, F2, B1, B2].")
    _warn_if_units_break_dispersion(mu, eps0)

    if single_pass_omega:
        fundamental = solve_multilayer_boundary_single_pass(
            top_known=incident_omega,
            top_unknown_basis=reflected_basis_omega,
            layer_unknown_basis=layer_basis_omega,
            substrate_unknown_basis=substrate_basis_omega,
            thicknesses=thicknesses,
            mu=mu,
            writeback=single_pass_omega_writeback,
        )
    else:
        fundamental = solve_multilayer_boundary(
            top_known=incident_omega,
            top_unknown_basis=reflected_basis_omega,
            layer_unknown_basis=layer_basis_omega,
            substrate_unknown_basis=substrate_basis_omega,
            thicknesses=thicknesses,
            mu=mu,
        )

    layer_sources = []
    layer_fields = []
    layer_known_2omega = []
    for d_voigt_lab, epsilon_2omega_lab, solved_layer in zip(
        layer_d_voigt_lab,
        layer_epsilon_2omega_lab,
        fundamental.layer_unknowns,
    ):
        f1, f2, b1, b2 = solved_layer
        sources = multilayer_sources(d_voigt_lab, f1, f2, b1, b2)
        fields = solve_multilayer_inhomogeneous_fields(
            epsilon_2omega_lab,
            sources,
            mu=mu,
            eps0=eps0,
            condition_threshold=condition_threshold,
            solution_policy=inhomogeneous_solution_policy,
        )
        layer_sources.append(sources)
        layer_fields.append(fields)
        layer_known_2omega.append([_field_to_wave(field) for field in _fields_for_source_policy(fields, inhomogeneous_source_policy)])

    shg = solve_multilayer_shg_boundary(
        reflected_basis_2omega=reflected_basis_2omega,
        layer_homogeneous_basis_2omega=layer_homogeneous_basis_2omega,
        substrate_basis_2omega=substrate_basis_2omega,
        layer_inhomogeneous_2omega=layer_known_2omega,
        thicknesses=thicknesses,
        mu=mu,
        single_pass=single_pass_2omega,
    )
    return MultilayerSHGWorkflowResult(
        fundamental=fundamental,
        layer_sources=layer_sources,
        layer_inhomogeneous_fields=layer_fields,
        shg=shg,
    )


def solve_multilayer_shg_from_tensors(
    *,
    incident_index_omega: complex,
    top_index_2omega: complex,
    incident_theta_rad: float,
    incident_polarization: str,
    layer_epsilon_omega_lab: list[np.ndarray],
    substrate_epsilon_omega_lab: np.ndarray,
    layer_d_voigt_lab: list[np.ndarray],
    layer_epsilon_2omega_lab: list[np.ndarray],
    substrate_epsilon_2omega_lab: np.ndarray,
    thicknesses: list[float],
    omega: float = 1.0,
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
    layer_shaarp_basis_omega: list[ShaarpEigenBasis | None] | None = None,
    substrate_shaarp_basis_omega: ShaarpEigenBasis | None = None,
    layer_shaarp_basis_2omega: list[ShaarpEigenBasis | None] | None = None,
    substrate_shaarp_basis_2omega: ShaarpEigenBasis | None = None,
) -> MultilayerSHGWorkflowResult:
    """Build homogeneous bases from tensors, then run the SHG workflow.

    Current scope: isotropic incident/reflected top medium, anisotropic
    lab-frame layer/substrate tensors, and supplied layer SHG tensors. The
    2 omega bases use the SHG tangential target
    `incident_index_omega * sin(incident_theta_rad)`.
    """

    omega_basis = build_multilayer_omega_basis(
        incident_index=incident_index_omega,
        incident_theta_rad=incident_theta_rad,
        incident_polarization=incident_polarization,
        layer_epsilon_lab=layer_epsilon_omega_lab,
        substrate_epsilon_lab=substrate_epsilon_omega_lab,
        omega=omega,
        layer_shaarp_basis=layer_shaarp_basis_omega,
        substrate_shaarp_basis=substrate_shaarp_basis_omega,
    )
    omega_2_basis = build_multilayer_2omega_basis(
        top_index_2omega=top_index_2omega,
        tangential_index_omega=incident_index_omega,
        incident_theta_rad=incident_theta_rad,
        layer_epsilon_2omega_lab=layer_epsilon_2omega_lab,
        substrate_epsilon_2omega_lab=substrate_epsilon_2omega_lab,
        omega_2=2 * omega,
        layer_shaarp_basis_2omega=layer_shaarp_basis_2omega,
        substrate_shaarp_basis_2omega=substrate_shaarp_basis_2omega,
    )
    return solve_multilayer_shg_from_fundamental(
        incident_omega=omega_basis.incident,
        reflected_basis_omega=omega_basis.reflected_basis,
        layer_basis_omega=omega_basis.layer_basis,
        substrate_basis_omega=omega_basis.substrate_basis,
        layer_d_voigt_lab=layer_d_voigt_lab,
        layer_epsilon_2omega_lab=layer_epsilon_2omega_lab,
        reflected_basis_2omega=omega_2_basis.reflected_basis,
        layer_homogeneous_basis_2omega=omega_2_basis.layer_basis,
        substrate_basis_2omega=omega_2_basis.substrate_basis,
        thicknesses=thicknesses,
        mu=mu,
        eps0=eps0,
        condition_threshold=condition_threshold,
        inhomogeneous_source_policy=inhomogeneous_source_policy,
        inhomogeneous_solution_policy=inhomogeneous_solution_policy,
        single_pass_omega=single_pass_omega,
        single_pass_omega_writeback=single_pass_omega_writeback,
        single_pass_2omega=single_pass_2omega,
    )


def solve_multilayer_shg_from_tensors_jones(
    *,
    incident_index_omega: complex,
    top_index_2omega: complex,
    incident_theta_rad: float,
    incident_jones_sp: tuple[complex, complex],
    layer_epsilon_omega_lab: list[np.ndarray],
    substrate_epsilon_omega_lab: np.ndarray,
    layer_d_voigt_lab: list[np.ndarray],
    layer_epsilon_2omega_lab: list[np.ndarray],
    substrate_epsilon_2omega_lab: np.ndarray,
    thicknesses: list[float],
    omega: float = 1.0,
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
    layer_shaarp_basis_omega: list[ShaarpEigenBasis | None] | None = None,
    substrate_shaarp_basis_omega: ShaarpEigenBasis | None = None,
    layer_shaarp_basis_2omega: list[ShaarpEigenBasis | None] | None = None,
    substrate_shaarp_basis_2omega: ShaarpEigenBasis | None = None,
) -> MultilayerSHGWorkflowResult:
    """Tensor workflow with arbitrary incident Jones amplitudes `(s, p)`."""

    omega_basis = build_multilayer_omega_basis_jones(
        incident_index=incident_index_omega,
        incident_theta_rad=incident_theta_rad,
        incident_jones_sp=incident_jones_sp,
        layer_epsilon_lab=layer_epsilon_omega_lab,
        substrate_epsilon_lab=substrate_epsilon_omega_lab,
        omega=omega,
        layer_shaarp_basis=layer_shaarp_basis_omega,
        substrate_shaarp_basis=substrate_shaarp_basis_omega,
    )
    omega_2_basis = build_multilayer_2omega_basis(
        top_index_2omega=top_index_2omega,
        tangential_index_omega=incident_index_omega,
        incident_theta_rad=incident_theta_rad,
        layer_epsilon_2omega_lab=layer_epsilon_2omega_lab,
        substrate_epsilon_2omega_lab=substrate_epsilon_2omega_lab,
        omega_2=2 * omega,
        layer_shaarp_basis_2omega=layer_shaarp_basis_2omega,
        substrate_shaarp_basis_2omega=substrate_shaarp_basis_2omega,
    )
    return solve_multilayer_shg_from_fundamental(
        incident_omega=omega_basis.incident,
        reflected_basis_omega=omega_basis.reflected_basis,
        layer_basis_omega=omega_basis.layer_basis,
        substrate_basis_omega=omega_basis.substrate_basis,
        layer_d_voigt_lab=layer_d_voigt_lab,
        layer_epsilon_2omega_lab=layer_epsilon_2omega_lab,
        reflected_basis_2omega=omega_2_basis.reflected_basis,
        layer_homogeneous_basis_2omega=omega_2_basis.layer_basis,
        substrate_basis_2omega=omega_2_basis.substrate_basis,
        thicknesses=thicknesses,
        mu=mu,
        eps0=eps0,
        condition_threshold=condition_threshold,
        inhomogeneous_source_policy=inhomogeneous_source_policy,
        inhomogeneous_solution_policy=inhomogeneous_solution_policy,
        single_pass_omega=single_pass_omega,
        single_pass_omega_writeback=single_pass_omega_writeback,
        single_pass_2omega=single_pass_2omega,
    )


def solve_multilayer_shg_from_system(
    system: MultilayerSystem,
    *,
    incident_polarization: str = "p",
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
) -> MultilayerSHGWorkflowResult:
    """Run the tensor workflow from a `MultilayerSystem` configuration.

    Current scope is deliberately narrow: the first/top medium must be
    isotropic at omega and 2omega, and at least one internal layer is required.
    Internal layer and substrate material tensors are rotated to the lab frame
    using each material's configured orientation. Thicknesses are interpreted
    in microns with `omega = 2*pi / wavelength_um`.
    """

    setup = _system_setup(system)
    return solve_multilayer_shg_from_tensors(
        incident_index_omega=setup["top_index_omega"],
        top_index_2omega=setup["top_index_2omega"],
        incident_theta_rad=setup["incident_theta_rad"],
        incident_polarization=incident_polarization,
        layer_epsilon_omega_lab=setup["layer_epsilon_omega_lab"],
        substrate_epsilon_omega_lab=setup["substrate_epsilon_omega_lab"],
        layer_d_voigt_lab=setup["layer_d_voigt_lab"],
        layer_epsilon_2omega_lab=setup["layer_epsilon_2omega_lab"],
        substrate_epsilon_2omega_lab=setup["substrate_epsilon_2omega_lab"],
        thicknesses=setup["thicknesses"],
        omega=setup["omega"],
        mu=mu,
        eps0=eps0,
        condition_threshold=condition_threshold,
        inhomogeneous_source_policy=inhomogeneous_source_policy,
        inhomogeneous_solution_policy=inhomogeneous_solution_policy,
        single_pass_omega=single_pass_omega,
        single_pass_omega_writeback=single_pass_omega_writeback,
        single_pass_2omega=single_pass_2omega,
        layer_shaarp_basis_omega=setup["layer_shaarp_basis_omega"],
        substrate_shaarp_basis_omega=setup["substrate_shaarp_basis_omega"],
        layer_shaarp_basis_2omega=setup["layer_shaarp_basis_2omega"],
        substrate_shaarp_basis_2omega=setup["substrate_shaarp_basis_2omega"],
    )


def solve_multilayer_shg_from_system_jones(
    system: MultilayerSystem,
    *,
    incident_jones_sp: tuple[complex, complex],
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
    setup: dict | None = None,
) -> MultilayerSHGWorkflowResult:
    """Run the system workflow with explicit incident Jones amplitudes `(s, p)`.

    ``setup`` (optional) is a precomputed ``_system_setup(system)`` dict. The
    ONLY angle-dependent entry is ``incident_theta_rad``; everything else (eps
    tensors, eigen-bases, d-tensor, indices, thicknesses, omega) is angle-
    independent. Incidence-angle sweeps may therefore compute ``_system_setup``
    ONCE and pass a copy with ``incident_theta_rad`` overridden per angle -- this
    is byte-identical to recomputing it (the downstream tensor solver receives
    every quantity, including the angle, as explicit kwargs and never reads
    ``system``). Defaults to recomputing it (exact prior behavior)."""

    if setup is None:
        setup = _system_setup(system)
    return solve_multilayer_shg_from_tensors_jones(
        incident_index_omega=setup["top_index_omega"],
        top_index_2omega=setup["top_index_2omega"],
        incident_theta_rad=setup["incident_theta_rad"],
        incident_jones_sp=incident_jones_sp,
        layer_epsilon_omega_lab=setup["layer_epsilon_omega_lab"],
        substrate_epsilon_omega_lab=setup["substrate_epsilon_omega_lab"],
        layer_d_voigt_lab=setup["layer_d_voigt_lab"],
        layer_epsilon_2omega_lab=setup["layer_epsilon_2omega_lab"],
        substrate_epsilon_2omega_lab=setup["substrate_epsilon_2omega_lab"],
        thicknesses=setup["thicknesses"],
        omega=setup["omega"],
        mu=mu,
        eps0=eps0,
        condition_threshold=condition_threshold,
        inhomogeneous_source_policy=inhomogeneous_source_policy,
        inhomogeneous_solution_policy=inhomogeneous_solution_policy,
        single_pass_omega=single_pass_omega,
        single_pass_omega_writeback=single_pass_omega_writeback,
        single_pass_2omega=single_pass_2omega,
        layer_shaarp_basis_omega=setup["layer_shaarp_basis_omega"],
        substrate_shaarp_basis_omega=setup["substrate_shaarp_basis_omega"],
        layer_shaarp_basis_2omega=setup["layer_shaarp_basis_2omega"],
        substrate_shaarp_basis_2omega=setup["substrate_shaarp_basis_2omega"],
    )


def solve_multilayer_shg_from_system_polarimetry(
    system: MultilayerSystem,
    *,
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
    setup: dict | None = None,
) -> MultilayerSHGWorkflowResult:
    """Run the system workflow using scalar `phi_deg` and `ellipticity_deg`.

    This follows the current Python polarimetry convention:
    `s = sin(phi) exp(i ellipticity)`, `p = cos(phi)`.

    ``setup`` (optional) forwards a precomputed ``_system_setup`` to the Jones
    solver -- see that function's note. Used by incidence-angle sweeps to avoid
    recomputing the angle-independent setup at every angle (byte-identical).
    """

    return solve_multilayer_shg_from_system_jones(
        system,
        incident_jones_sp=incident_jones_from_polarimetry(system),
        mu=mu,
        eps0=eps0,
        condition_threshold=condition_threshold,
        inhomogeneous_source_policy=inhomogeneous_source_policy,
        inhomogeneous_solution_policy=inhomogeneous_solution_policy,
        single_pass_omega=single_pass_omega,
        single_pass_omega_writeback=single_pass_omega_writeback,
        single_pass_2omega=single_pass_2omega,
        setup=setup,
    )


def solve_multilayer_shg_system_sweep(
    system: MultilayerSystem,
    *,
    theta_deg: np.ndarray | list[float] | tuple[float, ...] | None = None,
    incident_polarization: str = "p",
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
) -> MultilayerSHGSystemSweepResult:
    """Run `solve_multilayer_shg_from_system` over incidence angles."""

    if theta_deg is None:
        values = np.atleast_1d(np.asarray(system.polarimetry.theta_deg, dtype=float))
    else:
        values = np.atleast_1d(np.asarray(theta_deg, dtype=float))
    if values.ndim != 1:
        raise ValueError("theta_deg must be a scalar or one-dimensional sequence.")

    results = []
    for theta in values:
        pol = replace(system.polarimetry, theta_deg=float(theta))
        theta_system = replace(system, polarimetry=pol)
        results.append(
            solve_multilayer_shg_from_system(
                theta_system,
                incident_polarization=incident_polarization,
                mu=mu,
                eps0=eps0,
                condition_threshold=condition_threshold,
                inhomogeneous_solution_policy=inhomogeneous_solution_policy,
                single_pass_omega=single_pass_omega,
                single_pass_omega_writeback=single_pass_omega_writeback,
                single_pass_2omega=single_pass_2omega,
            )
        )

    return MultilayerSHGSystemSweepResult(
        theta_deg=values.astype(float),
        results=results,
        fundamental_residual_norm=np.array([np.linalg.norm(result.fundamental.residual) for result in results]),
        shg_residual_norm=np.array([np.linalg.norm(result.shg.residual) for result in results]),
        reflected_2omega_field_norm2=np.array(
            [
                sum(float(np.real(np.vdot(wave.electric, wave.electric))) for wave in result.shg.reflected_2omega)
                for result in results
            ]
        ),
    )


def solve_multilayer_shg_polarimetry_sweep(
    system: MultilayerSystem,
    *,
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
) -> MultilayerSHGPolarimetrySweepResult:
    """Run the staged workflow over broadcast polarimetry arrays.

    ``inhomogeneous_source_policy`` selects which inhomogeneous source waves are
    carried (default ``"all"``). To reproduce the original Mathematica SHAARP.ml
    ``SampleRotate`` (forward-only sources), pass ``"forward_only"`` -- the solve
    then matches live SHAARP.ml to ~2e-14 (see
    ``tests/test_polarimetry_reference_comparison.py``)."""

    theta, phi, psi, ellipticity = _broadcast_polarimetry_degrees(system)
    results = []
    amplitudes = []
    intensities = []
    fundamental_residuals = []
    shg_residuals = []

    for idx in np.ndindex(theta.shape):
        pol = replace(
            system.polarimetry,
            theta_deg=float(theta[idx]),
            phi_deg=float(phi[idx]),
            psi_deg=float(psi[idx]),
            ellipticity_deg=float(ellipticity[idx]),
        )
        point_system = replace(system, polarimetry=pol)
        result = solve_multilayer_shg_from_system_polarimetry(
            point_system,
            mu=mu,
            eps0=eps0,
            condition_threshold=condition_threshold,
            inhomogeneous_source_policy=inhomogeneous_source_policy,
            inhomogeneous_solution_policy=inhomogeneous_solution_policy,
            single_pass_omega=single_pass_omega,
            single_pass_omega_writeback=single_pass_omega_writeback,
            single_pass_2omega=single_pass_2omega,
        )
        amplitude, intensity = analyze_reflected_2omega_polarimetry(result.shg, point_system)
        results.append(result)
        amplitudes.append(amplitude)
        intensities.append(intensity)
        fundamental_residuals.append(np.linalg.norm(result.fundamental.residual))
        shg_residuals.append(np.linalg.norm(result.shg.residual))

    return MultilayerSHGPolarimetrySweepResult(
        theta_deg=theta,
        phi_deg=phi,
        psi_deg=psi,
        ellipticity_deg=ellipticity,
        shape=theta.shape,
        results=results,
        analyzer_amplitude=np.asarray(amplitudes, dtype=complex).reshape(theta.shape),
        intensity=np.asarray(intensities, dtype=float).reshape(theta.shape),
        fundamental_residual_norm=np.asarray(fundamental_residuals, dtype=float).reshape(theta.shape),
        shg_residual_norm=np.asarray(shg_residuals, dtype=float).reshape(theta.shape),
    )


def solve_multilayer_shg_sample_azimuth_sweep(
    system: MultilayerSystem,
    *,
    sample_azimuth_deg: np.ndarray | list[float] | tuple[float, ...] | float,
    rotate_top: bool = False,
    rotate_substrate: bool = False,
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "forward_only",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
    mrassumption: int = 0,
) -> MultilayerSHGSampleAzimuthSweepResult:
    """Sweep physical sample azimuth separately from polarizer/analyzer angles.

    SHAARP.ml `SampleRotate` follows the same effective f4NL source selection
    as the validated Maker path: only forward inhomogeneous source waves enter
    the nonlinear boundary solve. Use `inhomogeneous_source_policy="all"` for
    the broader Python diagnostic/physical-sum variant.

    `mrassumption` is a convenience selector mirroring the Maker sweep: 0=full,
    1=JK (single-pass omega no-writeback + single-pass 2 omega), 2=HH (single-pass
    omega writeback + full 2 omega). It maps onto the single_pass_* flags below;
    validated vs live SHAARP.ml SampleRotate (4-layer azimuth scan) to ~4.7e-15 in
    tests/test_jkhh_samplerotate_agreement.py.
    """

    if mrassumption not in (0, 1, 2):
        raise ValueError("mrassumption must be 0 (full), 1 (JK), or 2 (HH)")
    if mrassumption == 2:  # HH
        single_pass_omega = True
    elif mrassumption == 1:  # JK
        single_pass_omega = True
        single_pass_omega_writeback = False
        single_pass_2omega = True

    pol = system.polarimetry
    try:
        azimuth, theta, phi, psi, ellipticity = np.broadcast_arrays(
            np.asarray(sample_azimuth_deg, dtype=float),
            np.asarray(pol.theta_deg, dtype=float),
            np.asarray(pol.phi_deg, dtype=float),
            np.asarray(pol.psi_deg, dtype=float),
            np.asarray(pol.ellipticity_deg, dtype=float),
        )
    except ValueError as exc:
        raise ValueError("Sample azimuth and polarimetry arrays must be broadcast-compatible.") from exc

    results = []
    reflected_parallel_amplitudes = []
    reflected_perpendicular_amplitudes = []
    transmitted_parallel_amplitudes = []
    transmitted_perpendicular_amplitudes = []
    reflected_parallel_intensities = []
    reflected_perpendicular_intensities = []
    transmitted_parallel_intensities = []
    transmitted_perpendicular_intensities = []
    fundamental_residuals = []
    shg_residuals = []

    for idx in np.ndindex(azimuth.shape):
        pol_i = replace(
            pol,
            theta_deg=float(theta[idx]),
            phi_deg=float(phi[idx]),
            psi_deg=float(psi[idx]),
            ellipticity_deg=float(ellipticity[idx]),
        )
        point_system = replace(system, polarimetry=pol_i)
        rotated_system = with_sample_azimuth_deg(
            point_system,
            float(azimuth[idx]),
            rotate_top=rotate_top,
            rotate_substrate=rotate_substrate,
        )
        result = solve_multilayer_shg_from_system_polarimetry(
            rotated_system,
            mu=mu,
            eps0=eps0,
            condition_threshold=condition_threshold,
            inhomogeneous_source_policy=inhomogeneous_source_policy,
            inhomogeneous_solution_policy=inhomogeneous_solution_policy,
            single_pass_omega=single_pass_omega,
            single_pass_omega_writeback=single_pass_omega_writeback,
            single_pass_2omega=single_pass_2omega,
        )
        r_para, r_perp = sample_rotate_reflected_2omega_amplitudes(result.shg, float(psi[idx]))
        t_para, t_perp = sample_rotate_transmitted_2omega_amplitudes(result.shg, float(psi[idx]))
        results.append(result)
        reflected_parallel_amplitudes.append(r_para)
        reflected_perpendicular_amplitudes.append(r_perp)
        transmitted_parallel_amplitudes.append(t_para)
        transmitted_perpendicular_amplitudes.append(t_perp)
        reflected_parallel_intensities.append(float(abs(r_para) ** 2))
        reflected_perpendicular_intensities.append(float(abs(r_perp) ** 2))
        transmitted_parallel_intensities.append(float(abs(t_para) ** 2))
        transmitted_perpendicular_intensities.append(float(abs(t_perp) ** 2))
        fundamental_residuals.append(np.linalg.norm(result.fundamental.residual))
        shg_residuals.append(np.linalg.norm(result.shg.residual))

    r_para_amp = np.asarray(reflected_parallel_amplitudes, dtype=complex).reshape(azimuth.shape)
    r_para_intensity = np.asarray(reflected_parallel_intensities, dtype=float).reshape(azimuth.shape)
    return MultilayerSHGSampleAzimuthSweepResult(
        sample_azimuth_deg=azimuth,
        theta_deg=theta,
        phi_deg=phi,
        psi_deg=psi,
        ellipticity_deg=ellipticity,
        shape=azimuth.shape,
        results=results,
        analyzer_amplitude=r_para_amp,
        intensity=r_para_intensity,
        reflected_parallel_amplitude=r_para_amp,
        reflected_perpendicular_amplitude=np.asarray(reflected_perpendicular_amplitudes, dtype=complex).reshape(azimuth.shape),
        transmitted_parallel_amplitude=np.asarray(transmitted_parallel_amplitudes, dtype=complex).reshape(azimuth.shape),
        transmitted_perpendicular_amplitude=np.asarray(transmitted_perpendicular_amplitudes, dtype=complex).reshape(azimuth.shape),
        reflected_parallel_intensity=r_para_intensity,
        reflected_perpendicular_intensity=np.asarray(reflected_perpendicular_intensities, dtype=float).reshape(azimuth.shape),
        transmitted_parallel_intensity=np.asarray(transmitted_parallel_intensities, dtype=float).reshape(azimuth.shape),
        transmitted_perpendicular_intensity=np.asarray(transmitted_perpendicular_intensities, dtype=float).reshape(azimuth.shape),
        fundamental_residual_norm=np.asarray(fundamental_residuals, dtype=float).reshape(azimuth.shape),
        shg_residual_norm=np.asarray(shg_residuals, dtype=float).reshape(azimuth.shape),
    )


def solve_multilayer_maker_fringes_sweep(
    system: MultilayerSystem,
    *,
    theta_deg: Sequence[float] | None = None,
    theta_min_deg: float = -45.0,
    theta_max_deg: float = 45.0,
    theta_step_deg: float = 1.0,
    mu: float = 1.0,
    eps0: float = 1.0,
    condition_threshold: float = 1e12,
    inhomogeneous_source_policy: str = "forward_only",
    inhomogeneous_solution_policy: str = "solve",
    single_pass_omega: bool = False,
    single_pass_omega_writeback: bool = True,
    single_pass_2omega: bool = False,
    mrassumption: int = 0,
    transmitted_wave_policy: str = "shaarp_ml_selected",
) -> MultilayerMakerFringesSweepResult:
    """Run a staged SHAARP.ml Maker fringes incidence-angle scan.

    SHAARP.ml builds `MFList = MF[mats, theta, mrassumption] /@ Range[...]`
    and copies `listMFpara` / `listMFperp` as transmitted 2omega intensities
    at analyzer angles `psi` and `psi + 90 deg`. This function mirrors that
    first-class output shape using the current Python multilayer workflow.
    It is Mathematica `MFList` validated: live SHAARP.ml agreement to ~1e-15
    across multilayer depth 4-9 layers including the 739-angle 0.1-deg fine grid
    and the docs quartz+Au case (see VALIDATION.md).

    `mrassumption` selects the SHAARP.ml MF assumption mode: 0 = full (default),
    1 = JK (Jerphagnon-Kurtz) and 2 = HH (Herman-Hayden) -- both implemented via
    the sequential single-pass SHG assembly and comprehensively validated vs live
    SHAARP.ml (~5e-15, both entry points, depth-bracketed 4/5/9-layer).
    """

    if mrassumption not in (0, 1, 2):
        raise ValueError("mrassumption must be 0 (full), 1 (JK), or 2 (HH)")
    if mrassumption == 2:  # HH: single-pass omega fundamental (writeback), full 2 omega SHG
        single_pass_omega = True
    elif mrassumption == 1:  # JK: single-pass omega (no writeback) + single-pass 2 omega SHG
        single_pass_omega = True
        single_pass_omega_writeback = False
        single_pass_2omega = True

    theta_values = _maker_theta_grid(theta_deg, theta_step_deg, theta_min_deg, theta_max_deg)
    results = []
    parallel_amplitudes = []
    perpendicular_amplitudes = []
    parallel_intensities = []
    perpendicular_intensities = []
    fundamental_residuals = []
    shg_residuals = []

    # Compute the angle-independent setup ONCE (eps tensors, eigen-bases, d-tensor,
    # indices, thicknesses, omega). Only `incident_theta_rad` changes per angle, so
    # overriding that single entry per angle is byte-identical to recomputing the
    # whole setup -- and avoids the dominant per-angle eigen-decomposition cost, so
    # fine incidence-angle scans (continuous Maker fringes) stay responsive.
    base_setup = _system_setup(system)

    for theta in theta_values:
        pol = replace(system.polarimetry, theta_deg=float(theta))
        point_system = replace(system, polarimetry=pol)
        point_setup = dict(base_setup)
        point_setup["incident_theta_rad"] = float(pol.radians()[0])
        result = solve_multilayer_shg_from_system_polarimetry(
            point_system,
            mu=mu,
            eps0=eps0,
            condition_threshold=condition_threshold,
            inhomogeneous_source_policy=inhomogeneous_source_policy,
            inhomogeneous_solution_policy=inhomogeneous_solution_policy,
            single_pass_omega=single_pass_omega,
            single_pass_omega_writeback=single_pass_omega_writeback,
            single_pass_2omega=single_pass_2omega,
            setup=point_setup,
        )
        transmitted_waves = _transmitted_waves_for_maker_policy(result.shg, transmitted_wave_policy)
        para_amp, para_intensity = analyze_transmitted_2omega_waves(
            transmitted_waves,
            analyzer_jones_from_polarimetry(point_system),
        )
        perp_analyzer = analyzer_jones_from_psi_deg(float(pol.psi_deg) + 90.0)
        perp_amp, perp_intensity = analyze_transmitted_2omega_waves(transmitted_waves, perp_analyzer)
        results.append(result)
        parallel_amplitudes.append(para_amp)
        perpendicular_amplitudes.append(perp_amp)
        parallel_intensities.append(para_intensity)
        perpendicular_intensities.append(perp_intensity)
        fundamental_residuals.append(np.linalg.norm(result.fundamental.residual))
        shg_residuals.append(np.linalg.norm(result.shg.residual))

    return MultilayerMakerFringesSweepResult(
        theta_deg=theta_values,
        parallel_amplitude=np.asarray(parallel_amplitudes, dtype=complex),
        perpendicular_amplitude=np.asarray(perpendicular_amplitudes, dtype=complex),
        parallel_intensity=np.asarray(parallel_intensities, dtype=float),
        perpendicular_intensity=np.asarray(perpendicular_intensities, dtype=float),
        selected_transmitted_wave_count=len(_transmitted_waves_for_maker_policy(results[0].shg, transmitted_wave_policy))
        if results
        else 0,
        results=results,
        fundamental_residual_norm=np.asarray(fundamental_residuals, dtype=float),
        shg_residual_norm=np.asarray(shg_residuals, dtype=float),
    )


def incident_jones_from_polarimetry(system: MultilayerSystem) -> tuple[complex, complex]:
    """Return incident `(s, p)` amplitudes from scalar Python `Polarimetry`."""

    _, phi, _, ellipticity = system.polarimetry.radians()
    phi_array = np.asarray(phi, dtype=float)
    ellipticity_array = np.asarray(ellipticity, dtype=float)
    if phi_array.shape != () or ellipticity_array.shape != ():
        raise ValueError("incident_jones_from_polarimetry currently requires scalar phi_deg and ellipticity_deg.")
    return (
        complex(np.sin(float(phi_array)) * np.exp(1j * float(ellipticity_array))),
        complex(np.cos(float(phi_array))),
    )


def analyzer_jones_from_polarimetry(system: MultilayerSystem) -> tuple[complex, complex]:
    """Return analyzer `(s, p)` amplitudes from scalar Python `psi_deg`."""

    _, _, psi, _ = system.polarimetry.radians()
    psi_array = np.asarray(psi, dtype=float)
    if psi_array.shape != ():
        raise ValueError("analyzer_jones_from_polarimetry currently requires scalar psi_deg.")
    return analyzer_jones_from_psi_deg(float(np.rad2deg(psi_array)))


def analyzer_jones_from_psi_deg(psi_deg: float) -> tuple[complex, complex]:
    """Return analyzer `(s, p)` amplitudes from a scalar analyzer angle."""

    psi = np.deg2rad(float(psi_deg))
    return (complex(np.sin(psi)), complex(np.cos(psi)))


def reflected_2omega_jones_sp(shg: MultilayerSHGBoundaryResult) -> tuple[complex, complex]:
    """Extract reflected 2omega `(s, p)` amplitudes from solved wave-frame fields.

    SHAARP.ml extends each wave with `A = RotationMatrix[theta].T. EE` and
    sums analyzer components in that per-wave frame. Anisotropic modes can have
    distinct propagation angles, so projecting a summed lab-frame E field onto
    one mode's p direction is not equivalent.
    """

    reflected = shg.reflected_2omega
    if len(reflected) != 2:
        raise ValueError("Expected exactly two reflected 2omega waves ordered [s, p].")
    return _sum_wave_frame_jones_sp(reflected)


def transmitted_2omega_jones_sp(shg: MultilayerSHGBoundaryResult) -> tuple[complex, complex]:
    """Project transmitted substrate 2omega waves onto SHAARP.ml `(s, p)`.

    SHAARP.ml Maker fringes report transmitted SHG. This staged projection
    follows the Mathematica `A` convention wave-by-wave:
    `A = RotationMatrix[theta].T. EE`, then `(s, p) = (A_y, A_x)`.
    """

    transmitted = shg.substrate_2omega
    if len(transmitted) != 2:
        raise ValueError("Expected exactly two transmitted substrate 2omega waves.")
    return _sum_wave_frame_jones_sp(transmitted)


def shaarp_ml_selected_transmitted_2omega_jones_sp(shg: MultilayerSHGBoundaryResult) -> tuple[complex, complex]:
    """Return the transmitted `(s, p)` component used by SHAARP.ml Maker `MF`.

    The Mathematica `MF` path first selects transmitted SHG waves through its
    `wTs/wTp` association logic. For the current anisotropic Maker validation
    cases, that leaves one transmitted nonlinear wave present and one absent.
    This helper intentionally mirrors that SHAARP.ml numerical workflow.
    """

    return _sum_wave_frame_jones_sp(_transmitted_waves_for_maker_policy(shg, "shaarp_ml_selected"))


def analyze_reflected_2omega(
    shg: MultilayerSHGBoundaryResult,
    analyzer_jones_sp: tuple[complex, complex],
) -> tuple[complex, float]:
    """Project reflected 2omega onto analyzer amplitudes `(s, p)`."""

    s_component, p_component = reflected_2omega_jones_sp(shg)
    s_analyzer, p_analyzer = analyzer_jones_sp
    amplitude = s_analyzer * s_component + p_analyzer * p_component
    return amplitude, float(abs(amplitude) ** 2)


def analyze_reflected_2omega_polarimetry(
    shg: MultilayerSHGBoundaryResult,
    system: MultilayerSystem,
) -> tuple[complex, float]:
    """Project reflected 2omega using scalar `psi_deg` from `system`."""

    return analyze_reflected_2omega(shg, analyzer_jones_from_polarimetry(system))


def analyze_transmitted_2omega(
    shg: MultilayerSHGBoundaryResult,
    analyzer_jones_sp: tuple[complex, complex],
) -> tuple[complex, float]:
    """Project transmitted 2omega onto analyzer amplitudes `(s, p)`."""

    s_component, p_component = transmitted_2omega_jones_sp(shg)
    s_analyzer, p_analyzer = analyzer_jones_sp
    amplitude = s_analyzer * s_component + p_analyzer * p_component
    return amplitude, float(abs(amplitude) ** 2)


def analyze_transmitted_2omega_waves(
    waves: list[Wave],
    analyzer_jones_sp: tuple[complex, complex],
) -> tuple[complex, float]:
    """Project selected transmitted waves onto analyzer amplitudes `(s, p)`."""

    s_component, p_component = _sum_wave_frame_jones_sp(waves)
    s_analyzer, p_analyzer = analyzer_jones_sp
    amplitude = s_analyzer * s_component + p_analyzer * p_component
    return amplitude, float(abs(amplitude) ** 2)


def analyze_transmitted_2omega_polarimetry(
    shg: MultilayerSHGBoundaryResult,
    system: MultilayerSystem,
) -> tuple[complex, float]:
    """Project transmitted 2omega using scalar `psi_deg` from `system`."""

    return analyze_transmitted_2omega(shg, analyzer_jones_from_polarimetry(system))


def sample_rotate_reflected_2omega_amplitudes(
    shg: MultilayerSHGBoundaryResult,
    psi_deg: float,
) -> tuple[complex, complex]:
    """Return SHAARP.ml SampleRotate reflected `(parallel, perpendicular)` amplitudes."""

    psi = np.deg2rad(float(psi_deg))
    s_component, p_component = reflected_2omega_jones_sp(shg)
    parallel = np.cos(psi) * p_component - np.sin(psi) * s_component
    perpendicular = np.sin(psi) * p_component + np.cos(psi) * s_component
    return complex(parallel), complex(perpendicular)


def sample_rotate_transmitted_2omega_amplitudes(
    shg: MultilayerSHGBoundaryResult,
    psi_deg: float,
    *,
    transmitted_wave_policy: str = "shaarp_ml_selected",
) -> tuple[complex, complex]:
    """Return SHAARP.ml SampleRotate transmitted `(parallel, perpendicular)` amplitudes."""

    psi = np.deg2rad(float(psi_deg))
    waves = _transmitted_waves_for_maker_policy(shg, transmitted_wave_policy)
    s_component, p_component = _sum_wave_frame_jones_sp(waves)
    parallel = np.cos(psi) * p_component + np.sin(psi) * s_component
    perpendicular = -np.sin(psi) * p_component + np.cos(psi) * s_component
    return complex(parallel), complex(perpendicular)


def _field_to_wave(field: InhomogeneousField) -> Wave:
    return Wave(k=field.k, electric=field.electric, omega=field.omega)


def _fields_for_source_policy(fields: MultilayerInhomogeneousFields, policy: str) -> list[InhomogeneousField]:
    """Select SHAARP.ml `winhAssumption`-style inhomogeneous wave groups."""

    if policy == "forward_only":
        return [fields.f1f1, fields.f2f2, fields.f1f2]
    if policy == "forward_backward":
        return [fields.f1f1, fields.f2f2, fields.f1f2, fields.b1b1, fields.b2b2, fields.b1b2]
    if policy == "all":
        return fields.as_list()
    raise ValueError("inhomogeneous_source_policy must be 'forward_only', 'forward_backward', or 'all'.")


def _transmitted_waves_for_maker_policy(shg: MultilayerSHGBoundaryResult, policy: str) -> list[Wave]:
    if policy == "physical_sum":
        return shg.substrate_2omega
    if policy == "shaarp_ml_selected":
        waves = shg.substrate_2omega
        if not waves:
            return []
        # SHAARP.ml's MF uses ONE specific transmitted wave; Python's slot [0] reproduces that
        # choice on every gated live-Mathematica reference (anisotropic substrates: two DISTINCT
        # modes with stable, meaningful ordering -- an unconditional dominant-amplitude pick
        # breaks 14 of those gates by grabbing the other mode at some angles). The ONE failure of
        # slot [0] is the DEGENERATE case (isotropic substrate): the two transmitted modes share
        # k, the eigen-solve's basis rotation is arbitrary per angle, and at scattered angles the
        # whole physical field hops into slot [1], leaving slot [0] EXACTLY empty -- which read as
        # spurious exact-0 dives in the Maker channels (caught by the continuity lens,
        # live SHAARP.ml is continuous there). So: keep slot [0] -- and ONLY when it
        # is empty while another slot carries the field (the degenerate-swap signature), follow
        # the field. Verified: the 14 gated references stay byte-identical, and the LiNbO3 swap
        # window matches live SHAARP.ml to 3.4e-11 with zero dropouts.
        norms = [float(np.linalg.norm(np.asarray(w.electric))) for w in waves]
        best = max(norms)
        # DEGENERATE substrate pair (isotropic substrate: both transmitted modes share the SAME k).
        # The eigen-solve's basis inside that degenerate subspace is arbitrary PER ANGLE, so the
        # physical field can SPLIT across the two slots by an angle-flickering rotation; slot [0]
        # alone then carries a basis-dependent projection that dives toward zero whenever the
        # arbitrary basis lands perpendicular to the analyzer. That shipped as a comb of spurious
        # near-zero dives in the Maker curves of glass-substrate film systems (near-phase-matched LiNbO3 films showed it worst). Because the two modes share
        # ONE plane wave, their COHERENT SUM is the unique, basis-independent physical field --
        # identical to live SHAARP.ml, whose stable eigen-basis keeps the whole field in one slot.
        # Anisotropic substrates (two DISTINCT k -- every gated live-Mathematica reference) never
        # enter this branch and keep the slot-[0] selection byte-identical.
        if len(waves) == 2:
            k0 = np.asarray(waves[0].k, dtype=complex)
            k1 = np.asarray(waves[1].k, dtype=complex)
            k_scale = max(float(np.linalg.norm(k0)), float(np.linalg.norm(k1)), 1e-30)
            if float(np.linalg.norm(k0 - k1)) <= 1e-9 * k_scale:
                summed = replace(
                    waves[0],
                    electric=np.asarray(waves[0].electric, dtype=complex)
                    + np.asarray(waves[1].electric, dtype=complex),
                )
                return [summed]
        if best > 0.0 and norms[0] <= 1e-9 * best:
            return [waves[int(np.argmax(norms))]]
        return [waves[0]]
    raise ValueError("transmitted_wave_policy must be 'shaarp_ml_selected' or 'physical_sum'.")


def _epsilon_lab(material, *, omega: bool) -> np.ndarray:
    rotation = material.orientation.rotation_matrix()
    tensor = material.eps_w() if omega else material.eps_2w()
    return rotate_rank2_crystal_to_lab(tensor, rotation)


def _d_voigt_lab(material) -> np.ndarray:
    rotation = material.orientation.rotation_matrix()
    return chi2_to_voigt(rotate_rank3_crystal_to_lab(voigt_to_chi2(material.d_voigt()), rotation))


def _isotropic_index(epsilon: np.ndarray, label: str) -> complex:
    eps = np.asarray(epsilon, dtype=complex)
    if eps.shape != (3, 3):
        raise ValueError(f"{label} must be a 3x3 tensor after material parsing.")
    scalar = np.trace(eps) / 3
    if not np.allclose(eps, np.eye(3, dtype=complex) * scalar, atol=1e-10):
        raise ValueError(f"{label} must be isotropic for the current system-level workflow.")
    return np.sqrt(scalar)


def _system_setup(system: MultilayerSystem) -> dict:
    system.validate()
    if len(system.layers) < 3:
        raise ValueError("The system workflow requires top, at least one internal layer, and substrate.")
    theta, _, _, _ = system.polarimetry.radians()
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.shape != ():
        raise ValueError("solve_multilayer_shg_from_system currently requires a scalar theta_deg.")

    top = system.layers[0].material
    internal_layers = system.layers[1:-1]
    substrate = system.layers[-1].material
    return {
        "incident_theta_rad": float(theta_array),
        "top_index_omega": _isotropic_index(top.eps_w(), "top epsilon_omega"),
        "top_index_2omega": _isotropic_index(top.eps_2w(), "top epsilon_2omega"),
        "layer_epsilon_omega_lab": [_epsilon_lab(layer.material, omega=True) for layer in internal_layers],
        "layer_epsilon_2omega_lab": [_epsilon_lab(layer.material, omega=False) for layer in internal_layers],
        "layer_shaarp_basis_omega": [_shaarp_eigen_basis(layer.material, omega=True) for layer in internal_layers],
        "layer_shaarp_basis_2omega": [_shaarp_eigen_basis(layer.material, omega=False) for layer in internal_layers],
        "layer_d_voigt_lab": [
            _d_voigt_lab(layer.material) if layer.shg_active else np.zeros((3, 6), dtype=complex)
            for layer in internal_layers
        ],
        "thicknesses": [float(layer.thickness_um) for layer in internal_layers],
        "omega": 2 * np.pi / system.wavelength_um,
        "substrate_epsilon_omega_lab": _epsilon_lab(substrate, omega=True),
        "substrate_epsilon_2omega_lab": _epsilon_lab(substrate, omega=False),
        "substrate_shaarp_basis_omega": _shaarp_eigen_basis(substrate, omega=True),
        "substrate_shaarp_basis_2omega": _shaarp_eigen_basis(substrate, omega=False),
    }


def _shaarp_eigen_basis(material, *, omega: bool) -> ShaarpEigenBasis:
    return make_shaarp_eigen_basis(
        material.eps_w() if omega else material.eps_2w(),
        material.orientation.rotation_matrix(),
        material.structure.point_group,
    )


def _sum_wave_frame_jones_sp(waves: list[Wave]) -> tuple[complex, complex]:
    s_amplitude = 0.0j
    p_amplitude = 0.0j
    for wave in waves:
        a_x, a_y = _wave_frame_a_xy(wave)
        p_amplitude += a_x
        s_amplitude += a_y
    return complex(s_amplitude), complex(p_amplitude)


def _wave_frame_a_xy(wave: Wave) -> tuple[complex, complex]:
    kx = complex(wave.k[0])
    kz = complex(wave.k[2])
    norm = np.sqrt(kx * kx + kz * kz)
    if abs(norm) == 0:
        raise ValueError("Cannot construct wave-frame analyzer components from zero transverse wavevector.")
    sin_theta = kx / norm
    cos_theta = kz / norm
    electric = np.asarray(wave.electric, dtype=complex)
    a_x = cos_theta * electric[0] - sin_theta * electric[2]
    a_y = electric[1]
    return complex(a_x), complex(a_y)


def _broadcast_polarimetry_degrees(system: MultilayerSystem) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pol = system.polarimetry
    try:
        return np.broadcast_arrays(
            np.asarray(pol.theta_deg, dtype=float),
            np.asarray(pol.phi_deg, dtype=float),
            np.asarray(pol.psi_deg, dtype=float),
            np.asarray(pol.ellipticity_deg, dtype=float),
        )
    except ValueError as exc:
        raise ValueError("Polarimetry theta/phi/psi/ellipticity arrays must be broadcast-compatible.") from exc


def _maker_theta_grid(
    theta_deg: Sequence[float] | None,
    theta_step_deg: float,
    theta_min_deg: float,
    theta_max_deg: float,
) -> np.ndarray:
    if theta_deg is not None:
        values = np.asarray(theta_deg, dtype=float)
    else:
        if abs(theta_max_deg) >= 89.0:
            theta_max_deg = float(np.sign(theta_max_deg) * 89.0)
        if theta_step_deg <= 0:
            raise ValueError("theta_step_deg must be positive.")
        if theta_max_deg < theta_min_deg:
            raise ValueError("theta_max_deg must be greater than or equal to theta_min_deg.")
        count = int(np.floor((theta_max_deg - theta_min_deg) / theta_step_deg + 1e-12)) + 1
        values = theta_min_deg + theta_step_deg * np.arange(count, dtype=float)
    if values.ndim != 1:
        raise ValueError("theta_deg must be a one-dimensional sequence.")
    if np.any(values <= -90.0) or np.any(values >= 90.0):
        raise ValueError("Maker fringes incidence angles must satisfy -90 < theta_deg < 90.")
    return values
