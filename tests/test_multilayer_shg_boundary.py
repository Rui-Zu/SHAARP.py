import math
import unittest

import numpy as np

from shaarp.anisotropic import solve_snell_modes
from shaarp.config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry
from shaarp.linear import mode_to_wave
from shaarp.multilayer_basis import make_anisotropic_layer_basis, make_anisotropic_substrate_basis
from shaarp.multilayer_shg_boundary import (
    analyze_reflected_2omega,
    analyze_reflected_2omega_polarimetry,
    analyze_transmitted_2omega,
    analyze_transmitted_2omega_polarimetry,
    analyze_transmitted_2omega_waves,
    analyzer_jones_from_psi_deg,
    analyzer_jones_from_polarimetry,
    incident_jones_from_polarimetry,
    reflected_2omega_jones_sp,
    transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
    solve_multilayer_shg_boundary,
    solve_multilayer_shg_from_fundamental,
    solve_multilayer_shg_from_system_jones,
    solve_multilayer_shg_from_system_polarimetry,
    solve_multilayer_shg_from_system,
    solve_multilayer_shg_from_tensors,
    solve_multilayer_shg_from_tensors_jones,
    solve_multilayer_shg_polarimetry_sweep,
    solve_multilayer_shg_sample_azimuth_sweep,
    solve_multilayer_shg_system_sweep,
    _transmitted_waves_for_maker_policy,
)
from shaarp.nonlinear import multilayer_sources, solve_multilayer_inhomogeneous_fields
from shaarp.waves import Wave, make_isotropic_wave


class MultilayerSHGBoundaryTests(unittest.TestCase):
    def test_two_layer_shg_boundary_with_known_sources_has_small_residual(self):
        theta_air = math.radians(17.0)
        n_air = 1.0
        n_layers = [1.52, 1.83]
        n_sub = 1.28
        omega = 2.0
        theta_layers = [float(np.real(np.arcsin(n_air * np.sin(theta_air) / n + 0j))) for n in n_layers]
        theta_sub = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_sub + 0j)))

        layer_basis = [
            _layer_basis(n_layers[0], theta_layers[0], omega),
            _layer_basis(n_layers[1], theta_layers[1], omega),
        ]
        layer_sources = [
            [
                _source_wave(n_layers[0], theta_layers[0], omega, [0.05 + 0.02j, 0.03j, -0.01]),
                _source_wave(n_layers[0], theta_layers[0], omega, [-0.02j, 0.04, 0.015 + 0.01j]),
            ],
            [
                _source_wave(n_layers[1], theta_layers[1], omega, [0.025, -0.035j, 0.02 + 0.005j]),
            ],
        ]

        result = solve_multilayer_shg_boundary(
            reflected_basis_2omega=[
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=omega),
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=omega),
            ],
            layer_homogeneous_basis_2omega=layer_basis,
            substrate_basis_2omega=[
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="s", omega=omega),
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="p", omega=omega),
            ],
            layer_inhomogeneous_2omega=layer_sources,
            thicknesses=[0.31, 0.46],
            mu=1.0,
        )

        self.assertEqual(result.coefficients.shape, (12,))
        self.assertEqual(len(result.reflected_2omega), 2)
        self.assertEqual(len(result.layer_homogeneous_2omega), 2)
        self.assertEqual(len(result.substrate_2omega), 2)
        self.assertTrue(np.linalg.norm(result.residual) < 1e-10)
        self.assertGreater(np.linalg.norm(result.coefficients), 0.0)

    def test_layer_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            solve_multilayer_shg_boundary(
                reflected_basis_2omega=[],
                layer_homogeneous_basis_2omega=[[]],
                substrate_basis_2omega=[],
                layer_inhomogeneous_2omega=[],
                thicknesses=[1.0],
                mu=1.0,
            )

    def test_single_layer_shg_boundary_accepts_ten_solved_multilayer_sources(self):
        fundamental_omega = 1.4
        omega_2 = 2 * fundamental_omega
        theta_layer = math.radians(11.0)
        theta_air_2 = math.radians(18.0)
        theta_sub_2 = math.radians(13.0)
        d = np.array(
            [
                [0.3 + 0.2j, -0.4j, 0.7, 0.2 - 0.1j, 0.05j, -0.6],
                [0.1j, 0.5 + 0.15j, -0.25, 0.45j, 0.8, -0.2 + 0.05j],
                [-0.15, 0.25j, 0.35 + 0.2j, -0.5, 0.65j, 0.4],
            ],
            dtype=complex,
        )
        f1 = make_isotropic_wave(n=1.7 + 0.03j, theta_rad=theta_layer, polarization="p", omega=fundamental_omega)
        f2 = make_isotropic_wave(n=1.9 + 0.04j, theta_rad=theta_layer, polarization="s", omega=fundamental_omega)
        b1 = make_isotropic_wave(
            n=1.7 + 0.03j,
            theta_rad=theta_layer,
            polarization="p",
            direction_sign_z=-1,
            omega=fundamental_omega,
        )
        b2 = make_isotropic_wave(
            n=1.9 + 0.04j,
            theta_rad=theta_layer,
            polarization="s",
            direction_sign_z=-1,
            omega=fundamental_omega,
        )
        epsilon_2omega = np.array(
            [
                [2.6 + 0.15j, 0.08 - 0.03j, -0.05 + 0.02j],
                [0.04 + 0.07j, 3.1 + 0.11j, 0.06 - 0.01j],
                [-0.02 + 0.05j, 0.03 + 0.04j, 3.7 + 0.09j],
            ],
            dtype=complex,
        )
        fields = solve_multilayer_inhomogeneous_fields(
            epsilon_2omega,
            multilayer_sources(d, f1, f2, b1, b2),
            mu=1.0,
            eps0=1.0,
        )

        result = solve_multilayer_shg_boundary(
            reflected_basis_2omega=[
                make_isotropic_wave(n=1.0, theta_rad=theta_air_2, polarization="s", direction_sign_z=-1, omega=omega_2),
                make_isotropic_wave(n=1.0, theta_rad=theta_air_2, polarization="p", direction_sign_z=-1, omega=omega_2),
            ],
            layer_homogeneous_basis_2omega=[_layer_basis(1.8 + 0.04j, theta_layer, omega_2)],
            substrate_basis_2omega=[
                make_isotropic_wave(n=1.35 + 0.02j, theta_rad=theta_sub_2, polarization="s", omega=omega_2),
                make_isotropic_wave(n=1.35 + 0.02j, theta_rad=theta_sub_2, polarization="p", omega=omega_2),
            ],
            layer_inhomogeneous_2omega=[[_field_to_wave(field) for field in fields.as_list()]],
            thicknesses=[0.27],
            mu=1.0,
        )

        self.assertEqual(len(result.layer_known[0]), 10)
        self.assertEqual(result.coefficients.shape, (8,))
        self.assertTrue(np.linalg.norm(result.residual) < 1e-9)
        self.assertGreater(np.linalg.norm(result.coefficients), 0.0)

    def test_workflow_solves_fundamental_then_ten_source_shg_boundary(self):
        theta_air = math.radians(14.0)
        theta_layer = math.radians(9.0)
        theta_sub = math.radians(11.0)
        omega = 1.2
        omega_2 = 2 * omega
        d = np.array(
            [
                [0.4 + 0.1j, -0.2j, 0.6, 0.15, 0.05j, -0.3],
                [0.2j, 0.5, -0.1 + 0.05j, 0.35j, 0.7, -0.25],
                [-0.1, 0.2j, 0.45 + 0.1j, -0.4, 0.55j, 0.25],
            ],
            dtype=complex,
        )
        epsilon_2omega = np.array(
            [
                [2.5 + 0.09j, 0.05 - 0.02j, -0.03 + 0.01j],
                [0.02 + 0.04j, 2.9 + 0.08j, 0.04 - 0.02j],
                [-0.01 + 0.03j, 0.02 + 0.03j, 3.4 + 0.07j],
            ],
            dtype=complex,
        )

        result = solve_multilayer_shg_from_fundamental(
            incident_omega=[make_isotropic_wave(n=1.0, theta_rad=theta_air, polarization="p", omega=omega)],
            reflected_basis_omega=[
                make_isotropic_wave(n=1.0, theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=omega),
                make_isotropic_wave(n=1.0, theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=omega),
            ],
            layer_basis_omega=[_layer_basis(1.65 + 0.02j, theta_layer, omega)],
            substrate_basis_omega=[
                make_isotropic_wave(n=1.28 + 0.01j, theta_rad=theta_sub, polarization="s", omega=omega),
                make_isotropic_wave(n=1.28 + 0.01j, theta_rad=theta_sub, polarization="p", omega=omega),
            ],
            layer_d_voigt_lab=[d],
            layer_epsilon_2omega_lab=[epsilon_2omega],
            reflected_basis_2omega=[
                make_isotropic_wave(n=1.0, theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=omega_2),
                make_isotropic_wave(n=1.0, theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=omega_2),
            ],
            layer_homogeneous_basis_2omega=[_layer_basis(1.8 + 0.03j, theta_layer, omega_2)],
            substrate_basis_2omega=[
                make_isotropic_wave(n=1.34 + 0.02j, theta_rad=theta_sub, polarization="s", omega=omega_2),
                make_isotropic_wave(n=1.34 + 0.02j, theta_rad=theta_sub, polarization="p", omega=omega_2),
            ],
            thicknesses=[0.22],
            mu=1.0,
            eps0=1.0,
        )

        self.assertTrue(np.linalg.norm(result.fundamental.residual) < 1e-10)
        self.assertEqual(len(result.layer_sources), 1)
        self.assertEqual(len(result.layer_sources[0].as_list()), 10)
        self.assertEqual(len(result.layer_inhomogeneous_fields[0].as_list()), 10)
        self.assertTrue(all(np.linalg.norm(field.residual) < 1e-10 for field in result.layer_inhomogeneous_fields[0].as_list()))
        self.assertTrue(np.linalg.norm(result.shg.residual) < 1e-9)
        self.assertGreater(np.linalg.norm(result.shg.coefficients), 0.0)

    def test_workflow_requires_four_omega_layer_waves(self):
        with self.assertRaisesRegex(ValueError, "four waves"):
            solve_multilayer_shg_from_fundamental(
                incident_omega=[],
                reflected_basis_omega=[],
                layer_basis_omega=[[]],
                substrate_basis_omega=[],
                layer_d_voigt_lab=[np.eye(3, 6)],
                layer_epsilon_2omega_lab=[np.eye(3)],
                reflected_basis_2omega=[],
                layer_homogeneous_basis_2omega=[[]],
                substrate_basis_2omega=[],
                thicknesses=[1.0],
            )

    def test_tensor_workflow_generates_bases_and_solves_complex_nondiagonal_case(self):
        theta = math.radians(12.0)
        omega = 1.15
        eps_w_layer = np.array(
            [
                [1.62**2, 0.025, -0.012],
                [0.025, 1.78**2, 0.018],
                [-0.012, 0.018, 2.05**2],
            ],
            dtype=complex,
        )
        eps_w_sub = np.array(
            [
                [1.31**2, 0.01, 0.0],
                [0.01, 1.39**2, -0.015],
                [0.0, -0.015, 1.47**2],
            ],
            dtype=complex,
        )
        eps_2w_layer = np.array(
            [
                [2.7 + 0.08j, 0.04 - 0.01j, -0.02 + 0.015j],
                [0.04 - 0.01j, 3.1 + 0.06j, 0.03 - 0.02j],
                [-0.02 + 0.015j, 0.03 - 0.02j, 3.6 + 0.05j],
            ],
            dtype=complex,
        )
        eps_2w_sub = np.array(
            [
                [1.8 + 0.04j, 0.02 - 0.01j, 0.0],
                [0.02 - 0.01j, 2.1 + 0.03j, 0.01 + 0.005j],
                [0.0, 0.01 + 0.005j, 2.4 + 0.02j],
            ],
            dtype=complex,
        )
        d = np.array(
            [
                [0.4 + 0.1j, -0.2j, 0.6, 0.15, 0.05j, -0.3],
                [0.2j, 0.5, -0.1 + 0.05j, 0.35j, 0.7, -0.25],
                [-0.1, 0.2j, 0.45 + 0.1j, -0.4, 0.55j, 0.25],
            ],
            dtype=complex,
        )

        result = solve_multilayer_shg_from_tensors(
            incident_index_omega=1.0,
            top_index_2omega=1.08 + 0.01j,
            incident_theta_rad=theta,
            incident_polarization="p",
            layer_epsilon_omega_lab=[eps_w_layer],
            substrate_epsilon_omega_lab=eps_w_sub,
            layer_d_voigt_lab=[d],
            layer_epsilon_2omega_lab=[eps_2w_layer],
            substrate_epsilon_2omega_lab=eps_2w_sub,
            thicknesses=[0.24],
            omega=omega,
            mu=1.0,
            eps0=1.0,
        )

        expected_source_kx = 2 * omega * np.sin(theta)
        self.assertTrue(np.linalg.norm(result.fundamental.residual) < 1e-10)
        self.assertEqual(len(result.layer_sources[0].as_list()), 10)
        self.assertTrue(all(np.isclose(source.k[0], expected_source_kx, atol=1e-9) for source in result.layer_sources[0].as_list()))
        self.assertTrue(all(np.linalg.norm(field.residual) < 1e-9 for field in result.layer_inhomogeneous_fields[0].as_list()))
        self.assertTrue(np.linalg.norm(result.shg.residual) < 1e-8)

    def test_manual_basis_workflow_accepts_anisotropic_top_medium(self):
        theta = math.radians(9.0)
        omega = 1.1
        tangential_index = 1.0
        top_eps_w = np.array(
            [
                [1.3 + 0.02j, 0.03 - 0.01j, 0.01j],
                [0.03 - 0.01j, 1.6 + 0.03j, -0.02],
                [0.01j, -0.02, 1.9 + 0.04j],
            ],
            dtype=complex,
        )
        top_eps_2w = np.array(
            [
                [1.55 + 0.04j, 0.02 + 0.01j, -0.01j],
                [0.02 + 0.01j, 1.75 + 0.05j, 0.03],
                [-0.01j, 0.03, 2.05 + 0.06j],
            ],
            dtype=complex,
        )
        layer_eps_w = np.diag([1.62**2, 1.78**2, 2.05**2]).astype(complex)
        layer_eps_2w = np.diag([2.7 + 0.08j, 3.1 + 0.06j, 3.6 + 0.05j]).astype(complex)
        sub_eps_w = np.diag([1.31**2, 1.39**2, 1.47**2]).astype(complex)
        sub_eps_2w = np.diag([1.8 + 0.04j, 2.1 + 0.03j, 2.4 + 0.02j]).astype(complex)
        d = np.array(
            [
                [0.35 + 0.08j, -0.12j, 0.5, 0.11, 0.04j, -0.27],
                [0.18j, 0.42, -0.09 + 0.04j, 0.31j, 0.63, -0.22],
                [-0.08, 0.17j, 0.41 + 0.09j, -0.34, 0.48j, 0.21],
            ],
            dtype=complex,
        )
        top_forward = solve_snell_modes(top_eps_w, incident_index=tangential_index, incident_theta_rad=theta)
        top_backward = solve_snell_modes(top_eps_w, incident_index=tangential_index, incident_theta_rad=theta, reflected=True)
        top_2w_basis = make_anisotropic_layer_basis(
            top_eps_2w,
            tangential_index=tangential_index,
            incident_theta_rad=theta,
            omega=2 * omega,
        )

        result = solve_multilayer_shg_from_fundamental(
            incident_omega=[mode_to_wave(top_forward.fast, omega=omega)],
            reflected_basis_omega=[mode_to_wave(top_backward.fast, omega=omega), mode_to_wave(top_backward.slow, omega=omega)],
            layer_basis_omega=[
                make_anisotropic_layer_basis(layer_eps_w, tangential_index=tangential_index, incident_theta_rad=theta, omega=omega)
            ],
            substrate_basis_omega=make_anisotropic_substrate_basis(
                sub_eps_w,
                tangential_index=tangential_index,
                incident_theta_rad=theta,
                omega=omega,
            ),
            layer_d_voigt_lab=[d],
            layer_epsilon_2omega_lab=[layer_eps_2w],
            reflected_basis_2omega=top_2w_basis[2:],
            layer_homogeneous_basis_2omega=[
                make_anisotropic_layer_basis(layer_eps_2w, tangential_index=tangential_index, incident_theta_rad=theta, omega=2 * omega)
            ],
            substrate_basis_2omega=make_anisotropic_substrate_basis(
                sub_eps_2w,
                tangential_index=tangential_index,
                incident_theta_rad=theta,
                omega=2 * omega,
            ),
            thicknesses=[0.19],
            mu=1.0,
            eps0=1.0,
        )

        self.assertEqual(len(result.shg.reflected_2omega), 2)
        self.assertTrue(np.linalg.norm(result.fundamental.residual) < 1e-8)
        self.assertTrue(all(np.linalg.norm(field.residual) < 1e-8 for field in result.layer_inhomogeneous_fields[0].as_list()))
        self.assertTrue(np.linalg.norm(result.shg.residual) < 1e-7)

    def test_tensor_jones_workflow_pure_p_matches_polarization_workflow(self):
        theta = math.radians(12.0)
        omega = 1.15
        eps_w_layer = np.diag([1.62**2, 1.78**2, 2.05**2]).astype(complex)
        eps_w_sub = np.diag([1.31**2, 1.39**2, 1.47**2]).astype(complex)
        eps_2w_layer = np.diag([2.7 + 0.08j, 3.1 + 0.06j, 3.6 + 0.05j]).astype(complex)
        eps_2w_sub = np.diag([1.8 + 0.04j, 2.1 + 0.03j, 2.4 + 0.02j]).astype(complex)
        d = np.array(
            [
                [0.4 + 0.1j, -0.2j, 0.6, 0.15, 0.05j, -0.3],
                [0.2j, 0.5, -0.1 + 0.05j, 0.35j, 0.7, -0.25],
                [-0.1, 0.2j, 0.45 + 0.1j, -0.4, 0.55j, 0.25],
            ],
            dtype=complex,
        )

        pure = solve_multilayer_shg_from_tensors(
            incident_index_omega=1.0,
            top_index_2omega=1.08 + 0.01j,
            incident_theta_rad=theta,
            incident_polarization="p",
            layer_epsilon_omega_lab=[eps_w_layer],
            substrate_epsilon_omega_lab=eps_w_sub,
            layer_d_voigt_lab=[d],
            layer_epsilon_2omega_lab=[eps_2w_layer],
            substrate_epsilon_2omega_lab=eps_2w_sub,
            thicknesses=[0.24],
            omega=omega,
            mu=1.0,
            eps0=1.0,
        )
        jones = solve_multilayer_shg_from_tensors_jones(
            incident_index_omega=1.0,
            top_index_2omega=1.08 + 0.01j,
            incident_theta_rad=theta,
            incident_jones_sp=(0.0, 1.0),
            layer_epsilon_omega_lab=[eps_w_layer],
            substrate_epsilon_omega_lab=eps_w_sub,
            layer_d_voigt_lab=[d],
            layer_epsilon_2omega_lab=[eps_2w_layer],
            substrate_epsilon_2omega_lab=eps_2w_sub,
            thicknesses=[0.24],
            omega=omega,
            mu=1.0,
            eps0=1.0,
        )

        np.testing.assert_allclose(jones.fundamental.coefficients, pure.fundamental.coefficients, atol=1e-12)
        np.testing.assert_allclose(jones.shg.coefficients, pure.shg.coefficients, atol=1e-12)

    def test_system_workflow_rotates_configured_material_tensors(self):
        orientation = CrystalOrientation.from_cubic_miller_surface((1, 2, 3))
        structure = CrystalStructure(point_group="1")
        top = Material(
            name="air",
            structure=structure,
            orientation=CrystalOrientation(),
            epsilon_omega=np.eye(3),
            epsilon_2omega=np.eye(3) * (1.04**2),
            d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
        )
        layer = Material(
            name="rotated-layer",
            structure=structure,
            orientation=orientation,
            epsilon_omega=np.diag([1.58**2, 1.76**2, 2.01**2]),
            epsilon_2omega=np.diag([2.65 + 0.05j, 3.05 + 0.04j, 3.55 + 0.03j]),
            d_voigt_pm_v=np.array(
                [
                    [0.35 + 0.08j, -0.12j, 0.5, 0.11, 0.04j, -0.27],
                    [0.18j, 0.42, -0.09 + 0.04j, 0.31j, 0.63, -0.22],
                    [-0.08, 0.17j, 0.41 + 0.09j, -0.34, 0.48j, 0.21],
                ],
                dtype=complex,
            ),
        )
        substrate = Material(
            name="substrate",
            structure=structure,
            orientation=CrystalOrientation(),
            epsilon_omega=np.diag([1.32**2, 1.41**2, 1.49**2]),
            epsilon_2omega=np.diag([1.8 + 0.02j, 2.05 + 0.02j, 2.3 + 0.01j]),
            d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
        )
        system = MultilayerSystem(
            wavelength_um=1.064,
            polarimetry=Polarimetry(theta_deg=10.0),
            layers=[
                Layer("top", top, shg_active=False),
                Layer("film", layer, thickness_um=0.18, shg_active=True),
                Layer("substrate", substrate, shg_active=False),
            ],
        )

        result = solve_multilayer_shg_from_system(system, incident_polarization="p", mu=1.0, eps0=1.0)

        self.assertTrue(np.linalg.norm(result.fundamental.residual) < 1e-9)
        self.assertEqual(len(result.layer_sources[0].as_list()), 10)
        self.assertTrue(all(np.linalg.norm(field.residual) < 1e-8 for field in result.layer_inhomogeneous_fields[0].as_list()))
        self.assertTrue(np.linalg.norm(result.shg.residual) < 1e-7)

    def test_system_workflow_rejects_anisotropic_top_medium(self):
        structure = CrystalStructure(point_group="1")
        top = Material(
            name="bad-top",
            structure=structure,
            orientation=CrystalOrientation(),
            epsilon_omega=np.diag([1.0, 1.1, 1.0]),
            epsilon_2omega=np.eye(3),
            d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
        )
        layer = Material(
            name="layer",
            structure=structure,
            orientation=CrystalOrientation(),
            epsilon_omega=np.diag([2.0, 2.3, 2.6]),
            epsilon_2omega=np.diag([2.2, 2.5, 2.8]),
            d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
        )
        system = MultilayerSystem(
            wavelength_um=1.0,
            polarimetry=Polarimetry(theta_deg=5.0),
            layers=[
                Layer("top", top, shg_active=False),
                Layer("film", layer, thickness_um=0.1),
                Layer("substrate", layer, shg_active=False),
            ],
        )
        with self.assertRaisesRegex(ValueError, "top epsilon_omega must be isotropic"):
            solve_multilayer_shg_from_system(system)

    def test_system_jones_pure_p_matches_polarization_workflow(self):
        system = _configured_system(theta_deg=10.0)

        pure = solve_multilayer_shg_from_system(system, incident_polarization="p", mu=1.0, eps0=1.0)
        jones = solve_multilayer_shg_from_system_jones(system, incident_jones_sp=(0.0, 1.0), mu=1.0, eps0=1.0)

        np.testing.assert_allclose(jones.fundamental.coefficients, pure.fundamental.coefficients, atol=1e-12)
        np.testing.assert_allclose(jones.shg.coefficients, pure.shg.coefficients, atol=1e-12)

    def test_system_polarimetry_uses_phi_and_ellipticity_for_incident_jones(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=10.0, phi_deg=30.0, ellipticity_deg=20.0),
            layers=base.layers,
        )
        jones = incident_jones_from_polarimetry(system)
        self.assertTrue(np.isclose(jones[0], np.sin(np.deg2rad(30.0)) * np.exp(1j * np.deg2rad(20.0))))
        self.assertTrue(np.isclose(jones[1], np.cos(np.deg2rad(30.0))))

        result = solve_multilayer_shg_from_system_polarimetry(system, mu=1.0, eps0=1.0)

        self.assertTrue(np.linalg.norm(result.fundamental.residual) < 1e-9)
        self.assertTrue(np.linalg.norm(result.shg.residual) < 1e-7)

    def test_reflected_2omega_jones_components_and_analyzer_intensity(self):
        system = _configured_system(theta_deg=10.0)
        result = solve_multilayer_shg_from_system(system, incident_polarization="p", mu=1.0, eps0=1.0)

        s_component, p_component = reflected_2omega_jones_sp(result.shg)
        s_amp, s_intensity = analyze_reflected_2omega(result.shg, (1.0, 0.0))
        p_amp, p_intensity = analyze_reflected_2omega(result.shg, (0.0, 1.0))
        mixed_amp, mixed_intensity = analyze_reflected_2omega(result.shg, (1 / np.sqrt(2), 1 / np.sqrt(2)))

        self.assertTrue(np.isclose(s_amp, s_component))
        self.assertTrue(np.isclose(p_amp, p_component))
        self.assertTrue(np.isclose(s_intensity, abs(s_component) ** 2))
        self.assertTrue(np.isclose(p_intensity, abs(p_component) ** 2))
        self.assertTrue(np.isclose(mixed_amp, (s_component + p_component) / np.sqrt(2)))
        self.assertTrue(np.isclose(mixed_intensity, abs((s_component + p_component) / np.sqrt(2)) ** 2))

    def test_transmitted_2omega_jones_components_and_analyzer_intensity(self):
        system = _configured_system(theta_deg=10.0)
        result = solve_multilayer_shg_from_system(system, incident_polarization="p", mu=1.0, eps0=1.0)

        s_component, p_component = transmitted_2omega_jones_sp(result.shg)
        s_amp, s_intensity = analyze_transmitted_2omega(result.shg, (1.0, 0.0))
        p_amp, p_intensity = analyze_transmitted_2omega(result.shg, (0.0, 1.0))
        mixed_amp, mixed_intensity = analyze_transmitted_2omega(result.shg, (1 / np.sqrt(2), 1 / np.sqrt(2)))

        self.assertTrue(np.isclose(s_amp, s_component))
        self.assertTrue(np.isclose(p_amp, p_component))
        self.assertTrue(np.isclose(s_intensity, abs(s_component) ** 2))
        self.assertTrue(np.isclose(p_intensity, abs(p_component) ** 2))
        self.assertTrue(np.isclose(mixed_amp, (s_component + p_component) / np.sqrt(2)))
        self.assertTrue(np.isclose(mixed_intensity, abs((s_component + p_component) / np.sqrt(2)) ** 2))

    def test_analyzer_polarimetry_uses_scalar_psi(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=10.0, psi_deg=90.0),
            layers=base.layers,
        )
        result = solve_multilayer_shg_from_system(system, incident_polarization="p", mu=1.0, eps0=1.0)

        analyzer = analyzer_jones_from_polarimetry(system)
        amplitude, intensity = analyze_reflected_2omega_polarimetry(result.shg, system)
        s_component, _ = reflected_2omega_jones_sp(result.shg)

        self.assertTrue(np.isclose(analyzer[0], 1.0))
        self.assertTrue(np.isclose(analyzer[1], 0.0))
        self.assertTrue(np.isclose(amplitude, s_component))
        self.assertTrue(np.isclose(intensity, abs(s_component) ** 2))

    def test_analyzer_jones_from_psi_deg_matches_polarimetry_helper(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=10.0, psi_deg=35.0),
            layers=base.layers,
        )

        self.assertEqual(analyzer_jones_from_psi_deg(35.0), analyzer_jones_from_polarimetry(system))

    def test_polarimetry_sweep_broadcasts_phi_psi_and_returns_intensity(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(
                theta_deg=10.0,
                phi_deg=np.array([0.0, 30.0, 60.0]),
                psi_deg=np.array([0.0, 45.0, 90.0]),
                ellipticity_deg=15.0,
            ),
            layers=base.layers,
        )

        sweep = solve_multilayer_shg_polarimetry_sweep(system, mu=1.0, eps0=1.0)

        self.assertEqual(sweep.shape, (3,))
        self.assertEqual(len(sweep.results), 3)
        self.assertEqual(sweep.intensity.shape, (3,))
        self.assertTrue(np.all(sweep.fundamental_residual_norm < 1e-9))
        self.assertTrue(np.all(sweep.shg_residual_norm < 1e-7))
        self.assertTrue(np.allclose(sweep.intensity, np.abs(sweep.analyzer_amplitude) ** 2))

        first_system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=10.0, phi_deg=0.0, psi_deg=0.0, ellipticity_deg=15.0),
            layers=base.layers,
        )
        first = solve_multilayer_shg_from_system_polarimetry(first_system, mu=1.0, eps0=1.0)
        expected_amplitude, expected_intensity = analyze_reflected_2omega_polarimetry(first.shg, first_system)
        self.assertTrue(np.isclose(sweep.analyzer_amplitude[0], expected_amplitude))
        self.assertTrue(np.isclose(sweep.intensity[0], expected_intensity))

    def test_polarimetry_sweep_rejects_incompatible_shapes(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(
                theta_deg=np.array([8.0, 12.0]),
                phi_deg=np.array([0.0, 30.0, 60.0]),
                psi_deg=0.0,
            ),
            layers=base.layers,
        )
        with self.assertRaisesRegex(ValueError, "broadcast-compatible"):
            solve_multilayer_shg_polarimetry_sweep(system, mu=1.0, eps0=1.0)

    def test_sample_azimuth_sweep_rotates_sample_separately_from_phi(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=10.0, phi_deg=30.0, psi_deg=45.0, ellipticity_deg=10.0),
            layers=base.layers,
        )

        sweep = solve_multilayer_shg_sample_azimuth_sweep(
            system,
            sample_azimuth_deg=np.array([0.0, 25.0, 50.0]),
            mu=1.0,
            eps0=1.0,
        )

        self.assertEqual(sweep.shape, (3,))
        self.assertEqual(len(sweep.results), 3)
        np.testing.assert_allclose(sweep.sample_azimuth_deg, [0.0, 25.0, 50.0])
        np.testing.assert_allclose(sweep.phi_deg, [30.0, 30.0, 30.0])
        self.assertTrue(np.all(sweep.fundamental_residual_norm < 1e-9))
        self.assertTrue(np.all(sweep.shg_residual_norm < 1e-7))
        self.assertTrue(np.allclose(sweep.intensity, np.abs(sweep.analyzer_amplitude) ** 2))
        self.assertEqual(sweep.reflected_parallel_intensity.shape, (3,))
        self.assertEqual(sweep.reflected_perpendicular_intensity.shape, (3,))
        self.assertEqual(sweep.transmitted_parallel_intensity.shape, (3,))
        self.assertEqual(sweep.transmitted_perpendicular_intensity.shape, (3,))
        np.testing.assert_allclose(sweep.reflected_parallel_intensity, np.abs(sweep.reflected_parallel_amplitude) ** 2)
        np.testing.assert_allclose(sweep.reflected_perpendicular_intensity, np.abs(sweep.reflected_perpendicular_amplitude) ** 2)
        np.testing.assert_allclose(sweep.transmitted_parallel_intensity, np.abs(sweep.transmitted_parallel_amplitude) ** 2)
        np.testing.assert_allclose(sweep.transmitted_perpendicular_intensity, np.abs(sweep.transmitted_perpendicular_amplitude) ** 2)
        np.testing.assert_allclose(sweep.analyzer_amplitude, sweep.reflected_parallel_amplitude)
        np.testing.assert_allclose(sweep.intensity, sweep.reflected_parallel_intensity)

        first = solve_multilayer_shg_from_system_polarimetry(system, mu=1.0, eps0=1.0)
        expected_amplitude, expected_intensity = analyze_reflected_2omega_polarimetry(first.shg, system)
        self.assertFalse(np.isclose(sweep.analyzer_amplitude[0], expected_amplitude))
        self.assertFalse(np.isclose(sweep.intensity[0], expected_intensity))

    def test_sample_azimuth_sweep_broadcasts_with_polarimetry_arrays(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(
                theta_deg=np.array([8.0, 12.0]),
                phi_deg=np.array([20.0, 40.0]),
                psi_deg=0.0,
                ellipticity_deg=5.0,
            ),
            layers=base.layers,
        )

        sweep = solve_multilayer_shg_sample_azimuth_sweep(
            system,
            sample_azimuth_deg=np.array([0.0, 30.0]),
            mu=1.0,
            eps0=1.0,
        )

        self.assertEqual(sweep.shape, (2,))
        np.testing.assert_allclose(sweep.theta_deg, [8.0, 12.0])
        np.testing.assert_allclose(sweep.phi_deg, [20.0, 40.0])
        np.testing.assert_allclose(sweep.sample_azimuth_deg, [0.0, 30.0])
        self.assertTrue(np.all(sweep.fundamental_residual_norm < 1e-9))
        self.assertTrue(np.all(sweep.shg_residual_norm < 1e-7))

    def test_sample_azimuth_sweep_rejects_incompatible_shapes(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=np.array([8.0, 12.0]), phi_deg=30.0),
            layers=base.layers,
        )

        with self.assertRaisesRegex(ValueError, "Sample azimuth and polarimetry arrays must be broadcast-compatible"):
            solve_multilayer_shg_sample_azimuth_sweep(system, sample_azimuth_deg=np.array([0.0, 20.0, 40.0]))

    def test_system_sweep_runs_multiple_incidence_angles(self):
        system = _configured_system(theta_deg=[6.0, 10.0, 14.0])

        sweep = solve_multilayer_shg_system_sweep(system, incident_polarization="p", mu=1.0, eps0=1.0)

        np.testing.assert_allclose(sweep.theta_deg, [6.0, 10.0, 14.0])
        self.assertEqual(len(sweep.results), 3)
        self.assertTrue(np.all(sweep.fundamental_residual_norm < 1e-9))
        self.assertTrue(np.all(sweep.shg_residual_norm < 1e-7))
        self.assertTrue(np.all(np.isfinite(sweep.reflected_2omega_field_norm2)))

    def test_system_sweep_accepts_explicit_theta_override(self):
        system = _configured_system(theta_deg=10.0)

        sweep = solve_multilayer_shg_system_sweep(system, theta_deg=[8.0, 12.0], incident_polarization="s", mu=1.0, eps0=1.0)

        np.testing.assert_allclose(sweep.theta_deg, [8.0, 12.0])
        self.assertEqual(len(sweep.results), 2)
        self.assertTrue(np.all(sweep.fundamental_residual_norm < 1e-9))
        self.assertTrue(np.all(sweep.shg_residual_norm < 1e-7))

    def test_system_sweep_rejects_multidimensional_theta(self):
        system = _configured_system(theta_deg=10.0)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            solve_multilayer_shg_system_sweep(system, theta_deg=[[8.0, 12.0]])

    def test_maker_fringes_sweep_returns_shaarp_ml_copy_lists(self):
        base = _configured_system(theta_deg=10.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=10.0, phi_deg=30.0, psi_deg=45.0, ellipticity_deg=10.0),
            layers=base.layers,
        )

        sweep = solve_multilayer_maker_fringes_sweep(
            system,
            theta_min_deg=-10.0,
            theta_max_deg=10.0,
            theta_step_deg=10.0,
            mu=1.0,
            eps0=1.0,
        )

        np.testing.assert_allclose(sweep.theta_deg, [-10.0, 0.0, 10.0])
        self.assertEqual(len(sweep.results), 3)
        self.assertTrue(np.all(sweep.fundamental_residual_norm < 1e-9))
        self.assertTrue(np.all(sweep.shg_residual_norm < 1e-7))
        np.testing.assert_allclose(sweep.parallel_intensity, np.abs(sweep.parallel_amplitude) ** 2)
        np.testing.assert_allclose(sweep.perpendicular_intensity, np.abs(sweep.perpendicular_amplitude) ** 2)
        copy_lists = sweep.shaarp_ml_copy_lists()
        self.assertEqual(len(copy_lists), 2)
        np.testing.assert_allclose(copy_lists[0], sweep.list_mf_para)
        np.testing.assert_allclose(copy_lists[1], sweep.list_mf_perp)
        np.testing.assert_allclose(copy_lists[0][:, 0], [-10.0, 0.0, 10.0])
        np.testing.assert_allclose(copy_lists[1][:, 0], [-10.0, 0.0, 10.0])

        direct_system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=-10.0, phi_deg=30.0, psi_deg=45.0, ellipticity_deg=10.0),
            layers=base.layers,
        )
        direct = solve_multilayer_shg_from_system_polarimetry(
            direct_system,
            mu=1.0,
            eps0=1.0,
            inhomogeneous_source_policy="forward_only",
        )
        selected_waves = _transmitted_waves_for_maker_policy(direct.shg, "shaarp_ml_selected")
        self.assertEqual(len(selected_waves), sweep.selected_transmitted_wave_count)
        para_amp, para_intensity = analyze_transmitted_2omega_waves(
            selected_waves,
            analyzer_jones_from_polarimetry(direct_system),
        )
        perp_amp, perp_intensity = analyze_transmitted_2omega_waves(selected_waves, analyzer_jones_from_psi_deg(135.0))
        self.assertTrue(np.isclose(sweep.parallel_amplitude[0], para_amp))
        self.assertTrue(np.isclose(sweep.parallel_intensity[0], para_intensity))
        self.assertTrue(np.isclose(sweep.perpendicular_amplitude[0], perp_amp))
        self.assertTrue(np.isclose(sweep.perpendicular_intensity[0], perp_intensity))

    def test_maker_fringes_keeps_shaarp_selected_and_physical_sum_modes_distinct(self):
        base = _configured_system(theta_deg=0.0)
        system = MultilayerSystem(
            wavelength_um=base.wavelength_um,
            polarimetry=Polarimetry(theta_deg=0.0, phi_deg=30.0, psi_deg=45.0, ellipticity_deg=10.0),
            layers=base.layers,
        )

        selected = solve_multilayer_maker_fringes_sweep(
            system,
            theta_deg=[0.0],
            mu=1.0,
            eps0=1.0,
            transmitted_wave_policy="shaarp_ml_selected",
        )
        physical = solve_multilayer_maker_fringes_sweep(
            system,
            theta_deg=[0.0],
            mu=1.0,
            eps0=1.0,
            transmitted_wave_policy="physical_sum",
        )

        self.assertEqual(selected.selected_transmitted_wave_count, 1)
        self.assertEqual(physical.selected_transmitted_wave_count, 2)
        self.assertGreater(abs(selected.parallel_amplitude[0] - physical.parallel_amplitude[0]), 1e-5)
        self.assertGreater(abs(selected.perpendicular_amplitude[0] - physical.perpendicular_amplitude[0]), 1e-5)

    def test_maker_fringes_sweep_clamps_max_angle_like_gui(self):
        system = _configured_system(theta_deg=10.0)

        sweep = solve_multilayer_maker_fringes_sweep(
            system,
            theta_min_deg=0.0,
            theta_max_deg=91.0,
            theta_step_deg=40.0,
            mu=1.0,
            eps0=1.0,
        )

        np.testing.assert_allclose(sweep.theta_deg, [0.0, 40.0, 80.0])

    def test_maker_fringes_sweep_rejects_invalid_angle_grid(self):
        system = _configured_system(theta_deg=10.0)

        with self.assertRaisesRegex(ValueError, "positive"):
            solve_multilayer_maker_fringes_sweep(system, theta_step_deg=0.0)
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            solve_multilayer_maker_fringes_sweep(system, theta_min_deg=10.0, theta_max_deg=0.0)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            solve_multilayer_maker_fringes_sweep(system, theta_deg=[[0.0, 10.0]])
        with self.assertRaisesRegex(ValueError, "-90 < theta_deg < 90"):
            solve_multilayer_maker_fringes_sweep(system, theta_deg=[-90.0])


def _layer_basis(n, theta, omega):
    return [
        make_isotropic_wave(n=n, theta_rad=theta, polarization="s", omega=omega),
        make_isotropic_wave(n=n, theta_rad=theta, polarization="p", omega=omega),
        make_isotropic_wave(n=n, theta_rad=theta, polarization="s", direction_sign_z=-1, omega=omega),
        make_isotropic_wave(n=n, theta_rad=theta, polarization="p", direction_sign_z=-1, omega=omega),
    ]


def _source_wave(n, theta, omega, electric):
    basis = make_isotropic_wave(n=n, theta_rad=theta, polarization="p", omega=omega)
    return Wave(k=basis.k, electric=np.array(electric, dtype=complex), omega=omega)


def _field_to_wave(field):
    return Wave(k=field.k, electric=field.electric, omega=field.omega)


def _configured_system(theta_deg):
    orientation = CrystalOrientation.from_cubic_miller_surface((1, 2, 3))
    structure = CrystalStructure(point_group="1")
    top = Material(
        name="air",
        structure=structure,
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3),
        epsilon_2omega=np.eye(3) * (1.04**2),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    layer = Material(
        name="rotated-layer",
        structure=structure,
        orientation=orientation,
        epsilon_omega=np.diag([1.58**2, 1.76**2, 2.01**2]),
        epsilon_2omega=np.diag([2.65 + 0.05j, 3.05 + 0.04j, 3.55 + 0.03j]),
        d_voigt_pm_v=np.array(
            [
                [0.35 + 0.08j, -0.12j, 0.5, 0.11, 0.04j, -0.27],
                [0.18j, 0.42, -0.09 + 0.04j, 0.31j, 0.63, -0.22],
                [-0.08, 0.17j, 0.41 + 0.09j, -0.34, 0.48j, 0.21],
            ],
            dtype=complex,
        ),
    )
    substrate = Material(
        name="substrate",
        structure=structure,
        orientation=CrystalOrientation(),
        epsilon_omega=np.diag([1.32**2, 1.41**2, 1.49**2]),
        epsilon_2omega=np.diag([1.8 + 0.02j, 2.05 + 0.02j, 2.3 + 0.01j]),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    return MultilayerSystem(
        wavelength_um=1.064,
        polarimetry=Polarimetry(theta_deg=theta_deg),
        layers=[
            Layer("top", top, shg_active=False),
            Layer("film", layer, thickness_um=0.18, shg_active=True),
            Layer("substrate", substrate, shg_active=False),
        ],
    )


if __name__ == "__main__":
    unittest.main()
