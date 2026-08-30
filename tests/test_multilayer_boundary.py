import math
import unittest

import numpy as np

from shaarp.multilayer_boundary import _tangential, build_multilayer_boundary_system, solve_multilayer_boundary
from shaarp.waves import Wave
from shaarp.waves import make_isotropic_wave


class MultilayerBoundaryTests(unittest.TestCase):
    def test_tangential_order_matches_solve_fresnel_n_threaded_equations(self):
        wave = Wave(
            k=np.array([0.2, 0.0, 1.1], dtype=complex),
            electric=np.array([1.0 + 2.0j, 3.0 - 1.0j, 0.5], dtype=complex),
            omega=2.0,
        )
        h = wave.magnetic(mu=1.0)

        np.testing.assert_allclose(
            _tangential([wave], mu=1.0),
            np.array([wave.electric[0], wave.electric[1], h[0], h[1]], dtype=complex),
            atol=1e-12,
        )

    def test_single_layer_boundary_system_has_small_residual(self):
        theta_air = math.radians(25.0)
        n_air = 1.0
        n_layer = 1.6
        n_sub = 1.2
        theta_layer = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_layer + 0j)))
        theta_sub = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_sub + 0j)))

        solution = solve_multilayer_boundary(
            top_known=[
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="s", omega=1.3),
            ],
            top_unknown_basis=[
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=1.3),
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=1.3),
            ],
            layer_unknown_basis=[
                [
                    make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="s", omega=1.3),
                    make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="p", omega=1.3),
                    make_isotropic_wave(
                        n=n_layer,
                        theta_rad=theta_layer,
                        polarization="s",
                        direction_sign_z=-1,
                        omega=1.3,
                    ),
                    make_isotropic_wave(
                        n=n_layer,
                        theta_rad=theta_layer,
                        polarization="p",
                        direction_sign_z=-1,
                        omega=1.3,
                    ),
                ]
            ],
            substrate_unknown_basis=[
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="s", omega=1.3),
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="p", omega=1.3),
            ],
            thicknesses=[0.7],
            mu=1.0,
        )
        self.assertEqual(solution.coefficients.shape, (8,))
        self.assertTrue(np.linalg.norm(solution.residual) < 1e-10)
        self.assertTrue(np.all(np.isfinite(solution.coefficients)))

    def test_boundary_system_exposes_mathematica_equation_and_unknown_order(self):
        theta_air = math.radians(25.0)
        n_air = 1.0
        n_layer = 1.6
        n_sub = 1.2
        theta_layer = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_layer + 0j)))
        theta_sub = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_sub + 0j)))
        top_known = [make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="s", omega=1.3)]
        top_basis = [
            make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=1.3),
            make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=1.3),
        ]
        layer_basis = [
            [
                make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="s", omega=1.3),
                make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="p", omega=1.3),
                make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="s", direction_sign_z=-1, omega=1.3),
                make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="p", direction_sign_z=-1, omega=1.3),
            ]
        ]
        substrate_basis = [
            make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="s", omega=1.3),
            make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="p", omega=1.3),
        ]

        system = build_multilayer_boundary_system(
            top_known=top_known,
            top_unknown_basis=top_basis,
            layer_unknown_basis=layer_basis,
            substrate_unknown_basis=substrate_basis,
            thicknesses=[0.7],
            mu=1.0,
        )
        solution = solve_multilayer_boundary(
            top_known=top_known,
            top_unknown_basis=top_basis,
            layer_unknown_basis=layer_basis,
            substrate_unknown_basis=substrate_basis,
            thicknesses=[0.7],
            mu=1.0,
        )

        self.assertEqual(system.matrix.shape, (8, 8))
        self.assertEqual(system.rhs.shape, (8,))
        self.assertEqual(
            system.equation_labels,
            ["top:E_x", "top:E_y", "top:H_x", "top:H_y", "bottom:E_x", "bottom:E_y", "bottom:H_x", "bottom:H_y"],
        )
        self.assertEqual(
            system.equation_metadata[3],
            {
                "row_index": 3,
                "label": "top:H_y",
                "interface": "top",
                "interface_index": 0,
                "component": "H_y",
                "field": "H",
                "axis": "y",
            },
        )
        self.assertEqual(
            system.equation_metadata[4],
            {
                "row_index": 4,
                "label": "bottom:E_x",
                "interface": "bottom",
                "interface_index": 1,
                "component": "E_x",
                "field": "E",
                "axis": "x",
            },
        )
        self.assertEqual(
            system.unknown_labels,
            [
                "top_reflected_mode_1",
                "top_reflected_mode_2",
                "layer_1_forward_mode_1",
                "layer_1_forward_mode_2",
                "layer_1_backward_mode_1",
                "layer_1_backward_mode_2",
                "substrate_forward_mode_1",
                "substrate_forward_mode_2",
            ],
        )
        self.assertEqual(
            system.unknown_metadata[0],
            {
                "column_index": 0,
                "label": "top_reflected_mode_1",
                "region": "top",
                "layer_index": None,
                "basis_index": 1,
                "propagation": "reflected",
                "mode_index": 1,
            },
        )
        self.assertEqual(
            system.unknown_metadata[4],
            {
                "column_index": 4,
                "label": "layer_1_backward_mode_1",
                "region": "layer",
                "layer_index": 1,
                "basis_index": 3,
                "propagation": "backward",
                "mode_index": 1,
            },
        )
        np.testing.assert_allclose(system.matrix @ solution.coefficients, system.rhs, atol=1e-10)
        np.testing.assert_allclose(np.linalg.solve(system.matrix, system.rhs), solution.coefficients, atol=1e-12)

    def test_two_layer_boundary_system_has_small_residual(self):
        theta_air = math.radians(18.0)
        n_air = 1.0
        n_layers = [1.45, 1.9]
        n_sub = 1.25
        theta_layers = [float(np.real(np.arcsin(n_air * np.sin(theta_air) / n + 0j))) for n in n_layers]
        theta_sub = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_sub + 0j)))

        layer_basis = []
        for n, theta in zip(n_layers, theta_layers):
            layer_basis.append(
                [
                    make_isotropic_wave(n=n, theta_rad=theta, polarization="s", omega=0.9),
                    make_isotropic_wave(n=n, theta_rad=theta, polarization="p", omega=0.9),
                    make_isotropic_wave(n=n, theta_rad=theta, polarization="s", direction_sign_z=-1, omega=0.9),
                    make_isotropic_wave(n=n, theta_rad=theta, polarization="p", direction_sign_z=-1, omega=0.9),
                ]
            )

        solution = solve_multilayer_boundary(
            top_known=[make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="p", omega=0.9)],
            top_unknown_basis=[
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=0.9),
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=0.9),
            ],
            layer_unknown_basis=layer_basis,
            substrate_unknown_basis=[
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="s", omega=0.9),
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="p", omega=0.9),
            ],
            thicknesses=[0.35, 0.5],
            mu=1.0,
        )
        self.assertEqual(solution.coefficients.shape, (12,))
        self.assertTrue(np.linalg.norm(solution.residual) < 1e-10)
        self.assertTrue(np.all(np.isfinite(solution.coefficients)))

    def test_single_layer_boundary_with_known_inhomogeneous_source_has_small_residual(self):
        theta_air = math.radians(15.0)
        n_air = 1.0
        n_layer = 1.7
        n_sub = 1.3
        omega = 2.0
        theta_layer = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_layer + 0j)))
        theta_sub = float(np.real(np.arcsin(n_air * np.sin(theta_air) / n_sub + 0j)))
        source = make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="p", omega=omega)
        source = type(source)(
            k=source.k,
            electric=np.array([0.08 + 0.03j, -0.05j, -0.02 + 0.01j], dtype=complex),
            omega=source.omega,
        )

        solution = solve_multilayer_boundary(
            top_known=[],
            layer_known=[[source]],
            top_unknown_basis=[
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=omega),
                make_isotropic_wave(n=n_air, theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=omega),
            ],
            layer_unknown_basis=[
                [
                    make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="s", omega=omega),
                    make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="p", omega=omega),
                    make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="s", direction_sign_z=-1, omega=omega),
                    make_isotropic_wave(n=n_layer, theta_rad=theta_layer, polarization="p", direction_sign_z=-1, omega=omega),
                ]
            ],
            substrate_unknown_basis=[
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="s", omega=omega),
                make_isotropic_wave(n=n_sub, theta_rad=theta_sub, polarization="p", omega=omega),
            ],
            thicknesses=[0.42],
            mu=1.0,
        )

        self.assertEqual(solution.coefficients.shape, (8,))
        self.assertTrue(np.linalg.norm(solution.residual) < 1e-10)
        self.assertGreater(np.linalg.norm(solution.coefficients), 0.0)


if __name__ == "__main__":
    unittest.main()
