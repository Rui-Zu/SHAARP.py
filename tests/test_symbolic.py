import unittest

import numpy as np
import sympy as sp

from shaarp.nonlinear import compute_pnl_voigt
from shaarp.nonlinear import (
    NonlinearSource,
    single_interface_sources,
    solve_inhomogeneous_field,
    solve_single_interface_inhomogeneous_fields,
)
from shaarp.symbolic import (
    SymbolicNonlinearSource,
    SymbolicWave,
    boundary_residual_symbolic,
    compute_pnl_voigt_symbolic,
    d_voigt_symbolic,
    make_isotropic_wave_symbolic,
    partial_pnl_expressions,
    rotate_d_voigt_symbolic,
    single_interface_sources_symbolic,
    solve_boundary_continuity_symbolic,
    solve_fresnel_boundary_symbolic,
    solve_inhomogeneous_field_symbolic,
    solve_isotropic_linear_interface_symbolic,
    solve_isotropic_single_interface_shg_symbolic,
    solve_single_interface_inhomogeneous_fields_symbolic,
    solve_single_interface_shg_boundary_symbolic,
    solve_single_interface_shg_known_waves_symbolic,
)
from shaarp.boundary import boundary_residual, isotropic_fresnel_coefficients, solve_boundary_continuity, solve_fresnel_boundary
from shaarp.waves import make_isotropic_wave
from shaarp.waves import Wave


class SymbolicTests(unittest.TestCase):
    def test_point_group_43m_matches_shaarp_partial_pattern(self):
        d = d_voigt_symbolic("-43m")
        d14 = sp.Symbol("d14")
        expected = sp.Matrix(
            [
                [0, 0, 0, d14, 0, 0],
                [0, 0, 0, 0, d14, 0],
                [0, 0, 0, 0, 0, d14],
            ]
        )
        self.assertEqual(d, expected)

    def test_point_group_32_matches_shaarp_partial_pattern(self):
        d = d_voigt_symbolic("32")
        d11 = sp.Symbol("d11")
        d14 = sp.Symbol("d14")
        expected = sp.Matrix(
            [
                [d11, -d11, 0, d14, 0, 0],
                [0, 0, 0, 0, -d14, -d11],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertEqual(d, expected)

    def test_point_group_3m_matches_shaarp_partial_pattern(self):
        d = d_voigt_symbolic("3m")
        d15 = sp.Symbol("d15")
        d22 = sp.Symbol("d22")
        d31 = sp.Symbol("d31")
        d33 = sp.Symbol("d33")
        expected = sp.Matrix(
            [
                [0, 0, 0, 0, d15, -d22],
                [-d22, d22, 0, d15, 0, 0],
                [d31, d31, d33, 0, 0, 0],
            ]
        )
        self.assertEqual(d, expected)

    def test_mathematica_point_group_labels_match_ascii_aliases(self):
        aliases = {
            "Overscript[4, _]": "-4",
            "Overscript[4, _]2m": "-42m",
            "Overscript[6, _]": "-6",
            "Overscript[6, _]m2": "-6m2",
            "Overscript[4, _]3m": "-43m",
            "\\!\\(\\*OverscriptBox[\\(4\\), \\(_\\)]\\)": "-4",
            "\\!\\(\\*OverscriptBox[\\(4\\), \\(_\\)]\\)2m": "-42m",
            "\\!\\(\\*OverscriptBox[\\(4\\), \\(_\\)]\\)3m": "-43m",
            "\\!\\(\\*OverscriptBox[\\(6\\), \\(_\\)]\\)": "-6",
            "\\!\\(\\*OverscriptBox[\\(6\\), \\(_\\)]\\)m2": "-6m2",
            "\\[Infinity]": "inf",
            "\\[Infinity]m": "infm",
            "\\[Infinity]2": "inf2",
        }
        for mathematica_label, ascii_label in aliases.items():
            with self.subTest(mathematica_label=mathematica_label):
                self.assertEqual(d_voigt_symbolic(mathematica_label), d_voigt_symbolic(ascii_label))

    def test_symbolic_compute_pnl_matches_numeric_same_and_mixed_sources(self):
        d_symbols = sp.Matrix([[sp.Symbol(f"d{i}{j}") for j in range(1, 7)] for i in range(1, 4)])
        e1_symbols = sp.Matrix(sp.symbols("e1x e1y e1z"))
        e2_symbols = sp.Matrix(sp.symbols("e2x e2y e2z"))
        d_values = np.array(
            [
                [0.1 + 0.02j, -0.2, 0.03j, 0.4, -0.5j, 0.6],
                [-0.15j, 0.25, -0.35, 0.45j, 0.55, -0.65],
                [0.7, -0.8j, 0.9, -1.0, 1.1j, -1.2],
            ],
            dtype=complex,
        )
        e1 = np.array([0.3 + 0.1j, -0.2, 0.5j], dtype=complex)
        e2 = np.array([-0.4j, 0.6 + 0.2j, -0.7], dtype=complex)

        substitutions = {}
        for i in range(3):
            for j in range(6):
                substitutions[d_symbols[i, j]] = d_values[i, j]
        for symbol, value in zip(e1_symbols, e1):
            substitutions[symbol] = value
        for symbol, value in zip(e2_symbols, e2):
            substitutions[symbol] = value

        symbolic_same = compute_pnl_voigt_symbolic(d_symbols, e1_symbols).subs(substitutions)
        symbolic_mixed = compute_pnl_voigt_symbolic(d_symbols, e1_symbols, e2_symbols, factor=2).subs(substitutions)

        same = np.array([complex(value.evalf()) for value in symbolic_same], dtype=complex)
        mixed = np.array([complex(value.evalf()) for value in symbolic_mixed], dtype=complex)

        np.testing.assert_allclose(same, compute_pnl_voigt(d_values, e1, e1), atol=1e-12)
        np.testing.assert_allclose(mixed, compute_pnl_voigt(d_values, e1, e2, factor=2), atol=1e-12)

    def test_partial_pnl_expressions_returns_three_sources(self):
        ex, ey, ez = sp.symbols("ex ey ez")
        ox, oy, oz = sp.symbols("ox oy oz")
        out = partial_pnl_expressions("3m", sp.Matrix([ex, ey, ez]), sp.Matrix([ox, oy, oz]))
        self.assertEqual(set(out), {"ee", "oo", "eo"})
        self.assertEqual(out["ee"].shape, (3, 1))
        self.assertEqual(out["oo"].shape, (3, 1))
        self.assertEqual(out["eo"].shape, (3, 1))

    def test_symbolic_d_rotation_identity_preserves_tensor(self):
        d = d_voigt_symbolic("1")
        rotated = rotate_d_voigt_symbolic(d, sp.eye(3))
        self.assertEqual(rotated, d)

    def test_symbolic_d_rotation_matches_direct_numeric_loop(self):
        d_values = np.array(
            [
                [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
                [0.7, -0.8, 0.9, -1.0, 1.1, -1.2],
                [1.3, -1.4, 1.5, -1.6, 1.7, -1.8],
            ],
            dtype=float,
        )
        angle = 0.37
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        d_symbols = sp.Matrix([[sp.Symbol(f"d{i}{j}") for j in range(1, 7)] for i in range(1, 4)])
        a_symbols = sp.Matrix([[sp.Symbol(f"a{i}{j}") for j in range(1, 4)] for i in range(1, 4)])
        rotated = rotate_d_voigt_symbolic(d_symbols, a_symbols, simplify=False)
        substitutions = {}
        for i in range(3):
            for j in range(6):
                substitutions[d_symbols[i, j]] = d_values[i, j]
            for j in range(3):
                substitutions[a_symbols[i, j]] = rotation[i, j]

        symbolic_values = np.array(
            [[float(rotated[i, j].subs(substitutions).evalf()) for j in range(6)] for i in range(3)]
        )
        np.testing.assert_allclose(symbolic_values, _direct_rotate_d_voigt(d_values, rotation), atol=1e-12)

    def test_symbolic_s_fresnel_boundary_matches_numeric_solver(self):
        n1, n2 = 1.0, 1.7
        theta_i = 0.41
        theta_t = float(np.real(isotropic_fresnel_coefficients(n1, n2, theta_i)["theta_t"]))

        symbolic = _symbolic_isotropic_boundary_coeffs(n1, n2, theta_i, theta_t, "s")
        numeric = _numeric_isotropic_boundary_coeffs(n1, n2, theta_i, theta_t, "s")
        np.testing.assert_allclose(symbolic, numeric, atol=1e-12)

    def test_symbolic_p_fresnel_boundary_matches_numeric_solver(self):
        n1, n2 = 1.0, 1.7
        theta_i = 0.41
        theta_t = float(np.real(isotropic_fresnel_coefficients(n1, n2, theta_i)["theta_t"]))

        symbolic = _symbolic_isotropic_boundary_coeffs(n1, n2, theta_i, theta_t, "p")
        numeric = _numeric_isotropic_boundary_coeffs(n1, n2, theta_i, theta_t, "p")
        np.testing.assert_allclose(symbolic, numeric, atol=1e-12)

    def test_symbolic_isotropic_linear_interface_matches_numeric_complex_indices(self):
        n1 = 1.0 + 0.015j
        n2 = 1.68 + 0.07j
        theta_i = 0.37
        for polarization in ("s", "p"):
            symbolic = solve_isotropic_linear_interface_symbolic(
                incident_index=n1,
                transmitted_index=n2,
                incident_theta_rad=theta_i,
                incident_polarization=polarization,
                mu=1,
                simplify=False,
            )
            theta_t = np.arcsin(n1 * np.sin(theta_i) / n2)
            numeric = _numeric_isotropic_boundary_coeffs(n1, n2, theta_i, theta_t, polarization)
            symbolic_coeffs = np.array([complex(value.evalf()) for value in symbolic.coefficients], dtype=complex)
            np.testing.assert_allclose(symbolic_coeffs, numeric, atol=1e-12)

    def test_symbolic_boundary_with_known_bottom_source_matches_numeric_residual(self):
        n1, n2 = 1.0, 1.5
        theta_i = 0.29
        theta_t = float(np.real(isotropic_fresnel_coefficients(n1, n2, theta_i)["theta_t"]))
        top_known = []
        bottom_known = [
            make_isotropic_wave_symbolic(
                n=n2,
                theta_rad=theta_t,
                polarization="s",
                amplitude=0.13 + 0.07j,
                omega=2,
            )
        ]
        top_basis = [
            make_isotropic_wave_symbolic(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1, omega=2),
            make_isotropic_wave_symbolic(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1, omega=2),
        ]
        bottom_basis = [
            make_isotropic_wave_symbolic(n=n2, theta_rad=theta_t, polarization="s", omega=2),
            make_isotropic_wave_symbolic(n=n2, theta_rad=theta_t, polarization="p", omega=2),
        ]
        top_unknown, bottom_unknown, coeffs = solve_boundary_continuity_symbolic(
            top_known=top_known,
            bottom_known=bottom_known,
            top_unknown_basis=top_basis,
            bottom_unknown_basis=bottom_basis,
            mu=1,
        )
        residual = boundary_residual_symbolic(top_known + top_unknown, bottom_known + bottom_unknown, mu=1)
        residual_values = np.array([complex(value.evalf()) for value in residual], dtype=complex)
        self.assertLess(np.linalg.norm(residual_values), 1e-12)

        numeric_top_known = []
        numeric_bottom_known = [
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s", amplitude=0.13 + 0.07j, omega=2)
        ]
        numeric_top_basis = [
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1, omega=2),
            make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1, omega=2),
        ]
        numeric_bottom_basis = [
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s", omega=2),
            make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="p", omega=2),
        ]
        numeric_top, numeric_bottom, numeric_coeffs = solve_boundary_continuity(
            top_known=numeric_top_known,
            bottom_known=numeric_bottom_known,
            top_unknown_basis=numeric_top_basis,
            bottom_unknown_basis=numeric_bottom_basis,
            mu=1,
        )
        coeff_values = np.array([complex(value.evalf()) for value in coeffs], dtype=complex)
        np.testing.assert_allclose(coeff_values, numeric_coeffs, atol=1e-12)
        numeric_residual = boundary_residual(numeric_top_known + numeric_top, numeric_bottom_known + numeric_bottom, mu=1)
        self.assertLess(np.linalg.norm(numeric_residual), 1e-12)

    def test_symbolic_inhomogeneous_field_matches_numeric_solver(self):
        epsilon = np.array(
            [
                [(2.2 + 0.08j) ** 2, 0.04 + 0.01j, -0.03j],
                [0.04 + 0.01j, (2.5 + 0.06j) ** 2, 0.02 - 0.01j],
                [-0.03j, 0.02 - 0.01j, (2.9 + 0.05j) ** 2],
            ],
            dtype=complex,
        )
        k = np.array([0.31 + 0.02j, 0.0, 1.27 + 0.08j], dtype=complex)
        polarization = np.array([0.2 - 0.04j, -0.1 + 0.07j, 0.05 + 0.09j], dtype=complex)
        source = NonlinearSource(label="test", k=k, polarization=polarization, omega=2.0)
        numeric = solve_inhomogeneous_field(epsilon, source, mu=1, eps0=1)

        symbolic_source = SymbolicNonlinearSource(
            label="test",
            k=sp.Matrix(k),
            polarization=sp.Matrix(polarization),
            omega=sp.Float(2.0),
        )
        symbolic = solve_inhomogeneous_field_symbolic(sp.Matrix(epsilon), symbolic_source, mu=1, eps0=1, simplify=False)
        electric = np.array([complex(value.evalf()) for value in symbolic.electric], dtype=complex)
        residual = np.array([complex(value.evalf()) for value in symbolic.residual], dtype=complex)

        np.testing.assert_allclose(electric, numeric.electric, atol=1e-12)
        self.assertLess(np.linalg.norm(residual), 1e-12)

    def test_symbolic_single_interface_sources_and_inhomogeneous_fields_match_numeric(self):
        d_voigt = np.array(
            [
                [0.2 + 0.01j, -0.1, 0.04j, 0.3, -0.2j, 0.15],
                [-0.05j, 0.12, -0.08, 0.11j, 0.07, -0.09],
                [0.18, -0.16j, 0.22, -0.14, 0.13j, -0.06],
            ],
            dtype=complex,
        )
        epsilon = np.diag([(1.7 + 0.02j) ** 2, (1.9 + 0.03j) ** 2, (2.1 + 0.04j) ** 2])
        fast = Wave(
            k=np.array([0.25 + 0.01j, 0.0, 1.1 + 0.03j], dtype=complex),
            electric=np.array([0.4, 0.2j, -0.1 + 0.05j], dtype=complex),
            omega=1.0,
        )
        slow = Wave(
            k=np.array([0.21 + 0.02j, 0.0, 1.25 + 0.04j], dtype=complex),
            electric=np.array([-0.05j, 0.35, 0.12 - 0.03j], dtype=complex),
            omega=1.0,
        )
        numeric_sources = single_interface_sources(d_voigt, fast, slow)
        numeric_fields = solve_single_interface_inhomogeneous_fields(epsilon, numeric_sources, mu=1, eps0=1)

        symbolic_fast = SymbolicWave(k=sp.Matrix(fast.k), electric=sp.Matrix(fast.electric), omega=1)
        symbolic_slow = SymbolicWave(k=sp.Matrix(slow.k), electric=sp.Matrix(slow.electric), omega=1)
        symbolic_sources = single_interface_sources_symbolic(sp.Matrix(d_voigt), symbolic_fast, symbolic_slow)
        symbolic_fields = solve_single_interface_inhomogeneous_fields_symbolic(
            sp.Matrix(epsilon),
            symbolic_sources,
            mu=1,
            eps0=1,
            simplify=False,
        )

        for label in ("ee", "oo", "eo"):
            symbolic_source = getattr(symbolic_sources, label)
            numeric_source = getattr(numeric_sources, label)
            symbolic_pol = np.array([complex(value.evalf()) for value in symbolic_source.polarization], dtype=complex)
            np.testing.assert_allclose(symbolic_pol, numeric_source.polarization, atol=1e-12)

            symbolic_field = getattr(symbolic_fields, label)
            numeric_field = getattr(numeric_fields, label)
            symbolic_e = np.array([complex(value.evalf()) for value in symbolic_field.electric], dtype=complex)
            symbolic_residual = np.array([complex(value.evalf()) for value in symbolic_field.residual], dtype=complex)
            np.testing.assert_allclose(symbolic_e, numeric_field.electric, atol=1e-12)
            self.assertLess(np.linalg.norm(symbolic_residual), 1e-11)

    def test_symbolic_inhomogeneous_field_nondiagonal_complex_epsilon_matches_numeric(self):
        d_voigt = _dense_complex_symbolic_d(scale=0.9)
        epsilon = np.array(
            [
                [(1.62 + 0.03j) ** 2, 0.025 + 0.008j, -0.014 + 0.006j],
                [0.025 + 0.008j, (1.84 + 0.04j) ** 2, 0.017 - 0.005j],
                [-0.014 + 0.006j, 0.017 - 0.005j, (2.05 + 0.05j) ** 2],
            ],
            dtype=complex,
        )
        fast_omega = Wave(
            k=np.array([0.18 + 0.02j, 0.0, 1.08 + 0.04j], dtype=complex),
            electric=np.array([0.25 + 0.03j, -0.12j, 0.18 - 0.02j], dtype=complex),
            omega=1.0,
        )
        slow_omega = Wave(
            k=np.array([0.16 + 0.01j, 0.0, 1.22 + 0.05j], dtype=complex),
            electric=np.array([-0.08j, 0.28 + 0.02j, -0.11 + 0.04j], dtype=complex),
            omega=1.0,
        )
        numeric_sources = single_interface_sources(d_voigt, fast_omega, slow_omega)
        numeric_inhomogeneous = solve_single_interface_inhomogeneous_fields(epsilon, numeric_sources, mu=1, eps0=1)
        symbolic_sources = single_interface_sources_symbolic(
            sp.Matrix(d_voigt),
            _symbolic_wave_from_numeric(fast_omega),
            _symbolic_wave_from_numeric(slow_omega),
        )
        symbolic_inhomogeneous = solve_single_interface_inhomogeneous_fields_symbolic(
            sp.Matrix(epsilon),
            symbolic_sources,
            mu=1,
            eps0=1,
            simplify=False,
        )
        for label in ("ee", "oo", "eo"):
            symbolic_field = getattr(symbolic_inhomogeneous, label)
            numeric_field = getattr(numeric_inhomogeneous, label)
            symbolic_e = np.array([complex(value.evalf()) for value in symbolic_field.electric], dtype=complex)
            symbolic_residual = np.array([complex(value.evalf()) for value in symbolic_field.residual], dtype=complex)
            np.testing.assert_allclose(symbolic_e, numeric_field.electric, atol=1e-12)
            self.assertLess(np.linalg.norm(symbolic_residual), 1e-11)
        self.assertGreater(np.max(np.abs(np.imag(epsilon))), 0.0)
        self.assertGreater(np.max(np.abs(epsilon - np.diag(np.diag(epsilon)))), 0.0)

    def test_symbolic_single_interface_shg_boundary_matches_numeric_boundary_stage(self):
        n_top = 1.0 + 0.01j
        n_bottom = 1.72 + 0.04j
        theta_top = 0.34
        theta_bottom = np.arcsin(n_top * np.sin(theta_top) / n_bottom)
        reflected_basis_numeric = [
            make_isotropic_wave(n=n_top, theta_rad=theta_top, polarization="s", direction_sign_z=-1, omega=2),
            make_isotropic_wave(n=n_top, theta_rad=theta_top, polarization="p", direction_sign_z=-1, omega=2),
        ]
        transmitted_basis_numeric = [
            make_isotropic_wave(n=n_bottom, theta_rad=theta_bottom, polarization="s", omega=2),
            make_isotropic_wave(n=n_bottom, theta_rad=theta_bottom, polarization="p", omega=2),
        ]
        inhomogeneous_numeric = [
            make_isotropic_wave(n=n_bottom, theta_rad=theta_bottom, polarization="s", amplitude=0.07 + 0.02j, omega=2),
            make_isotropic_wave(n=n_bottom, theta_rad=theta_bottom, polarization="p", amplitude=-0.03 + 0.05j, omega=2),
            make_isotropic_wave(n=n_bottom, theta_rad=theta_bottom, polarization="s", amplitude=0.02 - 0.01j, omega=2),
        ]
        numeric_reflected, numeric_transmitted, numeric_coeffs = solve_boundary_continuity(
            top_known=[],
            bottom_known=inhomogeneous_numeric,
            top_unknown_basis=reflected_basis_numeric,
            bottom_unknown_basis=transmitted_basis_numeric,
            mu=1,
        )
        numeric_residual = boundary_residual(numeric_reflected, numeric_transmitted + inhomogeneous_numeric, mu=1)

        reflected_basis_symbolic = [_symbolic_wave_from_numeric(wave) for wave in reflected_basis_numeric]
        transmitted_basis_symbolic = [_symbolic_wave_from_numeric(wave) for wave in transmitted_basis_numeric]
        inhomogeneous_symbolic = [_symbolic_wave_from_numeric(wave) for wave in inhomogeneous_numeric]
        symbolic_result = solve_single_interface_shg_boundary_symbolic(
            reflected_basis=reflected_basis_symbolic,
            transmitted_basis=transmitted_basis_symbolic,
            inhomogeneous=inhomogeneous_symbolic,
            mu=1,
            simplify=False,
        )
        symbolic_coeffs = np.array([complex(value.evalf()) for value in symbolic_result.coefficients], dtype=complex)
        symbolic_residual = np.array([complex(value.evalf()) for value in symbolic_result.boundary_residual], dtype=complex)

        np.testing.assert_allclose(symbolic_coeffs, numeric_coeffs, atol=1e-12)
        self.assertLess(np.linalg.norm(symbolic_residual), 1e-12)
        self.assertLess(np.linalg.norm(numeric_residual), 1e-12)

    def test_symbolic_single_interface_shg_known_waves_matches_numeric_pipeline(self):
        d_voigt = np.array(
            [
                [0.14 + 0.02j, -0.08, 0.05j, 0.22, -0.13j, 0.09],
                [-0.04j, 0.11, -0.07, 0.10j, 0.06, -0.05],
                [0.17, -0.12j, 0.19, -0.10, 0.08j, -0.03],
            ],
            dtype=complex,
        )
        epsilon = np.diag([(1.8 + 0.03j) ** 2, (2.0 + 0.04j) ** 2, (2.2 + 0.05j) ** 2])
        fast_omega = Wave(
            k=np.array([0.22 + 0.01j, 0.0, 1.05 + 0.03j], dtype=complex),
            electric=np.array([0.35, 0.18j, -0.08 + 0.04j], dtype=complex),
            omega=1.0,
        )
        slow_omega = Wave(
            k=np.array([0.19 + 0.02j, 0.0, 1.20 + 0.04j], dtype=complex),
            electric=np.array([-0.04j, 0.31, 0.10 - 0.02j], dtype=complex),
            omega=1.0,
        )
        reflected_basis = [
            make_isotropic_wave(n=1.0 + 0.01j, theta_rad=0.28, polarization="s", direction_sign_z=-1, omega=2),
            make_isotropic_wave(n=1.0 + 0.01j, theta_rad=0.28, polarization="p", direction_sign_z=-1, omega=2),
        ]
        n_bottom = 1.75 + 0.04j
        theta_bottom = np.arcsin((1.0 + 0.01j) * np.sin(0.28) / n_bottom)
        transmitted_basis = [
            make_isotropic_wave(n=n_bottom, theta_rad=theta_bottom, polarization="s", omega=2),
            make_isotropic_wave(n=n_bottom, theta_rad=theta_bottom, polarization="p", omega=2),
        ]

        numeric_sources = single_interface_sources(d_voigt, fast_omega, slow_omega)
        numeric_inhomogeneous = solve_single_interface_inhomogeneous_fields(epsilon, numeric_sources, mu=1, eps0=1)
        numeric_inhomogeneous_waves = [
            Wave(k=numeric_inhomogeneous.ee.k, electric=numeric_inhomogeneous.ee.electric, omega=numeric_inhomogeneous.ee.omega),
            Wave(k=numeric_inhomogeneous.oo.k, electric=numeric_inhomogeneous.oo.electric, omega=numeric_inhomogeneous.oo.omega),
            Wave(k=numeric_inhomogeneous.eo.k, electric=numeric_inhomogeneous.eo.electric, omega=numeric_inhomogeneous.eo.omega),
        ]
        _, _, numeric_coeffs = solve_boundary_continuity(
            top_known=[],
            bottom_known=numeric_inhomogeneous_waves,
            top_unknown_basis=reflected_basis,
            bottom_unknown_basis=transmitted_basis,
            mu=1,
        )

        symbolic_result = solve_single_interface_shg_known_waves_symbolic(
            epsilon_2omega_lab=sp.Matrix(epsilon),
            d_voigt_lab=sp.Matrix(d_voigt),
            transmitted_fast_omega=_symbolic_wave_from_numeric(fast_omega),
            transmitted_slow_omega=_symbolic_wave_from_numeric(slow_omega),
            reflected_basis_2omega=[_symbolic_wave_from_numeric(wave) for wave in reflected_basis],
            transmitted_basis_2omega=[_symbolic_wave_from_numeric(wave) for wave in transmitted_basis],
            mu=1,
            eps0=1,
            simplify=False,
        )
        symbolic_coeffs = np.array([complex(value.evalf()) for value in symbolic_result.boundary.coefficients], dtype=complex)
        symbolic_residual = np.array([complex(value.evalf()) for value in symbolic_result.boundary.boundary_residual], dtype=complex)
        np.testing.assert_allclose(symbolic_coeffs, numeric_coeffs, atol=1e-11)
        self.assertLess(np.linalg.norm(symbolic_residual), 1e-10)

    def test_symbolic_single_interface_shg_known_waves_dense_complex_basis_matches_numeric_pipeline(self):
        d_voigt = np.array(
            [
                [0.14 + 0.02j, -0.08, 0.05j, 0.22, -0.13j, 0.09],
                [-0.04j, 0.11, -0.07, 0.10j, 0.06, -0.05],
                [0.17, -0.12j, 0.19, -0.10, 0.08j, -0.03],
            ],
            dtype=complex,
        )
        epsilon = np.diag([(1.8 + 0.03j) ** 2, (2.0 + 0.04j) ** 2, (2.2 + 0.05j) ** 2])
        fast_omega = Wave(
            k=np.array([0.22 + 0.01j, 0.0, 1.05 + 0.03j], dtype=complex),
            electric=np.array([0.35, 0.18j, -0.08 + 0.04j], dtype=complex),
            omega=1.0,
        )
        slow_omega = Wave(
            k=np.array([0.19 + 0.02j, 0.0, 1.20 + 0.04j], dtype=complex),
            electric=np.array([-0.04j, 0.31, 0.10 - 0.02j], dtype=complex),
            omega=1.0,
        )
        reflected_basis = [
            make_isotropic_wave(n=1.0 + 0.01j, theta_rad=0.28, polarization="s", direction_sign_z=-1, omega=2),
            make_isotropic_wave(n=1.0 + 0.01j, theta_rad=0.28, polarization="p", direction_sign_z=-1, omega=2),
        ]
        transmitted_basis = [
            Wave(
                k=np.array([0.41 + 0.02j, 0.0, 2.0 + 0.08j], dtype=complex),
                electric=np.array([0.2, 0.7j, -0.05], dtype=complex),
                omega=2.0,
            ),
            Wave(
                k=np.array([0.39 + 0.03j, 0.0, 2.2 + 0.09j], dtype=complex),
                electric=np.array([-0.1j, 0.15, 0.65], dtype=complex),
                omega=2.0,
            ),
        ]

        numeric_sources = single_interface_sources(d_voigt, fast_omega, slow_omega)
        numeric_inhomogeneous = solve_single_interface_inhomogeneous_fields(epsilon, numeric_sources, mu=1, eps0=1)
        numeric_inhomogeneous_waves = [
            Wave(k=numeric_inhomogeneous.ee.k, electric=numeric_inhomogeneous.ee.electric, omega=numeric_inhomogeneous.ee.omega),
            Wave(k=numeric_inhomogeneous.oo.k, electric=numeric_inhomogeneous.oo.electric, omega=numeric_inhomogeneous.oo.omega),
            Wave(k=numeric_inhomogeneous.eo.k, electric=numeric_inhomogeneous.eo.electric, omega=numeric_inhomogeneous.eo.omega),
        ]
        _, _, numeric_coeffs = solve_boundary_continuity(
            top_known=[],
            bottom_known=numeric_inhomogeneous_waves,
            top_unknown_basis=reflected_basis,
            bottom_unknown_basis=transmitted_basis,
            mu=1,
        )

        symbolic_result = solve_single_interface_shg_known_waves_symbolic(
            epsilon_2omega_lab=sp.Matrix(epsilon),
            d_voigt_lab=sp.Matrix(d_voigt),
            transmitted_fast_omega=_symbolic_wave_from_numeric(fast_omega),
            transmitted_slow_omega=_symbolic_wave_from_numeric(slow_omega),
            reflected_basis_2omega=[_symbolic_wave_from_numeric(wave) for wave in reflected_basis],
            transmitted_basis_2omega=[_symbolic_wave_from_numeric(wave) for wave in transmitted_basis],
            mu=1,
            eps0=1,
            simplify=False,
        )
        symbolic_coeffs = np.array([complex(value.evalf()) for value in symbolic_result.boundary.coefficients], dtype=complex)
        symbolic_residual = np.array([complex(value.evalf()) for value in symbolic_result.boundary.boundary_residual], dtype=complex)
        np.testing.assert_allclose(symbolic_coeffs, numeric_coeffs, atol=1e-11)
        self.assertLess(np.linalg.norm(symbolic_residual), 1e-10)

    def test_symbolic_isotropic_single_interface_shg_matches_numeric_pipeline(self):
        n1w = 1.0 + 0.01j
        n2w = 1.55 + 0.04j
        n1_2w = 1.04 + 0.02j
        n2_2w = 1.78 + 0.05j
        theta_i = 0.31
        d_voigt = np.array(
            [
                [0.11 + 0.02j, -0.05, 0.04j, 0.18, -0.09j, 0.07],
                [-0.03j, 0.08, -0.06, 0.05j, 0.04, -0.02],
                [0.13, -0.10j, 0.16, -0.08, 0.06j, -0.01],
            ],
            dtype=complex,
        )
        epsilon_2w = np.eye(3, dtype=complex) * n2_2w**2

        symbolic = solve_isotropic_single_interface_shg_symbolic(
            transmitted_epsilon_2omega=sp.Matrix(epsilon_2w),
            d_voigt_lab=sp.Matrix(d_voigt),
            incident_index_omega=n1w,
            transmitted_index_omega=n2w,
            incident_index_2omega=n1_2w,
            transmitted_index_2omega=n2_2w,
            incident_theta_rad=theta_i,
            incident_polarization="p",
            mu=1,
            eps0=1,
            simplify=False,
        )

        theta_w = np.arcsin(n1w * np.sin(theta_i) / n2w)
        incident = [make_isotropic_wave(n=n1w, theta_rad=theta_i, polarization="p", omega=1)]
        reflected_w = [
            make_isotropic_wave(n=n1w, theta_rad=theta_i, polarization="s", direction_sign_z=-1, omega=1),
            make_isotropic_wave(n=n1w, theta_rad=theta_i, polarization="p", direction_sign_z=-1, omega=1),
        ]
        transmitted_w = [
            make_isotropic_wave(n=n2w, theta_rad=theta_w, polarization="s", omega=1),
            make_isotropic_wave(n=n2w, theta_rad=theta_w, polarization="p", omega=1),
        ]
        _, numeric_transmitted_w, numeric_linear_coeffs = solve_fresnel_boundary(
            incident,
            reflected_w,
            transmitted_w,
            mu=1,
        )
        theta_r_2w = np.arcsin(n1w * np.sin(theta_i) / n1_2w)
        theta_t_2w = np.arcsin(n1w * np.sin(theta_i) / n2_2w)
        reflected_2w = [
            make_isotropic_wave(n=n1_2w, theta_rad=theta_r_2w, polarization="s", direction_sign_z=-1, omega=2),
            make_isotropic_wave(n=n1_2w, theta_rad=theta_r_2w, polarization="p", direction_sign_z=-1, omega=2),
        ]
        transmitted_2w = [
            make_isotropic_wave(n=n2_2w, theta_rad=theta_t_2w, polarization="s", omega=2),
            make_isotropic_wave(n=n2_2w, theta_rad=theta_t_2w, polarization="p", omega=2),
        ]
        numeric_sources = single_interface_sources(d_voigt, numeric_transmitted_w[0], numeric_transmitted_w[1])
        numeric_inhomogeneous = solve_single_interface_inhomogeneous_fields(epsilon_2w, numeric_sources, mu=1, eps0=1)
        numeric_inhomogeneous_waves = [
            Wave(k=numeric_inhomogeneous.ee.k, electric=numeric_inhomogeneous.ee.electric, omega=numeric_inhomogeneous.ee.omega),
            Wave(k=numeric_inhomogeneous.oo.k, electric=numeric_inhomogeneous.oo.electric, omega=numeric_inhomogeneous.oo.omega),
            Wave(k=numeric_inhomogeneous.eo.k, electric=numeric_inhomogeneous.eo.electric, omega=numeric_inhomogeneous.eo.omega),
        ]
        _, _, numeric_shg_coeffs = solve_boundary_continuity(
            top_known=[],
            bottom_known=numeric_inhomogeneous_waves,
            top_unknown_basis=reflected_2w,
            bottom_unknown_basis=transmitted_2w,
            mu=1,
        )

        symbolic_linear_coeffs = np.array([complex(value.evalf()) for value in symbolic.linear_omega.coefficients], dtype=complex)
        symbolic_shg_coeffs = np.array([complex(value.evalf()) for value in symbolic.shg.boundary.coefficients], dtype=complex)
        symbolic_residual = np.array([complex(value.evalf()) for value in symbolic.shg.boundary.boundary_residual], dtype=complex)

        np.testing.assert_allclose(symbolic_linear_coeffs, numeric_linear_coeffs, atol=1e-12)
        np.testing.assert_allclose(symbolic_shg_coeffs, numeric_shg_coeffs, atol=1e-10)
        self.assertLess(np.linalg.norm(symbolic_residual), 1e-10)

    def test_symbolic_isotropic_shg_normal_incidence_all_complex_properties(self):
        n1w = 1.0 + 0.012j
        n2w = 1.52 + 0.045j
        n1_2w = 1.03 + 0.018j
        n2_2w = 1.76 + 0.055j
        d_voigt = _dense_complex_symbolic_d()
        epsilon_2w = sp.eye(3) * n2_2w**2

        symbolic = solve_isotropic_single_interface_shg_symbolic(
            transmitted_epsilon_2omega=epsilon_2w,
            d_voigt_lab=sp.Matrix(d_voigt),
            incident_index_omega=n1w,
            transmitted_index_omega=n2w,
            incident_index_2omega=n1_2w,
            transmitted_index_2omega=n2_2w,
            incident_theta_rad=0.0,
            incident_polarization="s",
            mu=1,
            eps0=1,
            simplify=False,
        )
        symbolic_residual = np.array([complex(value.evalf()) for value in symbolic.shg.boundary.boundary_residual], dtype=complex)
        self.assertLess(np.linalg.norm(symbolic_residual), 1e-10)
        self.assertGreater(np.max(np.abs(np.imag(d_voigt))), 0.0)
        self.assertGreater(abs(np.imag(n2_2w**2)), 0.0)
        self.assertTrue(abs(complex(symbolic.reflected_basis_2omega[0].k[0].evalf())) < 1e-12)
        self.assertTrue(abs(complex(symbolic.transmitted_basis_2omega[0].k[0].evalf())) < 1e-12)

    def test_symbolic_isotropic_shg_phase_matched_same_complex_dielectric(self):
        n1w = 1.0 + 0.02j
        n2 = 1.68 + 0.06j
        theta_i = 0.26
        d_voigt = _dense_complex_symbolic_d(scale=0.8)
        epsilon_same = sp.eye(3) * n2**2

        symbolic = solve_isotropic_single_interface_shg_symbolic(
            transmitted_epsilon_2omega=epsilon_same,
            d_voigt_lab=sp.Matrix(d_voigt),
            incident_index_omega=n1w,
            transmitted_index_omega=n2,
            incident_index_2omega=n1w,
            transmitted_index_2omega=n2,
            incident_theta_rad=theta_i,
            incident_polarization="p",
            mu=1,
            eps0=1,
            simplify=False,
        )
        symbolic_residual = np.array([complex(value.evalf()) for value in symbolic.shg.boundary.boundary_residual], dtype=complex)
        omega_theta = complex(symbolic.linear_omega.theta_transmitted.evalf())
        two_omega_theta = complex((sp.asin(n1w * sp.sin(theta_i) / n2)).evalf())
        self.assertLess(np.linalg.norm(symbolic_residual), 1e-10)
        self.assertTrue(np.isclose(omega_theta, two_omega_theta, atol=1e-12))
        self.assertGreater(np.max(np.abs(np.imag(d_voigt))), 0.0)
        self.assertGreater(abs(np.imag(n2**2)), 0.0)


def _direct_rotate_d_voigt(d_voigt, rotation):
    voigt = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
    out = np.zeros((3, 6), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                col = voigt[j, k]
                total = 0.0
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            total += rotation[i, l] * rotation[j, m] * rotation[k, n] * d_voigt[l, voigt[m, n]]
                out[i, col] = total
    return out


def _symbolic_isotropic_boundary_coeffs(n1, n2, theta_i, theta_t, polarization):
    incident = [make_isotropic_wave_symbolic(n=n1, theta_rad=theta_i, polarization=polarization)]
    reflected_basis = [
        make_isotropic_wave_symbolic(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1),
        make_isotropic_wave_symbolic(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1),
    ]
    transmitted_basis = [
        make_isotropic_wave_symbolic(n=n2, theta_rad=theta_t, polarization="s"),
        make_isotropic_wave_symbolic(n=n2, theta_rad=theta_t, polarization="p"),
    ]
    _, _, coeffs = solve_fresnel_boundary_symbolic(incident, reflected_basis, transmitted_basis, mu=1)
    return np.array([complex(value.evalf()) for value in coeffs], dtype=complex)


def _numeric_isotropic_boundary_coeffs(n1, n2, theta_i, theta_t, polarization):
    incident = [make_isotropic_wave(n=n1, theta_rad=theta_i, polarization=polarization)]
    reflected_basis = [
        make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="s", direction_sign_z=-1),
        make_isotropic_wave(n=n1, theta_rad=theta_i, polarization="p", direction_sign_z=-1),
    ]
    transmitted_basis = [
        make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="s"),
        make_isotropic_wave(n=n2, theta_rad=theta_t, polarization="p"),
    ]
    _, _, coeffs = solve_fresnel_boundary(incident, reflected_basis, transmitted_basis, mu=1)
    return coeffs


def _symbolic_wave_from_numeric(wave):
    return SymbolicWave(k=sp.Matrix(wave.k), electric=sp.Matrix(wave.electric), omega=wave.omega)


def _dense_complex_symbolic_d(scale=1.0):
    return scale * np.array(
        [
            [0.21 + 0.04j, -0.06 + 0.02j, 0.03 - 0.01j, 0.16 + 0.05j, -0.11j, 0.09 + 0.03j],
            [-0.02 + 0.08j, 0.12 - 0.05j, -0.07 + 0.03j, 0.06j, 0.10 - 0.04j, -0.05 + 0.01j],
            [0.14 - 0.02j, -0.08j, 0.19 + 0.06j, -0.12 + 0.04j, 0.05 + 0.10j, -0.03j],
        ],
        dtype=complex,
    )


class DTensorStructureTests(unittest.TestCase):
    """Headless analytical-dij structure summary (gui_shg_tensor_inputs_and_analytical_dij)."""

    def test_3m_structure(self):
        from shaarp.symbolic import d_tensor_structure

        s = d_tensor_structure("3m")
        self.assertEqual(s.independent_constants, ["d15", "d22", "d31", "d33"])
        self.assertEqual(s.independent_count, 4)
        entries = dict(s.nonzero_entries)
        # the standard 3m relations between Voigt entries
        self.assertEqual(entries["d16"], "-d22")
        self.assertEqual(entries["d21"], "-d22")
        self.assertEqual(entries["d24"], "d15")
        self.assertEqual(entries["d32"], "d31")

    def test_cubic_43m_single_constant(self):
        from shaarp.symbolic import d_tensor_structure

        s = d_tensor_structure("-43m")
        self.assertEqual(s.independent_constants, ["d14"])
        entries = dict(s.nonzero_entries)
        self.assertEqual(entries["d25"], "d14")
        self.assertEqual(entries["d36"], "d14")

    def test_triclinic_has_18_and_matches_d_voigt_symbolic(self):
        from shaarp.symbolic import d_tensor_structure, d_voigt_symbolic

        s = d_tensor_structure("1")
        self.assertEqual(s.independent_count, 18)
        d = d_voigt_symbolic("1")
        nz = sum(1 for r in range(3) for c in range(6) if d[r, c] != 0)
        self.assertEqual(len(s.nonzero_entries), nz)


if __name__ == "__main__":
    unittest.main()
