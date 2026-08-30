"""Symbolic layer-thickness (h) multilayer boundary solve (gui_layer_thickness_symbolic_h).

The symbolic-h solve keeps each layer thickness as a sympy symbol (eigenmodes are
numeric -- they depend on the permittivity tensor, not on h; h enters only through
the exp(i k_z h) propagation phase). Substituting the real thickness must reproduce
the validated numeric ``solve_multilayer_boundary`` exactly (to machine precision),
which is itself live-Mathematica-validated through the Maker-fringe / sample-rotation
agreement -- so this transitively validates the closed-form-in-h amplitudes.
"""

import math
import unittest

import numpy as np
import sympy as sp

from shaarp.multilayer_boundary import solve_multilayer_boundary
from shaarp.multilayer_boundary_symbolic import solve_multilayer_boundary_symbolic_thickness
from shaarp.waves import make_isotropic_wave

OMEGA = 1.3


def _snell(n1, n2, theta1):
    return math.asin(n1 * math.sin(theta1) / n2)


def _wave(n, theta, pol, dz=1):
    return make_isotropic_wave(n=n, theta_rad=theta, polarization=pol, direction_sign_z=dz, omega=OMEGA)


class SymbolicThicknessBoundaryTests(unittest.TestCase):
    def test_single_film_symbolic_h_reduces_to_numeric(self):
        n_air, n_film, n_sub = 1.0, 2.1, 1.5
        th_a = math.radians(25.0)
        th_f = _snell(n_air, n_film, th_a)
        th_s = _snell(n_air, n_sub, th_a)
        h0 = 0.7

        top_known = [_wave(n_air, th_a, "s")]
        top_basis = [_wave(n_air, th_a, "s", -1), _wave(n_air, th_a, "p", -1)]
        film_basis = [
            _wave(n_film, th_f, "s"), _wave(n_film, th_f, "p"),
            _wave(n_film, th_f, "s", -1), _wave(n_film, th_f, "p", -1),
        ]
        sub_basis = [_wave(n_sub, th_s, "s"), _wave(n_sub, th_s, "p")]

        numeric = solve_multilayer_boundary(
            top_known=top_known, top_unknown_basis=top_basis,
            layer_unknown_basis=[film_basis], substrate_unknown_basis=sub_basis, thicknesses=[h0],
        )
        ref = np.asarray(numeric.coefficients, dtype=complex)

        h = sp.Symbol("h", real=True)
        symbolic = solve_multilayer_boundary_symbolic_thickness(
            top_known=top_known, top_unknown_basis=top_basis,
            layer_unknown_basis=[film_basis], substrate_unknown_basis=sub_basis, thickness_symbols=[h],
        )
        np.testing.assert_allclose(np.asarray(symbolic.at(h0), dtype=complex), ref, atol=1e-11, rtol=1e-11)

        # Robust analytical-agreement check: do NOT compare the (messy) symbolic form.
        # Evaluate the closed-form expression at many generic thicknesses and require it
        # to match the numeric solve at EACH (Schwartz-Zippel: agreement at enough generic
        # points => identical functions). This validates the form over the domain, not at one point.
        for hv in (0.05, 0.17, 0.31, 0.5, 0.73, 0.9, 1.2, 1.55, 2.0, 3.3):
            numeric_hv = solve_multilayer_boundary(
                top_known=top_known, top_unknown_basis=top_basis,
                layer_unknown_basis=[film_basis], substrate_unknown_basis=sub_basis, thicknesses=[hv],
            )
            np.testing.assert_allclose(
                np.asarray(symbolic.at(hv), dtype=complex),
                np.asarray(numeric_hv.coefficients, dtype=complex),
                atol=1e-10, rtol=1e-10,
            )
        # genuinely a closed-form function of h (not a constant)
        self.assertTrue(any(h in expr.free_symbols for expr in symbolic.coefficients))
        self.assertGreater(
            np.max(np.abs(np.asarray(symbolic.at(0.31), dtype=complex) - np.asarray(symbolic.at(2.0), dtype=complex))),
            1e-6,
        )

    def test_two_layer_symbolic_h_reduces_to_numeric(self):
        n_air, n1, n2, n_sub = 1.0, 2.1, 1.7, 1.5
        th_a = math.radians(20.0)
        th_1 = _snell(n_air, n1, th_a)
        th_2 = _snell(n_air, n2, th_a)
        th_s = _snell(n_air, n_sub, th_a)
        h1, h2 = 0.6, 0.45

        top_known = [_wave(n_air, th_a, "s")]
        top_basis = [_wave(n_air, th_a, "s", -1), _wave(n_air, th_a, "p", -1)]
        layer1 = [_wave(n1, th_1, "s"), _wave(n1, th_1, "p"), _wave(n1, th_1, "s", -1), _wave(n1, th_1, "p", -1)]
        layer2 = [_wave(n2, th_2, "s"), _wave(n2, th_2, "p"), _wave(n2, th_2, "s", -1), _wave(n2, th_2, "p", -1)]
        sub_basis = [_wave(n_sub, th_s, "s"), _wave(n_sub, th_s, "p")]

        numeric = solve_multilayer_boundary(
            top_known=top_known, top_unknown_basis=top_basis,
            layer_unknown_basis=[layer1, layer2], substrate_unknown_basis=sub_basis, thicknesses=[h1, h2],
        )
        ref = np.asarray(numeric.coefficients, dtype=complex)

        sh1, sh2 = sp.symbols("h1 h2", real=True)
        symbolic = solve_multilayer_boundary_symbolic_thickness(
            top_known=top_known, top_unknown_basis=top_basis,
            layer_unknown_basis=[layer1, layer2], substrate_unknown_basis=sub_basis, thickness_symbols=[sh1, sh2],
        )
        got = np.asarray(symbolic.at(h1, h2), dtype=complex)
        np.testing.assert_allclose(got, ref, atol=1e-11, rtol=1e-11)

    def test_symbolic_boundary_with_known_source_reduces_to_numeric(self):
        """Boundary-half of the symbolic-h 2-omega SHG solve: a KNOWN inhomogeneous
        source wave inside the film moves to the RHS with symbolic propagation, and
        the closed-form-in-h amplitudes reproduce the numeric solve over a sweep."""
        from dataclasses import replace

        n_air, n_film, n_sub = 1.0, 2.1, 1.5
        th_a = math.radians(22.0)
        th_f = _snell(n_air, n_film, th_a)
        th_s = _snell(n_air, n_sub, th_a)

        top_known = [_wave(n_air, th_a, "s")]
        top_basis = [_wave(n_air, th_a, "s", -1), _wave(n_air, th_a, "p", -1)]
        film_basis = [_wave(n_film, th_f, "s"), _wave(n_film, th_f, "p"),
                      _wave(n_film, th_f, "s", -1), _wave(n_film, th_f, "p", -1)]
        sub_basis = [_wave(n_sub, th_s, "s"), _wave(n_sub, th_s, "p")]
        base = _wave(n_film, th_f, "s")
        source = replace(base, electric=base.electric * (0.3 + 0.2j))  # fixed in-film source

        h = sp.Symbol("h", real=True)
        symbolic = solve_multilayer_boundary_symbolic_thickness(
            top_known=top_known, top_unknown_basis=top_basis, layer_unknown_basis=[film_basis],
            substrate_unknown_basis=sub_basis, thickness_symbols=[h], layer_known=[[source]],
        )
        for hv in (0.1, 0.25, 0.4, 0.6, 0.85, 1.1, 1.5, 2.2):
            numeric = solve_multilayer_boundary(
                top_known=top_known, top_unknown_basis=top_basis, layer_unknown_basis=[film_basis],
                substrate_unknown_basis=sub_basis, thicknesses=[hv], layer_known=[[source]],
            )
            np.testing.assert_allclose(
                np.asarray(symbolic.at(hv), dtype=complex),
                np.asarray(numeric.coefficients, dtype=complex), atol=1e-10, rtol=1e-10,
            )

    def test_complex_absorbing_film_symbolic_h_reduces_to_numeric(self):
        """ML partial-analytical (symbolic-h) must handle a COMPLEX (absorbing) film:
        the closed-form-in-h amplitudes carry exp(i k_z h) with a COMPLEX k_z (so the
        film attenuates), and substituting the thickness must reproduce the numeric
        solve over a sweep. This is the ML analogue of the SI complex-coefficient
        validation -- the unified method must work for absorbing media on both sides."""
        import cmath

        n_air = 1.0
        n_film = complex(2.1, 0.45)   # absorbing film, Im(n) > 0
        n_sub = complex(1.5, 0.08)    # weakly absorbing substrate
        th_a = math.radians(23.0)

        def csnell(n1, n2, th1):
            return cmath.asin(n1 * cmath.sin(th1) / n2)

        th_f = csnell(n_air, n_film, th_a)
        th_s = csnell(n_air, n_sub, th_a)

        top_known = [_wave(n_air, th_a, "s")]
        top_basis = [_wave(n_air, th_a, "s", -1), _wave(n_air, th_a, "p", -1)]
        film_basis = [
            _wave(n_film, th_f, "s"), _wave(n_film, th_f, "p"),
            _wave(n_film, th_f, "s", -1), _wave(n_film, th_f, "p", -1),
        ]
        sub_basis = [_wave(n_sub, th_s, "s"), _wave(n_sub, th_s, "p")]

        h = sp.Symbol("h", real=True)
        symbolic = solve_multilayer_boundary_symbolic_thickness(
            top_known=top_known, top_unknown_basis=top_basis,
            layer_unknown_basis=[film_basis], substrate_unknown_basis=sub_basis, thickness_symbols=[h],
        )
        for hv in (0.05, 0.2, 0.5, 0.9, 1.4, 2.1, 3.0):
            numeric = solve_multilayer_boundary(
                top_known=top_known, top_unknown_basis=top_basis,
                layer_unknown_basis=[film_basis], substrate_unknown_basis=sub_basis, thicknesses=[hv],
            )
            np.testing.assert_allclose(
                np.asarray(symbolic.at(hv), dtype=complex),
                np.asarray(numeric.coefficients, dtype=complex), atol=1e-10, rtol=1e-10,
            )
        # the +z film modes must DECAY (Im k_z > 0) so exp(i k_z h) attenuates with depth
        for w in (film_basis[0], film_basis[1]):  # forward s, p
            self.assertGreater(complex(w.k[2]).imag, 0.0, "forward film mode must decay into the absorber")

    def test_at_validates_thickness_count(self):
        h = sp.Symbol("h", real=True)
        top_known = [_wave(1.0, 0.0, "s")]
        top_basis = [_wave(1.0, 0.0, "s", -1), _wave(1.0, 0.0, "p", -1)]
        film = [_wave(2.0, 0.0, "s"), _wave(2.0, 0.0, "p"), _wave(2.0, 0.0, "s", -1), _wave(2.0, 0.0, "p", -1)]
        sub = [_wave(1.5, 0.0, "s"), _wave(1.5, 0.0, "p")]
        sol = solve_multilayer_boundary_symbolic_thickness(
            top_known=top_known, top_unknown_basis=top_basis,
            layer_unknown_basis=[film], substrate_unknown_basis=sub, thickness_symbols=[h],
        )
        with self.assertRaises(ValueError):
            sol.at(0.5, 0.6)


if __name__ == "__main__":
    unittest.main()
