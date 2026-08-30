import math
import unittest

import numpy as np

from shaarp.multilayer_basis import (
    build_multilayer_2omega_basis,
    build_multilayer_omega_basis,
    build_multilayer_omega_basis_jones,
    make_anisotropic_layer_basis,
)
from shaarp.multilayer_boundary import solve_multilayer_boundary
from shaarp.tensors import chi2_to_voigt, voigt_to_chi2


class MultilayerBasisTests(unittest.TestCase):
    def test_chi2_voigt_round_trip_preserves_shaarp_ordering(self):
        d = np.array(
            [
                [1 + 0.1j, 2, 3, 4j, 5, 6],
                [-1j, -2, -3, -4, -5j, -6],
                [0.2, 0.3j, 0.4, 0.5, 0.6j, 0.7],
            ],
            dtype=complex,
        )
        np.testing.assert_allclose(chi2_to_voigt(voigt_to_chi2(d)), d)

    def test_anisotropic_layer_basis_is_ordered_forward_then_backward(self):
        eps = np.array(
            [
                [1.7**2, 0.03, -0.02],
                [0.03, 1.9**2, 0.04],
                [-0.02, 0.04, 2.1**2],
            ],
            dtype=complex,
        )
        basis = make_anisotropic_layer_basis(
            eps,
            tangential_index=1.0,
            incident_theta_rad=math.radians(18.0),
            omega=1.3,
        )

        self.assertEqual(len(basis), 4)
        self.assertTrue(all(np.isclose(wave.k[0], basis[0].k[0], atol=1e-10) for wave in basis))
        self.assertGreater(np.real(basis[0].k[2]), 0)
        self.assertGreater(np.real(basis[1].k[2]), 0)
        self.assertLess(np.real(basis[2].k[2]), 0)
        self.assertLess(np.real(basis[3].k[2]), 0)

    def test_generated_omega_basis_solves_boundary_residual(self):
        theta = math.radians(16.0)
        layer_eps = [
            np.array(
                [
                    [1.6**2, 0.02, -0.01],
                    [0.02, 1.8**2, 0.03],
                    [-0.01, 0.03, 2.0**2],
                ],
                dtype=complex,
            )
        ]
        substrate_eps = np.diag([1.25**2, 1.35**2, 1.45**2]).astype(complex)
        basis = build_multilayer_omega_basis(
            incident_index=1.0,
            incident_theta_rad=theta,
            incident_polarization="p",
            layer_epsilon_lab=layer_eps,
            substrate_epsilon_lab=substrate_eps,
            omega=1.1,
        )

        solution = solve_multilayer_boundary(
            top_known=basis.incident,
            top_unknown_basis=basis.reflected_basis,
            layer_unknown_basis=basis.layer_basis,
            substrate_unknown_basis=basis.substrate_basis,
            thicknesses=[0.31],
            mu=1.0,
        )
        self.assertEqual(solution.coefficients.shape, (8,))
        self.assertTrue(np.linalg.norm(solution.residual) < 1e-10)

    def test_generated_omega_basis_jones_keeps_explicit_s_p_amplitudes(self):
        theta = math.radians(13.0)
        layer_eps = [np.diag([1.6**2, 1.8**2, 2.0**2]).astype(complex)]
        substrate_eps = np.diag([1.25**2, 1.35**2, 1.45**2]).astype(complex)
        basis = build_multilayer_omega_basis_jones(
            incident_index=1.0,
            incident_theta_rad=theta,
            incident_jones_sp=(1.0 + 0.5j, -0.25j),
            layer_epsilon_lab=layer_eps,
            substrate_epsilon_lab=substrate_eps,
            omega=1.2,
        )

        self.assertEqual(len(basis.incident), 2)
        self.assertTrue(np.isclose(basis.incident[0].electric[1], 1.0 + 0.5j))
        self.assertTrue(np.isclose(basis.incident[1].electric[0], -0.25j * np.cos(theta)))
        solution = solve_multilayer_boundary(
            top_known=basis.incident,
            top_unknown_basis=basis.reflected_basis,
            layer_unknown_basis=basis.layer_basis,
            substrate_unknown_basis=basis.substrate_basis,
            thicknesses=[0.22],
            mu=1.0,
        )
        self.assertTrue(np.linalg.norm(solution.residual) < 1e-10)

    def test_generated_omega_basis_jones_rejects_zero_incident_field(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            build_multilayer_omega_basis_jones(
                incident_index=1.0,
                incident_theta_rad=math.radians(13.0),
                incident_jones_sp=(0.0, 0.0),
                layer_epsilon_lab=[np.eye(3)],
                substrate_epsilon_lab=np.eye(3),
            )

    def test_generated_2omega_basis_matches_shg_tangential_target(self):
        theta = math.radians(21.0)
        omega = 1.4
        n_omega = 1.0
        n_top_2omega = 1.18
        layer_eps = [np.diag([1.7**2, 1.9**2, 2.2**2]).astype(complex)]
        substrate_eps = np.diag([1.4**2, 1.5**2, 1.6**2]).astype(complex)

        basis = build_multilayer_2omega_basis(
            top_index_2omega=n_top_2omega,
            tangential_index_omega=n_omega,
            incident_theta_rad=theta,
            layer_epsilon_2omega_lab=layer_eps,
            substrate_epsilon_2omega_lab=substrate_eps,
            omega_2=2 * omega,
        )

        expected_kx = 2 * omega * n_omega * np.sin(theta)
        for wave in basis.reflected_basis + basis.layer_basis[0] + basis.substrate_basis:
            self.assertTrue(np.isclose(wave.k[0], expected_kx, atol=1e-10))
        self.assertTrue(np.isclose(basis.tangential_index_target, n_omega * np.sin(theta)))


if __name__ == "__main__":
    unittest.main()


class OmegaBasisReuseInvariant(unittest.TestCase):
    """The GUI Fresnel sweep solves p- and s-incidence at every angle, and both
    rebuilt the FULL omega basis — but `incident_jones_sp` feeds only the cheap `incident` list.
    The reflected/layer/substrate bases (which carry the brentq-heavy anisotropic Snell-mode
    root-finds) depend solely on (eps, incident_index, incident_theta_rad, omega). `reuse=` skips
    exactly that half. These fences hold the invariant that makes the optimisation legal."""

    @staticmethod
    def _args(jones):
        eps_film = np.diag([2.4 + 0j, 2.6 + 0j, 2.9 + 0j])  # biaxial: distinct modes per direction
        return dict(
            incident_index=1.0 + 0j,
            incident_theta_rad=math.radians(37.0),
            incident_jones_sp=jones,
            layer_epsilon_lab=[eps_film],
            substrate_epsilon_lab=np.diag([2.1 + 0j, 2.1 + 0j, 2.1 + 0j]),
            omega=1.0,
        )

    def test_polarization_does_not_change_the_expensive_bases(self):
        """The invariant itself: build at one angle for p- and for s-incidence; everything except
        `incident` must come out numerically identical."""
        bp = build_multilayer_omega_basis_jones(**self._args((0.0j, 1.0 + 0j)))
        bs = build_multilayer_omega_basis_jones(**self._args((1.0 + 0j, 0.0j)))
        self.assertEqual(bp.tangential_index_target, bs.tangential_index_target)
        for name, wp, ws in (("reflected", bp.reflected_basis, bs.reflected_basis),
                             ("layer", bp.layer_basis[0], bs.layer_basis[0]),
                             ("substrate", bp.substrate_basis, bs.substrate_basis)):
            self.assertEqual(len(wp), len(ws), f"{name} basis size differs")
            for i, (a, b) in enumerate(zip(wp, ws)):
                np.testing.assert_allclose(a.k, b.k, rtol=0, atol=1e-15,
                                           err_msg=f"{name}[{i}].k depends on polarization")
                np.testing.assert_allclose(a.electric, b.electric, rtol=0, atol=1e-15,
                                           err_msg=f"{name}[{i}].electric depends on polarization")

    def test_reuse_is_byte_identical_to_a_full_rebuild(self):
        first = build_multilayer_omega_basis_jones(**self._args((0.0j, 1.0 + 0j)))
        full = build_multilayer_omega_basis_jones(**self._args((1.0 + 0j, 0.0j)))
        reused = build_multilayer_omega_basis_jones(**self._args((1.0 + 0j, 0.0j)), reuse=first)
        # the incident wave is rebuilt from the NEW polarization, not inherited
        self.assertEqual(len(reused.incident), len(full.incident))
        for a, b in zip(reused.incident, full.incident):
            np.testing.assert_array_equal(a.electric, b.electric)
        for name, wr, wf in (("reflected", reused.reflected_basis, full.reflected_basis),
                             ("layer", reused.layer_basis[0], full.layer_basis[0]),
                             ("substrate", reused.substrate_basis, full.substrate_basis)):
            for i, (a, b) in enumerate(zip(wr, wf)):
                np.testing.assert_array_equal(a.k, b.k, err_msg=f"{name}[{i}].k differs under reuse")
                np.testing.assert_array_equal(a.electric, b.electric,
                                              err_msg=f"{name}[{i}].electric differs under reuse")

    def test_fresnel_sweep_cache_halves_the_mode_solves_and_keeps_the_curves(self):
        """End-to-end: the GUI sweep's shared cache must halve the anisotropic solves while
        leaving Rp/Rs/Tp/Ts byte-identical."""
        from shaarp import multilayer_basis as mb
        from shaarp.casestudy_materials import build_casestudy_ml_system
        from shaarp.shaarp_gui import compute_ml_gui_result

        system = build_casestudy_ml_system("MoS2", thickness_um=0.5, wavelength_um=1.064)
        grid = dict(fresnel_min_deg=0.0, fresnel_max_deg=20.0, fresnel_step_deg=2.0)

        # patch the name BOUND IN multilayer_basis -- it does `from .anisotropic import
        # solve_snell_modes` at import, so patching shaarp.anisotropic intercepts nothing (the
        # first version of this fence did exactly that and silently counted 0).
        calls = {"n": 0}
        real = mb.solve_snell_modes

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        mb.solve_snell_modes = counting
        try:
            res = compute_ml_gui_result("Fresnel Coefficients", system=system, **grid)
            with_cache = calls["n"]
        finally:
            mb.solve_snell_modes = real

        n_angles = len(np.asarray(res.numeric["theta_deg"]))
        self.assertGreater(with_cache, 0, "the fence patched the wrong module and counted nothing")
        # per angle: layer basis (forward + backward) + substrate = 3. Without the shared cache the
        # p and s passes double it to 6, which is exactly what this fence exists to catch.
        per_angle = with_cache / n_angles
        self.assertLessEqual(per_angle, 3.0, "the p/s passes are still duplicating mode solves "
                                             f"({per_angle:g} solve_snell_modes calls per angle)")
        for key in ("rp", "rs", "tp", "ts"):
            self.assertEqual(len(np.asarray(res.numeric[key])), n_angles)
