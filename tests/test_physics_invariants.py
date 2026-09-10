"""First-principles PHYSICS invariants on the GUI outputs -- "mechanics != meaning".

The layout harness checks figures aren't clipped; these check the CONTENT is physically right.
This layer exists because the schematic shipped with the reflected omega and 2omega rays
drawn at DIFFERENT angles -- it rendered cleanly (passed every "does it draw" check) but was
physically wrong. Encoding the known invariants as assertions converts that class of "looks fine but
is wrong" bug into a test failure. Every physics bug found in review should add an assertion here.
"""

from __future__ import annotations

import math
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from shaarp.shaarp_gui import (
    build_constrained_d_voigt,
    build_schematic_figure,
    compute_ml_gui_result,
    point_group_free_components,
)


def _arrow_vectors(ax):
    """(head - tail) vectors of every annotation arrow on a schematic axes."""
    vecs = []
    for a in ax.texts:
        if getattr(a, "arrow_patch", None) is not None:
            head = np.asarray(a.xy, dtype=float)
            tail = np.asarray(a.get_position(), dtype=float)
            vecs.append(head - tail)
    return vecs


class SchematicRayInvariants(unittest.TestCase):
    """Reflected omega and 2omega leave into the SAME medium at the SAME (specular) angle = theta_i.
    They must be drawn COLLINEAR. (Only the transmitted rays split, via crystal dispersion.)"""

    def test_reflected_omega_and_2omega_are_collinear_and_specular(self):
        for theta in (20.0, 45.0, 60.0):
            fig = build_schematic_figure([("air", None), ("crystal", None)], theta_deg=theta)
            vecs = _arrow_vectors(fig.axes[0])
            up_right = [v for v in vecs if v[0] > 1e-6 and v[1] > 1e-6]  # the two reflected rays
            self.assertEqual(len(up_right), 2,
                             f"theta={theta}: expected exactly 2 up-going reflected rays (omega+2omega)")
            (vx1, vy1), (vx2, vy2) = up_right
            cross = vx1 * vy2 - vy1 * vx2
            norm = math.hypot(vx1, vy1) * math.hypot(vx2, vy2)
            self.assertLess(abs(cross) / norm, 1e-6,
                            f"theta={theta}: reflected omega and 2omega are NOT collinear")
            # and the reflected ray is specular: angle from the (vertical) normal == theta_i
            ang = math.degrees(math.atan2(vx1, vy1))
            self.assertAlmostEqual(ang, theta, places=3,
                                   msg=f"reflected ray angle {ang:.2f} != incidence {theta}")
            matplotlib.pyplot.close(fig)

    def test_normal_incidence_rays_are_vertical(self):
        """theta_i = 0 must draw VERTICAL rays (true normal incidence), not the ~5 deg tilt the old
        max(5,..) clamp produced. Up-going (reflected omega+2omega) and down-going (incident omega +
        transmitted 2omega) rays must all be vertical (|vx| ~ 0)."""
        fig = build_schematic_figure([("air", None), ("crystal", None)], theta_deg=0.0)
        vecs = _arrow_vectors(fig.axes[0])
        up = [v for v in vecs if v[1] > 1e-6]      # reflected omega + reflected 2omega
        down = [v for v in vecs if v[1] < -1e-6]   # incident omega + transmitted 2omega
        self.assertEqual(len(up), 2, "theta=0: expected 2 up-going reflected rays")
        self.assertGreaterEqual(len(down), 1, "theta=0: expected a down-going incident ray")
        for vx, vy in up + down:
            self.assertLess(abs(vx), 1e-6, f"theta=0: ray is not vertical (vx={vx:.4f})")
        matplotlib.pyplot.close(fig)

    def test_high_angle_drawn_angle_tracks_input(self):
        """At theta_i = 80 deg the drawn reflected ray must be at 80 deg -- i.e. the old min(75,..)
        upper clamp (which capped the picture at 75 deg) is gone and the schematic tracks the input."""
        theta = 80.0
        fig = build_schematic_figure([("air", None), ("crystal", None)], theta_deg=theta)
        up_right = [v for v in _arrow_vectors(fig.axes[0]) if v[0] > 1e-6 and v[1] > 1e-6]
        self.assertEqual(len(up_right), 2, "theta=80: expected 2 up-going reflected rays")
        vx1, vy1 = up_right[0]
        ang = math.degrees(math.atan2(vx1, vy1))
        self.assertAlmostEqual(ang, theta, places=3,
                               msg=f"drawn angle {ang:.2f} != {theta} (upper clamp not removed)")
        matplotlib.pyplot.close(fig)


class FresnelEnergyInvariants(unittest.TestCase):
    """Linear Fresnel on the default (lossless) stack must conserve energy and stay physical."""

    def test_R_plus_T_unity_bounds_and_normal_incidence_degeneracy(self):
        r = compute_ml_gui_result("Fresnel Coefficients", theta_min_deg=0.0, theta_max_deg=89.0,
                                  theta_step_deg=1.0)
        rp, rs, tp, ts = (np.asarray(r.numeric[k], float) for k in ("rp", "rs", "tp", "ts"))
        th = np.asarray(r.numeric["theta_deg"], float)
        # energy conservation (lossless): R + T == 1 per polarization
        self.assertLess(float(np.max(np.abs(rp + tp - 1.0))), 1e-6, "R_p + T_p != 1 (energy)")
        self.assertLess(float(np.max(np.abs(rs + ts - 1.0))), 1e-6, "R_s + T_s != 1 (energy)")
        # physical bounds
        for arr, name in ((rp, "R_p"), (rs, "R_s"), (tp, "T_p"), (ts, "T_s")):
            self.assertGreaterEqual(float(arr.min()), -1e-9, f"{name} < 0")
            self.assertLessEqual(float(arr.max()), 1.0 + 1e-9, f"{name} > 1")
        # at normal incidence p and s are degenerate
        i0 = int(np.argmin(np.abs(th)))
        self.assertAlmostEqual(float(rp[i0]), float(rs[i0]), places=6,
                               msg="R_p != R_s at normal incidence")

    def test_R_plus_T_unity_with_media_that_are_not_index_matched(self):
        """The same invariant on stacks where it can actually FAIL.

        The test above runs on the default stack, whose substrate is ``blank_linear`` with n = 1.
        There the obliquity factor Re(n_exit cos th_exit)/Re(n_inc cos th_inc) is identically 1, so
        R+T=1 held even while the transmitted curves were bare |t|^2 rather than a power
        transmittance -- it passed on the one stack that could not expose the bug, and stayed
        vacuous for the whole life of the multilayer Fresnel path.

        These stacks break the index match on the exit side, on the incident side, and on both, so
        the factor is genuinely exercised. Cross-checked against tmm and inkstone in
        tests/test_isotropic_stack_reference_comparison.py.
        """
        from shaarp.api import run_fresnel_sweep
        from shaarp.config import Layer, MultilayerSystem
        from shaarp.layer_stack import _material_from_iso

        def stack(n_incident, n_substrate):
            return MultilayerSystem(
                wavelength_um=0.633,
                layers=[
                    Layer(name="ambient", material=_material_from_iso(n_incident, n_incident, "ambient"),
                          thickness_um=None, shg_active=False),
                    Layer(name="film", material=_material_from_iso(2.35, 2.35, "film"),
                          thickness_um=0.5, shg_active=False),
                    Layer(name="substrate", material=_material_from_iso(n_substrate, n_substrate, "substrate"),
                          thickness_um=None, shg_active=False),
                ],
            )

        # (incident, substrate, max angle): non-air exit; non-air entrance; both, and unequal.
        #
        # The water->air row stops at 45 deg on purpose. Going from a dense to a rare medium there
        # is a critical angle at arcsin(1.00/1.33) = 48.75 deg, and beyond it the multilayer path
        # raises "No real Snell root found" -- total internal reflection is not supported by
        # solve_snell_modes. That is a pre-existing limitation of the solver, unrelated to the
        # transmittance weighting this test guards, so the sweep stays inside the propagating
        # regime rather than asserting anything about TIR.
        for n_inc, n_sub, theta_max in ((1.0, 1.52, 85.0), (1.33, 1.0, 45.0), (1.34, 1.52, 85.0)):
            with self.subTest(n_incident=n_inc, n_substrate=n_sub):
                theta = np.arange(0.0, theta_max + 1e-9, 5.0)
                result = run_fresnel_sweep(
                    stack(n_inc, n_sub), theta,
                    options={"workflow": "gui_multilayer",
                             "transmitted_wave_policy": "physical_sum",
                             "mrassumption": 0},
                )
                rp, rs, tp, ts = (np.asarray(result.numeric[k], float) for k in ("rp", "rs", "tp", "ts"))
                self.assertLess(float(np.max(np.abs(rp + tp - 1.0))), 1e-9,
                                "R_p + T_p != 1 for n_inc=%s n_sub=%s" % (n_inc, n_sub))
                self.assertLess(float(np.max(np.abs(rs + ts - 1.0))), 1e-9,
                                "R_s + T_s != 1 for n_inc=%s n_sub=%s" % (n_inc, n_sub))
                for arr, name in ((rp, "R_p"), (rs, "R_s"), (tp, "T_p"), (ts, "T_s")):
                    self.assertGreaterEqual(float(arr.min()), -1e-9, "%s < 0" % name)
                    self.assertLessEqual(float(arr.max()), 1.0 + 1e-9, "%s > 1" % name)

    def test_legacy_amplitude_transmittance_does_not_conserve_energy(self):
        """Guards the guard: proves the case above WOULD fail without the obliquity factor.

        An invariant test that cannot fail is worse than none, which is exactly how this bug
        survived. Requesting transmittance='amplitude' reproduces the pre-fix behaviour, and R+T
        must then be visibly wrong. If this ever starts conserving energy, the test above has gone
        vacuous again and its stacks need re-choosing.
        """
        from shaarp.api import run_fresnel_sweep
        from shaarp.config import Layer, MultilayerSystem
        from shaarp.layer_stack import _material_from_iso

        system = MultilayerSystem(
            wavelength_um=0.633,
            layers=[
                Layer(name="ambient", material=_material_from_iso(1.0, 1.0, "ambient"),
                      thickness_um=None, shg_active=False),
                Layer(name="film", material=_material_from_iso(2.35, 2.35, "film"),
                      thickness_um=0.5, shg_active=False),
                Layer(name="substrate", material=_material_from_iso(1.52, 1.52, "substrate"),
                      thickness_um=None, shg_active=False),
            ],
        )
        legacy = run_fresnel_sweep(
            system, np.arange(0.0, 86.0, 5.0),
            options={"workflow": "gui_multilayer", "transmitted_wave_policy": "physical_sum",
                     "mrassumption": 0, "transmittance": "amplitude"},
        )
        rs = np.asarray(legacy.numeric["rs"], float)
        ts = np.asarray(legacy.numeric["ts"], float)
        self.assertGreater(float(np.max(np.abs(rs + ts - 1.0))), 0.1,
                           "the legacy |t|^2 convention now conserves energy -- this stack no "
                           "longer discriminates, so the invariant test above proves nothing")


class DTensorSymmetryInvariants(unittest.TestCase):
    """The symmetry-constrained SHG d tensor must obey the point group: every independent component
    is nonzero, and symmetry-derived relations hold (e.g. -43m: d14 = d25 = d36, all else zero)."""

    def test_each_free_component_is_nonzero(self):
        for pg in ("-43m", "3m", "mm2"):
            free = point_group_free_components(pg)
            d = build_constrained_d_voigt(pg, {(r, c): 1.0 for (r, c, _n) in free})
            for (r, c, _n) in free:
                self.assertNotEqual(complex(d[r, c]), 0j, f"{pg}: free d[{r},{c}] is zero")

    def test_cubic_43bar_m_equalities_and_sparsity(self):
        d = build_constrained_d_voigt("-43m", {(0, 3): 1.0})
        nonzero = {(r, c) for r in range(3) for c in range(6) if abs(d[r, c]) > 0}
        self.assertEqual(nonzero, {(0, 3), (1, 4), (2, 5)}, "-43m allows only d14, d25, d36")
        self.assertEqual(complex(d[0, 3]), complex(d[1, 4]))  # d14 == d25
        self.assertEqual(complex(d[1, 4]), complex(d[2, 5]))  # d25 == d36

    def test_orthorhombic_mm2_exact_pattern(self):
        free = point_group_free_components("mm2")
        d = build_constrained_d_voigt("mm2", {(r, c): 1.0 for (r, c, _n) in free})
        nonzero = {(r, c) for r in range(3) for c in range(6) if abs(d[r, c]) > 0}
        self.assertEqual(nonzero, {(r, c) for (r, c, _n) in free},
                         "mm2 nonzeros must equal exactly its independent components")


if __name__ == "__main__":
    unittest.main()
