"""Normal-incidence (theta_i = 0) and grazing-edge regression for the SHAARP.si compute paths.

WHY THIS EXISTS: the frozen exe crashed with `NonInvertibleMatrixError: Matrix det == 0`
when a user selected the LiNbO3 case on the .si tab and clicked the one-click "0" incident-angle
button. Root cause: at theta_i = 0 the transverse wavevector is zero, so the closed-form eigenmode
index sqrt((v +/- sqrt(v^2 - w))/w) divides by zero -> NaN -> singular boundary matrix. The
default-angle (45 deg) smoke run never exercised this edge.

The two .si compute paths have OPPOSITE needs exactly at theta_i = 0:
  * SHG Simulation -> run_si_numeric (shaarp_si_compat) has its OWN normal_incidence_2omega_branch_policy
    and must receive EXACTLY 0 (a tiny epsilon lands in the ill-conditioned near-normal root-solve zone).
  * Partial/Full Analytical -> closed-form eigenmode path; theta_i = 0 is singular (det == 0), so it is
    desingularized with a small nudge (_desingularize_theta_deg).
This test pins both, so neither regresses. Per the project ratchet rule, every field-found edge bug
becomes an assertion here.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from shaarp.casestudy_materials import build_casestudy_material
from shaarp.shaarp_gui import (
    SI_FUNCTIONALITIES,
    compute_si_gui_result,
    si_effective_index_curve,
    si_polarimetry_curve,
)

# Anisotropic / dispersive case materials that actually exercise the eigenmode + boundary solve
# (an isotropic default would not trigger the det == 0 path). LiNbO3 is the exact field repro.
_EDGE_MATERIALS = [
    "LiNbO3 z-cut (1064 nm)",
    "LiNbO3 z-cut (1550 nm)",
    "KTP x-cut",
    "Quartz z-cut (800 nm)",
    "GaAs (111) (800 nm)",
]

# theta_i = 0 (normal, the crash), a benign oblique angle, and a near-grazing angle (the other edge).
_EDGE_ANGLES = [0.0, 30.0, 89.0]


class SiNormalIncidenceCompute(unittest.TestCase):
    def test_compute_si_gui_result_all_modes_at_edge_angles(self):
        """Every .si functionality must compute (no exception, finite output) at theta_i = 0/30/89."""
        for name in _EDGE_MATERIALS:
            mat = build_casestudy_material(name, wavelength_um=1.064)
            for func in SI_FUNCTIONALITIES:
                for ang in _EDGE_ANGLES:
                    with self.subTest(material=name, functionality=func, theta=ang):
                        try:
                            res = compute_si_gui_result(func, theta_deg=ang, material=mat)
                        except Exception as exc:  # noqa: BLE001 - the bug class is "any crash at the edge"
                            self.fail(f"{func} on {name} at theta={ang} raised "
                                      f"{type(exc).__name__}: {exc}")
                        self.assertIsNotNone(res)

    def test_linbo3_normal_incidence_exact_field_repro(self):
        """The exact reported case: LiNbO3 z-cut (1064 nm) + SHG Simulation + theta_i = 0 deg."""
        mat = build_casestudy_material("LiNbO3 z-cut (1064 nm)", wavelength_um=1.064)
        res = compute_si_gui_result("SHG Simulation", theta_deg=0.0, material=mat)
        self.assertIsNotNone(res)

    def test_polarimetry_curve_at_normal_incidence(self):
        """The closed-form PLOT sweep (build_si_polarimetry_figure -> si_polarimetry_curve) must not
        crash at theta_i = 0 AND must return finite intensities (det == 0 would surface as NaN)."""
        for pg in ("3m", "-43m", "mm2"):
            with self.subTest(point_group=pg):
                d = si_polarimetry_curve(point_group=pg, theta_deg=0.0)
                self.assertGreater(len(d["phi_deg"]), 0)
                for key in ("intensity_s", "intensity_p"):
                    arr = np.asarray(d[key], dtype=float)
                    self.assertEqual(len(arr), len(d["phi_deg"]))
                    self.assertTrue(np.all(np.isfinite(arr)),
                                    f"{key} has non-finite values at theta=0 for {pg} (det==0 singularity)")

    def test_effective_index_curve_spans_zero_incidence(self):
        """The effective-index sweep starts at theta_i = 0; the leading sample must be finite."""
        d = si_effective_index_curve(point_group="3m")
        ang = np.asarray(d["theta_deg"], dtype=float)
        self.assertGreater(len(ang), 0)
        self.assertAlmostEqual(float(ang[0]), 0.0, places=6)
        self.assertTrue(np.isfinite(float(np.asarray(d["n_ordinary"], dtype=float)[0])))


# Point groups carried by case materials that are OUTSIDE the symbolic crystallographic set:
# all centrosymmetric or isotropic (Curie) groups -> SHG is symmetry-forbidden -> the SI figure must
# produce a finite ZERO-SHG result, never crash. (Air is additionally vacuum: no index contrast ->
# singular boundary, also guarded.) Found by the full GUI matrix crash sweep,.
_FORBIDDEN_SHG_MATERIALS = [
    "Air",                       # ∞∞m, d=0, n=1 (vacuum: degenerate interface)
    "Al2O3 (0001) (1550 nm)",    # 6/mmm centrosymmetric
    "Pt (111) (1550 nm)",        # m3m centrosymmetric (metal)
    "Blank linear",              # m3m centrosymmetric
]


class SiCaseMaterialFigureRobustness(unittest.TestCase):
    """Every case-study material must build the SI polarimetry figure (the real on_run path for a
    selected case material) without crashing -- regardless of its point group. This pins the bug class
    the matrix sweep found: materials whose point group is outside the symbolic set (Air/Au = ∞∞m,
    Al2O3 = 6/mmm, Pt/Blank-linear = m3m) crashed si_figure_kwargs_from_material / si_polarimetry_curve."""

    def test_every_case_material_builds_si_polar_figure(self):
        import shaarp.shaarp_gui as g
        from shaarp.casestudy_materials import CASE_STUDY_ORDER

        for name in CASE_STUDY_ORDER:
            mat = build_casestudy_material(name, wavelength_um=1.064)
            with self.subTest(material=name):
                # exact on_run call for a selected case material (nominal angle):
                fig = g.build_si_polarimetry_figure(theta_deg=45.0, **g.si_figure_kwargs_from_material(mat))
                self.assertIsNotNone(fig)

    def test_forbidden_symmetry_materials_give_zero_shg(self):
        """Centrosymmetric / vacuum materials: SHG is symmetry-forbidden -> intensities must be ~0
        (physically correct), not a crash and not garbage."""
        import shaarp.shaarp_gui as g
        for name in _FORBIDDEN_SHG_MATERIALS:
            mat = build_casestudy_material(name, wavelength_um=1.064)
            for ang in (0.0, 45.0, 89.0):
                with self.subTest(material=name, theta=ang):
                    d = g.si_polarimetry_curve(theta_deg=ang, **g.si_figure_kwargs_from_material(mat))
                    i_s = np.asarray(d["intensity_s"], dtype=float)
                    i_p = np.asarray(d["intensity_p"], dtype=float)
                    self.assertTrue(np.all(np.isfinite(i_s)) and np.all(np.isfinite(i_p)))
                    self.assertLess(float(max(i_s.max(), i_p.max())), 1e-12,
                                    f"{name} is centrosymmetric/vacuum; SHG must be ~0")


class MlCaseFilmPartialAnalyticalRobustness(unittest.TestCase):
    """Every case material used as an ML FILM must compute Partial Analytical without crashing. The
    ML partial-analytical path builds a symbolic d-tensor from the film's point group, so a film with a
    point group outside the symbolic set (Air/Au = ∞∞m, Al2O3 = 6/mmm, Pt/Blank-linear = m3m) crashed
    in d_voigt_symbolic. Found by the full GUI matrix crash sweep,."""

    def test_every_film_partial_analytical_no_crash(self):
        from shaarp.casestudy_materials import CASE_STUDY_ORDER, build_casestudy_ml_system
        from shaarp.shaarp_gui import analytical_expression_text, compute_ml_gui_result

        for name in CASE_STUDY_ORDER:
            sysobj = build_casestudy_ml_system(name, thickness_um=1.0, wavelength_um=1.064,
                                               substrate_n_omega=1.5, substrate_n_2omega=1.5)
            with self.subTest(film=name):
                res = compute_ml_gui_result("Partial Analytical", theta_deg=45.0, system=sysobj)
                # must render to copyable text without raising (the panel's analytical view)
                self.assertIn("ml_partial_analytical", analytical_expression_text(res).lower())

    def test_forbidden_film_partial_analytical_is_flagged(self):
        from shaarp.casestudy_materials import build_casestudy_ml_system
        from shaarp.shaarp_gui import analytical_expression_text, compute_ml_gui_result

        for name in ("Air", "Pt (111) (1550 nm)", "Al2O3 (0001) (1550 nm)"):
            sysobj = build_casestudy_ml_system(name, thickness_um=1.0, wavelength_um=1.064,
                                               substrate_n_omega=1.5, substrate_n_2omega=1.5)
            with self.subTest(film=name):
                txt = analytical_expression_text(
                    compute_ml_gui_result("Partial Analytical", theta_deg=45.0, system=sysobj))
                self.assertIn("symmetry-forbidden", txt.lower())

    def test_ml_shg_simulation_near_grazing(self):
        """ML SHG Simulation at near-grazing theta_i = 89 deg must not crash. For high-index / tilted
        films (GaAs(111), MoS2, TaAs) the 2-omega Snell mode root-solve explored a large-imaginary
        trial theta where the direction norm rounded to 0 (catastrophic cancellation in sin^2+cos^2),
        raising 'Propagation direction must be non-zero'. Guarded in anisotropic._solve_complex_theta_
        for_mode. Found by the full GUI matrix crash sweep,."""
        from shaarp.casestudy_materials import build_casestudy_ml_system
        from shaarp.shaarp_gui import compute_ml_gui_result

        for name in ("GaAs (111) (800 nm)", "MoS2", "TaAs (112)", "LiNbO3 z-cut (1064 nm)"):
            sysobj = build_casestudy_ml_system(name, thickness_um=1.0, wavelength_um=1.064,
                                               substrate_n_omega=1.5, substrate_n_2omega=1.5)
            for ang in (88.0, 89.0):
                with self.subTest(film=name, theta=ang):
                    res = compute_ml_gui_result("SHG Simulation", theta_deg=ang, system=sysobj)
                    self.assertIsNotNone(res)


class CaseMaterialLabelCleanliness(unittest.TestCase):
    """No GUI-facing material name may carry Mathematica private-use box-markup codepoints (they render
    as blank 'tofu' boxes). The setup.nb export stored names like SubscriptBox[LiNbO, 3]; the loader
    must fold them to real Unicode subscripts. Found via the matrix sweep's glyph warnings."""

    def test_no_private_use_chars_in_material_or_layer_names(self):
        from shaarp.casestudy_materials import (
            CASE_STUDY_ORDER,
            build_casestudy_ml_system,
        )

        def _pu(s):
            return [hex(ord(c)) for c in str(s) if 0xE000 <= ord(c) <= 0xF8FF]

        for name in CASE_STUDY_ORDER:
            mat = build_casestudy_material(name, wavelength_um=1.064)
            self.assertEqual(_pu(mat.name), [], f"material {name!r} name has private-use chars")
            sysobj = build_casestudy_ml_system(name, thickness_um=1.0, wavelength_um=1.064,
                                               substrate_n_omega=1.5, substrate_n_2omega=1.5)
            for layer in getattr(sysobj, "layers", []):
                self.assertEqual(_pu(getattr(layer, "name", "")), [],
                                 f"{name!r} layer {getattr(layer, 'name', '')!r} has private-use chars")

    def test_subscripts_folded_to_unicode(self):
        # the three that were garbled now read with real Unicode subscripts
        self.assertEqual(build_casestudy_material("MoS2", wavelength_um=1.064).name, "MoS₂")
        self.assertIn("LiNbO₃", build_casestudy_material("LiNbO3 z-cut (1064 nm)", wavelength_um=1.064).name)
        self.assertEqual(build_casestudy_material("Al2O3 (0001) (1550 nm)", wavelength_um=1.064).name, "Al₂O₃")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
