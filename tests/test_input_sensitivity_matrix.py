"""INPUT-SENSITIVITY matrix: every output must RESPOND to every input that should matter.

WHY (tightening T1 after kept finding what the checks missed): the / F24ml /
F25 family shared one mechanical signature -- an output silently failed to depend on an input it
must depend on (theta ignored, the layer stack ignored, "Full" identical to "Partial"). Detecting
that class needs NO knowledge of the right answer, only of what SHOULD matter: perturb each input,
assert the output changes -- or, where invariance is BY DESIGN (the fully-symbolic closed form is
theta-independent), assert the invariance AND that the output declares it.

Cells run at the compute level (the same calls the GUI's Update makes), headless and reasonably
fast. Every cell failure message names the ignored input -- the exact diagnosis a user would
otherwise have to reverse-engineer from a wrong plot.
"""

from __future__ import annotations

import hashlib
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from shaarp.casestudy_materials import (
    build_casestudy_material,
    build_casestudy_ml_system,
)
from shaarp.shaarp_gui import (
    analytical_expression_text,
    compute_ml_gui_result,
    compute_si_gui_result,
    si_figure_kwargs_from_material,
    si_polarimetry_curve,
)


def _fp_numeric(result) -> str:
    """Stable fingerprint of a SHAARPResult's numeric payload."""
    h = hashlib.sha256()
    for key in sorted((result.numeric or {}).keys()):
        arr = np.asarray(result.numeric[key])
        h.update(key.encode())
        h.update(np.round(np.asarray(arr, dtype=complex), 12).tobytes())
    return h.hexdigest()


def _fp_stages(result) -> str:
    h = hashlib.sha256()
    for key in ("reflected_p_2omega", "reflected_s_2omega", "analyzed_intensity"):
        h.update(str((result.stages or {}).get(key, "")).encode())
    return h.hexdigest()


def _fp_curve(curve: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(k for k, v in curve.items() if isinstance(v, (list, np.ndarray))):
        h.update(key.encode())
        h.update(np.round(np.asarray(curve[key], dtype=float), 12).tobytes())
    return h.hexdigest()


class SiNumericSensitivity(unittest.TestCase):
    def test_theta_matters(self):
        mat = build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)
        a = _fp_numeric(compute_si_gui_result("SHG Simulation", theta_deg=30.0, material=mat))
        b = _fp_numeric(compute_si_gui_result("SHG Simulation", theta_deg=45.0, material=mat))
        self.assertNotEqual(a, b, "SI SHG Simulation ignores the incident angle")

    def test_material_matters(self):
        a = _fp_numeric(compute_si_gui_result(
            "SHG Simulation", theta_deg=45.0,
            material=build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)))
        b = _fp_numeric(compute_si_gui_result(
            "SHG Simulation", theta_deg=45.0,
            material=build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=0.8)))
        self.assertNotEqual(a, b, "SI SHG Simulation ignores the selected material")

    def test_wavelength_matters_for_dispersive_material(self):
        # Quartz z-cut carries a REAL wavelength-varying table in the original export. (The first
        # version of this cell used "LiNbO3 (dispersive)" -- whose name promised dispersion the
        # ORIGINAL setup.nb never defined (constant n1w=2.294778...); T1 exposed that and the
        # material is now honestly named "LiNbO3 z-cut (1064 nm)".)
        a = _fp_numeric(compute_si_gui_result(
            "SHG Simulation", theta_deg=45.0,
            material=build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=0.8)))
        b = _fp_numeric(compute_si_gui_result(
            "SHG Simulation", theta_deg=45.0,
            material=build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=1.064)))
        self.assertNotEqual(a, b, "SI SHG Simulation ignores the wavelength (dispersion)")


class SiPolarimetrySettingsSensitivity(unittest.TestCase):
    def _kw(self):
        mat = build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=0.8)
        kw = si_figure_kwargs_from_material(mat)
        kw.pop("point_group")
        return kw

    def test_ellipticity_matters(self):
        kw = self._kw()
        a = _fp_curve(si_polarimetry_curve("32", theta_deg=45.0, ellipticity_deg=0.0, **kw))
        b = _fp_curve(si_polarimetry_curve("32", theta_deg=45.0, ellipticity_deg=30.0, **kw))
        self.assertNotEqual(a, b, "polarimetry curve ignores the incident ellipticity")

    def test_fixed_analyzer_matters(self):
        kw = self._kw()
        a = _fp_curve(si_polarimetry_curve("32", theta_deg=45.0, analyzer_deg=None, **kw))
        b = _fp_curve(si_polarimetry_curve("32", theta_deg=45.0, analyzer_deg=45.0, **kw))
        self.assertNotEqual(a, b, "polarimetry curve ignores the fixed-analyzer angle")

    def test_fix_polarizer_mode_matters(self):
        kw = self._kw()
        a = _fp_curve(si_polarimetry_curve("32", theta_deg=45.0, fixed_phi_deg=None, **kw))
        b = _fp_curve(si_polarimetry_curve("32", theta_deg=45.0, fixed_phi_deg=30.0, **kw))
        self.assertNotEqual(a, b, "polarimetry curve ignores the Fix-Polarizer phi")

    def test_corotating_offset_matters_and_obeys_identity(self):
        """FB1: the analyzer-polarizer offset must change the co-rotating channels, and the
        90-deg-offset parallel channel must equal the 0-deg perpendicular channel exactly."""
        kw = self._kw()
        a = si_polarimetry_curve("32", theta_deg=45.0, corotating_offset_deg=0.0, **kw)
        b = si_polarimetry_curve("32", theta_deg=45.0, corotating_offset_deg=30.0, **kw)
        self.assertNotEqual(_fp_curve(a), _fp_curve(b), "co-rotating channels ignore the offset")
        c = si_polarimetry_curve("32", theta_deg=45.0, corotating_offset_deg=90.0, **kw)
        d = float(np.max(np.abs(np.asarray(a["intensity_perpendicular"])
                                - np.asarray(c["intensity_parallel"]))))
        self.assertLess(d, 1e-12, "offset identity perp(0) == par(90) violated")


class SiAnalyticalSensitivity(unittest.TestCase):
    def test_partial_depends_on_theta(self):
        mat = build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=0.8)
        a = _fp_stages(compute_si_gui_result("Partial Analytical", theta_deg=30.0, material=mat))
        b = _fp_stages(compute_si_gui_result("Partial Analytical", theta_deg=45.0, material=mat))
        self.assertNotEqual(a, b, "Partial Analytical ignores the incident angle (F24 class)")

    def test_partial_depends_on_material_eps_same_point_group(self):
        # LiNbO3 z-cut and LiOsO3 are BOTH 3m -- only their eps values differ
        a = _fp_stages(compute_si_gui_result(
            "Partial Analytical", theta_deg=45.0,
            material=build_casestudy_material("LiNbO3 z-cut (1550 nm)", wavelength_um=1.55)))
        b = _fp_stages(compute_si_gui_result(
            "Partial Analytical", theta_deg=45.0,
            material=build_casestudy_material("LiOsO3", wavelength_um=1.55)))
        self.assertNotEqual(a, b, "Partial Analytical ignores the material's epsilon (F24)")

    def test_partial_depends_on_orientation(self):
        # GaAs(111) vs Blank nonlinear: BOTH -43m; only the orientation differs materially
        a = _fp_stages(compute_si_gui_result(
            "Partial Analytical", theta_deg=45.0,
            material=build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)))
        b = _fp_stages(compute_si_gui_result(
            "Partial Analytical", theta_deg=45.0,
            material=build_casestudy_material("Blank nonlinear", wavelength_um=0.8)))
        self.assertNotEqual(a, b, "Partial Analytical ignores the crystal orientation")

    def test_full_is_theta_invariant_BY_DESIGN_and_says_so(self):
        mat = build_casestudy_material("LiNbO3 z-cut (1550 nm)", wavelength_um=1.55)
        ra = compute_si_gui_result("Full Analytical", theta_deg=30.0, material=mat)
        rb = compute_si_gui_result("Full Analytical", theta_deg=45.0, material=mat)
        self.assertEqual(_fp_stages(ra), _fp_stages(rb),
                         "Full-symbolic should be theta-independent (theta is a symbol)")
        text = analytical_expression_text(ra)
        self.assertIn("theta_i (symbolic)", text,
                      "theta-invariance must be DECLARED in the provenance header")

    def test_full_differs_from_partial(self):
        mat = build_casestudy_material("LiNbO3 z-cut (1550 nm)", wavelength_um=1.55)
        full = _fp_stages(compute_si_gui_result("Full Analytical", theta_deg=45.0, material=mat))
        part = _fp_stages(compute_si_gui_result("Partial Analytical", theta_deg=45.0, material=mat))
        self.assertNotEqual(full, part, "Full Analytical is indistinguishable from Partial (F25)")


class SiPolarimetryPaperReplication(unittest.TestCase):
    """Fences for two defects found by replicating the 2022 paper's
    Fig. 4 (GaAs (111), 800 nm) through the GUI curve path. rotating a mathematically
    ISOTROPIC eps leaves ~1e-15 float noise between the principal values; the symbolic solver's
    exact-equality degeneracy check missed it and the biaxial D-formula divided by that noise
    (NaN curves at most incident angles). the curve path cast eps and d to REAL, silently
    applying the paper's eps_R approximation (the paper itself shows it misestimates d by ~20%)."""

    def _kw(self):
        mat = build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)
        kw = si_figure_kwargs_from_material(mat)
        return mat, kw, kw.pop("point_group")

    def test_rotated_isotropic_curve_finite_at_all_angles(self):
        _mat, kw, pg = self._kw()
        for th in (0.0, 5.0, 15.0, 30.0, 44.0, 45.0, 60.0, 75.0, 89.0):
            c = si_polarimetry_curve(pg, theta_deg=th, **kw)
            for key in ("intensity_p", "intensity_s"):
                arr = np.asarray(c[key], dtype=float)
                self.assertTrue(bool(np.all(np.isfinite(arr))),
                                f"GaAs(111) {key} non-finite at theta={th} (regression)")

    def test_curve_matches_full_numeric_reference(self):
        import math

        from shaarp.shg import solve_single_interface_shg
        from shaarp.tensors import rotate_d_voigt_crystal_to_lab
        from shaarp.shaarp_gui import _epsilon_lab_of

        mat, kw, pg = self._kw()
        eps_w_lab = _epsilon_lab_of(mat, omega=True)
        eps_2_lab = _epsilon_lab_of(mat, omega=False)
        rot = np.asarray(mat.orientation.rotation_matrix(), dtype=float)
        d_lab = np.asarray(rotate_d_voigt_crystal_to_lab(np.asarray(mat.d_voigt_pm_v), rot),
                           dtype=complex)
        c = si_polarimetry_curve(pg, theta_deg=30.0, n_phi=361, **kw)
        phi_deg = np.asarray(c["phi_deg"])
        for phi0 in (0.0, 40.0, 137.0):
            k = int(np.argmin(np.abs(phi_deg - phi0)))
            jones = (math.sin(math.radians(phi0)), math.cos(math.radians(phi0)))
            r = solve_single_interface_shg(
                eps_w_lab, eps_2_lab, d_lab, incident_index_omega=1.0,
                incident_index_2omega=1.0, incident_theta_rad=math.radians(30.0),
                incident_jones=jones, omega=1.0, mu=1.0, eps0=1.0)
            n_ip = float(np.sum(np.abs(np.asarray(r.reflected_p_2omega.electric)) ** 2))
            n_is = float(np.sum(np.abs(np.asarray(r.reflected_s_2omega.electric)) ** 2))
            self.assertLess(abs(float(c["intensity_p"][k]) - n_ip) / n_ip, 1e-9,
                            f"curve I_p deviates from the validated numeric solver at phi={phi0}")
            self.assertLess(abs(float(c["intensity_s"][k]) - n_is) / max(n_is, 1e-25), 1e-9,
                            f"curve I_s deviates from the validated numeric solver at phi={phi0}")

    def test_complex_eps_matters_the_paper_20_percent(self):
        _mat, kw, pg = self._kw()
        kw_real = dict(kw)
        kw_real["eps_omega_principal"] = tuple(float(np.real(x)) for x in kw["eps_omega_principal"])
        kw_real["eps_2omega_principal"] = tuple(float(np.real(x)) for x in kw["eps_2omega_principal"])
        c_full = si_polarimetry_curve(pg, theta_deg=30.0, **kw)
        c_real = si_polarimetry_curve(pg, theta_deg=30.0, **kw_real)
        ratio = float(np.sqrt(np.max(c_full["intensity_p"]) / np.max(c_real["intensity_p"])))
        # the 2022 paper Fig 4(e): eps_R-fitted d36 = 216/267 = 0.81 of the full value
        self.assertTrue(0.75 < ratio < 0.90,
                        f"complex-eps effect ratio {ratio:.3f} outside the paper's ~20% window "
                        "(regression) dropped again?)")


class MlSensitivity(unittest.TestCase):
    def _sys(self, name="Quartz z-cut (800 nm)", t=1.0, wl=0.8, **kw):
        return build_casestudy_ml_system(name, thickness_um=t, wavelength_um=wl, **kw)

    def test_shg_sim_depends_on_theta_and_assumption_and_thickness(self):
        s = self._sys()
        base = _fp_numeric(compute_ml_gui_result("SHG Simulation", theta_deg=20.0, system=s))
        self.assertNotEqual(base, _fp_numeric(compute_ml_gui_result(
            "SHG Simulation", theta_deg=45.0, system=s)), "ML SHG ignores theta")
        self.assertNotEqual(base, _fp_numeric(compute_ml_gui_result(
            "SHG Simulation", theta_deg=20.0, system=s,
            assumption="Jerphagnon & Kurtz Assumption (No MR)")), "ML SHG ignores the assumption")
        self.assertNotEqual(base, _fp_numeric(compute_ml_gui_result(
            "SHG Simulation", theta_deg=20.0, system=self._sys(t=3.0))),
            "ML SHG ignores the film thickness")

    def test_maker_depends_on_grid_assumption_wavelength(self):
        s = self._sys()
        grid = dict(theta_min_deg=5.0, theta_max_deg=15.0, theta_step_deg=1.0)
        base = compute_ml_gui_result("Maker Fringes", system=s, **grid)
        n_base = len(np.asarray(base.numeric["theta_deg"]))
        wider = compute_ml_gui_result("Maker Fringes", system=s, theta_min_deg=5.0,
                                      theta_max_deg=25.0, theta_step_deg=1.0)
        self.assertGreater(len(np.asarray(wider.numeric["theta_deg"])), n_base,
                           "Maker ignores the scan range")
        finer = compute_ml_gui_result("Maker Fringes", system=s, theta_min_deg=5.0,
                                      theta_max_deg=15.0, theta_step_deg=0.5)
        self.assertGreater(len(np.asarray(finer.numeric["theta_deg"])), n_base,
                           "Maker ignores the scan step")
        hh = compute_ml_gui_result(
            "Maker Fringes", system=s,
            assumption="Herman & Hayden Assumption (MR only for 2ω Homo Waves)", **grid)
        self.assertNotEqual(_fp_numeric(base), _fp_numeric(hh), "Maker ignores the assumption")
        other_wl = compute_ml_gui_result("Maker Fringes",
                                         system=self._sys(name="Quartz x-cut (1064 nm)", wl=1.064),
                                         **grid)
        self.assertNotEqual(_fp_numeric(base), _fp_numeric(other_wl),
                            "Maker ignores the material/wavelength")

    def test_fresnel_depends_on_thickness_and_substrate(self):
        grid = dict(theta_min_deg=5.0, theta_max_deg=60.0, theta_step_deg=2.0)
        a = _fp_numeric(compute_ml_gui_result("Fresnel Coefficients", system=self._sys(t=1.0), **grid))
        b = _fp_numeric(compute_ml_gui_result("Fresnel Coefficients", system=self._sys(t=5.0), **grid))
        self.assertNotEqual(a, b, "Fresnel ignores the film thickness")
        c = _fp_numeric(compute_ml_gui_result(
            "Fresnel Coefficients", system=self._sys(substrate_n_omega=1.7, substrate_n_2omega=1.72),
            **grid))
        self.assertNotEqual(a, c, "Fresnel ignores the substrate index")

    def test_partial_analytical_depends_on_substrate(self):
        a = str(compute_ml_gui_result("Partial Analytical", theta_deg=20.0,
                                      system=self._sys()).stages["reflected_p_2omega"])
        b = str(compute_ml_gui_result("Partial Analytical", theta_deg=20.0,
                                      system=self._sys(substrate_n_omega=1.7, substrate_n_2omega=1.72)
                                      ).stages["reflected_p_2omega"])
        self.assertNotEqual(a, b, "ML Partial Analytical ignores the substrate (F24ml class)")

    def test_metal_film_fresnel_full_range_is_finite(self):
        """FB8b follow-up: the fixed 0-90 Fresnel range must survive metallic films -- the Pt film
        hits an isolated singular boundary solve at exactly 85.0 deg (removable; masked and
        interpolated like the Maker singularities). Found by the release sweep."""
        import numpy as np
        s = build_casestudy_ml_system("Pt (111) (1550 nm)", thickness_um=1.0, wavelength_um=1.55)
        r = compute_ml_gui_result("Fresnel Coefficients", theta_step_deg=0.5, system=s)
        for key in ("rp", "rs", "tp", "ts"):
            arr = np.asarray(r.numeric[key], dtype=float)
            self.assertTrue(bool(np.all(np.isfinite(arr))), f"{key} has non-finite points")

    def test_analytical_dij_mixing_and_h_substitution(self):
        """FB2: a KNOWN d component must vanish as a symbol (substituted) while unknowns remain;
        the analytical-h toggle must remove the bare thickness symbol when substituted."""
        r = compute_ml_gui_result("Partial Analytical", theta_deg=20.0, system=self._sys(),
                                  analytical_d_known={"d11": 2.3}, analytical_h_value=1.0)
        ep = str(r.stages["reflected_p_2omega"])
        self.assertIn("d14", ep, "unknown component must stay symbolic")
        self.assertNotIn("d11", ep, "known component must be substituted (FB2)")
        base = str(compute_ml_gui_result("Partial Analytical", theta_deg=20.0,
                                         system=self._sys()).stages["reflected_p_2omega"])
        self.assertIn("d11", base, "without mixing, d11 must be symbolic")
        self.assertNotEqual(ep, base)


class SiGeneralOrientationCurve(unittest.TestCase):
    """Fences from a GUI report: a lab eps that is NOT
    principal-aligned (the TaAs (112) tilt of a uniaxial crystal) previously fell back SILENTLY
    to the crystal-frame z-cut approximation in si_figure_kwargs_from_material -- the case-study
    plot never showed the tilt, so a user switching the orientation to identity saw no change.
    Now the general case routes through the validated numeric arbitrary-Jones solver."""

    def test_numeric_branch_equals_symbolic_where_closed_form_exists(self):
        """Forcing the numeric branch on a principal-aligned uniaxial case must reproduce the
        symbolic closed-form curve (incl. ellipticity and analyzer modes) -- the convention
        contract (channel order, Jones/ellipticity phase, analyzer formula)."""
        pg = "3m"
        base = dict(theta_deg=40.0, n_omega=2.2, n_2omega=2.3, n_omega_e=2.25, n_2omega_e=2.35,
                    d_free={(0, 4): 1.2, (1, 1): -0.3, (2, 0): 0.9, (2, 2): 2.5}, n_phi=13)
        eps_w = np.diag([2.2 ** 2, 2.2 ** 2, 2.25 ** 2]).astype(complex)
        eps_2 = np.diag([2.3 ** 2, 2.3 ** 2, 2.35 ** 2]).astype(complex)
        for extra in ({}, {"ellipticity_deg": 30.0}, {"analyzer_deg": 25.0}):
            sym = si_polarimetry_curve(pg, **base, **extra)
            num = si_polarimetry_curve(pg, **base, **extra,
                                       eps_omega_lab_full=eps_w, eps_2omega_lab_full=eps_2)
            for key in sym:
                if not (isinstance(sym[key], np.ndarray) and key.startswith("intensity")):
                    continue
                if key.endswith("_trans"):
                    continue  # transmitted channel is symbolic-only best-effort (by design)
                a = np.asarray(sym[key], dtype=float)
                b = np.asarray(num[key], dtype=float)
                scale = float(np.max(np.abs(a))) + 1e-300
                self.assertLess(float(np.max(np.abs(a - b))) / scale, 1e-8,
                                f"numeric branch deviates from closed form on {key} ({extra})")
            peak = float(np.max(np.asarray(sym["intensity_p"], dtype=float)))
            self.assertGreater(peak, 1e-8, "vacuous zero curves")

    def test_taas_112_routes_general_and_differs_from_zcut(self):
        """The TaAs (112) case material must produce the general-orientation kwargs, and its
        exact curve must differ decisively from the old silent z-cut approximation (the exact
        (112) I_p/I_s peak ratio is ~74.6 vs the z-cut's ~4.6)."""
        mat = build_casestudy_material("TaAs (112)", wavelength_um=0.8)
        kw = si_figure_kwargs_from_material(mat)
        pg = kw.pop("point_group")
        self.assertIn("eps_omega_lab_full", kw,
                      "TaAs (112) no longer routes through the general-orientation branch (F40)")
        cur = si_polarimetry_curve(pg, theta_deg=45.0, n_phi=37, **kw)
        kw_zcut = {k: v for k, v in kw.items()
                   if k not in ("eps_omega_lab_full", "eps_2omega_lab_full", "d_voigt_lab_full")}
        cur_z = si_polarimetry_curve(pg, theta_deg=45.0, n_phi=37, **kw_zcut)
        ip, is_ = (float(np.max(np.asarray(cur[k], dtype=float)))
                   for k in ("intensity_p", "intensity_s"))
        self.assertGreater(ip, 0.0)
        self.assertGreater(is_, 0.0)
        ratio = ip / is_
        self.assertTrue(65.0 < ratio < 85.0,
                        f"exact TaAs (112) Ip/Is = {ratio:.1f} drifted from ~74.6 (F40 baseline)")
        rel = (float(np.max(np.abs(np.asarray(cur["intensity_p"]) - np.asarray(cur_z["intensity_p"]))))
               / (ip + 1e-300))
        self.assertGreater(rel, 0.5,
                           "exact (112) curve no longer differs from the z-cut approximation -- "
                           "the general branch silently fell back again (regression)")

    def test_effective_index_tile_uses_full_lab_eps(self):
        """The effective-index tile must consume the full lab eps under tilt: the tilted
        second-mode index curve differs from the scalar z-cut route and stays finite."""
        from shaarp.shaarp_gui import si_effective_index_curve

        mat = build_casestudy_material("TaAs (112)", wavelength_um=0.8)
        kw = si_figure_kwargs_from_material(mat)
        full = si_effective_index_curve("4mm", n_omega=kw["n_omega"], n_omega_e=kw["n_omega_e"],
                                        eps_omega_lab_full=kw["eps_omega_lab_full"], n_theta=19)
        scal = si_effective_index_curve("4mm", n_omega=kw["n_omega"], n_omega_e=kw["n_omega_e"],
                                        n_theta=19)
        for out in (full, scal):
            for key in ("n_ordinary", "n_extraordinary"):
                self.assertTrue(bool(np.all(np.isfinite(np.asarray(out[key], dtype=float)))))
        self.assertGreater(float(np.max(np.abs(np.asarray(full["n_extraordinary"])
                                               - np.asarray(scal["n_extraordinary"])))), 1e-3,
                           "tilted-mode index curve identical to the scalar z-cut route")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
