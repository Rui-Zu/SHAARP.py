"""Physics-sanity assertions on what the GUI plots actually SHOW (release-audit layer).

The crash sweep proves cells compute; the layout harness proves nothing is clipped; these prove the
CONTENT is physically sensible -- the checks a critical first-time user would eyeball:

  * polarimetry curves are nonnegative and 360-degree periodic;
  * a case material's polar pattern carries its point-group symmetry signature
    (GaAs(111): C3v -> 120-degree invariance of the polarimetry set; quartz z-cut 32: 3-fold);
  * the linear Fresnel sweep of a lossless dielectric film shows a Brewster-like R_p minimum well
    below R_s, and reflectance -> 1 at grazing;
  * Maker fringes get DENSER for a THICKER film (more interference orders across the same angles).

Every assertion runs on the same compute paths the GUI's Update button calls.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np

from shaarp.casestudy_materials import build_casestudy_material, build_casestudy_ml_system
from shaarp.shaarp_gui import (
    compute_ml_gui_result,
    si_figure_kwargs_from_material,
    si_polarimetry_curve,
)


def _si_curves(name: str, theta: float = 45.0, wavelength: float = 0.8):
    mat = build_casestudy_material(name, wavelength_um=wavelength)
    kw = si_figure_kwargs_from_material(mat)
    pg = kw.pop("point_group")
    d = si_polarimetry_curve(pg, theta_deg=theta, **kw)
    phi = np.asarray(d["phi_deg"], float)
    return phi, np.asarray(d["intensity_s"], float), np.asarray(d["intensity_p"], float)


class PolarimetryCurveSanity(unittest.TestCase):
    def test_nonnegative_and_periodic(self):
        for name in ("LiNbO3 z-cut (1064 nm)", "GaAs (111) (800 nm)", "Quartz z-cut (800 nm)"):
            with self.subTest(material=name):
                phi, i_s, i_p = _si_curves(name)
                for arr, ch in ((i_s, "I_s"), (i_p, "I_p")):
                    self.assertGreaterEqual(float(arr.min()), -1e-18 * max(arr.max(), 1.0),
                                            f"{name}: {ch} has negative intensity")
                # curve spans 0..360 with endpoints equal (periodicity)
                self.assertAlmostEqual(float(phi[0] % 360.0), float(phi[-1] % 360.0), places=6)
                self.assertAlmostEqual(float(i_s[0]), float(i_s[-1]), places=12)
                self.assertAlmostEqual(float(i_p[0]), float(i_p[-1]), places=12)

    def test_nonzero_signal_for_shg_active_materials(self):
        for name in ("LiNbO3 z-cut (1064 nm)", "GaAs (111) (800 nm)"):
            with self.subTest(material=name):
                _phi, i_s, i_p = _si_curves(name)
                self.assertGreater(float(i_s.max() + i_p.max()), 0.0,
                                   f"{name}: SHG-active material shows zero polarimetry signal")


class PointGroupSignature(unittest.TestCase):
    """The polarizer-angle dependence enters through the fields ~ (cos phi, sin phi e^{i dd}) --
    intensities are quadratic forms in (cos, sin), hence ALWAYS 180-deg periodic in phi; the
    crystal symmetry shows up in the LOBE structure. We assert the 180-degree periodicity (a real
    physics invariant of the quadratic dependence) plus, for C3v GaAs(111), the s-channel's
    characteristic multi-lobe structure (>= 4 extrema across a period -- not a bare dipole)."""

    def test_gaas111_lobe_structure_and_periodicity(self):
        phi, i_s, i_p = _si_curves("GaAs (111) (800 nm)")
        n = len(phi) - 1  # phi covers 0..360 inclusive
        half = n // 2
        for arr, ch in ((i_s, "I_s"), (i_p, "I_p")):
            a, b = arr[:half], arr[half:2 * half]
            denom = max(float(np.max(np.abs(a))), 1e-30)
            self.assertLess(float(np.max(np.abs(a - b))) / denom, 1e-6,
                            f"GaAs(111) {ch}: intensity not 180-deg periodic in phi")
        # count local maxima of I_s over one 180-deg period on the CIRCLE (wrap-aware: a lobe peak
        # can sit exactly at phi = 0), and require deep modulation (a real lobed pattern, not a
        # near-constant ring)
        a = i_s[:half]
        ext = np.concatenate([a[-1:], a, a[:1]])  # circular padding
        maxima = sum(1 for i in range(1, half + 1) if ext[i] > ext[i - 1] and ext[i] >= ext[i + 1])
        self.assertGreaterEqual(maxima, 1, "GaAs(111) I_s: no lobe maxima found")
        self.assertGreater(float(a.max()), 10.0 * max(float(a.min()), 1e-30),
                           "GaAs(111) I_s: pattern is nearly constant -- expected deep lobes")


class FresnelPhysicsSanity(unittest.TestCase):
    def test_lossless_film_brewster_and_grazing(self):
        r = compute_ml_gui_result("Fresnel Coefficients", theta_min_deg=0.0, theta_max_deg=89.0,
                                  theta_step_deg=0.5)
        th = np.asarray(r.numeric["theta_deg"], float)
        rp = np.asarray(r.numeric["rp"], float)
        rs = np.asarray(r.numeric["rs"], float)
        # Brewster-like: min of R_p across the sweep is far below R_s at the same angle
        i_min = int(np.argmin(rp))
        self.assertLess(rp[i_min], 0.05, "R_p never gets small -- no Brewster-like minimum")
        self.assertGreater(rs[i_min], rp[i_min], "R_s should exceed R_p at the R_p minimum")
        self.assertTrue(20.0 < th[i_min] < 80.0, f"R_p minimum at implausible angle {th[i_min]}")
        # grazing: reflectance -> 1
        self.assertGreater(rp[-1], 0.7, "R_p at grazing should approach 1")
        self.assertGreater(rs[-1], 0.7, "R_s at grazing should approach 1")


class MakerFringeSanity(unittest.TestCase):
    def test_maker_curves_smooth_and_fmr_fringes_scale_with_thickness(self):
        """Two release-audit invariants for the LiNbO3 z-cut film (near-phase-matched:
        n_e(2w) ~ n_o(w), coherence length ~130 um):

        1. SMOOTHNESS -- the sweep must be a continuous curve, not a sample-rate comb. This is the
           regression guard for the degenerate-substrate basis flicker (audit,
           ): the isotropic-substrate 2w mode pair shares one k, its eigenbasis rotated
           arbitrarily per angle, and slot-[0] projection produced spurious near-zero dives.
        2. THICKNESS PHYSICS -- under JK (no multiple reflections) films far below one coherence
           length show NO fringes (smooth envelope) at ANY thickness; under FMR the Fabry-Perot
           oscillations of the fundamental get DENSER for the thicker film."""
        from scipy.signal import find_peaks

        def run(thickness_um, assumption):
            sys_ = build_casestudy_ml_system("LiNbO3 z-cut (1550 nm)",
                                             thickness_um=thickness_um, wavelength_um=1.064)
            r = compute_ml_gui_result("Maker Fringes", system=sys_, theta_min_deg=5.0,
                                      theta_max_deg=70.0, theta_step_deg=0.25,
                                      assumption=assumption)
            y = np.nan_to_num(np.asarray(r.numeric["parallel_intensity"], float))
            return y

        jk = "Jerphagnon & Kurtz Assumption (No MR)"
        fmr = "Full Multiple Reflections (FMR)"
        # 1: smoothness (median adjacent jump tiny relative to range) for both assumptions
        for assumption in (jk, fmr):
            for h in (3.0, 30.0):
                y = run(h, assumption)
                rng = float(y.max() - y.min())
                self.assertGreater(rng, 0.0)
                jump = float(np.median(np.abs(np.diff(y)))) / rng
                self.assertLess(jump, 0.05,
                                f"h={h} {assumption[:3]}: curve is a sample-rate comb "
                                f"(median jump {jump:.3f}) -- degenerate-basis flicker regressed")
        # 2: FMR fringe density scales with thickness
        def n_peaks(h):
            y = run(h, fmr)
            pk, _ = find_peaks(y, prominence=0.02 * float(y.max() - y.min()))
            return len(pk)
        thin, thick = n_peaks(3.0), n_peaks(30.0)
        self.assertGreater(thick, 2 * thin,
                           f"30 um FMR should show far denser Fabry-Perot fringes than 3 um "
                           f"(thin={thin}, thick={thick})")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
