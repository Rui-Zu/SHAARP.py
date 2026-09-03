"""Fence: the paper-reproduction cases (benchmarks/paper_cases.py) keep matching the published figures'
key numeric signatures -- crystal-class symmetry, the relative p/s channel scaling the panels annotate,
and the multilayer observables (reflected vs transmitted). Guards the two flagship notebooks against
a silent regression in the case builders or the underlying solver.
"""
import unittest

import numpy as np


def _peak_phi(curve, key):
    phi = np.asarray(curve["phi_deg"], float)
    return float(phi[int(np.argmax(np.asarray(curve[key], float)))])


def _dist_to_set(angle_deg, targets, period=360.0):
    return min(abs(((angle_deg - t + period / 2) % period) - period / 2) for t in targets)


class SIPaperCaseTests(unittest.TestCase):
    def test_all_si_cases_build_and_are_nonzero(self):
        from benchmarks.paper_cases import SI_CASES
        for c in SI_CASES.values():
            curves = c.polar_curves(n_phi=181)
            peak = max(float(np.max(curves[t]["intensity_p"]) + np.max(curves[t]["intensity_s"]))
                       for t in c.thetas_deg)
            self.assertTrue(np.isfinite(peak) and peak > 0, f"{c.fig} gave zero/non-finite SHG")

    def test_channel_scaling_matches_panels(self):
        """KTP: I_s >> I_p (paper scales I_p x10). LiNbO3: I_p >> I_s (paper scales I_s x15)."""
        from benchmarks.paper_cases import SI_CASES
        def peaks(key):
            c = SI_CASES[key]; cur = c.polar_curves(n_phi=181)
            ip = max(float(np.max(cur[t]["intensity_p"])) for t in c.thetas_deg)
            is_ = max(float(np.max(cur[t]["intensity_s"])) for t in c.thetas_deg)
            return ip, is_
        # Exact published-physics bands (exactness campaign): with the paper's own
        # Table-1 800-nm d tensors, KTP gives Is/Ip = 10.4 (matches the panel's x10 -- near-unity
        # channel calibration) and LiNbO3 gives Ip/Is = 18.2 (the panel's x15 is the EXPERIMENTAL
        # display ratio including the x/y detector calibration the original fit models -- fitting
        # notebook "800 nm x-LNO Different Intensity of x and y").
        # KTP with the paper's Table-1 tensor gives Is/Ip = 8.7; the panel's x10 is the
        # experimental display ratio (the xKTP fit's second-channel amplitudes are ~1.15-1.37,
        # and 8.7 x ~1.15 = 10 -- the same channel-calibration semantics as LiNbO3's x15).
        ip_k, is_k = peaks("fig6")      # KTP (100)
        self.assertTrue(7.9 < is_k / ip_k < 9.6,
                        f"KTP Is/Ip = {is_k/ip_k:.2f} drifted from the published-physics 8.7 (panel x10)")
        ip_l, is_l = peaks("fig5")      # LiNbO3 (11-20)
        self.assertTrue(16.3 < ip_l / is_l < 20.1,
                        f"LiNbO3 Ip/Is = {ip_l/is_l:.2f} drifted from the published-physics 18.2")

    def test_symmetry_of_lobe_positions(self):
        """LiNbO3 (11-20) I_p is a dumbbell along 0/180; KTP (100) I_p is a 4-lobe near 45/135/..."""
        from benchmarks.paper_cases import SI_CASES
        c5 = SI_CASES["fig5"]; cur5 = c5.polar_curves(n_phi=361)[c5.thetas_deg[-1]]
        self.assertLess(_dist_to_set(_peak_phi(cur5, "intensity_p"), (0.0, 180.0, 360.0)), 20.0,
                        "LiNbO3 I_p peak not along the 0/180 dumbbell axis")
        c6 = SI_CASES["fig6"]; cur6 = c6.polar_curves(n_phi=361)[c6.thetas_deg[-1]]
        self.assertLess(_dist_to_set(_peak_phi(cur6, "intensity_p"), (45.0, 135.0, 225.0, 315.0)), 20.0,
                        "KTP I_p peak not on the 45/135 four-lobe axis")


class TestSiFig7TaAsEffectiveIndex(unittest.TestCase):
    """Fence: published si-2022 Fig 7(b) — TaAs (112) eo-mode effective index vs incident angle.

    F45 ( — earlier text here said "F42,", the wrong session
    number) closed the open item 'TaAs registry azimuth': SHAARP.si V1.04
    propagates its transmitted waves DOWNWARD (u = (sin θ, 0, −cos θ)), while this port (like
    SHAARP.ml setup.nb solveSnell, its validation oracle) propagates UPWARD, so the verbatim
    V1.04 Z-axes-in-lab matrix consumed as-is z-mirrors the oblique (112) tilt and MIRRORS the
    published trend. build_casestudy_material now applies the exact 180° lab-azimuth
    compensation (_SI_FRAME_AZIMUTH_DEG). The pinned endpoint values below are the LIVE
    SHAARP.si V1.04 replication run (its own hklConvert (112)/[1,-1,0] state + its downward
    convention, Wolfram 14.3), which equals the published panel: n_eo RISES
    4.3006 → 4.3512 and k_eo FALLS 2.1684 → 2.0541 across θi = 0.5° → 80°. The wrong (mirrored)
    azimuth gives 4.2547/2.3469 at 80° — far outside the bands, so this fence DISCRIMINATES the
    exact route from the plausible-looking mirror (fence rule)."""

    def _eo_index(self, material, theta_deg):
        import math

        from shaarp.anisotropic import solve_snell_modes
        from shaarp.tensors import rotate_rank2_crystal_to_lab

        eps_lab = rotate_rank2_crystal_to_lab(material.epsilon_omega,
                                              material.orientation.rotation_matrix())
        modes = solve_snell_modes(eps_lab, 1.0 + 0j, math.radians(theta_deg))
        n_o = complex(np.sqrt(np.asarray(material.epsilon_omega, dtype=complex)[0, 0]))
        cands = [complex(m.refractive_index) for m in (modes.fast, modes.slow)]
        eo = [n for n in cands if abs(n - n_o) > 0.3]
        self.assertEqual(len(eo), 1, f"eo branch not identifiable at {theta_deg} deg: {cands}")
        return eo[0]

    def test_orientation_carries_the_si_frame_azimuth_fix(self):
        """The built material's c axis must lean toward −L1 (published panel-(a) cross-section
        drawing), i.e. rows = verbatim registry rows · diag(−1,−1,1)."""
        from shaarp.casestudy_materials import build_casestudy_material

        mat = build_casestudy_material("TaAs (112)", wavelength_um=0.8)
        c_lab = np.asarray(mat.orientation.rotation_matrix(), dtype=float)[2]
        np.testing.assert_allclose(c_lab, [-0.9228762828632939, 0.0, 0.38509656779622115],
                                   atol=1e-9, err_msg="TaAs (112) si-frame azimuth fix missing")

    def test_effective_index_endpoints_and_trend_match_published(self):
        from shaarp.casestudy_materials import build_casestudy_material

        mat = build_casestudy_material("TaAs (112)", wavelength_um=0.8)
        n_lo = self._eo_index(mat, 0.5)
        n_hi = self._eo_index(mat, 80.0)
        # live V1.04 values: 4.300614+2.168417j (0.5 deg), 4.351173+2.054143j (80 deg)
        self.assertAlmostEqual(n_lo.real, 4.300614, delta=5e-3,
                               msg=f"n_eo(0.5deg) = {n_lo.real:.6f} off the live-V1.04 4.3006")
        self.assertAlmostEqual(n_lo.imag, 2.168417, delta=5e-3,
                               msg=f"k_eo(0.5deg) = {n_lo.imag:.6f} off the live-V1.04 2.1684")
        self.assertAlmostEqual(n_hi.real, 4.351173, delta=5e-3,
                               msg=f"n_eo(80deg) = {n_hi.real:.6f} off the live-V1.04 4.3512")
        self.assertAlmostEqual(n_hi.imag, 2.054143, delta=5e-3,
                               msg=f"k_eo(80deg) = {n_hi.imag:.6f} off the live-V1.04 2.0541")
        self.assertGreater(n_hi.real, n_lo.real,
                           "published Fig 7(b) trend lost: n_eo must RISE with theta_i")
        self.assertLess(n_hi.imag, n_lo.imag,
                        "published Fig 7(b) trend lost: k_eo must FALL with theta_i")


class MLPaperCaseTests(unittest.TestCase):
    def test_fig6_heterostructure_reflected_nonzero_transmitted_blocked(self):
        """ZnO//Pt//Al2O3: Pt is opaque at 1550 nm -> transmitted SHG ~0, reflected SHG is the signal."""
        from benchmarks.paper_cases import ASSUMPTION_HH, ml_fig6_system, ml_maker, ml_polar
        s = ml_fig6_system()
        _, ip_r, is_r = ml_polar(s, theta_deg=45.0, channel="reflected", n_phi=37)
        refl_peak = float(max(np.max(ip_r), np.max(is_r)))
        self.assertGreater(refl_peak, 1e-6, "reflected SHG of the ZnO stack should be non-negligible")
        _, i_trans = ml_maker(s, ASSUMPTION_HH, th_max=40.0, step=5.0)
        self.assertLess(float(np.max(i_trans)) / refl_peak, 1e-3,
                        "transmitted SHG should be ~0 through the opaque 200 nm Pt electrode")

    def test_fig4d_published_central_bump_and_mirror_amplification(self):
        """Published Fig 4(d) EXACT inputs (h0 = 121.18 um -- the generating cell's ``hMRP1``, Au = 13.9 nm;
        see ml_fig4d_system's provenance): the plain-FMR curve at normal incidence sits at the published
        fraction of its own peak. The author's closed-form FMR model gives FMR(0)/FMR(max) = 3.000/14.881 =
        0.2016 at this thickness (my solver 0.212); the ~0.35 bump seen in the published panel is the
        EXPERIMENT / the theta+h+lambda-averaged curve, not the raw FMR (the raw FMR shows the 0.22 centre
        dip with 0.30 side peaks at +-3 deg). NOTE the value is a fringe-phase quantity: at the earlier
        121.0 um it read 0.41 -- one reason the thickness must be the published 121.18."""
        from benchmarks.paper_cases import ASSUMPTION_FMR, ml_fig4d_system, ml_maker
        s = ml_fig4d_system()
        # two windows, both FRINGE-RESOLVED (0.05 deg; the ~0.4-0.5 deg fringes alias at >=0.2):
        # near-normal for the bump value, 30-45 deg for the envelope peak the bump is measured against
        _, i_0 = ml_maker(s, ASSUMPTION_FMR, th_min=0.0, th_max=2.0, step=0.05)
        _, i_w = ml_maker(s, ASSUMPTION_FMR, th_min=30.0, th_max=45.0, step=0.05)
        bump = float(i_0[0] / np.max(i_w))
        self.assertGreater(bump, 0.16, f"(d) centre value lost (got {bump:.3f}, published model 0.2016)")
        self.assertLess(bump, 0.26, f"(d) centre value too large (got {bump:.3f}, published model 0.2016)")

    def test_fig4_published_display_recipe_numbers(self):
        """The panel-(d) display recipe (FIG4_PUBLISHED: common-{min,max} Rescale, FMR x1.07, HH x1.5)
        reproduces the published panel's landmark values from the author's own model exports at 121.18 um:
        HH x1.5 peak 0.786, HH x1.5 at theta=0 0.342, FMR x1.07 at theta=0 0.216 (pixel-verified against
        the published figure crop). Pure bookkeeping on the bundled references -- no solver."""
        from benchmarks.paper_figures import FIG4_PUBLISHED, _fig4d_fmr_reference, _fig4d_hh_reference, _rescale
        rec = FIG4_PUBLISHED["d"]
        self.assertEqual(rec["h_um"], 121.18)
        th_h, i_h = _fig4d_hh_reference()
        th_f, i_f = _fig4d_fmr_reference()
        lo = min(float(np.min(i_h)), float(np.min(i_f))); hi = max(float(np.max(i_h)), float(np.max(i_f)))
        y_h = rec["hh_scale"] * _rescale(i_h, lo, hi); y_f = rec["fmr_scale"] * _rescale(i_f, lo, hi)
        self.assertAlmostEqual(float(np.max(y_h)), 0.786, delta=0.005)
        self.assertAlmostEqual(float(y_h[0]), 0.342, delta=0.005)
        self.assertAlmostEqual(float(y_f[0]), 0.216, delta=0.005)
        self.assertAlmostEqual(float(np.max(y_f)), 1.07, delta=1e-9)
        # grids: HH 0.01 deg (5001 pts to 50 deg), FMR 0.02 deg
        self.assertAlmostEqual(float(np.median(np.diff(th_h))), 0.01, places=6)
        self.assertAlmostEqual(float(np.median(np.diff(th_f))), 0.02, places=6)

    def test_fig4d_rendered_centre_values_track_the_published_panel(self):
        """The RENDERED panel-(d) recipe (my FMR through the actual figure rescale on my-FMR u his-HH) at the
        fringe-phase-sensitive theta = 0 point: FMR x1.07 centre. Published pixel value 0.216 (his model 0.2157);
        my solver renders 0.228 (my FMR(0) sits 5.4% above his closed-form model there while the shape corr is
        0.9994). Fenced honestly at the measured residual so a regression (or a fix) is visible -- added after an
        independent adversarial review pass flagged that the recipe fence alone was self-referential."""
        from benchmarks.paper_cases import ASSUMPTION_FMR, ml_fig4d_system, ml_maker
        from benchmarks.paper_figures import FIG4_PUBLISHED, _fig4d_hh_reference, _rescale
        rec = FIG4_PUBLISHED["d"]
        _, i_0 = ml_maker(ml_fig4d_system(), ASSUMPTION_FMR, th_min=0.0, th_max=1.0, step=0.05)
        _, i_w = ml_maker(ml_fig4d_system(), ASSUMPTION_FMR, th_min=30.0, th_max=45.0, step=0.05)
        _, i_hh = _fig4d_hh_reference()
        lo = min(0.0, float(np.min(i_hh))); hi = max(float(np.max(i_w)), float(np.max(i_hh)))
        centre = rec["fmr_scale"] * float(_rescale(i_0[0], lo, hi))
        self.assertTrue(0.20 < centre < 0.24, f"rendered FMR x1.07 centre {centre:.4f} (published 0.216, expected ~0.228)")

    def test_fig4d_shaarp_py_HH_in_author_geometry_matches_his_HH_model(self):
        """Panel (d) HH is now SHAARP.py's own HH computed the way the author's published HH model was built:
        quartz on an Au HALF-SPACE (the model's 2w Fabry-Perot constant |r1 r2| = 0.0559 = quartz->air x
        quartz->bulk-Au, no Au-thickness dependence) with his beam-frame projection of the transmitted field
        (IT2wPout = |(Inverse[RNum].E_t)[[1]]|^2). Fence on the fringe-resolved 30-45 deg window at 0.05 deg
        (his 0.01-deg model interpolated onto my grid): shape corr >= 0.995, peak ratio within 0.97-1.02
        (measured 0.998 / 0.992). Near normal incidence the port sits ~4.6% above ALL of his SLAB closed forms
        (bare and Au, FMR and HH alike) -- a shared systematic, not an HH ingredient, so it is not fenced here."""
        from benchmarks.paper_figures import ml_fig4d_hh_author_geometry, _fig4d_hh_reference
        th, i = ml_fig4d_hh_author_geometry(th_min=30.0, th_max=45.0, step=0.05)
        hr, ir = _fig4d_hh_reference()
        his = np.interp(th, hr, ir)
        corr = float(np.corrcoef(i, his)[0, 1]); mag = float(np.max(i) / np.max(his))
        self.assertGreater(corr, 0.995, f"SHAARP.py HH (author geometry) vs his HH model corr {corr:.4f}")
        self.assertTrue(0.97 < mag < 1.02, f"peak ratio my/his {mag:.4f} (expected ~0.992)")

    def test_fig4b_my_bare_slab_curves_match_his_models(self):
        """Panel (b): my FMR / HH / JK for the bare 123.6 um Z-quartz slab reproduce the author's three
        published bare-slab models (fig4b_*_reference.csv, h0 = 123.6) -- shape corr >= 0.998 and peak
        magnitude within 5% -- on a fringe-resolved window (30-45 deg, 0.05 deg; his denser curves are
        interpolated onto my grid, never the reverse)."""
        from benchmarks.paper_cases import ASSUMPTION_FMR, ASSUMPTION_HH, ASSUMPTION_JK, ml_fig4_system, ml_maker
        from benchmarks.paper_figures import _fig4b_reference
        s = ml_fig4_system(au=False)
        for kind, asm in (("fmr", ASSUMPTION_FMR), ("hh", ASSUMPTION_HH), ("jk", ASSUMPTION_JK)):
            th, i = ml_maker(s, asm, th_min=30.0, th_max=45.0, step=0.05)
            his_th, his_i = _fig4b_reference(kind)
            his = np.interp(th, his_th, his_i)
            corr = float(np.corrcoef(i, his)[0, 1]); mag = float(np.max(i) / np.max(his))
            self.assertGreater(corr, 0.998, f"(b) {kind.upper()} vs the author's model corr {corr:.4f}")
            self.assertTrue(0.95 < mag < 1.05, f"(b) {kind.upper()} peak ratio my/his {mag:.4f}")

    def test_fig4d_my_FMR_matches_his_closed_form_FMR(self):
        """Fig 4(d) method-equivalence fence (option B). My numeric FMR reproduces the author's OWN
        published closed-form FMR model for the Quartz+backside-Au panel (QuartzAuSimuMRP1S0p02.mx,
        h0 = 121.18 um, phi = 0; embedded as benchmarks/fig4d_fmr_reference.csv) to corr >= 0.998 on the
        shared angular window (measured 0.9994 at the published 121.18 um; peak ratio 0.998 on the 0.05 deg grid,
        1.0001 on a 0.02 deg grid -- the fine-fringe peak is sampling-sensitive). This FMR curve is the PHYSICS RESULT; the panel-(d) HH
        curve is embedded from the author's HH .mx (fig4d_hh_reference.csv) because my numeric single-pass-omega
        HH diverges for the strong 13.9 nm Au reflector -- HH is the paper's deliberately-failing
        illustrative curve, so the physics claim rides on THIS fence, not on HH.

        Comparison discipline: sample both curves on MY 0.05 deg grid, interpolating only his DENSER
        (0.02 deg) reference. The reverse (linear-interpolating my coarse curve onto his fine grid) is an
        aliasing artifact that reads ~0.988 and understates the true agreement."""
        from benchmarks.paper_cases import ASSUMPTION_FMR, ml_fig4d_system, ml_maker
        from benchmarks.paper_figures import _fig4d_fmr_reference
        his_th, his_i = _fig4d_fmr_reference()
        th_f, i_f = ml_maker(ml_fig4d_system(), ASSUMPTION_FMR, th_max=45.0, step=0.05)
        his_on_mine = np.interp(th_f, his_th, his_i)          # interp only the denser curve
        corr = float(np.corrcoef(i_f / (np.max(i_f) or 1.0),
                                 his_on_mine / (np.max(his_on_mine) or 1.0))[0, 1])
        self.assertGreater(corr, 0.998,
                           f"my FMR vs the author's closed-form FMR corr {corr:.4f} < 0.998 (Fig 4d equivalence)")
        # magnitudes agree too (raw a.u. peaks): my 14.852 (0.05 deg grid; 14.883 at 0.02 deg) vs his 14.881 at 121.18 um
        mag = float(np.max(i_f) / np.max(his_i))
        self.assertTrue(0.95 < mag < 1.05, f"FMR peak magnitude ratio {mag:.4f} off (my/his)")

    def test_fig4d_HH_reference_sits_below_FMR(self):
        """Fig 4(d) embedded-HH sanity (option B). the author's HH model (fig4d_hh_reference.csv), even after the
        paper's x1.5 display factor, stays BELOW his FMR reference peak -- HH under-captures the total
        transmitted SHG (the whole point of the panel: FMR departs upward with the Au mirror)."""
        from benchmarks.paper_figures import _fig4d_hh_reference, _fig4d_fmr_reference
        _th_h, i_h = _fig4d_hh_reference()
        _th_f, i_f = _fig4d_fmr_reference()
        ratio = float(np.max(i_h) / np.max(i_f))
        self.assertTrue(0.3 < ratio < 0.75, f"HH/FMR peak ratio {ratio:.3f} off (expected 0.524 at 121.18 um)")
        self.assertLess(1.5 * ratio, 1.0, "HH x1.5 must stay below the FMR peak on the normalized axis")

    def test_fig3_hh_and_jk_envelopes_are_close(self):
        """300 um X-cut quartz: HH and JK share nearly the same Maker envelope (differ only in fine fringes)."""
        from benchmarks.paper_cases import ASSUMPTION_HH, ASSUMPTION_JK, ml_fig3_system, ml_maker
        s = ml_fig3_system()
        _, i_hh = ml_maker(s, ASSUMPTION_HH, th_max=60.0, step=2.0)
        _, i_jk = ml_maker(s, ASSUMPTION_JK, th_max=60.0, step=2.0)
        ratio = float(np.max(i_hh) / np.max(i_jk))
        self.assertTrue(0.8 < ratio < 1.25, f"HH/JK envelope peak ratio {ratio:.3f} not ~1")

    def test_figS7_mos2_sixfold_and_twist_bilayer_stronger(self):
        """Twist bilayer MoS2 (SI Fig S7): monolayer RA-SHG is six-fold; the coherent bilayer is stronger."""
        from benchmarks.paper_cases import ml_mos2_system, ml_ra_reflected_polar
        az, ip_mono, _ = ml_ra_reflected_polar(ml_mos2_system([0.0]), n_azimuth=97)
        y = ip_mono[:-1] - float(np.mean(ip_mono[:-1]))          # drop duplicated 360deg endpoint
        dominant_fold = int(np.argmax(np.abs(np.fft.rfft(y))[1:]) + 1)
        self.assertEqual(dominant_fold, 6, f"monolayer MoS2 RA-SHG not six-fold (got {dominant_fold})")
        _, ip_bi, _ = ml_ra_reflected_polar(ml_mos2_system([25.0, 0.0]), n_azimuth=97)
        self.assertGreater(float(np.max(ip_bi)), 1.3 * float(np.max(ip_mono)),
                           "twist bilayer MoS2 should emit MORE than a single monolayer (coherent add)")

    def test_figS7a_mos2_kappa_bounded_and_oscillatory(self):
        """Published SI Fig S7(a): the bilayer coherence kappa(dTheta=0, t) on the Al2O3 wafer stays
        a proper coherence factor (bounded by ~1) and OSCILLATES with the substrate FP phase."""
        from benchmarks.paper_cases import ml_mos2_kappa_sweep
        t = np.linspace(480.0, 481.5, 16)          # ~2 fringe periods, cheap
        _, kappa = ml_mos2_kappa_sweep(t)
        self.assertTrue(np.all(kappa > 0.3) and np.all(kappa < 1.05),
                        f"kappa left the coherence band: [{kappa.min():.3f}, {kappa.max():.3f}]")
        self.assertGreater(float(np.ptp(kappa)), 0.01,
                           "kappa should oscillate with the substrate Fabry-Perot phase")

    def test_fig7_stacking_order_controls_interference(self):
        """Fig 7(d): flipping LNO's polar axis P (±L1) against quartz(001)[100] flips the two-crystal
        SHG interference — one stack is enhanced, the other suppressed, and the two LNO-alone cases
        are identical (a monolayer's intensity is invariant under P flip)."""
        from benchmarks.paper_cases import ml_fig7_case, ml_polar
        il1_phi0 = {}
        for case in (1, 2, 3, 4):
            _, ip, _ = ml_polar(ml_fig7_case(case), theta_deg=0.5, channel="transmitted", n_phi=13)
            il1_phi0[case] = float(ip[0])   # I_L1 at phi = 0
        self.assertAlmostEqual(il1_phi0[1], il1_phi0[4], delta=0.02 * il1_phi0[1],
                               msg="LNO-alone intensity should be invariant under the P flip")
        hi, lo = max(il1_phi0[2], il1_phi0[3]), min(il1_phi0[2], il1_phi0[3])
        self.assertGreater(hi / lo, 1.05,
                           "stacking-order (P sign) should modulate the two-crystal SHG interference")
        self.assertGreater(hi, il1_phi0[1], "constructive stack should exceed LNO alone")
        self.assertLess(lo, il1_phi0[1], "destructive stack should fall below LNO alone")


class PaperFidelityFixes(unittest.TestCase):
    """Fences for the display-fidelity fixes that raised the reproduced ♯SHAARP.ml figures
    to the published treatment: finer (0.02°) fine fringes, the author's θ+h+λ effective-thickness Maker
    averaging, the computed polar ×N display scaling, and the new Fig 8c κ-vs-twist panel."""

    def test_fig3_fine_fringe_smooth_and_matches_analytic(self):
        """Fix 1: the fine-fringe HH curve swept at 0.02° gives ≥8 samples per ~0.2° fringe period
        (smooth, not the aliased 0.05°) and matches the byte-exact analytic HH on a shared grid."""
        from benchmarks.herman_hayden_maker import herman_hayden_quartz_reference_curve
        from benchmarks.paper_cases import ASSUMPTION_HH, ML_CASES, ml_maker
        fine_step = 0.02
        self.assertGreaterEqual(0.2 / fine_step, 8.0)   # >=8 pts per ~0.2 deg fringe -> smooth line
        s = ML_CASES["fig3"].system_builder()
        th, i = ml_maker(s, ASSUMPTION_HH, th_min=24.0, th_max=26.0, step=fine_step)
        n = int(round((26.0 - 24.0) / fine_step)) + 1
        _thr, ir = herman_hayden_quartz_reference_curve(24.0, 26.0, n)
        corr = float(np.corrcoef(i / (np.max(i) or 1.0), ir / (np.max(ir) or 1.0))[0, 1])
        self.assertGreater(corr, 0.99, f"fine-fringe HH vs analytic corr {corr:.4f} < 0.99 (aliased?)")

    def test_fig4_averaged_fmr_is_smoother_than_raw(self):
        """Fix 2: the θ+h+λ (effective-thickness broadening + 3° θ boxcar) averaged FMR is finite and
        SMOOTHER than the raw FMR fringes -- the author's method that lets the Expt scan match FMR."""
        from benchmarks.paper_cases import ASSUMPTION_FMR, ml_fig4d_system, ml_maker
        from benchmarks.paper_figures import ml_fig4_averaged_fmr
        _th_a, i_av = ml_fig4_averaged_fmr(step=0.5, th_max=20.0, n_h=5)   # small/fast
        self.assertTrue(np.all(np.isfinite(i_av)) and float(np.max(i_av)) > 0)
        _th, i_raw = ml_maker(ml_fig4d_system(), ASSUMPTION_FMR, th_max=20.0, step=0.5)
        self.assertLess(float(np.std(np.diff(i_av))), float(np.std(np.diff(i_raw))),
                        "averaged FMR not smoother than raw FMR (broadening not applied)")

    def test_polar_display_factors_are_rui_computed(self):
        """Fix 3: the interference figure lifts the weak L2 channel by the author's computed ×N (int(1.1·g1/g2)
        cross-channel ≈ the paper's ×61, plus large per-case factors) instead of per-channel fill."""
        from benchmarks.paper_figures import ml_interference_figure
        _fig, st = ml_interference_figure(n_phi=37)
        self.assertTrue(40 <= st["L2_cross_factor"] <= 90,
                        f"L2 cross factor {st['L2_cross_factor']} not near the paper's ×61")
        self.assertGreater(max(st["L2_case_factors"]), 10,
                           "no strong per-case weak-channel lift (×N scaling not applied)")

    def test_fig8c_kappa_follows_cos3theta(self):
        """Fix 4: κ(Δθ) for the twist bilayer starts coherent (~cos0>0), crosses toward the anti-phase
        cos(3·60°)=−1 minimum -- i.e. it tracks cos(3Δθ), decreasing monotonically over 0→60°."""
        from benchmarks.paper_cases import ml_mos2_kappa_twist_sweep
        tw, k = ml_mos2_kappa_twist_sweep((0.0, 30.0, 60.0), substrate_um=504.0, n_azimuth=25)
        self.assertGreater(k[0], 0.5, f"κ(0°)={k[0]:.3f} not coherent (~1)")
        self.assertLess(k[2], -0.4, f"κ(60°)={k[2]:.3f} not anti-coherent (~cos180=−1)")
        self.assertTrue(k[0] > k[1] > k[2], f"κ not monotonically decreasing over twist: {k}")


if __name__ == "__main__":
    unittest.main()
