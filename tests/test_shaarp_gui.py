"""Merged SHAARP.si + SHAARP.ml GUI (shaarp/shaarp_gui.py) -- Phase 1: navigation skeleton.

The original SHAARP ships two separate Mathematica GUIs; this package merges them into ONE
ipywidgets panel with a top-level Tab to navigate between the single-interface (.si) and multilayer
(.ml) interfaces. The per-tab compute logic is pure/no-widget (compute_si_gui_result /
compute_ml_gui_result) so it is unit-testable headlessly; this test pins that every Functionality
mode of each interface routes a VALIDATED solver facade, and that the merged panel builds with both
navigable tabs.
"""

import unittest

import numpy as np

from shaarp.shaarp_gui import (
    ML_FUNCTIONALITIES,
    SHG_POINT_GROUPS,
    SI_FUNCTIONALITIES,
    build_constrained_d_voigt,
    build_maker_figure,
    build_si_polarimetry_figure,
    compute_ml_gui_result,
    compute_si_gui_result,
    count_discontinuities,
    default_ml_system,
    mask_maker_singularities,
    point_group_free_components,
    si_polarimetry_curve,
)


class AnalyticalDerivationSteps(unittest.TestCase):
    """The full-analytical panel must expose the published
    derivation stages (transmitted omega fields -> P_NL -> inhomogeneous 2omega fields) that
    build up to the final reflected amplitude -- each the ACTUAL captured object, validated vs
    the published supplementary (tasks #37-40)."""

    def test_full_analytical_yields_four_typeset_steps(self):
        from shaarp.shaarp_gui import analytical_derivation_items, compute_si_gui_result
        r = compute_si_gui_result("Full Analytical", theta_deg=45.0)
        steps = analytical_derivation_items(r)
        # Step 0 (FA-1 Stage 2) is the layered definition chain: the final amplitude is written in
        # named intermediates, so the quantities that used to appear inline (phi, theta_i, d_ij)
        # now live in their definitions -- which must therefore be part of the shown derivation.
        self.assertEqual([h.split(" — ")[0] for h, _, _ in steps],
                         ["Step 0", "Step 1", "Step 2", "Step 3"])
        headings = " ".join(h for h, _, _ in steps)
        self.assertIn("Named intermediates", headings)
        self.assertIn("Transmitted fundamental", headings)
        self.assertIn("Nonlinear polarization", headings)
        self.assertIn("Inhomogeneous", headings)
        bodies = " ".join(b for _, _, b in steps)
        self.assertIn("d<sub>14</sub>", bodies)     # d14 typeset subscript in P_NL / inhom
        self.assertIn("φ", bodies)                  # phi typeset
        for raw in ("**", "sqrt(", "theta_i", "n_omega"):
            self.assertNotIn(raw, bodies, f"raw token {raw!r} leaked into a derivation step")

    def test_numeric_run_has_no_derivation_steps(self):
        from shaarp.shaarp_gui import analytical_derivation_items, compute_si_gui_result
        r = compute_si_gui_result("SHG Simulation", theta_deg=45.0)
        self.assertEqual(analytical_derivation_items(r), [])


class MathDisplayHtml(unittest.TestCase):
    """F33 fence: the analytical panel's typeset layer must render the original's notation and
    must NEVER leak into the machine-readable layer (Copy / .txt export use the raw text)."""

    def test_screenshot_fragment_typesets(self):
        from shaarp.shaarp_gui import math_display_html
        s = ("sqrt(n_omega**2 - sin(theta_i)**2)*(-32*d14*sqrt(n_omega**2 - sin(theta_i)**2)"
             "*sin(phi)*sin(theta_i)*cos(phi)*cos(theta_i)**2")
        out = math_display_html(s)
        for needle in ("n<sub>ω</sub>", "<sup>2</sup>", "d<sub>14</sub>",
                       "θ<sub>i</sub>", "φ", "√(", "·"):
            self.assertIn(needle, out)
        for raw in ("**", "theta_i", "n_omega", "sqrt(", "phi"):
            self.assertNotIn(raw, out, f"raw token {raw!r} leaked into the typeset view")

    def test_display_floats_rounded_to_six_significant_digits(self):
        # the ML partial expressions carry full-precision float coefficients; the DISPLAY rounds
        # them to ~6 significant digits (like the original's typeset output) while the raw layer
        # keeps sympy's full precision
        from shaarp.shaarp_gui import math_display_html
        out = math_display_html("d15*(-0.00114472595030629 - 2.21595031103526e-17)")
        self.assertNotIn("0.00114472595030629", out)
        self.assertIn("0.00114473", out)
        self.assertIn("2.21595e-17", out)

    def test_mathematica_format_export(self):
        """the author's request: closed forms in Wolfram Language syntax. The conversion
        was verified numerically against a LIVE Wolfram kernel (wolframscript) at build time
        (agreement to 16 digits); this fence pins the format contract: WL function heads and
        power syntax, and NO underscore symbol names (underscores are Blank patterns in WL)."""
        import re
        from shaarp.shaarp_gui import (
            analytical_expression_mathematica,
            compute_ml_gui_result,
            compute_si_gui_result,
            default_ml_system,
        )
        si = compute_si_gui_result("Full Analytical", theta_deg=45.0)
        m = analytical_expression_mathematica(si)
        self.assertIn("^(1/2)", m)          # roots in WL power syntax (sympy's Sqrt form)
        self.assertIn("Sin[", m)            # WL function heads, not sin(
        self.assertIn("\[Theta]i", m)
        self.assertIn("\[CurlyPhi]", m)
        self.assertIn("n\[Omega]", m)
        self.assertNotIn("**", m)
        body = re.sub(r"\(\*.*?\*\)", "", m, flags=re.S)  # comments may carry raw names
        for raw in ("theta_i", "n_omega", "sqrt(", "sin("):
            self.assertNotIn(raw, body, f"raw sympy token {raw!r} leaked into WL output")
        # a thickness is symbolic ONLY when its layer is flagged, so drive the exponentials
        # deliberately. Until F61 an unflagged stack still produced a bare `h` (I don't see SHG analytical symbol but only see a symbol h, what is that?") -- which is
        # why this fence used to pass with no flag set at all.
        import dataclasses

        base = default_ml_system()
        layers = list(base.layers)
        layers[1] = dataclasses.replace(layers[1], analytic_h=True)   # row 2 -> h2
        ml = compute_ml_gui_result(
            "Partial Analytical", system=dataclasses.replace(base, layers=layers))
        m2 = analytical_expression_mathematica(ml)
        self.assertTrue(("Exp[" in m2) or ("E^" in m2), "complex phases must be WL exponentials")
        self.assertIn("h2", m2, "the thickness symbol must carry the layer's ROW number (F61)")
        self.assertNotIn("**", m2)
        m3 = analytical_expression_mathematica(
            compute_ml_gui_result("Partial Analytical", system=base))
        self.assertNotIn("Exp[", m3,
                         "an unflagged stack must substitute every thickness (F61)")

    def test_ellipse_tile_square_limits_for_degenerate_linear_case(self):
        """With linear polarization the beam 'ellipse' is a
        degenerate line; the tile must keep FIXED symmetric limits + equal aspect so the box
        stays square instead of autoscaling the flat axis to float noise."""
        from shaarp.layer_stack import build_system_from_stack, default_stack
        from shaarp.shaarp_gui import (
            build_ml_polarimetry_figure,
            ml_beam_ellipses,
            ml_polarimetry_curve,
        )
        sysm = build_system_from_stack(default_stack(), wavelength_um=1.064)
        curve = ml_polarimetry_curve(sysm, theta_deg=20.0, n_phi=13)
        ell = ml_beam_ellipses(sysm, theta_deg=20.0)  # ellipticity 0 -> linear -> line
        fig = build_ml_polarimetry_figure(curve, point_group="3m", ellipses=ell)
        try:
            ax_e = [a for a in fig.axes if a.get_xlabel().startswith("$E_s")][0]
            self.assertEqual(tuple(round(v, 6) for v in ax_e.get_xlim()), (-1.1, 1.1))
            self.assertEqual(tuple(round(v, 6) for v in ax_e.get_ylim()), (-1.1, 1.1))
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_html_and_text_layers_coexist(self):
        from shaarp.shaarp_gui import (
            analytical_expression_html,
            analytical_expression_text,
            compute_si_gui_result,
        )
        res = compute_si_gui_result("Full Analytical", theta_deg=45.0)
        raw = analytical_expression_text(res)
        html = analytical_expression_html(res)
        self.assertIn("**", raw, "machine-readable layer must keep sympy notation")
        self.assertIn("<sup>", html)
        self.assertIn("<sub>", html)
        self.assertNotIn("**", html.replace("<sup>", "").replace("</sup>", ""),
                         "typeset layer must not show raw ** powers")

class SiTabComputeTests(unittest.TestCase):
    def test_absorbing_principal_aligned_curve_routes_numeric(self):
        """TaAs's complex absorbing eps entered
        at a principal-aligned (z-cut/Miller-001) orientation used to enter the full SYMBOLIC
        closed form, where sympy's expand on complex uniaxial radicals never finishes. Absorbing
        principal-aligned crystals must route through the validated numeric per-phi solver (the
        closed form's own 1e-12..1e-16 reference) — pinned here by making the symbolic constructor
        unreachable and requiring a finite, nonzero curve in bounded time."""
        import numpy as np
        from unittest import mock

        from shaarp import shaarp_gui as sg

        taas_w = (6.2591 + 18.8118j, 6.2591 + 18.8118j, 15.2221 + 18.0536j)
        taas_2w = (3.6748 + 13.3883j, 3.6748 + 13.3883j, 3.4997 + 14.6893j)
        d_free = {(0, 4): -92.2879, (1, 3): -92.2879,
                  (2, 0): 21.3876, (2, 1): 21.3876, (2, 2): 746.8152}
        # the curve function imports the symbolic solver lazily -> patch it at its SOURCE module
        with mock.patch(
                "shaarp.symbolic.solve_si_shg_full_analytical_symbolic",
                side_effect=AssertionError("symbolic closed form must not run for absorbing eps")):
            curve = sg.si_polarimetry_curve(
                "4mm", theta_deg=0.0, d_free=d_free, n_phi=13,
                eps_omega_principal=taas_w, eps_2omega_principal=taas_2w)
        i_p = np.asarray(curve["intensity_p"], dtype=float)
        i_s = np.asarray(curve["intensity_s"], dtype=float)
        self.assertTrue(np.all(np.isfinite(i_p)) and np.all(np.isfinite(i_s)))
        self.assertGreater(float(np.max(i_p) + np.max(i_s)), 0.0,
                           "absorbing 4mm z-cut curve unexpectedly all-zero")

    def test_every_si_functionality_returns_a_result(self):
        expected = {
            "SHG Simulation": "si_numeric_shaarp_si_compat",
            "Partial Analytical": "si_full_analytical_polarimetry",
            "Full Analytical": "si_full_analytical_polarimetry",
        }
        self.assertEqual(set(expected), set(SI_FUNCTIONALITIES))
        for func, kind in expected.items():
            res = compute_si_gui_result(func, point_group="-43m", theta_deg=45.0)
            self.assertEqual(res.kind, kind, f"si {func!r} routed the wrong backend")

    def test_unknown_si_functionality_raises(self):
        with self.assertRaises(ValueError):
            compute_si_gui_result("bogus")


class MlTabComputeTests(unittest.TestCase):
    def test_every_ml_functionality_returns_a_result(self):
        expected = {
            "SHG Simulation": "ml_numeric",
            "Maker Fringes": "maker_fringes",
            "Fresnel Coefficients": "fresnel_sweep",
            "Partial Analytical": "ml_partial_analytical_polarimetry",
        }
        self.assertEqual(set(expected), set(ML_FUNCTIONALITIES))
        for func, kind in expected.items():
            res = compute_ml_gui_result(
                func, point_group="-43m", theta_deg=20.0, theta_max_deg=20.0, theta_step_deg=10.0
            )
            self.assertEqual(res.kind, kind, f"ml {func!r} routed the wrong backend")

    def test_unknown_ml_functionality_raises(self):
        with self.assertRaises(ValueError):
            compute_ml_gui_result("bogus")

    def test_assumption_panel_routes_mrassumption(self):
        # the SHAARP.ml Assumptions panel: Full (FMR) vs JK must give genuinely different Maker
        # results (multiple reflections vs forward-only), and an unknown assumption must raise.
        full = compute_ml_gui_result("Maker Fringes", theta_min_deg=10.0, theta_max_deg=14.0,
                                     theta_step_deg=1.0, assumption="Full (FMR)")
        jk = compute_ml_gui_result("Maker Fringes", theta_min_deg=10.0, theta_max_deg=14.0,
                                   theta_step_deg=1.0, assumption="JK")
        a = np.abs(np.asarray(full.numeric["parallel_intensity"]))
        b = np.abs(np.asarray(jk.numeric["parallel_intensity"]))
        self.assertGreater(float(np.max(np.abs(a - b))) / float(a.max()), 0.01,
                           "Full vs JK must differ (multiple-reflection physics)")
        with self.assertRaises(ValueError):
            compute_ml_gui_result("Maker Fringes", assumption="bogus")

    def test_system_presets_resolve(self):
        from shaarp.shaarp_gui import ML_SYSTEM_PRESETS, resolve_ml_system_preset

        # the paper-stack presets (case-study fidelity audit): Fig 4 / Fig 6 / Fig 7
        self.assertEqual(list(ML_SYSTEM_PRESETS), [
            "Quartz + Au (Fig 4, 800 nm)", "ZnO / Pt / Al2O3 (Fig 6, 1550 nm)",
            "LiNbO3 / Quartz (Fig 7, 1550 nm)"])
        quartz = resolve_ml_system_preset("Quartz + Au (Fig 4, 800 nm)")
        self.assertEqual(len(quartz.layers), 4)
        self.assertAlmostEqual(float(quartz.layers[1].thickness_um), 121.2, places=6)
        # legacy names (pre-audit sessions/notebooks) still resolve
        legacy = resolve_ml_system_preset("Quartz + Au (docs, validated)")
        self.assertAlmostEqual(float(legacy.layers[1].thickness_um), 121.2, places=6)
        retired = resolve_ml_system_preset("LiNbO3 film (10 um)")
        self.assertAlmostEqual(float(retired.layers[1].thickness_um), 10.0, places=6)
        fig6 = resolve_ml_system_preset("ZnO / Pt / Al2O3 (Fig 6, 1550 nm)")
        self.assertEqual([lyr.shg_active for lyr in fig6.layers], [False, True, False, False])
        self.assertAlmostEqual(float(fig6.layers[1].thickness_um), 0.159, places=9)
        self.assertAlmostEqual(float(fig6.wavelength_um), 1.55, places=9)
        fig7 = resolve_ml_system_preset("LiNbO3 / Quartz (Fig 7, 1550 nm)")
        self.assertEqual([lyr.shg_active for lyr in fig7.layers], [False, True, True, False])
        self.assertEqual([float(lyr.thickness_um) for lyr in fig7.layers[1:3]], [50.0, 35.0])
        self.assertAlmostEqual(float(fig7.polarimetry.theta_deg), 0.5, places=9)
        default = resolve_ml_system_preset(None)
        self.assertEqual(default.layers[1].material.structure.point_group, "3m")  # LiNbO3
        with self.assertRaises(ValueError):
            resolve_ml_system_preset("bogus")

    def test_fresnel_functionality_is_continuous_and_energy_conserving(self):
        # the continuity lens + energy conservation on the new Fresnel output: both transmission
        # channels simultaneously populated (the tp/ts slot-swap bug is fixed at the source),
        # all four curves continuous, and R+T = 1 for the lossless film.
        res = compute_ml_gui_result("Fresnel Coefficients", theta_min_deg=0.0, theta_max_deg=80.0,
                                    theta_step_deg=2.0)
        rp = np.abs(np.asarray(res.numeric["rp"], dtype=float))
        rs = np.abs(np.asarray(res.numeric["rs"], dtype=float))
        tp = np.abs(np.asarray(res.numeric["tp"], dtype=float))
        ts = np.abs(np.asarray(res.numeric["ts"], dtype=float))
        self.assertEqual(int(np.sum(tp == 0.0)) + int(np.sum(ts == 0.0)), 0, "no tp/ts slot-swap holes")
        for name, y in (("rp", rp), ("rs", rs), ("tp", tp), ("ts", ts)):
            self.assertEqual(count_discontinuities(y), 0, f"{name} must be continuous")
        self.assertAlmostEqual(float(tp[0]), float(ts[0]), places=9, msg="normal incidence is degenerate")
        self.assertTrue(np.allclose(rp + tp, 1.0, atol=1e-6) and np.allclose(rs + ts, 1.0, atol=1e-6),
                        "lossless film must conserve energy: R + T = 1")

    def test_fresnel_figure_builds(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from shaarp.shaarp_gui import build_fresnel_figure

        res = compute_ml_gui_result("Fresnel Coefficients", theta_min_deg=0.0, theta_max_deg=60.0,
                                    theta_step_deg=5.0)
        fig = build_fresnel_figure(res)
        self.assertEqual(len(fig.axes[0].lines), 4)  # rp, rs, tp, ts
        plt.close(fig)


class PointGroupConstraintTests(unittest.TestCase):
    """Phase 2: the GUI's point-group control auto-constrains the SHG d-tensor (faithful behavior)."""

    def test_twenty_three_shg_point_groups(self):
        # the original's "Noncentrosymmetric ->" popup = 20 crystallographic classes + the
        # three SHG-active Curie groups (SHAARP.ml.nb:5191).
        self.assertEqual(len(SHG_POINT_GROUPS), 23)
        for pg in ("-43m", "3m", "mm2", "32", "1", "∞", "∞m", "∞2"):
            self.assertIn(pg, SHG_POINT_GROUPS)
        for pg in ("m3m", "6/mmm", "432", "-1"):
            self.assertNotIn(pg, SHG_POINT_GROUPS)

    def test_free_components_match_known_classes(self):
        self.assertEqual([(r, c) for r, c, _ in point_group_free_components("-43m")], [(0, 3)])
        self.assertEqual(
            {n for _r, _c, n in point_group_free_components("3m")}, {"d15", "d22", "d31", "d33"}
        )
        self.assertEqual(
            {n for _r, _c, n in point_group_free_components("mm2")}, {"d15", "d24", "d31", "d32", "d33"}
        )

    def test_constrained_tensor_obeys_3m_relations(self):
        # enter only the independent components; the relations must follow automatically
        d = build_constrained_d_voigt("3m", {(0, 4): 4.6, (1, 1): 2.2, (2, 0): 4.6, (2, 2): 27.0})
        self.assertTrue(np.isclose(d[0, 5], -2.2))   # d16 = -d22
        self.assertTrue(np.isclose(d[1, 0], -2.2))   # d21 = -d22
        self.assertTrue(np.isclose(d[1, 3], 4.6))    # d24 = d15
        self.assertTrue(np.isclose(d[2, 1], 4.6))    # d32 = d31
        self.assertTrue(np.isclose(d[2, 2], 27.0))   # d33

    def test_constrained_tensor_obeys_minus43m_relation(self):
        d = build_constrained_d_voigt("-43m", {(0, 3): 0.4})
        self.assertTrue(np.isclose(d[0, 3], 0.4) and np.isclose(d[1, 4], 0.4) and np.isclose(d[2, 5], 0.4))
        # everything else zero
        self.assertTrue(np.isclose(np.abs(d).sum(), 1.2))


class OutputPlotTests(unittest.TestCase):
    """Phase 3a: the GUI renders the iconic outputs -- .si polar SHG polarimetry, .ml Maker fringes."""

    def test_si_polarimetry_curve_is_real_for_several_point_groups(self):
        for pg in ("-43m", "3m", "mm2", "4mm"):
            c = si_polarimetry_curve(pg, theta_deg=45.0)
            self.assertEqual(c["intensity_p"].shape, (181,))
            self.assertTrue(np.all(np.isfinite(c["intensity_p"])) and np.all(np.isfinite(c["intensity_s"])))
            # a genuine polarimetry curve varies with phi
            self.assertGreater(float(np.ptp(c["intensity_p"]) + np.ptp(c["intensity_s"])), 0.0)

    def test_si_polar_figure_builds(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = build_si_polarimetry_figure("3m", theta_deg=45.0, n_omega=2.2, n_2omega=2.3,
                                          n_omega_e=2.26, n_2omega_e=2.36)
        # the original SHAARP.si SHG-Simulation output = FOUR analysis tiles: two reflected polar
        # plots (I_p, I_s) + the effective-index line plot + the incident-ellipticity locus.
        self.assertEqual(len(fig.axes), 4)
        polar = [ax for ax in fig.axes if ax.name == "polar"]
        cart = [ax for ax in fig.axes if ax.name != "polar"]
        self.assertEqual(len(polar), 2)  # I_p + I_s reflected
        self.assertEqual(len(cart), 2)   # effective index + ellipticity
        self.assertTrue(all(len(ax.lines) == 1 for ax in polar))      # one trace per polar
        self.assertEqual(len(cart[0].lines), 2)  # effective index: n^1 + n^2
        plt.close(fig)
        # fixed-analyzer psi now renders in the SAME 4-tile layout as Rotating:
        # two reflected polar plots (analyzer at psi and psi+90) + effective index + ellipticity.
        fig2 = build_si_polarimetry_figure("3m", theta_deg=45.0, analyzer_deg=30.0)
        self.assertEqual(len(fig2.axes), 4)
        self.assertEqual(len([ax for ax in fig2.axes if ax.name == "polar"]), 2)
        plt.close(fig2)

    def test_ml_polarimetry_four_polar(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from shaarp.layer_stack import build_system_from_stack, default_stack
        from shaarp.shaarp_gui import build_ml_polarimetry_figure, ml_polarimetry_curve

        sysm = build_system_from_stack(default_stack(), wavelength_um=1.064)
        c = ml_polarimetry_curve(sysm, n_phi=37, theta_deg=20.0)  # the four .ml SHG-sim channels
        for k in ("intensity_p", "intensity_s", "intensity_p_trans", "intensity_s_trans"):
            a = np.asarray(c[k], dtype=float)
            self.assertGreater(float(a.max()), 0.0, f"{k} must be nonzero")
            self.assertGreater(float(a.max() - a.min()), 0.0, f"{k} must vary with phi")
        fig = build_ml_polarimetry_figure(c, point_group="3m", assumption_label="Full Multiple Reflections (FMR)")
        self.assertEqual(len(fig.axes), 4)
        self.assertTrue(all(ax.name == "polar" for ax in fig.axes))
        plt.close(fig)

    def test_ml_polar_respects_assumption(self):
        """The ML SHG-Simulation 4-polar must actually change with the FMR / JK / HH assumption
        (not just relabel) -- regression for the bug where it always used the FMR path."""
        import numpy as np

        from shaarp.layer_stack import build_system_from_stack, default_stack
        from shaarp.shaarp_gui import ml_polarimetry_curve

        sysm = build_system_from_stack(default_stack(), wavelength_um=1.064)
        fmr = ml_polarimetry_curve(sysm, n_phi=19, theta_deg=20.0, mrassumption=0)
        jk = ml_polarimetry_curve(sysm, n_phi=19, theta_deg=20.0, mrassumption=1)
        hh = ml_polarimetry_curve(sysm, n_phi=19, theta_deg=20.0, mrassumption=2)
        self.assertGreater(float(np.max(np.abs(np.asarray(fmr["intensity_p"]) - np.asarray(jk["intensity_p"])))),
                           1e-6, "4-polar must differ between FMR and JK")
        self.assertGreater(float(np.max(np.abs(np.asarray(fmr["intensity_p"]) - np.asarray(hh["intensity_p"])))),
                           1e-6, "4-polar must differ between FMR and HH")

    def test_si_effective_index_curve_is_snell(self):
        import numpy as np

        from shaarp.shaarp_gui import si_effective_index_curve

        c = si_effective_index_curve("3m", n_omega=2.20, n_omega_e=2.30, n_theta=46)
        self.assertLess(float(np.ptp(c["n_ordinary"])), 1e-6)        # ordinary ~ flat vs angle
        self.assertGreater(float(np.ptp(c["n_extraordinary"])), 1e-3)  # extraordinary varies (Snell)

    def test_fix_polarizer_sweeps_analyzer(self):
        """Fix Polarizer: hold phi, sweep the analyzer psi -> SI single I(psi) polar; ML reflected +
        transmitted I(psi) (the original's Rotate-Analyzer-at-fixed-phi mode)."""
        import numpy as np

        from shaarp.layer_stack import build_system_from_stack, default_stack
        from shaarp.shaarp_gui import ml_polarimetry_curve, si_polarimetry_curve

        si = si_polarimetry_curve("3m", theta_deg=45.0, n_omega=2.2, n_2omega=2.3, fixed_phi_deg=30.0)
        self.assertIn("intensity_analyzed", si)
        self.assertEqual(si["fixed_phi_deg"], 30.0)
        self.assertGreater(float(np.ptp(si["intensity_analyzed"])), 0.0)
        ml = ml_polarimetry_curve(build_system_from_stack(default_stack(), wavelength_um=1.064),
                                  n_phi=25, theta_deg=20.0, fixed_phi_deg=45.0)
        for k in ("intensity_reflected", "intensity_transmitted"):
            self.assertGreater(float(np.ptp(ml[k])), 0.0, f"{k} must vary with psi")

    def test_analytical_expression_items_structured(self):
        """The analytical output is structured into ordered, labeled (label, text) blocks (the
        original's separate copyable I(phi,psi) expressions); numeric results raise ValueError."""
        from shaarp.shaarp_gui import (
            analytical_expression_items,
            analytical_expression_text,
            compute_ml_gui_result,
            compute_si_gui_result,
        )

        res = compute_si_gui_result("Full Analytical", point_group="-43m", theta_deg=45.0)
        items = analytical_expression_items(res)
        self.assertTrue(any("reflected" in label or "analyzed" in label for label, _t in items))
        self.assertIn("d14", "\n".join(t for _l, t in items))  # the symbolic d coefficient survives
        self.assertIn("closed form", analytical_expression_text(res))  # labeled blocks
        num = compute_ml_gui_result("Maker Fringes", theta_min_deg=0, theta_max_deg=10, theta_step_deg=5)
        with self.assertRaises(ValueError):
            analytical_expression_items(num)

    def test_si_notebook_case_materials(self):
        """The extra SHAARP.si case-study materials (TaAs, GaAs@1064 complex, LBO, KBBF, LiOsO3) load
        with their exact notebook constants and drive the SI polarimetry figure without error."""
        import shaarp.casestudy_materials as cm
        from shaarp.casestudy_materials import CASE_STUDY_ORDER, build_casestudy_material
        from shaarp.shaarp_gui import build_si_polarimetry_figure, si_figure_kwargs_from_material

        for name in ("TaAs (112)", "GaAs (111) (1064 nm)", "LiB3O5 (LBO)", "KBBF", "LiOsO3"):
            self.assertIn(name, CASE_STUDY_ORDER)
            m = build_casestudy_material(name)
            self.assertEqual(m.d_voigt_pm_v.shape, (3, 6))
        # exact constants preserved (no hallucination): TaAs giant d33, GaAs@1064 complex d14
        jt = cm._data()["materials"]["TaAs (112)"]
        self.assertAlmostEqual(jt["d_re"][2][2], 746.8152, places=3)
        jg = cm._data()["materials"]["GaAs (111) (1064 nm)"]
        self.assertAlmostEqual(jg["d_re"][0][3], 165.0, places=3)
        self.assertAlmostEqual(jg["d_im"][0][3], -475.0, places=3)
        # SI figure builds for a complex (TaAs) and a real (LBO) case-study material
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name in ("TaAs (112)", "LiB3O5 (LBO)"):
            kw = si_figure_kwargs_from_material(build_casestudy_material(name))
            pg = kw.pop("point_group")
            fig = build_si_polarimetry_figure(pg, theta_deg=45.0, **kw)
            self.assertGreaterEqual(len(fig.axes), 1)
            plt.close(fig)

    def test_full_complex_tensor_entry(self):
        """Full complex 3x3 eps + 3x6 d entry flows into the SI material and the ML film (the
        original's full-matrix dielectric/SHG entry; cores are complex-capable)."""
        import numpy as np

        from shaarp.shaarp_gui import build_custom_ml_system, build_custom_si_material, ml_polarimetry_curve

        epsw = [[5 + 1j, 0, 0], [0, 5 + 1j, 0], [0, 0, 6 + 2j]]
        eps2 = [[5.5, 0, 0], [0, 5.5, 0], [0, 0, 6.5]]
        dfull = [[0, 0, 0, 0, 7, 0], [0, 0, 0, 7, 0, 0], [3, 3, 9, 0, 0, 0]]
        m = build_custom_si_material("-43m", eps_omega_full=epsw, eps_2omega_full=eps2, d_full=dfull)
        self.assertTrue(np.allclose(m.epsilon_omega, np.array(epsw)))   # z-cut identity -> as entered
        self.assertTrue(np.allclose(m.d_voigt_pm_v, np.array(dfull)))
        sysm = build_custom_ml_system("-43m", film_eps_omega_full=epsw, film_eps_2omega_full=eps2,
                                      film_d_full=dfull, thickness_um=1.0, wavelength_um=1.0)
        self.assertTrue(np.allclose(sysm.layers[1].material.d_voigt_pm_v, np.array(dfull)))
        c = ml_polarimetry_curve(sysm, n_phi=13, theta_deg=20.0)  # ML solver honors full complex tensors
        self.assertGreater(float(np.max(c["intensity_p"])), 0.0)

    def test_maker_figure_builds_from_result(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res = compute_ml_gui_result("Maker Fringes", theta_min_deg=0.0, theta_max_deg=20.0, theta_step_deg=5.0)
        fig = build_maker_figure(res)
        self.assertEqual(len(fig.axes[0].lines), 2)  # parallel + perpendicular
        plt.close(fig)

    def test_default_ml_system_is_a_real_crystal(self):
        # the .ml default is a physical LiNbO3 film (NOT the artificial point-group-1 demo material)
        sysm = default_ml_system()
        film = sysm.layers[1]
        self.assertEqual(film.material.structure.point_group, "3m")  # LiNbO3 z-cut
        self.assertGreater(float(film.thickness_um), 0.0)


class MakerSingularityMaskingTests(unittest.TestCase):
    """A Maker fringe is physically CONTINUOUS; the field solve returns a sentinel (exact ~0, or a
    huge spike) at isolated eigenmode-degeneracy angles. mask_maker_singularities flags ONLY those
    (a physical minimum is smoothly small, never exactly 0 amid finite neighbors)."""

    def test_real_maker_curve_is_natively_continuous(self):
        """The continuity LENS as a gate ('every plot should be continuous; if
        any plot is non-continuous, something is off'). The transmitted Maker per-channel split
        used to dive to exact-0 at scattered angles (eigenmode-ordering swap, reading transmitted
        slot [0]); the fix selects the DOMINANT transmitted wave -- verified vs live SHAARP.ml
        (continuous, matches to 3.4e-11; benchmarks/verify_linbo3_swap_vs_mathematica.py). This
        asserts the RAW solver output is continuous over a window that previously had multiple
        exact-0 dropouts -- a regression here means a real solver bug, NOT something to mask."""
        res = compute_ml_gui_result("Maker Fringes", theta_min_deg=0.0, theta_max_deg=12.0, theta_step_deg=0.25)
        i_par = np.asarray(res.numeric["parallel_intensity"], dtype=float)
        self.assertGreater(float(i_par.max()), 1.0, "strong LiNbO3 signal expected (not a symmetry null)")
        self.assertEqual(int(np.sum(i_par == 0.0)), 0, "no exact-0 sentinel may appear in the raw Maker curve")
        self.assertEqual(count_discontinuities(i_par), 0, "raw Maker curve must be natively continuous")

    def test_isolated_sentinel_zeros_are_masked_singles_and_pairs(self):
        y = np.array([0.5, 0.51, 0.0, 0.53, 0.0, 0.0, 0.56, 0.58], dtype=float)  # a single + a pair of 0s
        masked, n = mask_maker_singularities(y)
        self.assertEqual(n, 3)
        self.assertTrue(np.isnan(masked[2]) and np.isnan(masked[4]) and np.isnan(masked[5]))
        # the genuine signal points are untouched
        self.assertTrue(np.allclose(masked[[0, 1, 3, 6, 7]], y[[0, 1, 3, 6, 7]]))

    def test_isolated_spike_is_masked(self):
        y = np.array([1.0, 1.1, 1.2, 50.0, 1.3, 1.4], dtype=float)  # one huge isolated spike
        masked, n = mask_maker_singularities(y)
        self.assertEqual(n, 1)
        self.assertTrue(np.isnan(masked[3]))

    def test_physical_minimum_is_not_masked(self):
        # a smooth fringe dipping toward (but not exactly) zero must NOT be masked
        x = np.linspace(0, np.pi, 41)
        y = np.sin(x) ** 2  # smooth, touches ~0 at the ends only, smoothly
        masked, n = mask_maker_singularities(y)
        self.assertEqual(n, 0, "a smoothly-small physical minimum must not be flagged as singular")

    def test_physical_interior_zero_minimum_is_not_masked(self):
        # a Maker fringe genuinely oscillates DOWN TO ZERO at interior minima (small neighbors);
        # these are continuous and must NOT be flagged -- only isolated holes in a HIGH region are.
        x = np.linspace(0, 6 * np.pi, 600)
        y = np.abs(np.sin(x))  # smooth, reaches exactly 0 at interior minima, with small neighbors
        masked, n = mask_maker_singularities(y)
        self.assertEqual(n, 0, "physical interior zero-minima (smooth dips) must not be masked")

    def test_sentinel_hole_in_high_region_is_masked_but_low_zero_is_not(self):
        # high smooth curve ~10 with one isolated exact-0 hole (sentinel) -> masked;
        # plus a region genuinely dipping to 0 (small neighbors) -> NOT masked.
        y = np.array([10.0, 10.1, 0.0, 10.2, 10.3, 1.0, 0.2, 0.0, 0.2, 1.0, 10.0, 10.1], dtype=float)
        masked, n = mask_maker_singularities(y)
        self.assertTrue(np.isnan(masked[2]), "isolated 0 amid ~10 neighbors is a sentinel -> masked")
        self.assertFalse(np.isnan(masked[7]), "0 at the bottom of a smooth dip (small neighbors) -> kept")
        self.assertEqual(n, 1)

    def test_all_zero_curve_is_not_masked(self):
        y = np.zeros(10)
        _masked, n = mask_maker_singularities(y)
        self.assertEqual(n, 0, "a genuinely-zero curve has no singular holes")


class CustomMaterialEntryTests(unittest.TestCase):
    """P2b: custom material entry -- indices + the point group's FREE d components, with the
    symmetry relations applied automatically (the faithful constrained SHG-tensor entry)."""

    def test_custom_si_material_applies_symmetry_and_indices(self):
        from shaarp.shaarp_gui import build_custom_si_material

        m = build_custom_si_material(
            "3m", n_omega=2.232, n_2omega=2.325, n_omega_e=2.156, n_2omega_e=2.233,
            d_free={(0, 4): 4.6, (1, 1): 2.2, (2, 0): 4.6, (2, 2): 27.0},
        )
        d = np.real(m.d_voigt())
        self.assertAlmostEqual(d[0, 5], -2.2, places=9)   # d16 = -d22 (applied, not entered)
        self.assertAlmostEqual(d[1, 3], 4.6, places=9)    # d24 = d15
        self.assertAlmostEqual(d[2, 2], 27.0, places=9)
        self.assertAlmostEqual(float(np.real(m.eps_w()[2, 2])), 2.156**2, places=9)   # eps_z = ne^2
        self.assertAlmostEqual(float(np.real(m.eps_w()[0, 0])), 2.232**2, places=9)   # eps_x = no^2

    def test_custom_ml_system_builds_with_constrained_film(self):
        from shaarp.shaarp_gui import build_custom_ml_system

        s = build_custom_ml_system("mm2", film_n_omega=2.2, thickness_um=0.5, wavelength_um=0.8,
                                   d_free={(2, 2): 10.0})
        self.assertEqual(len(s.layers), 3)
        self.assertAlmostEqual(float(s.layers[1].thickness_um), 0.5, places=9)
        self.assertAlmostEqual(float(s.wavelength_um), 0.8, places=9)
        self.assertAlmostEqual(float(np.real(s.layers[1].material.d_voigt()[2, 2])), 10.0, places=9)
        self.assertFalse(s.layers[2].shg_active)

    def test_uniaxial_polarimetry_curve_differs_from_isotropic(self):
        iso = si_polarimetry_curve("3m", theta_deg=45.0, n_omega=2.232, n_2omega=2.325)
        uni = si_polarimetry_curve("3m", theta_deg=45.0, n_omega=2.232, n_2omega=2.325,
                                   n_omega_e=2.156, n_2omega_e=2.233)
        self.assertTrue(np.all(np.isfinite(uni["intensity_p"])))
        rel = float(np.max(np.abs(iso["intensity_p"] - uni["intensity_p"]))) / float(iso["intensity_p"].max())
        self.assertGreater(rel, 1e-3, "the extraordinary index must change the polarimetry")


class SchematicTests(unittest.TestCase):
    """P4 slice: the 2D optical-setup schematic (the originals' orientation aid)."""

    def test_schematic_builds_for_three_and_four_layer_stacks(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        from shaarp.shaarp_gui import build_schematic_figure

        for layers in (
            [("air", None), ("film", 10.0), ("substrate", None)],
            [("air in", None), ("Z-cut quartz", 121.2), ("Au coating", 0.0139), ("air out", None)],
        ):
            fig = build_schematic_figure(layers, theta_deg=25.0, wavelength_um=0.8)
            ax = fig.axes[0]
            bands = [p for p in ax.patches if isinstance(p, Rectangle)]
            self.assertEqual(len(bands), len(layers), "one band per medium")
            texts = " ".join(t.get_text() for t in ax.texts)
            for name, _t in layers:
                self.assertIn(name, texts, f"layer {name!r} must be labeled")
            self.assertIn("121.2", texts) if len(layers) == 4 else None
            plt.close(fig)

    def test_schematic_requires_two_media(self):
        from shaarp.shaarp_gui import build_schematic_figure

        with self.assertRaises(ValueError):
            build_schematic_figure([("air", None)], theta_deg=20.0)


class PresetStoreAndExportTests(unittest.TestCase):
    """P4b: the originals' Preset 1-4 save/recall panel + the Copy-button data-export analog."""

    def test_preset_store_roundtrip_and_clear(self):
        from shaarp.shaarp_gui import PresetStore

        store = PresetStore()
        self.assertEqual(store.n_slots, 4)
        snap = {"point_group": "3m", "n_w": 2.232, "d_free": {"2,2": 27.0}}
        store.save(1, snap, label="LiNbO3")
        self.assertTrue(store.filled(1))
        self.assertFalse(store.filled(0))
        back = store.recall(1)
        self.assertEqual(back, snap)
        self.assertIsNot(back, snap, "recall must return a copy, not the stored object")
        self.assertEqual(store.label(1), "LiNbO3")
        self.assertIsNone(store.recall(0), "an empty slot recalls None")
        store.clear()
        self.assertFalse(store.filled(1))
        with self.assertRaises(ValueError):
            store.save(7, snap)

    def test_export_payload_is_json_serializable(self):
        import json

        from shaarp.shaarp_gui import export_result_payload

        res = compute_ml_gui_result("Maker Fringes", theta_min_deg=10.0, theta_max_deg=12.0, theta_step_deg=1.0)
        payload = export_result_payload(res)
        self.assertEqual(payload["kind"], "maker_fringes")
        self.assertIn("parallel_intensity", payload["numeric"])
        text = json.dumps(payload)  # must not raise (complex arrays exported as {re, im})
        self.assertGreater(len(text), 100)
        amp = payload["numeric"]["parallel_amplitude"]
        self.assertIn("re", amp)
        self.assertIn("im", amp)

    def test_panel_has_preset_and_export_buttons(self):
        try:
            import ipywidgets as widgets
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.shaarp_gui import make_shaarp_gui

        def _all(w):
            yield w
            for ch in getattr(w, "children", []) or []:
                yield from _all(ch)

        gui = make_shaarp_gui()
        for tab in gui.children:
            labels = {w.description for w in _all(tab) if isinstance(w, widgets.Button)}
            for needed in ("Save", "Recall", "Clear all", "Export data"):
                self.assertIn(needed, labels, f"{needed!r} button missing from a tab")


class MergedPanelNavigationTests(unittest.TestCase):
    def test_panel_has_two_navigable_si_ml_tabs(self):
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.shaarp_gui import make_shaarp_gui

        gui = make_shaarp_gui()
        self.assertEqual(len(gui.children), 2, "merged GUI must have exactly the .si and .ml tabs")
        self.assertIn("SHAARP.si", gui.get_title(0))
        self.assertIn("SHAARP.ml", gui.get_title(1))

    def test_point_group_dropdown_offers_the_shg_groups(self):
        try:
            import ipywidgets as widgets
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.shaarp_gui import make_shaarp_gui

        gui = make_shaarp_gui()
        stack = [gui]
        found = None
        while stack:
            w = stack.pop()
            if isinstance(w, widgets.Dropdown) and w.description == "Point group":
                found = w
                break
            stack.extend(list(getattr(w, "children", []) or []))
        self.assertIsNotNone(found, "point-group dropdown not found")
        self.assertEqual(tuple(found.options), SHG_POINT_GROUPS)

    def test_d_free_fields_rebuild_on_point_group_change(self):
        try:
            import ipywidgets as widgets
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.shaarp_gui import make_shaarp_gui

        def _all(w):
            yield w
            for ch in getattr(w, "children", []) or []:
                yield from _all(ch)

        gui = make_shaarp_gui()
        si_tab = gui.children[0]
        pg = next(w for w in _all(si_tab) if isinstance(w, widgets.Dropdown) and w.description == "Point group")
        pg.value = "-43m"
        labels_43m = {w.description for w in _all(si_tab)
                      if isinstance(w, widgets.FloatText) and w.description.startswith("d")}
        self.assertEqual(labels_43m, {"d14"}, "-43m exposes exactly its one free component")
        pg.value = "3m"
        labels_3m = {w.description for w in _all(si_tab)
                     if isinstance(w, widgets.FloatText) and w.description.startswith("d")}
        self.assertEqual(labels_3m, {"d15", "d22", "d31", "d33"}, "3m exposes its four free components")

    def test_ml_system_dropdown_offers_custom_film(self):
        try:
            import ipywidgets as widgets
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.shaarp_gui import make_shaarp_gui

        def _all(w):
            yield w
            for ch in getattr(w, "children", []) or []:
                yield from _all(ch)

        gui = make_shaarp_gui()
        ml_tab = gui.children[1]
        sysdd = next(w for w in _all(ml_tab) if isinstance(w, widgets.Dropdown) and w.description == "System")
        self.assertIn("Custom film (use fields)", tuple(sysdd.options))


class AnalyticalExpressionTests(unittest.TestCase):
    """P4b: the originals' copyable analytical output -- the closed-form polarimetry expression
    text extracted from analytical results (SI full + ML partial), shown in a Textarea and
    exported as .txt alongside the JSON data export."""

    def test_si_full_analytical_expression_text(self):
        from shaarp.shaarp_gui import analytical_expression_text

        res = compute_si_gui_result("Full Analytical", point_group="-43m", theta_deg=45.0)
        text = analytical_expression_text(res)
        # structured/labeled blocks (faithful per-expression output) + the symbolic content
        for need in ("reflected p", "reflected s", "closed form",
                     "phi", "d14", "si_full_analytical_polarimetry"):
            self.assertIn(need, text, f"SI closed-form text must contain {need!r}")
        self.assertGreater(len(text), 200, "a real closed form, not a stub")

    def test_ml_partial_analytical_expression_text(self):
        from shaarp.shaarp_gui import analytical_expression_text

        res = compute_ml_gui_result("Partial Analytical", point_group="3m", theta_deg=20.0)
        text = analytical_expression_text(res)
        for need in ("reflected p", "reflected s", "phi", "d15", "h"):
            self.assertIn(need, text, f"ML closed-form text must contain {need!r}")

    def test_numeric_result_raises(self):
        from shaarp.shaarp_gui import analytical_expression_text

        res = compute_si_gui_result("SHG Simulation", theta_deg=30.0)
        with self.assertRaises(ValueError):
            analytical_expression_text(res)

    def test_panel_has_expression_area(self):
        try:
            import ipywidgets as widgets
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.shaarp_gui import make_shaarp_gui

        def _all(w):
            yield w
            for ch in getattr(w, "children", []) or []:
                yield from _all(ch)

        gui = make_shaarp_gui()
        for tab in gui.children:
            area = next(w for w in _all(tab)
                        if isinstance(w, widgets.Textarea) and w.description == "Closed form")
            self.assertEqual(area.layout.display, "none", "hidden until an analytical run fills it")


class Schematic3DTests(unittest.TestCase):
    """P4b: the 3D sample-stack schematic (translucent slabs + incidence-plane rays)."""

    def test_builds_for_si_and_ml_layer_lists(self):
        import matplotlib

        matplotlib.use("Agg")
        from shaarp.shaarp_gui import build_schematic_3d_figure

        for layer_list in (
            [("air", None), ("crystal (-43m)", None)],
            [("air in", None), ("LiNbO3 film", 10.0), ("substrate", None)],
        ):
            fig = build_schematic_3d_figure(layer_list, theta_deg=30.0)
            try:
                ax = fig.axes[0]
                self.assertEqual(ax.name, "3d", "must be a 3D axes")
                self.assertEqual(len(ax.collections), len(layer_list), "one slab per layer")
                self.assertGreaterEqual(len(ax.lines), 3, "incident + reflected + normal rays")
            finally:
                import matplotlib.pyplot as plt

                plt.close(fig)


class OrientationEntryTests(unittest.TestCase):
    """P4b crystal-orientation entry: the faithful SHAARP hklConvert Miller mode (lattice-aware
    surface hkl + in-plane uvw), validated against the setup.nb quartz preset axes and the
    package's cubic Miller reference."""

    def test_zcut_identity_default(self):
        from shaarp.config import CrystalStructure
        from shaarp.shaarp_gui import build_orientation

        s = CrystalStructure(point_group="3m")
        R = build_orientation(s).rotation_matrix()
        self.assertLess(float(np.max(np.abs(R - np.eye(3)))), 1e-15)

    def test_quartz_001_120_matches_setupnb_identity(self):
        # the setup.nb Quartz$zCut preset stores plane (0,0,1) + vertical (1,2,0) with stored
        # axes == identity for the trigonal lattice (a=b=4.913, c=5.405, gamma=120).
        from shaarp.config import CrystalStructure
        from shaarp.shaarp_gui import build_orientation

        quartz = CrystalStructure(point_group="32", a=4.91304, b=4.91304, c=5.40463,
                                  alpha_deg=90.0, beta_deg=90.0, gamma_deg=120.0)
        o = build_orientation(quartz, "Miller (hkl + in-plane uvw)",
                              surface_hkl=(0, 0, 1), in_plane_uvw=(1, 2, 0))
        R = o.rotation_matrix()
        self.assertLess(float(np.max(np.abs(R - np.eye(3)))), 1e-12,
                        "quartz (001)/[120] must reproduce the setup.nb identity orientation")

    def test_cubic_111_maps_normal_to_lab_z(self):
        # the ASCII '-43m' label must canonicalize (alias fix) and the (111) plane normal must
        # land on the lab surface normal; rank-1 lab vector = R.T @ crystal vector.
        from shaarp.config import CrystalStructure
        from shaarp.shaarp_gui import build_orientation

        gaas = CrystalStructure(point_group="-43m", a=5.65, b=5.65, c=5.65)
        R = build_orientation(gaas, "Miller (hkl + in-plane uvw)",
                              surface_hkl=(1, 1, 1), in_plane_uvw=(1, -1, 0)).rotation_matrix()
        self.assertLess(float(np.max(np.abs(R @ R.T - np.eye(3)))), 1e-12, "must be a rotation")
        v = R.T @ (np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))
        err = min(float(np.max(np.abs(v - np.array([0.0, 0.0, 1.0])))),
                  float(np.max(np.abs(v + np.array([0.0, 0.0, 1.0])))))
        self.assertLess(err, 1e-12, "(111) normal must map to the lab normal")

    def test_invalid_in_plane_uvw_raises(self):
        from shaarp.config import CrystalStructure
        from shaarp.shaarp_gui import build_orientation

        gaas = CrystalStructure(point_group="-43m", a=5.65, b=5.65, c=5.65)
        with self.assertRaises(ValueError):
            build_orientation(gaas, "Miller (hkl + in-plane uvw)",
                              surface_hkl=(1, 1, 1), in_plane_uvw=(0, 0, 1))  # not in the plane

    def test_unknown_orientation_mode_raises(self):
        from shaarp.config import CrystalStructure
        from shaarp.shaarp_gui import build_orientation

        with self.assertRaises(ValueError):
            build_orientation(CrystalStructure(point_group="1"), "nonsense mode")

    def test_orientation_figure_readable_when_axes_overlap(self):
        """The crystal-axes (Zi) vs lab-axes (Li) figure must stay readable
        even when the two triads COINCIDE (z-cut/identity: Zi == Li). It carries a legend
        distinguishing solid-black lab from dashed-crimson crystal, and the Zi/Li labels are pushed
        to opposite sides of the shared axis so they never print on top of each other."""
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        from shaarp.config import CrystalOrientation
        from shaarp.shaarp_gui import build_orientation_axes_figure

        fig = build_orientation_axes_figure(CrystalOrientation(np.eye(3)))
        ax = fig.axes[0]
        # the legend moved to the FIGURE level (a bottom two-column row -- the compact
        # 340 px canvas clipped the old outside-right axes legend); accept either home.
        leg = ax.get_legend() or (fig.legends[0] if fig.legends else None)
        self.assertIsNotNone(leg, "orientation figure must carry a lab-vs-crystal legend")
        labels = {t.get_text() for t in leg.get_texts()}
        self.assertEqual(labels, {"lab $L_i$", "crystal $Z_i$"},
                         "legend must distinguish lab Li (solid) from crystal Zi (dashed)")
        # both the Zi and Li axis-name labels are present (six directional labels) so the identity
        # case still shows all six, offset apart rather than stacked
        texts = {t.get_text() for t in ax.texts}
        for need in ("$L_1$", "$L_2$", "$L_3$", "$Z_1$", "$Z_2$", "$Z_3$"):
            self.assertIn(need, texts)

    def test_crystal_physics_directions_mode(self):
        # the original's third orientation mode: direct Z1/Z2/Z3-in-lab entry
        from shaarp.config import CrystalStructure
        from shaarp.shaarp_gui import build_orientation

        s = CrystalStructure(point_group="3m")
        ident = build_orientation(s, "Crystal Physics Directions (Z1,Z2,Z3)", z_axes=np.eye(3))
        self.assertTrue(np.allclose(ident.rotation_matrix(), np.eye(3)))
        rot = build_orientation(s, "Crystal Physics Directions (Z1,Z2,Z3)",
                                z_axes=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        R = rot.rotation_matrix()
        self.assertLess(float(np.max(np.abs(R @ R.T - np.eye(3)))), 1e-9)
        # a non-orthonormal triad is rejected (not silently turned into a non-rotation)
        with self.assertRaises(ValueError):
            build_orientation(s, "Crystal Physics Directions (Z1,Z2,Z3)",
                              z_axes=[[1, 0, 0], [1, 0, 0], [0, 0, 1]])

    def test_custom_material_carries_miller_orientation_and_lattice(self):
        from shaarp.shaarp_gui import build_custom_si_material

        m = build_custom_si_material(
            "-43m", d_free={(0, 3): 0.4}, lattice=(5.65, 5.65, 5.65, 90.0, 90.0, 90.0),
            orientation_mode="Miller (hkl + in-plane uvw)",
            surface_hkl=(1, 1, 1), in_plane_uvw=(1, -1, 0),
        )
        self.assertEqual(m.structure.a, 5.65)
        R = m.orientation.rotation_matrix()
        self.assertGreater(float(np.max(np.abs(R - np.eye(3)))), 0.1,
                           "Miller (111) orientation must be non-identity")

    def test_custom_ml_system_threads_orientation_to_film(self):
        from shaarp.shaarp_gui import build_custom_ml_system

        sysm = build_custom_ml_system(
            "-43m", d_free={(0, 3): 0.4}, lattice=(5.65, 5.65, 5.65, 90.0, 90.0, 90.0),
            orientation_mode="Miller (hkl + in-plane uvw)",
            surface_hkl=(1, 1, 1), in_plane_uvw=(1, -1, 0),
        )
        R = sysm.layers[1].material.orientation.rotation_matrix()
        self.assertGreater(float(np.max(np.abs(R - np.eye(3)))), 0.1)

    def test_miller_matches_validated_cubic_reference_up_to_azimuth(self):
        # cross-validation against the package's previously-validated GaAs(111) orientation
        # (from_cubic_miller_surface, used by the worked polarimetry example): the SHAARP
        # hklConvert convention puts in_plane_uvw -> lab L2 while the cubic reference puts its
        # hint -> lab L1, so the two must agree up to EXACTLY a +90 deg azimuth about the normal.
        from shaarp.config import CrystalOrientation, CrystalStructure
        from shaarp.shaarp_gui import build_orientation

        gaas = CrystalStructure(point_group="-43m", a=5.65, b=5.65, c=5.65)
        Rm = build_orientation(gaas, "Miller (hkl + in-plane uvw)",
                               surface_hkl=(1, 1, 1), in_plane_uvw=(1, -1, 0)).rotation_matrix()
        Rc = CrystalOrientation.from_cubic_miller_surface(
            np.array([1.0, 1.0, 1.0]), in_plane_hint=np.array([1.0, -1.0, 0.0])).rotation_matrix()
        Az = Rm.T @ Rc  # the lab-frame map between the two conventions
        expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertLess(float(np.max(np.abs(Az - expected))), 1e-12,
                        "Miller mode must equal the validated cubic reference up to +90 deg azimuth")

    def test_ascii_barred_label_canonicalizes_for_physics_axes(self):
        # regression for the alias fix: shaarp_physics_axes must accept the package-wide ASCII
        # barred spellings, not only the Mathematica OverscriptBox forms.
        from shaarp.config import CrystalStructure

        for pg in ("-43m", "-42m", "-6m2", "-4", "-6"):
            axes = CrystalStructure(point_group=pg).shaarp_physics_axes()
            self.assertEqual(np.asarray(axes).shape, (3, 3), f"physics axes must build for {pg!r}")

    def test_gui_orientation_widgets_present(self):
        try:
            import ipywidgets as widgets
        except ImportError:
            self.skipTest("ipywidgets not installed")
        from shaarp.shaarp_gui import make_shaarp_gui

        def _all(w):
            yield w
            for ch in getattr(w, "children", []) or []:
                yield from _all(ch)

        gui = make_shaarp_gui()
        si_tab = gui.children[0]
        odd = next(w for w in _all(si_tab)
                   if isinstance(w, widgets.Dropdown) and w.description == "Orientation")
        self.assertIn("Miller (hkl + in-plane uvw)", tuple(odd.options))
        # Miller entry fields exist (lattice + hkl + uvw)
        descs = {w.description for w in _all(si_tab) if isinstance(w, (widgets.FloatText, widgets.IntText))}
        for need in ("a (A)", "gamma", "h", "k", "l", "u", "v", "w"):
            self.assertIn(need, descs)


class ValidationMetadataGracefulTests(unittest.TestCase):
    """A MISSING validation-metadata benchmark file must never crash an actual compute.

    Regression for the frozen-exe FileNotFoundError: the GUI requests
    include_validation_summary=True, which loads benchmarks/*.json for provenance metadata;
    PyInstaller bundled shaarp/ data but not the top-level benchmarks/ tree, so the ML SHG
    Simulation Update crashed in the exe. The validation metadata is provenance only -- the
    physics compute + result.validation.status must still succeed if those files are absent."""

    def test_ml_si_compute_graceful_when_benchmarks_missing(self):
        import shaarp.api as api

        orig = api._load_project_json

        def _missing(rel):
            raise FileNotFoundError(f"simulated missing bundled file: {rel}")

        api._load_project_json = _missing
        try:
            r = compute_ml_gui_result("SHG Simulation", system_preset="Quartz + Au (Fig 4, 800 nm)")
            self.assertTrue(str(r.validation.status))  # status still set (independent of the files)
            art = r.stages.get("mathematica_validation_artifacts", {})
            self.assertEqual(art.get("status"), "validation_metadata_unavailable_in_this_build")
            # the other compute modes must also survive a missing benchmark file
            compute_ml_gui_result("Maker Fringes", system_preset="Quartz + Au (Fig 4, 800 nm)",
                                  theta_min_deg=0.0, theta_max_deg=10.0, theta_step_deg=5.0)
            compute_ml_gui_result("Fresnel Coefficients", theta_min_deg=0.0, theta_max_deg=10.0,
                                  theta_step_deg=5.0)
            compute_si_gui_result("SHG Simulation", point_group="3m")
        finally:
            api._load_project_json = orig


if __name__ == "__main__":
    unittest.main()
