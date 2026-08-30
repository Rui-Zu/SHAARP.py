"""REFEREE layer: cross-check every user-facing output surface against an independent computation.

WHY (extend your evaluation to cover those areas you failed to inspect"): the
earlier checks verified CONSISTENCY (labels present, panels sync, nothing crashes, nothing clipped)
but not CONTENT -- whether the labeled object is the right physical object, derived from the user's
actual inputs, delivering each mode's promise. the author's exe test exposed that gap (amplitudes
labeled as intensities; F24/F24ml generic default constants instead of the panel/stack values;
"Full" identical to "Partial"). This module encodes the missing review as permanent gates:

  A. EXPRESSION <-> PLOT: the analytical closed form, numerically evaluated with the material's own
     d values, must reproduce the SHAPE of the plotted polarimetry curves (a wrong point group,
     wrong epsilon, or missing orientation rotation breaks the match).
  B. INPUT PROVENANCE: the ML film closed form must depend on the user's theta and the actual film
     material (regression for l), and say so in its provenance header.
  C. LEGEND <-> DATA: the curve labeled R_p must BE the rp array (and Maker parallel, SI I_p tile)
     -- a silent legend/data swap is invisible to every layout/crash check.
  D. MODE CONTRACT: "Full Analytical" must actually be symbolic in theta/indices where promised,
     and honestly say when it falls back; "Partial" must say what it substituted.
"""

from __future__ import annotations

import re
import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
import sympy as sp

from shaarp.casestudy_materials import (
    build_casestudy_material,
    build_casestudy_ml_system,
)
from shaarp.shaarp_gui import (
    analytical_expression_text,
    build_fresnel_figure,
    build_maker_figure,
    compute_ml_gui_result,
    compute_si_gui_result,
    point_group_free_components,
    si_figure_kwargs_from_material,
    si_polarimetry_curve,
)


def _material_d_values(mat) -> dict[str, complex]:
    """Map symbolic component names (d11, d15, ...) to the material's numeric values."""
    d = np.asarray(mat.d_voigt_pm_v, dtype=complex)
    pg = mat.structure.point_group
    return {name: complex(d[r, c]) for (r, c, name) in point_group_free_components(pg)}


def _eval_closed_form(expr_str: str, d_map: dict[str, complex], phi: np.ndarray) -> np.ndarray:
    expr = sp.sympify(expr_str)
    subs = {sp.Symbol(k): v for k, v in d_map.items() if sp.Symbol(k) in expr.free_symbols}
    expr = expr.subs(subs)
    fn = sp.lambdify(sp.Symbol("phi", real=True), expr, modules="numpy")
    vals = np.asarray(fn(phi), dtype=complex)
    if vals.shape != phi.shape:  # constant expression
        vals = np.full(phi.shape, complex(vals))
    return vals


class ExpressionMatchesPlot(unittest.TestCase):
    """A. The Partial closed form (material-true constants) reproduces the plotted curve's SHAPE."""

    def _check(self, name: str, wavelength: float):
        mat = build_casestudy_material(name, wavelength_um=wavelength)
        res = compute_si_gui_result("Partial Analytical", theta_deg=45.0, material=mat)
        d_map = _material_d_values(mat)
        kw = si_figure_kwargs_from_material(mat)
        pg = kw.pop("point_group")
        curve = si_polarimetry_curve(pg, theta_deg=45.0, **kw)
        phi = np.deg2rad(np.asarray(curve["phi_deg"], float))
        for stage_key, curve_key in (("reflected_p_2omega", "intensity_p"),
                                     ("reflected_s_2omega", "intensity_s")):
            with self.subTest(material=name, channel=curve_key):
                amps = _eval_closed_form(str(res.stages[stage_key]), d_map, phi)
                i_expr = np.abs(amps) ** 2
                i_plot = np.asarray(curve[curve_key], float)
                mask = i_plot > 1e-6 * float(i_plot.max())
                self.assertTrue(mask.any(), f"{name}/{curve_key}: plotted curve is all ~zero")
                ratio = i_expr[mask] / i_plot[mask]
                spread = float(np.std(ratio) / max(abs(np.mean(ratio)), 1e-30))
                self.assertLess(spread, 1e-6,
                                f"{name}/{curve_key}: closed form does not match the plotted curve "
                                f"shape (ratio spread {spread:.2e}) -- wrong pg/eps/orientation?")

    def test_linbo3_uniaxial(self):
        self._check("LiNbO3 z-cut (1064 nm)", 1.064)

    def test_gaas111_rotated(self):
        self._check("GaAs (111) (800 nm)", 0.8)

    def test_quartz_32(self):
        self._check("Quartz z-cut (800 nm)", 0.8)

    def test_linbo3_xcut_oriented(self):
        # x-cut: the optic axis lies IN PLANE (lab eps_x != eps_y) -- the case the old z-cut
        # approximation silently got wrong for both the plot and the expression
        self._check("LiNbO3 x-cut (1550 nm)", 1.55)


class MlExpressionUsesTheStack(unittest.TestCase):
    """B. Regression for F24ml: the film closed form derives from the user's theta and stack."""

    def test_theta_and_film_change_the_expression(self):
        s1 = build_casestudy_ml_system("LiNbO3 z-cut (1550 nm)", thickness_um=3.0, wavelength_um=1.064)
        s2 = build_casestudy_ml_system("LiOsO3", thickness_um=3.0, wavelength_um=1.064)  # same pg 3m
        e_20 = str(compute_ml_gui_result("Partial Analytical", theta_deg=20.0,
                                         system=s1).stages["reflected_p_2omega"])
        e_45 = str(compute_ml_gui_result("Partial Analytical", theta_deg=45.0,
                                         system=s1).stages["reflected_p_2omega"])
        e_other = str(compute_ml_gui_result("Partial Analytical", theta_deg=20.0,
                                            system=s2).stages["reflected_p_2omega"])
        self.assertNotEqual(e_20, e_45, "theta does not influence the ML closed form (F24ml)")
        self.assertNotEqual(e_20, e_other, "the film material does not influence the ML closed form")

    def test_provenance_header(self):
        s = build_casestudy_ml_system("LiNbO3 z-cut (1550 nm)", thickness_um=3.0, wavelength_um=1.064)
        r = compute_ml_gui_result("Partial Analytical", theta_deg=20.0, system=s)
        text = analytical_expression_text(r)
        self.assertIn("(substituted)", text)
        self.assertIn("film =", text)


class LegendMatchesData(unittest.TestCase):
    """C. The curve carrying a label IS the array of that name (no silent legend/data swap)."""

    def _line_by_label(self, fig, needle: str):
        for ax in fig.axes:
            for line in ax.get_lines():
                if needle in (line.get_label() or ""):
                    return np.asarray(line.get_ydata(), float)
        return None

    def test_fresnel_legend_mapping(self):
        r = compute_ml_gui_result("Fresnel Coefficients", theta_min_deg=5.0, theta_max_deg=60.0,
                                  theta_step_deg=1.0)
        fig = build_fresnel_figure(r)
        for label_needle, key in (("R_p", "rp"), ("R_s", "rs"), ("T_p", "tp"), ("T_s", "ts")):
            with self.subTest(curve=key):
                y = self._line_by_label(fig, label_needle)
                self.assertIsNotNone(y, f"no plotted line labeled like {label_needle!r}")
                ref = np.asarray(r.numeric[key], float)
                self.assertEqual(len(y), len(ref))
                self.assertLess(float(np.max(np.abs(y - ref))), 1e-12,
                                f"line labeled {label_needle} does not plot numeric[{key!r}]")

    def test_maker_parallel_mapping(self):
        r = compute_ml_gui_result("Maker Fringes", theta_min_deg=5.0, theta_max_deg=40.0,
                                  theta_step_deg=0.5)
        fig = build_maker_figure(r, assumption_label="test")
        ref = np.asarray(r.numeric["parallel_intensity"], float)
        best = None
        for ax in fig.axes:
            for line in ax.get_lines():
                y = np.asarray(line.get_ydata(), float)
                if len(y) == len(ref):
                    finite = np.isfinite(y) & np.isfinite(ref)
                    err = float(np.max(np.abs(y[finite] - ref[finite]))) if finite.any() else np.inf
                    lab = (line.get_label() or "")
                    if err < 1e-9 and ("par" in lab.lower() or "∥" in lab or "\\parallel" in lab):
                        best = lab
        self.assertIsNotNone(best, "no Maker line labeled parallel plots parallel_intensity")


class ModeContract(unittest.TestCase):
    """D. Each analytical mode delivers (and declares) exactly what it promises."""

    def test_full_is_symbolic_for_uniaxial_identity(self):
        mat = build_casestudy_material("LiNbO3 z-cut (1064 nm)", wavelength_um=1.064)
        r = compute_si_gui_result("Full Analytical", theta_deg=45.0, material=mat)
        text = analytical_expression_text(r)
        self.assertIn("theta_i (symbolic)", text)
        # The amplitude is now written in NAMED intermediates (FA-1 Stage 2), so theta_i / n_omega
        # live in their definitions -- which travel WITH the expressions in this text. Checking the
        # whole text keeps the original intent (genuinely symbolic, nothing substituted) and the
        # float-literal check below makes it stricter than the old symbol-presence test.
        self.assertIn("theta_i", text)
        self.assertIn("n_omega", text)
        amplitude = str(r.stages["reflected_p_2omega"])
        self.assertEqual(re.findall(r"\d+\.\d{3,}", amplitude), [],
                         "numeric literals in a Full Analytical amplitude = a silent substitution")

    def test_partial_declares_substitution_and_is_numeric(self):
        mat = build_casestudy_material("LiNbO3 z-cut (1064 nm)", wavelength_um=1.064)
        r = compute_si_gui_result("Partial Analytical", theta_deg=45.0, material=mat)
        text = analytical_expression_text(r)
        self.assertIn("(substituted)", text)
        self.assertNotIn("theta_i", str(r.stages["reflected_p_2omega"]))

    def test_full_is_symbolic_even_for_a_rotated_material(self):
        # WAS test_full_falls_back_honestly_for_rotated_material: GaAs (111) used to fall back and
        # say "impractical". It no longer falls back -- rotating only rotates d (done symbolically)
        # and its lab eps stays diagonal, so the layered emitter gives the full closed form. The
        # honest-fallback CONTRACT is not dropped, it moves to the case that genuinely cannot be
        # done (see test_full_states_its_reason_when_no_closed_form_exists).
        mat = build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)
        r = compute_si_gui_result("Full Analytical", theta_deg=45.0, material=mat)
        text = analytical_expression_text(r)
        self.assertIn("theta_i (symbolic)", text)
        self.assertNotIn("impractical", text)
        self.assertEqual(re.findall(r"\d+\.\d{3,}", str(r.stages["reflected_p_2omega"])), [])

    def test_full_states_its_reason_when_no_closed_form_exists(self):
        # NO SILENT FALLBACK. TaAs (112) has a NON-DIAGONAL lab eps, where the principal-aligned
        # closed form does not apply (forcing it disagrees with the validated numeric solver by
        # ~6e+00 relative -- it would be wrong, not merely missing). The panel must say so and
        # point at the numeric path, rather than quietly showing a substituted expression.
        mat = build_casestudy_material("TaAs (112)", wavelength_um=0.8)
        r = compute_si_gui_result("Full Analytical", theta_deg=45.0, material=mat)
        text = analytical_expression_text(r)
        self.assertIn("no closed form", text)
        self.assertIn("OFF-DIAGONAL", text)
        self.assertIn("SHG Simulation", text)  # tells the user where to go instead

    def test_amplitude_vs_intensity_labels(self):
        mat = build_casestudy_material("LiNbO3 z-cut (1064 nm)", wavelength_um=1.064)
        r = compute_si_gui_result("Partial Analytical", theta_deg=45.0, material=mat)
        text = analytical_expression_text(r)
        self.assertIn("field amplitude", text)          # E blocks say what they are
        self.assertIn("analyzed SHG intensity", text)    # the |...|^2 block is the intensity
        self.assertNotIn("I_p^{2ω}(φ,ψ)  —  reflected", text)  # the old mislabel must not return


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
