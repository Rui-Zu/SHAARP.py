"""R4 (partial closure): every TEXT OBJECT in the GUI's figures must be clean.

The S11 incident shipped a figure whose mathtext title had been corrupted by a collapsed
backslash escape (a vertical-tab control character swallowed the leading letter -> the title
rendered as "(arphi"), and NOTHING raised: outside math mode matplotlib silently draws around
control characters. Pixel-level golden images are brittle across matplotlib versions, but the
TEXT OBJECTS are not -- so this gate walks every Text artist in a representative set of the
GUI's figures and asserts the strings are structurally sound:

  * no control characters (the S11 signature: a collapsed escape like \\v or \\t),
  * no U+FFFD replacement characters (mojibake signature),
  * balanced ``$`` delimiters (an odd count means a broken mathtext span).

This catches the S11 class MECHANICALLY, at gate time, on every figure listed -- no PNG
inspection required. (Rendering each figure is still exercised elsewhere; this gate is about
the silent-garbling case rendering does not catch.)
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
from matplotlib.text import Text

from shaarp.casestudy_materials import (
    build_casestudy_material,
    build_casestudy_ml_system,
)
from shaarp.shaarp_gui import (
    build_orientation_axes_figure,
    build_schematic_figure,
    build_si_polarimetry_figure,
    compute_ml_gui_result,
    si_figure_kwargs_from_material,
)


def _text_defects(fig) -> list[str]:
    """All structurally-broken text strings in a figure (empty list = clean)."""
    bad = []
    for artist in fig.findobj(Text):
        s = artist.get_text()
        if not s:
            continue
        if any(ord(c) < 32 and c != "\n" for c in s):
            bad.append(f"control char in {s!r}")
        if "�" in s:
            bad.append(f"replacement char in {s!r}")
        if s.count("$") % 2 == 1:
            bad.append(f"unbalanced $ in {s!r}")
    return bad


class FigureTextIntegrity(unittest.TestCase):
    def assertCleanText(self, fig, label: str) -> None:
        defects = _text_defects(fig)
        self.assertFalse(defects, f"{label}: {defects[:4]}")

    def test_detector_catches_the_s11_class(self):
        # negative control: a collapsed-escape corruption OUTSIDE math mode renders silently,
        # so only this scan would catch it -- the detector must flag it.
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_title("I(\x0barphi)  vs  angle")  # \v collapsed out of \varphi
        try:
            self.assertTrue(_text_defects(fig), "detector missed a control-char corruption")
        finally:
            plt.close(fig)

    def test_si_polarimetry_figures(self):
        mat = build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=0.8)
        kw = si_figure_kwargs_from_material(mat)
        for label, extra in (("standard", {}),
                             ("fixed analyzer", {"analyzer_deg": 30.0}),
                             ("co-rotating", {"corotating_offset_deg": 15.0})):
            fig = build_si_polarimetry_figure(theta_deg=45.0, **kw, **extra)
            try:
                self.assertCleanText(fig, f"SI polarimetry ({label})")
            finally:
                import matplotlib.pyplot as plt
                plt.close(fig)

    def test_schematic_all_assumptions(self):
        for assumption, sub in (("Full Multiple Reflections (FMR)", "Forward + Backward waves"),
                                ("Jerphagnon & Kurtz Assumption (No MR)", None),
                                ("Herman & Hayden Assumption (MR only for 2ω Homo Waves)", None)):
            fig = build_schematic_figure(
                [("air", None), ("quartz film", 50.0), ("substrate", None)],
                theta_deg=30.0, assumption=assumption, fmr_submode=sub)
            try:
                self.assertCleanText(fig, f"schematic ({assumption[:14]})")
            finally:
                import matplotlib.pyplot as plt
                plt.close(fig)

    def test_orientation_axes_figure(self):
        mat = build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)
        fig = build_orientation_axes_figure(mat.orientation, title="GaAs (111)")
        try:
            self.assertCleanText(fig, "orientation axes")
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_ml_maker_and_fresnel_figures(self):
        from shaarp.shaarp_gui import build_maker_figure, build_fresnel_figure

        s = build_casestudy_ml_system("Quartz z-cut (800 nm)", thickness_um=50.0,
                                      wavelength_um=0.8)
        r = compute_ml_gui_result("Maker Fringes", system=s, theta_min_deg=0.0,
                                  theta_max_deg=30.0, theta_step_deg=1.0,
                                  assumption="Jerphagnon & Kurtz Assumption (No MR)")
        fig = build_maker_figure(r, assumption_label="JK")
        try:
            self.assertCleanText(fig, "Maker figure")
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
        rf = compute_ml_gui_result("Fresnel Coefficients", system=s, theta_step_deg=2.0)
        fig = build_fresnel_figure(rf)
        try:
            self.assertCleanText(fig, "Fresnel figure")
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
