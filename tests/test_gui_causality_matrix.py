"""GUI CAUSALITY MATRIX (post-mortem gate).

Why this file exists: the GUI matrix sweep is a CRASH/render sweep and the
input-sensitivity matrix exercises the PURE-COMPUTE functions, so the SEAM between them -- the Qt
signal wiring that turns a user edit into different rendered curves -- was never evaluated. A DEAD
CONTROL (the crystal-orientation group while a case study was active) crashed nothing, rendered a
self-consistent screen, and passed every gate; it was found by clicking. This module closes that
class: it drives each curated input control THE WAY A USER WOULD (activated / setValue / setText),
presses Update / Run, and asserts the RENDERED curve data actually changed. A control that changes
nothing fails by name.

Design notes:
- Snapshot = the concatenated line data of every matplotlib canvas on the tab (plot + Maker +
  Fresnel), so a change that only re-renders a different tile still counts, and a figure swapped
  to a different mode (different line count) counts trivially.
- Steps are SEQUENTIAL on one window (each asserts against the previous state), so the whole
  matrix costs one window build and ~a dozen Updates.
- Physics guard: each step perturbs an input that MUST matter in the state it runs in (e.g. the
  z-axes rows are exercised in "Crystal Physics Directions" mode with a y-tilt -- an Rz twist of a
  4mm crystal is a genuine physical invariance and would be a false failure).
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False

import numpy as np


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SiCausalityMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()
        tabs = cls.win.findChild(QtWidgets.QTabWidget)
        cls.si = tabs.widget(0)
        cls.canvas_cls = FigureCanvasQTAgg

    def _spin(self, tip_key):
        from shaarp.desktop_app import TOOLTIPS
        cands = [s for s in self.si.findChildren(QtWidgets.QDoubleSpinBox)
                 if s.toolTip() == TOOLTIPS[tip_key]]
        self.assertTrue(cands, f"no spinbox with tooltip {tip_key!r}")
        return cands[0]

    def _combo_with_item(self, text, exclude=()):
        c = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                 if c.findText(text) >= 0 and c not in exclude)
        return c

    def _update(self):
        btn = next(b for b in self.si.findChildren(QtWidgets.QPushButton)
                   if b.text() == "Update / Run")
        btn.click()
        self.app.processEvents()

    def _snapshot(self):
        out = []
        for cv in self.si.findChildren(self.canvas_cls):
            if cv.objectName().startswith("orient_canvas"):
                continue  # input-panel triad, not a result surface
            for ax in cv.figure.axes:
                for ln in ax.lines:
                    out.append(np.asarray(ln.get_ydata(), dtype=float))
        return out

    def _assert_changed(self, before, after, label):
        if len(before) != len(after) or any(a.shape != b.shape for a, b in zip(before, after)):
            return  # different figure structure = the control had an effect
        scale = max((float(np.nanmax(np.abs(a))) for a in before if a.size), default=0.0) + 1e-300
        diff = max((float(np.nanmax(np.abs(a - b))) for a, b in zip(before, after) if a.size),
                   default=0.0)
        self.assertGreater(diff / scale, 1e-6,
                           f"DEAD CONTROL: {label} changed nothing in the rendered curves "
                           "(F40 class -- the widget is not wired into the compute)")

    def test_si_input_controls_all_drive_the_result(self):
        case = self._combo_with_item("TaAs (112)")
        orient = self._combo_with_item("z-cut (identity)", exclude=(case,))

        # baseline: the TaAs (112) case study (exact tilted route)
        case.setCurrentText("TaAs (112)")
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        self.assertTrue(any(a.size for a in snap), "baseline produced no curves")

        # 1) THE F40 reproduction: user switches orientation mode to identity
        orient.setCurrentText("z-cut (identity)")
        orient.activated.emit(orient.currentIndex())
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "Orientation Mode (case -> z-cut identity)")
        snap = nxt

        # 2) z-axes rows: user enters a y-tilt in Crystal Physics Directions mode
        # (an x/z tilt genuinely changes 4mm physics; an Rz twist would NOT -- see header)
        orient.setCurrentText("Crystal Physics Directions (Z1,Z2,Z3)")
        orient.activated.emit(orient.currentIndex())
        self.app.processEvents()
        import math
        c20, s20 = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
        ry20 = [[c20, 0.0, s20], [0.0, 1.0, 0.0], [-s20, 0.0, c20]]
        z_edits = [s for s in self.si.findChildren(QtWidgets.QDoubleSpinBox)
                   if s.toolTip() and "Z" in s.toolTip() and s.decimals() >= 4] or None
        if z_edits is None or len(z_edits) < 9:
            from shaarp.desktop_app import TOOLTIPS
            z_edits = [s for s in self.si.findChildren(QtWidgets.QDoubleSpinBox)
                       if s.toolTip() == TOOLTIPS["z_axes"]]
        self.assertGreaterEqual(len(z_edits), 9, "could not locate the 9 z-axes cells")
        for k, widget in enumerate(z_edits[:9]):
            widget.setValue(float(ry20[k // 3][k % 3]))
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "z-axes rows (identity -> 20-deg y-tilt)")
        snap = nxt

        # 3) incident angle
        theta = self._spin("theta")
        theta.setValue(30.0)
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "incident angle theta 45 -> 30")
        snap = nxt

        # 4) incident ellipticity
        ell = self._spin("ellipticity")
        ell.setValue(30.0)
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "ellipticity 0 -> 30 deg")
        snap = nxt

        # 5) an SHG d-tensor cell (the case's d33 = 746.815 still fills the grid)
        d_cell = next(e for e in self.si.findChildren(QtWidgets.QLineEdit)
                      if "746.8" in e.text())
        d_cell.setText("500")
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "d33 tensor cell 746.8 -> 500")
        snap = nxt

        # 6) wavelength on the SI tab is INERT BY DESIGN since the case-study fidelity audit
        # the restored original ♯SHAARP.si palette is single-wavelength
        # everywhere (the original tool has NO wavelength input -- lambda enters only through
        # the entered indices), so every SI case stores a single-point or constant grid and a
        # wavelength change must NOT alter the curves. The lambda -> dispersion -> curve
        # causality is exercised on the ML tab (MlCausalityMatrix, dispersive film row).
        # Pin the inertness so a regression that wires lambda into hidden SI state shows up.
        from shaarp.casestudy_materials import build_casestudy_material
        si_key, wl_a, wl_b = "GaAs (111) (800 nm)", 0.8, 1.4
        eps_a = np.asarray(build_casestudy_material(si_key, wavelength_um=wl_a).epsilon_omega)
        eps_b = np.asarray(build_casestudy_material(si_key, wavelength_um=wl_b).epsilon_omega)
        self.assertLess(float(np.max(np.abs(eps_a - eps_b))), 1e-12,
                        f"{si_key} registry data became dispersive -- revisit this subtest")
        case.setCurrentText("GaAs (111)")
        self.app.processEvents()
        wl = self._spin("wavelength")
        wl.setValue(wl_a)
        self._update()
        snap = self._snapshot()
        wl.setValue(wl_b)
        self._update()
        nxt = self._snapshot()
        scale = max((float(np.nanmax(np.abs(a))) for a in snap if a.size), default=0.0) + 1e-300
        diff = max((float(np.nanmax(np.abs(a - b))) for a, b in zip(snap, nxt) if a.size),
                   default=0.0)
        self.assertLess(diff / scale, 1e-12,
                        "SI wavelength changed the curves of a single-lambda palette case -- "
                        "lambda must stay inert on the restored original SI palette")


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class MlCausalityMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()
        tabs = cls.win.findChild(QtWidgets.QTabWidget)
        cls.ml = tabs.widget(1)
        cls.canvas_cls = FigureCanvasQTAgg

    def _update(self):
        btn = next(b for b in self.ml.findChildren(QtWidgets.QPushButton)
                   if b.text() == "Update / Run")
        btn.click()
        self.app.processEvents()

    def _snapshot(self):
        out = []
        for cv in self.ml.findChildren(self.canvas_cls):
            if cv.objectName().startswith("orient_canvas"):
                continue
            for ax in cv.figure.axes:
                for ln in ax.lines:
                    out.append(np.asarray(ln.get_ydata(), dtype=float))
        return out

    def _assert_changed(self, before, after, label):
        if len(before) != len(after) or any(a.shape != b.shape for a, b in zip(before, after)):
            return
        scale = max((float(np.nanmax(np.abs(a))) for a in before if a.size), default=0.0) + 1e-300
        diff = max((float(np.nanmax(np.abs(a - b))) for a, b in zip(before, after) if a.size),
                   default=0.0)
        self.assertGreater(diff / scale, 1e-6,
                           f"DEAD CONTROL: {label} changed nothing in the rendered curves "
                           "(F40 class -- the widget is not wired into the compute)")

    def test_ml_input_controls_all_drive_the_result(self):
        from shaarp.desktop_app import TOOLTIPS

        from . import gui_harness as gh
        preset = gh.ml_case_combo(self.ml)
        film_item = gh.ml_film_labels(self.ml)[0]  # first palette film row
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        gh.select_ml_film(self.ml, self.app, film_item)
        func.setCurrentText("Maker Fringes")
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        self.assertTrue(any(a.size for a in snap), "ML Maker baseline produced no curves")

        # 1) film thickness -> fringe pattern must shift. the layer editor's per-layer
        # spin is the SINGLE thickness input (the standalone film-thickness row is deleted);
        # select_ml_film leaves the editor on the film row, so this edit targets the film.
        thick = [s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
                 if s.toolTip() == TOOLTIPS["thickness"]]
        self.assertEqual(len(thick), 1,
                         "F55 contract: exactly ONE thickness spin (the layer editor's)")
        case_combo = gh.ml_case_combo(self.ml)
        mode_before = case_combo.currentText()
        thick[0].setValue(thick[0].value() * 3.0 + 1.0)
        self.app.processEvents()
        self.assertEqual(case_combo.currentText(), mode_before,
                         "a thickness edit in a simple film mode must STAY in the mode "
                         "(F55) - flipping to the N-layer editor would make the mode "
                         "unable to set its own thickness")
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "film thickness x3")
        snap = nxt

        # 2) multiple-reflection assumption
        assum = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                     if "assumption" in c.toolTip().lower())
        assum.setCurrentIndex((assum.currentIndex() + 1) % assum.count())
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "multiple-reflection assumption switch")
        snap = nxt

        # 3) wavelength (dispersive film preset)
        wl = [s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
              if s.toolTip() == TOOLTIPS["wavelength"]]
        self.assertTrue(wl, "no wavelength spinbox found")
        wl[0].setValue(1.3)
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "wavelength -> 1.3 um")


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SiCausalityMatrixExtended(unittest.TestCase):
    """The surfaces the first
    matrix did NOT drive -- the Miller orientation mode, the hkl cells, the analyzer/polarizer
    VALUE spins (not just modes), the point-group combo, and the widget->ANALYTICAL-OUTPUT seam
    (the expr box is a result surface too; a theta edit that doesn't reach the closed form is the
    same dead-control class as F40)."""

    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()
        cls.si = cls.win.findChild(QtWidgets.QTabWidget).widget(0)
        cls.canvas_cls = FigureCanvasQTAgg

    def _update(self):
        next(b for b in self.si.findChildren(QtWidgets.QPushButton)
             if b.text() == "Update / Run").click()
        self.app.processEvents()

    def _snapshot(self):
        out = []
        for cv in self.si.findChildren(self.canvas_cls):
            if cv.objectName().startswith("orient_canvas"):
                continue
            for ax in cv.figure.axes:
                for ln in ax.lines:
                    out.append(np.asarray(ln.get_ydata(), dtype=float))
        return out

    def _assert_changed(self, before, after, label):
        if len(before) != len(after) or any(a.shape != b.shape for a, b in zip(before, after)):
            return
        scale = max((float(np.nanmax(np.abs(a))) for a in before if a.size), default=0.0) + 1e-300
        diff = max((float(np.nanmax(np.abs(a - b))) for a, b in zip(before, after) if a.size),
                   default=0.0)
        self.assertGreater(diff / scale, 1e-6,
                           f"DEAD CONTROL: {label} changed nothing in the rendered curves")

    def _spins(self, tip_key):
        from shaarp.desktop_app import TOOLTIPS
        return [s for s in self.si.findChildren(QtWidgets.QDoubleSpinBox)
                if s.toolTip() == TOOLTIPS[tip_key]]

    def test_miller_mode_and_hkl_cells_drive_the_result(self):
        case = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        orient = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                      if c.findText("Miller (hkl + in-plane uvw)") >= 0 and c is not case)
        case.setCurrentText("GaAs (111)")
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        # user switches to Miller (001): -43m (001) vs (111) are decisively different patterns
        orient.setCurrentText("Miller (hkl + in-plane uvw)")
        orient.activated.emit(orient.currentIndex())
        self.app.processEvents()
        self.assertEqual(case.currentText(), "GaAs (111)",
                         "F57: a Miller-mode edit STAYS under the case (edited working copy)")
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "orientation mode case-(111) -> Miller (001)")
        snap = nxt
        # hkl cells (integer QSpinBox, tooltip key "hkl"): tilt the cubic cut -- for -43m any
        # surface change alters the SHG pattern. In-plane [uvw] must stay perpendicular to the
        # new surface, so set BOTH consistently: (011) surface with [100] in-plane.
        from shaarp.desktop_app import TOOLTIPS
        hkl = [s for s in self.si.findChildren(QtWidgets.QSpinBox)
               if s.toolTip() == TOOLTIPS["hkl"]]
        self.assertEqual(len(hkl), 3, "could not locate the 3 hkl cells")
        hkl[1].setValue(1)  # (0,0,1) -> (0,1,1); default uvw [100] remains in-plane
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "surface hkl (001) -> (011) in Miller mode")

    def test_analyzer_and_polarizer_value_spins_drive_the_result(self):
        case = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        case.setCurrentText("GaAs (111)")
        self.app.processEvents()
        analyzer_mode = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                             if c.findText("Fix Analyzer") >= 0)
        polarizer_mode = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                              if c.findText("Fix Polarizer") >= 0)
        # fixed analyzer: psi VALUE must reach the compute (same figure structure both runs)
        analyzer_mode.setCurrentText("Fix Analyzer")
        for s in self._spins("analyzer"):
            s.setValue(20.0)
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        for s in self._spins("analyzer"):
            s.setValue(65.0)
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "fixed-analyzer psi 20 -> 65")
        # fix polarizer: phi0 VALUE must reach the compute
        analyzer_mode.setCurrentText("Rotate Analyzer")   # 2-option analyzer (offset=0 -> parallel+perp)
        polarizer_mode.setCurrentText("Fix Polarizer")
        for s in self._spins("polarizer"):
            s.setValue(40.0)
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        for s in self._spins("polarizer"):
            s.setValue(75.0)
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "fix-polarizer phi 40 -> 75")
        # co-rotating analyzer: the OFFSET value must reach the compute (its spin carries a
        # custom tooltip, not the shared "analyzer" tip -- discover it by that text)
        polarizer_mode.setCurrentText("Rotate Polarizer")
        analyzer_mode.setCurrentText("Rotate Analyzer")   # Rotating + nonzero offset co-rotates the analyzer
        off_spin = next(s for s in self.si.findChildren(QtWidgets.QDoubleSpinBox)
                        if "offset angle" in s.toolTip())
        off_spin.setValue(0.0)
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        off_spin.setValue(25.0)
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "co-rotating analyzer offset 0 -> 25")

    def test_point_group_combo_drives_custom_result(self):
        case = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                    if c.findText("Custom (use fields)") >= 0)
        case.setCurrentText("Custom (use fields)")
        pg = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                  if c.findText("-43m") >= 0 and c.findText("3m") >= 0)
        pg.setCurrentText("-43m")
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        pg.setCurrentText("3m")
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "point group -43m -> 3m (custom mode)")

    def test_theta_reaches_the_partial_analytical_expression(self):
        """The ANALYTICAL OUTPUT is a result surface too: a theta edit that never reaches the
        closed form is the class on a text panel instead of a plot."""
        func = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        partial = next(func.itemText(i) for i in range(func.count())
                       if "Partial" in func.itemText(i))  # display label, e.g. "... Expression"
        func.setCurrentText(partial)
        theta = self._spins("theta")[0]
        expr = self.si.findChild(QtWidgets.QTextEdit, "expr_box")
        theta.setValue(30.0)
        self._update()
        raw_a = expr.property("raw_text") or ""
        self.assertTrue(raw_a.strip(), "Partial Analytical produced no expression")
        theta.setValue(55.0)
        self._update()
        raw_b = expr.property("raw_text") or ""
        self.assertTrue(raw_b.strip(), "Partial Analytical produced no expression at theta=55")
        self.assertNotEqual(raw_a, raw_b,
                            "DEAD CONTROL: theta edit does not reach the Partial Analytical "
                            "closed form (widget->expression seam)")


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SiExportCausality(unittest.TestCase):
    """F41 finding: the EXPORT is a result surface too. The SI SHG-Simulation
    'last_result' comes from the compat workflow at its CANONICAL incident polarization -- it
    never tracked the panel's polarimetry controls, and the payload carried no curves at all
    (the original GUI's Copy exported the plotted lists). Fixed: the plotted curve rides in
    payload['polarimetry_curves'] with a numeric_note declaring what the stage block is; this
    fence asserts the curves are present AND respond to a polarimetry edit."""

    def test_export_payload_carries_settings_dependent_curves(self):
        import json

        from shaarp.desktop_app import TOOLTIPS, build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        case.setCurrentText("GaAs (111)")
        app.processEvents()
        run_btn = next(b for b in si.findChildren(QtWidgets.QPushButton)
                       if b.text() == "Update / Run")
        ell = [s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
               if s.toolTip() == TOOLTIPS["ellipticity"]][0]
        # The startup analyzer default is "Fix Analyzer"; this test probes the
        # RICHER co-rotating export channels, so select Rotating explicitly (self-contained).
        analyzer_mode = next(c for c in si.findChildren(QtWidgets.QComboBox)
                             if c.findText("Fix Analyzer") >= 0)
        analyzer_mode.setCurrentText("Rotate Analyzer")
        app.processEvents()

        def run_and_payload(ell_deg):
            ell.setValue(ell_deg)
            run_btn.click()
            app.processEvents()
            return si._build_export_payload()

        p0 = run_and_payload(0.0)
        self.assertIn("polarimetry_curves", p0,
                      "SI export payload no longer carries the plotted curves (regression)")
        self.assertIn("numeric_note", p0, "export payload lost its numeric-block provenance note")
        # Rotating analyzer -> co-rotating channels (intensity_parallel/perpendicular), NOT the fixed
        # p/s channels (analyzer fix: "Rotate Analyzer" must actually co-rotate).
        # The export must still carry a non-empty, settings-dependent plotted curve.
        i0 = p0["polarimetry_curves"].get("intensity_parallel")
        self.assertTrue(i0 and max(map(abs, i0)) > 0, "exported intensity_parallel empty/zero")
        p1 = run_and_payload(35.0)
        i1 = p1["polarimetry_curves"].get("intensity_parallel")
        self.assertNotEqual(i0, i1,
                            "EXPORT STALENESS: the exported curves do not track the panel's "
                            "polarimetry settings (F41 class)")
        json.dumps(p1)  # the payload must remain JSON-serializable end-to-end


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class MlCausalityMatrixExtended(unittest.TestCase):
    """ML surfaces the first matrix did not drive -- the SHG-Simulation polarimetry controls,
    the N-layer stack editor (per-layer material / SHG-active), substrate indices, the Maker scan
    range, and the Calculation-Controls Fresnel checkbox."""

    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()
        cls.ml = cls.win.findChild(QtWidgets.QTabWidget).widget(1)
        cls.canvas_cls = FigureCanvasQTAgg

    def _update(self):
        next(b for b in self.ml.findChildren(QtWidgets.QPushButton)
             if b.text() == "Update / Run").click()
        self.app.processEvents()

    def _snapshot(self):
        out = []
        for cv in self.ml.findChildren(self.canvas_cls):
            if cv.objectName().startswith("orient_canvas"):
                continue
            for ax in cv.figure.axes:
                for ln in ax.lines:
                    out.append(np.asarray(ln.get_ydata(), dtype=float))
        return out

    def _assert_changed(self, before, after, label):
        if len(before) != len(after) or any(a.shape != b.shape for a, b in zip(before, after)):
            return
        scale = max((float(np.nanmax(np.abs(a))) for a in before if a.size), default=0.0) + 1e-300
        diff = max((float(np.nanmax(np.abs(a - b))) for a, b in zip(before, after) if a.size),
                   default=0.0)
        self.assertGreater(diff / scale, 1e-6,
                           f"DEAD CONTROL: {label} changed nothing in the rendered curves")

    def _spins(self, tip_key):
        from shaarp.desktop_app import TOOLTIPS
        return [s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
                if s.toolTip() == TOOLTIPS[tip_key]]

    def test_shg_simulation_polarimetry_controls_drive_the_result(self):
        from . import gui_harness as gh
        preset = gh.ml_case_combo(self.ml)
        film = next(c for c in gh.ml_film_labels(self.ml) if "x-cut" in c)
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        gh.select_ml_film(self.ml, self.app, film)
        func.setCurrentText("SHG Simulation")
        self.app.processEvents()
        theta = self._spins("theta")[0]
        theta.setValue(20.0)
        self._update()
        snap = self._snapshot()
        self.assertTrue(any(a.size for a in snap), "ML SHG Simulation produced no curves")
        theta.setValue(45.0)
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "ML incident angle 20 -> 45")
        snap = nxt
        ell = self._spins("ellipticity")[0]
        ell.setValue(30.0)
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "ML ellipticity 0 -> 30")

    def test_nlayer_editor_drives_the_result(self):
        from shaarp.desktop_app import TOOLTIPS

        preset = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                      if c.findText("N-layer stack (editor)") >= 0)
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        preset.setCurrentText("N-layer stack (editor)")
        func.setCurrentText("Maker Fringes")
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        self.assertTrue(any(a.size for a in snap), "N-layer Maker baseline produced no curves")
        # change an interior layer's MATERIAL -- must reach the stack compute
        lay_sel = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                       if c.toolTip() == TOOLTIPS["layer_select"])
        lay_mat = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                       if c.toolTip() == TOOLTIPS["layer_material"])
        # pick the middle (film-like) layer
        lay_sel.setCurrentIndex(min(1, lay_sel.count() - 1))
        self.app.processEvents()
        old_mat = lay_mat.currentText()
        new_mat = next(t for t in (lay_mat.itemText(i) for i in range(lay_mat.count()))
                       if t != old_mat and "Quartz" in t) if lay_mat.findText(
                           "Quartz z-cut · 800 nm") >= 0 else lay_mat.itemText(
                           (lay_mat.currentIndex() + 1) % lay_mat.count())
        lay_mat.setCurrentText(new_mat)
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, f"N-layer editor: layer material {old_mat!r} -> {new_mat!r}")
        snap = nxt
        # there is no SHG-active switch -- the POINT GROUP decides. Moving the film to a
        # centrosymmetric group (the original's "Centrosymmetric ->" list) kills its source.
        pg = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                  if c.toolTip() == TOOLTIPS["point_group"])
        # self-contained baseline (the class shares one window, so an earlier test may have
        # left this row at m3m already): an ACTIVE group first, then the inactive one
        pg.setCurrentText("3m")
        pg.activated.emit(pg.currentIndex())
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        pg.setCurrentText("m3m")
        pg.activated.emit(pg.currentIndex())
        self.app.processEvents()
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "N-layer editor: point group 3m -> m3m (SHG-inactive)")

    def test_substrate_indices_and_scan_range_drive_maker(self):
        from shaarp.desktop_app import TOOLTIPS

        from . import gui_harness as gh
        preset = gh.ml_case_combo(self.ml)
        film = next(c for c in gh.ml_film_labels(self.ml) if "Quartz" in c)
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        gh.select_ml_film(self.ml, self.app, film)
        func.setCurrentText("Maker Fringes")
        self.app.processEvents()
        self._update()
        snap = self._snapshot()
        # the substrate medium is edited through the DIELECTRIC TENSOR grids — select the
        # last (substrate) row, whose eps grids mirror n^2 * I, type a new diagonal, and the
        # Maker fringes must respond (mode must not switch).
        lay_sel = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                       if c.toolTip() == TOOLTIPS["layer_select"])
        lay_sel.setCurrentIndex(lay_sel.count() - 1)
        self.app.processEvents()
        # an isotropic medium is ONE number per frequency — the scalar index rows are the
        # entry surface (the 3x3 grids are hidden for these rows).
        n_spins = [s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
                   if "ISOTROPIC layer" in s.toolTip()]
        self.assertEqual(len(n_spins), 2, "expected the n(w)/n(2w) rows")
        self.assertAlmostEqual(n_spins[0].value(), 1.45, places=6,
                               msg="substrate row selected -> the index rows must mirror 1.45")
        mode_before = preset.currentText()
        for spin in n_spins:  # n(w) and n(2w) -> 2.4
            spin.setValue(2.4)
            self.app.processEvents()
        self.assertEqual(preset.currentText(), mode_before,
                         "a substrate eps edit must stay in the mode (F57)")
        self._update()
        nxt = self._snapshot()
        self._assert_changed(snap, nxt, "substrate eps 1.45^2 -> 2.4^2 via the tensor grids")
        snap = nxt
        # scan range: shrinking theta_max must shorten the swept arrays (structure change) or
        # at minimum change the data
        th_boxes = [s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
                    if abs(s.maximum() - 89.9) < 1e-6]
        self.assertTrue(th_boxes, "no theta_max scan spin found")
        th_boxes[0].setValue(25.0)
        self._update()
        nxt = self._snapshot()
        if len(nxt) == len(snap) and all(a.shape == b.shape for a, b in zip(snap, nxt)):
            self._assert_changed(snap, nxt, "Maker scan range theta_max -> 25")

    def test_fresnel_mode_populates_the_fresnel_tab(self):
        """The 'Fresnel Coefficients' FUNCTIONALITY is the one control (the old
        Generate-checkbox is gone) -- selecting it computes the sweep into the Fresnel tab
        and switches the output tabs to it."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from . import gui_harness as gh
        film = next(c for c in gh.ml_film_labels(self.ml) if "Quartz" in c)
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        gh.select_ml_film(self.ml, self.app, film)
        func.setCurrentText("Fresnel Coefficients")
        self.app.processEvents()
        self._update()
        canvas = next(cv for cv in self.ml.findChildren(FigureCanvasQTAgg)
                      if cv.objectName() == "fresnel_canvas")
        lines = [ln for ax in canvas.figure.axes for ln in ax.lines]
        self.assertTrue(lines, "Fresnel mode must populate the Fresnel canvas")
        tabs = next(t for t in self.ml.findChildren(QtWidgets.QTabWidget)
                    if any(t.tabText(i) == "Fresnel Coefficients" for i in range(t.count())))
        self.assertEqual(tabs.tabText(tabs.currentIndex()), "Fresnel Coefficients",
                         "Fresnel mode must switch the output tabs to its plot")
