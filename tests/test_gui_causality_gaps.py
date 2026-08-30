"""Phase C: the causality steps that close every enumerated coverage gap
(tests/gui_harness.py COVERAGE_RULES "todo" rows -> "covered" as each lands here).

Same discipline as the matrices: drive controls AS A USER, assert the RESULT SURFACE
changes (curves / expression / clipboard / widget mirrors), with physics guards so genuine
invariances are never misread as dead controls.
"""

import math
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False

import numpy as np


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SiGapSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window
        from . import gui_harness as gh

        cls.gh = gh
        cls.app = _app()
        cls.win = build_main_window()
        cls.si = cls.win.findChild(QtWidgets.QTabWidget).widget(0)

    def _update(self):
        self.gh.click_update(self.si, self.app)

    def _snap(self):
        return self.gh.snapshot_curves(self.si)

    def _changed(self, before, after, label):
        self.assertTrue(self.gh.curves_changed(before, after),
                        f"DEAD CONTROL: {label} changed nothing in the rendered curves")

    def test_uvw_and_lattice_cells_drive_miller_result(self):
        """(0,1,1) surface fixed; rotating the in-plane [uvw] and stretching the lattice both
        change the cut geometry of a -43m crystal -> the pattern must follow."""
        case = self.gh.combo_with_item(self.si, "GaAs (111)")
        orient = self.gh.combo_with_item(self.si, "Miller (hkl + in-plane uvw)", exclude=(case,))
        case.setCurrentText("GaAs (111)")
        self.app.processEvents()
        orient.setCurrentText("Miller (hkl + in-plane uvw)")
        orient.activated.emit(orient.currentIndex())
        self.app.processEvents()
        from shaarp.desktop_app import TOOLTIPS
        hkl = [s for s in self.si.findChildren(QtWidgets.QSpinBox)
               if s.toolTip() == TOOLTIPS["hkl"]]
        uvw = [s for s in self.si.findChildren(QtWidgets.QSpinBox)
               if s.toolTip() == TOOLTIPS["uvw"]]
        for s, v in zip(hkl, (0, 1, 1)):
            s.setValue(v)
        for s, v in zip(uvw, (1, 0, 0)):
            s.setValue(v)
        self.app.processEvents()
        self._update()
        snap = self._snap()
        # in-plane [100] -> [0,1,-1] (still in the (011) plane) rotates the cut about the normal
        for s, v in zip(uvw, (0, 1, -1)):
            s.setValue(v)
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "in-plane [uvw] rotation in Miller mode")
        snap = nxt
        # lattice: F63 locks a cubic cell (a=b=c), so first move the custom crystal to a
        # tetragonal group (c free), then stretch c -- the (011) geometry changes. 422 keeps a
        # SINGLE free component (d14, like -43m): under a general Miller rotation the SI
        # analytical path's CAS cost grows steeply with the component count (4mm took 1086 s).
        pg = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                  if c.findText("-43m") >= 0 and c.findText("3m") >= 0)
        pg.setCurrentText("422")
        pg.activated.emit(pg.currentIndex())
        self.app.processEvents()
        self._update()
        snap = self._snap()
        lat = self.gh.spins_with_tip(self.si, "lattice")
        self.assertGreaterEqual(len(lat), 6)
        self.assertTrue(lat[2].isEnabled(), "c is free under a tetragonal group")
        lat[2].setValue(2.0)  # c
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "lattice c 1 -> 2 in Miller mode")

    def test_eps_cells_drive_the_result(self):
        case = self.gh.combo_with_item(self.si, "Custom (use fields)")
        case.setCurrentText("Custom (use fields)")
        self.app.processEvents()
        self._update()
        snap = self._snap()
        cells = [e for e in self.si.findChildren(QtWidgets.QLineEdit)
                 if e.toolTip() and "full 3" in e.toolTip().lower()] or None
        if cells is None or len(cells) < 36:
            from shaarp.desktop_app import TOOLTIPS
            cells = [e for e in self.si.findChildren(QtWidgets.QLineEdit)
                     if e.toolTip() == TOOLTIPS["full_tensor"]]
        self.assertGreaterEqual(len(cells), 36, "could not locate the tensor grids")
        # cell #0 = eps_w[0,0] in creation order: bump it decisively
        old = cells[0].text()
        cells[0].setText("9.5")
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, f"eps(w)[0,0] cell {old!r} -> 9.5")

    def test_theta_slider_and_quick_buttons_mirror_the_spin(self):
        theta = self.gh.spins_with_tip(self.si, "theta")[0]
        slider = next(s for s in self.si.findChildren(QtWidgets.QSlider)
                      if s.toolTip() == theta.toolTip())
        slider.setValue(37)
        self.app.processEvents()
        self.assertAlmostEqual(theta.value(), 37.0, places=6,
                               msg="theta slider does not drive the theta spin")
        btn30 = next(b for b in self.si.findChildren(QtWidgets.QPushButton)
                     if b.text() == "30")
        btn30.click()
        self.app.processEvents()
        self.assertAlmostEqual(theta.value(), 30.0, places=6,
                               msg="quick-angle button does not set the theta spin")

    def test_quick_buttons_show_the_selected_setting(self):
        """The preset rows must SHOW which setting is active. The
        highlight tracks the spin from EVERY source -- preset click, typed value (custom ->
        nothing lit; the free input is the fallback when presets don't fit), and clicking the
        already-active preset must not toggle its highlight off."""
        theta = self.gh.spins_with_tip(self.si, "theta")[0]
        # scope to the THETA row: "45"/"0" also exist in the fixed-phi and psi rows, and a flat
        # text->button dict would keep the wrong (disabled) row's button
        anchor = next(b for b in self.si.findChildren(QtWidgets.QPushButton)
                      if b.text() == "15" and b.isCheckable())
        btns = {b.text(): b for b in anchor.parent().findChildren(QtWidgets.QPushButton)}
        self.assertGreaterEqual(len(btns), 6, "quick-angle buttons are not checkable (F44)")
        btns["30"].click()
        self.app.processEvents()
        self.assertTrue(btns["30"].isChecked(), "active preset is not highlighted")
        self.assertFalse(btns["45"].isChecked())
        btns["30"].click()  # clicking the ACTIVE preset must keep it highlighted
        self.app.processEvents()
        self.assertTrue(btns["30"].isChecked(),
                        "clicking the active preset toggled its highlight off")
        theta.setValue(45.0)  # typed/programmatic value -> highlight follows
        self.app.processEvents()
        self.assertTrue(btns["45"].isChecked(), "highlight does not track a typed value")
        self.assertFalse(btns["30"].isChecked())
        theta.setValue(37.0)  # CUSTOM value: nothing lit, the spin carries it
        self.app.processEvents()
        self.assertFalse(any(b.isChecked() for b in btns.values()),
                         "a custom typed value must light no preset button")

    def test_update_buttons_are_equivalent(self):
        """All Update surfaces (input panel, output panel, window-global) drive the same run."""
        from shaarp.desktop_app import TOOLTIPS

        case = self.gh.combo_with_item(self.si, "GaAs (111)")
        case.setCurrentText("GaAs (111)")
        self.app.processEvents()
        runs = [b for b in self.si.findChildren(QtWidgets.QPushButton)
                if b.toolTip() == TOOLTIPS["run"]]
        self.assertGreaterEqual(len(runs), 2, "expected two run-tipped Update buttons on the page")
        tabs = self.win.findChild(QtWidgets.QTabWidget)
        tabs.setCurrentIndex(0)
        page_ids = set(map(id, runs))
        top = next(b for b in self.win.findChildren(QtWidgets.QPushButton)
                   if b.toolTip() == TOOLTIPS["run"] and id(b) not in page_ids
                   and not tabs.widget(1).isAncestorOf(b))
        snaps = []
        for b in [*runs, top]:
            b.click()
            self.app.processEvents()
            snaps.append(self._snap())
        base = snaps[0]
        self.assertTrue(any(a.size for a in base), "baseline Update produced no curves")
        for k, s in enumerate(snaps[1:], 1):
            self.assertFalse(self.gh.curves_changed(base, s),
                             f"Update surface #{k} produced a DIFFERENT result than surface #0")

    def test_copy_buttons_put_tracking_content_on_the_clipboard(self):
        func = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        partial = next(func.itemText(i) for i in range(func.count())
                       if "Partial" in func.itemText(i))
        func.setCurrentText(partial)
        theta = self.gh.spins_with_tip(self.si, "theta")[0]
        from shaarp.desktop_app import TOOLTIPS
        copy_sym = next(b for b in self.si.findChildren(QtWidgets.QPushButton)
                        if b.toolTip() == TOOLTIPS["closed_form"])
        copy_mma = next(b for b in self.si.findChildren(QtWidgets.QPushButton)
                        if b.toolTip().startswith("Copy the closed-form expressions in Wolf"))
        cb = QtWidgets.QApplication.clipboard()

        def run_and_copy(th):
            theta.setValue(th)
            self._update()
            cb.clear()
            copy_sym.click()
            self.app.processEvents()
            sym = cb.text()
            cb.clear()
            copy_mma.click()
            self.app.processEvents()
            return sym, cb.text()

        sym_a, mma_a = run_and_copy(30.0)
        self.assertTrue(sym_a.strip(), "Copy closed form put nothing on the clipboard")
        self.assertTrue(mma_a.strip(), "Copy Mathematica form put nothing on the clipboard")
        sym_b, mma_b = run_and_copy(55.0)
        self.assertNotEqual(sym_a, sym_b, "clipboard closed form does not track theta")
        self.assertNotEqual(mma_a, mma_b, "clipboard Mathematica form does not track theta")

    def test_preset_slots_round_trip(self):
        from shaarp.desktop_app import TOOLTIPS
        pg = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                  if c.findText("-43m") >= 0 and c.findText("3m") >= 0)
        wl = self.gh.spins_with_tip(self.si, "wavelength")[0]
        label = next(e for e in self.si.findChildren(QtWidgets.QLineEdit)
                     if e.toolTip() == TOOLTIPS["presets"])
        slots = [b for b in self.si.findChildren(QtWidgets.QPushButton)
                 if b.toolTip() == TOOLTIPS["presets"] and b.text().startswith("Preset")]
        clear = next(b for b in self.si.findChildren(QtWidgets.QPushButton)
                     if b.toolTip() == TOOLTIPS["presets"] and b.text() == "Clear Presets")
        self.assertTrue(slots, "no preset slot buttons found")
        pg.setCurrentText("3m")
        wl.setValue(1.234)
        label.setText("gap-step")
        slots[0].click()          # save
        self.app.processEvents()
        self.assertIn("gap-step", slots[0].text(), "preset label did not reach the slot button")
        pg.setCurrentText("-43m")
        wl.setValue(0.8)
        slots[0].click()          # recall
        self.app.processEvents()
        self.assertEqual(pg.currentText(), "3m", "preset recall did not restore the point group")
        self.assertAlmostEqual(wl.value(), 1.234, places=6,
                               msg="preset recall did not restore the wavelength")
        clear.click()
        self.app.processEvents()
        self.assertEqual(slots[0].text(), "Preset 1", "Clear Presets did not reset the slot")


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class MlGapSteps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window
        from . import gui_harness as gh

        cls.gh = gh
        cls.app = _app()
        cls.win = build_main_window()
        cls.ml = cls.win.findChild(QtWidgets.QTabWidget).widget(1)

    def _update(self):
        self.gh.click_update(self.ml, self.app)

    def _snap(self):
        return self.gh.snapshot_curves(self.ml)

    def _changed(self, before, after, label):
        self.assertTrue(self.gh.curves_changed(before, after),
                        f"DEAD CONTROL: {label} changed nothing in the rendered curves")

    def _select_custom_film_from(self, film_name):
        """Select a film row (stripped palette label, case-study fidelity audit), then
        flip to Custom film via a user orientation edit so the panel keeps the film's tensors
        (the mechanism, reused as a fixture)."""
        return self.gh.select_ml_film(self.ml, self.app, film_name)

    def test_custom_film_orientation_and_tensor_cells(self):
        from shaarp.desktop_app import TOOLTIPS
        preset = self._select_custom_film_from("LiNbO3 z-cut · 1550 nm")
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText("Maker Fringes")
        self.app.processEvents()
        self._update()
        snap = self._snap()
        # ML z-cells: user tilt (Crystal Physics Directions) -> flips to Custom film + changes
        orient = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                      if c.findText("Crystal Physics Directions (Z1,Z2,Z3)") >= 0
                      and c is not preset and c is not func)
        orient.setCurrentText("Crystal Physics Directions (Z1,Z2,Z3)")
        orient.activated.emit(orient.currentIndex())
        self.app.processEvents()
        z = [s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
             if s.toolTip() == TOOLTIPS["z_axes"]]
        self.assertGreaterEqual(len(z), 9)
        c20, s20 = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
        ry20 = [c20, 0.0, s20, 0.0, 1.0, 0.0, -s20, 0.0, c20]
        for w, v in zip(z[:9], ry20):
            w.setValue(v)
        self.app.processEvents()
        self.assertEqual(preset.currentText().strip(), "LiNbO3 z-cut · 1550 nm",
                         "F57: a z-cell edit STAYS under the film case (edited working copy)")
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "ML z-axes 20-deg tilt on the custom film")
        snap = nxt
        # (note: a point-group pick under a case no longer changes the compute by itself —
        # the FULL tensor grids drive the custom build ("full tensors win"), and staying under
        # the case means populate refills the case's own d rather than a bare symmetry
        # template. The tensor-CELL steps below carry the panel-edit causality.)
        cells = [e for e in self.ml.findChildren(QtWidgets.QLineEdit)
                 if e.toolTip() == TOOLTIPS["full_tensor"]]
        self.assertGreaterEqual(len(cells), 36)
        cells[0].setText("7.7")  # eps_w[0,0]
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "ML eps(w)[0,0] cell -> 7.7")
        snap = nxt
        d_cell = cells[18 + 14]  # d row 3 col 3 area; any d cell of the -43m grid
        d_cell.setText("3.3")
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "ML d cell -> 3.3")

    def test_ml_miller_mode_cells(self):
        preset = self._select_custom_film_from("GaAs (111) (800 nm)")
        # Deterministic thickness: an earlier test in this class leaves h = 11 um, and GaAs's
        # SHG at 400 nm is strongly absorbed (eps_2w = 16.94+2.73j) -- through 11 um the
        # transmitted Maker signal decays to ~1e-30 numerical noise, where the (001)-vs-(011)
        # comparison below is meaningless. 0.5 um keeps the signal healthy. write it
        # AFTER the mode switch (the template rebuild would overwrite an earlier write) on
        # the editor's film row, where the simple mode remembers it.
        self.gh.spins_with_tip(self.ml, "thickness")[0].setValue(0.5)
        self.app.processEvents()
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText("Maker Fringes")
        orient = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                      if c.findText("Miller (hkl + in-plane uvw)") >= 0
                      and c is not preset and c is not func)
        orient.setCurrentText("Miller (hkl + in-plane uvw)")
        orient.activated.emit(orient.currentIndex())
        self.app.processEvents()
        from shaarp.desktop_app import TOOLTIPS
        hkl = [s for s in self.ml.findChildren(QtWidgets.QSpinBox)
               if s.toolTip() == TOOLTIPS["hkl"]]
        uvw = [s for s in self.ml.findChildren(QtWidgets.QSpinBox)
               if s.toolTip() == TOOLTIPS["uvw"]]
        for s, v in zip(hkl, (0, 0, 1)):
            s.setValue(v)
        for s, v in zip(uvw, (1, 0, 0)):
            s.setValue(v)
        self.app.processEvents()
        self._update()
        snap = self._snap()
        for s, v in zip(hkl, (0, 1, 1)):
            s.setValue(v)
        for s, v in zip(uvw, (1, 0, 0)):
            s.setValue(v)
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "ML Miller (001) -> (011)")

    def test_ml_polarimetry_value_controls(self):
        preset = self._select_custom_film_from("LiNbO3 x-cut · 1550 nm")
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText("SHG Simulation")
        self.app.processEvents()
        analyzer_mode = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                             if c.findText("Fix Analyzer") >= 0)
        polarizer_mode = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                              if c.findText("Fix Polarizer") >= 0)
        analyzer_mode.setCurrentText("Fix Analyzer")
        for s in self.gh.spins_with_tip(self.ml, "analyzer"):
            s.setValue(20.0)
        self.app.processEvents()
        self._update()
        snap = self._snap()
        for s in self.gh.spins_with_tip(self.ml, "analyzer"):
            s.setValue(65.0)
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "ML fixed-analyzer psi 20 -> 65")
        # fix polarizer value
        analyzer_mode.setCurrentText("Rotate Analyzer")   # 2-option analyzer (offset=0 -> parallel+perp)
        polarizer_mode.setCurrentText("Fix Polarizer")
        for s in self.gh.spins_with_tip(self.ml, "polarizer"):
            s.setValue(40.0)
        self.app.processEvents()
        self._update()
        snap = self._snap()
        for s in self.gh.spins_with_tip(self.ml, "polarizer"):
            s.setValue(75.0)
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "ML fix-polarizer phi 40 -> 75")
        # co-rotating offset value (2-option analyzer: Rotating + a nonzero offset co-rotates)
        polarizer_mode.setCurrentText("Rotate Polarizer")
        analyzer_mode.setCurrentText("Rotate Analyzer")
        off = next(s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
                   if "offset angle" in s.toolTip())
        off.setValue(0.0)
        self.app.processEvents()
        self._update()
        snap = self._snap()
        off.setValue(25.0)
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "ML co-rotating offset 0 -> 25")

    def test_scan_range_layer_editor_and_submode(self):
        from shaarp.desktop_app import TOOLTIPS
        preset = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                      if c.findText("N-layer stack (editor)") >= 0)
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        preset.setCurrentText("N-layer stack (editor)")
        func.setCurrentText("Maker Fringes")
        self.app.processEvents()
        self._update()
        snap = self._snap()
        # scan range: min/max/step must all reach the sweep (structure or data changes)
        rng = self.gh.spins_with_tip(self.ml, "theta_range")
        self.assertGreaterEqual(len(rng), 3, "expected theta min/max/step spins")
        for spin, val, label in ((rng[0], 5.0, "theta_min -> 5"),
                                 (rng[1], 35.0, "theta_max -> 35"),
                                 (rng[2], max(0.5, rng[2].value() * 2), "theta_step x2")):
            spin.setValue(float(val))
            self._update()
            nxt = self._snap()
            if not self.gh.curves_changed(snap, nxt):
                self.fail(f"DEAD CONTROL: scan {label} changed nothing")
            snap = nxt
        # n_layers: deeper stack changes the physics (new default layer inserted)
        n_layers = next(s for s in self.ml.findChildren(QtWidgets.QSpinBox)
                        if s.toolTip() == TOOLTIPS["n_layers"])
        n_layers.setValue(n_layers.value() + 1)
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "n_layers +1")
        snap = nxt
        # layer thickness: F55 made the editor's per-layer spin the SINGLE thickness input
        # (the standalone film row is deleted). Drive it on a FINITE layer (half-infinite
        # layers disable it -- correct physics).
        thick = self.gh.spins_with_tip(self.ml, "thickness")
        self.assertEqual(len(thick), 1,
                         "F55 contract: exactly ONE thickness spin (the layer editor's)")
        layer_spin = thick[0]
        lay_sel = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                       if c.toolTip() == TOOLTIPS["layer_select"])
        finite_idx = None
        for idx in range(lay_sel.count()):
            lay_sel.setCurrentIndex(idx)
            self.app.processEvents()
            if layer_spin.isEnabled():
                finite_idx = idx
                if idx > 0:  # interior layer preferred
                    break
        self.assertIsNotNone(finite_idx, "no finite (thickness-enabled) layer found in the stack")
        lay_sel.setCurrentIndex(finite_idx)
        self.app.processEvents()
        layer_spin.setValue(layer_spin.value() * 3.0 + 2.0)
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "N-layer per-layer thickness x3 (finite layer)")
        snap = nxt
        # FMR submode: retained-source policy for inhomogeneous backward waves; needs a stack
        # with backside reflection to matter -- the N-layer stack qualifies
        assum = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                     if c.toolTip() == TOOLTIPS["assumptions"])
        assum.setCurrentText(next(assum.itemText(i) for i in range(assum.count())
                                  if "Full Multiple Reflections" in assum.itemText(i)))
        sub = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                   if c.toolTip() == TOOLTIPS["fmr_submode"])
        self.app.processEvents()
        self._update()
        snap = self._snap()
        sub.setCurrentIndex((sub.currentIndex() + 1) % sub.count())
        self.app.processEvents()
        self._update()
        nxt = self._snap()
        self._changed(snap, nxt, "FMR submode switch")

    def test_scan_preset_rows_light_up_at_startup(self):
        """The reported case: scan defaults 0 / 45 / 0.1 -- the matching
        theta_min / theta_max / step preset buttons must be highlighted from the first paint."""
        rng = self.gh.spins_with_tip(self.ml, "theta_range")
        self.assertGreaterEqual(len(rng), 3)
        for spin, preset_text in ((rng[0], "0"), (rng[1], "45"), (rng[2], "0.1")):
            spin.setValue(float(preset_text))
            self.app.processEvents()
            lit = [b.text() for b in self.ml.findChildren(QtWidgets.QPushButton)
                   if b.isCheckable() and b.isChecked()]
            self.assertIn(preset_text, lit,
                          f"scan preset {preset_text!r} not highlighted when active (lit: {lit})")

    def test_scan_polarimetry_relevance_gating(self):
        """The input groups the SELECTED
        functionality does not consume collapse + gain a '— not used' title hint; the relevant
        ones expand. Maker/Fresnel use Scan Range (not polarimetry rotation); SHG Simulation +
        analytical use Polarimetry Settings (not the scan sweep)."""
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        groups = {b.title().split("   ")[0]: b for b in self.ml.findChildren(QtWidgets.QGroupBox)}
        # the shared scan group split into per-mode sections
        scan = groups["Maker Fringes Scan Range"]
        fres = groups["Fresnel Coefficients Scan Range"]
        pol = groups["Polarimetry Settings"]

        def pick(substr):
            func.setCurrentText(next(func.itemText(i) for i in range(func.count())
                                     if substr in func.itemText(i)))
            self.app.processEvents()

        pick("Maker")
        self.assertTrue(scan.isChecked(), "Maker Fringes Scan Range collapsed in Maker Fringes mode")
        self.assertFalse(fres.isChecked(), "Fresnel section not collapsed in Maker Fringes mode")
        # polarimetry stays OPEN in Maker Fringes -- its input phi and detection psi live there
        self.assertTrue(pol.isChecked(), "Polarimetry must stay open in Maker Fringes (F70)")
        pick("Fresnel")
        self.assertTrue(fres.isChecked(), "Fresnel section collapsed in Fresnel Coefficients mode")
        self.assertFalse(scan.isChecked(), "Maker section not collapsed in Fresnel mode")
        self.assertFalse(pol.isChecked(), "Polarimetry not collapsed in Fresnel mode")
        self.assertIn("not used", pol.title(), "collapsed group missing the '— not used' hint")
        pick("SHG Simulation")
        self.assertTrue(pol.isChecked(), "Polarimetry collapsed in SHG Simulation")
        self.assertFalse(scan.isChecked(), "Maker section not collapsed in SHG Simulation")
        self.assertFalse(fres.isChecked(), "Fresnel section not collapsed in SHG Simulation")
        # a hinted/collapsed group must remain EXPANDABLE for pre-setting (grill requirement):
        scan.setChecked(True)
        self.app.processEvents()
        th = self.gh.spins_with_tip(self.ml, "theta_range")
        self.assertTrue(th and th[0].isEnabled(),
                        "scan spins not editable when a collapsed group is re-expanded")

    def test_scan_presets_are_inline_with_their_spin(self):
        """Each scan preset button shares its spin's ROW (not a separate labelled row) --
        assert a preset button and its spin have the same parent row widget."""
        from shaarp.desktop_app import TOOLTIPS
        th_min = next(s for s in self.ml.findChildren(QtWidgets.QDoubleSpinBox)
                      if s.toolTip() == TOOLTIPS["theta_range"])
        row = th_min.parent()
        btns = [b for b in row.findChildren(QtWidgets.QPushButton) if b.isCheckable()]
        self.assertTrue(btns, "scan spin's row carries no inline preset buttons (F48)")
        self.assertLessEqual(len(self.gh.spins_with_tip(self.ml, "theta_range")), 3,
                             "expected exactly the 3 scan spins")

    def test_layer_name_reaches_the_layer_list(self):
        from shaarp.desktop_app import TOOLTIPS
        preset = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                      if c.findText("N-layer stack (editor)") >= 0)
        preset.setCurrentText("N-layer stack (editor)")
        self.app.processEvents()
        name_edit = next(e for e in self.ml.findChildren(QtWidgets.QLineEdit)
                         if e.toolTip().startswith("Optional layer name"))
        lay_sel = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                       if c.toolTip() == TOOLTIPS["layer_select"])
        name_edit.setText("MyTestLayer")
        name_edit.editingFinished.emit()
        self.app.processEvents()
        items = [lay_sel.itemText(i) for i in range(lay_sel.count())]
        self.assertTrue(any("MyTestLayer" in t for t in items),
                        f"FB3 layer name did not reach the layer list ({items})")

    def test_analytical_checkboxes_drive_the_partial_expression(self):
        preset = self._select_custom_film_from("Quartz z-cut · 800 nm")
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        partial = next(func.itemText(i) for i in range(func.count())
                       if "Partial" in func.itemText(i))
        func.setCurrentText(partial)
        h_chk = next(c for c in self.ml.findChildren(QtWidgets.QCheckBox)
                     if c.toolTip().startswith("Original .ml 'analytical h'"))
        d_chk = next(c for c in self.ml.findChildren(QtWidgets.QCheckBox)
                     if c.toolTip().startswith("Original .ml 'analytical dij'"))
        expr = self.ml.findChild(QtWidgets.QTextEdit, "expr_box")

        def run_raw():
            self._update()
            return expr.property("raw_text") or ""

        h_chk.setChecked(True)
        raw_h_on = run_raw()
        self.assertTrue(raw_h_on.strip(), "ML Partial produced no expression")
        h_chk.setChecked(False)
        raw_h_off = run_raw()
        self.assertNotEqual(raw_h_on, raw_h_off,
                            "DEAD CONTROL: analytical-h checkbox does not change the expression")
        d_chk.setChecked(not d_chk.isChecked())
        raw_d = run_raw()
        self.assertNotEqual(raw_h_off, raw_d,
                            "DEAD CONTROL: analytical-dij checkbox does not change the expression")

    def test_per_layer_analytic_d_drives_the_partial_expression(self):
        """The PER-LAYER 'analytical dij' flag (the original's per-layer button,
        SHAARP.ml.nb:7256-7292) must change the closed form -- distinct from the stack-wide
        known/unknown-d switch in the SHG Tensor group."""
        self._select_custom_film_from("Quartz z-cut · 800 nm")
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText(next(func.itemText(i) for i in range(func.count())
                                 if "Partial" in func.itemText(i)))
        # there is now ONE analytical-dij control and it lives in the SHG-Tensor panel
        # (you have two analytical toggle ... you should only keep that in d tensor").
        per_layer_d = next(c for c in self.ml.findChildren(QtWidgets.QCheckBox)
                           if c.toolTip().startswith("Original .ml 'analytical dij'"))
        expr = self.ml.findChild(QtWidgets.QTextEdit, "expr_box")
        self._update()
        before = expr.property("raw_text") or ""
        self.assertTrue(before.strip(), "ML Partial produced no expression")
        per_layer_d.setChecked(not per_layer_d.isChecked())
        self.app.processEvents()
        self._update()
        after = expr.property("raw_text") or ""
        self.assertNotEqual(before, after,
                            "DEAD CONTROL: the per-layer analytical-dij flag did not change "
                            "the closed form")

    def test_eps_entry_mode_converts_the_display_without_changing_the_physics(self):
        """Switching between
        permittivity and refractive-index entry CONVERTS what is on screen (eps = n.n) and must
        leave the COMPUTED result untouched -- it is a change of units, not of physics. This is
        an equal-results fence, so it fails both if the conversion is wrong (numbers unchanged)
        and if the mode leaks into the compute (curves changed)."""
        from shaarp.desktop_app import TOOLTIPS, _parse_complex

        self._select_custom_film_from("Quartz z-cut · 800 nm")
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText("SHG Simulation")
        self.app.processEvents()
        mode = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip() == TOOLTIPS["eps_mode"])
        cells = [e for e in self.ml.findChildren(QtWidgets.QLineEdit)
                 if e.toolTip() == TOOLTIPS["full_tensor"]]
        self.assertGreaterEqual(len(cells), 36)
        self._update()
        before_curves = self._snap()
        before_text = [c.text() for c in cells]
        eps_shown = _parse_complex(cells[0].text())
        mode.setCurrentIndex(1)  # -> refractive index
        self.app.processEvents()
        n_shown = _parse_complex(cells[0].text())
        # the cells render 6 significant digits, so compare RELATIVELY -- an exact-decimal
        # check would fail on display rounding rather than on the conversion
        self.assertLess(abs(abs(n_shown) ** 2 - abs(eps_shown)) / abs(eps_shown), 1e-5,
                        "index mode must display sqrt(eps), i.e. eps = n*n")
        self._update()
        after_curves = self._snap()
        self.assertFalse(self.gh.curves_changed(before_curves, after_curves),
                         "the entry MODE changed the computed physics -- it must only change "
                         "the units the values are typed in")
        mode.setCurrentIndex(0)  # back to permittivity
        self.app.processEvents()
        self.assertLess(abs(abs(_parse_complex(cells[0].text())) - abs(eps_shown)) / abs(eps_shown),
                        1e-5, "switching back must restore the original permittivity")
        # HYGIENE: the round trip re-renders at 6 significant digits, so it leaves ~1e-5 drift
        # in the grids. This class shares ONE window across tests, so restore the exact text --
        # otherwise the drift silently perturbs whichever fence runs next.
        for cell, text in zip(cells, before_text):
            cell.setText(text)
        self.app.processEvents()

    def test_schematic_rays_follow_the_incidence_angle(self):
        """At theta_i = 0
        EVERY arrow in the schematic -- the main fan AND the FB5 multiple-reflection ladders
        (omega, 2-omega, inhomogeneous) -- must be VERTICAL; at oblique incidence the ladder
        passes must actually slant. Legibility offsets may separate parallel arrows but must
        never tilt one."""
        import matplotlib.text as mtext

        from shaarp.shaarp_gui import build_schematic_figure

        def arrow_vectors(theta):
            fig = build_schematic_figure(
                [("air in", None), ("quartz", 121.2), ("Au", 0.0139), ("air out", None)],
                theta_deg=theta, wavelength_um=0.8,
                assumption="Full Multiple Reflections (FMR)",
                fmr_submode="Forward + Backward + Standing waves")
            ax = fig.axes[0]
            vecs = []
            for ch in ax.get_children():
                if isinstance(ch, mtext.Annotation) and ch.arrowprops is not None:
                    (x1, y1), (x0, y0) = ch.xy, ch.xyann
                    vecs.append((x1 - x0, y1 - y0))
            return vecs

        v0 = arrow_vectors(0.0)
        self.assertGreaterEqual(len(v0), 8, "schematic lost its arrows")
        bad = [f"({dx:+.3f},{dy:+.3f})" for dx, dy in v0 if abs(dx) > 1e-6]
        self.assertFalse(bad,
                         "NON-VERTICAL arrows at normal incidence (regression): " + ", ".join(bad))
        v45 = arrow_vectors(45.0)
        slanted = [1 for dx, dy in v45 if abs(dx) > 0.05]
        self.assertGreaterEqual(len(slanted), 6,
                                "arrows do not slant at oblique incidence -- the angle is not wired")

    def test_maker_mode_tracks_inputs_and_other_modes_leave_it_frozen(self):
        """The 'Maker Fringes' FUNCTIONALITY is the one control (the Generate-checkbox is
        gone). In Maker mode the canvas tracks input changes on every Update; in any other mode
        nothing recomputes it (a stale canvas keeps its last curves)."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._select_custom_film_from("Quartz z-cut · 800 nm")
        func = next(c for c in self.ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        thick = self.gh.spins_with_tip(self.ml, "thickness")[0]
        small = [cv for cv in self.ml.findChildren(FigureCanvasQTAgg)
                 if cv.objectName() == "maker_canvas"]
        self.assertTrue(small, "maker canvas not found by name")

        def snap_small():
            out = []
            for cv in small:
                for ax in cv.figure.axes:
                    for ln in ax.lines:
                        out.append(np.asarray(ln.get_ydata(), dtype=float))
            return out

        func.setCurrentText("Maker Fringes")
        thick.setValue(3.0)
        self.app.processEvents()
        self._update()
        base = snap_small()
        self.assertTrue(any(a.size for a in base), "Maker canvas empty in Maker Fringes mode")
        thick.setValue(7.0)
        self.app.processEvents()
        self._update()
        tracked = snap_small()
        self.assertTrue(self.gh.curves_changed(base, tracked),
                        "DEAD MODE: Maker canvas does not track inputs in Maker Fringes mode")
        func.setCurrentText("SHG Simulation")
        thick.setValue(11.0)
        self.app.processEvents()
        self._update()
        frozen = snap_small()
        self.assertFalse(self.gh.curves_changed(tracked, frozen),
                         "Maker canvas recomputed outside its mode (F67: nothing but the "
                         "Maker Fringes functionality drives it)")
