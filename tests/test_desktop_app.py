"""Desktop (PySide6) app smoke tests -- run headlessly via the Qt offscreen platform.

The desktop app is the .exe-packaged shell over the SAME validated compute cores as the
notebook GUI; these tests pin that the window builds, both interface tabs exist, and the
original-documentation-sourced tooltips are wired onto the controls.
"""

import os
import re
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets

    HAVE_QT = True
except ImportError:
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class DesktopAppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_analytical_progress_message_sets_expectation(self):
        """R9 (closed): analytical (CAS) modes must announce that the
        FIRST solve for a configuration can take minutes -- otherwise a slow machine reads the
        symbolic wait as a hang. Non-analytical modes keep the plain computing text."""
        from shaarp.desktop_app import _progress_format_for

        for canon in ("Partial Analytical", "Full Analytical", "Partial Analytical Expressions"):
            msg = _progress_format_for(canon)
            self.assertIn("minutes", msg)
            self.assertIn("cached", msg)
        for canon in ("SHG Simulation", "Maker Fringes", "Fresnel Coefficients", None):
            self.assertEqual(_progress_format_for(canon), "Computing... %p%")

    def test_layer_edit_stays_under_the_example(self):
        """Editing
        a layer field under a NAMED preset keeps the example selected, and the WORKING COPY
        carries the edit — Update computes the modified copy (never the preset's fixed stack
        while the editor shows different numbers)."""
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        ml = win.findChild(QtWidgets.QTabWidget).widget(1)
        preset = next(c for c in ml.findChildren(QtWidgets.QComboBox)
                      if c.findText("Quartz + Au (Fig 4, 800 nm)") >= 0)
        preset.setCurrentText("Quartz + Au (Fig 4, 800 nm)")
        edit = next(c for c in ml.findChildren(QtWidgets.QComboBox)
                    # row 0 is the AMBIENT (unnumbered — the original's convention:
                    # interior media are the numbered layers 1..N). Identify the editor combo
                    # by that role label, never by a "1:" prefix.
                    if c.count() and ("ambient" in c.itemText(0) or "air in" in c.itemText(0)))
        edit.setCurrentIndex(1)
        QtWidgets.QApplication.processEvents()
        thick = next(s for s in ml.findChildren(QtWidgets.QDoubleSpinBox)
                     if "thickness" in s.toolTip().lower() and s.maximum() >= 1000.0)
        thick.setValue(100.0)  # a user edit to the layer editor
        self.assertEqual(preset.currentText(), "Quartz + Au (Fig 4, 800 nm)",
                         "an edit under a named example must STAY under it (F57)")
        from shaarp.layer_stack import decode_stack
        got = decode_stack(ml._ml_stack_payload()["stack"])
        self.assertAlmostEqual(float(got[1]["thickness_um"]), 100.0, places=6,
                               msg="the working copy must carry the user's edit (the F34 "
                                   "silent-preset-compute class)")
        case_grp = next(g for g in ml.findChildren(QtWidgets.QGroupBox)
                        if g.title().startswith("Case Study"))
        self.assertIn("edited", case_grp.title().lower())

    def test_orientation_edit_stays_under_the_case(self):
        """Changing the crystal-orientation controls while a case
        study is selected must flip the case selector to Custom so the user's orientation is what
        gets computed -- previously TaAs (112) + Mode 'z-cut (identity)' kept showing AND computing
        the (112) tilt (the case's stored orientation silently won over the user's choice)."""
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("TaAs (112)") >= 0)
        orient = next(c for c in si.findChildren(QtWidgets.QComboBox)
                      if c.findText("z-cut (identity)") >= 0 and c is not case)
        case.setCurrentText("TaAs (112)")
        # programmatic sync mirrors the case's rotated orientation WITHOUT flipping the case
        self.assertEqual(case.currentText(), "TaAs (112)",
                         "case sync must not be mistaken for a user orientation edit")
        self.assertTrue(orient.currentText().startswith("Crystal Physics Directions"),
                        "TaAs (112) is a rotated cut -- sync should mirror it")
        # USER selects identity: activated fires only on real interaction -> flips to Custom
        orient.setCurrentText("z-cut (identity)")
        orient.activated.emit(orient.currentIndex())
        self.assertEqual(case.currentText(), "TaAs (112)",
                         "F57: a user orientation edit STAYS under the case (edited-marked)")
        case_grp = next(g for g in si.findChildren(QtWidgets.QGroupBox)
                        if g.title().startswith("Case Study"))
        self.assertIn("edited", case_grp.title().lower(),
                      "the case group must mark the example as edited")
        # the panel keeps the case's tensors: d15 cell still carries TaAs 92.2879
        d_texts = [e.text() for e in si.findChildren(QtWidgets.QLineEdit) if "92.28" in e.text()]
        self.assertTrue(d_texts, "flip must keep the case's d values in the panel (case d + user orientation)")
        # re-selecting the case must NOT false-flip back (sync is signal-blocked)
        case.setCurrentText("TaAs (112)")
        self.assertEqual(case.currentText(), "TaAs (112)")

    def test_wavelength_clamp_note_appears_outside_dispersion_grid(self):
        """R2 (closed): a wavelength outside the case-study
        material's exported dispersion grid clamps to the grid end -- the GUI must SAY so
        instead of clamping silently."""
        from shaarp.casestudy_materials import casestudy_lambda_range
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        note = si.findChild(QtWidgets.QLabel, "wl_note_si")
        self.assertIsNotNone(note, "wl_note_si label missing from the SI tab")
        # SI palette label "GaAs (111)" (case-study fidelity audit) resolves to the registry key
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        case.setCurrentText("GaAs (111)")
        from shaarp.desktop_app import TOOLTIPS

        # locate by tooltip (identity), not by decimals+maximum. A value-based locator
        # silently matches NOTHING when the bound moves -- that is how the gui-smoke lost half
        # its coverage this release (14 -> 7) while still reporting success.
        wl = next(s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
                  if s.toolTip() == TOOLTIPS["wavelength"])
        lo, hi = casestudy_lambda_range("GaAs (111) (800 nm)")
        wl.setValue(hi + 5.0)  # beyond the tabulated grid -> clamped
        self.assertTrue(note.isVisibleTo(si), "clamp note not shown for out-of-grid wavelength")
        self.assertIn("clamped", note.text())
        self.assertIn("GaAs (111) (800 nm)", note.text())
        wl.setValue(0.8)  # back inside the grid -> note hidden
        self.assertFalse(note.isVisibleTo(si), "clamp note stuck on for an in-grid wavelength")

    def test_window_builds_with_both_interface_tabs(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertEqual(titles, ["SHAARP.si (single interface)", "SHAARP.ml (multilayer)"])

    def test_tooltips_are_wired_and_docs_sourced(self):
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        # every tooltip text is non-empty, and the wording carries the original docs' phrasing
        self.assertTrue(all(v.strip() for v in TOOLTIPS.values()))
        self.assertIn("point group symmetry are automatically imposed", TOOLTIPS["shg_tensor"])
        self.assertIn("perpendicular", TOOLTIPS["uvw"])
        win = build_main_window()
        tipped = [w for w in win.findChildren(QtWidgets.QWidget) if w.toolTip().strip()]
        self.assertGreater(len(tipped), 40, "most controls must carry tooltips")

    def test_preset_buttons_and_run_button_exist(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        buttons = {b.text() for b in win.findChildren(QtWidgets.QPushButton)}
        for need in ("Update / Run", "Preset 1", "Preset 4", "Clear Presets", "Export data"):
            self.assertIn(need, buttons)

    def test_polarimetry_settings_controls_present(self):
        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        descs = {w.toolTip()[:24] for w in si.findChildren(QtWidgets.QWidget)}
        combos = [c.currentText() for c in si.findChildren(QtWidgets.QComboBox)]
        # ellipticity + analyzer-mode controls exist on the SI tab
        self.assertTrue(any(t.startswith("The incident electric") for t in descs), "ellipticity tip missing")
        analyzer = next((c for c in si.findChildren(QtWidgets.QComboBox)
                         if c.findText("Fix Analyzer") >= 0), None)
        self.assertIsNotNone(analyzer, "analyzer combo missing")
        items = [analyzer.itemText(i) for i in range(analyzer.count())]
        self.assertEqual(items, ["Rotate Analyzer", "Fix Analyzer"],
                         "analyzer combo should be the 2-option Rotating / Fixed (matching ♯SHAARP.si)")

    def test_analyzer_defaults_to_fixed_on_both_tabs(self):
        """Rui: the analyzer starts in 'Fix Analyzer' mode. The dependent controls
        must match: ψ value + quick buttons live, the (Rotating-only) offset field greyed."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)
        for ti, tag in ((0, "SI"), (1, "ML")):
            page = tabs.widget(ti)
            analyzer = next(c for c in page.findChildren(QtWidgets.QComboBox)
                            if c.findText("Fix Analyzer") >= 0)
            self.assertEqual(analyzer.currentText(), "Fix Analyzer",
                             f"{tag}: analyzer must default to Fixed")
            psi = [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                   if s.toolTip() == TOOLTIPS["analyzer"]]
            self.assertTrue(psi and all(s.isEnabled() for s in psi),
                            f"{tag}: fixed-analyzer ψ spin must start enabled")
            offset = [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                      if s.toolTip().startswith("Analyzer–polarizer offset")]
            self.assertTrue(offset and not any(s.isEnabled() for s in offset),
                            f"{tag}: the Rotating-only offset field must start greyed")

    def test_case_selection_syncs_orientation_panel(self):
        """Rui: 'when I select case study, the orientation does not seem to be
        automatically updated'. Selecting a tilted case must leave the orientation panel SHOWING
        that orientation: mode = Crystal Physics Directions, the Z1/Z2/Z3 cells carrying the case's
        rotation matrix, and those cells ENABLED (the signal-blocked mode sync used to skip the
        per-mode enable refresh, leaving the freshly filled cells greyed = 'not updated')."""
        import numpy as np

        from shaarp.casestudy_materials import build_casestudy_material
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("TaAs (112)") >= 0)
        case.setCurrentText("TaAs (112)")
        QtWidgets.QApplication.processEvents()
        orient = next(c for c in si.findChildren(QtWidgets.QComboBox)
                      if c.findText("Miller (hkl + in-plane uvw)") >= 0 and c is not case)
        self.assertTrue(orient.currentText().startswith("Crystal Physics"),
                        "tilted case must switch the orientation mode display")
        z = [s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
             if s.toolTip() == TOOLTIPS["z_axes"] and isinstance(s, QtWidgets.QDoubleSpinBox)]
        z = [s for s in z if s.decimals() == 6][:9]
        self.assertEqual(len(z), 9, "expected the 9 Z1/Z2/Z3 cells")
        self.assertTrue(all(s.isEnabled() for s in z),
                        "Z cells must be ENABLED after a case sync (stale-grey bug)")
        shown = np.array([s.value() for s in z]).reshape(3, 3)
        expected = np.asarray(
            build_casestudy_material("TaAs (112)").orientation.rotation_matrix(), dtype=float)
        self.assertLess(float(np.max(np.abs(shown - expected))), 5e-6,
                        "Z cells must display the case's stored orientation matrix")
        # hkl/uvw stay greyed in Crystal Physics mode (their stale values are visibly inactive)
        hkl = [s for s in si.findChildren(QtWidgets.QSpinBox) if s.toolTip() == TOOLTIPS["hkl"]]
        self.assertTrue(hkl and not any(s.isEnabled() for s in hkl),
                        "hkl cells must be greyed while a Crystal-Physics case is shown")

    def test_si_curve_ellipticity_and_fixed_analyzer(self):
        import numpy as np

        from shaarp.shaarp_gui import si_polarimetry_curve

        kw = dict(theta_deg=45.0, n_omega=2.2, n_2omega=2.3,
                  d_free={(0, 3): 1.0, (1, 1): 1.0, (2, 0): 1.0, (2, 2): 1.0})
        base = si_polarimetry_curve("3m", **kw)
        ell = si_polarimetry_curve("3m", ellipticity_deg=45.0, **kw)
        self.assertGreater(float(np.max(np.abs(np.array(base["intensity_p"]) - np.array(ell["intensity_p"])))), 0.0)
        ana = si_polarimetry_curve("3m", analyzer_deg=30.0, **kw)
        self.assertIn("intensity_analyzed", ana)
        self.assertEqual(ana["analyzer_deg"], 30.0)

    def test_branding_header_present_with_logo_and_authors(self):
        from PySide6 import QtGui

        from shaarp.desktop_app import BRANDING_HTML, _load_logo_pixmap, build_main_window

        # the bundled #SHAARP.ml logo loads, and the banner names the SHAARP authors + acknowledgment
        pix = _load_logo_pixmap(QtGui)
        self.assertIsNotNone(pix, "bundled #SHAARP.ml logo failed to load")
        for need in ("SHAARP.py", "Gopalan", "acknowledge"):
            self.assertIn(need, BRANDING_HTML)
        win = build_main_window()
        labels = [w.text() for w in win.findChildren(QtWidgets.QLabel) if w.text()]
        self.assertTrue(any("Anisotropic Rotational Polarimetry" in t for t in labels))

    def test_help_menu_has_user_guide(self):
        from shaarp.desktop_app import USER_GUIDE_HTML, build_main_window

        win = build_main_window()
        help_actions = []
        for m in win.menuBar().findChildren(QtWidgets.QMenu):
            if m.title() == "Help":
                help_actions = [a.text() for a in m.actions()]
        self.assertIn("User Guide", help_actions)
        for need in ("Functionality", "Polarimetry", "Assumptions"):
            self.assertIn(need, USER_GUIDE_HTML)

    def test_fmr_submodes_distinct_and_present(self):
        import numpy as np

        from shaarp.desktop_app import build_main_window
        from shaarp.layer_stack import build_system_from_stack, default_stack
        from shaarp.shaarp_gui import FMR_SUBMODES, compute_ml_gui_result

        sysm = build_system_from_stack(default_stack(), wavelength_um=1.064)
        curves = {}
        for sub in FMR_SUBMODES:  # faithful labels: Forward waves only / + Backward / + Standing
            r = compute_ml_gui_result("Maker Fringes", theta_min_deg=0, theta_max_deg=30,
                                      theta_step_deg=5, assumption="Full Multiple Reflections (FMR)",
                                      fmr_submode=sub, system=sysm)
            curves[sub] = np.asarray(r.numeric["parallel_intensity"], float)
        # backward waves change the result vs the validated forward-only default
        self.assertGreater(float(np.max(np.abs(
            curves["Forward waves only"] - curves["Forward + Backward waves"]))), 1e-6)
        # legacy short labels still accepted (back-compat aliases)
        r_alias = compute_ml_gui_result("Maker Fringes", theta_min_deg=0, theta_max_deg=30,
                                        theta_step_deg=5, assumption="Full (FMR)",
                                        fmr_submode="No backward waves", system=sysm)
        self.assertEqual(len(r_alias.numeric["parallel_intensity"]), len(curves["Forward waves only"]))
        win = build_main_window()
        ml = win.findChild(QtWidgets.QTabWidget).widget(1)
        fc = next((c for c in ml.findChildren(QtWidgets.QComboBox)
                   if "Forward + Backward + Standing waves" in [c.itemText(i) for i in range(c.count())]), None)
        self.assertIsNotNone(fc, "FMR backward/standing sub-mode dropdown missing")

    def test_full_analytical_shows_collapsible_derivation_steps(self):
        """A Full-Analytical run populates collapsible Step 1/2/3 group boxes above the
        final expression box; a numeric run clears them."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)
        page = tabs.widget(0)
        func = next(c for c in page.findChildren(W.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        run = next(b for b in page.findChildren(W.QPushButton) if b.text() == "Update / Run")
        func.setCurrentText("Full Analytical Expression"); run.click(); self.app.processEvents()
        steps = [b for b in page.findChildren(W.QGroupBox) if b.isCheckable() and b.title().startswith("Step ")]
        # Step 0 (the layered definition chain, FA-1 Stage 2) + the three published stages
        self.assertEqual(len(steps), 4, "expected 4 derivation-step boxes for Full Analytical")
        self.assertTrue(all(not b.isChecked() for b in steps), "steps should start collapsed")
        func.setCurrentText("SHG Simulation"); run.click(); self.app.processEvents()
        steps2 = [b for b in page.findChildren(W.QGroupBox) if b.isCheckable() and b.title().startswith("Step ")]
        self.assertEqual(len(steps2), 0, "numeric run must clear the derivation steps")

    def test_run_click_drives_si_analytical_and_ml_maker(self):
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)

        def drive(tab_index, functionality):
            page = tabs.widget(tab_index)
            func = next(c for c in page.findChildren(W.QComboBox)
                        if c.toolTip().startswith("Choose what to calculate"))
            func.setCurrentText(functionality)
            next(b for b in page.findChildren(W.QPushButton) if b.text() == "Update / Run").click()
            self.app.processEvents()
            expr = page.findChild(W.QTextEdit, "expr_box")  # rich-text typeset view
            status = [L for L in page.findChildren(W.QLabel)
                      if L.text().startswith("Validation:")][0]
            # the label now shows HUMAN wording; the raw workflow tag lives in its tooltip
            return status.toolTip(), expr

        s, expr = drive(0, "Full Analytical Expression")  # faithful display label
        self.assertIn("si_full_analytical_polarimetry", s)
        raw = expr.property("raw_text") or ""
        self.assertIn("d14", raw, "machine-readable closed form must populate (Copy/export layer)")
        # the DISPLAY is typeset -- Greek + real sub/superscripts, no raw sympy tokens
        shown = expr.toPlainText()
        self.assertIn("θ", shown, "typeset view must show Greek theta")
        self.assertNotIn("**", shown, "typeset view must not show raw ** powers")
        # the WHOLE display is typeset now (incl. the symbols provenance line); raw names
        # live only in the Copy/export layer
        self.assertNotIn("theta_i", shown, "typeset view must not show raw symbol names")
        self.assertNotIn("n_omega", shown, "typeset view must not show raw symbol names")
        s, expr = drive(1, "Maker Fringes")
        self.assertIn("maker", s)
        self.assertEqual(expr.toPlainText(), "", "numeric run must clear the closed-form box")

    def test_progress_time_collapsible_and_update_buttons(self):
        """Visual-parity controls vs the original GUI: top progress bar, 'Time Used' readout,
        collapsible sub-panels, and Update buttons at the original's three locations."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        # top progress bar (the original's "...% Completed" bar)
        bar = win.findChild(W.QProgressBar)
        self.assertIsNotNone(bar, "top progress bar missing")
        # idle state reads "Ready" before any Update (becomes "100% Completed" after a run)
        self.assertEqual(bar.format(), "Ready")
        # 'Time Used' readout, one per tab (the original's output 'Time Used' line)
        labels = [L.text() for L in win.findChildren(W.QLabel)]
        self.assertGreaterEqual(sum(t.startswith("Time Used") for t in labels), 2)
        # Update at the original's three places: global top + input-bottom + output-bottom
        btn_texts = [b.text() for b in win.findChildren(W.QPushButton)]
        self.assertIn("Update / Run", btn_texts)              # input-panel bottom
        self.assertGreaterEqual(btn_texts.count("Update"), 3)  # 1 global header + 1 output per tab
        # collapsible sub-panels: the input group boxes are checkable (expand/collapse)
        groups = [g for g in win.findChildren(W.QGroupBox) if g.isCheckable()]
        self.assertGreaterEqual(len(groups), 12, "input sub-panels should be collapsible")

    def test_global_top_update_recomputes_active_tab(self):
        """The global top Update button (in the header, outside the tabs) recomputes the
        active tab and the 'Time Used' readout populates."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        tabs = win.findChild(W.QTabWidget)
        tabs.setCurrentIndex(1)  # ML tab
        ml = tabs.widget(1)
        func = next(c for c in ml.findChildren(W.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText("Maker Fringes")
        header_update = next(b for b in win.findChildren(W.QPushButton)
                             if b.text() == "Update" and not tabs.isAncestorOf(b))
        header_update.click()
        self.app.processEvents()
        status = [L for L in ml.findChildren(W.QLabel) if L.text().startswith("Validation:")][0]
        self.assertIn("maker", status.toolTip())
        tused = [L for L in ml.findChildren(W.QLabel) if L.text().startswith("Time Used")][0]
        self.assertNotIn("-- s", tused.text(), "Time Used must populate after a compute")

    def test_exactly_one_live_ellipticity_control_per_mode(self):
        """Maker Fringes has its own -90..90 Delta-delta, so in that mode BOTH
        ellipticity boxes were enabled and editable while the compute path read only the Maker one
        -- a second live input for one physical quantity, silently ignored. House rule R-B is ONE
        control per concept: the Maker box owns Delta-delta in Maker Fringes, the general box owns
        it everywhere else, and never both at once."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        ml = win.findChild(QtWidgets.QTabWidget).widget(1)
        func = next(c for c in ml.findChildren(W.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        general = [x for x in ml.findChildren(W.QDoubleSpinBox)
                   if x.toolTip() == TOOLTIPS["ellipticity"]]
        maker = [x for x in ml.findChildren(W.QDoubleSpinBox)
                 if x.toolTip() == TOOLTIPS["maker_ellipticity"]]
        self.assertEqual((len(general), len(maker)), (1, 1))
        general, maker = general[0], maker[0]
        for mode, want_general, want_maker in (
                ("SHG Simulation", True, False),
                ("Maker Fringes", False, True),
                ("Partial Analytical Expressions", True, False),
                ("Fresnel Coefficients", False, False)):
            func.setCurrentText(mode)
            self.app.processEvents()
            self.assertEqual(general.isEnabled(), want_general,
                             f"{mode}: general Δδ enabled-state wrong")
            self.assertEqual(maker.isEnabled(), want_maker,
                             f"{mode}: Maker Δδ enabled-state wrong")
            self.assertFalse(general.isEnabled() and maker.isEnabled(),
                             f"{mode}: TWO live inputs for one quantity (R-B duplication)")
        func.setCurrentText("SHG Simulation")
        self.app.processEvents()

    def test_ps_convention_note_is_visible_on_both_tabs(self):
        """The original .ml prints the p/s convention as a VISIBLE row in the polarimetry
        column. Ours was tooltip-only -- hover-only for the convention that decides whether a
        reported number means p- or s-polarized.

        Fenced because the gate's own review found this row shipping with nothing asserting it:
        gui_harness has no QLabel coverage rule, so a label row is invisible to the coverage
        ratchet and could be deleted with every check still green."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)
        for idx, tab in ((0, "SI"), (1, "ML")):
            page = tabs.widget(idx)
            hits = [L for L in page.findChildren(W.QLabel)
                    if "p-polarized" in L.text() and "s-polarized" in L.text()]
            self.assertTrue(hits, f"{tab}: the p/s convention must be VISIBLE, not tooltip-only")
            txt = hits[0].text()
            self.assertIn("0", txt, f"{tab}: the note must state the 0° convention")
            self.assertIn("90", txt, f"{tab}: the note must state the 90° convention")
            self.assertTrue(hits[0].isVisibleTo(page), f"{tab}: the p/s note must not be hidden")

    def test_orientation_view_toggle_switches_the_figure(self):
        """The crystal-axes view gains a 2D companion, default 3D.

        The view was 3D-only; 2D looks down the surface normal, which reads an in-plane azimuth
        far better. Pinned here because a toggle whose two states are never both exercised is the
        same dormant-check class this project keeps rediscovering."""
        from PySide6 import QtWidgets as W

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)
        for idx, tab in ((0, "si"), (1, "ml")):
            page = tabs.widget(idx)
            combo = [c for c in page.findChildren(W.QComboBox)
                     if c.toolTip() == TOOLTIPS["orientation_view"]]
            self.assertEqual(len(combo), 1, f"{tab}: expected exactly one orientation-view combo")
            combo = combo[0]
            self.assertEqual(combo.currentText(), "3D", f"{tab}: must default to 3D")
            canvas = page.findChild(FigureCanvasQTAgg, f"orient_canvas_{tab}")
            self.assertIsNotNone(canvas)
            combo.setCurrentText("2D (top view)")
            self.app.processEvents()
            self.assertEqual(canvas.figure.axes[0].name, "rectilinear",
                             f"{tab}: 2D must render a flat axes")
            combo.setCurrentText("3D")
            self.app.processEvents()
            self.assertEqual(canvas.figure.axes[0].name, "3d",
                             f"{tab}: 3D must render an mplot3d axes")

    def test_panel_defaults_match_rui_directive(self):
        """Rui: "default at normal incidence for all cases. for both si and ml, default
        analyzer to fixed and rotating polarizer, fix sample rotation."

        Nothing asserted these before, so the directive was enforced by nothing -- and the
        incidence default is load-bearing: the gui-smoke derived its angle set from it, so moving
        it silently halved that sweep's coverage. Pin all four on both tabs."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import TOOLTIPS, build_main_window

        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)
        for idx, tab in ((0, "SI"), (1, "ML")):
            page = tabs.widget(idx)

            def combo(item):
                return next((c for c in page.findChildren(W.QComboBox)
                             if c.findText(item) >= 0), None)

            theta = [t for t in page.findChildren(W.QDoubleSpinBox)
                     if t.toolTip() == TOOLTIPS["theta"]]
            self.assertEqual(len(theta), 1, f"{tab}: expected exactly one incidence spin")
            self.assertEqual(theta[0].value(), 0.0,
                             f"{tab}: must default to NORMAL INCIDENCE")
            self.assertEqual(combo("Rotate Polarizer").currentText(), "Rotate Polarizer",
                             f"{tab}: polarizer must default to ROTATING")
            self.assertEqual(combo("Rotate Analyzer").currentText(), "Fix Analyzer",
                             f"{tab}: analyzer must default to FIXED")
            sample = combo("Rotate Sample")
            if sample is not None:  # ML only -- .si has no sample-rotation control
                self.assertEqual(sample.currentText(), "Fix Sample",
                                 f"{tab}: sample rotation must default to FIXED")

    def test_faithful_functionality_labels_and_view_modes(self):
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        tabs = win.findChild(QtWidgets.QTabWidget)
        si, ml = tabs.widget(0), tabs.widget(1)

        def func_combo(page):
            return next(c for c in page.findChildren(W.QComboBox)
                        if c.toolTip().startswith("Choose what to calculate"))

        # the Functionality dropdown lists COMPUTE MODES ONLY. The original's
        # view-only entries (User Guide / Set Material Properties / 2D|3D Schematics) were removed as
        # redundant -- the Guide is on the Help menu + startup tab, the schematic is a persistent
        # banner, and the crystal-axes (Zi vs Li) view moved next to the orientation inputs. Exact
        # lists (order + content) pin the contract so a regression that re-adds a view mode fails here.
        cs = func_combo(si)
        si_items = [cs.itemText(i) for i in range(cs.count())]
        self.assertEqual(
            si_items,
            ["SHG Simulation", "Partial Analytical Expression", "Full Analytical Expression"])
        cm = func_combo(ml)
        ml_items = [cm.itemText(i) for i in range(cm.count())]
        self.assertEqual(
            ml_items,
            ["SHG Simulation", "Maker Fringes", "Fresnel Coefficients",
             "Partial Analytical Expressions"])  # exactly 4; RA is a Polarimetry toggle
        # never call it bare "Maker" -- every user-visible group title,
        # tab label, and combo item spells "Maker Fringes" in full.
        _bare = re.compile(r"\bMaker\b(?!\s+[Ff]ringes)")
        for page in (si, ml):
            for w in page.findChildren(W.QGroupBox) + page.findChildren(W.QLabel):
                txt = w.title() if isinstance(w, W.QGroupBox) else w.text()
                self.assertIsNone(_bare.search(txt or ""),
                                  f"bare 'Maker' in user-visible text: {txt!r}")
        for tw in win.findChildren(W.QTabWidget):
            for i in range(tw.count()):
                self.assertIsNone(_bare.search(tw.tabText(i)),
                                  f"bare 'Maker' tab label: {tw.tabText(i)!r}")
        for gone in ("User Guide", "Set Material Properties", "2D Schematics", "3D Schematics"):
            self.assertNotIn(gone, si_items)
            self.assertNotIn(gone, ml_items)
        # the crystal-axes orientation view now lives in the input panel, by an objectName the
        # test can find; the always-on 3D "sample stack" output pane is gone.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.assertIsNotNone(si.findChild(FigureCanvasQTAgg, "orient_canvas_si"),
                             "SI orientation view must live in the input panel (F38)")
        self.assertIsNotNone(ml.findChild(FigureCanvasQTAgg, "orient_canvas_ml"),
                             "ML orientation view must live in the input panel (F38)")

    def test_full_tensor_entry_controls(self):
        """The tensor entry is the Mathematica full-matrix layout: ε(ω) and ε(2ω) as 3×3 SYMMETRIC
        grids (ε_ij = ε_ji) and d as the full 3×6 Voigt matrix (all components editable). These are
        the primary, always-on entry (no 'use full' checkbox)."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import _parse_complex, _wire_symmetric_grid, build_main_window

        self.assertEqual(_parse_complex("6.26+18.81 I"), complex(6.26, 18.81))  # Mathematica 'I'
        self.assertEqual(_parse_complex("2+3j"), complex(2, 3))
        self.assertEqual(_parse_complex(""), 0j)
        win = build_main_window()
        si = win.findChild(W.QTabWidget).widget(0)
        titles = [g.title() for g in si.findChildren(W.QGroupBox)]
        self.assertTrue(any("Dielectric Tensors" in t and "symmetric" in t for t in titles),
                        "full symmetric dielectric ε matrices missing")
        self.assertTrue(any("SHG Tensor" in t and "Voigt" in t for t in titles),
                        "full 3×6 Voigt d matrix missing")
        # 3x3 eps(w) + 3x3 eps(2w) + 3x6 d = 24 complex matrix line-edits (plus other fields)
        self.assertGreaterEqual(len(si.findChildren(W.QLineEdit)), 24)
        # ε symmetry constraint: editing one off-diagonal cell mirrors its transpose partner.
        cells = [[W.QLineEdit("0") for _ in range(3)] for _ in range(3)]
        _wire_symmetric_grid(cells)
        cells[0][2].setText("0.7")
        self.assertEqual(cells[2][0].text(), "0.7")

    def test_guide_tab_references_and_scrollable_plots(self):
        """The output lands on an instruction 'Guide' tab at startup (carrying the SHAARP references +
        DOE acknowledgment), and each plot tab is wrapped in a QScrollArea so the full multi-tile
        figure is navigable (auto-fits width, scrolls vertically)."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import USER_GUIDE_HTML, build_main_window

        self.assertIn("npj Comput", USER_GUIDE_HTML)        # how-to-cite references present
        self.assertIn("DE-SC0020145", USER_GUIDE_HTML)      # DOE acknowledgment present
        self.assertIn("s41524-022-00930-4", USER_GUIDE_HTML)  # the .si paper DOI
        win = build_main_window()
        si = win.findChild(W.QTabWidget).widget(0)
        ot = next(t for t in si.findChildren(W.QTabWidget))
        labels = [ot.tabText(i) for i in range(ot.count())]
        self.assertIn("Guide", labels)
        self.assertEqual(ot.tabText(ot.currentIndex()), "Guide")  # startup = instruction page
        self.assertIsInstance(ot.widget(labels.index("Polar Plots")), W.QScrollArea)

    def test_analytical_mode_switches_to_expression_tab(self):
        """Analytical modes surface the closed form in its own 'Analytical Expression' output tab;
        sim modes return to the Polar Plots tab (so the expression is never buried below the fold)."""
        from PySide6 import QtWidgets as W

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        ot = next(t for t in si.findChildren(W.QTabWidget))
        self.assertIn("Analytical Expression", [ot.tabText(i) for i in range(ot.count())])
        func = next(c for c in si.findChildren(W.QComboBox) if c.toolTip().startswith("Choose what to calculate"))
        run = lambda: (next(b for b in si.findChildren(W.QPushButton) if b.text() == "Update / Run").click(),
                       self.app.processEvents())
        func.setCurrentText("Full Analytical Expression"); run()
        self.assertEqual(ot.tabText(ot.currentIndex()), "Analytical Expression")
        func.setCurrentText("SHG Simulation"); run()
        self.assertEqual(ot.tabText(ot.currentIndex()), "Polar Plots")

    def test_schematic_canvases_sized_for_full_render(self):
        """The output banner carries ONE compact 2D optical-setup schematic
        that fills the width; the crystal-axes (Zi vs Li) orientation view moved into the input
        panel and is a compact 3D mplot3d canvas width-capped near-square (<=480) so it renders the
        whole box + labels (a wide-short 3D canvas clipped them). The window minimum width must also
        stay small enough that the geometry/output panel is never clipped on a normal laptop."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        # the orientation view lives in the input panel now, width-capped near-square
        orient = si.findChild(FigureCanvasQTAgg, "orient_canvas_si")
        self.assertIsNotNone(orient, "orientation view canvas missing from the input panel (F38)")
        self.assertLessEqual(orient.maximumWidth(), 480,
                             "orientation view (3D) must be width-capped near-square to render fully")
        self.assertGreaterEqual(orient.minimumHeight(), 240, "orientation view should be compact")
        # compact context canvases: the 2D banner + the orientation view
        schem = [c for c in si.findChildren(FigureCanvasQTAgg) if 240 <= c.minimumHeight() <= 320]
        self.assertGreaterEqual(len(schem), 2, "SI tab needs the 2D banner + the orientation view")
        # The window must not force a huge minimum width (the non-wrapping banner once pinned it ~1388,
        # clipping the geometry/output panel on smaller laptops). Keep it laptop-friendly.
        # FONT NORMALIZATION: the offscreen Qt platform's Windows font-database lookup
        # can fail machine-wide (QFontInfo resolves an EMPTY family; every string measures ~2x wide),
        # inflating ALL minimum-size hints. The laptop-fit intent is a bound at NORMAL font metrics,
        # so scale the pixel bound by the measured advance of a reference button label relative to
        # its normal-resolution width (~200 px at Segoe UI 9). scale == 1 on a healthy font stack.
        from PySide6 import QtGui
        ref_advance = QtGui.QFontMetrics(win.font()).horizontalAdvance(
            "Copy closed form (Python/SymPy)")
        scale = max(1.0, ref_advance / 200.0)
        self.assertLessEqual(win.minimumSizeHint().width(), 1100 * scale,
                             f"window min width must stay laptop-friendly (font scale {scale:.2f})")

    def test_startup_guidance_and_ready_progress(self):
        """Ease-of-use: before the first Update the result canvas shows a 'Getting started' hint (so a
        first-time user knows what to do), and the idle progress bar reads 'Ready' rather than a
        confusing '100% Completed' before anything has run."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from shaarp.desktop_app import build_main_window

        win = build_main_window()
        bars = win.findChildren(QtWidgets.QProgressBar)
        self.assertTrue(any(b.format() == "Ready" for b in bars),
                        "idle progress should read 'Ready'")
        ml = win.findChild(QtWidgets.QTabWidget).widget(1)
        texts = [t.get_text()
                 for c in ml.findChildren(FigureCanvasQTAgg)
                 for ax in c.figure.axes for t in ax.texts]
        self.assertTrue(any("Getting started" in t for t in texts),
                        "startup result canvas must show a getting-started hint")

    def test_faithful_assumption_labels(self):
        from shaarp.shaarp_gui import FMR_SUBMODES, ML_ASSUMPTIONS

        for need in ("Full Multiple Reflections (FMR)", "Jerphagnon & Kurtz Assumption (No MR)",
                     "Herman & Hayden Assumption (MR only for 2ω Homo Waves)"):
            self.assertIn(need, ML_ASSUMPTIONS)
        for need in ("Forward waves only", "Forward + Backward waves", "Forward + Backward + Standing waves"):
            self.assertIn(need, FMR_SUBMODES)


if __name__ == "__main__":
    unittest.main()
