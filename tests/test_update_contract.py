"""F49 tail fences: the Update contract + stale banner (A1), re-entry guard (C3), auto-session (C5).

The user's core complaint (grill-2 Q1) was "links not dynamically updated / plots not updated based on
set parameters." Root cause: the orientation triad refreshed live while the schematic + plots only
refreshed on Update, so mid-edit the screen was half-live and read as broken. The decided contract is
POST-UPDATE COMPLETENESS: after an Update every display equals a function of the current inputs, and a
stale banner makes the "you changed inputs, press Update" boundary explicit. These fences pin:

  * A1 changing a compute-relevant input raises the banner; a completed Update clears it;
  * Update contract after Update the schematic angle readout equals the incidence spin;
  * C3 the Update buttons are disabled during compute and re-armed after (and a re-entrant Update
        is a no-op);
  * C5 the auto-session round-trips through the fixed on-disk path (save -> fresh window -> restore).
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False


def _si_page(win):
    return win.findChild(QtWidgets.QTabWidget).widget(0)


def _theta_spin(page):
    # locate by tooltip (identity), not by maximum() -- the SI max moved 89 -> 89.9.
    from shaarp.desktop_app import TOOLTIPS

    return next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                if s.toolTip() == TOOLTIPS["theta"])


def _case_combo(page, needle="GaAs (111)"):
    return next(c for c in page.findChildren(QtWidgets.QComboBox) if c.findText(needle) >= 0)


def _schematic_angle_texts(page):
    import matplotlib.text as mtext
    fig = page._schematic_canvas.figure
    return [t.get_text() for t in fig.findobj(mtext.Text)
            if t.get_text() and "°" in t.get_text()]  # the angle label carries the degree sign


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class UpdateContractAndStaleBanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()

    def test_stale_banner_raises_on_input_change_and_clears_on_update(self):
        from . import gui_harness as gh

        si = _si_page(self.win)
        _case_combo(si).setCurrentText("GaAs (111)")
        self.app.processEvents()
        gh.click_update(si, self.app)
        # isHidden() (not isVisible()) is the offscreen-safe check: without a top-level show() every
        # child reports isVisible()==False, but isHidden() reflects the explicit setVisible state.
        self.assertFalse(si._is_stale(), "banner should be clear right after an Update")
        self.assertTrue(si._stale_banner.isHidden())
        # a compute-relevant input change flips it stale
        _theta_spin(si).setValue(31.0)
        self.app.processEvents()
        self.assertTrue(si._is_stale(), "changing the incidence angle must raise the stale banner")
        self.assertFalse(si._stale_banner.isHidden())
        # the next Update clears it again
        gh.click_update(si, self.app)
        self.assertFalse(si._is_stale(), "a completed Update must clear the stale banner")
        self.assertTrue(si._stale_banner.isHidden())

    def test_update_syncs_the_schematic_angle_readout(self):
        """The post-Update contract, concretely: the schematic's incidence-angle readout equals the
        incidence spin. (This is exactly the class that read as 'links not updating' when the schematic
        lagged the live triad.)"""
        from . import gui_harness as gh

        si = _si_page(self.win)
        _case_combo(si).setCurrentText("GaAs (111)")
        theta = _theta_spin(si)
        for val in (22.0, 58.0):
            theta.setValue(val)
            self.app.processEvents()
            gh.click_update(si, self.app)
            texts = _schematic_angle_texts(si)
            self.assertTrue(any(str(int(val)) in t for t in texts),
                            f"schematic angle readout did not reflect theta={val}: {texts}")

    def test_update_buttons_rearm_after_a_run(self):
        """C3: the guard disables the Update buttons during compute and re-enables them after, so the
        window is never left with a dead Update button."""
        from . import gui_harness as gh

        si = _si_page(self.win)
        _case_combo(si).setCurrentText("GaAs (111)")
        self.app.processEvents()
        gh.click_update(si, self.app)
        runs = [b for b in si.findChildren(QtWidgets.QPushButton)
                if b.text() in ("Update", "Update / Run")]
        self.assertTrue(runs)
        for b in runs:
            self.assertTrue(b.isEnabled(), "an Update button was left disabled after the run finished")

    def test_reentrant_update_is_a_noop(self):
        """C3: while an Update is in flight (state['running'] set) a second Update must early-return,
        not re-enter the solver. Proven by: with the running flag set, changing theta and clicking
        Update leaves the curve unchanged (the guard skipped the recompute)."""
        from . import gui_harness as gh

        si = _si_page(self.win)
        _case_combo(si).setCurrentText("GaAs (111)")
        self.app.processEvents()
        gh.click_update(si, self.app)
        first = gh.snapshot_curves(si)
        si._state["running"] = True  # pretend a compute is already running
        try:
            _theta_spin(si).setValue(_theta_spin(si).value() + 15.0)  # a change that WOULD move the curve
            run_btn = next(b for b in si.findChildren(QtWidgets.QPushButton)
                           if b.text() in ("Update", "Update / Run"))
            run_btn.click()
            self.app.processEvents()
            second = gh.snapshot_curves(si)
            self.assertFalse(gh.curves_changed(first, second),
                             "re-entrant Update recomputed despite the running guard")
        finally:
            si._state["running"] = False


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class AutoSessionRoundTrip(unittest.TestCase):
    def test_autosession_persists_and_restores_from_disk(self):
        from shaarp.desktop_app import (_autosave_session, _last_session_path,
                                        _restore_last_session, build_main_window)

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        si = _si_page(win)
        _case_combo(si, "TaAs (112)").setCurrentText("TaAs (112)")
        _theta_spin(si).setValue(33.0)  # the incidence spin is integer-degree; use a whole value
        app.processEvents()

        path = _last_session_path()
        if os.path.exists(path):
            os.remove(path)
        _autosave_session(win)
        self.assertTrue(os.path.exists(path), "auto-session file was not written")

        # a fresh window starts at defaults (build_main_window never auto-restores -- only main() does)
        win2 = build_main_window()
        si2 = _si_page(win2)
        case2 = _case_combo(si2, "TaAs (112)")
        theta2 = _theta_spin(si2)
        self.assertNotEqual(case2.currentText(), "TaAs (112)",
                            "fresh window unexpectedly started on TaAs -- default-state assumption broke")
        n = _restore_last_session(win2)
        self.assertGreater(n, 100, "restore reported suspiciously few inputs")
        self.assertEqual(case2.currentText(), "TaAs (112)", "case selection did not restore")
        self.assertAlmostEqual(theta2.value(), 33.0, places=3, msg="incidence angle did not restore")
        try:
            os.remove(path)  # tidy; harmless if it lingers (tests never auto-restore)
        except OSError:
            pass

    def test_corrupt_autosession_never_blocks_launch(self):
        from shaarp.desktop_app import (_last_session_path, _restore_last_session,
                                        build_main_window)

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with open(_last_session_path(), "w", encoding="utf-8") as fh:
            fh.write("{ this is not valid json ]")
        win = build_main_window()
        self.assertEqual(_restore_last_session(win), 0,
                         "a corrupt auto-session must restore nothing (and not raise)")
        try:
            os.remove(_last_session_path())
        except OSError:
            pass


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class MaterialLibrary(unittest.TestCase):
    """A personal, named, on-disk material library so a custom material survives across
    sessions. Reuses the validated session collect/apply machinery (no physics path), so the fence is
    a save -> fresh-window -> load round-trip by name, plus delete."""

    NAME = "unittest-tmp-material"

    def tearDown(self):
        from shaarp.desktop_app import _delete_library_material
        _delete_library_material(self.NAME)

    def test_named_material_round_trips_and_deletes(self):
        from shaarp.desktop_app import (_apply_library_material, _delete_library_material,
                                        _load_material_library, _save_material_to_library,
                                        build_main_window)

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        si = _si_page(win)
        _case_combo(si, "TaAs (112)").setCurrentText("TaAs (112)")
        _theta_spin(si).setValue(41.0)
        app.processEvents()

        _delete_library_material(self.NAME)  # start clean
        self.assertEqual(_save_material_to_library(win, self.NAME), self.NAME)
        self.assertIn(self.NAME, _load_material_library(), "saved material not listed in the library")

        # a fresh window starts at defaults; loading the named material must bring the setup back
        win2 = build_main_window()
        si2 = _si_page(win2)
        case2 = _case_combo(si2, "TaAs (112)")
        theta2 = _theta_spin(si2)
        self.assertNotEqual(case2.currentText(), "TaAs (112)")
        n = _apply_library_material(win2, self.NAME)
        self.assertGreater(n, 100, "library apply reported suspiciously few inputs")
        self.assertEqual(case2.currentText(), "TaAs (112)", "material case did not restore from library")
        self.assertAlmostEqual(theta2.value(), 41.0, places=3)

        # delete removes it from the library
        self.assertTrue(_delete_library_material(self.NAME))
        self.assertNotIn(self.NAME, _load_material_library())

    def test_empty_name_is_rejected_and_missing_load_is_safe(self):
        from shaarp.desktop_app import (_apply_library_material, _save_material_to_library,
                                        build_main_window)

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        with self.assertRaises(ValueError):
            _save_material_to_library(win, "   ")
        self.assertEqual(_apply_library_material(win, "no-such-material-xyz"), 0,
                         "loading a missing library entry must be a safe no-op")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)


class EnablednessContract(unittest.TestCase):
    """RELEVANCE CONTRACT -- the class of defect found by using the app.

    He reported that selecting a case study left the orientation panel greyed out ("the orientation
    does not seem to be automatically updated"). Every gate passed, because the greyed cells held
    the CORRECT VALUES -- test_update_contract checked values and visibility, never ENABLEDNESS.
    (Root cause, _sync_case_panel sets the mode combo with signals BLOCKED, so the
    currentTextChanged hook that re-enables the mode's own fields never ran.)

    The invariant asserted here: after ANY programmatic change of the orientation mode -- including
    the signal-blocked path a case-study selection takes -- every input belonging to the ACTIVE
    mode must be enabled, and the inputs belonging to the other modes must not be. A control that
    shows the right number but cannot be edited is a defect, not a display."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _pump(self, n=6):
        for _ in range(n):
            self.app.processEvents()

    @staticmethod
    def _orient_controls(page):
        """The mode combo plus the Miller (hkl/uvw) and Crystal-Physics (Z1..Z3) input groups.

        The combo is found STRUCTURALLY (the only one whose items are ORIENTATION_MODES) rather
        than by tooltip prose, so rewording a tooltip cannot silently empty this selector and turn
        the test green. The inputs are spin boxes, not line edits, and are matched on the distinctive
        phrases of their own tooltip keys ("hkl"/"uvw"/"z_axes")."""
        from shaarp.shaarp_gui import ORIENTATION_MODES
        combo = next(c for c in page.findChildren(QtWidgets.QComboBox)
                     if [c.itemText(i) for i in range(c.count())] == list(ORIENTATION_MODES))
        edits = {"miller": [], "zmode": []}
        for e in page.findChildren(QtWidgets.QAbstractSpinBox):
            tip = e.toolTip()
            if "Miller indices (hkl)" in tip or "perpendicular to the plane of incidence" in tip:
                edits["miller"].append(e)
            elif "Crystal Physics Directions" in tip:
                edits["zmode"].append(e)
        return combo, edits

    def test_active_orientation_mode_inputs_are_enabled_for_every_mode(self):
        from shaarp.desktop_app import build_main_window
        from shaarp.shaarp_gui import ORIENTATION_MODES
        win = build_main_window()
        win.resize(1366, 768)
        win.show()
        self._pump()
        try:
            page = _si_page(win)
            combo, edits = self._orient_controls(page)
            self.assertTrue(edits["miller"], "no Miller inputs found -- selector is stale")
            self.assertTrue(edits["zmode"], "no crystal-physics inputs found -- selector is stale")
            for mode in ORIENTATION_MODES:
                with self.subTest(mode=mode):
                    combo.setCurrentText(mode)
                    self._pump()
                    active = ("miller" if mode.startswith("Miller")
                              else "zmode" if mode.startswith("Crystal Physics") else None)
                    for group, widgets in edits.items():
                        want = (group == active)
                        for w in widgets:
                            self.assertEqual(
                                w.isEnabled(), want,
                                f"{mode}: {group} input {w.toolTip()[:40]!r} "
                                f"isEnabled={w.isEnabled()}, expected {want}")
        finally:
            win.close()

    def test_case_study_selection_leaves_the_active_mode_editable(self):
        # THE REGRESSION FENCE for a case study sets the orientation mode with signals
        # BLOCKED, so the enabling hook must be invoked explicitly afterwards. Values were always
        # right here; only the enabled state was wrong, which is why value-checking gates passed.
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        win.resize(1366, 768)
        win.show()
        self._pump()
        try:
            page = _si_page(win)
            combo, edits = self._orient_controls(page)
            case = _case_combo(page, "GaAs (111)")
            case.setCurrentText(next(case.itemText(i) for i in range(case.count())
                                     if "GaAs (111)" in case.itemText(i)))
            case.activated.emit(case.currentIndex())
            self._pump()
            mode = combo.currentText()
            active = ("miller" if mode.startswith("Miller")
                      else "zmode" if mode.startswith("Crystal Physics") else None)
            if active is not None:
                for w in edits[active]:
                    self.assertTrue(
                        w.isEnabled(),
                        f"case study selected mode {mode!r} but its input "
                        f"{w.toolTip()[:40]!r} stayed disabled (F51a class)")
        finally:
            win.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class StackModeOwnershipContract(unittest.TestCase):
    """Stack-mode inputs must TELL you who owns them.

    With the "Quartz + Au (Fig 4, 800 nm)" preset selected, the schematic said lambda = 0.8 um while
    the Wavelength Setting spin still showed the user's stale value -- the preset's own wavelength
    drives the compute and the spin is simply ignored, with nothing on screen saying so ("your
    wavelength setting is not dynamically linked to the actual simulation"). Same class for the
    film-thickness spin (dead in preset AND N-layer modes -- "what do you mean by film thickness
    given you are already defining thickness for each layer") and the Substrate group (dead outside
    the 3-layer Custom/Film modes -- "what do you mean by substrate?").

    The contract pinned here, following the relevance-gating decision (hint + collapse, never
    disable -- pre-setting values for the next mode stays possible) and the post-Update display
    contract (every number on screen equals a function of the current inputs):

      * selecting a named preset SYNCS the wavelength/thickness spins to the preset's true values
        and marks their rows "set by the preset";
      * the N-layer editor marks film-thickness "per-layer in the stack editor";
      * the Substrate group carries a not-used hint outside Custom/Film modes and none inside them.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _pump(self, n=6):
        for _ in range(n):
            self.app.processEvents()

    @staticmethod
    def _ml_page(win):
        return win.findChild(QtWidgets.QTabWidget).widget(1)

    @staticmethod
    def _preset_combo(page):
        return next(c for c in page.findChildren(QtWidgets.QComboBox)
                    if c.findText("Custom film (use fields)") >= 0)

    @staticmethod
    def _spin_by_tip(page, phrase):
        return next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                    if phrase in s.toolTip())

    @staticmethod
    def _row_label(page, spin):
        for lay in page.findChildren(QtWidgets.QFormLayout):
            lbl = lay.labelForField(spin)
            if lbl is not None:
                return lbl
        return None

    @staticmethod
    def _editor_widgets(page):
        """(edit-layer combo, layer-thickness spin) — the single-source editor controls."""
        edit = next(c for c in page.findChildren(QtWidgets.QComboBox)
                    # row 0 is the AMBIENT (unnumbered — the original's convention:
                    # interior media are the numbered layers 1..N). Identify the editor combo
                    # by that role label, never by a "1:" prefix.
                    if c.count() and ("ambient" in c.itemText(0) or "air in" in c.itemText(0)))
        th = next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                  if s.maximum() >= 100000 and "Per-layer thickness" in s.toolTip())
        return edit, th

    def test_named_preset_loads_its_real_stack_into_the_editor(self):
        """F55 core (screenshot): the editor showed default LiNbO3/10um while Fig-4
        computed quartz-121.2/Au-0.0139. Selecting a preset must LOAD its real layers; lambda
        syncs on selection, a user edit is kept (dirty model), and re-selecting resets."""
        from shaarp.desktop_app import build_main_window
        from shaarp.shaarp_gui import resolve_ml_system_preset
        win = build_main_window()
        try:
            page = self._ml_page(win)
            combo = self._preset_combo(page)
            wl = self._spin_by_tip(page, "wavelength (in vacuum)")
            preset_name = next(combo.itemText(i) for i in range(combo.count())
                               if "Fig 4" in combo.itemText(i))
            sys_ = resolve_ml_system_preset(preset_name)
            combo.setCurrentText("Custom film (use fields)")  # leave the startup preset first
            self._pump()
            combo.setCurrentText(preset_name)
            self._pump()
            self.assertAlmostEqual(wl.value(), float(sys_.wavelength_um), places=9,
                                   msg="wavelength spin must SYNC to the preset's true lambda")
            wl.setValue(2.064)
            self._pump()
            self.assertAlmostEqual(wl.value(), 2.064, places=9,
                                   msg="F57: a lambda edit under the example is KEPT (dirty), "
                                       "not snapped back")
            combo.activated.emit(combo.currentIndex())  # re-pick the example = reset
            self._pump()
            self.assertAlmostEqual(wl.value(), float(sys_.wavelength_um), places=9,
                                   msg="re-selecting the example must reset lambda to its "
                                       "true value")
            edit, th = self._editor_widgets(page)
            self.assertEqual(edit.count(), len(sys_.layers),
                             "editor must hold exactly the preset's layers")
            expected = [float(L.thickness_um or 0.0) for L in sys_.layers]
            got = []
            for i in range(edit.count()):
                edit.setCurrentIndex(i)
                self._pump(2)
                got.append(float(th.value()))
            self.assertEqual(got, expected,
                             f"editor rows must mirror the preset's true thicknesses, got {got}")
            self.assertIn("set by the preset", self._row_label(page, wl).text())
        finally:
            win.close()

    def test_editing_a_preset_layer_flips_to_editor_with_a_faithful_copy(self):
        """Editing a layer under a named example STAYS under
        it — the working copy carries the edit, the OTHER layers keep the example's true
        values, and lambda remains the example's until the user edits it."""
        from shaarp.desktop_app import build_main_window
        from shaarp.shaarp_gui import resolve_ml_system_preset
        win = build_main_window()
        try:
            page = self._ml_page(win)
            combo = self._preset_combo(page)
            wl = self._spin_by_tip(page, "wavelength (in vacuum)")
            preset_name = next(combo.itemText(i) for i in range(combo.count())
                               if "Fig 4" in combo.itemText(i))
            preset_wl = float(resolve_ml_system_preset(preset_name).wavelength_um)
            combo.setCurrentText("Custom film (use fields)")
            self._pump()
            wl.setValue(1.2)  # the user's own lambda before entering the preset
            self._pump()
            combo.setCurrentText(preset_name)
            self._pump()
            edit, th = self._editor_widgets(page)
            edit.setCurrentIndex(1)  # the quartz layer (121.2 um)
            self._pump(2)
            th.setValue(5.0)  # user edit under the example
            self._pump()
            self.assertEqual(combo.currentText(), preset_name,
                             "F57: an edit under a named example STAYS under it (the F55 "
                             "flip-to-editor is superseded)")
            self.assertAlmostEqual(th.value(), 5.0, places=9, msg="the edited row keeps the edit")
            edit.setCurrentIndex(2)  # the Au layer must still carry the preset's 0.0139
            self._pump(2)
            self.assertAlmostEqual(th.value(), 0.0139, places=9,
                                   msg="the working copy must stay FAITHFUL - other layers keep "
                                       "the example's true values")
            self.assertAlmostEqual(wl.value(), preset_wl, places=9,
                                   msg="lambda stays the example's value while only a layer was "
                                       "edited")
            self.assertNotIn("set by", self._row_label(page, wl).text(),
                             "an edited example no longer claims preset ownership of lambda")
            wl.setValue(1.55)
            self._pump()
            self.assertAlmostEqual(wl.value(), 1.55, places=9,
                                   msg="lambda must be freely editable under the edited example")
        finally:
            win.close()

    def test_simple_modes_own_the_film_row_without_flipping(self):
        """In Custom-film / Film: modes the editor's film row IS the mode's thickness input:
        editing it must update the template and STAY in the mode (flipping out would make the
        simple modes unable to set thickness at all)."""
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        try:
            page = self._ml_page(win)
            combo = self._preset_combo(page)
            combo.setCurrentText("Custom film (use fields)")
            self._pump()
            edit, th = self._editor_widgets(page)
            self.assertEqual(edit.count(), 3, "simple modes are 3-layer templates")
            edit.setCurrentIndex(1)
            self._pump(2)
            th.setValue(2.75)
            self._pump()
            self.assertEqual(combo.currentText(), "Custom film (use fields)",
                             "a thickness edit in a simple mode must NOT flip to the editor")
            self.assertAlmostEqual(th.value(), 2.75, places=9)
            wl = self._spin_by_tip(page, "wavelength (in vacuum)")
            self.assertNotIn("set by", self._row_label(page, wl).text())
            # (the substrate isotropic entry moved into the eps grids — fenced by
            # LayerMediumPanelContract.test_simple_substrate_grid_edit_stays_and_drives)
        finally:
            win.close()

    def test_stack_editor_mode_keeps_ownership_and_substrate_hint(self):
        from shaarp.desktop_app import build_main_window
        from shaarp.shaarp_gui import resolve_ml_system_preset
        win = build_main_window()
        try:
            page = self._ml_page(win)
            combo = self._preset_combo(page)
            wl = self._spin_by_tip(page, "wavelength (in vacuum)")
            startup_wl = float(resolve_ml_system_preset(combo.currentText()).wavelength_um)
            combo.setCurrentText("N-layer stack (editor)")
            self._pump()
            self.assertNotIn("set by", self._row_label(page, wl).text(),
                             "wavelength still drives build_system_from_stack - no false hint")
            self.assertAlmostEqual(wl.value(), startup_wl, places=9,
                                   msg="combo-entering the editor from a preset inherits the "
                                       "preset's lambda too (the editor takes over the whole "
                                       "experiment)")
        finally:
            win.close()

    def test_wavelength_group_is_the_first_input_on_both_tabs(self):
        """Wavelength is an upfront experiment property — the FIRST input group,
        above even the case/preset selector, on both tabs."""
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        try:
            tabs = win.findChild(QtWidgets.QTabWidget)
            for idx, name in ((0, "SI"), (1, "ML")):
                page = tabs.widget(idx)
                scroll = page.findChild(QtWidgets.QScrollArea)
                host = scroll.widget() if scroll else page
                groups = [g for g in host.findChildren(QtWidgets.QGroupBox) if g.isCheckable()]
                self.assertTrue(groups, f"{name}: no input groups found")
                self.assertTrue(groups[0].title().startswith("Wavelength Setting"),
                                f"{name}: first input group must be Wavelength Setting, "
                                f"got {groups[0].title()!r}")
        finally:
            win.close()

    def test_wheel_never_changes_an_input_value(self):
        """"disable mouse scrolling to change values - this can easily go wrong when
        user scroll from sections to sections." Synthetic wheel events on EVERY spin, dropdown
        and slider in both input columns must leave the value/index unchanged (worst pre-fix
        case: the wheel landing on the case selector switched the whole simulated system)."""
        from PySide6 import QtCore, QtGui
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        try:
            tabs = win.findChild(QtWidgets.QTabWidget)
            checked = 0
            for idx in (0, 1):
                page = tabs.widget(idx)
                scroll = page.findChild(QtWidgets.QScrollArea)
                host = scroll.widget() if scroll else page
                targets = (list(host.findChildren(QtWidgets.QAbstractSpinBox))
                           + list(host.findChildren(QtWidgets.QComboBox))
                           + list(host.findChildren(QtWidgets.QSlider)))
                for w in targets:
                    is_combo = isinstance(w, QtWidgets.QComboBox)
                    before = w.currentIndex() if is_combo else w.value()
                    for delta in (120, -120):
                        ev = QtGui.QWheelEvent(
                            QtCore.QPointF(5, 5),
                            QtCore.QPointF(w.mapToGlobal(QtCore.QPoint(5, 5))),
                            QtCore.QPoint(0, 0), QtCore.QPoint(0, delta),
                            QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
                            QtCore.Qt.NoScrollPhase, False)
                        QtWidgets.QApplication.sendEvent(w, ev)
                    after = w.currentIndex() if is_combo else w.value()
                    self.assertEqual(before, after,
                                     f"wheel changed {type(w).__name__} "
                                     f"({w.toolTip()[:40]!r}) {before} -> {after}")
                    checked += 1
            # 80 controls measured across both columns at the time of writing; the floor guards
            # against the finder going vacuously empty, not against layout evolution.
            self.assertGreater(checked, 60, "the wheel fence should cover the whole input column")
        finally:
            win.close()

    def test_guide_and_about_cite_the_papers_as_links(self):
        """Rui: 'paper should be linked' - the DOIs rendered as plain text while the repo URLs
        right above them were clickable."""
        from shaarp.desktop_app import USER_GUIDE_HTML
        for doi in ("10.1038/s41524-022-00930-4", "10.1038/s41524-024-01229-2"):
            self.assertIn(f'href="https://doi.org/{doi}"', USER_GUIDE_HTML,
                          f"guide must link {doi}, not print it as inert text")


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class LayerMediumPanelContract(unittest.TestCase):
    """The Dielectric Tensors section displays/edits the SELECTED
    layer's medium — isotropic media show eps = n^2 * I (air = the identity), and edits
    NEVER auto-switch the mode: under a named example they modify the working copy in place
    (stale/dirty-marked; re-selecting the example resets), in the editor they convert or
    write through the selected row. Fences (i)-(vii) of the approved F57 plan."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _pump(self, n=3):
        for _ in range(n):
            self.app.processEvents()

    @staticmethod
    def _ml_page(win):
        return win.findChild(QtWidgets.QTabWidget).widget(1)

    @staticmethod
    def _preset_combo(page):
        return next(c for c in page.findChildren(QtWidgets.QComboBox)
                    if c.findText("Custom film (use fields)") >= 0)

    @staticmethod
    def _editor_widgets(page):
        edit = next(c for c in page.findChildren(QtWidgets.QComboBox)
                    # row 0 is the AMBIENT (unnumbered — the original's convention:
                    # interior media are the numbered layers 1..N). Identify the editor combo
                    # by that role label, never by a "1:" prefix.
                    if c.count() and ("ambient" in c.itemText(0) or "air in" in c.itemText(0)))
        th = next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                  if s.maximum() >= 100000 and "Per-layer thickness" in s.toolTip())
        return edit, th

    @staticmethod
    def _grids(page):
        """(eps_w_cells, eps_2w_cells, d_cells) as 2-D lists of the TENSOR-CELL QLineEdits.

        Filtered by the full_tensor tooltip, never by position: a QDoubleSpinBox owns an
        internal QLineEdit, so when F58 added the scalar index rows to the same group a
        positional slice silently shifted by two cells (caught by this fence going red)."""
        from PySide6 import QtWidgets as _qw

        from shaarp.desktop_app import TOOLTIPS
        groups = [g for g in page.findChildren(_qw.QGroupBox)
                  if g.title().startswith("Dielectric Tensors")
                  or g.title().startswith("SHG Tensor")]
        eps_grp = next(g for g in groups if g.title().startswith("Dielectric Tensors"))
        d_grp = next(g for g in groups if g.title().startswith("SHG Tensor"))
        tip = TOOLTIPS["full_tensor"]
        eps_edits = [e for e in eps_grp.findChildren(_qw.QLineEdit) if e.toolTip() == tip]
        d_edits = [e for e in d_grp.findChildren(_qw.QLineEdit) if e.toolTip() == tip]
        assert len(eps_edits) == 18 and len(d_edits) == 18, (
            f"expected 18 eps + 18 d cells, got {len(eps_edits)} / {len(d_edits)}")
        ew = [eps_edits[r * 3:(r + 1) * 3] for r in range(3)]
        e2 = [eps_edits[9 + r * 3:9 + (r + 1) * 3] for r in range(3)]
        dd = [d_edits[r * 6:(r + 1) * 6] for r in range(3)]
        return ew, e2, dd

    @staticmethod
    def _grid_vals(cells):
        out = []
        for row in cells:
            out.append([complex(str(e.text()).replace("j", "j") or "0") if e.text().strip()
                        else 0j for e in row])
        return out

    @staticmethod
    def _group(page, prefix):
        from PySide6 import QtWidgets as _qw
        return next(g for g in page.findChildren(_qw.QGroupBox)
                    if g.title().startswith(prefix))

    def _fresh(self):
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        return win, self._ml_page(win)

    def _enter_editor(self, page):
        combo = self._preset_combo(page)
        combo.setCurrentText("N-layer stack (editor)")
        self._pump()
        return combo

    def test_air_row_displays_identity_tensors(self):
        """(F-i) the author's screenshot class: selecting an AIR half-space must SHOW air — eps = I,
        point group 1, d = 0 — and the structure/orientation/d groups must carry the
        'not used: isotropic medium' hint. Red pre-the mirror's registry lookup for
        'air' raises and returns early, so the panels keep the PREVIOUS row's tensors."""
        from PySide6 import QtWidgets as _qw
        win, page = self._fresh()
        try:
            combo = self._enter_editor(page)  # default stack: air / LiNbO3 / air
            edit, _th = self._editor_widgets(page)
            edit.setCurrentIndex(1)  # LiNbO3 first, so stale mirror content is detectable
            self._pump(2)
            edit.setCurrentIndex(0)  # the AIR ambient row
            self._pump(2)
            ew, e2, dd = self._grids(page)
            for r in range(3):
                for c in range(3):
                    want = 1.0 if r == c else 0.0
                    self.assertAlmostEqual(complex(ew[r][c].text() or "0").real, want,
                                           places=6,
                                           msg=f"eps_w[{r}][{c}] must display AIR (identity), "
                                               f"got {ew[r][c].text()!r}")
            pg = next(c for c in page.findChildren(_qw.QComboBox)
                      if c.findText("-43m") >= 0 and c.findText("1") >= 0)
            self.assertEqual(pg.currentText(), "1",
                             "point group must show 1 for an isotropic medium")
            for r in range(3):
                for c in range(6):
                    self.assertAlmostEqual(complex(dd[r][c].text() or "0").real, 0.0,
                                           places=9, msg="d must be zero for air")
            self.assertIn("not used: isotropic medium",
                          self._group(page, "SHG Tensor").title())
            self.assertIn("not used: isotropic medium",
                          self._group(page, "Crystal Structure").title())
            self.assertIn("air", self._group(page, "Dielectric Tensors").title(),
                          "the tensor group's title must name the mirrored medium")
        finally:
            win.close()

    def test_editor_panel_edit_keeps_mode_and_converts_row(self):
        """(F-ii) A panel edit in EDITOR mode must not destroy the stack: mode stays, the
        edited row converts to Custom (fields), the other rows keep their values. Red
        pre-the unguarded flip hook switches to 'Custom film' and _enter_stack_mode
        replaces the 4-layer stack with the 3-layer template."""
        from PySide6 import QtWidgets as _qw
        win, page = self._fresh()
        try:
            combo = self._enter_editor(page)  # startup preset is Fig-4 -> editor inherits it
            edit, th = self._editor_widgets(page)
            self.assertEqual(edit.count(), 4, "editor should inherit Fig-4's 4-layer stack")
            edit.setCurrentIndex(1)  # quartz
            self._pump(2)
            lattice = [s for s in page.findChildren(_qw.QDoubleSpinBox)
                       if "lattice" in s.toolTip().lower() or "Lattice" in s.toolTip()]
            if not lattice:  # fall back: locate by the Crystal Structure group
                lattice = self._group(page, "Crystal Structure").findChildren(
                    _qw.QDoubleSpinBox)
            lattice[0].setValue(lattice[0].value() + 0.37)  # a user structural edit
            self._pump()
            self.assertEqual(combo.currentText(), "N-layer stack (editor)",
                             "a panel edit in editor mode must NOT switch the mode "
                             "(pre-F57 it flipped to Custom film and DESTROYED the stack)")
            self.assertEqual(edit.count(), 4, "the 4-layer stack must survive the edit")
            payload = page._ml_stack_payload()
            self.assertEqual(payload["stack"][1]["material"], "Custom (fields)",
                             "the edited row must convert to a Custom layer")
            self.assertAlmostEqual(float(payload["stack"][2]["thickness_um"]), 0.0139,
                                   places=9, msg="the Au row must keep the preset value")
        finally:
            win.close()

    def test_selecting_a_custom_row_does_not_flip_the_mode(self):
        """(F-ii-b) Selecting a Custom row re-loads its snapshot into the panels; those
        PROGRAMMATIC writes must not fire the ownership hooks. Red pre-restore()'s
        lattice writes are not signal-blocked and the flip hook has no _loading guard."""
        win, page = self._fresh()
        try:
            combo = self._enter_editor(page)
            edit, _th = self._editor_widgets(page)
            edit.setCurrentIndex(1)
            self._pump(2)
            from PySide6 import QtWidgets as _qw
            lat = self._group(page, "Crystal Structure").findChildren(_qw.QDoubleSpinBox)
            lat[0].setValue(lat[0].value() + 0.51)  # convert row 1 to Custom
            self._pump()
            edit.setCurrentIndex(0)
            self._pump(2)
            edit.setCurrentIndex(1)  # re-select the Custom row -> restore() replays panels
            self._pump(2)
            self.assertEqual(combo.currentText(), "N-layer stack (editor)",
                             "re-selecting a Custom row must not flip the mode")
        finally:
            win.close()

    def test_custom_editor_row_computes_the_grid_values(self):
        """(F-iii, the latent bug) A user-created Custom editor row must COMPUTE the tensors
        shown in the grids. Red pre-snapshot() omits the grids and the custom builder
        falls back to the dead parentless index spins (n = 2.0 -> eps 4.0)."""
        win, page = self._fresh()
        try:
            self._enter_editor(page)
            edit, _th = self._editor_widgets(page)
            edit.setCurrentIndex(1)
            self._pump(2)
            ew, _e2, _dd = self._grids(page)
            ew[0][0].setText("9")
            ew[0][0].setModified(True)  # what real typing sets (the commit path gates on it)
            ew[0][0].textEdited.emit("9")
            ew[0][0].editingFinished.emit()
            self._pump()
            from shaarp.layer_stack import decode_stack
            payload = page._ml_stack_payload()
            decoded = decode_stack(payload["stack"])
            spec = decoded[1]
            self.assertEqual(spec["material"], "Custom (fields)")
            got = spec.get("custom", {}).get("eps_omega_full")
            self.assertIsNotNone(got, "custom snapshot must carry the eps grids (pre-F57 "
                                      "it did not, and the compute used n=2.0 defaults)")
            self.assertAlmostEqual(complex(got[0][0]).real, 9.0, places=6)
            from shaarp.layer_stack import build_system_from_stack
            sys_ = build_system_from_stack(decoded, wavelength_um=0.8)
            self.assertAlmostEqual(complex(sys_.layers[1].material.eps_w()[0][0]).real,
                                   9.0, places=6,
                                   msg="the BUILT system must use the grid value, not the "
                                       "dead scalar spins' 4.0")
        finally:
            win.close()

    def test_top_row_rejects_anisotropic_entry(self):
        """(F-v) The incident medium must stay isotropic (solver contract): a non-scalar
        eps entered on row 1 is refused at commit with a statusbar hint and the display
        reverts. Red pre-F57 via the mode-flip (the edit flips to Custom film)."""
        win, page = self._fresh()
        try:
            combo = self._enter_editor(page)
            edit, _th = self._editor_widgets(page)
            edit.setCurrentIndex(0)  # the air ambient row
            self._pump(2)
            ew, _e2, _dd = self._grids(page)
            ew[0][1].setText("0.5")  # off-diagonal -> anisotropic
            ew[0][1].setModified(True)
            ew[0][1].textEdited.emit("0.5")
            ew[0][1].editingFinished.emit()
            self._pump()
            self.assertEqual(combo.currentText(), "N-layer stack (editor)",
                             "a refused top-row entry must not switch the mode")
            payload = page._ml_stack_payload()
            self.assertIn(payload["stack"][0]["material"],
                          ("air", "isotropic n (set below)"),
                          "row 1 must stay an isotropic medium after the refusal")
            self.assertIn("isotropic", win.statusBar().currentMessage().lower(),
                          "the refusal must be explained in the status bar")
            self.assertAlmostEqual(complex(ew[0][1].text() or "0").real, 0.0, places=9,
                                   msg="the refused entry must revert on the display")
        finally:
            win.close()

    def test_edits_stay_under_the_example(self):
        """(F-iv, core rule) 'stay under the example case ... reupdate the input back
        if you select quartz+Au again': a thickness edit AND a tensor edit under Fig-4 keep
        the combo on Fig-4 (title marked edited); re-selecting Fig-4 resets everything.
        Red pre-the thickness edit flips the combo to the N-layer editor."""
        win, page = self._fresh()
        try:
            combo = self._preset_combo(page)
            preset_name = next(combo.itemText(i) for i in range(combo.count())
                               if "Fig 4" in combo.itemText(i))
            combo.setCurrentText("Custom film (use fields)")
            self._pump()
            combo.setCurrentText(preset_name)
            self._pump()
            edit, th = self._editor_widgets(page)
            edit.setCurrentIndex(1)
            self._pump(2)
            th.setValue(5.0)  # a layer-field edit under the example
            self._pump()
            self.assertEqual(combo.currentText(), preset_name,
                             "an edit under a named example must STAY under it (F57; the "
                             "F55 flip-to-editor is superseded)")
            case_grp = next(g for g in page.findChildren(QtWidgets.QGroupBox)
                            if g.title().startswith("Case Study"))
            self.assertIn("edited", case_grp.title().lower(),
                          "the case group's title must mark the example as edited")
            payload = page._ml_stack_payload()
            self.assertAlmostEqual(float(payload["stack"][1]["thickness_um"]), 5.0,
                                   places=9, msg="the working copy must carry the edit")
            combo.activated.emit(combo.currentIndex())  # re-pick the same example = reset
            self._pump()
            payload = page._ml_stack_payload()
            self.assertAlmostEqual(float(payload["stack"][1]["thickness_um"]), 121.2,
                                   places=6,
                                   msg="re-selecting the example must reset its true values")
            self.assertNotIn("edited", case_grp.title().lower(),
                             "the edited mark must clear on reset")
        finally:
            win.close()

    @staticmethod
    def _medium_spins(page):
        """ the scalar refractive-index rows — the isotropic-medium entry surface."""
        return [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                if "ISOTROPIC layer" in s.toolTip()]

    def test_simple_substrate_scalar_edit_stays_and_drives(self):
        """(F-vi, form) The substrate entry is ONE number per frequency: selecting the
        substrate row shows n = 1.45 / 1.46 in the scalar rows, typing 1.7 stays in the mode
        and lands in the template memory + the stack row. Red pre-no scalar rows exist
        (the value could only be reached through the 3x3 grid)."""
        win, page = self._fresh()
        try:
            combo = self._preset_combo(page)
            combo.setCurrentText("Custom film (use fields)")
            self._pump()
            edit, _th = self._editor_widgets(page)
            edit.setCurrentIndex(2)  # the substrate row
            self._pump(2)
            n_w, n_2w = self._medium_spins(page)
            self.assertAlmostEqual(n_w.value(), 1.45, places=6,
                                   msg="selecting the substrate row must DISPLAY its index")
            self.assertAlmostEqual(n_2w.value(), 1.46, places=6)
            n_w.setValue(1.7)
            self._pump()
            self.assertEqual(combo.currentText(), "Custom film (use fields)",
                             "a substrate index edit must stay in the simple mode")
            payload = page._ml_stack_payload()
            self.assertAlmostEqual(float(payload["simple_template"]["bottom"][0]), 1.7,
                                   places=6)
            self.assertAlmostEqual(float(payload["stack"][-1]["iso_n"][0]), 1.7, places=6)
        finally:
            win.close()

    def test_isotropic_rows_show_one_number_and_hide_the_grids(self):
        """("given we are assuming isotropic, better to have one input rather than
        having the full tensor") An isotropic medium takes ONE number per frequency: the
        scalar rows are shown and the 3x3 grids hidden; an anisotropic layer is the reverse.
        Red pre-no scalar rows, grids always visible."""
        win, page = self._fresh()
        try:
            self._preset_combo(page)  # startup preset: air / quartz / Au / air
            edit, _th = self._editor_widgets(page)
            n_w, _n_2w = self._medium_spins(page)
            scalar_form = n_w.parentWidget()
            eps_label = next(l for l in page.findChildren(QtWidgets.QLabel)
                             if l.text().startswith("ε(ω)"))
            eps_block = eps_label.parentWidget()
            edit.setCurrentIndex(0)  # air ambient
            self._pump(2)
            self.assertFalse(scalar_form.isHidden(), "an isotropic row shows the index rows")
            self.assertTrue(eps_block.isHidden(), "an isotropic row hides the 3x3 grids")
            self.assertAlmostEqual(n_w.value(), 1.0, places=9, msg="air is n = 1")
            edit.setCurrentIndex(1)  # anisotropic quartz
            self._pump(2)
            self.assertTrue(scalar_form.isHidden(),
                            "an anisotropic layer hides the scalar index rows")
            self.assertFalse(eps_block.isHidden(),
                             "an anisotropic layer shows the full tensor grids")
        finally:
            win.close()

    def test_custom_film_update_uses_film_tensors_not_the_selected_rows_mirror(self):
        """(F-vii, the hazard itself introduces) With the substrate row selected the
        grids show n^2 * I — Update must still compute the FILM's tensors. Pinned as an
        invariant: the computed curves are identical whichever row is selected."""
        from . import gui_harness as gh
        win, page = self._fresh()
        try:
            combo = self._preset_combo(page)
            combo.setCurrentText("Custom film (use fields)")
            self._pump()
            func = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.toolTip().startswith("Choose what to calculate"))
            func.setCurrentText("SHG Simulation")
            self._pump()
            edit, _th = self._editor_widgets(page)
            edit.setCurrentIndex(1)  # film row selected
            self._pump(2)
            gh.click_update(page, QtWidgets.QApplication.instance())
            want = gh.snapshot_curves(page)
            edit.setCurrentIndex(2)  # substrate row selected -> grids re-mirror
            self._pump(2)
            gh.click_update(page, QtWidgets.QApplication.instance())
            got = gh.snapshot_curves(page)
            self.assertFalse(gh.curves_changed(want, got),
                             "changing the SELECTED row must not change the computed "
                             "physics - the film's tensors drive the simple mode")
        finally:
            win.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class F61PanelShowsWhatItComputes(unittest.TestCase):
    """One analytical-d control, in the panel that owns the tensor; the
    grid and the thickness field show the SYMBOLS the closed form will contain; and no
    programmatic write may move the functionality mode."""

    def _pump(self, app, n=5):
        for _ in range(n):
            app.processEvents()

    def _ml(self, win):
        return win.findChild(QtWidgets.QTabWidget).widget(1)

    def _boxes(self, page):
        ah = next(b for b in page.findChildren(QtWidgets.QCheckBox)
                  if b.text().startswith("analytical h"))
        ad = next(b for b in page.findChildren(QtWidgets.QCheckBox)
                  if b.text().startswith("analytical d"))
        return ah, ad

    def test_exactly_one_analytical_d_control_and_it_lives_with_the_tensor(self):
        """Rui item 4: two analytical-dij toggles existed; only the d-tensor one survives."""
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            d_boxes = [b for b in page.findChildren(QtWidgets.QCheckBox)
                       if "analytical d" in b.text()]
            self.assertEqual(len(d_boxes), 1,
                             "expected ONE analytical-dij control, found "
                             + repr([b.text() for b in d_boxes]))
            group = d_boxes[0].parent()
            while group is not None and not isinstance(group, QtWidgets.QGroupBox):
                group = group.parent()
            self.assertIsNotNone(group)
            self.assertTrue(group.title().startswith("SHG Tensor"),
                            "the analytical-dij control belongs with the d tensor, found it in "
                            + repr(group.title()))
        finally:
            win.close()

    def test_flagged_layer_shows_its_symbolic_tensor_and_reverts(self):
        """Rui item 5 + his decision: the grid shows the point group's pattern with the layer's
        suffix and hard 0 where symmetry forbids a component."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            edit = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.count() and c.itemText(0).startswith("1:"))
            _ah, ad = self._boxes(page)
            cells = [e for e in page.findChildren(QtWidgets.QLineEdit)
                     if e.toolTip() == TOOLTIPS["full_tensor"]]
            self.assertGreaterEqual(len(cells), 36)
            d_row0 = cells[18:24]
            edit.setCurrentIndex(1)  # row 2 = Z-cut quartz, point group 32
            self._pump(app)
            numeric = [c.text() for c in d_row0]
            ad.click()
            self._pump(app)
            shown = [c.text() for c in d_row0]
            self.assertEqual(shown[:2], ["d11m2", "-d11m2"],
                             "row 2 must show its SUFFIXED symbols, got " + repr(shown))
            self.assertEqual(shown[3], "d14m2")
            self.assertEqual([shown[2], shown[4], shown[5]], ["0", "0", "0"],
                             "symmetry-forbidden components must stay 0")
            ad.click()
            self._pump(app)
            self.assertEqual([c.text() for c in d_row0], numeric,
                             "clearing the flag must restore this layer's numbers")
        finally:
            win.close()

    def test_flagged_thickness_shows_its_symbol(self):
        """The same rule applied to thickness: the field must not show a number the expression
        does not contain."""
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            edit = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.count() and c.itemText(0).startswith("1:"))
            ah, _ad = self._boxes(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            ah.click()
            self._pump(app)
            shown = [e.text() for e in page.findChildren(QtWidgets.QLineEdit)
                     if e.isReadOnly() and e.text().startswith("h")]
            self.assertIn("h2", shown, "row 2 thickness must display h2, got " + repr(shown))
            ah.click()
            self._pump(app)
            shown = [e.text() for e in page.findChildren(QtWidgets.QLineEdit)
                     if e.isReadOnly() and e.text().startswith("h")]
            self.assertFalse(shown, "clearing the flag must bring the numeric field back")
        finally:
            win.close()

    def test_a_programmatic_check_does_not_move_the_mode(self):
        """The auto-pivot must answer to USER clicks only. A session restore writes checkbox
        state programmatically (gui_introspect.apply_session_state), and while the flags were
        connected to stateChanged that restore silently landed the app in Partial Analytical."""
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            func = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.toolTip().startswith("Choose what to calculate"))
            ah, ad = self._boxes(page)
            self.assertEqual(func.currentText(), "SHG Simulation")
            ah.setChecked(True)   # programmatic
            ad.setChecked(True)
            self._pump(app)
            self.assertEqual(func.currentText(), "SHG Simulation",
                             "a programmatic setChecked moved the functionality mode")
            ah.click()            # a real click MAY pivot (that is the feature)
            self._pump(app)
            self.assertEqual(func.currentText(), "Partial Analytical Expressions")
        finally:
            win.close()

    def test_startup_pins_both_tabs_to_shg_simulation(self):
        """Rui item 1. A restored session brings back materials/layers/angles but never leaves
        you in a slow analytical mode on launch."""
        import os
        import tempfile

        import shaarp.desktop_app as da

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = da.build_main_window()
        try:
            tabs = win.findChild(QtWidgets.QTabWidget)
            for idx in (0, 1):
                combo = next(c for c in tabs.widget(idx).findChildren(QtWidgets.QComboBox)
                             if c.toolTip().startswith("Choose what to calculate"))
                target = next(combo.itemText(i) for i in range(combo.count())
                              if "Partial Analytical" in combo.itemText(i))
                combo.setCurrentText(target)
            self._pump(app)
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "s.json")
                orig = da._last_session_path
                da._last_session_path = lambda: path
                try:
                    da._autosave_session(win)
                    win2 = da.build_main_window()
                    try:
                        da._restore_last_session(win2)
                        self._pump(app)
                        tabs2 = win2.findChild(QtWidgets.QTabWidget)
                        for idx in (0, 1):
                            combo = next(c for c in tabs2.widget(idx).findChildren(QtWidgets.QComboBox)
                                         if c.toolTip().startswith("Choose what to calculate"))
                            self.assertEqual(combo.currentText(), "SHG Simulation",
                                             "tab %d did not open on SHG Simulation" % idx)
                    finally:
                        win2.close()
                finally:
                    da._last_session_path = orig
        finally:
            win.close()

    def test_converting_a_layer_keeps_its_flags(self):
        """`_to_custom` rebuilt a spec without the flags, so editing a palette layer's
        tensors silently stopped it being symbolic."""
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            edit = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.count() and c.itemText(0).startswith("1:"))
            ah, _ad = self._boxes(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            ah.click()  # row 2 symbolic
            self._pump(app)
            lat = next(g for g in page.findChildren(QtWidgets.QGroupBox)
                       if g.title().startswith("Crystal Structure")).findChildren(
                           QtWidgets.QDoubleSpinBox)
            lat[0].setValue(lat[0].value() + 0.37)  # structural edit -> converts to Custom
            self._pump(app)
            stack = page._ml_stack_payload()["stack"]
            self.assertEqual(stack[1]["material"], "Custom (fields)")
            self.assertTrue(stack[1].get("analytic_h"),
                            "the conversion dropped the layer's analytical-h flag")
        finally:
            win.close()

    # ---- the grid is symbolic on EVERY display of a flagged layer -------
    # the GUI walkthrough DEFECT 1: flag row 2, visit row 3, return to row 2 -> the box re-ticked but
    # the grid showed the NUMERIC tensor (the symbolic paint lived only on the click path and
    # _sync_layer_crystal_view repainted numbers on row selection). Update then parsed every
    # number as KNOWN and the closed form carried NO d symbols under a "[d symbolic]" header.

    def _d_row0_and_edit(self, page):
        from shaarp.desktop_app import TOOLTIPS
        edit = next(c for c in page.findChildren(QtWidgets.QComboBox)
                    if c.toolTip() == TOOLTIPS["layer_select"])
        cells = [e for e in page.findChildren(QtWidgets.QLineEdit)
                 if e.toolTip() == TOOLTIPS["full_tensor"]]
        self.assertGreaterEqual(len(cells), 36)
        return cells[18:24], edit

    @staticmethod
    def _expr_raw(page):
        box = page.findChild(QtWidgets.QTextEdit, "expr_box")
        return (box.property("raw_text") or "") if box is not None else ""

    @classmethod
    def _expr_body(cls, page):
        """The closed form WITHOUT its '# symbols:' header (the header names known values
        by symbol, e.g. 'known d = d11m2=0.3', so body checks must exclude it)."""
        raw = cls._expr_raw(page)
        cut = raw.find("====")
        return raw[cut:] if cut >= 0 else raw

    def test_reselecting_a_flagged_row_keeps_its_symbolic_grid(self):
        """Leave the flagged row and come back: the grid must still show d11m2/-d11m2/0/d14m2
        and an Update must still carry those symbols."""
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import click_update

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            d_row0, edit = self._d_row0_and_edit(page)
            _ah, ad = self._boxes(page)
            edit.setCurrentIndex(1)  # row 2 = Z-cut quartz, point group 32
            self._pump(app)
            ad.click()
            self._pump(app)
            expected = ["d11m2", "-d11m2", "0", "d14m2", "0", "0"]
            self.assertEqual([c.text() for c in d_row0], expected)
            edit.setCurrentIndex(2)  # visit row 3 (Au)
            self._pump(app)
            edit.setCurrentIndex(1)  # and return
            self._pump(app)
            self.assertTrue(ad.isChecked(), "the flag itself must survive the round trip")
            self.assertEqual([c.text() for c in d_row0], expected,
                             "re-selecting a flagged row must show its SYMBOLIC grid, got "
                             + repr([c.text() for c in d_row0]))
            click_update(page, app)
            self._pump(app, 20)
            raw = self._expr_raw(page)
            self.assertTrue(raw.strip(), "Partial Analytical produced no expression")
            self.assertIn("d11m2", raw, "the closed form lost the layer's d symbols")
            self.assertIn("d14m2", raw)
        finally:
            win.close()

    def test_update_uses_stack_flags_not_the_displayed_row(self):
        """Flag row 2, then DISPLAY row 3 and Update: row 2 is still symbolic in the closed
        form -- the compute keys on the stack flags, not on which row is on screen."""
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import click_update

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            _d_row0, edit = self._d_row0_and_edit(page)
            _ah, ad = self._boxes(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            ad.click()
            self._pump(app)
            edit.setCurrentIndex(2)  # unflagged row on screen at Update time
            self._pump(app)
            self.assertFalse(ad.isChecked(), "row 3 carries no flag")
            click_update(page, app)
            self._pump(app, 20)
            raw = self._expr_raw(page)
            self.assertTrue(raw.strip(), "Partial Analytical produced no expression")
            self.assertIn("d11m2", raw,
                          "row 2's d symbols must be in the closed form whichever row is shown")
        finally:
            win.close()

    def test_analytical_d_control_is_absent_on_half_space_rows(self):
        """Rule extended to the d control: the two semi-infinite media show NO analytical
        controls -- not disabled, ABSENT."""
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            _cells, edit = self._d_row0_and_edit(page)
            ah, ad = self._boxes(page)
            n = edit.count()
            self.assertGreaterEqual(n, 3)
            for half in (0, n - 1):
                edit.setCurrentIndex(half)
                self._pump(app)
                self.assertTrue(ad.isHidden(),
                                f"analytical-dij control must be hidden on half-space row {half + 1}")
                self.assertTrue(ah.isHidden())
            edit.setCurrentIndex(1)
            self._pump(app)
            self.assertFalse(ad.isHidden())
            self.assertFalse(ah.isHidden())
        finally:
            win.close()

    # ---- F62b (remember 0.3, that could mean user only want portions of
    # coefficients to be analytical"): a KNOWN value typed over a symbol is stored on the
    # layer, re-shown on every display, and used by Update whichever row is on screen.
    # And (SUSPECT 3) a structural edit on a flagged row must snapshot the
    # layer's NUMERIC d, not the symbolic grid (which parsed as zeros).

    @staticmethod
    def _type_cell(cell, text):
        cell.setText(text)
        cell.setModified(True)
        cell.textEdited.emit(text)
        cell.editingFinished.emit()

    @staticmethod
    def _decoded_stack(page):
        from shaarp.layer_stack import decode_stack
        return decode_stack(page._ml_stack_payload()["stack"])

    def test_known_d_value_is_remembered_across_rows(self):
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import click_update

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            d_row0, edit = self._d_row0_and_edit(page)
            _ah, ad = self._boxes(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            ad.click()
            self._pump(app)
            self._type_cell(d_row0[0], "0.3")
            self._pump(app)
            shown = [c.text() for c in d_row0]
            self.assertEqual(shown[:2], ["0.3", "-0.3"],
                             "a known d11 shows as its number (and -d11 follows), got " + repr(shown))
            self.assertEqual(shown[3], "d14m2", "d14 stays symbolic")
            stack = self._decoded_stack(page)
            self.assertNotEqual(stack[1]["material"], "Custom (fields)",
                                "declaring a known component is not a material edit")
            self.assertEqual(stack[1].get("analytic_d_known"), {"d11": 0.3 + 0j})
            edit.setCurrentIndex(2)
            self._pump(app)
            edit.setCurrentIndex(1)
            self._pump(app)
            self.assertEqual([c.text() for c in d_row0][:4], ["0.3", "-0.3", "0", "d14m2"],
                             "the typed known value must survive a row switch")
            click_update(page, app)
            self._pump(app, 20)
            raw = self._expr_raw(page)
            self.assertTrue(raw.strip())
            body = self._expr_body(page)
            self.assertIn("d14m2", body)
            self.assertNotIn("d11m2", body, "a known d11 must be substituted, not symbolic")
        finally:
            win.close()

    def test_known_d_applies_whichever_row_is_displayed(self):
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import click_update

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            d_row0, edit = self._d_row0_and_edit(page)
            _ah, ad = self._boxes(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            ad.click()
            self._pump(app)
            self._type_cell(d_row0[0], "0.3")
            self._pump(app)
            edit.setCurrentIndex(2)  # Au on screen at Update time
            self._pump(app)
            click_update(page, app)
            self._pump(app, 20)
            raw = self._expr_raw(page)
            self.assertTrue(raw.strip())
            body = self._expr_body(page)
            self.assertIn("d14m2", body)
            self.assertNotIn("d11m2", body)
        finally:
            win.close()

    def test_flagged_row_shows_numbers_outside_partial_analytical(self):
        """The flags are inert in the numeric functionalities (the compute uses the
        numbers), so the grid shows numbers there and symbols only in Partial Analytical --
        caught by test_gui_causality_gaps on the Custom-film template, whose film row is
        flagged by default."""
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            d_row0, edit = self._d_row0_and_edit(page)
            _ah, ad = self._boxes(page)
            func = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.toolTip().startswith("Choose what to calculate"))
            edit.setCurrentIndex(1)
            self._pump(app)
            ad.click()  # pivots to Partial Analytical, grid symbolic
            self._pump(app)
            self.assertEqual([c.text() for c in d_row0][:2], ["d11m2", "-d11m2"])
            func.setCurrentText("SHG Simulation")  # user leaves PA with the flag still on
            self._pump(app)
            self.assertTrue(ad.isChecked())
            self.assertEqual([c.text() for c in d_row0][:2], ["0.3", "-0.3"],
                             "outside Partial Analytical the compute uses the NUMBERS, so the "
                             "grid must show them")
            func.setCurrentText("Partial Analytical Expressions")
            self._pump(app)
            self.assertEqual([c.text() for c in d_row0][:2], ["d11m2", "-d11m2"],
                             "re-entering Partial Analytical re-mirrors the symbols")
        finally:
            win.close()

    def test_structural_edit_on_flagged_row_snapshots_numeric_d(self):
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            page = self._ml(win)
            self._pump(app)
            d_row0, edit = self._d_row0_and_edit(page)
            _ah, ad = self._boxes(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            numeric = [c.text() for c in d_row0]
            self.assertEqual(numeric[:2], ["0.3", "-0.3"], "Z-cut quartz's own d11")
            ad.click()
            self._pump(app)
            lat = next(g for g in page.findChildren(QtWidgets.QGroupBox)
                       if g.title().startswith("Crystal Structure")).findChildren(
                           QtWidgets.QDoubleSpinBox)
            lat[0].setValue(lat[0].value() + 0.37)  # structural edit -> converts to Custom
            self._pump(app)
            stack = self._decoded_stack(page)
            self.assertEqual(stack[1]["material"], "Custom (fields)")
            self.assertTrue(stack[1].get("analytic_d"), "the flag survives the conversion")
            d00 = complex(stack[1]["custom"]["d_full"][0][0])
            self.assertAlmostEqual(d00.real, 0.3, places=6,
                                   msg="the Custom snapshot must carry the NUMERIC d, not zeros "
                                       "parsed from the symbolic grid")
            ad.click()  # clear the flag -> numbers back
            self._pump(app)
            self.assertEqual([c.text() for c in d_row0][:2], ["0.3", "-0.3"])
        finally:
            win.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class F63PointGroupDecidesActivity(unittest.TestCase):
    """No "SHG active" box -- the POINT GROUP decides, exactly as the original
    ♯SHAARP.ml GUI's two popups do (SHAARP.ml.nb:5191 Noncentrosymmetric / :5630 Centrosymmetric).
    An SHG-inactive group zeroes the d grid, collapses the SHG-Tensor group with a hint and
    disables the analytical-dij control; the crystal system LOCKS the dependent lattice cells."""

    ACTIVE = ("1", "2", "m", "mm2", "222", "3", "32", "3m", "4", "6", "-4", "4mm", "6mm",
              "422", "622", "-42m", "-6", "-6m2", "-43m", "23", "∞", "∞m", "∞2")
    INACTIVE = ("-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m3", "m3m",
                "432", "∞/m", "∞/mm", "∞∞", "∞∞m")

    def _pump(self, app, n=5):
        for _ in range(n):
            app.processEvents()

    @staticmethod
    def _page(win, k):
        return win.findChild(QtWidgets.QTabWidget).widget(k)

    @staticmethod
    def _pg(page):
        from shaarp.desktop_app import TOOLTIPS
        return next(c for c in page.findChildren(QtWidgets.QComboBox)
                    if c.toolTip() == TOOLTIPS["point_group"])

    @staticmethod
    def _d_cells(page):
        from shaarp.desktop_app import TOOLTIPS
        cells = [e for e in page.findChildren(QtWidgets.QLineEdit)
                 if e.toolTip() == TOOLTIPS["full_tensor"]]
        return cells[18:36]

    @staticmethod
    def _lattice(page):
        from shaarp.desktop_app import TOOLTIPS
        return [e for e in page.findChildren(QtWidgets.QDoubleSpinBox)
                if e.toolTip() == TOOLTIPS["lattice"]]

    @staticmethod
    def _dm_group(page):
        return next(g for g in page.findChildren(QtWidgets.QGroupBox)
                    if g.title().startswith("SHG Tensor"))

    @staticmethod
    def _ad(page):
        return next(b for b in page.findChildren(QtWidgets.QCheckBox)
                    if b.text().startswith("analytical d"))

    def _pick(self, pg, label, app):
        pg.setCurrentText(label)
        pg.activated.emit(pg.currentIndex())
        self._pump(app)

    def test_combo_lists_the_originals_two_popups(self):
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            for k in (0, 1):
                pg = self._pg(self._page(win, k))
                items = [pg.itemText(i) for i in range(pg.count())]
                expect = (["—  Noncentrosymmetric (SHG-active)  —", *self.ACTIVE,
                           "—  Centrosymmetric (SHG-inactive)  —", *self.INACTIVE])
                self.assertEqual(items, expect, f"tab {k}")
                model = pg.model()
                for i, t in enumerate(items):
                    self.assertEqual(model.item(i).isEnabled(), not t.startswith("—"), t)
                if k == 0:
                    self.assertEqual(pg.currentText(), "-43m")  # ML mirrors its selected row
                pg.setCurrentIndex(0)  # a header row can only be reached programmatically
                self._pump(app)
                self.assertFalse(pg.currentText().startswith("—"), "header rows are skipped")
        finally:
            win.close()

    def test_inactive_group_zeroes_d_and_disables_analytical_d(self):
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import combo_with_item

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            si = self._page(win, 0)
            case = combo_with_item(si, "Custom (use fields)")
            case.setCurrentText("Custom (use fields)")
            self._pump(app)
            pg, cells, g_dm = self._pg(si), self._d_cells(si), self._dm_group(si)
            # (the SI tab has no analytical-dij control -- that one is ML-only; see the ML test)
            self._pick(pg, "3m", app)
            self.assertTrue(any(c.text() not in ("0", "") for c in cells), "3m has a pattern")
            self._pick(pg, "m3m", app)
            self.assertTrue(all(c.text() == "0" for c in cells),
                            "an SHG-inactive group shows d = 0: " + repr([c.text() for c in cells]))
            self.assertIn("SHG-inactive", g_dm.title())
            self.assertFalse(g_dm.isChecked(), "the group collapses (hint + collapse)")
            self._pick(pg, "3m", app)
            self.assertTrue(any(c.text() not in ("0", "") for c in cells), "the pattern is back")
            self.assertNotIn("SHG-inactive", g_dm.title())
            self.assertTrue(g_dm.isChecked())
        finally:
            win.close()

    def test_crystal_system_locks_the_lattice(self):
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import combo_with_item

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            si = self._page(win, 0)
            case = combo_with_item(si, "Custom (use fields)")
            case.setCurrentText("Custom (use fields)")
            self._pump(app)
            pg, lat = self._pg(si), self._lattice(si)
            self.assertEqual(len(lat), 6)
            self._pick(pg, "-43m", app)
            self.assertEqual([e.isEnabled() for e in lat], [True, False, False, False, False, False])
            lat[0].setValue(5.0)
            self._pump(app)
            self.assertEqual([round(e.value(), 6) for e in lat], [5.0, 5.0, 5.0, 90.0, 90.0, 90.0],
                             "a -> b, c under a cubic group")
            self._pick(pg, "6mm", app)
            self.assertEqual([e.isEnabled() for e in lat], [True, False, True, False, False, False])
            self.assertEqual(round(lat[5].value(), 6), 120.0)
            lat[2].setValue(7.5)  # c is free
            self._pump(app)
            self.assertEqual(round(lat[2].value(), 6), 7.5)
            self._pick(pg, "1", app)
            self.assertTrue(all(e.isEnabled() for e in lat))
            self._pick(pg, "∞∞m", app)  # Curie isotropic locks like cubic
            self.assertEqual([e.isEnabled() for e in lat], [True, False, False, False, False, False])
        finally:
            win.close()

    def test_preset_after_a_film_case_mirrors_the_real_lattice(self):
        """the GUI walkthrough D4: the Fig-4 factory quartz carries a placeholder cell (a=b=c=1); after a
        single-film case the preset pick must still show the palette material's real lattice."""
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import ml_case_combo

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            ml = self._page(win, 1)
            self._pump(app)
            cc, lat, pg = ml_case_combo(ml), self._lattice(ml), self._pg(ml)
            for label in ("GaAs (111) (800 nm)", "Quartz + Au (Fig 4, 800 nm)"):
                cc.setCurrentText(label)
                cc.activated.emit(cc.currentIndex())
                self._pump(app)
            self.assertEqual(pg.currentText(), "32")
            self.assertEqual([round(e.value(), 3) for e in lat], [4.913, 4.913, 5.405, 90.0, 90.0, 120.0])
        finally:
            win.close()

    def test_ml_layer_activity_follows_its_point_group(self):
        """Fig-4 quartz (32) flagged analytical-d, then moved to m3m: the row converts to Custom,
        the flag and the known values are cleared, the grid is zero and the built system carries
        no source for that layer; an Update still runs."""
        from shaarp.desktop_app import TOOLTIPS, build_main_window
        from shaarp.layer_stack import build_system_from_stack, decode_stack
        from tests.gui_harness import click_update

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            ml = self._page(win, 1)
            self._pump(app)
            edit = next(c for c in ml.findChildren(QtWidgets.QComboBox)
                        if c.toolTip() == TOOLTIPS["layer_select"])
            pg, cells, ad = self._pg(ml), self._d_cells(ml), self._ad(ml)
            edit.setCurrentIndex(1)  # row 2 = Z-cut quartz, 32
            self._pump(app)
            self.assertEqual(pg.currentText(), "32")
            self.assertFalse(any(b.text() == "SHG active" for b in ml.findChildren(QtWidgets.QCheckBox)))
            ad.click()
            self._pump(app)
            self.assertEqual(cells[0].text(), "d11m2")
            self._pick(pg, "m3m", app)
            stack = decode_stack(ml._ml_stack_payload()["stack"])
            self.assertEqual(stack[1]["material"], "Custom (fields)")
            self.assertEqual(stack[1]["custom"]["point_group"], "m3m")
            self.assertFalse(stack[1].get("analytic_d"), "an inactive layer cannot stay analytical")
            self.assertFalse(ad.isEnabled())
            self.assertTrue(all(c.text() == "0" for c in cells))
            g_dm = self._dm_group(ml)
            self.assertIn("SHG-inactive", g_dm.title())
            self.assertFalse(g_dm.isChecked(), "the ML medium hint must not re-expand it")
            func = next(c for c in ml.findChildren(QtWidgets.QComboBox)
                        if c.toolTip().startswith("Choose what to calculate"))
            self.assertNotEqual(func.currentText(), "Partial Analytical Expressions",
                                "a point-group pick never pivots INTO Partial Analytical")
            sysm = build_system_from_stack(stack, wavelength_um=0.8)
            self.assertEqual([L.shg_active for L in sysm.layers], [False, False, False, False])
            click_update(ml, app)
            self._pump(app, 20)
            self._pick(pg, "32", app)
            self.assertTrue(ad.isEnabled())
            sysm = build_system_from_stack(decode_stack(ml._ml_stack_payload()["stack"]), wavelength_um=0.8)
            self.assertTrue(sysm.layers[1].shg_active, "an active group makes the layer a source again")
        finally:
            win.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class CaseOwnershipContract(unittest.TestCase):
    """"make sure those settings and input parameters are dynamically
    linked to the simulated case, unless those values are modified by users and pending update."

    The generalization of F53 from three ML fields to the WHOLE input surface. Legitimate states
    for any input field are exactly two: (a) it shows the value the simulated configuration
    actually uses, or (b) it shows a user edit with the stale banner marking it pending — and a
    user edit to a case-owned field must TAKE ownership (flip to Custom, the pattern) or
    snap back (the pattern). Nothing may sit silently ignored while looking live.

    The behavioral audit found three violations:
      V1 the wavelength spin showed 1.064 under the 800-nm GaAs (111) SI case (and lambda edits
          under a single-lambda case are clamp-ignored) — the exact class of the author's 2.064 screen;
      V2 the point-group combo under a case/preset repopulates the d-grid TEMPLATE while the
          compute keeps using the case — worse than dead, the panel becomes misleading;
      V3 the lattice spins under a case/preset are silently ignored (no flip, no hint).
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _pump(self, n=6):
        for _ in range(n):
            self.app.processEvents()

    @staticmethod
    def _wl_spin(page):
        return next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                    if "wavelength (in vacuum)" in s.toolTip())

    @staticmethod
    def _row_label(page, spin):
        for lay in page.findChildren(QtWidgets.QFormLayout):
            lbl = lay.labelForField(spin)
            if lbl is not None:
                return lbl
        return None

    @staticmethod
    def _pg_combo(page):
        return next(c for c in page.findChildren(QtWidgets.QComboBox)
                    if c.findText("3m") >= 0 and c.findText("mm2") >= 0)

    def test_si_wavelength_is_owned_by_a_single_lambda_case(self):
        """V1: selecting a SINGLE-LAMBDA case (TaAs (112): the registry grid is one 0.8-um point,
        measured, so any other spin value is clamp-ignored) must sync the lambda spin to the
        case's lambda and say so; a later write (user edit or session restore) snaps back while
        the case owns it. GaAs (111) is deliberately the counter-case below: its registry grid is
        a real 0.4-2.0 um dispersion table, so lambda genuinely DRIVES and stays user-owned."""
        from shaarp.desktop_app import build_main_window
        from shaarp.casestudy_materials import casestudy_lambda_range, resolve_case_label
        lam = float(casestudy_lambda_range(resolve_case_label("TaAs (112)"))[0])
        win = build_main_window()
        try:
            si = win.findChild(QtWidgets.QTabWidget).widget(0)
            case = _case_combo(si, "TaAs (112)")
            wl = self._wl_spin(si)
            wl.setValue(1.064)
            case.setCurrentText("TaAs (112)")
            self._pump()
            self.assertAlmostEqual(wl.value(), lam, places=9,
                                   msg="lambda must sync to the case's single-lambda grid value "
                                       "(the dataset defines lambda; 1.064 on screen is a lie)")
            lbl = self._row_label(si, wl)
            self.assertIn("set by the case", lbl.text())
            wl.setValue(1.55)
            self._pump()
            self.assertAlmostEqual(wl.value(), lam, places=9,
                                   msg="a write to the case-owned lambda must snap back "
                                       "(single-lambda data: any other value is clamp-ignored)")
            # counter-case: a DISPERSIVE material keeps lambda user-owned (it drives the lookup)
            case.setCurrentText("GaAs (111)")
            self._pump()
            self.assertNotIn("set by", self._row_label(si, wl).text(),
                             "GaAs (111) carries a real dispersion grid - lambda drives, no hint")
            wl.setValue(1.2)
            self._pump()
            self.assertAlmostEqual(wl.value(), 1.2, places=9,
                                   msg="dispersive-case lambda is user-owned; it must not snap")
            case.setCurrentText("Custom (use fields)")
            self._pump()
            self.assertNotIn("set by", self._row_label(si, wl).text(),
                             "hint must clear when the user owns lambda again")
        finally:
            win.close()

    def test_point_group_user_choice_stays_under_the_case(self):
        """V2 under F57 (stay under the example case"): a USER point-group pick while a
        case is selected keeps the case selected, marks it edited, and the edit is what gets
        computed; re-selecting the case resets."""
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        try:
            si = win.findChild(QtWidgets.QTabWidget).widget(0)
            case = _case_combo(si)
            case.setCurrentText("GaAs (111)")
            self._pump()
            pgc = self._pg_combo(si)
            idx = pgc.findText("mm2")
            pgc.setCurrentIndex(idx)
            pgc.activated.emit(idx)  # `activated` fires only on real user interaction
            self._pump()
            self.assertEqual(case.currentText(), "GaAs (111)",
                             "an edit under a case must STAY under it (F57 - no mode flip)")
            case_grp = next(g for g in si.findChildren(QtWidgets.QGroupBox)
                            if g.title().startswith("Case Study"))
            self.assertIn("edited", case_grp.title().lower(),
                          "the case group must mark the example as edited")
            case.activated.emit(case.currentIndex())  # re-pick = reset
            self._pump()
            self.assertNotIn("edited", case_grp.title().lower(),
                             "re-selecting the case must clear the edited mark")
        finally:
            win.close()

    def test_lattice_edit_takes_ownership_and_case_sync_does_not(self):
        """V3 under a user lattice edit under a case STAYS under it (edited-marked); the
        programmatic case-selection sync of those same spins must not mark anything
        (signal-blocked, lesson)."""
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        try:
            si = win.findChild(QtWidgets.QTabWidget).widget(0)
            case = _case_combo(si)
            case.setCurrentText("GaAs (111)")
            self._pump()
            self.assertEqual(case.currentText(), "GaAs (111)",
                             "selecting the case must not immediately flip itself (sync must be "
                             "signal-blocked)")
            lat = next(s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
                       if "crystallographic constants" in s.toolTip())
            lat.setValue(lat.value() + 0.37)
            self._pump()
            self.assertEqual(case.currentText(), "GaAs (111)",
                             "a lattice edit under a case must STAY under it (F57)")
            case_grp = next(g for g in si.findChildren(QtWidgets.QGroupBox)
                            if g.title().startswith("Case Study"))
            self.assertIn("edited", case_grp.title().lower(),
                          "the case group must mark the example as edited")
        finally:
            win.close()

    def test_ml_film_mode_dispersive_lambda_stays_user_owned(self):
        """V1 on the ML side, as MEASURED: every ML film dataset carries a real 0.4-2.0 um
        dispersion grid, so in Film mode lambda genuinely drives the lookup and must stay
        user-owned — synced/hinted ownership belongs to the named presets only."""
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        try:
            ml = win.findChild(QtWidgets.QTabWidget).widget(1)
            combo = next(c for c in ml.findChildren(QtWidgets.QComboBox)
                         if c.findText("Custom film (use fields)") >= 0)
            film = next(combo.itemText(i) for i in range(combo.count())
                        if "LiNbO3 x-cut" in combo.itemText(i))
            combo.setCurrentText(film)
            self._pump()
            wl = self._wl_spin(ml)
            lbl = self._row_label(ml, wl)
            self.assertNotIn("set by", lbl.text(),
                             "film datasets are dispersive - lambda drives; no ownership hint")
            wl.setValue(0.9)
            self._pump()
            self.assertAlmostEqual(wl.value(), 0.9, places=9,
                                   msg="film-mode lambda is user-owned; it must not snap")
        finally:
            win.close()

    def test_preset_values_do_not_leak_into_user_owned_modes(self):
        """The matrix sweep caught this: after the Fig-4 preset synced thickness to 121.2 um,
        selecting the MoS2 (800 nm) film computed a 121-um "monolayer" whose absorbing
        exp(i k_z h) overflowed (RA Scan OverflowError). Under F55 the thickness input IS the
        layer editor's film row: entering a preset loads the preset's real stack (121.2 shows
        in the editor); leaving it rebuilds the simple template from the REMEMBERED user
        thickness, and the stashed wavelength is restored."""
        from shaarp.desktop_app import build_main_window
        win = build_main_window()
        try:
            page = win.findChild(QtWidgets.QTabWidget).widget(1)
            combo = next(c for c in page.findChildren(QtWidgets.QComboBox)
                         if c.findText("Custom film (use fields)") >= 0)
            wl = self._wl_spin(page)
            edit, th = StackModeOwnershipContract._editor_widgets(page)
            preset_name = next(combo.itemText(i) for i in range(combo.count())
                               if "Fig 4" in combo.itemText(i))
            combo.setCurrentText("Custom film (use fields)")
            self._pump()
            wl.setValue(1.2)
            edit.setCurrentIndex(1)  # the film row of the 3-layer template
            self._pump(2)
            th.setValue(2.5)
            self._pump()
            combo.setCurrentText(preset_name)
            self._pump()
            edit.setCurrentIndex(1)  # the quartz row of the loaded preset stack
            self._pump(2)
            self.assertAlmostEqual(th.value(), 121.2, places=6)  # preset truth while it owns
            film = next(combo.itemText(i) for i in range(combo.count())
                        if "MoS2" in combo.itemText(i))
            combo.setCurrentText(film)
            self._pump()
            edit.setCurrentIndex(1)  # the MoS2 film row of the rebuilt template
            self._pump(2)
            self.assertAlmostEqual(th.value(), 2.5, places=9,
                                   msg="leaving the preset must restore the USER'S thickness - "
                                       "121.2 um leaking into a MoS2 monolayer overflowed the "
                                       "absorbing exp(i k_z h) in the sweep")
            # F57 retired the lambda stash: MoS2 (800 nm) is a single-lambda case, so the case-owned sync sets 0.8 - the preset's value never "leaks" as such
            self.assertAlmostEqual(wl.value(), 0.8, places=9,
                                   msg="film-mode lambda must be the CASE value (F54 sync); "
                                       "the retired stash no longer restores 1.2")
        finally:
            win.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class UserMaterials(unittest.TestCase):
    """"My Materials" -- save the current material under a name into a local
    JSON store, reuse it on both tabs, update / rename / delete it; built-in cases read-only; the
    old session-scoped Presets group hidden. The suite points the store at a temp file."""

    @classmethod
    def setUpClass(cls):
        import os
        import tempfile

        from shaarp import user_materials as um
        cls._um = um
        cls._prev_env = os.environ.get(um.ENV_OVERRIDE)
        cls._tmp = tempfile.TemporaryDirectory()
        os.environ[um.ENV_OVERRIDE] = os.path.join(cls._tmp.name, "user_materials.json")
        assert um.store_path().startswith(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        import os
        if cls._prev_env is None:
            os.environ.pop(cls._um.ENV_OVERRIDE, None)
        else:
            os.environ[cls._um.ENV_OVERRIDE] = cls._prev_env
        cls._tmp.cleanup()

    def setUp(self):
        for n in self._um.list_names():
            self._um.delete(n)

    @staticmethod
    def _pump(app, n=6):
        for _ in range(n):
            app.processEvents()

    @staticmethod
    def _page(win, k):
        return win.findChild(QtWidgets.QTabWidget).widget(k)

    @staticmethod
    def _combo(page, key):
        from shaarp.desktop_app import TOOLTIPS
        return next(c for c in page.findChildren(QtWidgets.QComboBox) if c.toolTip() == TOOLTIPS[key])

    @staticmethod
    def _d_cells(page):
        from shaarp.desktop_app import TOOLTIPS
        cells = [e for e in page.findChildren(QtWidgets.QLineEdit) if e.toolTip() == TOOLTIPS["full_tensor"]]
        return cells[18:36]

    @staticmethod
    def _wl(page):
        from shaarp.desktop_app import TOOLTIPS
        return next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox) if s.toolTip() == TOOLTIPS["wavelength"])

    @staticmethod
    def _buttons(page):
        from shaarp.desktop_app import TOOLTIPS
        by = {b.text(): b for b in page.findChildren(QtWidgets.QPushButton) if b.toolTip() == TOOLTIPS["my_materials"]}
        return by["Save current as new"], by["Update selected"], by["Rename…"], by["Delete selected"]

    @staticmethod
    def _type(cell, text):
        cell.setText(text)
        cell.setModified(True)
        cell.textEdited.emit(text)
        cell.editingFinished.emit()

    def _save_from_si(self, win, app, name, d11="7.7", lam=0.9):
        si = self._page(win, 0)
        case = self._combo(si, "case_study")
        case.setCurrentText("Custom (use fields)")
        self._pump(app)
        self._wl(si).setValue(lam)
        self._type(self._d_cells(si)[0], d11)
        self._pump(app)
        si._my_materials["name_edit"].setText(name)
        save_btn = self._buttons(si)[0]
        save_btn.click()  # the real button (no modal on save)
        self._pump(app)
        return si, case

    def test_save_lists_on_both_tabs_and_selects_it(self):
        from shaarp.desktop_app import build_main_window
        from shaarp.user_materials import USER_SECTION_HEADER

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            si, case = self._save_from_si(win, app, "my crystal")
            self.assertIn("my crystal", self._um.list_names())
            self.assertEqual(case.currentText(), "my crystal")
            hdr = case.findText(USER_SECTION_HEADER)
            self.assertGreaterEqual(hdr, 0)
            self.assertFalse(case.model().item(hdr).isEnabled(), "the section header is disabled")
            lm = self._combo(self._page(win, 1), "layer_material")
            self.assertGreaterEqual(lm.findText("my crystal"), 0)
            self.assertFalse(lm.model().item(lm.findText(USER_SECTION_HEADER)).isEnabled())
            self.assertEqual(si._my_materials["selected"](), "my crystal")
            self.assertTrue(all(b.isEnabled() for b in self._buttons(si)[1:]))
        finally:
            win.close()

    def test_fresh_window_reloads_it_single_lambda_and_updates(self):
        from shaarp.desktop_app import build_main_window
        from tests.gui_harness import click_update

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            self._save_from_si(win, app, "my crystal", d11="7.7", lam=0.9)
        finally:
            win.close()
        win2 = build_main_window()
        try:
            si = self._page(win2, 0)
            case = self._combo(si, "case_study")
            self._wl(si).setValue(1.3)
            case.setCurrentText("my crystal")
            case.activated.emit(case.currentIndex())
            self._pump(app)
            self.assertEqual(self._d_cells(si)[0].text(), "7.7", "the saved tensor mirrors into the panel")
            self.assertAlmostEqual(self._wl(si).value(), 0.9, places=9,
                                   msg="a saved material is single-lambda (F54 rule)")
            click_update(si, app)
            self._pump(app, 20)
            self.assertIn("complete", win2.statusBar().currentMessage().lower())
        finally:
            win2.close()

    def test_ml_layer_uses_the_user_material(self):
        from shaarp.desktop_app import build_main_window
        from shaarp.layer_stack import build_system_from_stack, decode_stack

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            self._save_from_si(win, app, "my crystal", d11="7.7")
            ml = self._page(win, 1)
            sel = self._combo(ml, "layer_select")
            lm = self._combo(ml, "layer_material")
            sel.setCurrentIndex(1)
            self._pump(app)
            lm.setCurrentText("my crystal")
            self._pump(app)
            stack = decode_stack(ml._ml_stack_payload()["stack"])
            self.assertEqual(stack[1]["material"], "my crystal")
            sysm = build_system_from_stack(stack, wavelength_um=0.9)
            self.assertEqual(sysm.layers[1].material.name, "my crystal")
            self.assertAlmostEqual(complex(sysm.layers[1].material.d_voigt()[0][0]).real, 7.7, places=9)
            self.assertEqual(ml._my_materials["selected"](), "my crystal")
        finally:
            win.close()

    def test_update_rename_delete_through_the_buttons(self):
        from shaarp.desktop_app import build_main_window
        from shaarp.user_materials import USER_SECTION_HEADER

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        yes = QtWidgets.QMessageBox.Yes
        orig_q, orig_t = QtWidgets.QMessageBox.question, QtWidgets.QInputDialog.getText
        QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: yes)
        QtWidgets.QInputDialog.getText = staticmethod(lambda *a, **k: ("my crystal 2", True))
        try:
            si, case = self._save_from_si(win, app, "my crystal", d11="7.7")
            _save, update, rename, delete = self._buttons(si)
            self._type(self._d_cells(si)[0], "8.8")
            self._pump(app)
            update.click()
            self._pump(app)
            self.assertAlmostEqual(complex(self._um.get("my crystal")["spec"]["d_full"][0][0]).real, 8.8)
            rename.click()
            self._pump(app)
            self.assertEqual(self._um.list_names(), ["my crystal 2"])
            self.assertEqual(case.currentText(), "my crystal 2")
            lm = self._combo(self._page(win, 1), "layer_material")
            self.assertGreaterEqual(lm.findText("my crystal 2"), 0)
            self.assertLess(lm.findText("my crystal"), 0)
            delete.click()
            self._pump(app)
            self.assertEqual(self._um.list_names(), [])
            self.assertEqual(case.currentText(), "Custom (use fields)", "deleted selection falls back")
            self.assertLess(case.findText(USER_SECTION_HEADER), 0, "empty store -> no section")
            self.assertLess(lm.findText("my crystal 2"), 0)
            self.assertIn("deleted", win.statusBar().currentMessage().lower())
            self.assertFalse(update.isEnabled())
        finally:
            QtWidgets.QMessageBox.question = orig_q
            QtWidgets.QInputDialog.getText = orig_t
            win.close()

    def test_rename_keeps_the_other_tabs_selection_and_headers_are_never_selected(self):
        """the GUI walkthrough D1 + D3."""
        from shaarp.desktop_app import build_main_window
        from shaarp.layer_stack import decode_stack
        from shaarp.user_materials import USER_SECTION_HEADER

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            si, case = self._save_from_si(win, app, "walk mat")
            ml = self._page(win, 1)
            sel, lm = self._combo(ml, "layer_select"), self._combo(ml, "layer_material")
            sel.setCurrentIndex(1)
            self._pump(app)
            lm.setCurrentText("walk mat")
            self._pump(app)
            self.assertEqual(ml._my_materials["rename"]("walk mat 2", confirm=False), "walk mat 2")
            self._pump(app)
            self.assertEqual(case.currentText(), "walk mat 2", "the SI selection follows the rename")
            self.assertEqual(lm.currentText(), "walk mat 2")
            self.assertEqual(decode_stack(ml._ml_stack_payload()["stack"])[1]["material"], "walk mat 2")
            # D3: a programmatic landing on the header row steps to a real entry
            for combo in (case, lm):
                combo.setCurrentIndex(combo.findText(USER_SECTION_HEADER))
                self._pump(app)
                self.assertNotEqual(combo.currentText(), USER_SECTION_HEADER)
            self.assertNotEqual(decode_stack(ml._ml_stack_payload()["stack"])[1]["material"], USER_SECTION_HEADER)
            # S2: an ML-side Update keeps the material's SAVED wavelength
            lm.setCurrentText("walk mat 2")
            self._pump(app)
            self.assertEqual(ml._my_materials["update"](confirm=False), "walk mat 2")
            self.assertAlmostEqual(self._um.get("walk mat 2")["wavelength_um"], 0.9, places=9)
        finally:
            win.close()

    def test_user_material_on_a_preset_row_survives_session_restore(self):
        """A user material assigned to a NON-edited row of a named preset
        survives collect -> fresh window -> apply."""
        import shaarp.desktop_app as da
        from shaarp.layer_stack import decode_stack

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = da.build_main_window()
        try:
            self._save_from_si(win, app, "sess mat")
            ml = self._page(win, 1)
            combo = self._combo(ml, "case_study")
            preset_name = next(combo.itemText(i) for i in range(combo.count())
                               if "Fig 4" in combo.itemText(i))
            combo.setCurrentText(preset_name)
            self._pump(app)
            sel, lm = self._combo(ml, "layer_select"), self._combo(ml, "layer_material")
            sel.setCurrentIndex(1)
            self._pump(app)
            lm.setCurrentText("sess mat")
            self._pump(app)
            sel.setCurrentIndex(2)  # park elsewhere (the S23 shape)
            self._pump(app)
            state = win._collect_session_state()
            payload = ml._ml_stack_payload()
            self.assertTrue(payload["dirty"])
        finally:
            win.close()
        win2 = da.build_main_window()
        try:
            win2._apply_session_state(state)
            ml2 = self._page(win2, 1)
            ml2._ml_stack_apply(payload)
            self._pump(app)
            got = decode_stack(ml2._ml_stack_payload()["stack"])
            self.assertEqual(got[1]["material"], "sess mat",
                             "user material on a non-edited preset row lost on restore (S23)")
        finally:
            win2.close()

    def test_built_in_names_are_protected_and_presets_group_hidden(self):
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        try:
            si = self._page(win, 0)
            with self.assertRaises(ValueError):
                si._my_materials["save"]("GaAs (111) (800 nm)", confirm=False)
            with self.assertRaises(ValueError):
                si._my_materials["save"]("", confirm=False)
            self.assertEqual(self._um.list_names(), [])
            g_pre = next(g for g in si.findChildren(QtWidgets.QGroupBox)
                         if g.title().startswith("Layer Properties Preset Values"))
            self.assertTrue(g_pre.isHidden(), "the session-scoped Presets group is hidden (F64)")
            self.assertTrue(any(b.text() == "Preset 1" for b in si.findChildren(QtWidgets.QPushButton)),
                            "…but still constructed")
            self.assertIsNotNone(next((g for g in si.findChildren(QtWidgets.QGroupBox)
                                       if g.title() == "My Materials"), None))
        finally:
            win.close()
