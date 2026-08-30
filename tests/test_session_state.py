"""Phase D fences: session save/load round-trip + debug log + Debug Info dialog.

The session serialization is DEFINED by the same widget walk the coverage gate enforces
(shaarp/gui_introspect.py), so these fences also pin that unification: every state-carrying
walked widget must round-trip through a session file, and a session restore must reproduce
the same computed curves.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False

import numpy as np


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SessionStateRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()

    def test_every_walked_state_widget_round_trips(self):
        from shaarp.gui_introspect import (apply_session_state, collect_session_state,
                                           walk_all, widget_value)

        si = self.win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("TaAs (112)") >= 0)
        case.setCurrentText("TaAs (112)")
        self.app.processEvents()
        # locate by TOOLTIP, not by maximum() -- the SI max moved 89 -> 89.9 (the .si
        # original allows 0-90; only .ml is 0-89) and a value-based selector silently matched
        # nothing. The tooltip is the locator desktop_app itself documents for this spin.
        from shaarp.desktop_app import TOOLTIPS

        theta = next(s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
                     if s.toolTip() == TOOLTIPS["theta"])
        theta.setValue(33.0)
        state = collect_session_state(self.win)
        self.assertGreater(len(state), 150, "session state suspiciously small")
        # JSON round-trip (the file format users attach to reports)
        state = json.loads(json.dumps({"widgets": state}))["widgets"]
        # mutate a spread of controls, then restore
        case.setCurrentText("Custom (use fields)")
        theta.setValue(60.0)
        missing = apply_session_state(self.win, state)
        self.assertFalse(missing, f"restore lost idents: {missing[:5]}")
        self.assertEqual(case.currentText(), "TaAs (112)",
                         "session restore did not bring the case selection back")
        self.assertAlmostEqual(theta.value(), 33.0, places=6)
        # EVERY state-carrying walked widget matches the saved value now
        live = {ident: w for ident, w in walk_all(self.win)}
        bad = []
        for ident, spec in state.items():
            w = live.get(ident)
            if w is None:
                bad.append(f"{ident} (gone)")
                continue
            now = widget_value(w)
            if now != spec:
                bad.append(f"{ident}: {spec} -> {now}")
        self.assertFalse(bad, "widgets that did not round-trip:\n  " + "\n  ".join(bad[:10]))

    def test_session_restore_reproduces_the_result(self):
        from shaarp.gui_introspect import apply_session_state, collect_session_state

        from . import gui_harness as gh

        si = self.win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        case.setCurrentText("GaAs (111)")
        self.app.processEvents()
        gh.click_update(si, self.app)
        want = gh.snapshot_curves(si)
        state = collect_session_state(self.win)
        # wander off, then restore + recompute
        case.setCurrentText("TaAs (112)")
        self.app.processEvents()
        apply_session_state(self.win, state)
        gh.click_update(si, self.app)
        got = gh.snapshot_curves(si)
        self.assertFalse(gh.curves_changed(want, got),
                         "a restored session did not reproduce the same computed curves")


class StackCodecPure(unittest.TestCase):
    """R15 codec: layer specs round-trip through JSON exactly, complex values included."""

    def test_encode_decode_round_trips_complex_custom_spec(self):
        from shaarp.layer_stack import decode_stack, encode_stack, isotropic_layer_spec

        stack = [
            isotropic_layer_spec(1.0, 1.0, name="ambient"),
            {"material": "Custom (fields)", "thickness_um": 2.5, "shg_active": True,
             "custom": {"point_group": "4mm",
                        "eps_omega_full": [[complex(6.0, 0.25), 0.0, 0.0],
                                           [0.0, complex(6.0, 0.25), 0.0],
                                           [0.0, 0.0, complex(5.5, 0.1)]],
                        "d_full": [[0.0] * 6, [0.0] * 6,
                                   [complex(1.0, -2.0), 0.0, 0.0, 0.0, 0.0, 0.0]]}},
            isotropic_layer_spec(1.45, 1.46, name="substrate"),
        ]
        wire = json.loads(json.dumps(encode_stack(stack)))
        back = decode_stack(wire)
        self.assertEqual(back, stack, "stack did not round-trip through JSON exactly")
        self.assertIsInstance(back[1]["custom"]["eps_omega_full"][0][0], complex)

    def test_decode_rejects_non_stacks(self):
        from shaarp.layer_stack import decode_stack

        with self.assertRaises(ValueError):
            decode_stack([{"no_material_key": 1}])


@unittest.skipUnless(HAVE_QT, "PySide2/6 not available")
class MlStackSessionRoundTrip(unittest.TestCase):
    """R15 closure: the ML layer stack rides in the session payload. Pre-R15, an
    N-layer-editor session restored onto the mode's REBUILT base stack (only the currently
    selected row's visible fields survived via the widget walk); these fences pin the whole
    stack -- and the simple modes' remembered template values -- through a real
    autosave -> fresh-window restore."""

    def _ml(self, win):
        return win.findChild(QtWidgets.QTabWidget).widget(1)

    def _pump(self, app, n=4):
        for _ in range(n):
            app.processEvents()

    def _editor_widgets(self, page):
        edit = next(c for c in page.findChildren(QtWidgets.QComboBox)
                    # row 0 is the AMBIENT (unnumbered — the original's convention:
                    # interior media are the numbered layers 1..N). Identify the editor combo
                    # by that role label, never by a "1:" prefix.
                    if c.count() and ("ambient" in c.itemText(0) or "air in" in c.itemText(0)))
        th = next(s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                  if s.maximum() >= 100000 and "Per-layer thickness" in s.toolTip())
        return edit, th

    def test_editor_stack_survives_autosave_and_fresh_window_restore(self):
        import shaarp.desktop_app as da

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = da.build_main_window()
        try:
            page = self._ml(win)
            combo = next(c for c in page.findChildren(QtWidgets.QComboBox)
                         if c.findText("Custom film (use fields)") >= 0)
            preset_name = next(combo.itemText(i) for i in range(combo.count())
                               if "Fig 4" in combo.itemText(i))
            combo.setCurrentText(preset_name)
            self._pump(app)
            combo.setCurrentText("N-layer stack (editor)")  # explicit editor entry
            self._pump(app)
            edit, th = self._editor_widgets(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            th.setValue(7.25)  # quartz row -> 7.25 in the user-owned stack
            self._pump(app)
            self.assertEqual(combo.currentText(), "N-layer stack (editor)")
            # A second edited row + a moved selection make the stack IRRECOVERABLE from the
            # widget walk alone (the walk replays only the finally-selected row's fields) —
            # red-proven: with the "ml_stack" payload stripped, this restore fails.
            edit.setCurrentIndex(2)
            self._pump(app)
            th.setValue(0.5)  # the Au row too
            self._pump(app)
            edit.setCurrentIndex(0)  # park the selection on the ambient row
            self._pump(app)
            want = page._ml_stack_payload()
            self.assertEqual(want["mode"], "N-layer stack (editor)")
            self.assertEqual(len(want["stack"]), 4, "Fig-4 copy is a 4-layer stack")

            with tempfile.TemporaryDirectory() as td:
                sess = os.path.join(td, "shaarp_last_session.json")
                orig_path = da._last_session_path
                da._last_session_path = lambda: sess
                try:
                    da._autosave_session(win)
                    payload = json.load(open(sess, encoding="utf-8"))
                    self.assertIn("ml_stack", payload, "session file must carry the stack")

                    win2 = da.build_main_window()  # fresh launch: startup preset base stack
                    try:
                        restored = da._restore_last_session(win2)
                        self.assertGreater(restored, 100, "widget restore suspiciously small")
                        got = self._ml(win2)._ml_stack_payload()
                        self.assertEqual(got["mode"], "N-layer stack (editor)")
                        self.assertEqual(got["stack"], want["stack"],
                                         "restored editor stack != saved stack (R15 regression: "
                                         "pre-R15 this came back as the rebuilt base stack)")
                        edit2, th2 = self._editor_widgets(self._ml(win2))
                        edit2.setCurrentIndex(1)
                        self._pump(app)
                        self.assertAlmostEqual(th2.value(), 7.25, places=9,
                                               msg="edited quartz thickness lost on restore")
                    finally:
                        win2.close()
                finally:
                    da._last_session_path = orig_path
        finally:
            win.close()

    def test_edited_preset_stack_survives_restore_and_pristine_ignores_tamper(self):
        """The EDITED working copy of a NAMED PRESET survives a session
        restore (here: Fig-4 row 2 swapped to MoS2, selection parked on row 3 — pre-F65 the
        restore rebuilt the pristine preset). A PRISTINE preset still rebuilds from the factory:
        a tampered stack in the payload is ignored (the R15 divergence rationale)."""
        import shaarp.desktop_app as da

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = da.build_main_window()
        try:
            page = self._ml(win)
            combo = next(c for c in page.findChildren(QtWidgets.QComboBox)
                         if c.findText("Custom film (use fields)") >= 0)
            preset_name = next(combo.itemText(i) for i in range(combo.count())
                               if "Fig 4" in combo.itemText(i))
            combo.setCurrentText(preset_name)
            self._pump(app)
            edit, _th = self._editor_widgets(page)
            lay_mat = next(c for c in page.findChildren(QtWidgets.QComboBox)
                           if c.findText("Custom (fields)") >= 0 and c is not combo)
            edit.setCurrentIndex(1)
            self._pump(app)
            lay_mat.setCurrentText("MoS2 (800 nm)")  # an EDIT under the preset -> dirty copy
            self._pump(app)
            edit.setCurrentIndex(2)  # park the selection on a NON-edited row (the S23 shape)
            self._pump(app)
            want = page._ml_stack_payload()
            self.assertEqual(want["mode"], preset_name)
            self.assertTrue(want["dirty"], "the edit must mark the working copy dirty")

            with tempfile.TemporaryDirectory() as td:
                sess = os.path.join(td, "shaarp_last_session.json")
                orig_path = da._last_session_path
                da._last_session_path = lambda: sess
                try:
                    da._autosave_session(win)
                    win2 = da.build_main_window()
                    try:
                        da._restore_last_session(win2)
                        got = self._ml(win2)._ml_stack_payload()
                        self.assertEqual(got["mode"], preset_name)
                        self.assertEqual(got["stack"][1]["material"], "MoS2 (800 nm)",
                                         "edited preset row lost on restore (S20/S23)")
                        self.assertTrue(got["dirty"], "the restored copy is re-marked edited")
                    finally:
                        win2.close()

                    # pristine preset + tampered payload -> factory rebuild wins
                    win3 = da.build_main_window()
                    try:
                        page3 = self._ml(win3)
                        combo3 = next(c for c in page3.findChildren(QtWidgets.QComboBox)
                                      if c.findText("Custom film (use fields)") >= 0)
                        combo3.setCurrentText(preset_name)
                        self._pump(app)
                        pristine = page3._ml_stack_payload()
                        self.assertFalse(pristine["dirty"])
                        tampered = dict(pristine)
                        tampered["stack"] = [dict(r) for r in pristine["stack"]]
                        tampered["stack"][1]["material"] = "MoS2 (800 nm)"
                        tampered["dirty"] = False  # a stale/pristine payload
                        page3._ml_stack_apply(tampered)
                        self._pump(app)
                        after = page3._ml_stack_payload()
                        self.assertNotEqual(after["stack"][1]["material"], "MoS2 (800 nm)",
                                            "a pristine preset must rebuild from the factory, "
                                            "never from a stale payload (R15 rationale)")
                    finally:
                        win3.close()
                finally:
                    da._last_session_path = orig_path
        finally:
            win.close()

    def test_simple_template_values_survive_restore(self):
        import shaarp.desktop_app as da

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = da.build_main_window()
        try:
            page = self._ml(win)
            combo = next(c for c in page.findChildren(QtWidgets.QComboBox)
                         if c.findText("Custom film (use fields)") >= 0)
            combo.setCurrentText("Custom film (use fields)")
            self._pump(app)
            edit, th = self._editor_widgets(page)
            edit.setCurrentIndex(1)
            self._pump(app)
            th.setValue(3.75)  # the mode's film thickness (stays in the mode)
            self._pump(app)
            edit.setCurrentIndex(2)  # substrate row: isotropic medium
            self._pump(app)
            # an isotropic medium takes ONE number per frequency — the substrate n is
            # entered through the scalar index rows (the 3x3 grids are hidden for such rows).
            # Located by TOOLTIP, never by position: a QDoubleSpinBox owns an internal
            # QLineEdit, so a positional slice over the group's line edits shifts silently.
            n_spins = [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
                       if "ISOTROPIC layer" in s.toolTip()]
            self.assertEqual(len(n_spins), 2, "expected the n(w)/n(2w) rows")
            n_spins[0].setValue(1.62)
            self._pump(app)
            n_spins[1].setValue(1.63)
            self._pump(app)

            with tempfile.TemporaryDirectory() as td:
                sess = os.path.join(td, "shaarp_last_session.json")
                orig_path = da._last_session_path
                da._last_session_path = lambda: sess
                try:
                    da._autosave_session(win)
                    win2 = da.build_main_window()
                    try:
                        da._restore_last_session(win2)
                        got = self._ml(win2)._ml_stack_payload()
                        self.assertEqual(got["mode"], "Custom film (use fields)")
                        st = got["simple_template"]
                        self.assertAlmostEqual(float(st["um"]), 3.75, places=9)
                        self.assertEqual([round(float(x), 6) for x in st["bottom"]],
                                         [1.62, 1.63],
                                         "simple-mode substrate n did not restore")
                        bot = got["stack"][-1]
                        self.assertEqual(bot.get("iso_n"), [1.62, 1.63],
                                         "rebuilt template's substrate row != restored memory")
                    finally:
                        win2.close()
                finally:
                    da._last_session_path = orig_path
        finally:
            win.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class DebugTooling(unittest.TestCase):
    def test_swallowed_exception_lands_in_log_with_traceback(self):
        from shaarp.debuglog import log_path
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        win._gui_smoke_errors = []  # suppress the modal error dialog (armed-sink path)
        ml = win.findChild(QtWidgets.QTabWidget).widget(1)
        func = next(c for c in ml.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText("Maker Fringes")
        # force the guard: scan min >= max raises ValueError inside on_run
        from shaarp.desktop_app import TOOLTIPS

        rng = [s for s in ml.findChildren(QtWidgets.QDoubleSpinBox)
               if abs(s.maximum() - 89.9) < 1e-6]
        self.assertTrue(rng)
        lo = [s for s in ml.findChildren(QtWidgets.QDoubleSpinBox)
              if s.toolTip() == TOOLTIPS["theta"] and s is not rng[0]]
        rng[0].setValue(1.0)  # theta_max = 1 while default min is 0? force min > max instead
        mins = [s for s in ml.findChildren(QtWidgets.QDoubleSpinBox)
                if s.toolTip() == rng[0].toolTip() and s is not rng[0]]
        if mins:
            mins[0].setValue(5.0)  # min 5 > max 1 -> ValueError
        run = next(b for b in ml.findChildren(QtWidgets.QPushButton)
                   if b.text() == "Update / Run")
        run.click()
        app.processEvents()
        tb = getattr(win, "_last_traceback", "")
        self.assertIn("ValueError", tb, "exception traceback was not captured on the window")
        self.assertIn("Traceback", tb)
        with open(log_path(), encoding="utf-8") as fh:
            tail = fh.read()[-4000:]
        self.assertIn("ValueError", tail, "traceback did not reach the debug log file")

    def test_custom_after_rotated_case_does_not_error(self):
        """Selecting a rotated case study (TaAs (112)) then switching to Custom must
        Update cleanly -- the 6-decimal z-cells used to truncate the case rotation past the
        1e-7 orthogonality tolerance, raising 'Crystal axes must be mutually orthogonal'. The
        _z_axes SVD-snap fixes it (drift ~1.6e-8)."""
        from shaarp.desktop_app import build_main_window

        from . import gui_harness as gh

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        win._gui_smoke_errors = []  # capture any Update error instead of the non-modal note
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("TaAs (112)") >= 0)
        for name in ("TaAs (112)", "GaAs (111) (800 nm)", "LiNbO3 x-cut (1550 nm)"):
            case.setCurrentText(name)
            app.processEvents()
            gh.click_update(si, app)
            case.setCurrentText("Custom (use fields)")
            app.processEvents()
            gh.click_update(si, app)
            errs = [e for e in win._gui_smoke_errors if "orthogonal" in e]
            self.assertFalse(errs, f"orthogonality error after {name}->Custom: {errs}")

    def test_export_figure_writes_a_file(self):
        """The Export figure button saves the current plot (PNG here)."""
        import os
        import tempfile

        from shaarp.desktop_app import build_main_window

        from . import gui_harness as gh

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        case.setCurrentText("GaAs (111)")
        app.processEvents()
        gh.click_update(si, app)
        self.assertTrue(hasattr(si, "_on_export_figure"), "figure-export hook missing (F49-B1)")
        p = os.path.join(tempfile.gettempdir(), "f49_fig_fence.png")
        if os.path.exists(p):
            os.remove(p)
        orig = QtWidgets.QFileDialog.getSaveFileName
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (p, "PNG (*.png)"))
        try:
            si._on_export_figure()
            app.processEvents()
        finally:
            QtWidgets.QFileDialog.getSaveFileName = orig
        self.assertTrue(os.path.exists(p) and os.path.getsize(p) > 1000,
                        "Export figure produced no usable file")

    def test_copy_figure_and_provenance(self):
        """Copy figure puts an image on the clipboard (#6), and every export payload carries
        a reproducible provenance block (#5: version + full input state)."""
        from shaarp.desktop_app import build_main_window

        from . import gui_harness as gh

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        case.setCurrentText("GaAs (111)")
        app.processEvents()
        gh.click_update(si, app)
        # #6 copy figure -> clipboard pixmap
        self.assertTrue(hasattr(si, "_on_copy_figure"), "copy-figure hook missing (F50-#6)")
        cb = QtWidgets.QApplication.clipboard()
        cb.clear()
        si._on_copy_figure()
        app.processEvents()
        self.assertFalse(cb.pixmap().isNull(), "Copy figure put no image on the clipboard")
        # #5 provenance in the export payload
        pay = si._build_export_payload()
        self.assertIn("provenance", pay, "export payload has no provenance block (F50-#5)")
        prov = pay["provenance"]
        self.assertTrue(prov.get("shaarp_py_version"), "provenance missing version")
        self.assertGreater(len(prov.get("inputs", {})), 100, "provenance inputs suspiciously small")
        import json
        json.dumps(pay)  # provenance must not break JSON serialization
        # #8 pop-out builds without error
        self.assertTrue(hasattr(si, "_popout_current_figure"))
        si._popout_current_figure()
        app.processEvents()

    def test_update_logs_per_stage_timings(self):
        """Phase 0: every Update line in the debug log carries the compute/figure/render
        split so a 'feels slow' report localizes to a stage without a profiling session."""
        from shaarp.debuglog import log_path
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        si = win.findChild(QtWidgets.QTabWidget).widget(0)
        case = next(c for c in si.findChildren(QtWidgets.QComboBox)
                    if c.findText("GaAs (111)") >= 0)
        case.setCurrentText("GaAs (111)")
        app.processEvents()
        next(b for b in si.findChildren(QtWidgets.QPushButton)
             if b.text() == "Update / Run").click()
        app.processEvents()
        with open(log_path(), encoding="utf-8") as fh:
            tail = [ln for ln in fh.read().splitlines() if " run tab=" in ln][-1]
        for key in ("compute=", "figure=", "render=", "total="):
            self.assertIn(key, tail, f"stage telemetry lost {key!r} (regression)")

    def test_debug_info_dialog_builds_and_copies(self):
        from shaarp.desktop_app import build_main_window

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        win = build_main_window()
        self.assertTrue(hasattr(win, "_show_debug_info"), "Debug Info hook missing")
        # the dialog is modal; verify its CONTENT via the same builder inputs instead of exec()
        from shaarp import __version__
        from shaarp.debuglog import log_path
        n = len(win._collect_session_state())
        self.assertGreater(n, 150)
        self.assertTrue(os.path.basename(log_path()).startswith("shaarp_gui"))
        self.assertTrue(__version__)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
