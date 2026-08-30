"""3b / fences: the incident-medium index input.

The GUI historically hardwired the incident medium to air on both tabs. F47-3b exposes it:

  * SI tab — an "Incident Medium (isotropic n)" group; the pair reaches the numeric polarimetry
    solver (`solve_single_interface_shg`, whose `incident_index_*` kwargs were always plumbed but
    never exposed). A non-air pair routes the closed form to the validated numeric branch — the
    symbolic derivation carries n0 = 1 structurally (kx = w*sin(theta)).
  * ML tab — the incident medium IS the stack's first layer (both half-spaces live in the
    layer editor; the simple templates carry isotropic-n rows). The Partial-Analytical closed
    form now reads the stack's true ambient instead of a hardwired 1.0.

Fence tiers (the scoping, -approved): equal-results at air (n = 1 reproduces every prior
result EXACTLY), causality (n != 1 must change the result), physics sanity (an index-matched
ambient kills the linear reflection — un-fakeable Fresnel physics).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    HAVE_QT = False


class SiCurveIncidentIndex(unittest.TestCase):
    KW = dict(theta_deg=45.0, n_omega=2.0, n_2omega=2.2, n_phi=61)

    def test_air_default_is_byte_identical(self):
        from shaarp.shaarp_gui import si_polarimetry_curve
        base = si_polarimetry_curve("-43m", **self.KW)
        same = si_polarimetry_curve("-43m", incident_index_omega=1.0,
                                    incident_index_2omega=1.0, **self.KW)
        for key in ("intensity_p", "intensity_s"):
            np.testing.assert_array_equal(np.asarray(base[key]), np.asarray(same[key]),
                                          err_msg=f"explicit n0=1 must be BYTE-identical ({key})")
        self.assertGreater(float(np.max(np.asarray(base["intensity_p"]))), 0.0,
                           "anti-vacuous: the baseline curve must be nonzero")

    def test_non_air_changes_the_curve(self):
        from shaarp.shaarp_gui import si_polarimetry_curve
        base = si_polarimetry_curve("-43m", **self.KW)
        immersed = si_polarimetry_curve("-43m", incident_index_omega=1.5,
                                        incident_index_2omega=1.52, **self.KW)
        diff = max(float(np.max(np.abs(np.asarray(base[k]) - np.asarray(immersed[k]))))
                   for k in ("intensity_p", "intensity_s"))
        rel = diff / float(np.max(np.abs(np.asarray(base["intensity_p"]))))
        self.assertGreater(rel, 1e-6,
                           "DEAD INPUT: a 1.5/1.52 incident medium changed nothing")

    def test_index_matched_ambient_kills_linear_reflection(self):
        """n0 = n_crystal (lossless isotropic): the interface disappears for the fundamental —
        the reflected omega amplitudes must vanish at machine precision. Un-fakeable Fresnel."""
        from shaarp.shg import solve_single_interface_shg

        n = 1.75
        eps = (np.eye(3) * n ** 2).astype(complex)
        d = np.zeros((3, 6), dtype=complex)
        d[0, 0] = 1.0  # any nonzero d; the LINEAR reflection is the observable here
        r = solve_single_interface_shg(
            eps, eps, d, incident_index_omega=n, incident_index_2omega=n,
            incident_theta_rad=np.deg2rad(35.0), incident_jones=(0.6, 0.8),
            omega=1.0, mu=1.0, eps0=1.0)
        lin = r.linear_omega
        refl = [abs(complex(lin.r_s)), abs(complex(lin.r_p))]
        self.assertLess(max(refl), 1e-10,
                        f"index-matched ambient must kill the linear reflection, got {refl}")


class MlPartialAnalyticalAmbient(unittest.TestCase):
    CASE = {"point_group": "-43m", "incident_theta_rad": 0.5}

    def test_air_default_renders_identical_expressions(self):
        from shaarp.api import run_ml_partial_analytical
        a = run_ml_partial_analytical(dict(self.CASE), {"workflow": "polarimetry"})
        b = run_ml_partial_analytical(
            dict(self.CASE, ambient_index_omega=1.0, ambient_index_2omega=1.0),
            {"workflow": "polarimetry"})
        self.assertEqual(a.stages["reflected_p_2omega"], b.stages["reflected_p_2omega"])
        self.assertEqual(a.stages["reflected_s_2omega"], b.stages["reflected_s_2omega"])
        self.assertNotIn("reflected_p_2omega = 0", str(a.stages["reflected_p_2omega"]),
                         "anti-vacuous: the -43m closed form must be nonzero")

    def test_non_air_ambient_changes_the_expressions(self):
        from shaarp.api import run_ml_partial_analytical
        a = run_ml_partial_analytical(dict(self.CASE), {"workflow": "polarimetry"})
        c = run_ml_partial_analytical(
            dict(self.CASE, ambient_index_omega=1.33, ambient_index_2omega=1.34),
            {"workflow": "polarimetry"})
        self.assertNotEqual(a.stages["reflected_p_2omega"], c.stages["reflected_p_2omega"],
                            "DEAD INPUT: the PA closed form ignored the ambient index")


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class SiGuiIncidentInput(unittest.TestCase):
    """The GUI wiring: the SI Incident Medium spins exist, default to air, and drive the
    rendered polarimetry curves through Update (the dead-control standard)."""

    @classmethod
    def setUpClass(cls):
        from shaarp.desktop_app import build_main_window
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.win = build_main_window()
        cls.si = cls.win.findChild(QtWidgets.QTabWidget).widget(0)

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def _pump(self, n=4):
        for _ in range(n):
            self.app.processEvents()

    def _update(self):
        next(b for b in self.si.findChildren(QtWidgets.QPushButton)
             if b.text() == "Update").click()
        self._pump(6)

    def _snap(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        out = []
        for cv in self.si.findChildren(FigureCanvasQTAgg):
            if cv.objectName().startswith("orient_canvas"):
                continue
            for ax in cv.figure.axes:
                for ln in ax.lines:
                    out.append(np.asarray(ln.get_ydata(), dtype=float))
        return out

    def test_incident_spins_drive_the_si_curves(self):
        spins = [s for s in self.si.findChildren(QtWidgets.QDoubleSpinBox)
                 if "INCIDENT medium" in s.toolTip()]
        self.assertEqual(len(spins), 2, "SI must expose the incident n(w)/n(2w) pair")
        self.assertEqual([s.value() for s in spins], [1.0, 1.0], "default = air")
        func = next(c for c in self.si.findChildren(QtWidgets.QComboBox)
                    if c.toolTip().startswith("Choose what to calculate"))
        func.setCurrentText("SHG Simulation")
        # PIN an oblique incidence -- do not inherit the panel default. At normal incidence
        # (the default) s and p are degenerate, so the ambient index is a pure amplitude factor
        # that the normalised comparison below divides out: measured diff/scale = 5.0e-13 at 0 deg
        # vs 9.1e-04 at 45 deg. The control is live either way; the FENCE is blind at 0, so the
        # angle it needs must be stated, not inherited.
        from shaarp.desktop_app import TOOLTIPS

        theta = next(t for t in self.si.findChildren(QtWidgets.QDoubleSpinBox)
                     if t.toolTip() == TOOLTIPS["theta"])
        _theta_restore = theta.value()
        theta.setValue(45.0)
        self._pump()
        self._update()
        before = self._snap()
        self.assertTrue(any(a.size for a in before), "SI baseline produced no curves")
        for s, v in zip(spins, (1.5, 1.52)):
            s.setValue(v)
        self._pump()
        self._update()
        after = self._snap()
        scale = max((float(np.nanmax(np.abs(a))) for a in before if a.size), default=0.0) + 1e-300
        diff = max((float(np.nanmax(np.abs(a - b)))
                    for a, b in zip(before, after)
                    if a.size and a.shape == b.shape), default=0.0)
        self.assertGreater(diff / scale, 1e-6,
                           "DEAD CONTROL: the incident-medium spins changed nothing in the "
                           "rendered SI curves")
        for s in spins:  # leave the shared window in the default state
            s.setValue(1.0)
        theta.setValue(_theta_restore)  # restore the incidence default for the shared window
        self._pump()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
