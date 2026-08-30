"""Gated agreement test: the FMR backward / standing-wave source policies vs LIVE SHAARP.ml.

The original FMR mode's winhAssumption sub-variants (0 = no backward, 1 = backward, 2 = backward +
standing) are reproduced by the Python multilayer solver's inhomogeneous_source_policy
(forward_only / forward_backward / all). These live-Mathematica reference MFLists were produced by
benchmarks/verify_fmr_backward_vs_mathematica.py (quartz+Au docs system, 20-30 deg). This test is
COMPARE-ONLY: it re-runs the Python policies and checks them against the stored references; it skips
cleanly if the references are absent (no Wolfram kernel needed at test time).
"""

import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.quartz_au_docs_case import build_quartz_au_system
from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

MR = Path(__file__).resolve().parents[1] / "benchmarks" / "mathematica_reference"
WINH_TO_POLICY = {0: "forward_only", 1: "forward_backward", 2: "all"}
TOL = 1e-6  # live-Mathematica agreement (measured ~1e-9; generous gate)


def _find_mflist(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "MFList":
                return v
            f = _find_mflist(v)
            if f is not None:
                return f
    if isinstance(o, list):
        for x in o:
            f = _find_mflist(x)
            if f is not None:
                return f
    return None


def _cval(x):
    return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)


class FmrBackwardLiveAgreementTests(unittest.TestCase):
    def setUp(self):
        self.window = [round(20.0 + 1.0 * i, 4) for i in range(11)]
        self.sysm = build_quartz_au_system()

    def _check(self, winh: int):
        ref = MR / f"fmr_winh{winh}_reference.json"
        if not ref.exists():
            self.skipTest(f"no live reference {ref.name} (run verify_fmr_backward_vs_mathematica)")
        mf = _find_mflist(json.loads(ref.read_text(encoding="utf-8")))
        self.assertIsNotNone(mf, "MFList missing from reference")
        mma_par = np.abs([_cval(row[1]) for row in mf])
        sw = solve_multilayer_maker_fringes_sweep(
            self.sysm, theta_deg=self.window, mrassumption=0,
            inhomogeneous_source_policy=WINH_TO_POLICY[winh], mu=1.0, eps0=1.0)
        py_par = np.abs(np.asarray(sw.parallel_intensity))
        err = float(np.max(np.abs(mma_par - py_par)))
        self.assertLess(err, TOL, f"winh={winh} ({WINH_TO_POLICY[winh]}) vs live SHAARP.ml: {err:.2e}")

    def test_winh0_forward_only_sanity(self):
        self._check(0)

    def test_winh1_backward_waves(self):
        self._check(1)

    def test_winh2_backward_standing_waves(self):
        self._check(2)

    def test_policies_are_distinct(self):
        # the three FMR sub-modes must give genuinely different curves (not a no-op control)
        curves = {}
        for winh, pol in WINH_TO_POLICY.items():
            sw = solve_multilayer_maker_fringes_sweep(
                self.sysm, theta_deg=self.window, mrassumption=0,
                inhomogeneous_source_policy=pol, mu=1.0, eps0=1.0)
            curves[winh] = np.abs(np.asarray(sw.parallel_intensity))
        self.assertGreater(float(np.max(np.abs(curves[0] - curves[1]))), 1e-6)
        self.assertGreater(float(np.max(np.abs(curves[0] - curves[2]))), 1e-6)


if __name__ == "__main__":
    unittest.main()
