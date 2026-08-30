"""Gated regression test for the 0.1-degree Maker-fringe agreement vs LIVE Mathematica.

This locks in the paper-rigor (0.1 deg) angular-sampling proof (Zu et al., npj Comput. Mater.
10, 64 (2024)) as an un-fakeable, tolerance-gated test: SHAARP.py reproduces the live
SHAARP.ml Maker MFList at 0.1 deg to machine precision. It is GATED on the stored live-Wolfram
reference (benchmarks/.../maker_0p1_pilot_reference_v1.json); if that artifact is absent
(e.g. on a machine without the Wolfram export), the test SKIPS rather than fabricating a pass.

Regenerate the reference with: python benchmarks/run_maker_0p1_pilot.py
"""

import json
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "benchmarks" / "mathematica_reference" / "maker_0p1_pilot_reference_v1.json"


def _find_mflist(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "MFList":
                return value
            found = _find_mflist(value)
            if found is not None:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = _find_mflist(item)
            if found is not None:
                return found
    return None


def _c(x):
    return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)


@unittest.skipUnless(REF.exists(), "live-Wolfram 0.1deg Maker reference absent; run benchmarks/run_maker_0p1_pilot.py")
class Maker0p1DegLiveMathematicaTests(unittest.TestCase):
    def test_shaarp_py_matches_live_mathematica_at_0p1_deg(self):
        from benchmarks.generate_maker_fringes_benchmarks import build_cases
        from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

        mf = _find_mflist(json.loads(REF.read_text(encoding="utf-8")))
        self.assertIsNotNone(mf, "reference has no MFList")
        theta = [float(_c(row[0]).real) for row in mf]
        mma_para = np.array([_c(row[1]) for row in mf])
        mma_perp = np.array([_c(row[2]) for row in mf])
        # step is 0.1 deg (paper rigor)
        steps = np.diff(theta)
        self.assertTrue(np.allclose(steps, 0.1, atol=1e-6), f"reference not at 0.1deg step: {steps[:3]}")

        # SHAARP.py via the SAME call the validated benchmark uses (run_case): mu=1, eps0=1.
        sweep = solve_multilayer_maker_fringes_sweep(
            build_cases()[0]["system"], theta_deg=theta, mrassumption=0, mu=1.0, eps0=1.0,
        )
        py_para = np.asarray(sweep.parallel_intensity)
        py_perp = np.asarray(sweep.perpendicular_intensity)

        scale = max(float(np.max(np.abs(mma_para))), float(np.max(np.abs(mma_perp))), 1e-300)
        self.assertGreater(scale, 1e-6, "near-zero reference -> vacuous")
        para_err = float(np.max(np.abs(mma_para - py_para)))
        perp_err = float(np.max(np.abs(mma_perp - py_perp)))
        self.assertLess(para_err, 1e-12 + 1e-9 * scale, f"0.1deg para deviates from live Mathematica: {para_err:.2e}")
        self.assertLess(perp_err, 1e-12 + 1e-9 * scale, f"0.1deg perp deviates from live Mathematica: {perp_err:.2e}")


if __name__ == "__main__":
    unittest.main()
