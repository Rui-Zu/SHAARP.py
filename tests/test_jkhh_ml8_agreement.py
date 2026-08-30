"""Gated agreement test for JK/HH at MAXIMUM depth -- a 9-LAYER stack (nM=7).

This is the deepest the single-pass f4L interior For-loop iterates: 9 layers ->
7 internal -> 6 interior-interface iterations. Together with the 4-layer (1
iteration) and 5-layer (2 iterations) tests it brackets the For-loop across its
range, confirming no depth-specific accumulation issue under JK/HH.

Validated vs live SHAARP.ml MF (ml8 9-layer case0, mrassumption 0/1/2;
jkhh_ml8_validation_targets_v1.json): full/JK/HH all match to ~1.6e-15.
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "jkhh_ml8_validation_targets_v1.json"
ATOL = 5e-4


def _supports_jkhh() -> bool:
    try:
        params = inspect.signature(solve_multilayer_maker_fringes_sweep).parameters
    except (TypeError, ValueError):
        return False
    return "mrassumption" in params


def _para_amp(pairs: list) -> np.ndarray:
    def num(x):
        return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)

    return np.array([abs(num(row[-1])) for row in pairs], dtype=float)


class JKHHMaxDepthTargetsTests(unittest.TestCase):
    def test_targets_present_and_distinct(self):
        self.assertTrue(TARGETS_PATH.exists(), f"missing {TARGETS_PATH}")
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        full = _para_amp(t["full_listMFpara"])
        self.assertGreater(np.max(np.abs(_para_amp(t["jk_listMFpara"]) - full)), 1e-5)
        self.assertGreater(np.max(np.abs(_para_amp(t["hh_listMFpara"]) - full)), 1e-5)


@unittest.skipUnless(
    _supports_jkhh(),
    "solver does not expose `mrassumption`; 9-layer JK/HH gate goes live once it does.",
)
class JKHHMaxDepthAgreementTests(unittest.TestCase):
    def _run(self, mrassumption: int) -> np.ndarray:
        from benchmarks.generate_maker_fringes_multilayer_ml8_benchmarks import build_ml8_cases

        case = build_ml8_cases(limit=1)[0]
        self.assertEqual(len(case["system"].layers), 9, "expected a 9-layer (nM=7) stack")
        sweep = solve_multilayer_maker_fringes_sweep(
            case["system"], theta_deg=case["theta_deg"], mu=1.0, eps0=1.0, mrassumption=mrassumption
        )
        para, _ = sweep.shaarp_ml_copy_lists()
        arr = np.array(para)
        col = arr[:, 1] if arr.ndim == 2 else arr
        return np.abs(col.astype(complex))

    def test_full_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(0), _para_amp(t["full_listMFpara"]), atol=ATOL)

    def test_jk_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(1), _para_amp(t["jk_listMFpara"]), atol=ATOL)

    def test_hh_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(2), _para_amp(t["hh_listMFpara"]), atol=ATOL)


if __name__ == "__main__":
    unittest.main()
