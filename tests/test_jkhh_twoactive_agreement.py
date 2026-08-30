"""Gated agreement test for JK/HH with TWO adjacent ACTIVE layers.

The single/4-layer/5-layer/point-group JK/HH tests all have the SHG source in the
FIRST internal layer only (the other internal layers are passive). This one uses a
pg2act stack with TWO adjacent active films, so the single-pass interior-interface
solve must carry inhomogeneous SHG source waves from a NON-first internal layer
(the `layer_inhomogeneous[i>0]` branch) -- a code path the single-active cases never
exercise.

Validated vs live SHAARP.ml MF exported under mrassumption 0/1/2 for the pg2act
two-active case0 (jkhh_2act_validation_targets_v1.json): full/JK/HH all match to ~1e-15.
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "jkhh_2act_validation_targets_v1.json"
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


class JKHHTwoActiveTargetsTests(unittest.TestCase):
    def test_targets_present_and_distinct(self):
        self.assertTrue(TARGETS_PATH.exists(), f"missing {TARGETS_PATH}")
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        full = _para_amp(t["full_listMFpara"])
        self.assertGreater(np.max(np.abs(_para_amp(t["jk_listMFpara"]) - full)), 1e-4)
        self.assertGreater(np.max(np.abs(_para_amp(t["hh_listMFpara"]) - full)), 1e-4)


@unittest.skipUnless(
    _supports_jkhh(),
    "solver does not expose `mrassumption`; two-active JK/HH gate goes live once it does.",
)
class JKHHTwoActiveAgreementTests(unittest.TestCase):
    def _run(self, mrassumption: int) -> np.ndarray:
        from benchmarks.generate_maker_fringes_pg2act_benchmarks import build_pg2act_cases

        case = build_pg2act_cases(limit=1)[0]
        active = [bool(getattr(layer, "shg_active", False)) for layer in case["system"].layers]
        self.assertGreaterEqual(sum(active), 2, "expected at least two SHG-active layers")
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
