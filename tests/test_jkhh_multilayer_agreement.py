"""Gated agreement test for the JK / HH Maker SHG assumption modes on a MULTILAYER
stack (4-layer: air / active-film / isotropic-interlayer / substrate), mrassumption 1/2.

Companion to test_jkhh_maker_agreement.py (single film). This exercises the f4L
interior-interface For-loop in solve_multilayer_boundary_single_pass: the single-
pass solve sweeps entrance -> interior interface(s) -> exit, carrying the per-layer
inhomogeneous SHG sources through each interface (JK), with the flagHH propagated-
forward writeback (HH).

Validated vs the live SHAARP.ml MF reference exported under mrassumption 0/1/2 for
the ml1 4-layer case (jkhh_ml_validation_targets_v1.json): full/JK/HH all match to
~2e-15. The gate goes live automatically once the sweep honors `mrassumption`.
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "jkhh_ml_validation_targets_v1.json"
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


class JKHHMultilayerTargetsPresentTests(unittest.TestCase):
    def test_targets_present_distinct_and_four_layer(self):
        self.assertTrue(TARGETS_PATH.exists(), f"missing {TARGETS_PATH}")
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        full, jk, hh = _para_amp(t["full_listMFpara"]), _para_amp(t["jk_listMFpara"]), _para_amp(t["hh_listMFpara"])
        self.assertEqual(len(full), 4)
        self.assertGreater(np.max(np.abs(jk - full)), 1e-3, "JK should differ from full")
        self.assertGreater(np.max(np.abs(hh - full)), 1e-3, "HH should differ from full")


@unittest.skipUnless(
    _supports_jkhh(),
    "solver does not expose `mrassumption`; multilayer JK/HH gate goes live once it does.",
)
class JKHHMultilayerAgreementTests(unittest.TestCase):
    def _run(self, mrassumption: int) -> np.ndarray:
        from benchmarks.generate_maker_fringes_multilayer_benchmarks import build_multilayer_cases

        case = build_multilayer_cases(limit=1)[0]
        self.assertEqual(len(case["system"].layers), 4, "expected a 4-layer multilayer case")
        sweep = solve_multilayer_maker_fringes_sweep(
            case["system"], theta_deg=case["theta_deg"], mu=1.0, eps0=1.0, mrassumption=mrassumption
        )
        arr = np.array(sweep.shaarp_ml_copy_lists()[0])
        col = arr[:, 1] if arr.ndim == 2 else arr
        return np.abs(col.astype(complex))

    def test_full_multilayer_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(0), _para_amp(t["full_listMFpara"]), atol=ATOL)

    def test_jk_multilayer_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(1), _para_amp(t["jk_listMFpara"]), atol=ATOL)

    def test_hh_multilayer_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(2), _para_amp(t["hh_listMFpara"]), atol=ATOL)


if __name__ == "__main__":
    unittest.main()
