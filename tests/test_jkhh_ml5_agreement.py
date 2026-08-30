"""Gated agreement test for JK/HH on a 5-LAYER stack (nM=3 internal layers).

Where test_jkhh_multilayer_agreement.py validates a 4-layer stack (nM=2 -> ONE
interior interface, so the f4L single-pass For-loop body runs once), this validates
a 5-layer ml2 stack (air / active-film / ANISOTROPIC passive interlayer / isotropic
interlayer / substrate; nM=3 -> TWO interior interfaces). It is the decisive check
that the ported f4L interior-interface For-loop genuinely ITERATES across more than
one interior interface, and that JK/HH handle an anisotropic (non-degenerate) passive
interior layer.

Validated vs live SHAARP.ml MF exported under mrassumption 0/1/2 for the ml2 5-layer
case0 (jkhh_ml5_validation_targets_v1.json): full/JK/HH all match to ~2e-15. The gate
goes live automatically once the sweep honors `mrassumption`.
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "jkhh_ml5_validation_targets_v1.json"
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


class JKHHFiveLayerTargetsTests(unittest.TestCase):
    def test_targets_present_distinct_and_five_layer(self):
        self.assertTrue(TARGETS_PATH.exists(), f"missing {TARGETS_PATH}")
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        full = _para_amp(t["full_listMFpara"])
        jk = _para_amp(t["jk_listMFpara"])
        hh = _para_amp(t["hh_listMFpara"])
        self.assertGreater(np.max(np.abs(jk - full)), 1e-4, "JK should differ from full")
        self.assertGreater(np.max(np.abs(hh - full)), 1e-4, "HH should differ from full")


@unittest.skipUnless(
    _supports_jkhh(),
    "solver does not expose `mrassumption`; 5-layer JK/HH gate goes live once it does.",
)
class JKHHFiveLayerAgreementTests(unittest.TestCase):
    def _run(self, mrassumption: int) -> np.ndarray:
        from benchmarks.generate_maker_fringes_multilayer_ml2_benchmarks import build_ml2_cases

        case = build_ml2_cases(limit=1)[0]
        self.assertEqual(len(case["system"].layers), 5, "expected a 5-layer (nM=3) stack")
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
