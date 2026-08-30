"""Gated agreement test for JK/HH with a SYMMETRY-CONSTRAINED point-group d-tensor.

The single-film / 4-layer / 5-layer JK/HH agreement tests use the build_system
default d. This one confirms JK/HH (mrassumption 1/2) also reproduce live SHAARP.ml
when the active film carries a real crystallographic point-group d-tensor (pg1
case0, class 3m) -- i.e. the assumption modes work with symmetry-constrained SHG
sources, the realistic crystal case.

Validated vs live SHAARP.ml MF exported under mrassumption 0/1/2 for pg1 case0
(jkhh_pg_validation_targets_v1.json): full/JK/HH all match to ~1.2e-15.
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "jkhh_pg_validation_targets_v1.json"
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


class JKHHPointGroupTargetsTests(unittest.TestCase):
    def test_targets_present_and_distinct(self):
        self.assertTrue(TARGETS_PATH.exists(), f"missing {TARGETS_PATH}")
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        full = _para_amp(t["full_listMFpara"])
        self.assertGreater(np.max(np.abs(_para_amp(t["jk_listMFpara"]) - full)), 1e-4)
        self.assertGreater(np.max(np.abs(_para_amp(t["hh_listMFpara"]) - full)), 1e-4)


@unittest.skipUnless(
    _supports_jkhh(),
    "solver does not expose `mrassumption`; point-group JK/HH gate goes live once it does.",
)
class JKHHPointGroupAgreementTests(unittest.TestCase):
    def _run(self, mrassumption: int) -> np.ndarray:
        from benchmarks.generate_maker_fringes_pointgroup_pg1_benchmarks import build_pg1_cases

        case = build_pg1_cases(limit=1)[0]
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
