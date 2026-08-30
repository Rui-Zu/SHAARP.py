"""Gated agreement test for the JK / HH Maker SHG assumption modes (mrassumption 1/2).

STATUS: the Python solver does NOT yet implement the JK (Jerphagnon-
Kurtz, sequential single-pass) or HH (Herman-Hayden) SHG assumption modes. SHAARP
selects them via `MF[..., assumption]` / `f4NL[..., iterative=True]` with flagJK /
flagHH, which changes BOTH the linear fundamental solve (iterative mode) and
the SHG boundary unknown set. Python currently implements only the full
(mrassumption 0) path, which is validated and which Python reproduces with the
default `inhomogeneous_source_policy="forward_only"`.

This test encodes the LIVE-Wolfram single-film validation targets (exported under
mrassumption 0/1/2 for ext1 case0; see jkhh_validation_targets_v1.json) and is
SKIPPED until the solver advertises a JK/HH entry point. The contract the future
implementation should satisfy to make this test live:

  * `solve_multilayer_maker_fringes_sweep(...)` accepts a keyword `mrassumption`
    (0=full, 1=JK, 2=HH); OR the module defines `SUPPORTS_JKHH = True` plus a
    sweep that honors `mrassumption`.
  * With mrassumption=1/2, the resulting |listMFpara| for ext1 case0 must match the
    saved JK/HH targets below to atol 5e-4.

When that lands, this test flips from skip to a real agreement gate (no edit
needed) -- the same pattern used for the sipg/pgml live-Wolfram milestones.
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "jkhh_validation_targets_v1.json"
ATOL = 5e-4


def _supports_jkhh() -> bool:
    """True once the solver exposes an mrassumption (JK/HH) entry point."""
    try:
        params = inspect.signature(solve_multilayer_maker_fringes_sweep).parameters
    except (TypeError, ValueError):
        return False
    if "mrassumption" in params:
        return True
    import shaarp.multilayer_shg_boundary as m

    return bool(getattr(m, "SUPPORTS_JKHH", False))


def _para_amp(pairs: list) -> np.ndarray:
    """|MFpara| from SHAARP-style [theta, value] complex pairs."""
    def num(x):
        return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)

    return np.array([abs(num(row[-1])) for row in pairs], dtype=float)


class JKHHTargetsArePresentTests(unittest.TestCase):
    """Always-on: the live-Wolfram JK/HH targets are persisted and well-formed."""

    def test_targets_file_present_and_distinct(self):
        self.assertTrue(TARGETS_PATH.exists(), f"missing {TARGETS_PATH}")
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        full = _para_amp(t["full_listMFpara"])
        jk = _para_amp(t["jk_listMFpara"])
        hh = _para_amp(t["hh_listMFpara"])
        self.assertEqual(len(full), 4)
        # Modes must be genuinely distinct (this is why JK/HH are worth validating).
        self.assertGreater(np.max(np.abs(jk - full)), 1e-3, "JK should differ from full")
        self.assertGreater(np.max(np.abs(hh - full)), 1e-3, "HH should differ from full")
        # Sanity: the documented magnitudes.
        np.testing.assert_allclose(full, [0.0521, 0.0544, 0.0361, 0.0089], atol=1e-3)
        np.testing.assert_allclose(jk, [0.0542, 0.0569, 0.0384, 0.0093], atol=1e-3)
        np.testing.assert_allclose(hh, [0.0478, 0.0488, 0.0326, 0.0083], atol=1e-3)


@unittest.skipUnless(
    _supports_jkhh(),
    "Python solver does not yet implement JK/HH (mrassumption 1/2) modes; "
    "this gate goes live automatically once solve_multilayer_maker_fringes_sweep honors `mrassumption`.",
)
class JKHHMakerAgreementTests(unittest.TestCase):
    def _run(self, mrassumption: int) -> np.ndarray:
        from benchmarks.generate_maker_fringes_ext_benchmarks import build_ext_maker_cases

        case = build_ext_maker_cases(limit=1)[0]
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
        try:
            result = self._run(1)
        except NotImplementedError:
            self.skipTest(
                "JK mode (mrassumption=1) not yet implemented; needs the sequential "
                "single-pass SHG assembly. HH (mrassumption=2) is validated."
            )
        np.testing.assert_allclose(result, _para_amp(t["jk_listMFpara"]), atol=ATOL)

    def test_hh_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(2), _para_amp(t["hh_listMFpara"]), atol=ATOL)


if __name__ == "__main__":
    unittest.main()
