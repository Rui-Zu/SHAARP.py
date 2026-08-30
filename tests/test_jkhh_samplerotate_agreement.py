"""Gated agreement test for JK/HH under the SampleRotate azimuth-scan entry point.

The other JK/HH tests validate the incidence-swept Maker (MF) path. This one
validates the SECOND entry point -- physical sample-azimuth rotation
(solve_multilayer_shg_sample_azimuth_sweep, SHAARP.ml SampleRotate) -- under the
JK/HH assumption modes. It composes the validated single-pass solve with the
validated azimuth rotation (the ml4 rotate-active-only path) and checks the
composition against a fresh live SHAARP.ml SampleRotate reference exported under
mrassumption 0/1/2 for the ml4 4-layer case (azimuths 0/22.5/45 deg).

Validated to ~4.7e-15 (jkhh_samplerotate_validation_targets_v1.json).
"""

import inspect
import json
import unittest
from pathlib import Path

import numpy as np

from shaarp.multilayer_shg_boundary import solve_multilayer_shg_sample_azimuth_sweep

ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "benchmarks" / "mathematica_reference" / "jkhh_samplerotate_validation_targets_v1.json"
ATOL = 5e-4


def _supports_jkhh_sr() -> bool:
    try:
        params = inspect.signature(solve_multilayer_shg_sample_azimuth_sweep).parameters
    except (TypeError, ValueError):
        return False
    return "mrassumption" in params


def _signal_cols(rows: list) -> np.ndarray:
    """The 4 signal columns (drop the azimuth label in col 0) of a sampleRotateList."""
    def num(x):
        return x["real"] if isinstance(x, dict) else x

    return np.array([[num(v) for v in row[1:]] for row in rows], dtype=float)


class JKHHSampleRotateTargetsTests(unittest.TestCase):
    def test_targets_present_and_distinct(self):
        self.assertTrue(TARGETS_PATH.exists(), f"missing {TARGETS_PATH}")
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        full = _signal_cols(t["full"])
        self.assertGreater(np.max(np.abs(_signal_cols(t["jk"]) - full)), 1e-5)
        self.assertGreater(np.max(np.abs(_signal_cols(t["hh"]) - full)), 1e-5)


@unittest.skipUnless(
    _supports_jkhh_sr(),
    "azimuth sweep does not expose `mrassumption`; SampleRotate JK/HH gate goes live once it does.",
)
class JKHHSampleRotateAgreementTests(unittest.TestCase):
    def _run(self, mrassumption: int) -> np.ndarray:
        from benchmarks.generate_sample_rotation_multilayer_ml4_benchmarks import (
            build_ml4_cases,
            _sample_rotate_list,
        )

        case = build_ml4_cases(limit=1)[0]
        sweep = solve_multilayer_shg_sample_azimuth_sweep(
            case["system"],
            sample_azimuth_deg=np.asarray(case["sample_rotation_deg"], dtype=float),
            mu=1.0,
            eps0=1.0,
            inhomogeneous_solution_policy="solve",
            mrassumption=mrassumption,
        )
        return _signal_cols(np.array(_sample_rotate_list(sweep)).tolist())

    def test_full_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(0), _signal_cols(t["full"]), atol=ATOL)

    def test_jk_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(1), _signal_cols(t["jk"]), atol=ATOL)

    def test_hh_matches_target(self):
        t = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
        np.testing.assert_allclose(self._run(2), _signal_cols(t["hh"]), atol=ATOL)


if __name__ == "__main__":
    unittest.main()
