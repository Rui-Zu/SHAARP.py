import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.generate_solver_stage_benchmarks import (
    build_cases,
    fingerprint_record,
    noncubic_extreme_structure,
    run_case,
)


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "solver_stage_benchmarks_v1.json"
)


class SolverStageBenchmarkTests(unittest.TestCase):
    def test_solver_stage_benchmark_file_covers_angles_orientations_and_polarizations(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(len(payload["cases"]), 36)
        self.assertGreaterEqual(len({case["theta_deg"] for case in payload["cases"]}), 6)
        self.assertGreaterEqual(len({case["orientation_index"] for case in payload["cases"]}), 4)
        self.assertEqual({case["incident_polarization"] for case in payload["cases"]}, {"s", "p"})
        for case in payload["cases"]:
            self.assertGreater(case["metrics"]["max_abs_d_imag"], 0.0)
            self.assertGreater(case["metrics"]["max_abs_epsilon_omega_imag"], 0.0)
            self.assertGreater(case["metrics"]["max_abs_epsilon_2omega_imag"], 0.0)
            self.assertAlmostEqual(case["metrics"]["orientation_det"], 1.0, places=12)
        miller_orientation = payload["cases"][-1]["inputs"]["orientation"]
        self.assertTrue(all(abs(miller_orientation[row][2]["real"]) > 0 for row in range(3)))
        reciprocal_normal = noncubic_extreme_structure().miller_plane_normal((1, 2, 3))
        saved_normal = np.array([miller_orientation[row][2]["real"] for row in range(3)])
        np.testing.assert_allclose(saved_normal, reciprocal_normal, atol=1e-12)

    def test_solver_stage_benchmark_residuals_are_small(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertLess(metrics["boundary_residual_l2"], 1e-9)
            self.assertLess(metrics["inhomogeneous_ee_residual_l2"], 1e-10)
            self.assertLess(metrics["inhomogeneous_oo_residual_l2"], 1e-10)
            self.assertLess(metrics["inhomogeneous_eo_residual_l2"], 1e-10)
            self.assertLess(metrics["tangential_kx_spread"], 1e-10)

    def test_solver_stage_benchmark_fingerprints_match_current_python(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        expected = {case["id"]: case["fingerprint"] for case in payload["cases"]}
        for case in build_cases():
            record = run_case(case)
            self.assertEqual(fingerprint_record(record), expected[record["id"]])


if __name__ == "__main__":
    unittest.main()
