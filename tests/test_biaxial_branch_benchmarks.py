import unittest
from pathlib import Path

import numpy as np

from benchmarks.generate_biaxial_branch_benchmarks import (
    build_cases,
    fingerprint_record,
    noncubic_extreme_structure,
    run_sequence,
)


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "biaxial_branch_benchmarks_v1.json"
)


class BiaxialBranchBenchmarkTests(unittest.TestCase):
    def test_biaxial_branch_benchmark_file_covers_angles_and_orientations(self):
        self.assertTrue(BENCHMARK_PATH.exists())
        import json

        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["sequence_count"], 6)
        self.assertEqual(payload["angle_count_per_sequence"], 6)
        self.assertEqual({sequence["orientation_index"] for sequence in payload["sequences"]}, {0, 1, 2, 3, 4, 5})
        for sequence in payload["sequences"]:
            self.assertEqual(len(sequence["steps"]), 6)
            self.assertGreater(sequence["metrics"]["max_abs_epsilon_offdiag"], 0.0)
            self.assertGreater(sequence["metrics"]["max_abs_epsilon_imag"], 0.0)
        miller_orientation = payload["sequences"][-1]["inputs"]["orientation"]
        self.assertTrue(all(abs(miller_orientation[row][2]["real"]) > 0 for row in range(3)))
        reciprocal_normal = noncubic_extreme_structure().miller_plane_normal((1, 2, 3))
        saved_normal = np.array([miller_orientation[row][2]["real"] for row in range(3)])
        np.testing.assert_allclose(saved_normal, reciprocal_normal, atol=1e-12)

    def test_biaxial_branch_benchmark_residuals_and_alignments_are_good(self):
        for case in build_cases():
            record = run_sequence(case)
            self.assertLess(record["metrics"]["max_residual"], 1e-8)
            self.assertGreater(record["metrics"]["min_branch_alignment"], 0.8)

    def test_biaxial_branch_benchmark_fingerprints_match_current_python(self):
        import json

        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        saved = {record["id"]: record for record in payload["sequences"]}
        for case in build_cases():
            record = run_sequence(case)
            self.assertIn(record["id"], saved)
            self.assertEqual(record["fingerprint"], saved[record["id"]]["fingerprint"])
            self.assertEqual(fingerprint_record(record), saved[record["id"]]["fingerprint"])


if __name__ == "__main__":
    unittest.main()
