import unittest
from pathlib import Path

from benchmarks.generate_maker_fine_chunk_coverage import build_coverage


ROOT = Path(__file__).resolve().parents[1]


class MakerFineChunkCoverageTests(unittest.TestCase):
    def test_coverage_validates_complete_fine_grid_partition(self):
        coverage = build_coverage(ROOT)

        self.assertEqual(coverage["status"], "complete_fine_maker_chunk_coverage")
        self.assertEqual(coverage["full_fine_grid_validated"], True)
        self.assertEqual(coverage["total_fine_angle_count"], 739)
        self.assertEqual(coverage["verified_angle_count"], 739)
        self.assertEqual(coverage["remaining_angle_count"], 0)
        self.assertEqual(coverage["verified_chunk_count"], 32)
        self.assertEqual(coverage["failed_comparison_count"], 0)

        # Every registered chunk must pass its live-Wolfram value comparison.
        for chunk in coverage["chunks"]:
            self.assertTrue(chunk["passes_value_comparison"], chunk["chunk_id"])

        # The chunks must form a complete, non-overlapping partition of each case:
        # contiguous 0..N-1 index coverage with no double-counted angle.
        expected_lengths = {
            "fine_maker_negative_normal_positive": 97,
            "fine_maker_high_oblique": 161,
            "fine_maker_fringe_resolution": 481,
        }
        per_case_indices: dict[str, list[int]] = {case: [] for case in expected_lengths}
        for chunk in coverage["chunks"]:
            per_case_indices[chunk["case_id"]].extend(range(chunk["start"], chunk["start"] + chunk["count"]))
        for case, length in expected_lengths.items():
            indices = per_case_indices[case]
            self.assertEqual(len(indices), len(set(indices)), f"overlap in {case}")
            self.assertEqual(sorted(indices), list(range(length)), f"non-contiguous coverage in {case}")
        self.assertEqual(sum(expected_lengths.values()), 739)

        chunks = {chunk["chunk_id"]: chunk for chunk in coverage["chunks"]}
        self.assertEqual(chunks["negative_normal_positive_start_4_count_5"]["theta_deg"], [-11.0, -10.75, -10.5, -10.25, -10.0])
        self.assertEqual(chunks["negative_normal_positive_center_start_48_count_5"]["theta_deg"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(chunks["negative_normal_positive_center_start_48_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["negative_normal_positive_positive_edge_start_92_count_5"]["theta_deg"], [11.0, 11.25, 11.5, 11.75, 12.0])
        self.assertTrue(chunks["negative_normal_positive_positive_edge_start_92_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["high_oblique_low_start_0_count_5"]["theta_deg"], [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertTrue(chunks["high_oblique_low_start_0_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["high_oblique_mid_start_80_count_5"]["theta_deg"], [40.0, 40.5, 41.0, 41.5, 42.0])
        self.assertTrue(chunks["high_oblique_mid_start_80_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["high_oblique_tail_start_156_count_5"]["theta_deg"], [78.0, 78.5, 79.0, 79.5, 80.0])
        self.assertTrue(chunks["high_oblique_tail_start_156_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["fringe_resolution_negative_edge_start_0_count_5"]["theta_deg"], [-60.0, -59.75, -59.5, -59.25, -59.0])
        self.assertTrue(chunks["fringe_resolution_negative_edge_start_0_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["fringe_resolution_center_start_240_count_5"]["theta_deg"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(chunks["fringe_resolution_center_start_240_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["negative_normal_positive_mid_start_24_count_5"]["theta_deg"], [-6.0, -5.75, -5.5, -5.25, -5.0])
        self.assertTrue(chunks["negative_normal_positive_mid_start_24_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["negative_normal_positive_upper_start_70_count_5"]["theta_deg"], [5.5, 5.75, 6.0, 6.25, 6.5])
        self.assertTrue(chunks["negative_normal_positive_upper_start_70_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["high_oblique_lowmid_start_40_count_5"]["theta_deg"], [20.0, 20.5, 21.0, 21.5, 22.0])
        self.assertTrue(chunks["high_oblique_lowmid_start_40_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["high_oblique_high_start_120_count_5"]["theta_deg"], [60.0, 60.5, 61.0, 61.5, 62.0])
        self.assertTrue(chunks["high_oblique_high_start_120_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["fringe_resolution_negative_mid_start_120_count_5"]["theta_deg"], [-30.0, -29.75, -29.5, -29.25, -29.0])
        self.assertTrue(chunks["fringe_resolution_negative_mid_start_120_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["fringe_resolution_positive_mid_start_360_count_5"]["theta_deg"], [30.0, 30.25, 30.5, 30.75, 31.0])
        self.assertTrue(chunks["fringe_resolution_positive_mid_start_360_count_5"]["passes_value_comparison"])
        self.assertEqual(chunks["fringe_resolution_positive_tail_start_476_count_5"]["theta_deg"], [59.0, 59.25, 59.5, 59.75, 60.0])
        self.assertTrue(chunks["fringe_resolution_positive_tail_start_476_count_5"]["passes_value_comparison"])


if __name__ == "__main__":
    unittest.main()
