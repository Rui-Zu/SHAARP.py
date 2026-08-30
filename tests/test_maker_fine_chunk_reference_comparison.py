import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import compare_maker_reference_payloads


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_reference_v1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_python_output_v1.json"
SCRIPT_PATH = ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_reference.wl"
HIGH_OBLIQUE_REFERENCE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_high_oblique_tail_reference_v1.json"
)
HIGH_OBLIQUE_PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_high_oblique_tail_python_output_v1.json"
HIGH_OBLIQUE_SCRIPT_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_high_oblique_tail_reference.wl"
)
NORMAL_CENTER_REFERENCE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_normal_center_reference_v1.json"
)
NORMAL_CENTER_PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_normal_center_python_output_v1.json"
NORMAL_CENTER_SCRIPT_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_normal_center_reference.wl"
)
HIGH_OBLIQUE_MID_REFERENCE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_high_oblique_mid_reference_v1.json"
)
HIGH_OBLIQUE_MID_PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_high_oblique_mid_python_output_v1.json"
HIGH_OBLIQUE_MID_SCRIPT_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_high_oblique_mid_reference.wl"
)
FRINGE_NEGATIVE_EDGE_REFERENCE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_fringe_negative_edge_reference_v1.json"
)
FRINGE_NEGATIVE_EDGE_PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_fringe_negative_edge_python_output_v1.json"
FRINGE_NEGATIVE_EDGE_SCRIPT_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_fringe_negative_edge_reference.wl"
)
FRINGE_CENTER_REFERENCE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_fringe_center_reference_v1.json"
)
FRINGE_CENTER_PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_fringe_center_python_output_v1.json"
FRINGE_CENTER_SCRIPT_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_fringe_center_reference.wl"
)
POSITIVE_EDGE_REFERENCE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_positive_edge_reference_v1.json"
)
POSITIVE_EDGE_PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_positive_edge_python_output_v1.json"
POSITIVE_EDGE_SCRIPT_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_positive_edge_reference.wl"
)
HIGH_OBLIQUE_LOW_REFERENCE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_high_oblique_low_reference_v1.json"
)
HIGH_OBLIQUE_LOW_PYTHON_PATH = ROOT / "benchmarks" / "maker_fine_chunk_high_oblique_low_python_output_v1.json"
HIGH_OBLIQUE_LOW_SCRIPT_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_high_oblique_low_reference.wl"
)


class MakerFineChunkReferenceComparisonTests(unittest.TestCase):
    def test_fine_chunk_reference_matches_python_slice_value_by_value(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["status"], "mathematica_reference_exported_with_documented_empty_wave_guard")
        self.assertEqual(reference["case_count"], 1)
        self.assertEqual(python["case_count"], 1)
        self.assertEqual(python["total_angle_count"], 5)

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_chunk_export_script_uses_chunk_input_and_output_paths(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_input_v1.json", text)
        self.assertIn("maker_fine_chunk_reference_v1.json", text)
        self.assertIn("export_maker_fringes_reference.wl", text)

    def test_high_oblique_tail_chunk_matches_python_slice_through_80_degrees(self):
        reference = json.loads(HIGH_OBLIQUE_REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(HIGH_OBLIQUE_PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["case_count"], 1)
        py_case = python["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_high_oblique")
        self.assertEqual(py_case["theta_deg"], [78.0, 78.5, 79.0, 79.5, 80.0])

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_high_oblique_chunk_script_uses_unique_paths(self):
        text = HIGH_OBLIQUE_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_high_oblique_tail_input_v1.json", text)
        self.assertIn("maker_fine_chunk_high_oblique_tail_reference_v1.json", text)

    def test_normal_center_chunk_matches_python_slice_near_normal_incidence(self):
        reference = json.loads(NORMAL_CENTER_REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(NORMAL_CENTER_PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["case_count"], 1)
        py_case = python["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_negative_normal_positive")
        self.assertEqual(py_case["theta_deg"], [0.0, 0.25, 0.5, 0.75, 1.0])

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_normal_center_chunk_script_uses_unique_paths(self):
        text = NORMAL_CENTER_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_normal_center_input_v1.json", text)
        self.assertIn("maker_fine_chunk_normal_center_reference_v1.json", text)

    def test_high_oblique_mid_chunk_matches_python_slice(self):
        reference = json.loads(HIGH_OBLIQUE_MID_REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(HIGH_OBLIQUE_MID_PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["case_count"], 1)
        py_case = python["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_high_oblique")
        self.assertEqual(py_case["theta_deg"], [40.0, 40.5, 41.0, 41.5, 42.0])

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_high_oblique_mid_chunk_script_uses_unique_paths(self):
        text = HIGH_OBLIQUE_MID_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_high_oblique_mid_input_v1.json", text)
        self.assertIn("maker_fine_chunk_high_oblique_mid_reference_v1.json", text)

    def test_fringe_resolution_negative_edge_chunk_matches_python_slice(self):
        reference = json.loads(FRINGE_NEGATIVE_EDGE_REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(FRINGE_NEGATIVE_EDGE_PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["case_count"], 1)
        py_case = python["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_fringe_resolution")
        self.assertEqual(py_case["theta_deg"], [-60.0, -59.75, -59.5, -59.25, -59.0])

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_fringe_resolution_negative_edge_chunk_script_uses_unique_paths(self):
        text = FRINGE_NEGATIVE_EDGE_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_fringe_negative_edge_input_v1.json", text)
        self.assertIn("maker_fine_chunk_fringe_negative_edge_reference_v1.json", text)

    def test_fringe_resolution_center_chunk_matches_python_slice(self):
        reference = json.loads(FRINGE_CENTER_REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(FRINGE_CENTER_PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["case_count"], 1)
        py_case = python["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_fringe_resolution")
        self.assertEqual(py_case["theta_deg"], [0.0, 0.25, 0.5, 0.75, 1.0])

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_fringe_resolution_center_chunk_script_uses_unique_paths(self):
        text = FRINGE_CENTER_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_fringe_center_input_v1.json", text)
        self.assertIn("maker_fine_chunk_fringe_center_reference_v1.json", text)

    def test_positive_edge_chunk_matches_python_slice(self):
        reference = json.loads(POSITIVE_EDGE_REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(POSITIVE_EDGE_PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["case_count"], 1)
        py_case = python["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_negative_normal_positive")
        self.assertEqual(py_case["theta_deg"], [11.0, 11.25, 11.5, 11.75, 12.0])

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_positive_edge_chunk_script_uses_unique_paths(self):
        text = POSITIVE_EDGE_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_positive_edge_input_v1.json", text)
        self.assertIn("maker_fine_chunk_positive_edge_reference_v1.json", text)

    def test_high_oblique_low_chunk_matches_python_slice(self):
        reference = json.loads(HIGH_OBLIQUE_LOW_REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(HIGH_OBLIQUE_LOW_PYTHON_PATH.read_text(encoding="utf-8"))

        self.assertEqual(reference["case_count"], 1)
        py_case = python["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_high_oblique")
        self.assertEqual(py_case["theta_deg"], [0.0, 0.5, 1.0, 1.5, 2.0])

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(len(results), 7)
        self.assertTrue(all(result.passed for result in results), results)

    def test_high_oblique_low_chunk_script_uses_unique_paths(self):
        text = HIGH_OBLIQUE_LOW_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("maker_fine_chunk_high_oblique_low_input_v1.json", text)
        self.assertIn("maker_fine_chunk_high_oblique_low_reference_v1.json", text)


if __name__ == "__main__":
    unittest.main()
