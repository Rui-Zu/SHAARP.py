import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import (
    build_maker_reference_agreement_summary,
    compare_maker_reference_payloads,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_reference_v1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_fine_sampling_output_v1.json"


class MakerFineReferenceComparisonTests(unittest.TestCase):
    def test_fine_sampling_reference_matches_all_compared_cases(self):
        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        results = compare_maker_reference_payloads(reference, python, atol=1e-10, rtol=1e-10)
        summary = build_maker_reference_agreement_summary(reference, python, atol=1e-10, rtol=1e-10)

        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(
            summary["nonsingular_case_ids"],
            ["fine_maker_fringe_resolution", "fine_maker_high_oblique", "fine_maker_negative_normal_positive"],
        )
        self.assertEqual(len(results), 21)
        high_oblique = [result for result in results if result.case_id == "fine_maker_high_oblique"]
        negative_normal = [result for result in results if result.case_id == "fine_maker_negative_normal_positive"]
        self.assertTrue(high_oblique)
        self.assertTrue(negative_normal)
        high_oblique_by_key = {result.key: result for result in high_oblique}
        for key in ("MFList[[All,{1,2}]]", "MFList[[All,{1,3}]]", "listMFpara", "listMFperp"):
            self.assertTrue(high_oblique_by_key[key].passed, key)
            self.assertLess(high_oblique_by_key[key].max_abs_error, 1e-10)
        for key in ("parallel_analyzer_amplitude", "perpendicular_analyzer_amplitude", "transmitted_two_omega_jones_sp"):
            self.assertTrue(high_oblique_by_key[key].passed, key)
            self.assertLess(high_oblique_by_key[key].max_abs_error, 3e-14)
        self.assertTrue(all(result.passed for result in negative_normal))
        shapes_by_case = {result.case_id: result.shape for result in results if result.key == "listMFpara"}
        self.assertEqual(shapes_by_case["fine_maker_negative_normal_positive"], (97, 2))
        self.assertEqual(shapes_by_case["fine_maker_high_oblique"], (161, 2))
        self.assertEqual(shapes_by_case["fine_maker_fringe_resolution"], (481, 2))

    def test_high_oblique_fine_intensities_match_through_80_degrees(self):
        from benchmarks.compare_mathematica_reference import cases_by_id, numeric_array

        reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))
        mma = cases_by_id(reference, suite_id="maker_fringes_workflow", output_key="mathematica_outputs")[
            "fine_maker_high_oblique"
        ]["outputs"]
        py = cases_by_id(python)["fine_maker_high_oblique"]["outputs"]

        mma_para = numeric_array(mma["listMFpara"])
        py_para = numeric_array(py["list_mf_para"])
        row_delta = abs(mma_para[:, 1] - py_para[:, 1])
        bad = [idx for idx, delta in enumerate(row_delta) if delta > 1e-10]

        self.assertEqual(bad, [])
        self.assertEqual(float(mma_para[160, 0].real), 80.0)


if __name__ == "__main__":
    unittest.main()
