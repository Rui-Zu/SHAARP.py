import copy
import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fine_to_coarse import compare_fine_to_coarse_payloads


COARSE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "maker_fringes_benchmarks_v1.json"


class MakerFineToCoarseComparisonTests(unittest.TestCase):
    def test_fine_to_coarse_comparison_passes_on_matching_overlap_values(self):
        coarse = _coarse_payload()
        fine = _fine_payload_from_coarse(coarse, "maker_fringes_negative_normal_positive")

        results = compare_fine_to_coarse_payloads(fine, coarse, atol=1e-12, rtol=1e-12)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.passed for result in results))

    def test_fine_to_coarse_comparison_fails_on_perturbed_overlap_value(self):
        coarse = _coarse_payload()
        fine = _fine_payload_from_coarse(coarse, "maker_fringes_negative_normal_positive")
        fine["cases"][0]["outputs"]["list_mf_para"][1][1]["real"] += 1e-4

        results = compare_fine_to_coarse_payloads(fine, coarse, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].case_id, "fine_maker_fringes_negative_normal_positive")
        self.assertEqual(failed[0].key, "list_mf_para")
        self.assertGreater(failed[0].max_abs_error, 1e-5)

    def test_fine_to_coarse_comparison_rejects_missing_source_case_metadata(self):
        coarse = _coarse_payload()
        fine = _fine_payload_from_coarse(coarse, "maker_fringes_negative_normal_positive")
        fine["cases"][0].pop("fine_sampling")

        with self.assertRaisesRegex(ValueError, "source_case_id"):
            compare_fine_to_coarse_payloads(fine, coarse, atol=1e-12, rtol=1e-12)

    def test_fine_to_coarse_comparison_rejects_mathematica_status(self):
        coarse = _coarse_payload()
        fine = _fine_payload_from_coarse(coarse, "maker_fringes_negative_normal_positive")
        fine["status"] = "mathematica_reference_exported"

        with self.assertRaisesRegex(ValueError, "Python-only"):
            compare_fine_to_coarse_payloads(fine, coarse, atol=1e-12, rtol=1e-12)


def _coarse_payload() -> dict:
    return json.loads(COARSE_PATH.read_text(encoding="utf-8"))


def _fine_payload_from_coarse(coarse: dict, source_case_id: str) -> dict:
    source = next(case for case in coarse["cases"] if case["id"] == source_case_id)
    fine_case = copy.deepcopy(source)
    fine_case["id"] = f"fine_{source_case_id}"
    fine_case["fine_sampling"] = {
        "source_case_id": source_case_id,
        "theta_step_deg": 0.25,
        "expected_residual_behavior": "small_shg_residual_expected",
    }
    return {
        "status": "python_slow_path_maker_fringes_fine_sampling_not_mathematica_validation",
        "cases": [fine_case],
    }


if __name__ == "__main__":
    unittest.main()
