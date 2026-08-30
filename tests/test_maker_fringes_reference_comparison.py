import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import (
    SUITE_ID,
    build_maker_reference_agreement_summary,
    compare_maker_reference_files,
    compare_maker_reference_payloads,
    write_maker_reference_agreement_summary,
)


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "maker_fringes_benchmarks_v1.json"
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "mathematica_reference" / "reference_template_v1.json"
MATHEMATICA_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_v1.json"
)


class MakerFringesReferenceComparisonTests(unittest.TestCase):
    def test_maker_reference_comparator_passes_on_matching_wolfram_shaped_values(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)

        results = compare_maker_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

        self.assertEqual(len(results), 21)
        self.assertTrue(all(result.passed for result in results))

    def test_maker_reference_comparator_fails_on_perturbed_mflist_value(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["mathematica_outputs"]["MFList"][0][1]["real"] += 1e-4

        results = compare_maker_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].case_id, "maker_fringes_negative_normal_positive")
        self.assertEqual(failed[0].key, "MFList[[All,{1,2}]]")
        self.assertGreater(failed[0].max_abs_error, 1e-5)

    def test_maker_reference_comparator_fails_on_perturbed_listmfperp_value(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["mathematica_outputs"]["listMFperp"][0][1]["real"] += 1e-4

        results = compare_maker_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].case_id, "maker_fringes_negative_normal_positive")
        self.assertEqual(failed[0].key, "listMFperp")
        self.assertGreater(failed[0].max_abs_error, 1e-5)

    def test_maker_reference_comparator_rejects_non_wolfram_source_in_strict_mode(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["source"] = "synthetic comparison fixture"

        with self.assertRaisesRegex(ValueError, "Mathematica or Wolfram"):
            compare_maker_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_maker_reference_comparator_rejects_unfilled_generated_template(self):
        python_payload = _python_payload()
        template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        template["source"] = "Mathematica SHAARP placeholder"

        with self.assertRaisesRegex(ValueError, "status"):
            compare_maker_reference_payloads(template, python_payload, atol=1e-12, rtol=1e-12)

    def test_maker_reference_comparator_rejects_missing_required_output(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["mathematica_outputs"].pop("listMFpara")

        with self.assertRaisesRegex(ValueError, "listMFpara"):
            compare_maker_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_maker_reference_comparator_rejects_unknown_case_id(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["id"] = "not_a_python_case"

        with self.assertRaisesRegex(ValueError, "not_a_python_case"):
            compare_maker_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_maker_reference_comparator_rejects_mflist_without_three_columns(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["mathematica_outputs"]["MFList"] = copy.deepcopy(
            reference["suites"][0]["items"][0]["mathematica_outputs"]["listMFpara"]
        )

        with self.assertRaisesRegex(ValueError, "MFList"):
            compare_maker_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_maker_reference_file_comparison_can_feed_report_writer(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["mathematica_outputs"]["listMFpara"][0][1]["real"] += 1e-4

        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = Path(tmpdir) / "reference.json"
            python_path = Path(tmpdir) / "python.json"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            python_path.write_text(json.dumps(python_payload), encoding="utf-8")

            results = compare_maker_reference_files(reference_path, python_path, atol=1e-12, rtol=1e-12)

        failed = [result for result in results if not result.passed]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].key, "listMFpara")

    def test_real_mathematica_maker_summary_passes_nonsingular_cases_and_keeps_phase_matching_diagnostic(self):
        reference = json.loads(MATHEMATICA_REFERENCE_PATH.read_text(encoding="utf-8"))
        python_payload = _python_payload()

        summary = build_maker_reference_agreement_summary(reference, python_payload, atol=1e-10, rtol=1e-10)

        self.assertEqual(
            summary["status"],
            "nonsingular_maker_outputs_match_with_phase_matching_diagnostic",
        )
        self.assertFalse(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], 2)
        self.assertEqual(summary["diagnostic_case_ids"], ["maker_fringes_phase_matching_like"])
        self.assertEqual(summary["nonsingular_pass_count"], 14)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        self.assertEqual(summary["diagnostic_fail_count"], 7)

        outputs = {item["key"]: item for item in summary["nonsingular_outputs"]}
        self.assertLessEqual(outputs["transmitted_two_omega_jones_sp"]["max_abs_error"], 1e-10)
        self.assertLessEqual(outputs["parallel_analyzer_amplitude"]["max_abs_error"], 1e-10)
        self.assertLessEqual(outputs["perpendicular_analyzer_amplitude"]["max_abs_error"], 1e-10)

    def test_real_mathematica_maker_summary_writer_creates_artifact(self):
        reference = json.loads(MATHEMATICA_REFERENCE_PATH.read_text(encoding="utf-8"))
        python_payload = _python_payload()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "maker_summary.json"
            summary = write_maker_reference_agreement_summary(
                output_path,
                reference,
                python_payload,
                atol=1e-10,
                rtol=1e-10,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), summary)
            self.assertEqual(summary["selected_transmitted_wave_policy"], "shaarp_ml_selected")


def _python_payload() -> dict:
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def _wolfram_shaped_reference_from_python(python_payload: dict) -> dict:
    return {
        "source": "Synthetic Wolfram comparison fixture",
        "status": "mathematica_reference_exported",
        "suites": [
            {
                "suite_id": SUITE_ID,
                "items": [
                    {
                        "id": case["id"],
                        "mathematica_outputs": _mathematica_outputs_from_python(case["outputs"]),
                    }
                    for case in python_payload["cases"]
                ],
            }
        ],
    }


def _mathematica_outputs_from_python(outputs: dict) -> dict:
    list_mf_para = copy.deepcopy(outputs["list_mf_para"])
    list_mf_perp = copy.deepcopy(outputs["list_mf_perp"])
    return {
        "MFList": [
            [copy.deepcopy(para[0]), copy.deepcopy(para[1]), copy.deepcopy(perp[1])]
            for para, perp in zip(list_mf_para, list_mf_perp)
        ],
        "listMFpara": list_mf_para,
        "listMFperp": list_mf_perp,
        "transmitted_two_omega_jones_sp": copy.deepcopy(outputs["transmitted_2omega_jones_sp"]),
        "parallel_analyzer_amplitude": copy.deepcopy(outputs["parallel_amplitude"]),
        "perpendicular_analyzer_amplitude": copy.deepcopy(outputs["perpendicular_amplitude"]),
    }


if __name__ == "__main__":
    unittest.main()
