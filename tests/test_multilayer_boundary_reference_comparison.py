import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.compare_multilayer_boundary_reference import (
    NUMERIC_OUTPUT_KEY_MAP,
    SUITE_ID,
    compare_boundary_reference_files,
    compare_boundary_reference_payloads,
)


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "multilayer_boundary_system_benchmarks_v1.json"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "mathematica_reference" / "reference_template_v1.json"
)


class MultilayerBoundaryReferenceComparisonTests(unittest.TestCase):
    def test_boundary_reference_comparator_passes_on_matching_wolfram_shaped_values(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)

        results = compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

        self.assertGreater(len(results), 100)
        self.assertTrue(all(result.passed for result in results))

    def test_boundary_reference_comparator_fails_on_perturbed_matrix_value(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        first_matrix = reference["suites"][0]["items"][0]["mathematica_outputs"]["solveFresnelN_matrix"]
        first_matrix[0][0]["real"] += 1e-4

        results = compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].case_id, "boundary_system_01")
        self.assertEqual(failed[0].key, "solveFresnelN_matrix")
        self.assertGreater(failed[0].max_abs_error, 1e-5)

    def test_boundary_reference_comparator_fails_on_metadata_mismatch(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        first_metadata = reference["suites"][0]["items"][0]["mathematica_outputs"]["unknown_metadata"]
        first_metadata[0]["propagation"] = "transmitted"

        results = compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].case_id, "boundary_system_01")
        self.assertEqual(failed[0].key, "unknown_metadata")

    def test_boundary_reference_comparator_fails_on_wave_value_mismatch(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        first_wave = reference["suites"][0]["items"][0]["mathematica_outputs"]["known_waves"][0]["wave"]
        first_wave["electric"][0]["real"] += 1e-4

        results = compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)
        failed = [result for result in results if not result.passed]

        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].case_id, "boundary_system_01")
        self.assertEqual(failed[0].key, "known_waves[0].wave.electric")
        self.assertGreater(failed[0].max_abs_error, 1e-5)

    def test_boundary_reference_comparator_rejects_non_wolfram_source_in_strict_mode(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["source"] = "synthetic comparison fixture"

        with self.assertRaisesRegex(ValueError, "Mathematica or Wolfram"):
            compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_boundary_reference_comparator_rejects_unfilled_generated_template(self):
        python_payload = _python_payload()
        template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        template["source"] = "Mathematica SHAARP placeholder"

        with self.assertRaisesRegex(ValueError, "status"):
            compare_boundary_reference_payloads(template, python_payload, atol=1e-12, rtol=1e-12)

    def test_boundary_reference_comparator_rejects_pending_status_even_with_values(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["status"] = "mathematica_reference_pending"

        with self.assertRaisesRegex(ValueError, "pending"):
            compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_boundary_reference_comparator_rejects_missing_numeric_value_but_allows_metadata_none(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        outputs = reference["suites"][0]["items"][0]["mathematica_outputs"]
        self.assertIsNone(outputs["unknown_metadata"][0]["layer_index"])
        outputs["solveFresnelN_rhs"][0] = None

        with self.assertRaisesRegex(ValueError, "solveFresnelN_rhs"):
            compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_boundary_reference_comparator_rejects_missing_required_output_key(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        outputs = reference["suites"][0]["items"][0]["mathematica_outputs"]
        outputs.pop("solveFresnelN_coefficients")

        with self.assertRaisesRegex(ValueError, "solveFresnelN_coefficients"):
            compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_boundary_reference_comparator_rejects_unknown_case_id(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["id"] = "not_a_python_case"

        with self.assertRaisesRegex(ValueError, "not_a_python_case"):
            compare_boundary_reference_payloads(reference, python_payload, atol=1e-12, rtol=1e-12)

    def test_boundary_reference_file_comparison_can_feed_report_writer(self):
        python_payload = _python_payload()
        reference = _wolfram_shaped_reference_from_python(python_payload)
        reference["suites"][0]["items"][0]["mathematica_outputs"]["solveFresnelN_rhs"][0]["real"] += 1e-4

        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = Path(tmpdir) / "reference.json"
            python_path = Path(tmpdir) / "python.json"
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
            python_path.write_text(json.dumps(python_payload), encoding="utf-8")

            results = compare_boundary_reference_files(
                reference_path,
                python_path,
                atol=1e-12,
                rtol=1e-12,
            )

        failed = [result for result in results if not result.passed]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].key, "solveFresnelN_rhs")


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
    mathematica_outputs = {
        mathematica_key: copy.deepcopy(outputs[python_key])
        for mathematica_key, python_key in NUMERIC_OUTPUT_KEY_MAP.items()
    }
    for key in (
        "equation_labels",
        "equation_metadata",
        "unknown_labels",
        "unknown_metadata",
        "unknown_basis_waves",
        "known_waves",
    ):
        mathematica_outputs[key] = copy.deepcopy(outputs[key])
    return mathematica_outputs


if __name__ == "__main__":
    unittest.main()
