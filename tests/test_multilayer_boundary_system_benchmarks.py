import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.generate_multilayer_boundary_system_benchmarks import build_cases, fingerprint_record, run_case


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "multilayer_boundary_system_benchmarks_v1.json"
)


class MultilayerBoundarySystemBenchmarkTests(unittest.TestCase):
    def test_boundary_system_benchmarks_cover_layer_counts_and_known_sources(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "python_solve_fresnel_n_system_regression_not_mathematica_validation")
        self.assertEqual(payload["case_count"], 6)
        self.assertEqual(len(payload["cases"]), 6)
        self.assertEqual({case["layer_count"] for case in payload["cases"]}, {1, 2, 3})
        self.assertEqual({case["known_source"] for case in payload["cases"]}, {False, True})
        self.assertIn(0.0, {case["theta_deg"] for case in payload["cases"]})
        self.assertGreaterEqual(max(case["theta_deg"] for case in payload["cases"]), 55.0)

    def test_boundary_system_benchmarks_store_matrix_rhs_and_labels(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            outputs = case["outputs"]
            metrics = case["metrics"]
            layer_count = case["layer_count"]
            expected_size = 4 * (layer_count + 1)

            self.assertEqual(metrics["matrix_shape"], [expected_size, expected_size])
            self.assertEqual(len(outputs["matrix"]), expected_size)
            self.assertEqual(len(outputs["rhs"]), expected_size)
            self.assertEqual(len(outputs["known_residual"]), expected_size)
            self.assertEqual(len(outputs["coefficients"]), expected_size)
            self.assertEqual(len(outputs["residual"]), expected_size)
            self.assertEqual(len(outputs["equation_labels"]), expected_size)
            self.assertEqual(len(outputs["equation_metadata"]), expected_size)
            self.assertEqual(len(outputs["unknown_labels"]), expected_size)
            self.assertEqual(len(outputs["unknown_metadata"]), expected_size)
            self.assertEqual(len(outputs["unknown_basis_waves"]), expected_size)
            self.assertEqual(outputs["equation_labels"][:4], ["top:E_x", "top:E_y", "top:H_x", "top:H_y"])
            self.assertEqual(outputs["equation_labels"][-4:], ["bottom:E_x", "bottom:E_y", "bottom:H_x", "bottom:H_y"])
            self.assertEqual(
                outputs["equation_labels"],
                [item["label"] for item in outputs["equation_metadata"]],
            )
            self.assertEqual(
                outputs["unknown_labels"],
                [item["label"] for item in outputs["unknown_metadata"]],
            )
            self.assertEqual(
                outputs["unknown_metadata"],
                [item["metadata"] for item in outputs["unknown_basis_waves"]],
            )
            self.assertGreaterEqual(len(outputs["known_waves"]), 1)

        first_labels = payload["cases"][0]["outputs"]["unknown_labels"]
        self.assertEqual(
            first_labels[:6],
            [
                "top_reflected_mode_1",
                "top_reflected_mode_2",
                "layer_1_forward_mode_1",
                "layer_1_forward_mode_2",
                "layer_1_backward_mode_1",
                "layer_1_backward_mode_2",
            ],
        )
        self.assertEqual(first_labels[-2:], ["substrate_forward_mode_1", "substrate_forward_mode_2"])
        first_equations = payload["cases"][0]["outputs"]["equation_metadata"]
        self.assertEqual(
            first_equations[:4],
            [
                {
                    "row_index": 0,
                    "label": "top:E_x",
                    "interface": "top",
                    "interface_index": 0,
                    "component": "E_x",
                    "field": "E",
                    "axis": "x",
                },
                {
                    "row_index": 1,
                    "label": "top:E_y",
                    "interface": "top",
                    "interface_index": 0,
                    "component": "E_y",
                    "field": "E",
                    "axis": "y",
                },
                {
                    "row_index": 2,
                    "label": "top:H_x",
                    "interface": "top",
                    "interface_index": 0,
                    "component": "H_x",
                    "field": "H",
                    "axis": "x",
                },
                {
                    "row_index": 3,
                    "label": "top:H_y",
                    "interface": "top",
                    "interface_index": 0,
                    "component": "H_y",
                    "field": "H",
                    "axis": "y",
                },
            ],
        )
        first_metadata = payload["cases"][0]["outputs"]["unknown_metadata"]
        self.assertEqual(
            first_metadata[4],
            {
                "column_index": 4,
                "label": "layer_1_backward_mode_1",
                "region": "layer",
                "layer_index": 1,
                "basis_index": 3,
                "propagation": "backward",
                "mode_index": 1,
            },
        )
        first_unknown_wave = payload["cases"][0]["outputs"]["unknown_basis_waves"][0]["wave"]
        np.testing.assert_allclose(_numeric_array(first_unknown_wave["k"]), np.array([0.0, 0.0, -1.7]), atol=1e-12)
        np.testing.assert_allclose(_numeric_array(first_unknown_wave["electric"]), np.array([0.0, 1.0, 0.0]), atol=1e-12)
        first_known_wave = payload["cases"][0]["outputs"]["known_waves"][0]["wave"]
        np.testing.assert_allclose(_numeric_array(first_known_wave["k"]), np.array([0.0, 0.0, 1.7]), atol=1e-12)
        np.testing.assert_allclose(_numeric_array(first_known_wave["electric"]), np.array([1.0, 0.0, 0.0]), atol=1e-12)
        source_cases = [case for case in payload["cases"] if case["known_source"]]
        self.assertTrue(any(record["role"] == "known_layer_source" for case in source_cases for record in case["outputs"]["known_waves"]))

    def test_boundary_system_benchmark_residuals_and_complexity_are_good(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            metrics = case["metrics"]
            self.assertLess(metrics["residual_l2"], 1e-9)
            self.assertGreater(metrics["condition_number"], 1.0)
            self.assertGreater(metrics["max_abs_matrix_imag"], 0.0)
            if case["known_source"]:
                self.assertGreater(metrics["max_abs_rhs_imag"], 0.0)

    def test_boundary_system_benchmark_fingerprints_match_current_python(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        expected = {case["id"]: case["fingerprint"] for case in payload["cases"]}
        for case in build_cases():
            record = run_case(case)
            self.assertEqual(fingerprint_record(record), expected[record["id"]])

    def test_boundary_system_matrix_solves_saved_coefficients(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        first = payload["cases"][0]
        matrix = _numeric_array(first["outputs"]["matrix"])
        rhs = _numeric_array(first["outputs"]["rhs"])
        coefficients = _numeric_array(first["outputs"]["coefficients"])

        np.testing.assert_allclose(matrix @ coefficients, rhs, atol=1e-10, rtol=1e-10)

    def test_boundary_system_matrix_and_rhs_reconstruct_from_saved_wave_records(self):
        payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            outputs = case["outputs"]
            matrix = _numeric_array(outputs["matrix"])
            known_residual = _numeric_array(outputs["known_residual"])
            rhs = _numeric_array(outputs["rhs"])

            reconstructed_matrix = np.column_stack(
                [
                    _residual_from_one_record(
                        record,
                        layer_count=case["layer_count"],
                        thicknesses=case["inputs"]["thicknesses"],
                    )
                    for record in outputs["unknown_basis_waves"]
                ]
            )
            reconstructed_known = np.sum(
                [
                    _residual_from_one_record(
                        record,
                        layer_count=case["layer_count"],
                        thicknesses=case["inputs"]["thicknesses"],
                    )
                    for record in outputs["known_waves"]
                ],
                axis=0,
            )

            np.testing.assert_allclose(reconstructed_matrix, matrix, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(reconstructed_known, known_residual, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(rhs, -known_residual, atol=1e-12, rtol=1e-12)


def _numeric_array(value):
    if isinstance(value, list):
        return np.array([_numeric_array(item) for item in value])
    return complex(value["real"], value["imag"])


def _residual_from_one_record(record: dict, *, layer_count: int, thicknesses: list[float]) -> np.ndarray:
    residual = np.zeros(4 * (layer_count + 1), dtype=complex)
    wave = record["wave"]
    metadata = record.get("metadata", record)
    tangential = _numeric_array(wave["tangential_eh"])
    layer_index = metadata["layer_index"]
    region = metadata["region"]

    if region == "top":
        residual[0:4] += tangential
        return residual
    if region == "substrate":
        residual[-4:] -= tangential
        return residual
    if region != "layer" or layer_index is None:
        raise AssertionError(f"Unexpected wave record metadata: {metadata}")

    layer_zero_index = int(layer_index) - 1
    if layer_zero_index == 0:
        residual[0:4] -= tangential
    else:
        start = 4 * layer_zero_index
        residual[start : start + 4] -= tangential

    propagated = _propagated_tangential(wave, thicknesses[layer_zero_index])
    if layer_zero_index == layer_count - 1:
        residual[-4:] += propagated
    else:
        start = 4 * (layer_zero_index + 1)
        residual[start : start + 4] += propagated
    return residual


def _propagated_tangential(wave: dict, thickness: float) -> np.ndarray:
    kz = _numeric_array(wave["k"])[2]
    phase = np.exp(1j * kz * thickness)
    return _numeric_array(wave["tangential_eh"]) * phase


if __name__ == "__main__":
    unittest.main()
