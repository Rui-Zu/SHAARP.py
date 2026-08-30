import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.reference_paths import resolve_reference_path
from benchmarks.compare_shaarp_si_branch_metadata_reference import (
    compare_linear_field_replay_self_contained_python,
    compare_linear_field_replay,
    compare_linear_field_replay_with_python_angles_and_eigensystem,
    compare_linear_field_replay_with_python_eigensystem,
)
from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.anisotropic import shaarp_ordered_generalized_eigensystem
from shaarp.si_compat import (
    select_shaarp_si_2omega_branch_indices,
    select_shaarp_si_omega_branch_indices,
    solve_shaarp_si_active_angle_roots,
    solve_shaarp_si_homogeneous_2omega,
    solve_shaarp_si_linear_omega,
)


ROOT = Path(__file__).resolve().parents[1]
BRANCH_REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_branch_metadata_reference_v1.json"
FRESNEL_REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_fresnel_stage_reference_v1.json"
ANGLE_REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_angle_stage_reference_v1.json"


class SHAARPSIBranchMetadataReferenceTests(unittest.TestCase):
    def test_branch_metadata_reference_contains_omega_and_two_omega_branch_indices(self):
        payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "branch_metadata_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(payload["numeric_branch_case_count"], 36)
        for case in payload["cases"]:
            with self.subTest(case_id=case["id"]):
                self.assertEqual(case["parse_status"], "parsed")
                outputs = case["outputs"]
                for key in ("omega_ext_k_index", "omega_ord_k_index", "two_omega_ext_k_index", "two_omega_ord_k_index"):
                    self.assertIn(outputs[key], (1, 2, 3))
                self.assertEqual(len(outputs["EigenwNumExt_values"]), 3)
                self.assertEqual(len(outputs["EigenwNumOrd_values"]), 3)
                self.assertEqual(len(outputs["Eigen2wNumExt_values"]), 3)
                self.assertEqual(len(outputs["Eigen2wNumOrd_values"]), 3)
                omega_ext_value = _complex_value(outputs["EigenwNumExt_values"][outputs["omega_ext_k_index"] - 1])
                omega_ord_value = _complex_value(outputs["EigenwNumOrd_values"][outputs["omega_ord_k_index"] - 1])
                two_omega_ext_value = _complex_value(outputs["Eigen2wNumExt_values"][outputs["two_omega_ext_k_index"] - 1])
                two_omega_ord_value = _complex_value(outputs["Eigen2wNumOrd_values"][outputs["two_omega_ord_k_index"] - 1])
                self.assertGreater(abs(omega_ext_value), 0.0)
                self.assertGreater(abs(omega_ord_value), 0.0)
                self.assertGreater(abs(two_omega_ext_value), 0.0)
                self.assertGreater(abs(two_omega_ord_value), 0.0)

    def test_branch_metadata_replays_linear_fresnel_fields_with_si_coordinate_convention(self):
        branch_payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))
        angle_payload = json.loads(ANGLE_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_linear_field_replay(
            branch_payload,
            fresnel_payload,
            angle_payload,
            atol=2e-6,
            rtol=0.0,
        )

        self.assertEqual(len(results), 36 * 3)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])

    def test_python_eigensystem_replays_linear_fresnel_fields_with_si_coordinate_convention(self):
        branch_payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))
        angle_payload = json.loads(ANGLE_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_linear_field_replay_with_python_eigensystem(
            branch_payload,
            fresnel_payload,
            angle_payload,
            atol=2e-6,
            rtol=0.0,
        )

        self.assertEqual(len(results), 36 * 3)
        field_failures = [
            result
            for result in results
            if not result.passed and not result.key.startswith("EwrootsNum")
        ]
        self.assertEqual(field_failures, [])
        coefficient_failures = [
            result.case_id
            for result in results
            if not result.passed and result.key.startswith("EwrootsNum")
        ]
        self.assertEqual(
            coefficient_failures,
            [
                "solver_stage_01",
                "solver_stage_19",
                "solver_stage_22",
                "solver_stage_30",
                "solver_stage_32",
                "solver_stage_35",
            ],
        )

    def test_python_angles_and_eigensystem_replay_linear_fresnel_fields_with_si_coordinate_convention(self):
        branch_payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_linear_field_replay_with_python_angles_and_eigensystem(
            branch_payload,
            fresnel_payload,
            atol=2e-6,
            rtol=0.0,
        )

        self.assertEqual(len(results), 36 * 3)
        field_failures = [
            result
            for result in results
            if not result.passed and not result.key.startswith("EwrootsNum")
        ]
        self.assertEqual(field_failures, [])

    def test_self_contained_python_replays_linear_fresnel_fields_with_si_coordinate_convention(self):
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_linear_field_replay_self_contained_python(
            fresnel_payload,
            atol=2e-6,
            rtol=0.0,
        )

        self.assertEqual(len(results), 36 * 3)
        field_failures = [
            result
            for result in results
            if not result.passed and not result.key.startswith("EwrootsNum")
        ]
        self.assertEqual(field_failures, [])

    def test_shaarp_si_linear_omega_function_matches_fresnel_stage_fields(self):
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping_file = resolve_reference_path(fresnel_payload["mapping_path"])
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}

        for case in fresnel_payload["cases"]:
            with self.subTest(case_id=case["id"]):
                mapped_case = mapping_by_id[case["id"]]
                variables = mapped_case["mathematica_variables"]
                result = solve_shaarp_si_linear_omega(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    theta_in_rad=np.deg2rad(float(mapped_case["theta_deg"])),
                    incident_polarization=str(mapped_case["incident_polarization"]),
                )
                np.testing.assert_allclose(
                    result.transmitted_ext.electric,
                    numeric_array(case["outputs"]["ETweo"]),
                    atol=2e-6,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    result.transmitted_ord.electric,
                    numeric_array(case["outputs"]["ETwo"]),
                    atol=2e-6,
                    rtol=0.0,
                )

    def test_python_shaarp_ordered_generalized_eigensystem_matches_branch_metadata_values(self):
        branch_payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))
        angle_payload = json.loads(ANGLE_REFERENCE_PATH.read_text(encoding="utf-8"))
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping_file = resolve_reference_path(fresnel_payload["mapping_path"])
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        angle_by_id = {case["id"]: case for case in angle_payload["cases"]}
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}

        for case in branch_payload["cases"]:
            with self.subTest(case_id=case["id"]):
                variables = mapping_by_id[case["id"]]["mathematica_variables"]
                a_matrix = numeric_array(variables["a"])
                epsilon_omega_lab = (
                    a_matrix
                    @ numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"])
                    @ a_matrix.T
                )
                angle_outputs = angle_by_id[case["id"]]["outputs"]
                branch_outputs = case["outputs"]

                for label, theta_key, values_key, vectors_key in (
                    ("ext", "ThetaExt", "EigenwNumExt_values", "EigenwNumExt_vectors"),
                    ("ord", "ThetaOrd", "EigenwNumOrd_values", "EigenwNumOrd_vectors"),
                ):
                    with self.subTest(label=label):
                        theta = numeric_array(angle_outputs[theta_key]).item()
                        direction = [np.sin(theta), 0.0, -np.cos(theta)]
                        actual_values, actual_vectors = shaarp_ordered_generalized_eigensystem(
                            direction,
                            epsilon_omega_lab,
                        )
                        expected_values = numeric_array(branch_outputs[values_key])
                        expected_vectors = numeric_array(branch_outputs[vectors_key])
                        np.testing.assert_allclose(actual_values, expected_values, atol=1e-12, rtol=1e-12)
                        for actual, expected in zip(actual_vectors, expected_vectors):
                            self.assertLess(_phase_aligned_error(actual, expected), 1e-10)

    def test_python_selects_active_omega_branch_indices_from_post_factor_labels(self):
        branch_payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping_file = resolve_reference_path(fresnel_payload["mapping_path"])
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}

        for case in branch_payload["cases"]:
            with self.subTest(case_id=case["id"]):
                variables = mapping_by_id[case["id"]]["mathematica_variables"]
                roots = solve_shaarp_si_active_angle_roots(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    theta_in_rad=np.deg2rad(float(mapping_by_id[case["id"]]["theta_deg"])),
                )
                indices = select_shaarp_si_omega_branch_indices(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    roots=roots,
                )
                outputs = case["outputs"]
                self.assertEqual(indices.ext_index, outputs["omega_ext_k_index"])
                self.assertEqual(indices.ord_index, outputs["omega_ord_k_index"])

    def test_python_selects_active_2omega_branch_indices_from_post_factor_labels(self):
        branch_payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping_file = resolve_reference_path(fresnel_payload["mapping_path"])
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}

        for case in branch_payload["cases"]:
            with self.subTest(case_id=case["id"]):
                variables = mapping_by_id[case["id"]]["mathematica_variables"]
                roots = solve_shaarp_si_active_angle_roots(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    theta_in_rad=np.deg2rad(float(mapping_by_id[case["id"]]["theta_deg"])),
                )
                indices = select_shaarp_si_2omega_branch_indices(
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    roots=roots,
                )
                outputs = case["outputs"]
                if case["id"] == "solver_stage_13":
                    self.assertEqual(float(mapping_by_id[case["id"]]["theta_deg"]), 0.0)
                    self.assertNotEqual(indices.ext_index, outputs["two_omega_ext_k_index"])
                    self.assertNotEqual(indices.ord_index, outputs["two_omega_ord_k_index"])
                else:
                    self.assertEqual(indices.ext_index, outputs["two_omega_ext_k_index"])
                    self.assertEqual(indices.ord_index, outputs["two_omega_ord_k_index"])

    def test_shaarp_si_homogeneous_2omega_matches_unambiguous_branch_fields(self):
        branch_payload = json.loads(BRANCH_REFERENCE_PATH.read_text(encoding="utf-8"))
        fresnel_payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping_file = resolve_reference_path(fresnel_payload["mapping_path"])
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}

        for case in branch_payload["cases"]:
            with self.subTest(case_id=case["id"]):
                variables = mapping_by_id[case["id"]]["mathematica_variables"]
                roots = solve_shaarp_si_active_angle_roots(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    theta_in_rad=np.deg2rad(float(mapping_by_id[case["id"]]["theta_deg"])),
                )
                waves = solve_shaarp_si_homogeneous_2omega(
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    roots=roots,
                )
                outputs = case["outputs"]
                if case["id"] == "solver_stage_13":
                    self.assertGreater(
                        _phase_aligned_error(waves.ext.electric, numeric_array(outputs["ET2wNumExt"])),
                        1e-3,
                    )
                    continue
                self.assertLess(
                    _phase_aligned_error(waves.ext.electric, numeric_array(outputs["ET2wNumExt"])),
                    1e-10,
                )
                self.assertLess(
                    _phase_aligned_error(waves.ord.electric, numeric_array(outputs["ET2wNumOrd"])),
                    1e-10,
                )


def _complex_value(value):
    return complex(value["real"], value["imag"])


def _phase_aligned_error(actual, expected):
    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    scale = np.vdot(expected, actual) / np.vdot(expected, expected)
    return float(np.linalg.norm(actual - scale * expected))


if __name__ == "__main__":
    unittest.main()
