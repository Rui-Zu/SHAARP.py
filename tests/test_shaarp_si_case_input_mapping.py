import json
import unittest
from pathlib import Path

from benchmarks.generate_shaarp_si_case_input_mapping import build_payload


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
MAPPING_PATH = REF_DIR / "shaarp_si_case_input_mapping_v1.json"
PREFLIGHT_PATH = REF_DIR / "shaarp_si_case_input_mapping_preflight_v1.json"


class SHAARPSICaseInputMappingTests(unittest.TestCase):
    def test_saved_mapping_matches_generator(self):
        expected = build_payload()
        actual = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)

    def test_mapping_is_not_validation_evidence(self):
        payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "input_mapping_not_validation_evidence")
        self.assertFalse(payload["validation_claim"])
        self.assertIn("does not run Mathematica", " ".join(payload["notes"]))

    def test_mapping_covers_all_solver_stage_cases_and_controls(self):
        payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["case_count"], 36)
        self.assertEqual({case["incident_polarization"] for case in payload["cases"]}, {"s", "p"})
        self.assertEqual({case["orientation_index"] for case in payload["cases"]}, {0, 1, 2, 3, 4, 5})
        polarizer_by_pol = {case["incident_polarization"]: case["mathematica_variables"]["PolarizerAngle"] for case in payload["cases"]}
        self.assertEqual(polarizer_by_pol["s"], 90.0)
        self.assertEqual(polarizer_by_pol["p"], 0.0)

    def test_tensor_mapping_reconstructs_python_lab_inputs(self):
        payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

        max_eps_w = max(case["reconstruction_checks"]["max_abs_epsilon_omega_lab_error"] for case in payload["cases"])
        max_eps_2w = max(case["reconstruction_checks"]["max_abs_epsilon_2omega_lab_error"] for case in payload["cases"])
        max_d = max(case["reconstruction_checks"]["max_abs_d_voigt_lab_error"] for case in payload["cases"])
        max_orth = max(case["reconstruction_checks"]["orientation_orthogonality_error"] for case in payload["cases"])
        self.assertLess(max_eps_w, 1e-12)
        self.assertLess(max_eps_2w, 1e-12)
        self.assertLess(max_d, 1e-12)
        self.assertLess(max_orth, 1e-12)

    def test_mapping_exports_shaarp_si_variable_names(self):
        first = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["cases"][0]
        variables = first["mathematica_variables"]
        expected = first["expected_python_lab_inputs"]

        for name in (
            "ThetaIn",
            "PolarizerAngle",
            "AnalyzerAngle",
            "a",
            "\\[CurlyEpsilon]\\[Omega]Cry",
            "\\[CurlyEpsilon]2\\[Omega]Cry",
            "dold",
        ):
            self.assertIn(name, variables)
        for name in ("epsilon_omega_lab", "epsilon_2omega_lab", "d_voigt_lab"):
            self.assertIn(name, expected)

    def test_mathematica_preflight_validates_mapping_convention_not_stage_values(self):
        payload = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_preflight_exported_not_shaarp_stage_values")
        self.assertFalse(payload["validation_claim"])
        self.assertEqual(payload["case_count"], 36)
        self.assertLess(payload["max_abs_epsilon_omega_lab_error"], 1e-12)
        self.assertLess(payload["max_abs_epsilon_2omega_lab_error"], 1e-12)
        self.assertLess(payload["max_abs_d_voigt_lab_error"], 1e-12)
        self.assertEqual(len(payload["cases"]), 36)


if __name__ == "__main__":
    unittest.main()
