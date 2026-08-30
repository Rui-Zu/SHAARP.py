import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REFERENCE_PATH = REF_DIR / "shaarp_si_angle_stage_reference_v1.json"


class SHAARPSIAngleStageReferenceTests(unittest.TestCase):
    def test_angle_stage_reference_is_exported_but_limited(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "mathematica_stage_reference_exported")
        self.assertEqual(payload["validation_claim"], "angle_stage_only_not_full_shaarp_agreement")
        self.assertEqual(payload["case_count"], 36)
        self.assertEqual(
            payload["region_ids"],
            [
                "principal_axis_and_index_setup",
                "active_wave_equation_setup",
                "active_backward_wave_equation_setup",
                "omega_angle_root_assignments_active_path",
                "two_omega_angle_root_assignments_active_path",
                "omega_angle_roots_active_numerical_path",
            ],
        )

    def test_angle_stage_roots_satisfy_mathematica_wave_equations(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        for key in (
            "max_omega_ord_root_abs_residual",
            "max_omega_ext_root_abs_residual",
            "max_two_omega_ord_root_abs_residual",
            "max_two_omega_ext_root_abs_residual",
        ):
            self.assertLess(payload[key], 1e-12)

        for case in payload["cases"]:
            self.assertEqual(case["parse_status"], "parsed")
            checks = case["checks"]
            self.assertLess(checks["omega_ord_root_abs_residual"], 1e-12)
            self.assertLess(checks["omega_ext_root_abs_residual"], 1e-12)
            self.assertLess(checks["two_omega_ord_root_abs_residual"], 1e-12)
            self.assertLess(checks["two_omega_ext_root_abs_residual"], 1e-12)

    def test_angle_stage_exports_expected_intermediates_as_complex_records(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        expected_outputs = (
            "ThetaOrd",
            "ThetaExt",
            "ThetaOrd2w",
            "ThetaExt2w",
            "nwNumOrd",
            "nwNumExt",
            "n2wNumOrd",
            "n2wNumExt",
            "ExtwFactor",
            "Ext2wFactor",
        )

        for case in payload["cases"]:
            outputs = case["outputs"]
            for key in expected_outputs:
                self.assertIn(key, outputs)
                self.assertEqual(set(outputs[key]), {"real", "imag"})
                self.assertTrue(math.isfinite(outputs[key]["real"]))
                self.assertTrue(math.isfinite(outputs[key]["imag"]))

    def test_angle_stage_captures_branch_factor_flips(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        ext_w = {case["outputs"]["ExtwFactor"]["real"] for case in payload["cases"]}
        ext_2w = {case["outputs"]["Ext2wFactor"]["real"] for case in payload["cases"]}
        self.assertEqual(ext_w, {-1.0, 1.0})
        self.assertEqual(ext_2w, {-1.0, 1.0})


if __name__ == "__main__":
    unittest.main()
