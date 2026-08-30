import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_shaarp_si_stage_reference import (
    compare_pnl_stage_reference,
    compare_signal_stage_reference,
    compare_inhomogeneous_stage_reference,
    compare_fresnel_stage_reference,
    compare_angle_stage_reference,
    compare_angle_stage_reference_against_shaarp_si_root_equations,
    compare_two_omega_boundary_stage_reference,
)
from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.nonlinear import compute_pnl_voigt
from shaarp.si_compat import (
    solve_shaarp_si_active_angle_roots,
    solve_shaarp_si_linear_omega,
    solve_shaarp_si_reflected_shg,
)


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
PNL_REFERENCE_PATH = REF_DIR / "shaarp_si_pnl_stage_reference_v1.json"
INHOMOGENEOUS_REFERENCE_PATH = REF_DIR / "shaarp_si_inhomogeneous_stage_reference_v1.json"
SIGNAL_REFERENCE_PATH = REF_DIR / "shaarp_si_signal_stage_reference_v1.json"
BOUNDARY_REFERENCE_PATH = REF_DIR / "shaarp_si_two_omega_boundary_stage_reference_v1.json"
FRESNEL_REFERENCE_PATH = REF_DIR / "shaarp_si_fresnel_stage_reference_v1.json"
ANGLE_REFERENCE_PATH = REF_DIR / "shaarp_si_angle_stage_reference_v1.json"


class SHAARPSIStagePythonComparisonTests(unittest.TestCase):
    def test_pnl_stage_reference_matches_python_compute_pnl_values(self):
        payload = json.loads(PNL_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_pnl_stage_reference(payload, atol=1e-12, rtol=1e-12)

        self.assertEqual(len(results), 36 * 3)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])

    def test_self_contained_linear_omega_fields_feed_pnl_stage_values(self):
        payload = json.loads(PNL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping = json.loads(Path(payload["mapping_path"]).read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}

        for case in payload["cases"]:
            with self.subTest(case_id=case["id"]):
                mapped_case = mapping_by_id[case["id"]]
                variables = mapped_case["mathematica_variables"]
                linear = solve_shaarp_si_linear_omega(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    theta_in_rad=np.deg2rad(float(mapped_case["theta_deg"])),
                    incident_polarization=str(mapped_case["incident_polarization"]),
                )
                d_shg = numeric_array(case["outputs"]["dSHG"])
                python_outputs = {
                    "P2eo": compute_pnl_voigt(d_shg, linear.transmitted_ext.electric, linear.transmitted_ext.electric),
                    "P2o": compute_pnl_voigt(d_shg, linear.transmitted_ord.electric, linear.transmitted_ord.electric),
                    "Peoo": compute_pnl_voigt(
                        d_shg,
                        linear.transmitted_ext.electric,
                        linear.transmitted_ord.electric,
                        factor=2,
                    ),
                }
                for key, actual in python_outputs.items():
                    np.testing.assert_allclose(actual, numeric_array(case["outputs"][key]), atol=1e-10, rtol=1e-10)

    def test_angle_stage_reference_is_not_validated_by_naive_scalar_snell_equations(self):
        payload = json.loads(ANGLE_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_angle_stage_reference(payload, atol=1e-12, rtol=1e-12)

        self.assertEqual(len(results), 36 * 4)
        failed = [result for result in results if not result.passed]
        self.assertGreater(len(failed), 0)
        self.assertTrue(any(result.case_id == "solver_stage_02" for result in failed))

    def test_angle_stage_reference_matches_shaarp_si_root_equations_and_branch_factors(self):
        payload = json.loads(ANGLE_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_angle_stage_reference_against_shaarp_si_root_equations(
            payload,
            residual_atol=1e-12,
            index_atol=1e-5,
        )

        self.assertEqual(len(results), 36 * 10)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])

    def test_python_solves_active_angle_roots_and_labels_from_case_inputs(self):
        payload = json.loads(ANGLE_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping = json.loads(Path(payload["mapping_path"]).read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}

        for case in payload["cases"]:
            with self.subTest(case_id=case["id"]):
                mapped_case = mapping_by_id[case["id"]]
                variables = mapped_case["mathematica_variables"]
                roots = solve_shaarp_si_active_angle_roots(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    theta_in_rad=np.deg2rad(float(mapped_case["theta_deg"])),
                )
                outputs = case["outputs"]
                np.testing.assert_allclose(roots.theta_ord, numeric_array(outputs["ThetaOrd"]).item(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(roots.theta_ext, numeric_array(outputs["ThetaExt"]).item(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(roots.theta_ord_2omega, numeric_array(outputs["ThetaOrd2w"]).item(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(roots.theta_ext_2omega, numeric_array(outputs["ThetaExt2w"]).item(), atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(roots.n_ord_label, numeric_array(outputs["nwNumOrd"]).item(), atol=1e-5, rtol=0.0)
                np.testing.assert_allclose(roots.n_ext_label, numeric_array(outputs["nwNumExt"]).item(), atol=1e-5, rtol=0.0)
                np.testing.assert_allclose(roots.n_ord_2omega_label, numeric_array(outputs["n2wNumOrd"]).item(), atol=1e-5, rtol=0.0)
                np.testing.assert_allclose(roots.n_ext_2omega_label, numeric_array(outputs["n2wNumExt"]).item(), atol=1e-5, rtol=0.0)
                self.assertEqual(roots.ext_factor, numeric_array(outputs["ExtwFactor"]).item())
                expected_ext_2omega_factor = numeric_array(outputs["Ext2wFactor"]).item()
                if case["id"] == "solver_stage_13":
                    self.assertEqual(float(mapped_case["theta_deg"]), 0.0)
                    self.assertLess(abs(numeric_array(outputs["ThetaExt2w"]).item()), 2e-15)
                    self.assertLess(abs(numeric_array(outputs["ThetaOrd2w"]).item()), 2e-15)
                    self.assertNotEqual(roots.ext_2omega_factor, expected_ext_2omega_factor)
                else:
                    self.assertEqual(roots.ext_2omega_factor, expected_ext_2omega_factor)

    def test_fresnel_stage_reference_closes_python_boundary_equations(self):
        payload = json.loads(FRESNEL_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_fresnel_stage_reference(payload, atol=1e-12, rtol=1e-12)

        self.assertEqual(len(results), 36 * 4)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])

    def test_inhomogeneous_stage_reference_matches_python_solve_inhomogeneous_field(self):
        payload = json.loads(INHOMOGENEOUS_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_inhomogeneous_stage_reference(payload, atol=1e-11, rtol=1e-11)

        self.assertEqual(len(results), 36 * 3)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])

    def test_two_omega_boundary_stage_reference_closes_python_boundary_equations(self):
        payload = json.loads(BOUNDARY_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_two_omega_boundary_stage_reference(payload, atol=1e-12, rtol=1e-12)

        self.assertEqual(len(results), 36 * 4)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])

    def test_self_contained_si_reflected_shg_matches_unambiguous_er2w_fields(self):
        boundary_payload = json.loads(BOUNDARY_REFERENCE_PATH.read_text(encoding="utf-8"))
        pnl_payload = json.loads(PNL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping = json.loads(Path(boundary_payload["mapping_path"]).read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}
        pnl_by_id = {case["id"]: case for case in pnl_payload["cases"]}

        for case in boundary_payload["cases"]:
            with self.subTest(case_id=case["id"]):
                mapped_case = mapping_by_id[case["id"]]
                variables = mapped_case["mathematica_variables"]
                result = solve_shaarp_si_reflected_shg(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    d_voigt_lab=numeric_array(pnl_by_id[case["id"]]["outputs"]["dSHG"]),
                    theta_in_rad=np.deg2rad(float(mapped_case["theta_deg"])),
                    incident_polarization=str(mapped_case["incident_polarization"]),
                    mu=1.0,
                    eps0=1.0,
                )
                expected = numeric_array(case["outputs"]["ER2w"])
                error = float(np.max(np.abs(result.reflected_total.electric - expected)))
                if case["id"] == "solver_stage_13":
                    self.assertGreater(error, 1e-4)
                else:
                    self.assertLess(error, 2e-6)

    def test_shaarp_reference_like_2omega_branch_policy_matches_all_er2w_fields(self):
        boundary_payload = json.loads(BOUNDARY_REFERENCE_PATH.read_text(encoding="utf-8"))
        pnl_payload = json.loads(PNL_REFERENCE_PATH.read_text(encoding="utf-8"))
        mapping = json.loads(Path(boundary_payload["mapping_path"]).read_text(encoding="utf-8"))
        mapping_by_id = {case["id"]: case for case in mapping["cases"]}
        pnl_by_id = {case["id"]: case for case in pnl_payload["cases"]}

        for case in boundary_payload["cases"]:
            with self.subTest(case_id=case["id"]):
                mapped_case = mapping_by_id[case["id"]]
                variables = mapped_case["mathematica_variables"]
                result = solve_shaarp_si_reflected_shg(
                    epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
                    epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
                    orientation_matrix=numeric_array(variables["a"]),
                    d_voigt_lab=numeric_array(pnl_by_id[case["id"]]["outputs"]["dSHG"]),
                    theta_in_rad=np.deg2rad(float(mapped_case["theta_deg"])),
                    incident_polarization=str(mapped_case["incident_polarization"]),
                    mu=1.0,
                    eps0=1.0,
                    normal_incidence_2omega_branch_policy="shaarp_reference_like",
                )
                np.testing.assert_allclose(
                    result.reflected_total.electric,
                    numeric_array(case["outputs"]["ER2w"]),
                    atol=2e-6,
                    rtol=0.0,
                )

    def test_signal_stage_reference_matches_python_intensity_formulas(self):
        payload = json.loads(SIGNAL_REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_signal_stage_reference(payload, atol=1e-12, rtol=1e-12)

        self.assertEqual(len(results), 36 * 7)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
