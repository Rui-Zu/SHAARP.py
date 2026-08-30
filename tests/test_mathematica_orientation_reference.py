import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_mathematica_reference import numeric_array
from shaarp.config import CrystalOrientation, CrystalStructure
from shaarp.tensors import (
    rotate_d_voigt_crystal_to_lab,
    rotate_rank2_crystal_to_lab,
    shaarp_qc_matrix,
    shaarp_qp_matrix,
)


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "orientation_reference_v1.json"
)
HEXAGONAL_DISCREPANCY_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "discrepancy_reports"
    / "orientation_hklconvert_hexagonal_point_group_failure.json"
)


class MathematicaOrientationReferenceTests(unittest.TestCase):
    def test_mathematica_orientation_reference_matches_python_values(self):
        if not REFERENCE_PATH.exists() or REFERENCE_PATH.stat().st_size == 0:
            self.skipTest("Mathematica orientation_reference_v1.json has not been exported yet.")
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Mathematica", payload["source"])
        self.assertEqual(payload["status"], "mathematica_reference_exported")
        self.assertEqual(payload["inputCellCount"], 49)
        self.assertEqual(payload["functionDownValues"]["hklConvert"], 1)
        self.assertEqual(payload["functionDownValues"]["QC2QP"], 1)
        self.assertEqual(payload["functionDownValues"]["extMater"], 1)
        self.assertEqual(payload["selectedKeys"]["dC"], [100, 67])
        self.assertEqual(payload["selectedKeys"]["dL"], [100, 76])
        self.assertGreaterEqual(payload["case_count"], 10)
        self.assertEqual(len(payload["cases"]), payload["case_count"])
        self.assertTrue(HEXAGONAL_DISCREPANCY_PATH.exists())

        for case in payload["cases"]:
            with self.subTest(case=case["id"]):
                self.assertNotIn("error", case, msg=case)
                inputs = case["inputs"]
                outputs = case["outputs"]
                structure = _structure_from_case(inputs)
                orientation = CrystalOrientation.from_shaarp_miller_surface(
                    structure,
                    numeric_array(inputs["plane_hkl"]).real,
                    numeric_array(inputs["vertical_uvw"]).real,
                )
                axes = orientation.rotation_matrix()
                epsilon_omega = numeric_array(inputs["epsilon_omega_crystal"])
                epsilon_2omega = numeric_array(inputs["epsilon_2omega_crystal"])
                d_voigt = numeric_array(inputs["d_voigt_crystal"])

                np.testing.assert_allclose(axes, numeric_array(outputs["orientation_axes"]), atol=1e-11, rtol=1e-11)
                np.testing.assert_allclose(shaarp_qc_matrix(axes), numeric_array(outputs["QC"]), atol=1e-11, rtol=1e-11)
                np.testing.assert_allclose(
                    rotate_rank2_crystal_to_lab(epsilon_omega, axes),
                    numeric_array(outputs["epsilon_omega_lab"]),
                    atol=1e-10,
                    rtol=1e-10,
                )
                np.testing.assert_allclose(
                    rotate_rank2_crystal_to_lab(epsilon_2omega, axes),
                    numeric_array(outputs["epsilon_2omega_lab"]),
                    atol=1e-10,
                    rtol=1e-10,
                )
                np.testing.assert_allclose(
                    rotate_d_voigt_crystal_to_lab(d_voigt, axes),
                    numeric_array(outputs["d_voigt_lab"]),
                    atol=1e-10,
                    rtol=1e-10,
                )
                np.testing.assert_allclose(
                    shaarp_qp_matrix(inputs["point_group"], epsilon_omega, axes),
                    numeric_array(outputs["QPOmega"]),
                    atol=1e-9,
                    rtol=1e-9,
                )
                np.testing.assert_allclose(
                    shaarp_qp_matrix(inputs["point_group"], epsilon_2omega, axes),
                    numeric_array(outputs["QP2Omega"]),
                    atol=1e-9,
                    rtol=1e-9,
                )


def _structure_from_case(inputs):
    a, b, c, alpha, beta, gamma = numeric_array(inputs["latcon"]).real
    return CrystalStructure(
        point_group=inputs["point_group"],
        a=float(a),
        b=float(b),
        c=float(c),
        alpha_deg=float(alpha),
        beta_deg=float(beta),
        gamma_deg=float(gamma),
    )


if __name__ == "__main__":
    unittest.main()
