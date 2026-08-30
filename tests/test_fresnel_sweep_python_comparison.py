import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_fresnel_sweep_reference import compare_fresnel_sweep_payload, compare_fresnel_sweep_payload_with_policy
from benchmarks.generate_maker_fringes_benchmarks import build_cases
from benchmarks.compare_mathematica_reference import numeric_array
from shaarp import run_fresnel_sweep


REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "fresnel_sweep_reference_v1.json"
)


class FresnelSweepPythonComparisonTests(unittest.TestCase):
    def test_fresnel_sweep_reference_matches_python_gui_shaped_values(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_fresnel_sweep_payload(payload, atol=1e-10, rtol=1e-10)

        self.assertGreater(len(results), 0)
        failed = [result for result in results if not result.passed]
        self.assertEqual(failed, [])

    def test_fresnel_sweep_physical_transmitted_sum_is_documented_caveat(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

        results = compare_fresnel_sweep_payload_with_policy(
            payload,
            atol=1e-10,
            rtol=1e-10,
            transmitted_wave_policy="physical_sum",
        )

        failed = [result for result in results if not result.passed]
        self.assertEqual({result.key for result in failed}, {"Tp", "Ts"})
        self.assertTrue(all(result.max_abs_error > 0.1 for result in failed))

    def test_public_fresnel_facade_can_run_gui_multilayer_selected_policy(self):
        payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        reference_case = payload["suites"][0]["items"][0]
        system = {case["id"]: case["system"] for case in build_cases()}[reference_case["id"]]

        result = run_fresnel_sweep(
            system,
            angle_grid=reference_case["inputs"]["theta_deg"],
            options={"workflow": "gui_multilayer", "transmitted_wave_policy": "shaarp_ml_selected"},
        )

        curves = reference_case["mathematica_outputs"]["listFresnel"]
        np.testing.assert_allclose(result.numeric["rp"], numeric_array(curves[0])[:, 1].real, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result.numeric["rs"], numeric_array(curves[1])[:, 1].real, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result.numeric["tp"], numeric_array(curves[2])[:, 1].real, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result.numeric["ts"], numeric_array(curves[3])[:, 1].real, atol=1e-10, rtol=1e-10)
        self.assertEqual(result.validation.mathematica_reference, "benchmarks/mathematica_reference/fresnel_sweep_reference_v1.json")
        self.assertEqual(
            result.validation.status,
            "gui_fresnel_list_selected_policy_mathematica_validated_with_transmitted_sum_caveat",
        )


if __name__ == "__main__":
    unittest.main()
