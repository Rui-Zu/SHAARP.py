"""Live-Wolfram validation of the multilayer polarimetry solve (gui_shg_simulation_polarimetry).

The reflected / transmitted 2-omega analyzer-projected intensities computed by
``solve_multilayer_shg_from_system_polarimetry`` match the original Mathematica
SHAARP.ml ``SampleRotate`` to ~1e-14 across 20 polarimetry settings (varied
theta / phi / psi / ellipticity with non-trivial baked orientations).

CONVENTION NOTE (a genuine finding from building this validation): the convenience
wrapper ``solve_multilayer_shg_polarimetry_sweep`` defaults to
``inhomogeneous_source_policy="all"``, which is NOT the SHAARP.ml convention
(SampleRotate uses forward-only sources: flagBackward = flagStandingWave = False).
This test therefore validates the underlying solve under ``"forward_only"`` — the
policy that matches the original package. (The azimuth is baked into each layer's
orientation by ``build_cases`` via ``with_sample_azimuth_deg``, so the Mathematica
reference is exported at a zero rotation step.)
"""
import json
import unittest
from pathlib import Path

import numpy as np

REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "mathematica_reference"
    / "polarimetry_reference_v1.json"
)


def _num(x):
    return x["real"] if isinstance(x, dict) else x


@unittest.skipUnless(REFERENCE_PATH.exists(), "polarimetry live-Wolfram reference not present")
class PolarimetryReferenceComparisonTests(unittest.TestCase):
    def test_reference_is_live_wolfram_sourced(self):
        ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertIn("Mathematica", ref["source"])
        self.assertIn("SampleRotate", ref["source"])
        self.assertEqual(ref["case_count"], 20)

    def test_polarimetry_solve_matches_live_mathematica_forward_only(self):
        from benchmarks.generate_multilayer_polarimetry_benchmarks import build_cases
        from shaarp.multilayer_shg_boundary import (
            sample_rotate_reflected_2omega_amplitudes,
            sample_rotate_transmitted_2omega_amplitudes,
            solve_multilayer_shg_from_system_polarimetry,
        )

        ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        items = {it["id"]: it for it in ref["suites"][0]["items"]}
        self.assertEqual(len(items), 20)

        checked = 0
        for case in build_cases():
            self.assertIn(case["id"], items)
            psi = float(case["psi_deg"])
            shg = solve_multilayer_shg_from_system_polarimetry(
                case["system"], inhomogeneous_source_policy="forward_only", mu=1.0, eps0=1.0
            ).shg
            r_par, r_perp = sample_rotate_reflected_2omega_amplitudes(shg, psi)
            t_par, t_perp = sample_rotate_transmitted_2omega_amplitudes(shg, psi)
            py = np.array([abs(r_par) ** 2, abs(r_perp) ** 2, abs(t_par) ** 2, abs(t_perp) ** 2])

            row = items[case["id"]]["mathematica_outputs"]["sampleRotateList"][0]
            mma = np.array([_num(row[1]), _num(row[2]), _num(row[3]), _num(row[4])])
            np.testing.assert_allclose(py, mma, atol=1e-10, rtol=1e-9)
            checked += 1
        self.assertEqual(checked, 20)

    def test_polarimetry_sweep_wrapper_matches_under_forward_only(self):
        """The public sweep wrapper solve_multilayer_shg_polarimetry_sweep, run under
        the SHAARP.ml-matching forward-only source policy, also matches live Mathematica
        SampleRotate to ~1e-14 (its default policy='all' is a different convention)."""
        from benchmarks.generate_multilayer_polarimetry_benchmarks import build_cases
        from shaarp.multilayer_shg_boundary import (
            sample_rotate_reflected_2omega_amplitudes,
            sample_rotate_transmitted_2omega_amplitudes,
            solve_multilayer_shg_polarimetry_sweep,
        )

        ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        items = {it["id"]: it for it in ref["suites"][0]["items"]}

        checked = 0
        for case in build_cases():
            psi = float(case["psi_deg"])
            sweep = solve_multilayer_shg_polarimetry_sweep(
                case["system"], inhomogeneous_source_policy="forward_only", mu=1.0, eps0=1.0
            )
            shg = sweep.results[0].shg
            r_par, r_perp = sample_rotate_reflected_2omega_amplitudes(shg, psi)
            t_par, t_perp = sample_rotate_transmitted_2omega_amplitudes(shg, psi)
            py = np.array([abs(r_par) ** 2, abs(r_perp) ** 2, abs(t_par) ** 2, abs(t_perp) ** 2])
            row = items[case["id"]]["mathematica_outputs"]["sampleRotateList"][0]
            mma = np.array([_num(row[1]), _num(row[2]), _num(row[3]), _num(row[4])])
            np.testing.assert_allclose(py, mma, atol=1e-10, rtol=1e-9)
            checked += 1
        self.assertEqual(checked, 20)


if __name__ == "__main__":
    unittest.main()
