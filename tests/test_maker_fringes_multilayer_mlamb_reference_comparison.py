"""GATED live-Mathematica comparison for a NON-AIR INCIDENT MEDIUM (tag: mlamb) — F58.

WHY THIS FENCE EXISTS. The SHAARP.py GUI lets the user set BOTH half-spaces of a stack. The
RELEASED Mathematica ♯SHAARP.ml GUI does not: `SHAARP.ml.nb:666-716` forces `m1 = setMater@Air[]`
and `:425-439` overwrites the exit slot (`allmaterials[[materialnumber+2 ;; ...]] = Table[mbot,...]`
with mbot = air), and the layer selector `Range[2, materialnumber+1]` reaches neither. Settable
half-spaces are therefore an EXTENSION of the original tool ("this is a
fundamentally different capability than what we have in mathematica where we hard coded both top
and bottom surface to be air"), and an extension gets live evidence rather than an assumption.

WHAT WAS ALREADY COVERED vs WHAT THIS ADDS.
  * NON-AIR EXIT medium: already certified — every multilayer manifest (ml1..ml8, pg*, pgml*,
    ext1, mlgraze, polarimetry, sample_rotation) carries an ABSORBING, ANISOTROPIC, ROTATED
    substrate, and the engine always accepted an arbitrary one (`setup.nb:6421` documents wSub as
    "waves into Substrate (can be Air)"; f4NL takes `mSub = mAll[[-1]]` with no vacuum branch).
  * NON-AIR ENTRANCE medium: never exercised before this set. All 27 prior input manifests set
    layer 0 to eps = I, so the Mathematica global
    `n0 = Sqrt[Mean[Eigenvalues[mats[[1]][epsOmegaC]]]]`
    (export_maker_fringes_reference.wl:119) was identically 1 in every live reference.

HOW THE ORACLE WAS DRIVEN WITHOUT TOUCHING IT. n0 is derived per case from layer 0 of the
manifest and consumed by `setwInc` as the ambient index
(`wInc = setWave[{w, n0 (w/c0){Sin,0,Cos}, n0 w/c0, theta, Einc}]`), so it scales k and k0 and
therefore the tangential k_x every Snell/Fresnel step chains off. Setting layer 0 to
eps = n^2 I in the PYTHON generator makes the original compute a non-air ambient — the Wolfram
wrapper is the usual 4-line path-override clone, with zero exporter edits.

MEASURED (live Wolfram 14.3): all output families agree, worst max_abs_error ~5.6e-14
across both cases. Kernel note: the run ended with an ACCESS VIOLATION at teardown AFTER both
cases completed (breadcrumbs "Finished case 1/2") and after the JSON was written and validated —
the reference is complete and live-sourced; the crash is a shutdown artifact, not a compute
failure (cf. the mlgraze S16 kernel deaths, which happened INSIDE L and produced no output).
"""

import json
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_maker_fringes_reference import (
    build_maker_reference_agreement_summary,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_mlamb.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_mlamb.json"
AIR_TWIN_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_mlamb_airtwin.json"
MANIFEST_PATH = (ROOT / "benchmarks" / "mathematica_reference"
                 / "maker_mathematica_inputs_mlamb.json")

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 2
EXPECTED_THETA_DEG = [20.0, 40.0, 60.0]
AMBIENT_N_OMEGA = 1.33
AMBIENT_N_2OMEGA = 1.34


@unittest.skipUnless(REFERENCE_PATH.exists(), "mlamb live Wolfram reference not yet exported")
class MakerFringesNonAirAmbientReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_reference_is_live_wolfram_sourced_9_layer(self):
        source = str(self.reference.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        self.assertEqual(self.reference.get("case_count"), EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        items = self.reference["suites"][0]["items"]
        self.assertFalse([item for item in items if "error" in item])
        for case in self.python["cases"]:
            self.assertEqual(case["layer_count"], 9)
            self.assertEqual(case["theta_deg"], EXPECTED_THETA_DEG)

    def test_the_manifest_really_carries_a_non_air_isotropic_ambient(self):
        """Guards the REASON the set exists. If layer 0 ever regressed to eps = I this reference
        would silently certify nothing new (it would be an ml8 duplicate at 20/40/60 deg), and if
        it were anisotropic the two sides would not be like-for-like: Mathematica averages
        eigenvalues for n0 while the port's _isotropic_index REQUIRES eps = scalar * I."""
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            eps = case["layers"][0]["epsilon_omega_crystal"]
            diag = [complex(eps[k][k]["real"], eps[k][k]["imag"]) for k in range(3)]
            off = [complex(eps[r][c]["real"], eps[r][c]["imag"])
                   for r in range(3) for c in range(3) if r != c]
            self.assertTrue(all(abs(v) <= 1e-12 for v in off),
                            "the ambient must be ISOTROPIC (no off-diagonal eps)")
            for v in diag:
                self.assertAlmostEqual(v.real, AMBIENT_N_OMEGA ** 2, places=9)
                self.assertLessEqual(abs(v.imag), 1e-12, "a real ambient keeps n0 real")
            self.assertGreater(abs(diag[0] - 1.0), 0.5,
                               "layer 0 must NOT be air — that is the whole point of mlamb "
                               "(every prior manifest pinned n0 = 1)")

    def test_mlamb_grid_matches_live_mathematica(self):
        summary = build_maker_reference_agreement_summary(
            self.reference, self.python, atol=ATOL, rtol=RTOL
        )
        self.assertEqual(summary["status"], "maker_outputs_match_all_compared_cases")
        self.assertTrue(summary["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(summary["nonsingular_case_count"], EXPECTED_CASE_COUNT)
        self.assertEqual(summary["nonsingular_fail_count"], 0)
        self.assertEqual(summary["diagnostic_case_ids"], [])
        self.assertTrue(summary["nonsingular_outputs"])
        for key_result in summary["nonsingular_outputs"]:
            self.assertEqual(key_result["comparison_status"], "passed")
            self.assertEqual(key_result["case_count"], EXPECTED_CASE_COUNT)
            self.assertLessEqual(key_result["max_abs_error"], ATOL)

    def test_agreement_is_machine_precision_not_merely_inside_the_gate(self):
        """Pin the ACHIEVED level (with headroom) so a silent degradation to ~1e-10 cannot ship
        while the loose 1e-9 gate still reads green (measured 5.6e-14 when frozen)."""
        summary = build_maker_reference_agreement_summary(
            self.reference, self.python, atol=ATOL, rtol=RTOL
        )
        worst = max(k["max_abs_error"] for k in summary["nonsingular_outputs"])
        self.assertLess(
            worst, 1e-12,
            f"non-air-ambient agreement degraded: max_abs_error={worst:.3e} across outputs "
            f"(the live comparison achieved ~5.6e-14 when this fence was frozen)")

    @unittest.skipUnless(AIR_TWIN_PATH.exists(), "mlamb air-ambient twin not generated")
    def test_the_ambient_index_actually_reaches_the_solve(self):
        """The un-fakeable half: agreement alone cannot prove the ambient MATTERS — if n0 were
        ignored on BOTH sides the curves would still match. The air twin runs the SAME stacks
        with layer 0 = air, and the observables must differ materially.

        (Compared on the ANALYZER AMPLITUDES, not on list_mf_para: SHAARP's copy lists lead with
        the ANGLE column, whose identical 20/40/60 values dominate a naive relative difference —
        that artifact made a first draft of this check read 8.5e-4 for what is really a 45%
        change at 20 deg and a 3x change at 60 deg.)"""
        twin = json.loads(AIR_TWIN_PATH.read_text(encoding="utf-8"))
        twin_by_id = {c["id"]: c for c in twin["cases"]}
        worst_rel = 0.0
        for case in self.python["cases"]:
            other = twin_by_id[case["id"]]
            for key in ("parallel_amplitude", "perpendicular_amplitude"):
                a = np.array([complex(v["real"], v["imag"]) for v in case["outputs"][key]])
                b = np.array([complex(v["real"], v["imag"]) for v in other["outputs"][key]])
                scale = max(float(np.max(np.abs(b))), 1e-300)
                worst_rel = max(worst_rel, float(np.max(np.abs(a - b))) / scale)
        self.assertGreater(
            worst_rel, 1e-3,
            f"the ambient index does not change the result (worst relative difference vs the "
            f"air twin = {worst_rel:.3e}) — n0 is not reaching the solve, so the agreement above "
            f"would be vacuous")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
