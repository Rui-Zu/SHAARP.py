"""Gated ML agreement test at GRAZING INCIDENCE (mlgraze) — the R1 angle band, multilayer side.

WHY THIS EXISTS. After the SI side's grazing hole was closed (sigraze: theta 75..89.5 deg,
agreement 3.6e-15), enumerating the MULTILAYER live references showed the same class of hole,
wider: every multilayer family — ml1..ml8, pgml, pgml2, pg2act — sweeps theta 0/20/40/60 deg
(polarimetry <= 52, sample rotation <= 42), so the multilayer certified ceiling was 60 deg while
the GUI exposes the full range and its matrix sweep drives theta = 89. The multilayer path is
where grazing is hardest: cos(theta_T) shrinks in every layer, exp(i k_z h) propagation phases
chain across seven internal interfaces, and the per-interface boundary blocks are
worst-conditioned.

THE SET. `benchmarks/generate_maker_fringes_multilayer_mlgraze_benchmarks.py` clones ml8 (the
DEEPEST live-validated stack: 9 layers, 2 anisotropic + 4 isotropic passive interlayers) with ONLY
the incidence angles changed — same RAW_CASES, same interlayer helpers, same build_system seeds —
so agreement here is attributable to the ANGLE. The film eps is complex with three distinct
principal values: absorbing AND biaxial as well as grazing.

WHY THE GRID STOPS AT 83 DEG (mapped): the live SHAARP.ml `f4NL` itself kills the
WolframKernel — silent process exit 0, no stderr, no output — above a CASE-DEPENDENT onset angle
(single-solve kernel sessions, 2-4 attempts per angle):

    case 1 (b1_o0): 75..87.5 pass | 88, 89, 89.5 die -> onset 88.0
    case 2 (b4_o2): 75..83 pass | 84 dies -> onset 84.0 <- binds the common grid
    case 3 (b8_o0): 75..87.5 pass | none found -> onset > 87.5
    case 4 (b15_o1): 75..86 pass | 87 dies -> onset 87.0

A breadcrumb-instrumented exporter localized the death INSIDE f4NL (the 'entering f4NL' crumb
lands, the 'returned' crumb never does), under the exact validated exporter configuration used
for ml1..ml8. The onset varying 84 -> >=88 across stacks differing only in deterministic seeds
and orientations shows a numerical boundary of the original's grazing solve, not one pathological
input. The Python port solves the full 75..89.5 range on the same inputs with ~1e-15 boundary
residuals — certification above 83 deg is blocked by the ORACLE, not the port. The multilayer
certified ceiling therefore moves 60 -> 83 deg (largest common band), and the per-case onsets are
a documented original-tool limitation (residual-risk register R1). Diagnostic footnote: this
initially masqueraded as machine memory pressure (multi-solve sessions died while free RAM sat
under 1 GB) until single-solve bisection separated the variables — the deaths were
angle-deterministic all along.

Skipped until the live Wolfram f4NL reference has been exported, matching the ml1..ml8 pattern.
"""

import json
import unittest
from pathlib import Path

from benchmarks.compare_maker_fringes_reference import (
    build_maker_reference_agreement_summary,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_mlgraze.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_mlgraze.json"

ATOL = 1e-9
RTOL = 1e-9
EXPECTED_CASE_COUNT = 4

# Every angle sits ABOVE the previous multilayer ceiling (60 deg) and BELOW the earliest mapped
# per-case f4NL kernel-death onset (84 deg, case 2) — the band every case can actually export.
PREVIOUS_MULTILAYER_THETA_CEILING_DEG = 60.0
EARLIEST_ORIGINAL_TOOL_KERNEL_DEATH_ONSET_DEG = 84.0
EXPECTED_THETA_DEG = [75.0, 78.0, 80.0, 81.0, 82.0, 83.0]


@unittest.skipUnless(REFERENCE_PATH.exists(), "mlgraze live Wolfram reference not yet exported")
class MakerFringesMultilayerGrazingReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.python = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

    def test_reference_is_live_wolfram_sourced_9_layer_grazing(self):
        source = str(self.reference.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        self.assertEqual(self.reference.get("case_count"), EXPECTED_CASE_COUNT)
        self.assertEqual(self.reference["functionDownValues"]["f4NL"], 1)
        items = self.reference["suites"][0]["items"]
        self.assertFalse([item for item in items if "error" in item])
        for case in self.python["cases"]:
            self.assertEqual(case["layer_count"], 9)

    def test_every_case_sweeps_only_the_newly_certified_grazing_band(self):
        """Guards the REASON the set exists, from both sides: at/below 60 deg it would certify
        nothing new (ml8 already covers it); at/above 84 deg the original f4NL kills the kernel
        for at least one case (case 2's mapped onset), so a uniform-grid reference claiming those
        angles could not have come from the live oracle."""
        for case in self.python["cases"]:
            self.assertEqual(case["theta_deg"], EXPECTED_THETA_DEG)
            for theta in case["theta_deg"]:
                self.assertGreater(
                    theta, PREVIOUS_MULTILAYER_THETA_CEILING_DEG,
                    f"theta={theta} is at or below the pre-existing multilayer ceiling "
                    f"({PREVIOUS_MULTILAYER_THETA_CEILING_DEG} deg, the ml1..ml8/pgml sweep top)")
                self.assertLess(
                    theta, EARLIEST_ORIGINAL_TOOL_KERNEL_DEATH_ONSET_DEG,
                    f"theta={theta} is at or above the earliest mapped f4NL kernel-death onset "
                    f"({EARLIEST_ORIGINAL_TOOL_KERNEL_DEATH_ONSET_DEG} deg, case b4_o2): the live "
                    f"oracle cannot export a uniform grid there, so such a reference cannot be "
                    f"genuine")

    def test_mlgraze_grid_matches_live_mathematica(self):
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
        while the loose 1e-9 gate still reads green — same regression-fence rationale as the
        sigraze twin on the SI side (measured there: 3.6e-15; ml8 at 0..60 deg: ~6.5e-15)."""
        summary = build_maker_reference_agreement_summary(
            self.reference, self.python, atol=ATOL, rtol=RTOL
        )
        worst = max(k["max_abs_error"] for k in summary["nonsingular_outputs"])
        self.assertLess(
            worst, 1e-12,
            f"multilayer grazing agreement degraded: max_abs_error={worst:.3e} across outputs "
            f"(the live comparison achieved ~1e-14 when this fence was frozen)")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
