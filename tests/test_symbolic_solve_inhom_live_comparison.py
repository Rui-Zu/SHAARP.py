import unittest

from benchmarks.compare_symbolic_solve_inhom_live import build_summary


class SymbolicSolveInhomLiveComparisonTests(unittest.TestCase):
    """The SYMBOLIC single-interface solveInhom (ee/oo/eo inhomogeneous 2omega
    fields) is validated against LIVE Wolfram numeric values: the symbolic
    expressions are evaluated at the exact inputs of all 20 live-Wolfram
    `solve_inhom_reference_v1.json` cases and compared to the Wolfram ee/oo/eo.
    This closes the Standing-Rule-4 gap (prior symbolic test only checked
    symbolic-vs-Python-numeric self-consistency)."""

    def test_symbolic_solve_inhom_matches_live_wolfram_all_cases(self):
        summary = build_summary(atol=1e-9, rtol=1e-9)
        self.assertEqual(summary["status"], "symbolic_solve_inhom_matches_live_wolfram")
        self.assertTrue(summary["full_symbolic_solve_inhom_agreement_claimed"])
        self.assertEqual(summary["case_count"], 20)
        self.assertEqual(summary["passing_case_count"], 20)
        self.assertEqual(summary["failing_case_count"], 0)
        self.assertLessEqual(summary["max_abs_error_over_all_cases"], 1e-9)

    def test_reference_is_live_wolfram_sourced(self):
        summary = build_summary(atol=1e-9, rtol=1e-9)
        source = str(summary.get("source", "")).lower()
        self.assertTrue("mathematica" in source or "wolfram" in source)
        # every case must report all three ee/oo/eo components passing
        for result in summary["results"]:
            self.assertEqual(set(result["components"]), {"ee", "oo", "eo"})
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
