import sys
import unittest
from pathlib import Path

from benchmarks.detect_wolfram_batch import _probe_script, _wolfram_path, default_candidate_paths, detect_wolfram_batch, probe_candidate


@unittest.skipUnless(sys.platform == "win32",
                     "Wolfram kernel discovery targets Windows install layouts; the assertions use "
                     "literal Windows paths (raw C:\\ strings parse as a single POSIX component)")
class WolframBatchDetectionTests(unittest.TestCase):
    def test_wolfram_path_uses_forward_slashes_for_kernel_strings(self):
        self.assertEqual(_wolfram_path(Path(r"C:\Users\test\probe.wl")), "C:/Users/test/probe.wl")

    def test_probe_script_exports_json_to_requested_path(self):
        script = _probe_script(Path(r"C:\temp\probe_output.json"))
        self.assertIn('Export["C:/temp/probe_output.json"', script)
        self.assertIn('"result" -> 2 + 2', script)
        self.assertIn("Quit[]", script)

    def test_probe_candidate_reports_missing_executable_without_running(self):
        result = probe_candidate(Path(r"C:\definitely_missing_wolfram_kernel.exe"), timeout=0.1)
        self.assertFalse(result["exists"])
        self.assertFalse(result["usable"])
        self.assertEqual(result["strategies"], [])

    def test_default_candidates_include_new_player_version_pattern(self):
        candidates = [str(path) for path in default_candidate_paths()]
        self.assertTrue(any(r"Wolfram Player\14.3\WolframKernel.exe" in path for path in candidates))
        self.assertTrue(any(r"Wolfram Engine\14.3\WolframKernel.exe" in path for path in candidates))
        self.assertFalse(any(r"Mathematica\13.3" in path for path in candidates))

    def test_mathematica_candidates_are_opt_in(self):
        candidates = [str(path) for path in default_candidate_paths(include_mathematica=True)]
        self.assertTrue(any(r"Wolfram Player\14.3\WolframKernel.exe" in path for path in candidates))
        self.assertTrue(any(r"Wolfram Engine\14.3\WolframKernel.exe" in path for path in candidates))
        self.assertTrue(any(r"Mathematica\13.3\WolframKernel.exe" in path for path in candidates))

    def test_detection_payload_records_free_runtime_scope_by_default(self):
        payload = detect_wolfram_batch([], timeout=0.1)
        self.assertEqual(payload["default_probe_scope"], "wolfram_player_and_engine")
        self.assertFalse(payload["usable_kernel_found"])


if __name__ == "__main__":
    unittest.main()
