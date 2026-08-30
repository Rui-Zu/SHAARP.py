import unittest
from pathlib import Path

from benchmarks.compare_multilayer_boundary_reference import compare_boundary_reference_files


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "solve_fresnel_n_reference_v1.json"
PYTHON_PATH = ROOT / "benchmarks" / "multilayer_boundary_system_benchmarks_v1.json"


class MathematicaSolveFresnelNReferenceTests(unittest.TestCase):
    def test_mathematica_solve_fresnel_n_reference_matches_python_values(self):
        results = compare_boundary_reference_files(
            REFERENCE_PATH,
            PYTHON_PATH,
            atol=1e-10,
            rtol=1e-10,
        )

        self.assertEqual(len(results), 555)
        self.assertTrue(all(result.passed for result in results))


if __name__ == "__main__":
    unittest.main()
