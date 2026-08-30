import json
import unittest
from pathlib import Path

from benchmarks.generate_point_group_tensor_patterns_from_si_source import build_reference
from shaarp.symbolic import d_voigt_symbolic


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "point_group_tensor_patterns_from_si_source_v1.json"


class PointGroupTensorSourceReferenceTests(unittest.TestCase):
    def test_saved_reference_matches_current_si_source_extraction(self):
        saved = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(saved, build_reference())

    def test_source_derived_point_group_patterns_match_python_symbolic_tensors(self):
        reference = build_reference()
        self.assertEqual(reference["status"], "source_derived_from_shaarp_si_partial_analytical_branch")
        self.assertGreaterEqual(reference["pattern_count"], 16)

        for pattern in reference["patterns"]:
            for label in pattern["point_group_labels"]:
                with self.subTest(label=label):
                    self.assertEqual(
                        _matrix_to_tokens(d_voigt_symbolic(label)),
                        pattern["matrix_tokens"],
                    )


def _matrix_to_tokens(matrix):
    return [[str(matrix[row, col]).replace("-", "-") for col in range(matrix.cols)] for row in range(matrix.rows)]


if __name__ == "__main__":
    unittest.main()
