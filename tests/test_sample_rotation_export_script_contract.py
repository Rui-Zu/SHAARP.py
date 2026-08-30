import unittest
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "benchmarks" / "mathematica_reference" / "export_sample_rotation_reference.wl"


class SampleRotationExportScriptContractTests(unittest.TestCase):
    def test_script_exports_constructed_output_payload_to_output_path(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")

        payload_index = text.index("outputPayload =")
        export_index = text.index("Export[\n  outputPath,")
        quit_index = text.index("Quit[]")

        self.assertLess(payload_index, export_index)
        self.assertLess(export_index, quit_index)
        self.assertIn("outputPayload", text[export_index:quit_index])
        self.assertIn('"JSON"', text[export_index:quit_index])

    def test_output_payload_suites_is_a_list_inside_outer_association(self):
        text = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertRegex(
            text,
            re.compile(
                r'"suites"\s*->\s*\{\s*<\|"suite_id"\s*->\s*"sample_rotation_workflow",\s*"items"\s*->\s*cases\|>\s*\}\s*\|>;',
                re.MULTILINE,
            ),
        )


if __name__ == "__main__":
    unittest.main()
