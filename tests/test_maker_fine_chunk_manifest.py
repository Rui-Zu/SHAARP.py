import json
import unittest
from pathlib import Path

from benchmarks.generate_maker_fine_chunk_manifest import build_chunk_pair, wolfram_chunk_script_text


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_mathematica_inputs_v1.json"
PYTHON_PATH = ROOT / "benchmarks" / "maker_fringes_fine_sampling_output_v1.json"


class MakerFineChunkManifestTests(unittest.TestCase):
    def test_chunk_pair_slices_input_angles_and_python_outputs_for_same_case(self):
        input_payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        python_payload = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        input_chunk, python_chunk = build_chunk_pair(
            input_payload,
            python_payload,
            case_id="fine_maker_negative_normal_positive",
            start=4,
            count=5,
        )

        self.assertEqual(input_chunk["case_count"], 1)
        self.assertEqual(input_chunk["total_angle_count"], 5)
        self.assertEqual(input_chunk["cases"][0]["id"], "fine_maker_negative_normal_positive")
        self.assertEqual(input_chunk["cases"][0]["theta_deg"], [-11.0, -10.75, -10.5, -10.25, -10.0])

        self.assertEqual(python_chunk["case_count"], 1)
        self.assertEqual(python_chunk["total_angle_count"], 5)
        py_case = python_chunk["cases"][0]
        self.assertEqual(py_case["id"], "fine_maker_negative_normal_positive")
        self.assertEqual(py_case["theta_deg"], [-11.0, -10.75, -10.5, -10.25, -10.0])
        self.assertEqual(py_case["angle_count"], 5)
        for value in py_case["outputs"].values():
            self.assertEqual(len(value), 5)

    def test_chunk_pair_rejects_ranges_outside_case_angles(self):
        input_payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        python_payload = json.loads(PYTHON_PATH.read_text(encoding="utf-8"))

        with self.assertRaisesRegex(ValueError, "outside available angles"):
            build_chunk_pair(
                input_payload,
                python_payload,
                case_id="fine_maker_negative_normal_positive",
                start=90,
                count=20,
            )

    def test_wolfram_chunk_script_text_uses_explicit_chunk_paths(self):
        """A path outside the reference directory is emitted literally: only the caller knows
        where a scratch file lives."""
        text = wolfram_chunk_script_text(
            input_path="D:/tmp/maker_chunk_input.json",
            output_path="D:/tmp/maker_chunk_reference.json",
            diagnostic_path="D:/tmp/maker_chunk_diagnostics.txt",
        )

        self.assertIn('MakerInputPath = "D:/tmp/maker_chunk_input.json"', text)
        self.assertIn('MakerOutputPath = "D:/tmp/maker_chunk_reference.json"', text)
        self.assertIn('MakerDiagnosticPath = "D:/tmp/maker_chunk_diagnostics.txt"', text)
        self.assertIn("export_maker_fringes_reference.wl", text)

    def test_reference_directory_paths_are_emitted_relative_to_the_checkout(self):
        """A file that ships beside the exporter must NOT be pinned to this machine."""
        ref_dir = ROOT / "benchmarks" / "mathematica_reference"
        text = wolfram_chunk_script_text(
            input_path=str(ref_dir / "maker_fine_chunk_input_v1.json"),
            output_path=str(ref_dir / "maker_fine_chunk_reference_v1.json"),
            diagnostic_path=str(ref_dir / "maker_fine_chunk_reference_diagnostics.txt"),
        )

        self.assertIn('MakerInputPath = SHAARPPaths`Ref["maker_fine_chunk_input_v1.json"]', text)
        self.assertIn('Get[SHAARPPaths`Ref["export_maker_fringes_reference.wl"]]', text)
        self.assertNotIn(":/", text.split("(* ---")[-1].split("*)")[-1])

    def test_generated_wrapper_matches_a_shipped_wrapper(self):
        """The generator and the 58 shipped wrappers must not drift apart -- regenerate one and
        compare it to the file on disk."""
        ref_dir = ROOT / "benchmarks" / "mathematica_reference"
        shipped = ref_dir / "export_maker_fine_chunk_fill_frng_125_60_reference.wl"
        if not shipped.is_file():
            self.skipTest(f"{shipped.name} is not present")
        slug = "fill_frng_125_60"
        regenerated = wolfram_chunk_script_text(
            input_path=str(ref_dir / f"maker_fine_chunk_{slug}_input_v1.json"),
            output_path=str(ref_dir / f"maker_fine_chunk_{slug}_reference_v1.json"),
            diagnostic_path=str(ref_dir / f"maker_fine_chunk_{slug}_diagnostics.txt"),
        )
        self.assertEqual(regenerated.strip(),
                         shipped.read_text(encoding="utf-8").strip(),
                         "the generator no longer reproduces the shipped wrapper")


if __name__ == "__main__":
    unittest.main()
