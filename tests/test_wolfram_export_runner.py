import json
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.run_wolfram_export_script import ExportRunResult, run_wolfram_export_script


class WolframExportRunnerTests(unittest.TestCase):
    def test_runner_uses_temp_output_and_copies_verified_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.wls"
            target = tmp_path / "target.json"
            source.write_text(
                'Export[FileNameJoin[{Directory[], "old.json"}], <|"result" -> 4|>, "JSON"];',
                encoding="utf-8",
            )

            def fake_runner(command, **kwargs):
                script_text = Path(command[2]).read_text(encoding="utf-8")
                marker = 'Export["'
                start = script_text.index(marker) + len(marker)
                end = script_text.index('"', start)
                temp_output = Path(script_text[start:end])
                temp_output.write_text('{"result":4}', encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="license warning")

            result = run_wolfram_export_script(
                source,
                target,
                executable=Path("fake-kernel.exe"),
                original_export_expression='FileNameJoin[{Directory[], "old.json"}]',
                runner=fake_runner,
            )

            self.assertIsInstance(result, ExportRunResult)
            self.assertTrue(result.output_created)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["result"], 4)
            self.assertIn("license warning", result.stderr)

    def test_runner_can_use_explicit_temp_root_for_wolfram_writable_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.wls"
            target = tmp_path / "target.json"
            temp_root = tmp_path / "wolfram-temp-root"
            source.write_text(
                'Export["old.json", <|"result" -> 4|>, "JSON"];',
                encoding="utf-8",
            )

            def fake_runner(command, **kwargs):
                temp_script = Path(command[2])
                self.assertIn(temp_root, temp_script.parents)
                script_text = temp_script.read_text(encoding="utf-8")
                marker = 'Export["'
                start = script_text.index(marker) + len(marker)
                end = script_text.index('"', start)
                temp_output = Path(script_text[start:end])
                self.assertIn(temp_root, temp_output.parents)
                temp_output.write_text('{"result":4}', encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = run_wolfram_export_script(
                source,
                target,
                executable=Path("fake-kernel.exe"),
                original_export_expression='"old.json"',
                temp_root=temp_root,
                runner=fake_runner,
            )

            self.assertTrue(result.output_created)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["result"], 4)

    def test_runner_replaces_quoted_path_literal_when_given_inner_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.wls"
            target = tmp_path / "target.json"
            source.write_text('Export["old.json", <|"result" -> 4|>, "JSON"];', encoding="utf-8")

            def fake_runner(command, **kwargs):
                script_text = Path(command[2]).read_text(encoding="utf-8")
                self.assertIn('Export["', script_text)
                self.assertNotIn('Export[""', script_text)
                marker = 'Export["'
                start = script_text.index(marker) + len(marker)
                end = script_text.index('"', start)
                Path(script_text[start:end]).write_text('{"result":4}', encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = run_wolfram_export_script(
                source,
                target,
                executable=Path("fake-kernel.exe"),
                original_export_expression="old.json",
                runner=fake_runner,
            )

            self.assertTrue(result.output_created)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["result"], 4)

    def test_runner_waits_briefly_for_exported_json_to_become_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.wls"
            target = tmp_path / "target.json"
            source.write_text('Export["old.json", <|"result" -> 4|>, "JSON"];', encoding="utf-8")

            def fake_runner(command, **kwargs):
                script_text = Path(command[2]).read_text(encoding="utf-8")
                marker = 'Export["'
                start = script_text.index(marker) + len(marker)
                end = script_text.index('"', start)
                temp_output = Path(script_text[start:end])
                threading.Timer(0.05, lambda: temp_output.write_text('{"result":4}', encoding="utf-8")).start()
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = run_wolfram_export_script(
                source,
                target,
                executable=Path("fake-kernel.exe"),
                original_export_expression='"old.json"',
                output_wait=1.0,
                runner=fake_runner,
            )

            self.assertTrue(result.output_created)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["result"], 4)

    def test_runner_can_keep_failed_temp_directory_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.wls"
            target = tmp_path / "target.json"
            temp_root = tmp_path / "wolfram-temp-root"
            source.write_text('Export["old.json", <|"result" -> 4|>, "JSON"];', encoding="utf-8")

            def fake_runner(command, **kwargs):
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = run_wolfram_export_script(
                source,
                target,
                executable=Path("fake-kernel.exe"),
                original_export_expression='"old.json"',
                temp_root=temp_root,
                output_wait=0.0,
                keep_temp_on_failure=True,
                runner=fake_runner,
            )

            self.assertFalse(result.output_created)
            self.assertTrue(Path(result.temp_output).parent.exists())
            self.assertTrue((Path(result.temp_output).parent / source.with_suffix(".wl").name).exists())

    def test_runner_rejects_scripts_without_expected_export_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.wls"
            target = tmp_path / "target.json"
            source.write_text('Export["somewhere.json", <|"result" -> 4|>, "JSON"];', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "original export expression"):
                run_wolfram_export_script(
                    source,
                    target,
                    executable=Path("fake-kernel.exe"),
                    original_export_expression='FileNameJoin[{Directory[], "old.json"}]',
                    runner=lambda *args, **kwargs: None,
                )


if __name__ == "__main__":
    unittest.main()
