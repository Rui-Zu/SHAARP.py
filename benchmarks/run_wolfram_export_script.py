from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


DEFAULT_WOLFRAM_EXECUTABLE = Path(r"C:\Program Files\Wolfram Research\Wolfram\14.3\WolframKernel.exe")


@dataclass(frozen=True)
class ExportRunResult:
    executable: str
    source_script: str
    target_output: str
    returncode: int
    stdout: str
    stderr: str
    output_created: bool
    temp_output: str


Runner = Callable[..., object]


def run_wolfram_export_script(
    source_script: Path,
    target_output: Path,
    *,
    executable: Path = DEFAULT_WOLFRAM_EXECUTABLE,
    original_export_expression: str,
    timeout: float = 120.0,
    temp_root: Path | None = None,
    output_wait: float = 5.0,
    keep_temp_on_failure: bool = False,
    runner: Runner = subprocess.run,
) -> ExportRunResult:
    """Run a Wolfram export script through a temp JSON file, then copy it.

    Wolfram batch execution is reliable when the script and its initial output live in a
    plain temporary directory. Exporting directly into a synced or deeply nested working
    directory can fail silently, so this helper verifies the temporary JSON before copying it
    into the repository.
    """

    source_script = Path(source_script)
    target_output = Path(target_output)
    text = source_script.read_text(encoding="utf-8")
    if original_export_expression not in text:
        raise ValueError("source script does not contain the original export expression.")

    temp_root_path = Path(temp_root) if temp_root is not None else None
    if temp_root_path is not None:
        temp_root_path.mkdir(parents=True, exist_ok=True)

    with _temporary_directory(prefix="shaarp_wolfram_export_", dir=temp_root_path) as (tmp, cleanup):
        tmp_path = Path(tmp)
        temp_output = tmp_path / target_output.name
        temp_script = tmp_path / source_script.with_suffix(".wl").name
        rewritten = _replace_export_expression(text, original_export_expression, temp_output)
        temp_script.write_text(rewritten, encoding="utf-8")

        # The kernel runs a COPY of the script from a temp directory, so the script cannot
        # locate the reference directory from its own path. Pass it explicitly.
        env = dict(os.environ)
        env["SHAARP_REF_DIR"] = str(source_script.parent.resolve())

        completed = runner(
            [str(executable), "-script", str(temp_script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        returncode = int(getattr(completed, "returncode", 0))
        stdout = str(getattr(completed, "stdout", ""))
        stderr = str(getattr(completed, "stderr", ""))
        output_created = _wait_for_output(temp_output, output_wait)
        if output_created:
            json.loads(temp_output.read_text(encoding="utf-8"))
            target_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(temp_output, target_output)
        if output_created or not keep_temp_on_failure:
            cleanup()
        return ExportRunResult(
            executable=str(executable),
            source_script=str(source_script),
            target_output=str(target_output),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            output_created=output_created,
            temp_output=str(temp_output),
        )


def _wolfram_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _replace_export_expression(text: str, original_export_expression: str, temp_output: Path) -> str:
    replacement = f'"{_wolfram_path(temp_output)}"'
    quoted_original = f'"{original_export_expression}"'
    if not original_export_expression.startswith('"') and quoted_original in text:
        return text.replace(quoted_original, replacement)
    return text.replace(original_export_expression, replacement)


def _wait_for_output(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        if path.exists():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


@contextmanager
def _temporary_directory(prefix: str, dir: Path | None):
    tmp = tempfile.mkdtemp(prefix=prefix, dir=dir)
    should_cleanup = True

    def cleanup() -> None:
        nonlocal should_cleanup
        if should_cleanup:
            shutil.rmtree(tmp, ignore_errors=True)
            should_cleanup = False

    try:
        yield tmp, cleanup
    finally:
        if should_cleanup:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Wolfram script that exports JSON through a verified temp path.")
    parser.add_argument("source_script", type=Path)
    parser.add_argument("target_output", type=Path)
    parser.add_argument("--executable", type=Path, default=DEFAULT_WOLFRAM_EXECUTABLE)
    parser.add_argument("--original-export-expression", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temp-root", type=Path, default=None)
    parser.add_argument("--output-wait", type=float, default=5.0)
    parser.add_argument("--keep-temp-on-failure", action="store_true")
    args = parser.parse_args(argv)
    result = run_wolfram_export_script(
        args.source_script,
        args.target_output,
        executable=args.executable,
        original_export_expression=args.original_export_expression,
        timeout=args.timeout,
        temp_root=args.temp_root,
        output_wait=args.output_wait,
        keep_temp_on_failure=args.keep_temp_on_failure,
    )
    print(json.dumps(result.__dict__, indent=2))
    if not result.output_created:
        raise SystemExit("Wolfram export did not create the expected JSON output.")


if __name__ == "__main__":
    main()
