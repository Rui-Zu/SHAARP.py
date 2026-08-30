from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "mathematica_reference" / "wolfram_batch_detection_v1.json"
DEFAULT_PLAYER_CANDIDATES = [
    Path(r"C:\Program Files\Wolfram Research\Wolfram Player\14.3\WolframKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Wolfram Player\14.3\MathKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Wolfram Player\13.0\WolframKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Wolfram Player\13.0\MathKernel.exe"),
]
DEFAULT_ENGINE_CANDIDATES = [
    Path(r"C:\Program Files\Wolfram Research\Wolfram Engine\14.3\WolframKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Wolfram Engine\14.3\MathKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Wolfram Engine\14.3\wolfram.exe"),
    Path(r"C:\Program Files\Wolfram Research\Wolfram Engine\14.3\math.exe"),
]
MATHEMATICA_CANDIDATES = [
    Path(r"C:\Program Files\Wolfram Research\Mathematica\14.3\WolframKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Mathematica\14.3\MathKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Mathematica\13.3\WolframKernel.exe"),
    Path(r"C:\Program Files\Wolfram Research\Mathematica\13.3\MathKernel.exe"),
    Path(r"C:\Program Files (x86)\Wolfram Research\WolframScript\wolframscript.exe"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect whether a local Wolfram kernel can run batch exports.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--candidate", action="append", type=Path, default=None)
    parser.add_argument(
        "--include-mathematica",
        action="store_true",
        help="Also probe Mathematica/wolframscript executables. Default probes Wolfram Player and Wolfram Engine only.",
    )
    args = parser.parse_args()

    payload = detect_wolfram_batch(
        args.candidate or default_candidate_paths(include_mathematica=args.include_mathematica),
        timeout=args.timeout,
        include_mathematica=args.include_mathematica,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote Wolfram batch detection report to {args.output}")
    print(f"usable_kernel_found={payload['usable_kernel_found']}")


def default_candidate_paths(*, include_mathematica: bool = False) -> list[Path]:
    discovered = []
    for root in [Path(r"C:\Program Files\Wolfram Research"), Path(r"C:\Program Files (x86)\Wolfram Research")]:
        if not root.exists():
            continue
        for executable_name in ["WolframKernel.exe", "MathKernel.exe", "wolframscript.exe"]:
            for path in root.glob(f"**/{executable_name}"):
                if include_mathematica or _is_default_free_runtime_path(path):
                    discovered.append(path)
    paths = []
    seen = set()
    base = [*DEFAULT_PLAYER_CANDIDATES, *DEFAULT_ENGINE_CANDIDATES]
    if include_mathematica:
        base.extend(MATHEMATICA_CANDIDATES)
    for path in [*base, *discovered]:
        key = str(path).lower()
        if key not in seen:
            paths.append(path)
            seen.add(key)
    return paths


def detect_wolfram_batch(
    candidates: list[Path],
    *,
    timeout: float = 20.0,
    include_mathematica: bool = False,
) -> dict[str, Any]:
    probes = [probe_candidate(path, timeout=timeout) for path in candidates]
    usable = [probe for probe in probes if probe["usable"]]
    return {
        "status": "wolfram_batch_detection",
        "default_probe_scope": "wolfram_player_and_engine" if not include_mathematica else "wolfram_player_engine_and_mathematica",
        "usable_kernel_found": bool(usable),
        "usable_candidates": [probe["executable"] for probe in usable],
        "notes": [
            "A Wolfram installation is only usable for automated SHAARP validation if it can execute code and export JSON.",
            "Default detection probes Wolfram Player and Wolfram Engine because an expired Mathematica license may make Mathematica executables unusable.",
            "Use --include-mathematica only when a valid Mathematica license is available or when explicitly testing Mathematica.",
            "Wolfram Player may be installed but still fail this batch-code probe; Wolfram Engine is the expected free command-line runtime.",
        ],
        "probes": probes,
    }


def _is_default_free_runtime_path(path: Path) -> bool:
    raw = str(path).lower()
    return "wolfram player" in raw or "wolfram engine" in raw


def probe_candidate(executable: Path, *, timeout: float) -> dict[str, Any]:
    if not executable.exists():
        return {
            "executable": str(executable),
            "exists": False,
            "usable": False,
            "strategies": [],
        }
    strategies = []
    with tempfile.TemporaryDirectory(prefix="shaarp_wolfram_probe_") as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "probe.wl"
        output = tmp_path / "probe_output.json"
        script.write_text(_probe_script(output), encoding="utf-8")
        strategies.append(_run_strategy("script", [str(executable), "-script", str(script)], output, timeout=timeout))
        if not strategies[-1]["usable"]:
            get_code = f'Get["{_wolfram_path(script)}"]'
            strategies.append(_run_strategy("run_get", [str(executable), "-noprompt", "-run", get_code], output, timeout=timeout))
    usable = any(strategy["usable"] for strategy in strategies)
    return {
        "executable": str(executable),
        "exists": True,
        "usable": usable,
        "strategies": strategies,
    }


def _run_strategy(name: str, command: list[str], output: Path, *, timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "usable": False,
            "timed_out": True,
            "returncode": None,
            "stdout": (exc.stdout or "")[:1000],
            "stderr": (exc.stderr or "")[:1000],
            "output_created": output.exists(),
        }
    output_created = output.exists()
    output_payload = None
    if output_created:
        try:
            output_payload = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            output_payload = None
    return {
        "name": name,
        "usable": bool(output_payload and output_payload.get("result") == 4),
        "timed_out": False,
        "returncode": completed.returncode,
        "stdout": completed.stdout[:1000],
        "stderr": completed.stderr[:1000],
        "output_created": output_created,
        "output_payload": output_payload,
    }


def _probe_script(output: Path) -> str:
    return "\n".join(
        [
            f'Export["{_wolfram_path(output)}", <|"source" -> "Wolfram batch probe", "result" -> 2 + 2, "version" -> $Version|>, "JSON"]',
            "Quit[]",
            "",
        ]
    )


def _wolfram_path(path: Path) -> str:
    return str(path).replace("\\", "/")


if __name__ == "__main__":
    main()
