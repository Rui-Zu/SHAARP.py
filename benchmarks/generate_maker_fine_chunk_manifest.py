from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from benchmarks.wolfram_paths import PREAMBLE


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "benchmarks" / "mathematica_reference"
DEFAULT_INPUT = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_mathematica_inputs_v1.json"
DEFAULT_PYTHON = ROOT / "benchmarks" / "maker_fringes_fine_sampling_output_v1.json"
DEFAULT_INPUT_OUT = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_input_v1.json"
DEFAULT_PYTHON_OUT = ROOT / "benchmarks" / "maker_fine_chunk_python_output_v1.json"
DEFAULT_SCRIPT_OUT = ROOT / "benchmarks" / "mathematica_reference" / "export_maker_fine_chunk_reference.wl"
DEFAULT_REFERENCE_OUT = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_reference_v1.json"
DEFAULT_DIAGNOSTIC_OUT = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_chunk_reference_diagnostics.txt"


def build_chunk_pair(
    input_payload: dict[str, Any],
    python_payload: dict[str, Any],
    *,
    case_id: str,
    start: int,
    count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if start < 0:
        raise ValueError("start must be non-negative.")
    if count <= 0:
        raise ValueError("count must be positive.")

    input_case = _case_by_id(input_payload, case_id)
    python_case = _case_by_id(python_payload, case_id)
    theta = list(input_case["theta_deg"])
    stop = start + count
    if stop > len(theta):
        raise ValueError(
            f"Requested chunk [{start}:{stop}] is outside available angles for {case_id!r} "
            f"({len(theta)} angles)."
        )

    theta_chunk = theta[start:stop]
    input_case_chunk = copy.deepcopy(input_case)
    input_case_chunk["theta_deg"] = theta_chunk

    python_case_chunk = copy.deepcopy(python_case)
    python_case_chunk["theta_deg"] = list(python_case["theta_deg"])[start:stop]
    python_case_chunk["angle_count"] = count
    python_case_chunk["outputs"] = {
        key: list(value)[start:stop] for key, value in python_case["outputs"].items()
    }

    input_chunk = {
        "benchmark_version": input_payload.get("benchmark_version", 1),
        "source": input_payload.get("source", "") + " (chunked)",
        "status": "python_fine_sampling_chunk_input_manifest_not_mathematica_validation",
        "case_count": 1,
        "total_angle_count": count,
        "chunk": {"case_id": case_id, "start": start, "count": count, "stop": stop},
        "notes": [
            "Chunked Mathematica input for resumable fine Maker validation.",
            "This is not a Mathematica validation result until exported by Wolfram.",
        ],
        "cases": [input_case_chunk],
    }
    python_chunk = {
        "status": python_payload.get("status", "") + "_chunked",
        "plan_version": python_payload.get("plan_version"),
        "case_count": 1,
        "total_angle_count": count,
        "mathematica_reference_required_for_agreement_claim": python_payload.get(
            "mathematica_reference_required_for_agreement_claim", True
        ),
        "chunk": {"case_id": case_id, "start": start, "count": count, "stop": stop},
        "cases": [python_case_chunk],
    }
    return input_chunk, python_chunk


def _case_by_id(payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in payload.get("cases", []):
        if case.get("id") == case_id:
            return case
    raise ValueError(f"Case id not found: {case_id!r}")


def wolfram_path_expression(path: str | Path) -> str:
    """Render ``path`` as a Wolfram expression for a generated exporter.

    A file inside this checkout's reference directory is emitted as ``SHAARPPaths`Ref["name"]``,
    so the generated script resolves against whatever checkout runs it. Anything else -- a scratch
    file under a temporary directory, say -- is emitted as a literal, since only the caller knows
    where it lives.
    """
    p = Path(str(path))
    try:
        inside = p.resolve().parent == REFERENCE_DIR.resolve()
    except OSError:
        inside = False
    if inside:
        return f'SHAARPPaths`Ref["{p.name}"]'
    return '"' + str(path).replace("\\", "/") + '"'


def wolfram_chunk_script_text(*, input_path: str, output_path: str, diagnostic_path: str) -> str:
    return "\n".join(
        [
            PREAMBLE,
            'BeginPackage["SHAARPPathOverrides`"];',
            f"MakerInputPath = {wolfram_path_expression(input_path)};",
            f"MakerOutputPath = {wolfram_path_expression(output_path)};",
            f"MakerDiagnosticPath = {wolfram_path_expression(diagnostic_path)};",
            "EndPackage[];",
            'Get[SHAARPPaths`Ref["export_maker_fringes_reference.wl"]];',
            "",
        ]
    )


def _wolfram_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched fine Maker input/Python output chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--input-out", type=Path, default=DEFAULT_INPUT_OUT)
    parser.add_argument("--python-out", type=Path, default=DEFAULT_PYTHON_OUT)
    parser.add_argument("--script-out", type=Path, default=DEFAULT_SCRIPT_OUT)
    parser.add_argument("--reference-out", type=Path, default=DEFAULT_REFERENCE_OUT)
    parser.add_argument("--diagnostic-out", type=Path, default=DEFAULT_DIAGNOSTIC_OUT)
    args = parser.parse_args()

    input_payload = json.loads(args.input.read_text(encoding="utf-8"))
    python_payload = json.loads(args.python.read_text(encoding="utf-8"))
    input_chunk, python_chunk = build_chunk_pair(
        input_payload,
        python_payload,
        case_id=args.case_id,
        start=args.start,
        count=args.count,
    )
    args.input_out.parent.mkdir(parents=True, exist_ok=True)
    args.python_out.parent.mkdir(parents=True, exist_ok=True)
    args.input_out.write_text(json.dumps(input_chunk, indent=2), encoding="utf-8")
    args.python_out.write_text(json.dumps(python_chunk, indent=2), encoding="utf-8")
    args.script_out.parent.mkdir(parents=True, exist_ok=True)
    args.script_out.write_text(
        wolfram_chunk_script_text(
            input_path=_wolfram_path(args.input_out.resolve()),
            output_path=_wolfram_path(args.reference_out.resolve()),
            diagnostic_path=_wolfram_path(args.diagnostic_out.resolve()),
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input_out": str(args.input_out),
                "python_out": str(args.python_out),
                "script_out": str(args.script_out),
                "reference_out": str(args.reference_out),
                "count": args.count,
            }
        )
    )


if __name__ == "__main__":
    main()
