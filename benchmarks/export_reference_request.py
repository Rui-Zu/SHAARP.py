from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PYTHON_BENCHMARK = Path(__file__).resolve().parent / "numeric_benchmarks_v1.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "mathematica_reference" / "reference_request_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark inputs requested from Mathematica SHAARP.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_reference_request(json.loads(args.python.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote Mathematica reference request to {args.output}")


def build_reference_request(python_payload: dict) -> dict:
    cases = []
    for case in python_payload["cases"]:
        cases.append(
            {
                "id": case["id"],
                "polarimetry": case["polarimetry"],
                "inputs": case["inputs"],
                "requested_outputs": [
                    "single_interface_intensity",
                    "multilayer_reflected_intensity",
                    "multilayer_transmitted_intensity",
                ],
            }
        )
    return {
        "status": "mathematica_reference_values_requested",
        "source_python_benchmark_version": python_payload.get("benchmark_version"),
        "case_count": len(cases),
        "notes": [
            "Run these cases in Mathematica SHAARP and export numeric outputs with matching case ids.",
            "This file is a request/input specification, not a Mathematica validation result.",
            "Use compare_mathematica_reference.py after Mathematica outputs are available.",
        ],
        "cases": cases,
    }


if __name__ == "__main__":
    main()
