from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG = ROOT / "benchmarks" / "mathematica_reference" / "sample_rotation_reference_debug.txt"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "mathematica_reference" / "sample_rotation_reference_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Mathematica SampleRotate InputForm debug values to JSON.")
    parser.add_argument("--debug", type=Path, default=DEFAULT_DEBUG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = convert_debug_text(args.debug.read_text(encoding="utf-8"))
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SampleRotate reference JSON to {args.output}")


def convert_debug_text(text: str) -> dict:
    cases = []
    case_pattern = re.compile(
        r'<\|"id" -> "(?P<id>[^"]+)", "source_case_id" -> "(?P<source>[^"]+)".*?'
        r'"sampleRotateList" -> \{\{(?P<table>.*?)\}\}\|>\|>',
        re.DOTALL,
    )
    for match in case_pattern.finditer(text):
        values = [_parse_wolfram_number(item) for item in re.findall(r'"real" -> ([^,|}]+)', match.group("table"))]
        if len(values) % 5 != 0:
            raise ValueError(f"Case {match.group('id')} has {len(values)} real values, not a multiple of 5.")
        rows = [values[idx : idx + 5] for idx in range(0, len(values), 5)]
        cases.append(
            {
                "id": match.group("id"),
                "source_case_id": match.group("source"),
                "mathematica_outputs": {
                    "sampleRotateList": rows,
                },
            }
        )
    if not cases:
        raise ValueError("No SampleRotate cases were parsed from Mathematica debug text.")
    return {
        "source": "Mathematica SHAARP.ml setup.nb SampleRotate reference values converted from InputForm debug export",
        "status": "mathematica_reference_exported_with_python_json_conversion",
        "case_count": len(cases),
        "suites": [
            {
                "suite_id": "sample_rotation_workflow",
                "items": cases,
            }
        ],
    }


def _parse_wolfram_number(value: str) -> float:
    return float(value.strip().replace("*^", "e"))


if __name__ == "__main__":
    main()
