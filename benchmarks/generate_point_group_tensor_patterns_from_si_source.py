from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
SOURCE_PATH = BENCHMARK_DIR / "mathematica_reference" / "shaarp_si_frontend_sections.json"
DEFAULT_OUTPUT = (
    BENCHMARK_DIR
    / "mathematica_reference"
    / "point_group_tensor_patterns_from_si_source_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SHAARP.si point-group d tensor patterns from source text.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reference = build_reference()
    args.output.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"Wrote point-group tensor source reference to {args.output}")


def build_reference() -> dict:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    block = _extract_partial_analytical_doldexp_block(source["records"])
    patterns = _parse_patterns(block)
    return {
        "reference_version": 1,
        "source": "Mathematica SHAARP.si front-end text extraction",
        "source_file": "mathematica_reference/shaarp_si_frontend_sections.json",
        "notebook_path": source["notebook_path"],
        "wolfram_version_at_extraction": source["wolfram_version"],
        "status": "source_derived_from_shaarp_si_partial_analytical_branch",
        "validation_level": "source_text_pattern_extraction_not_live_symbolic_execution",
        "note": (
            "Patterns are parsed from the SHAARP.si Partial Analytical Expressions "
            "doldExp branch. This validates Python's point-group tensor table against "
            "the extracted notebook source branch; it is not a full Wolfram execution "
            "of the complete analytical workflow."
        ),
        "pattern_count": len(patterns),
        "label_count": sum(len(pattern["point_group_labels"]) for pattern in patterns),
        "patterns": patterns,
    }


def _extract_partial_analytical_doldexp_block(records: list[dict]) -> str:
    candidates = [
        record["snippet"]
        for record in records
        if 'If [PointGroup[[1]] == "1"' in record["snippet"] and "progress=65;" in record["snippet"]
    ]
    if not candidates:
        raise ValueError("Could not find SHAARP.si Partial Analytical Expressions doldExp block.")
    snippet = max(candidates, key=len)
    start = snippet.index('If [PointGroup[[1]] == "1"')
    end = snippet.index("progress=65;", start)
    return snippet[start:end]


def _parse_patterns(block: str) -> list[dict]:
    patterns = []
    search_at = 0
    while True:
        assign_at = block.find("doldExp =", search_at)
        if assign_at < 0:
            break
        if_at = block.rfind("If [", 0, assign_at)
        expr_start = assign_at + len("doldExp =")
        expr_end = block.find("];", expr_start)
        if if_at < 0 or expr_end < 0:
            raise ValueError("Malformed SHAARP.si doldExp branch near offset {assign_at}.")

        condition = block[if_at + len("If [") : assign_at].strip()
        if condition.endswith(","):
            condition = condition[:-1].strip()
        labels = re.findall(r'PointGroup(?:\[\[1\]\])?\s*==\s*"([^"]+)"', condition)
        if not labels:
            raise ValueError(f"No point-group labels found in condition: {condition}")

        matrix_source = block[expr_start:expr_end].strip()
        patterns.append(
            {
                "condition_source": condition,
                "point_group_labels": labels,
                "matrix_tokens": _matrix_tokens(matrix_source),
            }
        )
        search_at = expr_end + 2
    return patterns


def _matrix_tokens(matrix_source: str) -> list[list[str]]:
    expr = re.sub(r"Subscript\[d,\s*(\d+)\]", r"d\1", matrix_source)
    expr = re.sub(r"\s+", "", expr)
    if not (expr.startswith("{{") and expr.endswith("}}")):
        raise ValueError(f"Expected a 3x6 matrix expression, got: {matrix_source[:80]}")
    rows = expr[2:-2].split("},{")
    tokens = [row.split(",") for row in rows]
    if len(tokens) != 3 or any(len(row) != 6 for row in tokens):
        raise ValueError(f"Expected a 3x6 token matrix, got: {tokens}")
    return tokens


if __name__ == "__main__":
    main()
