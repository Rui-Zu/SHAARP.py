from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
SNIPPET_PATH = REF_DIR / "shaarp_si_refined_region_texts_v1.json"
PARSE_PATH = REF_DIR / "shaarp_si_refined_region_parse.json"
OUTPUT_PATH = REF_DIR / "shaarp_si_stage_symbol_sources_v1.json"

REQUESTED_SYMBOLS = [
    "ThetaExt",
    "ThetaOrd",
    "nwNumExt",
    "nwNumOrd",
    "EwrootsNum",
    "E2wroots",
    "ERpNum",
    "ERsNum",
    "ER2w",
    "HR2w",
    "P2e",
    "P2o",
    "Peo",
    "P2eo",
    "Peoo",
    "Croots",
    "ETweo",
    "ETwo",
    "Vktweo",
    "Vktwo",
    "RSignalCrystal",
    "I2",
    "Fresnel",
]

LEADING_SYMBOL_RE = re.compile(r"^\s*[\{(]*\s*([A-Za-z][A-Za-z0-9]*)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.si refined-region symbol source map.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si stage symbol sources to {args.output}")


def build_payload() -> dict:
    snippets = json.loads(SNIPPET_PATH.read_text(encoding="utf-8"))
    parse = json.loads(PARSE_PATH.read_text(encoding="utf-8"))
    parse_by_id = {record["region_id"]: record for record in parse["records"]}
    regions = [_region_sources(record, parse_by_id.get(record["region_id"], {})) for record in snippets["records"]]
    symbol_sources: dict[str, list[dict]] = defaultdict(list)
    for region in regions:
        for assignment in region["assignments"]:
            if assignment["symbol"] in REQUESTED_SYMBOLS:
                symbol_sources[assignment["symbol"]].append(
                    {
                        "region_id": region["region_id"],
                        "line_index": assignment["line_index"],
                        "line_text": assignment["line_text"],
                        "region_active_status": region["active_status"],
                    }
                )
    missing_requested = [symbol for symbol in REQUESTED_SYMBOLS if symbol not in symbol_sources]
    return {
        "schema_version": 1,
        "source": "SHAARP.si refined-region assignment map generated from audited snippets",
        "status": "symbol_source_map_not_validation_evidence",
        "snippet_file": "benchmarks/mathematica_reference/shaarp_si_refined_region_texts_v1.json",
        "parse_file": "benchmarks/mathematica_reference/shaarp_si_refined_region_parse.json",
        "requested_symbols": REQUESTED_SYMBOLS,
        "missing_requested_symbols": missing_requested,
        "regions": regions,
        "symbol_sources": dict(sorted(symbol_sources.items())),
        "notes": [
            "Assignments are detected mechanically from snippet lines using simple Mathematica Set syntax.",
            "This map identifies exporter source regions; it does not validate numerical values.",
            "Commented regions are preserved but marked inactive so their assignments are not mistaken for active SHAARP.si behavior.",
        ],
    }


def _region_sources(record: dict, parse_record: dict) -> dict:
    assignments = []
    for offset, line in enumerate(record["text"].splitlines()):
        match = LEADING_SYMBOL_RE.match(line)
        if not match or not _has_set_assignment(line):
            continue
        assignments.append(
            {
                "symbol": match.group(1),
                "line_index": record["line_start"] + offset,
                "line_text": line.strip(),
            }
        )
    active_status = "commented_reference_only" if record["region_id"].startswith("commented_") else "active_or_setup_code"
    return {
        "region_id": record["region_id"],
        "line_start": record["line_start"],
        "line_end": record["line_end"],
        "parse_status": parse_record.get("parse_status", "unknown"),
        "active_status": active_status,
        "assignment_count": len(assignments),
        "assignments": assignments,
    }


def _has_set_assignment(line: str) -> bool:
    code = _strip_quoted_strings(line)
    for index, char in enumerate(code):
        if char != "=":
            continue
        previous_is_equals = index > 0 and code[index - 1] == "="
        next_is_equals = index + 1 < len(code) and code[index + 1] == "="
        if not previous_is_equals and not next_is_equals:
            return True
    return False


def _strip_quoted_strings(line: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for char in line:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            result.append(" ")
            continue
        if char == '"':
            in_string = True
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


if __name__ == "__main__":
    main()
