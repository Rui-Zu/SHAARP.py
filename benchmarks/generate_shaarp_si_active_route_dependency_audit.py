from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
SNIPPET_PATH = REF_DIR / "shaarp_si_refined_region_texts_v1.json"
CONTRACT_PATH = REF_DIR / "shaarp_si_stage_export_contract_v1.json"
OUTPUT_PATH = REF_DIR / "shaarp_si_active_route_dependency_audit_v1.json"


ASCII_SYMBOL_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")
ESCAPED_SYMBOL_RE = re.compile(r"\\\[[A-Za-z][A-Za-z0-9]*\](?:[A-Za-z0-9]*\\\[[A-Za-z][A-Za-z0-9]*\])*(?:[A-Za-z0-9]+)?")
ASSIGNMENT_RE = re.compile(r"[\{(;\s]*([A-Za-z][A-Za-z0-9]*)(?:\[[^\]]*\])?\s*=(?!=)")

BUILTINS_AND_LOCAL = {
    "Abs",
    "All",
    "Chop",
    "Clear",
    "ComplexExpand",
    "Conjugate",
    "ConstantArray",
    "Cos",
    "Cross",
    "Curl",
    "DebugFlag",
    "Degree",
    "Det",
    "Eigensystem",
    "Eigenvectors",
    "Element",
    "Exp",
    "Expand",
    "FindRoot",
    "Flatten",
    "For",
    "Functionality",
    "FullSimplify",
    "I",
    "If",
    "IdentityMatrix",
    "Inverse",
    "MatrixForm",
    "MatrixPower",
    "MatrixRank",
    "Max",
    "MaxValue",
    "Module",
    "NSolve",
    "Pi",
    "Print",
    "Quiet",
    "Re",
    "Reals",
    "Round",
    "SetPrecision",
    "Simplify",
    "Sin",
    "Sqrt",
    "Table",
    "Transpose",
    "TrigReduce",
    "Unevaluated",
    "ValueQ",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "progress",
    "x",
    "y",
    "z",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dependencies for SHAARP.si active export route.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si active-route dependency audit to {args.output}")


def build_payload() -> dict:
    snippets = _load_json(SNIPPET_PATH)
    contract = _load_json(CONTRACT_PATH)
    snippet_by_id = {record["region_id"]: record for record in snippets["records"]}
    active = _route_audit(contract["active_numeric_region_order"], snippet_by_id)
    fresnel = _route_audit(contract["fresnel_sweep_region_order"], snippet_by_id)
    return {
        "schema_version": 1,
        "source": "SHAARP.si active-route dependency audit from refined snippet text",
        "status": "dependency_audit_not_validation_evidence",
        "contract_file": "benchmarks/mathematica_reference/shaarp_si_stage_export_contract_v1.json",
        "snippet_file": "benchmarks/mathematica_reference/shaarp_si_refined_region_texts_v1.json",
        "active_numeric_route": active,
        "fresnel_sweep_route": fresnel,
        "notes": [
            "This is a lexical dependency audit for exporter construction, not a Mathematica evaluation.",
            "External symbols include user inputs, constants, GUI controls, and upstream notebook assignments that are outside the refined route.",
            "Escaped Mathematica symbols are preserved in their front-end text spelling.",
        ],
    }


def _route_audit(region_order: list[str], snippet_by_id: dict[str, dict]) -> dict:
    assigned_so_far: set[str] = set()
    region_records = []
    first_external_use: dict[str, dict] = {}
    all_assigned: set[str] = set()
    for region_id in region_order:
        record = snippet_by_id[region_id]
        region = _region_record(record, assigned_so_far)
        region_records.append(region)
        for symbol in region["unresolved_before_region"]:
            first_external_use.setdefault(
                symbol,
                {
                    "first_region_id": region_id,
                    "first_line_index": region["line_start"],
                },
            )
        assigned_so_far.update(region["assigned_symbols"])
        all_assigned.update(region["assigned_symbols"])
    return {
        "region_order": region_order,
        "assigned_symbols": sorted(all_assigned),
        "required_external_symbols": sorted(first_external_use),
        "first_external_use": dict(sorted(first_external_use.items())),
        "regions": region_records,
    }


def _region_record(record: dict, assigned_so_far: set[str]) -> dict:
    assigned = _assigned_symbols(record["text"])
    referenced = _referenced_symbols(record["text"])
    unresolved = sorted(symbol for symbol in referenced if symbol not in assigned_so_far and symbol not in assigned)
    return {
        "region_id": record["region_id"],
        "line_start": record["line_start"],
        "line_end": record["line_end"],
        "assigned_symbols": sorted(assigned),
        "referenced_symbols": sorted(referenced),
        "unresolved_before_region": unresolved,
    }


def _assigned_symbols(text: str) -> set[str]:
    assigned = set()
    for line in text.splitlines():
        stripped = _strip_strings_and_comments(line)
        for match in ASSIGNMENT_RE.finditer(stripped):
            assigned.add(match.group(1))
    return assigned


def _referenced_symbols(text: str) -> set[str]:
    stripped = _strip_strings_and_comments(text)
    escaped_symbols = set(ESCAPED_SYMBOL_RE.findall(stripped))
    stripped_without_escaped = ESCAPED_SYMBOL_RE.sub(" ", stripped)
    symbols = set(ASCII_SYMBOL_RE.findall(stripped_without_escaped))
    symbols.update(escaped_symbols)
    return {symbol for symbol in symbols if symbol not in BUILTINS_AND_LOCAL}


def _strip_strings_and_comments(text: str) -> str:
    chars = []
    in_string = False
    in_comment = False
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        char = text[index]
        if in_comment:
            if pair == "*)":
                in_comment = False
                chars.extend("  ")
                index += 2
            else:
                chars.append(" ")
                index += 1
            continue
        if in_string:
            if char == '"':
                in_string = False
            chars.append(" ")
            index += 1
            continue
        if pair == "(*":
            in_comment = True
            chars.extend("  ")
            index += 2
        elif char == '"':
            in_string = True
            chars.append(" ")
            index += 1
        else:
            chars.append(char)
            index += 1
    return "".join(chars)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
    "Or",
    "\\[Or]",
