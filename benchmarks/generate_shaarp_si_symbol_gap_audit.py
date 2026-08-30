from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from benchmarks.generate_shaarp_si_stage_symbol_sources import (
        LEADING_SYMBOL_RE,
        REQUESTED_SYMBOLS,
        _has_set_assignment,
    )
except ModuleNotFoundError:
    from generate_shaarp_si_stage_symbol_sources import REQUESTED_SYMBOLS, _has_set_assignment, LEADING_SYMBOL_RE


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
LINE_AUDIT_PATH = REF_DIR / "shaarp_si_candidate_region_line_audit.json"
SYMBOL_SOURCE_PATH = REF_DIR / "shaarp_si_stage_symbol_sources_v1.json"
OUTPUT_PATH = REF_DIR / "shaarp_si_symbol_gap_audit_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact requested SHAARP.si symbol gaps against line-audit text.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si symbol gap audit to {args.output}")


def build_payload() -> dict:
    line_audit = json.loads(LINE_AUDIT_PATH.read_text(encoding="utf-8"))
    symbol_sources = json.loads(SYMBOL_SOURCE_PATH.read_text(encoding="utf-8"))
    missing = symbol_sources["missing_requested_symbols"]
    numerical = next(item for item in line_audit["records"] if item["region_id"] == "si_numerical_shg_compute_candidate")
    lines = _unique_lines(numerical)
    assignments_by_symbol = _assignments_by_symbol(lines)
    records = [_gap_record(symbol, lines, assignments_by_symbol) for symbol in missing]
    return {
        "schema_version": 1,
        "source": "SHAARP.si exact requested-symbol gap audit from normalized PlainText line audit",
        "status": "gap_audit_not_validation_evidence",
        "line_audit": "benchmarks/mathematica_reference/shaarp_si_candidate_region_line_audit.json",
        "symbol_source_map": "benchmarks/mathematica_reference/shaarp_si_stage_symbol_sources_v1.json",
        "requested_symbols": REQUESTED_SYMBOLS,
        "missing_from_refined_symbol_sources": missing,
        "records": records,
        "notes": [
            "Exact assignment hits use the same assignment detector as the refined symbol-source map.",
            "Substring hits are diagnostic only and must not be treated as exact symbol matches.",
            "This audit resolves naming/routing gaps; it does not validate Mathematica values.",
        ],
    }


def _unique_lines(region: dict) -> list[dict]:
    records = (
        region["first_lines"]
        + region.get("target_window_lines", [])
        + region["interesting_lines"]
        + region["last_lines"]
    )
    by_index: dict[int, dict] = {}
    for record in records:
        by_index[record["line_index"]] = record
    return [by_index[key] for key in sorted(by_index)]


def _assignments_by_symbol(lines: list[dict]) -> dict[str, list[dict]]:
    assignments: dict[str, list[dict]] = {}
    for line in lines:
        text = line["text"]
        match = LEADING_SYMBOL_RE.match(text)
        if not match or not _has_set_assignment(text):
            continue
        symbol = match.group(1)
        assignments.setdefault(symbol, []).append(_line_summary(line))
    return assignments


def _gap_record(symbol: str, lines: list[dict], assignments_by_symbol: dict[str, list[dict]]) -> dict:
    exact_assignments = assignments_by_symbol.get(symbol, [])
    substring_hits = [_line_summary(line) for line in lines if symbol in line["text"]]
    assigned_symbols_containing_name = sorted(name for name in assignments_by_symbol if symbol in name and name != symbol)
    return {
        "symbol": symbol,
        "exact_assignment_count": len(exact_assignments),
        "exact_assignments": exact_assignments[:20],
        "substring_hit_count": len(substring_hits),
        "substring_hits_preview": substring_hits[:20],
        "assigned_symbols_containing_name": assigned_symbols_containing_name,
        "classification": _classification(symbol, exact_assignments, substring_hits, assigned_symbols_containing_name),
    }


def _classification(
    symbol: str,
    exact_assignments: list[dict],
    substring_hits: list[dict],
    assigned_symbols_containing_name: list[str],
) -> str:
    if exact_assignments:
        return "exact_assignment_outside_current_refined_snippets"
    if symbol in {"P2e", "Peo"} and assigned_symbols_containing_name:
        return "substring_only_of_active_pnl_symbols"
    if symbol == "I2":
        return "represented_by_specific_intensity_functions_not_exact_symbol"
    if symbol == "Fresnel" and substring_hits:
        return "sweep_table_symbol_outside_current_refined_snippets"
    if symbol in {"ThetaExt", "ThetaOrd"} and substring_hits:
        return "angle_symbol_referenced_outside_current_refined_snippets"
    return "unresolved_no_exact_assignment_in_line_audit"


def _line_summary(line: dict) -> dict:
    return {
        "line_index": line["line_index"],
        "offset": line["offset"],
        "text": line["text"],
    }


if __name__ == "__main__":
    main()
