from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
LINE_AUDIT_PATH = REF_DIR / "shaarp_si_candidate_region_line_audit.json"
MAP_PATH = REF_DIR / "shaarp_si_refined_extraction_map_v1.json"
OUTPUT_PATH = REF_DIR / "shaarp_si_refined_region_texts_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SHAARP.si refined region text snippets from audited lines.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si refined region texts to {args.output}")


def build_payload() -> dict:
    audit = json.loads(LINE_AUDIT_PATH.read_text(encoding="utf-8"))
    refined = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    numerical = next(item for item in audit["records"] if item["region_id"] == "si_numerical_shg_compute_candidate")
    lines = _line_index(numerical)
    snippets = [_snippet_record(lines, region) for region in refined["regions"]]
    return {
        "schema_version": 1,
        "source": "SHAARP.si refined region text snippets reconstructed from normalized PlainText line audit",
        "status": "snippet_export_not_validation_evidence",
        "coordinate_system": refined["coordinate_system"],
        "line_audit": "benchmarks/mathematica_reference/shaarp_si_candidate_region_line_audit.json",
        "refined_map": "benchmarks/mathematica_reference/shaarp_si_refined_extraction_map_v1.json",
        "records": snippets,
        "notes": [
            "Snippets are reconstructed from line-audit text using LF newlines.",
            "This avoids reopening the Mathematica front end during small parse probes.",
            "These snippets are routing evidence for exporter construction, not Mathematica value validation.",
        ],
    }


def _line_index(region: dict) -> dict[int, dict]:
    records = (
        region["first_lines"]
        + region.get("target_window_lines", [])
        + region["interesting_lines"]
        + region["last_lines"]
    )
    indexed: dict[int, dict] = {}
    for record in records:
        indexed[record["line_index"]] = record
    return indexed


def _snippet_record(lines: dict[int, dict], region: dict) -> dict:
    missing = [line_index for line_index in range(region["line_start"], region["line_end"] + 1) if line_index not in lines]
    text_lines = [] if missing else [lines[line_index]["text"] for line_index in range(region["line_start"], region["line_end"] + 1)]
    text = "\n".join(text_lines)
    return {
        "region_id": region["region_id"],
        "line_start": region["line_start"],
        "line_end": region["line_end"],
        "line_count": region["line_end"] - region["line_start"] + 1,
        "status": "missing_lines" if missing else "snippet_ready",
        "missing_lines": missing,
        "text": text,
        "byte_count": len(text.encode("utf-8")),
    }


if __name__ == "__main__":
    main()
