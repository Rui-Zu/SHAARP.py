from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
POSITIONS_PATH = REF_DIR / "shaarp_si_frontend_text_positions.json"
SECTIONS_PATH = REF_DIR / "shaarp_si_frontend_sections.json"
OUTPUT_PATH = REF_DIR / "shaarp_si_extraction_plan_v1.json"


def main() -> None:
    plan = build_plan()
    OUTPUT_PATH.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si extraction plan to {OUTPUT_PATH}")


def build_plan() -> dict:
    positions_payload = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
    sections_payload = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    plain_positions = _positions_for_format(positions_payload, "PlainText")
    input_positions = _positions_for_format(positions_payload, "InputText")
    summaries = {item["format"]: item for item in sections_payload["summaries"]}

    plain_regions = _candidate_regions(plain_positions)
    input_regions = _candidate_regions(input_positions)
    return {
        "schema_version": 1,
        "source": "SHAARP.si front-end text extraction plan from Mathematica-generated probe artifacts",
        "status": "candidate_extraction_plan_not_validation_evidence",
        "notebook_path": positions_payload["notebook_path"],
        "cell_index": 4,
        "inputCellCount": positions_payload["inputCellCount"],
        "rule": (
            "Ranges are candidate byte spans in front-end converted text. "
            "They guide targeted exporter work and must not be treated as Mathematica-vs-Python agreement."
        ),
        "formats": {
            "PlainText": {
                "byte_count": summaries["PlainText"]["byte_count"],
                "pattern_counts": summaries["PlainText"]["pattern_counts"],
                "candidate_regions": plain_regions,
            },
            "InputText": {
                "byte_count": summaries["InputText"]["byte_count"],
                "pattern_counts": summaries["InputText"]["pattern_counts"],
                "candidate_regions": input_regions,
            },
        },
        "next_actions": [
            "Extract and batch-parse the PlainText numerical core candidate before GUI output code.",
            "Identify minimal variable assignments needed to run the SI numerical core.",
            "Export stage records for omega waves, nonlinear polarization, inhomogeneous fields, two-omega waves, Fresnel coefficients, analyzer amplitudes, and final intensities.",
            "Extract full analytical export block separately from numerical GUI rendering.",
            "Compare exported Mathematica values against Python stage-by-stage before claiming agreement.",
        ],
    }


def _positions_for_format(payload: dict, format_name: str) -> dict[str, list[int]]:
    record = next(item for item in payload["records"] if item["format"] == format_name)
    result: dict[str, list[int]] = {}
    for item in record["patterns"]:
        result[item["pattern"]] = [pos[0] for pos in item["positions"]]
    return result


def _candidate_regions(positions: dict[str, list[int]]) -> list[dict]:
    first_shg = _first(positions, "SHG Simulations")
    first_partial = _first(positions, "Partial Analytical")
    first_full = _first(positions, "FullAnalytical")
    first_full_expression = _first(positions, "Full Analytical")
    first_export_full = _first(positions, 'Export["Full Analytical Expressions.nb"')
    second_manipulate = positions.get("Manipulate", [None])[-1]

    regions = [
        _region(
            "initialization_and_input_setup_candidate",
            1,
            first_shg,
            "Before the first SHG Simulations marker; contains global setup, controls, and physical variable preparation.",
        ),
        _region(
            "si_numerical_shg_compute_candidate",
            first_shg,
            first_partial,
            "Between first SHG Simulations and first Partial Analytical marker; likely numerical reflected SHG compute path.",
        ),
        _region(
            "si_partial_analytical_compute_candidate",
            first_partial,
            second_manipulate,
            "Between first Partial Analytical marker and later Manipulate/output region; likely partial analytical calculation body.",
        ),
        _region(
            "si_full_analytical_export_candidate",
            first_export_full,
            None,
            "Starts at the Full Analytical Expressions notebook export block; use with a bounded end once the export expression is bracket-matched.",
        ),
    ]
    if first_full is not None:
        regions.append(
            _region(
                "full_analytical_flag_setup_candidate",
                first_full,
                first_shg,
                "Early FullAnalyticalFlag/setup area, not the full analytical expression export body.",
            )
        )
    if first_full_expression is not None:
        regions.append(
            _region(
                "first_full_analytical_marker_candidate",
                first_full_expression,
                first_export_full,
                "First Full Analytical Expressions functionality marker before the explicit export block.",
            )
        )
    return regions


def _first(positions: dict[str, list[int]], pattern: str) -> int | None:
    values = positions.get(pattern, [])
    return values[0] if values else None


def _region(region_id: str, start: int | None, end: int | None, rationale: str) -> dict:
    return {
        "region_id": region_id,
        "start": start,
        "end": end,
        "byte_count": None if start is None or end is None else max(0, end - start + 1),
        "status": "candidate_not_validated",
        "rationale": rationale,
    }


if __name__ == "__main__":
    main()
