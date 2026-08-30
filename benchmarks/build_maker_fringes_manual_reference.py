from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.convert_wolfram_copied_output import parse_wolfram_real_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a strict Maker fringes Mathematica reference JSON from GUI-copied listMFpara/listMFperp text."
    )
    parser.add_argument("--list-mf-para", type=Path, required=True, help="Text file copied from SHAARP.ml listMFpara.")
    parser.add_argument("--list-mf-perp", type=Path, required=True, help="Text file copied from SHAARP.ml listMFperp.")
    parser.add_argument("--output", type=Path, required=True, help="Reference JSON file to create or update.")
    parser.add_argument("--case-id", required=True, help="Case id matching maker_fringes_benchmarks_v1.json.")
    parser.add_argument(
        "--source-note",
        default="Mathematica SHAARP.ml GUI copied Maker fringes output",
        help="Top-level source metadata. Must contain Mathematica or Wolfram for strict comparison.",
    )
    args = parser.parse_args()

    list_mf_para = _validate_pair_list(parse_wolfram_real_list(args.list_mf_para.read_text(encoding="utf-8")), "listMFpara")
    list_mf_perp = _validate_pair_list(parse_wolfram_real_list(args.list_mf_perp.read_text(encoding="utf-8")), "listMFperp")
    mf_list = _make_mf_list(list_mf_para, list_mf_perp)

    payload = _load_or_create(args.output, args.source_note)
    _upsert_case(
        payload,
        args.case_id,
        {
            "MFList": mf_list,
            "listMFpara": list_mf_para,
            "listMFperp": list_mf_perp,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Updated {args.output} with Maker fringes reference for {args.case_id}")


def _validate_pair_list(values: Any, label: str) -> list[list[float]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list of {{theta, value}} rows.")
    rows = []
    for idx, row in enumerate(values):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{label}[{idx}] must contain exactly two values: {{theta, value}}.")
        rows.append([float(row[0]), float(row[1])])
    return rows


def _make_mf_list(list_mf_para: list[list[float]], list_mf_perp: list[list[float]]) -> list[list[float]]:
    if len(list_mf_para) != len(list_mf_perp):
        raise ValueError("listMFpara and listMFperp must have the same row count.")
    mf_list = []
    for idx, (para, perp) in enumerate(zip(list_mf_para, list_mf_perp)):
        if abs(para[0] - perp[0]) > 1e-10:
            raise ValueError(f"Angle mismatch at row {idx}: listMFpara has {para[0]}, listMFperp has {perp[0]}.")
        mf_list.append([para[0], para[1], perp[1]])
    return mf_list


def _load_or_create(path: Path, source_note: str) -> dict:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = str(payload.get("source", ""))
        if "mathematica" not in source.lower() and "wolfram" not in source.lower():
            raise ValueError("Existing payload source must contain Mathematica or Wolfram.")
        return payload
    return {
        "source": source_note,
        "status": "mathematica_reference_exported_manual_gui_copy",
        "cases": [],
    }


def _upsert_case(payload: dict, case_id: str, outputs: dict) -> None:
    cases = payload.setdefault("cases", [])
    for case in cases:
        if case.get("id") == case_id:
            case["outputs"] = outputs
            return
    cases.append({"id": case_id, "outputs": outputs})


if __name__ == "__main__":
    main()
