from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import ComparisonResult, cases_by_id, numeric_array


DEFAULT_FINE = Path(__file__).resolve().parent / "maker_fringes_fine_sampling_output_v1.json"
DEFAULT_COARSE = Path(__file__).resolve().parent / "maker_fringes_benchmarks_v1.json"
OUTPUT_KEYS = ("list_mf_para", "list_mf_perp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare optional dense Maker fringes output against saved coarse Python Maker benchmarks at overlapping angles."
    )
    parser.add_argument("--fine", type=Path, default=DEFAULT_FINE, help="Fine-sampling output JSON.")
    parser.add_argument("--coarse", type=Path, default=DEFAULT_COARSE, help="Saved coarse Maker benchmark JSON.")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--angle-atol", type=float, default=1e-10)
    args = parser.parse_args()

    results = compare_fine_to_coarse_files(
        args.fine,
        args.coarse,
        atol=args.atol,
        rtol=args.rtol,
        angle_atol=args.angle_atol,
    )
    failed = [result for result in results if not result.passed]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.case_id} {result.key}: "
            f"shape={result.shape} max_abs={result.max_abs_error:.3e} max_rel={result.max_rel_error:.3e}"
        )
    if failed:
        raise SystemExit(f"{len(failed)} Maker fine-vs-coarse comparisons failed.")
    print(f"All {len(results)} Maker fine-vs-coarse comparisons passed.")


def compare_fine_to_coarse_files(
    fine_path: Path,
    coarse_path: Path,
    *,
    atol: float,
    rtol: float,
    angle_atol: float = 1e-10,
) -> list[ComparisonResult]:
    fine = json.loads(fine_path.read_text(encoding="utf-8"))
    coarse = json.loads(coarse_path.read_text(encoding="utf-8"))
    return compare_fine_to_coarse_payloads(fine, coarse, atol=atol, rtol=rtol, angle_atol=angle_atol)


def compare_fine_to_coarse_payloads(
    fine: dict[str, Any],
    coarse: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    angle_atol: float = 1e-10,
) -> list[ComparisonResult]:
    status = str(fine.get("status", "")).lower()
    source = str(fine.get("source", "")).lower()
    if status.startswith("mathematica") or "mathematica" in source or "wolfram" in source:
        raise ValueError("Fine-vs-coarse comparison is Python-only; use the Maker Mathematica comparator for Mathematica values.")
    fine_cases = cases_by_id(fine)
    coarse_cases = cases_by_id(coarse)
    results = []
    for fine_case_id, fine_case in sorted(fine_cases.items()):
        fine_sampling = fine_case.get("fine_sampling")
        if not isinstance(fine_sampling, dict) or "source_case_id" not in fine_sampling:
            raise ValueError(f"Fine case {fine_case_id} is missing fine_sampling.source_case_id.")
        source_case_id = fine_sampling["source_case_id"]
        if source_case_id not in coarse_cases:
            raise ValueError(f"Fine case {fine_case_id} references missing coarse source case {source_case_id}.")
        coarse_case = coarse_cases[source_case_id]
        for key in OUTPUT_KEYS:
            results.append(
                _compare_rows_at_coarse_angles(
                    fine_case_id,
                    key,
                    fine_case["outputs"][key],
                    coarse_case["outputs"][key],
                    atol=atol,
                    rtol=rtol,
                    angle_atol=angle_atol,
                )
            )
    return results


def _compare_rows_at_coarse_angles(
    case_id: str,
    key: str,
    fine_value: Any,
    coarse_value: Any,
    *,
    atol: float,
    rtol: float,
    angle_atol: float,
) -> ComparisonResult:
    fine = numeric_array(fine_value)
    coarse = numeric_array(coarse_value)
    if fine.ndim != 2 or fine.shape[1] != 2 or coarse.ndim != 2 or coarse.shape[1] != 2:
        raise ValueError(f"{key} must be two-column {{theta, value}} data.")

    selected = []
    for coarse_row in coarse:
        theta = coarse_row[0]
        matches = np.where(np.abs(fine[:, 0] - theta) <= angle_atol)[0]
        if len(matches) != 1:
            return ComparisonResult(case_id, key, False, float("inf"), float("inf"), tuple(fine.shape))
        selected.append(fine[matches[0]])
    selected_arr = np.asarray(selected, dtype=complex)
    delta = np.abs(selected_arr - coarse)
    scale = np.maximum(np.abs(coarse), np.finfo(float).tiny)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    passed = bool(np.allclose(selected_arr, coarse, atol=atol, rtol=rtol))
    return ComparisonResult(case_id, key, passed, max_abs, max_rel, tuple(selected_arr.shape))


if __name__ == "__main__":
    main()
