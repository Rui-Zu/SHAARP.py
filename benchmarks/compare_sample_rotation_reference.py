from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import ComparisonResult, cases_by_id, numeric_array


SUITE_ID = "sample_rotation_workflow"
DIAGNOSTIC_CASE_IDS = {"sample_rotation_phase_matching_like"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Mathematica SHAARP.ml SampleRotate values against Python.")
    parser.add_argument("--mathematica", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(__file__).resolve().parent / "sample_rotation_benchmarks_v1.json",
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    mathematica = json.loads(args.mathematica.read_text(encoding="utf-8"))
    python = json.loads(args.python.read_text(encoding="utf-8"))
    results = compare_sample_rotation_payloads(mathematica, python, atol=args.atol, rtol=args.rtol)
    if args.summary_out is not None:
        summary = build_sample_rotation_summary(results, mathematica, atol=args.atol, rtol=args.rtol)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))
        return
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.case_id} {result.key}: "
            f"shape={result.shape} max_abs={result.max_abs_error:.3e} max_rel={result.max_rel_error:.3e}"
        )
    failed = [result for result in results if not result.passed]
    if failed:
        raise SystemExit(f"{len(failed)} SampleRotate reference comparisons failed.")
    print(f"All {len(results)} SampleRotate reference comparisons passed.")


def compare_sample_rotation_payloads(
    mathematica: dict[str, Any],
    python: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    validate_sample_rotation_reference(mathematica)
    mma_cases = cases_by_id(mathematica, suite_id=SUITE_ID, output_key="mathematica_outputs")
    py_cases = cases_by_id(python)
    missing = sorted(set(mma_cases) - set(py_cases))
    if missing:
        raise ValueError(f"Python benchmark is missing SampleRotate cases: {missing}")
    results = []
    for case_id, mma_case in sorted(mma_cases.items()):
        mma = numeric_array(mma_case["outputs"]["sampleRotateList"])
        py = numeric_array(py_cases[case_id]["outputs"]["sample_rotate_list"])
        results.append(_compare_numeric(case_id, "sampleRotateList", mma, py, atol=atol, rtol=rtol))
    return results


def build_sample_rotation_summary(
    results: list[ComparisonResult],
    mathematica: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    nonsingular = [result for result in results if result.case_id not in DIAGNOSTIC_CASE_IDS]
    diagnostic = [result for result in results if result.case_id in DIAGNOSTIC_CASE_IDS]
    nonsingular_failed = [result for result in nonsingular if not result.passed]
    diagnostic_failed = [result for result in diagnostic if not result.passed]
    status = (
        "sample_rotation_outputs_match"
        if all(result.passed for result in results)
        else "sample_rotation_has_nonsingular_mismatch"
        if nonsingular_failed
        else "sample_rotation_nonsingular_match_with_phase_matching_diagnostic"
    )
    return {
        "source": mathematica.get("source"),
        "status": status,
        "full_sample_rotation_agreement_claimed": all(result.passed for result in results),
        "atol": atol,
        "rtol": rtol,
        "nonsingular_case_count": len(nonsingular),
        "nonsingular_fail_count": len(nonsingular_failed),
        "diagnostic_case_ids": sorted({result.case_id for result in diagnostic}),
        "diagnostic_fail_count": len(diagnostic_failed),
        "results": [_result_to_dict(result) for result in results],
    }


def validate_sample_rotation_reference(payload: dict[str, Any]) -> None:
    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("Strict SampleRotate comparison requires Mathematica or Wolfram source metadata.")
    cases = cases_by_id(payload, suite_id=SUITE_ID, output_key="mathematica_outputs")
    if not cases:
        raise ValueError("SampleRotate reference contains no cases.")
    for case_id, case in cases.items():
        outputs = case.get("outputs", {})
        if "sampleRotateList" not in outputs:
            raise ValueError(f"Case {case_id} is missing sampleRotateList.")
        arr = numeric_array(outputs["sampleRotateList"])
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError(f"Case {case_id}.sampleRotateList must have five columns.")


def _compare_numeric(case_id: str, key: str, mma: np.ndarray, py: np.ndarray, *, atol: float, rtol: float) -> ComparisonResult:
    mma = np.asarray(mma, dtype=complex)
    py = np.asarray(py, dtype=complex)
    if mma.shape != py.shape:
        return ComparisonResult(case_id, key, False, float("inf"), float("inf"), tuple(py.shape))
    delta = np.abs(py - mma)
    scale = np.maximum(np.abs(mma), np.finfo(float).tiny)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    return ComparisonResult(case_id, key, bool(np.allclose(py, mma, atol=atol, rtol=rtol)), max_abs, max_rel, tuple(py.shape))


def _result_to_dict(result: ComparisonResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "key": result.key,
        "comparison_status": "passed" if result.passed else "failed",
        "max_abs_error": result.max_abs_error,
        "max_rel_error": result.max_rel_error,
        "shape": list(result.shape),
    }


if __name__ == "__main__":
    main()
