from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import ComparisonResult, cases_by_id, numeric_array
from benchmarks.discrepancy_report import write_draft_reports_from_comparison_results


SUITE_ID = "maker_fringes_workflow"
REQUIRED_MATHEMATICA_OUTPUTS = ("MFList", "listMFpara", "listMFperp")
SUMMARY_PATH = Path(__file__).resolve().parent / "mathematica_reference" / "maker_fringes_comparison_summary_v1.json"
DIAGNOSTIC_CASE_IDS = {"maker_fringes_phase_matching_like", "fine_maker_phase_matching_like_local"}
DIRECT_OUTPUT_KEY_MAP = {
    "listMFpara": "list_mf_para",
    "listMFperp": "list_mf_perp",
    "transmitted_two_omega_jones_sp": "transmitted_2omega_jones_sp",
    "parallel_analyzer_amplitude": "parallel_amplitude",
    "perpendicular_analyzer_amplitude": "perpendicular_amplitude",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exported Mathematica SHAARP.ml Maker fringes references against Python fixtures."
    )
    parser.add_argument("--mathematica", type=Path, required=True, help="Filled Mathematica reference JSON.")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(__file__).resolve().parent / "maker_fringes_benchmarks_v1.json",
        help="Python Maker fringes benchmark JSON.",
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="Allow comparison against payloads without explicit Mathematica/Wolfram source metadata. Use only for tests.",
    )
    parser.add_argument(
        "--write-discrepancy-reports",
        type=Path,
        default=None,
        help="Optional directory where failed comparisons are written as draft discrepancy reports.",
    )
    parser.add_argument("--source-notebook-or-package", default="Github/SHAARP.ml/SHAARP.ml.nb")
    parser.add_argument("--mathematica-version-or-runtime", default="unknown")
    parser.add_argument("--shaarp-file-path-version-or-date", default="Github/SHAARP.ml/SHAARP.ml.nb")
    parser.add_argument("--python-git-or-worktree-state", default="unknown")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Write a machine-readable agreement summary that classifies known diagnostic cases.",
    )
    args = parser.parse_args()

    if args.summary_out is not None:
        mathematica = json.loads(args.mathematica.read_text(encoding="utf-8"))
        python = json.loads(args.python.read_text(encoding="utf-8"))
        summary = write_maker_reference_agreement_summary(
            args.summary_out,
            mathematica,
            python,
            atol=args.atol,
            rtol=args.rtol,
            strict_reference=not args.allow_unverified_source,
        )
        print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))
        return

    results = compare_maker_reference_files(
        args.mathematica,
        args.python,
        atol=args.atol,
        rtol=args.rtol,
        strict_reference=not args.allow_unverified_source,
    )
    failed = [result for result in results if not result.passed]
    if failed and args.write_discrepancy_reports is not None:
        paths = write_draft_reports_from_comparison_results(
            results,
            args.write_discrepancy_reports,
            source_notebook_or_package=args.source_notebook_or_package,
            mathematica_version_or_runtime=args.mathematica_version_or_runtime,
            shaarp_file_path_version_or_date=args.shaarp_file_path_version_or_date,
            python_git_or_worktree_state=args.python_git_or_worktree_state,
            first_failing_stage="Maker fringes transmitted SHG angle scan",
            tolerance={"atol": args.atol, "rtol": args.rtol},
            next_action="Inspect MFList/listMFpara/listMFperp normalization, analyzer convention, and transmitted wave summing.",
        )
        for path in paths:
            print(f"WROTE discrepancy draft: {path}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.case_id} {result.key}: "
            f"shape={result.shape} max_abs={result.max_abs_error:.3e} max_rel={result.max_rel_error:.3e}"
        )
    if failed:
        raise SystemExit(f"{len(failed)} Maker fringes reference comparisons failed.")
    print(f"All {len(results)} Maker fringes reference comparisons passed.")


def compare_maker_reference_files(
    mathematica_path: Path,
    python_path: Path,
    *,
    atol: float,
    rtol: float,
    strict_reference: bool = True,
) -> list[ComparisonResult]:
    mathematica = json.loads(mathematica_path.read_text(encoding="utf-8"))
    python = json.loads(python_path.read_text(encoding="utf-8"))
    return compare_maker_reference_payloads(
        mathematica,
        python,
        atol=atol,
        rtol=rtol,
        strict_reference=strict_reference,
    )


def compare_maker_reference_payloads(
    mathematica: dict[str, Any],
    python: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    strict_reference: bool = True,
) -> list[ComparisonResult]:
    if strict_reference:
        validate_maker_reference_payload(mathematica)

    mma_cases = cases_by_id(mathematica, suite_id=SUITE_ID, output_key="mathematica_outputs")
    py_cases = cases_by_id(python)
    missing = sorted(set(mma_cases) - set(py_cases))
    if missing:
        raise ValueError(f"Python benchmark is missing case ids: {missing}")

    results = []
    for case_id, mma_case in sorted(mma_cases.items()):
        py_case = py_cases[case_id]
        mma_outputs = _outputs(mma_case)
        py_outputs = _outputs(py_case)

        results.append(
            _compare_numeric(case_id, "MFList[[All,{1,2}]]", _mf_columns(mma_outputs["MFList"], (0, 1)), py_outputs["list_mf_para"], atol=atol, rtol=rtol)
        )
        results.append(
            _compare_numeric(case_id, "MFList[[All,{1,3}]]", _mf_columns(mma_outputs["MFList"], (0, 2)), py_outputs["list_mf_perp"], atol=atol, rtol=rtol)
        )
        for mma_key, py_key in DIRECT_OUTPUT_KEY_MAP.items():
            if mma_key in mma_outputs and py_key in py_outputs:
                results.append(_compare_numeric(case_id, mma_key, mma_outputs[mma_key], py_outputs[py_key], atol=atol, rtol=rtol))
    return results


def build_maker_reference_agreement_summary(
    mathematica: dict[str, Any],
    python: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    strict_reference: bool = True,
    diagnostic_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    diagnostic_case_ids = diagnostic_case_ids or DIAGNOSTIC_CASE_IDS
    results = compare_maker_reference_payloads(
        mathematica,
        python,
        atol=atol,
        rtol=rtol,
        strict_reference=strict_reference,
    )
    nonsingular = [result for result in results if result.case_id not in diagnostic_case_ids]
    diagnostic = [result for result in results if result.case_id in diagnostic_case_ids]
    nonsingular_failed = [result for result in nonsingular if not result.passed]
    diagnostic_failed = [result for result in diagnostic if not result.passed]
    nonsingular_case_ids = sorted({result.case_id for result in nonsingular})
    present_diagnostic_case_ids = sorted({result.case_id for result in diagnostic})
    status = (
        "nonsingular_maker_outputs_match_with_phase_matching_diagnostic"
        if not nonsingular_failed and diagnostic_failed
        else "maker_outputs_have_unresolved_nonsingular_mismatch"
        if nonsingular_failed
        else "maker_outputs_match_all_compared_cases"
    )
    return {
        "source": mathematica.get("source"),
        "status": status,
        "full_shaarp_ml_agreement_claimed": bool(results) and all(result.passed for result in results),
        "atol": atol,
        "rtol": rtol,
        "selected_transmitted_wave_policy": "shaarp_ml_selected",
        "nonsingular_case_count": len(nonsingular_case_ids),
        "nonsingular_case_ids": nonsingular_case_ids,
        "diagnostic_case_ids": present_diagnostic_case_ids,
        "nonsingular_pass_count": sum(1 for result in nonsingular if result.passed),
        "nonsingular_fail_count": len(nonsingular_failed),
        "diagnostic_pass_count": sum(1 for result in diagnostic if result.passed),
        "diagnostic_fail_count": len(diagnostic_failed),
        "nonsingular_outputs": _aggregate_results_by_key(nonsingular),
        "diagnostic_outputs": [_result_to_dict(result) for result in diagnostic],
        "notes": [
            "Nonsingular cases compare Mathematica SHAARP.ml MFList/listMFpara/listMFperp and analyzer amplitudes value-by-value.",
            "The phase-matching-like case is retained as a diagnostic because solveInhom is singular or nearly singular.",
            "This summary does not relax the strict comparator; it classifies the known diagnostic case separately.",
        ],
    }


def write_maker_reference_agreement_summary(
    path: Path,
    mathematica: dict[str, Any],
    python: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    strict_reference: bool = True,
) -> dict[str, Any]:
    summary = build_maker_reference_agreement_summary(
        mathematica,
        python,
        atol=atol,
        rtol=rtol,
        strict_reference=strict_reference,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def validate_maker_reference_payload(payload: dict[str, Any]) -> None:
    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("Strict Maker comparison requires source metadata containing Mathematica or Wolfram.")
    if "python" in source:
        raise ValueError("Strict Maker comparison refuses Python-generated payloads as Mathematica references.")

    status = str(payload.get("status", "")).lower()
    blocked_markers = (
        "fillable",
        "no_values",
        "not_yet",
        "requested",
        "pending",
        "not_mathematica_validation",
        "python_regression",
    )
    if any(marker in status for marker in blocked_markers):
        raise ValueError(f"Strict Maker comparison refuses unvalidated reference status: {payload.get('status')!r}.")

    cases = cases_by_id(payload, suite_id=SUITE_ID, output_key="mathematica_outputs")
    if not cases:
        raise ValueError("Strict Maker comparison requires non-empty reference cases.")
    for case in cases.values():
        case_id = case.get("id", "<missing>")
        outputs = _outputs(case)
        missing = sorted(set(REQUIRED_MATHEMATICA_OUTPUTS) - set(outputs))
        if missing:
            raise ValueError(f"Case {case_id} is missing required Maker Mathematica outputs: {missing}")
        for key in REQUIRED_MATHEMATICA_OUTPUTS:
            _reject_none(outputs[key], f"case {case_id}.outputs.{key}")
        mf = numeric_array(outputs["MFList"])
        if mf.ndim != 2 or mf.shape[1] < 3:
            raise ValueError(f"Case {case_id}.outputs.MFList must be a 2D array with at least three columns.")


def _reject_none(value: Any, path: str) -> None:
    if value is None:
        raise ValueError(f"Unfilled Mathematica output placeholder at {path}.")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_none(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _reject_none(item, f"{path}[{idx}]")


def _outputs(case: dict[str, Any]) -> dict[str, Any]:
    outputs = case.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"Case {case.get('id')} is missing outputs.")
    return outputs


def _mf_columns(value: Any, columns: tuple[int, int]) -> np.ndarray:
    arr = numeric_array(value)
    if arr.ndim != 2 or arr.shape[1] <= max(columns):
        raise ValueError("MFList must be a 2D array with enough columns for listMF extraction.")
    return arr[:, columns]


def _compare_numeric(
    case_id: str,
    key: str,
    mathematica_value: Any,
    python_value: Any,
    *,
    atol: float,
    rtol: float,
) -> ComparisonResult:
    mma = _as_numeric_array(mathematica_value)
    py = _as_numeric_array(python_value)
    if mma.shape != py.shape:
        return ComparisonResult(case_id, key, False, float("inf"), float("inf"), tuple(py.shape))
    delta = np.abs(py - mma)
    scale = np.maximum(np.abs(mma), np.finfo(float).tiny)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    return ComparisonResult(case_id, key, bool(np.allclose(py, mma, atol=atol, rtol=rtol)), max_abs, max_rel, tuple(py.shape))


def _aggregate_results_by_key(results: list[ComparisonResult]) -> list[dict[str, Any]]:
    by_key: dict[str, list[ComparisonResult]] = {}
    for result in results:
        by_key.setdefault(result.key, []).append(result)
    return [
        {
            "key": key,
            "comparison_status": "passed" if all(result.passed for result in key_results) else "failed",
            "case_count": len({result.case_id for result in key_results}),
            "max_abs_error": max(result.max_abs_error for result in key_results),
            "max_rel_error": max(result.max_rel_error for result in key_results),
        }
        for key, key_results in sorted(by_key.items())
    ]


def _result_to_dict(result: ComparisonResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "key": result.key,
        "comparison_status": "passed" if result.passed else "failed",
        "max_abs_error": result.max_abs_error,
        "max_rel_error": result.max_rel_error,
        "shape": list(result.shape),
    }


def _as_numeric_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=complex)
    return numeric_array(value)


if __name__ == "__main__":
    main()
