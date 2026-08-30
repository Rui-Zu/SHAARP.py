from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import (
    ComparisonResult,
    cases_by_id,
    numeric_array,
)
from benchmarks.discrepancy_report import write_draft_reports_from_comparison_results


SUITE_ID = "multilayer_boundary_system"
NUMERIC_OUTPUT_KEY_MAP = {
    "solveFresnelN_matrix": "matrix",
    "solveFresnelN_rhs": "rhs",
    "solveFresnelN_known_residual": "known_residual",
    "solveFresnelN_coefficients": "coefficients",
    "solveFresnelN_residual": "residual",
}
EXACT_OUTPUT_KEYS = (
    "equation_labels",
    "equation_metadata",
    "unknown_labels",
    "unknown_metadata",
)
WAVE_OUTPUT_KEYS = ("unknown_basis_waves", "known_waves")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exported Mathematica solveFresnelN boundary-system references against Python fixtures."
    )
    parser.add_argument("--mathematica", type=Path, required=True, help="Filled Mathematica reference JSON.")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(__file__).resolve().parent / "multilayer_boundary_system_benchmarks_v1.json",
        help="Python multilayer boundary-system benchmark JSON.",
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
    parser.add_argument("--source-notebook-or-package", default="Github/SHAARP.ml/setup.nb")
    parser.add_argument("--mathematica-version-or-runtime", default="unknown")
    parser.add_argument("--shaarp-file-path-version-or-date", default="Github/SHAARP.ml/setup.nb")
    parser.add_argument("--python-git-or-worktree-state", default="unknown")
    parser.add_argument("--first-failing-stage", default="2omega boundary matrix/RHS")
    args = parser.parse_args()

    results = compare_boundary_reference_files(
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
            first_failing_stage=args.first_failing_stage,
            tolerance={"atol": args.atol, "rtol": args.rtol},
            next_action="Inspect the exported solveFresnelN matrix/RHS and boundary wave records for this comparison key.",
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
        raise SystemExit(f"{len(failed)} multilayer boundary reference comparisons failed.")
    print(f"All {len(results)} multilayer boundary reference comparisons passed.")


def compare_boundary_reference_files(
    mathematica_path: Path,
    python_path: Path,
    *,
    atol: float,
    rtol: float,
    strict_reference: bool = True,
) -> list[ComparisonResult]:
    mathematica = json.loads(mathematica_path.read_text(encoding="utf-8"))
    python = json.loads(python_path.read_text(encoding="utf-8"))
    return compare_boundary_reference_payloads(
        mathematica,
        python,
        atol=atol,
        rtol=rtol,
        strict_reference=strict_reference,
    )


def compare_boundary_reference_payloads(
    mathematica: dict[str, Any],
    python: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    strict_reference: bool = True,
) -> list[ComparisonResult]:
    if strict_reference:
        validate_boundary_reference_payload(mathematica)

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

        for mma_key, py_key in NUMERIC_OUTPUT_KEY_MAP.items():
            results.append(_compare_numeric(case_id, mma_key, mma_outputs[mma_key], py_outputs[py_key], atol=atol, rtol=rtol))
        for key in EXACT_OUTPUT_KEYS:
            results.append(_compare_exact(case_id, key, mma_outputs[key], py_outputs[key]))
        for key in WAVE_OUTPUT_KEYS:
            results.extend(_compare_wave_records(case_id, key, mma_outputs[key], py_outputs[key], atol=atol, rtol=rtol))
    return results


def validate_boundary_reference_payload(payload: dict[str, Any]) -> None:
    """Reject placeholders while allowing intentional nulls in structured metadata."""

    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("Strict reference comparison requires source metadata containing Mathematica or Wolfram.")
    if "python" in source:
        raise ValueError("Strict reference comparison refuses Python-generated payloads as Mathematica references.")

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
        raise ValueError(f"Strict reference comparison refuses unvalidated reference status: {payload.get('status')!r}.")

    cases = cases_by_id(payload, suite_id=SUITE_ID, output_key="mathematica_outputs")
    if not cases:
        raise ValueError("Strict reference comparison requires non-empty reference cases.")
    required_keys = set(NUMERIC_OUTPUT_KEY_MAP) | set(EXACT_OUTPUT_KEYS) | set(WAVE_OUTPUT_KEYS)
    for case in cases.values():
        case_id = case.get("id", "<missing>")
        outputs = _outputs(case)
        missing = sorted(required_keys - set(outputs))
        if missing:
            raise ValueError(f"Case {case_id} is missing required Mathematica outputs: {missing}")
        for key in NUMERIC_OUTPUT_KEY_MAP:
            _reject_none(outputs[key], f"case {case_id}.outputs.{key}")
        for key in WAVE_OUTPUT_KEYS:
            _reject_none_in_wave_records(outputs[key], f"case {case_id}.outputs.{key}")


def _reject_none(value: Any, path: str) -> None:
    if value is None:
        raise ValueError(f"Unfilled Mathematica output placeholder at {path}.")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_none(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _reject_none(item, f"{path}[{idx}]")


def _reject_none_in_wave_records(records: Any, path: str) -> None:
    if records is None:
        raise ValueError(f"Unfilled Mathematica output placeholder at {path}.")
    if not isinstance(records, list):
        raise ValueError(f"{path} must be a list.")
    for idx, record in enumerate(records):
        wave = record.get("wave") if isinstance(record, dict) else None
        _reject_none(wave, f"{path}[{idx}].wave")


def _outputs(case: dict[str, Any]) -> dict[str, Any]:
    outputs = case.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"Case {case.get('id')} is missing outputs.")
    return outputs


def _compare_numeric(
    case_id: str,
    key: str,
    mathematica_value: Any,
    python_value: Any,
    *,
    atol: float,
    rtol: float,
) -> ComparisonResult:
    mma = numeric_array(mathematica_value)
    py = numeric_array(python_value)
    if mma.shape != py.shape:
        return ComparisonResult(case_id, key, False, float("inf"), float("inf"), tuple(py.shape))
    delta = np.abs(py - mma)
    scale = np.maximum(np.abs(mma), np.finfo(float).tiny)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    return ComparisonResult(case_id, key, bool(np.allclose(py, mma, atol=atol, rtol=rtol)), max_abs, max_rel, tuple(py.shape))


def _compare_exact(case_id: str, key: str, mathematica_value: Any, python_value: Any) -> ComparisonResult:
    passed = mathematica_value == python_value
    return ComparisonResult(case_id, key, passed, 0.0 if passed else float("inf"), 0.0 if passed else float("inf"), ())


def _compare_wave_records(
    case_id: str,
    key: str,
    mathematica_records: list[dict[str, Any]],
    python_records: list[dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    if len(mathematica_records) != len(python_records):
        return [ComparisonResult(case_id, key, False, float("inf"), float("inf"), (len(python_records),))]

    results = []
    for idx, (mma_record, py_record) in enumerate(zip(mathematica_records, python_records)):
        record_key = f"{key}[{idx}]"
        for metadata_key in ("metadata", "region", "layer_index", "basis_index", "role"):
            if metadata_key in mma_record or metadata_key in py_record:
                results.append(_compare_exact(case_id, f"{record_key}.{metadata_key}", mma_record.get(metadata_key), py_record.get(metadata_key)))
        mma_wave = mma_record.get("wave")
        py_wave = py_record.get("wave")
        if not isinstance(mma_wave, dict) or not isinstance(py_wave, dict):
            results.append(ComparisonResult(case_id, f"{record_key}.wave", False, float("inf"), float("inf"), ()))
            continue
        for wave_key in ("omega", "k", "electric", "magnetic", "tangential_eh"):
            results.append(_compare_numeric(case_id, f"{record_key}.wave.{wave_key}", mma_wave[wave_key], py_wave[wave_key], atol=atol, rtol=rtol))
    return results


if __name__ == "__main__":
    main()
