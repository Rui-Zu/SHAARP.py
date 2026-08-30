from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.discrepancy_report import write_draft_reports_from_comparison_results


DEFAULT_OUTPUT_KEYS = (
    "single_interface_intensity",
    "multilayer_reflected_intensity",
    "multilayer_transmitted_intensity",
)


@dataclass(frozen=True)
class ComparisonResult:
    case_id: str
    key: str
    passed: bool
    max_abs_error: float
    max_rel_error: float
    shape: tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare exported Mathematica SHAARP reference values against Python benchmark values."
    )
    parser.add_argument("--mathematica", type=Path, required=True, help="Mathematica reference JSON file.")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(__file__).resolve().parent / "numeric_benchmarks_v1.json",
        help="Python benchmark JSON file.",
    )
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument(
        "--keys",
        nargs="*",
        default=list(DEFAULT_OUTPUT_KEYS),
        help="Output keys to compare under each case's outputs object.",
    )
    parser.add_argument(
        "--suite-id",
        default=None,
        help="Suite id to compare when the Mathematica reference uses the generated suites/items template format.",
    )
    parser.add_argument(
        "--allow-unverified-source",
        action="store_true",
        help="Allow comparison against payloads without explicit Mathematica/Wolfram source metadata. Use only for comparator smoke tests.",
    )
    parser.add_argument(
        "--write-discrepancy-reports",
        type=Path,
        default=None,
        help="Optional directory where failed comparisons are written as draft discrepancy reports.",
    )
    parser.add_argument("--source-notebook-or-package", default="SHAARP")
    parser.add_argument("--mathematica-version-or-runtime", default="unknown")
    parser.add_argument("--shaarp-file-path-version-or-date", default="unknown")
    parser.add_argument("--python-git-or-worktree-state", default="unknown")
    parser.add_argument("--first-failing-stage", default="unknown")
    args = parser.parse_args()

    results = compare_reference_files(
        args.mathematica,
        args.python,
        keys=args.keys,
        atol=args.atol,
        rtol=args.rtol,
        strict_reference=not args.allow_unverified_source,
        suite_id=args.suite_id,
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
        raise SystemExit(f"{len(failed)} Mathematica reference comparisons failed.")
    print(f"All {len(results)} Mathematica reference comparisons passed.")


def compare_reference_files(
    mathematica_path: Path,
    python_path: Path,
    *,
    keys: list[str],
    atol: float,
    rtol: float,
    strict_reference: bool = True,
    suite_id: str | None = None,
) -> list[ComparisonResult]:
    mathematica = json.loads(mathematica_path.read_text(encoding="utf-8"))
    python = json.loads(python_path.read_text(encoding="utf-8"))
    return compare_payloads(
        mathematica,
        python,
        keys=keys,
        atol=atol,
        rtol=rtol,
        strict_reference=strict_reference,
        suite_id=suite_id,
    )


def compare_payloads(
    mathematica: dict[str, Any],
    python: dict[str, Any],
    *,
    keys: list[str],
    atol: float,
    rtol: float,
    strict_reference: bool = False,
    suite_id: str | None = None,
) -> list[ComparisonResult]:
    if strict_reference:
        validate_mathematica_reference_payload(mathematica, suite_id=suite_id)
    mathematica_cases = cases_by_id(mathematica, suite_id=suite_id, output_key="mathematica_outputs")
    python_cases = cases_by_id(python)
    missing = sorted(set(mathematica_cases) - set(python_cases))
    if missing:
        raise ValueError(f"Python benchmark is missing case ids: {missing}")

    results = []
    for case_id, mma_case in sorted(mathematica_cases.items()):
        py_case = python_cases[case_id]
        for key in keys:
            mma_value = case_output(mma_case, key)
            py_value = case_output(py_case, key)
            results.append(compare_array(case_id, key, mma_value, py_value, atol=atol, rtol=rtol))
    return results


def validate_mathematica_reference_payload(payload: dict[str, Any], *, suite_id: str | None = None) -> None:
    """Reject placeholders or Python-generated files as Mathematica references."""

    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("Strict reference comparison requires source metadata containing Mathematica or Wolfram.")
    if "python" in source:
        raise ValueError("Strict reference comparison refuses Python-generated payloads as Mathematica references.")

    status = str(payload.get("status", "")).lower()
    blocked_markers = ("not_yet", "requested", "pending", "not_mathematica_validation", "python_regression")
    if any(marker in status for marker in blocked_markers):
        raise ValueError(f"Strict reference comparison refuses unvalidated reference status: {payload.get('status')!r}.")

    cases = cases_by_id(payload, suite_id=suite_id, output_key="mathematica_outputs")
    if not cases:
        raise ValueError("Strict reference comparison requires non-empty reference cases.")
    for case in cases.values():
        case_id = case.get("id", "<missing>")
        outputs = case.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError(f"Case {case_id} must contain non-empty Mathematica outputs.")
        _reject_unfilled_placeholders(outputs, f"case {case_id}.outputs")


def _reject_unfilled_placeholders(value: Any, path: str) -> None:
    if value is None:
        raise ValueError(f"Unfilled Mathematica output placeholder at {path}.")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_unfilled_placeholders(item, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _reject_unfilled_placeholders(item, f"{path}[{idx}]")


def cases_by_id(
    payload: dict[str, Any],
    *,
    suite_id: str | None = None,
    output_key: str = "outputs",
) -> dict[str, dict[str, Any]]:
    cases = _payload_cases(payload, suite_id=suite_id, output_key=output_key)
    if not isinstance(cases, list):
        raise ValueError("Payload must contain a cases list or a suites/items list.")
    out = {}
    for case in cases:
        case_id = case.get("id")
        if not case_id:
            raise ValueError("Every case must contain an id.")
        out[str(case_id)] = case
    return out


def _payload_cases(payload: dict[str, Any], *, suite_id: str | None, output_key: str) -> list[dict[str, Any]] | None:
    cases = payload.get("cases")
    if cases is not None:
        return cases
    suites = payload.get("suites")
    if suites is None:
        return None
    if not isinstance(suites, list) or not suites:
        raise ValueError("Payload suites must be a non-empty list.")
    if suite_id is None:
        if len(suites) != 1:
            raise ValueError("Suite-form payload has multiple suites; pass --suite-id.")
        suite = suites[0]
    else:
        matches = [suite for suite in suites if suite.get("suite_id") == suite_id]
        if not matches:
            raise ValueError(f"Suite-form payload is missing suite_id {suite_id!r}.")
        suite = matches[0]
    items = suite.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Suite {suite.get('suite_id')} must contain an items list.")
    normalized = []
    for item in items:
        case = dict(item)
        if "outputs" not in case and output_key in case:
            case["outputs"] = case[output_key]
        normalized.append(case)
    return normalized


def case_output(case: dict[str, Any], key: str) -> Any:
    outputs = case.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"Case {case.get('id')} does not contain an outputs object.")
    if key not in outputs:
        raise ValueError(f"Case {case.get('id')} is missing outputs.{key}.")
    return outputs[key]


def compare_array(
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
    rel = delta / scale
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    passed = bool(np.allclose(py, mma, atol=atol, rtol=rtol))
    return ComparisonResult(case_id, key, passed, max_abs, max_rel, tuple(py.shape))


def numeric_array(value: Any) -> np.ndarray:
    return np.asarray(convert_numeric(value), dtype=complex)


def convert_numeric(value: Any) -> Any:
    if isinstance(value, dict):
        if {"real", "imag"} <= set(value):
            return complex(float(value["real"]), float(value["imag"]))
        if {"re", "im"} <= set(value):
            return complex(float(value["re"]), float(value["im"]))
        raise ValueError(f"Unsupported numeric object keys: {sorted(value)}")
    if isinstance(value, list):
        return [convert_numeric(item) for item in value]
    if isinstance(value, (int, float)):
        return value
    raise ValueError(f"Unsupported numeric value: {value!r}")


if __name__ == "__main__":
    main()
