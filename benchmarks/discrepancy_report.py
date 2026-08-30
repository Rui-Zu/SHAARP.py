from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_CLASSIFICATIONS = (
    "python_bug",
    "mathematica_shaarp_bug_or_ambiguity",
    "convention_mismatch",
    "insufficient_reference_data",
    "numerical_tolerance_or_conditioning",
    "singular_or_phase_matched_case",
    "unresolved",
)

REQUIRED_FIELDS = (
    "case_id",
    "source_notebook_or_package",
    "mathematica_version_or_runtime",
    "shaarp_file_path_version_or_date",
    "python_git_or_worktree_state",
    "first_failing_stage",
    "expected_mathematica_value",
    "python_value",
    "absolute_error",
    "relative_error",
    "tolerance",
    "intermediate_values_needed_to_reproduce",
    "classification",
    "next_action",
)

SHAARP_BUG_REQUIRED_FIELDS = (
    "minimal_reproducible_case",
    "equation_or_residual_check",
    "why_python_result_follows_documented_equations",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or validate SHAARP discrepancy triage reports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template", help="Write a blank discrepancy report template.")
    template_parser.add_argument("--output", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a filled discrepancy report.")
    validate_parser.add_argument("path", type=Path)

    args = parser.parse_args()
    if args.command == "template":
        report = blank_discrepancy_report()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote discrepancy report template to {args.output}")
    elif args.command == "validate":
        report = json.loads(args.path.read_text(encoding="utf-8"))
        validate_discrepancy_report(report)
        print(f"Discrepancy report is valid: {args.path}")


def blank_discrepancy_report() -> dict[str, Any]:
    return {
        "report_version": 1,
        "status": "draft_not_validated",
        "case_id": None,
        "source_notebook_or_package": None,
        "mathematica_version_or_runtime": None,
        "shaarp_file_path_version_or_date": None,
        "python_git_or_worktree_state": None,
        "first_failing_stage": None,
        "expected_mathematica_value": None,
        "python_value": None,
        "absolute_error": None,
        "relative_error": None,
        "tolerance": None,
        "intermediate_values_needed_to_reproduce": {},
        "classification": "unresolved",
        "next_action": None,
        "minimal_reproducible_case": None,
        "equation_or_residual_check": None,
        "why_python_result_follows_documented_equations": None,
        "notes": [],
    }


def draft_report_from_comparison_result(
    result: Any,
    *,
    source_notebook_or_package: str,
    mathematica_version_or_runtime: str,
    shaarp_file_path_version_or_date: str,
    python_git_or_worktree_state: str,
    first_failing_stage: str,
    tolerance: dict[str, float],
    expected_mathematica_value: Any = "see Mathematica reference payload",
    python_value: Any = "see Python benchmark payload",
    intermediate_values_needed_to_reproduce: dict[str, Any] | None = None,
    classification: str = "unresolved",
    next_action: str = "Isolate the first failing stage and inspect intermediate values.",
) -> dict[str, Any]:
    """Create a draft discrepancy report from a failed comparator result.

    The root cause is intentionally not inferred. Use the default `unresolved`
    classification until the first failing stage has been investigated.
    """

    report = blank_discrepancy_report()
    report.update(
        {
            "status": "draft_from_comparison_failure",
            "case_id": str(result.case_id),
            "source_notebook_or_package": source_notebook_or_package,
            "mathematica_version_or_runtime": mathematica_version_or_runtime,
            "shaarp_file_path_version_or_date": shaarp_file_path_version_or_date,
            "python_git_or_worktree_state": python_git_or_worktree_state,
            "first_failing_stage": first_failing_stage,
            "expected_mathematica_value": expected_mathematica_value,
            "python_value": python_value,
            "absolute_error": float(result.max_abs_error),
            "relative_error": float(result.max_rel_error),
            "tolerance": tolerance,
            "intermediate_values_needed_to_reproduce": intermediate_values_needed_to_reproduce or {
                "comparison_key": str(result.key),
                "comparison_shape": list(result.shape),
            },
            "classification": classification,
            "next_action": next_action,
        }
    )
    return report


def write_draft_reports_from_comparison_results(
    results: list[Any],
    output_dir: Path,
    *,
    source_notebook_or_package: str,
    mathematica_version_or_runtime: str,
    shaarp_file_path_version_or_date: str,
    python_git_or_worktree_state: str,
    first_failing_stage: str,
    tolerance: dict[str, float],
    next_action: str = "Isolate the first failing stage and inspect intermediate values.",
) -> list[Path]:
    failed = [result for result in results if not result.passed]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, result in enumerate(failed, start=1):
        report = draft_report_from_comparison_result(
            result,
            source_notebook_or_package=source_notebook_or_package,
            mathematica_version_or_runtime=mathematica_version_or_runtime,
            shaarp_file_path_version_or_date=shaarp_file_path_version_or_date,
            python_git_or_worktree_state=python_git_or_worktree_state,
            first_failing_stage=first_failing_stage,
            tolerance=tolerance,
            next_action=next_action,
        )
        path = output_dir / f"{index:03d}_{_safe_filename(result.case_id)}_{_safe_filename(result.key)}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def validate_discrepancy_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("Discrepancy report must be a JSON object.")
    missing = [field for field in REQUIRED_FIELDS if field not in report]
    if missing:
        raise ValueError(f"Discrepancy report is missing required fields: {missing}")

    classification = report["classification"]
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ValueError(f"Unsupported discrepancy classification: {classification!r}")

    for field in REQUIRED_FIELDS:
        if field == "python_git_or_worktree_state":
            continue
        if _is_empty(report[field]):
            raise ValueError(f"Discrepancy report field {field!r} must be filled.")

    if not isinstance(report["intermediate_values_needed_to_reproduce"], dict):
        raise ValueError("'intermediate_values_needed_to_reproduce' must be an object.")

    for field in ("absolute_error", "relative_error"):
        _require_numeric(report[field], field)
    _validate_tolerance(report["tolerance"])

    if classification == "mathematica_shaarp_bug_or_ambiguity":
        for field in SHAARP_BUG_REQUIRED_FIELDS:
            if field not in report or _is_empty(report[field]):
                raise ValueError(f"SHAARP bug/ambiguity reports require filled field {field!r}.")


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _require_numeric(value: Any, field: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Discrepancy report field {field!r} must be numeric.")


def _validate_tolerance(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("'tolerance' must be an object containing 'atol' and 'rtol'.")
    for field in ("atol", "rtol"):
        if field not in value:
            raise ValueError(f"'tolerance' is missing {field!r}.")
        _require_numeric(value[field], f"tolerance.{field}")


def _safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text).strip("_") or "item"


if __name__ == "__main__":
    main()
