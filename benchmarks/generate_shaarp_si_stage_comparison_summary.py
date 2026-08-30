from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from benchmarks.compare_shaarp_si_stage_reference import (
    compare_angle_stage_reference,
    compare_angle_stage_reference_against_shaarp_si_root_equations,
    compare_fresnel_stage_reference,
    compare_inhomogeneous_stage_reference,
    compare_pnl_stage_reference,
    compare_signal_stage_reference,
    compare_two_omega_boundary_stage_reference,
)


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
OUTPUT_PATH = REF_DIR / "shaarp_si_stage_comparison_summary_v1.json"


def build_summary() -> dict:
    stage_specs = [
        (
            "pnl",
            REF_DIR / "shaarp_si_pnl_stage_reference_v1.json",
            compare_pnl_stage_reference,
            "passed",
            {"atol": 1e-12, "rtol": 1e-12},
        ),
        (
            "inhomogeneous_croots",
            REF_DIR / "shaarp_si_inhomogeneous_stage_reference_v1.json",
            compare_inhomogeneous_stage_reference,
            "passed",
            {"atol": 1e-11, "rtol": 1e-11},
        ),
        (
            "omega_fresnel_boundary",
            REF_DIR / "shaarp_si_fresnel_stage_reference_v1.json",
            compare_fresnel_stage_reference,
            "passed",
            {"atol": 1e-12, "rtol": 1e-12},
        ),
        (
            "two_omega_boundary",
            REF_DIR / "shaarp_si_two_omega_boundary_stage_reference_v1.json",
            compare_two_omega_boundary_stage_reference,
            "passed",
            {"atol": 1e-12, "rtol": 1e-12},
        ),
        (
            "signal_intensity",
            REF_DIR / "shaarp_si_signal_stage_reference_v1.json",
            compare_signal_stage_reference,
            "passed",
            {"atol": 1e-12, "rtol": 1e-12},
        ),
        (
            "angle_root_equations",
            REF_DIR / "shaarp_si_angle_stage_reference_v1.json",
            compare_angle_stage_reference_against_shaarp_si_root_equations,
            "passed",
            {"residual_atol": 1e-12, "index_atol": 1e-5},
        ),
        (
            "angle_root_labels",
            REF_DIR / "shaarp_si_angle_stage_reference_v1.json",
            compare_angle_stage_reference,
            "diagnostic_only",
            {"atol": 1e-12, "rtol": 1e-12},
        ),
    ]
    stages = [
        _stage_summary(stage_id, reference_path, comparator, expected_status, kwargs)
        for stage_id, reference_path, comparator, expected_status, kwargs in stage_specs
    ]
    return {
        "schema_version": 1,
        "status": "partial_stage_value_comparison_not_full_shaarp_si_agreement",
        "source": "SHAARP.si Mathematica stage references compared with Python stage formulas",
        "full_shaarp_si_agreement_claimed": False,
        "passed_value_comparison_count": sum(
            stage["comparison_count"] for stage in stages if stage["comparison_status"] == "passed"
        ),
        "diagnostic_only_count": sum(
            stage["comparison_count"] for stage in stages if stage["comparison_status"] == "diagnostic_only"
        ),
        "stages": stages,
        "next_action": (
            "Use passed stages to build facade-level SI stage records; keep the scalar angle check "
            "as a diagnostic guard against mis-validating anisotropic branch labels."
        ),
    }


def _stage_summary(
    stage_id: str,
    reference_path: Path,
    comparator: Callable[..., list],
    expected_status: str,
    comparator_kwargs: dict,
) -> dict:
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    results = comparator(payload, **comparator_kwargs)
    failures = [result for result in results if not result.passed]
    if expected_status == "passed":
        comparison_status = "passed" if not failures else "failed"
    else:
        comparison_status = "diagnostic_only"
    return {
        "stage_id": stage_id,
        "reference_path": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
        "comparison_status": comparison_status,
        "comparison_count": len(results),
        "failed_count": len(failures),
        "max_abs_error": max((result.max_abs_error for result in results), default=0.0),
        "max_rel_error": max((result.max_rel_error for result in results), default=0.0),
        "tolerance": comparator_kwargs,
    }


def main() -> None:
    payload = build_summary()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si stage comparison summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
