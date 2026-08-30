from __future__ import annotations

import argparse
import json
from pathlib import Path


PLAN_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
DEFAULT_OUTPUT = BENCHMARK_DIR / "maker_fringes_fine_sampling_plan_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the slow-path Maker fringes fine-sampling plan.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_plan(), indent=2), encoding="utf-8")
    print(f"Wrote Maker fringes fine-sampling plan to {args.output}")


def build_plan() -> dict:
    cases = [
        _case(
            "fine_maker_negative_normal_positive",
            source_case_id="maker_fringes_negative_normal_positive",
            theta_min_deg=-12.0,
            theta_max_deg=12.0,
            theta_step_deg=0.25,
            purpose="Resolve fine structure around negative, normal, and positive incidence.",
            expected_residual_behavior="small_shg_residual_expected",
        ),
        _case(
            "fine_maker_high_oblique",
            source_case_id="maker_fringes_moderate_high_oblique",
            theta_min_deg=0.0,
            theta_max_deg=80.0,
            theta_step_deg=0.5,
            purpose="Check smoothness and consistency over a broad high-oblique transmitted Maker scan.",
            expected_residual_behavior="small_shg_residual_expected",
        ),
        _case(
            "fine_maker_fringe_resolution",
            source_case_id="maker_fringes_moderate_high_oblique",
            theta_min_deg=-60.0,
            theta_max_deg=60.0,
            theta_step_deg=0.25,
            purpose="Dense symmetric scan intended to catch missed Maker-fringe oscillations.",
            expected_residual_behavior="small_shg_residual_expected",
        ),
        _case(
            "fine_maker_phase_matching_like_local",
            source_case_id="maker_fringes_phase_matching_like",
            theta_min_deg=-5.0,
            theta_max_deg=5.0,
            theta_step_deg=0.1,
            purpose="Local dense diagnostic near the phase-matching-like singular regime.",
            expected_residual_behavior="diagnostic_extreme_not_small_residual_claim",
        ),
    ]
    return {
        "plan_version": PLAN_VERSION,
        "status": "slow_path_fine_sampling_plan_not_routine_test_not_mathematica_validation",
        "routine_suite_should_not_run_fine_grid": True,
        "mathematica_reference_required_for_agreement_claim": True,
        "slow_runner": "benchmarks/run_maker_fringes_fine_sampling.py --confirm-slow",
        "total_planned_angle_count": sum(case["theta_count"] for case in cases),
        "acceptance_checks": [
            "all computed intensities finite",
            "all non-diagnostic cases keep small fundamental and SH residuals",
            "diagnostic phase-matching-like case records residual behavior without forcing a small-residual claim",
            "coarse-grid samples agree with maker_fringes_benchmarks_v1.json at matching theta values",
            "Mathematica MFList/listMFpara/listMFperp values match within explicit tolerances before any SHAARP agreement claim",
        ],
        "cases": cases,
    }


def _case(
    case_id: str,
    *,
    source_case_id: str,
    theta_min_deg: float,
    theta_max_deg: float,
    theta_step_deg: float,
    purpose: str,
    expected_residual_behavior: str,
) -> dict:
    theta_values = _theta_values(theta_min_deg, theta_max_deg, theta_step_deg)
    return {
        "id": case_id,
        "source_case_id": source_case_id,
        "theta_min_deg": theta_min_deg,
        "theta_max_deg": theta_max_deg,
        "theta_step_deg": theta_step_deg,
        "theta_count": len(theta_values),
        "theta_deg": theta_values,
        "purpose": purpose,
        "expected_residual_behavior": expected_residual_behavior,
    }


def _theta_values(theta_min_deg: float, theta_max_deg: float, theta_step_deg: float) -> list[float]:
    if theta_step_deg <= 0:
        raise ValueError("theta_step_deg must be positive.")
    count = int((theta_max_deg - theta_min_deg) / theta_step_deg + 1e-9) + 1
    return [round(theta_min_deg + idx * theta_step_deg, 10) for idx in range(count)]


if __name__ == "__main__":
    main()
