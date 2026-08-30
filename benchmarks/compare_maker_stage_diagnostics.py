from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_mathematica_reference import numeric_array  # noqa: E402
from benchmarks.generate_maker_fringes_benchmarks import build_cases  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    solve_multilayer_shg_from_system_polarimetry,
)
from shaarp.config import Polarimetry  # noqa: E402
from dataclasses import replace  # noqa: E402


STAGE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_stage_diagnostics_v1.json"
SUMMARY_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_stage_comparison_summary_v1.json"
SINGULAR_CASE_IDS = {"maker_fringes_phase_matching_like"}


@dataclass(frozen=True)
class StageDelta:
    case_id: str
    theta_deg: float
    stage: str
    max_abs: float


def build_stage_agreement_summary(payload: dict | None = None, *, atol: float = 1e-10) -> dict:
    """Summarize the Maker f4NL stage comparison without hiding caveats.

    The raw comparison intentionally reports direct index deltas, best-order
    deltas, and absent Mathematica wave slots. This summary promotes only the
    stage comparisons whose interpretation is currently understood:
    nonsingular best-match fields, SHAARP.ml selected transmitted waves, and
    forward inhomogeneous sources. Phase-matching and homogeneous-layer
    ordering are recorded as diagnostics rather than pass/fail claims.
    """

    payload = payload or json.loads(STAGE_PATH.read_text(encoding="utf-8"))
    deltas = build_stage_deltas(payload)
    nonsingular_case_ids = sorted(
        case["id"] for case in payload["cases"] if case["id"] not in SINGULAR_CASE_IDS
    )
    singular_case_ids = sorted(case["id"] for case in payload["cases"] if case["id"] in SINGULAR_CASE_IDS)

    nonsingular = [delta for delta in deltas if delta.case_id in nonsingular_case_ids]
    singular = [delta for delta in deltas if delta.case_id in singular_case_ids]

    stages = [
        _stage_status(
            "linear_reflected_best_match",
            "Omega reflected waves match Mathematica after wave-order best matching.",
            _finite_max(
                nonsingular,
                lambda stage: stage.startswith("linear_reflected[") and ".best_" in stage,
            ),
            atol,
        ),
        _stage_status(
            "linear_transmitted_selected_wave",
            "Omega transmitted wave 0 matches the SHAARP.ml GUI selected-wave convention.",
            _finite_max(
                nonsingular,
                lambda stage: stage.startswith("linear_transmitted[0].best_"),
            ),
            atol,
        ),
        _stage_status(
            "inhomogeneous_forward_sources",
            "Layer-1 forward-only inhomogeneous SHG source waves match Mathematica after best matching.",
            _finite_max(
                nonsingular,
                lambda stage: stage.startswith("inhomogeneous_layer_1[") and ".best_" in stage,
            ),
            atol,
        ),
        _stage_status(
            "nonlinear_transmitted_selected_wave",
            "2omega transmitted selected wave matches Mathematica for nonsingular Maker cases.",
            _finite_max(
                nonsingular,
                lambda stage: stage.startswith("nonlinear_transmitted[") and ".best_" in stage,
            ),
            atol,
        ),
        _stage_status(
            "homogeneous_layer_2omega_boundary",
            "Forward/backward homogeneous 2omega layer waves match Mathematica after comparing the nonlinear boundary records.",
            _finite_max(
                nonsingular,
                lambda stage: (
                    stage.startswith("forward_homogeneous_layer_1[")
                    or stage.startswith("backward_homogeneous_layer_1[")
                )
                and ".best_" in stage,
            ),
            atol,
        ),
        {
            "stage_id": "transmitted_absent_slots",
            "comparison_status": "documented_caveat",
            "max_abs_error": None,
            "case_count": _count_cases(
                nonsingular,
                lambda stage: stage.endswith(".present_mismatch") and "transmitted" in stage,
            ),
            "notes": (
                "Mathematica emits absent/non-Association transmitted wave slots where Python "
                "keeps both eigenwave records; this is tied to the selected-wave policy."
            ),
        },
        {
            "stage_id": "phase_matching_case",
            "comparison_status": "diagnostic_only",
            "max_abs_error": _finite_max(singular, lambda stage: True),
            "case_count": len(singular_case_ids),
            "notes": (
                "The phase-matching-like case is intentionally excluded from small-residual "
                "agreement claims because solveInhom is singular or nearly singular."
            ),
        },
    ]

    return {
        "source": payload.get("source"),
        "status": "partial_stage_agreement_with_selected_wave_and_phase_matching_caveats",
        "full_shaarp_ml_agreement_claimed": False,
        "atol": atol,
        "nonsingular_case_count": len(nonsingular_case_ids),
        "nonsingular_case_ids": nonsingular_case_ids,
        "singular_case_ids": singular_case_ids,
        "selected_transmitted_wave_policy": "shaarp_ml_selected",
        "stages": stages,
    }


def write_stage_agreement_summary(path: Path, payload: dict | None = None, *, atol: float = 1e-10) -> dict:
    summary = build_stage_agreement_summary(payload, atol=atol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_stage_deltas(payload: dict | None = None) -> list[StageDelta]:
    payload = payload or json.loads(STAGE_PATH.read_text(encoding="utf-8"))
    mma_cases = {case["id"]: case for case in payload["cases"]}
    py_cases = {case["id"]: case for case in build_cases()}
    deltas: list[StageDelta] = []
    for case_id, mma_case in mma_cases.items():
        py_case = py_cases[case_id]
        for point in mma_case["points"]:
            theta = float(point["theta_deg"])
            system = replace(
                py_case["system"],
                polarimetry=replace(py_case["system"].polarimetry, theta_deg=theta),
            )
            result = solve_multilayer_shg_from_system_polarimetry(
                system,
                mu=1.0,
                eps0=1.0,
                inhomogeneous_source_policy="forward_only",
            )
            deltas.extend(compare_point(case_id, theta, point, result))
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Write a machine-readable agreement summary JSON instead of printing raw deltas.",
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    payload = json.loads(STAGE_PATH.read_text(encoding="utf-8"))
    if args.summary_out is not None:
        summary = write_stage_agreement_summary(args.summary_out, payload, atol=args.atol)
        print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))
        return

    deltas = build_stage_deltas(payload)
    for delta in sorted(deltas, key=lambda item: (item.case_id, item.theta_deg, item.stage)):
        print(f"{delta.case_id:38s} theta={delta.theta_deg:7.2f} {delta.stage:42s} max_abs={delta.max_abs:.15g}")


def compare_point(case_id: str, theta: float, point: dict, result) -> list[StageDelta]:
    deltas = []
    deltas.extend(
        _compare_wave_list(
            case_id,
            theta,
            "linear_reflected",
            point["linear_reflected"],
            result.fundamental.top_unknown,
        )
    )
    deltas.extend(
        _compare_wave_list(
            case_id,
            theta,
            "linear_transmitted",
            point["linear_transmitted"],
            result.fundamental.substrate_unknown,
        )
    )
    deltas.extend(
        _compare_wave_list(
            case_id,
            theta,
            "forward_homogeneous_layer_1",
            point["forward_homogeneous_by_layer"][0],
            result.shg.layer_homogeneous_2omega[0][:2],
        )
    )
    deltas.extend(
        _compare_wave_list(
            case_id,
            theta,
            "backward_homogeneous_layer_1",
            point["backward_homogeneous_by_layer"][0],
            result.shg.layer_homogeneous_2omega[0][2:],
        )
    )
    py_inh = result.shg.layer_known[0]
    deltas.extend(_compare_wave_list(case_id, theta, "inhomogeneous_layer_1", point["inhomogeneous_by_layer"][0], py_inh))
    deltas.extend(_compare_wave_list(case_id, theta, "nonlinear_transmitted", point["nonlinear_transmitted"], result.shg.substrate_2omega))
    return deltas


def _compare_wave_list(case_id: str, theta: float, label: str, mma_waves: list[dict], py_waves: list) -> list[StageDelta]:
    deltas = []
    count = min(len(mma_waves), len(py_waves))
    for idx in range(count):
        mma_wave = mma_waves[idx]
        py_wave = py_waves[idx]
        if not mma_wave.get("present", False):
            deltas.append(StageDelta(case_id, theta, f"{label}[{idx}].present_mismatch", float("inf")))
            continue
        deltas.append(_delta(case_id, theta, f"{label}[{idx}].k", _wave_value(mma_wave, "k"), py_wave.k))
        deltas.append(_delta(case_id, theta, f"{label}[{idx}].electric", _wave_value(mma_wave, "electric"), py_wave.electric))
    if len(mma_waves) != len(py_waves):
        deltas.append(StageDelta(case_id, theta, f"{label}.length", float(abs(len(mma_waves) - len(py_waves)))))
    for mma_idx, mma_wave in enumerate(mma_waves):
        if not mma_wave.get("present", False):
            continue
        best_k = min(
            (float(np.max(np.abs(_wave_value(mma_wave, "k") - py_wave.k))), py_idx)
            for py_idx, py_wave in enumerate(py_waves)
        )
        best_e = min(
            (float(np.max(np.abs(_wave_value(mma_wave, "electric") - py_wave.electric))), py_idx)
            for py_idx, py_wave in enumerate(py_waves)
        )
        deltas.append(StageDelta(case_id, theta, f"{label}[{mma_idx}].best_k_py[{best_k[1]}]", best_k[0]))
        deltas.append(StageDelta(case_id, theta, f"{label}[{mma_idx}].best_electric_py[{best_e[1]}]", best_e[0]))
    return deltas


def _wave_value(wave: dict, key: str) -> np.ndarray:
    return numeric_array(wave[key])


def _delta(case_id: str, theta: float, stage: str, mma: np.ndarray, py: np.ndarray) -> StageDelta:
    mma_arr = np.asarray(mma, dtype=complex)
    py_arr = np.asarray(py, dtype=complex)
    if mma_arr.shape != py_arr.shape:
        return StageDelta(case_id, theta, stage + ".shape", float("inf"))
    return StageDelta(case_id, theta, stage, float(np.max(np.abs(py_arr - mma_arr))) if mma_arr.size else 0.0)


def _stage_status(stage_id: str, notes: str, max_abs_error: float | None, atol: float) -> dict:
    passed = max_abs_error is not None and max_abs_error <= atol
    return {
        "stage_id": stage_id,
        "comparison_status": "passed" if passed else "failed",
        "max_abs_error": max_abs_error,
        "tolerance": atol,
        "notes": notes,
    }


def _finite_max(deltas: list[StageDelta], predicate) -> float | None:
    values = [delta.max_abs for delta in deltas if predicate(delta.stage) and np.isfinite(delta.max_abs)]
    return max(values) if values else None


def _count_cases(deltas: list[StageDelta], predicate) -> int:
    return len({delta.case_id for delta in deltas if predicate(delta.stage)})


if __name__ == "__main__":
    main()
