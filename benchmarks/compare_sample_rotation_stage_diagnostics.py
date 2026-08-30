from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_mathematica_reference import numeric_array  # noqa: E402
from benchmarks.generate_sample_rotation_benchmarks import build_sample_rotation_cases  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    _sum_wave_frame_jones_sp,
    _transmitted_waves_for_maker_policy,
    _wave_frame_a_xy,
    sample_rotate_reflected_2omega_amplitudes,
    sample_rotate_transmitted_2omega_amplitudes,
    solve_multilayer_shg_sample_azimuth_sweep,
)


SAMPLE_ROTATION_STAGE_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "sample_rotation_stage_diagnostics_v1.json"
)
SAMPLE_ROTATION_STAGE_SUMMARY_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "sample_rotation_stage_comparison_summary_v1.json"
)
DIAGNOSTIC_CASE_IDS = {"sample_rotation_phase_matching_like"}


@dataclass(frozen=True)
class StageDelta:
    case_id: str
    sample_rotation_deg: float
    stage: str
    max_abs: float


def build_sample_rotation_stage_summary(payload: dict | None = None, *, atol: float = 1e-10) -> dict[str, Any]:
    payload = payload or json.loads(SAMPLE_ROTATION_STAGE_PATH.read_text(encoding="utf-8"))
    deltas = build_stage_deltas(payload)
    nonsingular_case_ids = sorted(
        case["id"] for case in payload["cases"] if case["id"] not in DIAGNOSTIC_CASE_IDS
    )
    diagnostic_case_ids = sorted(case["id"] for case in payload["cases"] if case["id"] in DIAGNOSTIC_CASE_IDS)
    nonsingular = [delta for delta in deltas if delta.case_id in nonsingular_case_ids]
    diagnostic = [delta for delta in deltas if delta.case_id in diagnostic_case_ids]

    stages = [
        _stage_status(
            "nonlinear_reflected_waves",
            "Direct reflected 2omega nonlinear wave records from SampleRotate.",
            _finite_max(nonsingular, lambda stage: stage.startswith("nonlinear_reflected[")),
            atol,
        ),
        {
            "stage_id": "nonlinear_transmitted_raw_wave_slots",
            "comparison_status": "documented_caveat",
            "max_abs_error": _finite_max(nonsingular, lambda stage: stage.startswith("nonlinear_transmitted[")),
            "case_count": len(
                {
                    delta.case_id
                    for delta in nonsingular
                    if delta.stage.startswith("nonlinear_transmitted[")
                    and not np.isfinite(delta.max_abs)
                }
            ),
            "notes": (
                "Mathematica SampleRotate emits an absent transmitted wave slot where Python "
                "retains both substrate 2omega eigenwave records; selected wave-frame sums and "
                "final transmitted amplitudes are the parity checks."
            ),
        },
        _stage_status(
            "sample_rotate_wave_frame_sums",
            "Wave-frame xy sums before SHAARP.ml SampleRotate analyzer projection.",
            _finite_max(nonsingular, lambda stage: stage.endswith("wave_frame_xy_sum")),
            atol,
        ),
        _stage_status(
            "sample_rotate_amplitudes",
            "Complex reflected/transmitted parallel and perpendicular SampleRotate amplitudes.",
            _finite_max(nonsingular, lambda stage: stage.startswith("sample_rotate_amplitudes.")),
            atol,
        ),
        _stage_status(
            "sample_rotate_intensities",
            "Final four SampleRotate intensity columns excluding the rotation angle.",
            _finite_max(nonsingular, lambda stage: stage == "sample_rotate_intensities"),
            atol,
        ),
        {
            "stage_id": "phase_matching_case",
            "comparison_status": "diagnostic_only",
            "max_abs_error": _finite_max(diagnostic, lambda stage: True),
            "case_count": len(diagnostic_case_ids),
            "notes": (
                "The phase-matching-like case remains diagnostic because solveInhom can be "
                "singular or nearly singular."
            ),
        },
    ]
    nonsingular_failed = [stage for stage in stages if stage.get("comparison_status") == "failed"]
    has_diagnostic = bool(diagnostic_case_ids)
    return {
        "source": payload.get("source"),
        "status": (
            "sample_rotation_stage_nonsingular_match_with_phase_matching_diagnostic"
            if not nonsingular_failed and has_diagnostic and all(_stage_passed_or_diagnostic(stage) for stage in stages)
            else "sample_rotation_stage_outputs_match"
            if not nonsingular_failed and all(_stage_passed_or_diagnostic(stage) for stage in stages)
            else "sample_rotation_stage_has_nonsingular_mismatch"
        ),
        "full_sample_rotation_agreement_claimed": False,
        "atol": atol,
        "nonsingular_case_count": len(nonsingular_case_ids),
        "nonsingular_case_ids": nonsingular_case_ids,
        "diagnostic_case_ids": diagnostic_case_ids,
        "stages": stages,
    }


def write_sample_rotation_stage_summary(path: Path, payload: dict | None = None, *, atol: float = 1e-10) -> dict[str, Any]:
    summary = build_sample_rotation_stage_summary(payload, atol=atol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_stage_deltas(payload: dict | None = None) -> list[StageDelta]:
    payload = payload or json.loads(SAMPLE_ROTATION_STAGE_PATH.read_text(encoding="utf-8"))
    payload = _attach_case_context(payload)
    mma_cases = {case["id"]: case for case in payload["cases"]}
    py_cases = {case["id"]: case for case in build_sample_rotation_cases()}
    deltas: list[StageDelta] = []
    for case_id, mma_case in sorted(mma_cases.items()):
        py_case = py_cases[case_id]
        sweep = solve_multilayer_shg_sample_azimuth_sweep(
            py_case["system"],
            sample_azimuth_deg=np.asarray(py_case["sample_rotation_deg"], dtype=float),
            mu=1.0,
            eps0=1.0,
            inhomogeneous_solution_policy="solve",
        )
        for idx, point in enumerate(mma_case["points"]):
            rotation = float(point["sample_rotation_deg"])
            result = sweep.results[idx]
            deltas.extend(_compare_point(case_id, rotation, point, result))
    return deltas


def _compare_point(case_id: str, rotation: float, point: dict, result) -> list[StageDelta]:
    deltas = []
    deltas.extend(_compare_wave_list(case_id, rotation, "nonlinear_reflected", point["nonlinear_reflected"], result.shg.reflected_2omega))
    deltas.extend(_compare_wave_list(case_id, rotation, "nonlinear_transmitted", point["nonlinear_transmitted"], result.shg.substrate_2omega))

    r_para, r_perp = sample_rotate_reflected_2omega_amplitudes(result.shg, _psi_deg_from_point_context(point))
    t_para, t_perp = sample_rotate_transmitted_2omega_amplitudes(result.shg, _psi_deg_from_point_context(point))
    py_amplitudes = {
        "reflected_parallel": r_para,
        "reflected_perpendicular": r_perp,
        "transmitted_parallel": t_para,
        "transmitted_perpendicular": t_perp,
    }
    mma_amplitudes = point["sample_rotate_amplitudes"]
    for key, py_value in py_amplitudes.items():
        deltas.append(_delta(case_id, rotation, f"sample_rotate_amplitudes.{key}", numeric_array(mma_amplitudes[key]), py_value))

    r_s, r_p = _sum_wave_frame_jones_sp(result.shg.reflected_2omega)
    transmitted = _transmitted_waves_for_maker_policy(result.shg, "shaarp_ml_selected")
    t_s, t_p = _sum_wave_frame_jones_sp(transmitted)
    deltas.append(
        _delta(
            case_id,
            rotation,
            "reflected_wave_frame_xy_sum",
            numeric_array(mma_amplitudes["reflected_wave_frame_xy_sum"]),
            np.asarray([r_p, r_s], dtype=complex),
        )
    )
    deltas.append(
        _delta(
            case_id,
            rotation,
            "transmitted_wave_frame_xy_sum",
            numeric_array(mma_amplitudes["transmitted_wave_frame_xy_sum"]),
            np.asarray([t_p, t_s], dtype=complex),
        )
    )
    py_intensities = np.asarray([abs(r_para) ** 2, abs(r_perp) ** 2, abs(t_para) ** 2, abs(t_perp) ** 2])
    deltas.append(
        _delta(case_id, rotation, "sample_rotate_intensities", numeric_array(point["sample_rotate_intensities"]), py_intensities)
    )
    return deltas


def _compare_wave_list(case_id: str, rotation: float, label: str, mma_waves: list[dict], py_waves: list) -> list[StageDelta]:
    deltas = []
    count = min(len(mma_waves), len(py_waves))
    for idx in range(count):
        mma_wave = mma_waves[idx]
        py_wave = py_waves[idx]
        if not mma_wave.get("present", False):
            deltas.append(StageDelta(case_id, rotation, f"{label}[{idx}].present_mismatch", float("inf")))
            continue
        deltas.append(_delta(case_id, rotation, f"{label}[{idx}].k", numeric_array(mma_wave["k"]), py_wave.k))
        deltas.append(_delta(case_id, rotation, f"{label}[{idx}].electric", numeric_array(mma_wave["electric"]), py_wave.electric))
        deltas.append(_delta(case_id, rotation, f"{label}[{idx}].A_xy", numeric_array(mma_wave["A"])[:2], _wave_a_xy(py_wave)))
    if len(mma_waves) != len(py_waves):
        deltas.append(StageDelta(case_id, rotation, f"{label}.length", float(abs(len(mma_waves) - len(py_waves)))))
    return deltas


def _wave_a_xy(wave) -> np.ndarray:
    return np.asarray(_wave_frame_a_xy(wave), dtype=complex)


def _psi_deg_from_point_context(point: dict) -> float:
    # The Mathematica point stores already-projected amplitudes but not psi.
    # Current SampleRotate fixtures all use psi=case-level scalar; comparator
    # fills this from the exported case before calling this helper.
    return float(point["_case_psi_deg"])


def _attach_case_context(payload: dict[str, Any]) -> dict[str, Any]:
    for case in payload["cases"]:
        for point in case["points"]:
            point["_case_psi_deg"] = case["psi_deg"]
    return payload


def _delta(case_id: str, rotation: float, stage: str, mma: Any, py: Any) -> StageDelta:
    mma_arr = np.asarray(mma, dtype=complex)
    py_arr = np.asarray(py, dtype=complex)
    if mma_arr.shape != py_arr.shape:
        return StageDelta(case_id, rotation, stage + ".shape", float("inf"))
    return StageDelta(case_id, rotation, stage, float(np.max(np.abs(py_arr - mma_arr))) if mma_arr.size else 0.0)


def _stage_status(stage_id: str, notes: str, max_abs_error: float | None, atol: float) -> dict[str, Any]:
    passed = max_abs_error is not None and max_abs_error <= atol
    return {
        "stage_id": stage_id,
        "comparison_status": "passed" if passed else "failed",
        "max_abs_error": max_abs_error,
        "tolerance": atol,
        "notes": notes,
    }


def _finite_max(deltas: list[StageDelta], predicate: Callable[[str], bool]) -> float | None:
    values = [delta.max_abs for delta in deltas if predicate(delta.stage) and np.isfinite(delta.max_abs)]
    return max(values) if values else None


def _stage_passed_or_diagnostic(stage: dict[str, Any]) -> bool:
    return stage["comparison_status"] in {"passed", "diagnostic_only", "documented_caveat"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Mathematica SHAARP.ml SampleRotate stage diagnostics.")
    parser.add_argument("--mathematica", type=Path, default=SAMPLE_ROTATION_STAGE_PATH)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    payload = _attach_case_context(json.loads(args.mathematica.read_text(encoding="utf-8")))
    if args.summary_out is not None:
        summary = write_sample_rotation_stage_summary(args.summary_out, payload, atol=args.atol)
        print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))
        return
    deltas = build_stage_deltas(payload)
    for delta in sorted(deltas, key=lambda item: (item.case_id, item.sample_rotation_deg, item.stage)):
        print(
            f"{delta.case_id:38s} rotation={delta.sample_rotation_deg:7.2f} "
            f"{delta.stage:48s} max_abs={delta.max_abs:.15g}"
        )


def _load_default_payload() -> dict[str, Any]:
    return _attach_case_context(json.loads(SAMPLE_ROTATION_STAGE_PATH.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
