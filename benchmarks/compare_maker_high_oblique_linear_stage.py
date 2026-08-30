from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_mathematica_reference import numeric_array  # noqa: E402
from benchmarks.generate_maker_fringes_benchmarks import build_cases  # noqa: E402
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_system_polarimetry  # noqa: E402


LINEAR_STAGE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_high_oblique_80_linear_stage_v1.json"
SUMMARY_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_high_oblique_80_linear_stage_summary_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SHAARP.ml f4L linear waves at the high-oblique 80 degree endpoint.")
    parser.add_argument("--mathematica", type=Path, default=LINEAR_STAGE_PATH)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    payload = json.loads(args.mathematica.read_text(encoding="utf-8"))
    summary = build_linear_stage_summary(payload, atol=args.atol)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))
        return
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_linear_stage_summary(payload: dict[str, Any] | None = None, *, atol: float = 1e-10) -> dict[str, Any]:
    payload = payload or json.loads(LINEAR_STAGE_PATH.read_text(encoding="utf-8"))
    deltas = build_linear_stage_deltas(payload)
    finite = [
        delta["max_abs_error"]
        for delta in deltas
        if delta["max_abs_error"] is not None and np.isfinite(delta["max_abs_error"])
    ]
    max_abs = max(finite) if finite else None
    passed = max_abs is not None and max_abs <= atol
    return {
        "source": payload.get("source"),
        "status": (
            "python_direct_linear_stage_matches_mathematica"
            if passed
            else "python_direct_linear_stage_mismatch_at_high_oblique_endpoint"
        ),
        "agreement_claimed": bool(passed),
        "case_id": payload["case_id"],
        "theta_deg": payload["theta_deg"],
        "first_failing_stage": "omega_fundamental_boundary_waves",
        "max_abs_error": max_abs,
        "tolerance": atol,
        "deltas": deltas,
    }


def build_linear_stage_deltas(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = _python_direct_result(float(payload["theta_deg"]))
    comparisons = [
        # SHAARP.ml f4L returns reflected waves as p,s for this branch; the
        # Python boundary basis stores the isotropic top unknowns as s,p.
        ("linear_reflected", payload["linear_reflected"], [result.fundamental.top_unknown[1], result.fundamental.top_unknown[0]]),
        ("linear_transmitted", payload["linear_transmitted"], result.fundamental.substrate_unknown),
        ("forward_layer_waves", payload["forward_layer_waves"][0], result.fundamental.layer_unknowns[0][:2]),
        ("backward_layer_waves", payload["backward_layer_waves"][0], result.fundamental.layer_unknowns[0][2:]),
    ]
    deltas: list[dict[str, Any]] = []
    for label, mma_waves, py_waves in comparisons:
        deltas.extend(_compare_wave_list(label, mma_waves, py_waves))
    return deltas


def _python_direct_result(theta_deg: float):
    case = next(case for case in build_cases() if case["id"] == "maker_fringes_moderate_high_oblique")
    system = replace(case["system"], polarimetry=replace(case["system"].polarimetry, theta_deg=theta_deg))
    return solve_multilayer_shg_from_system_polarimetry(
        system,
        mu=1.0,
        eps0=1.0,
        inhomogeneous_source_policy="forward_only",
        inhomogeneous_solution_policy="solve",
    )


def _compare_wave_list(label: str, mma_waves: list[dict[str, Any]], py_waves: list[Any]) -> list[dict[str, Any]]:
    deltas = []
    if len(mma_waves) != len(py_waves):
        return [{"stage": f"{label}.length", "max_abs_error": float("inf")}]
    for idx, (mma_wave, py_wave) in enumerate(zip(mma_waves, py_waves)):
        if not mma_wave.get("present", True) or "k" not in mma_wave or "electric" not in mma_wave:
            deltas.append(
                {
                    "stage": f"{label}[{idx}]",
                    "max_abs_error": None,
                    "status": "mathematica_wave_absent_or_nonnumeric",
                    "mathematica_head": mma_wave.get("head"),
                    "mathematica_input_form": mma_wave.get("input_form"),
                }
            )
            continue
        for key, py_value in (("k", py_wave.k), ("electric", py_wave.electric)):
            mma = numeric_array(mma_wave[key])
            py = np.asarray(py_value, dtype=complex)
            error = float(np.max(np.abs(py - mma))) if py.size else 0.0
            deltas.append({"stage": f"{label}[{idx}].{key}", "max_abs_error": error})
    return deltas


if __name__ == "__main__":
    main()
