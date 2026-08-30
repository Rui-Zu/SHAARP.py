from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_mathematica_reference import numeric_array  # noqa: E402
from benchmarks.generate_maker_fringes_benchmarks import build_cases  # noqa: E402
from shaarp.nonlinear import curl_curl_matrix  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    _epsilon_lab,
    solve_multilayer_shg_from_system_polarimetry,
)


STAGE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_stage_diagnostics_v1.json"
OUTPUT_PATH = ROOT / "benchmarks" / "phase_matching_solve_inhom_diagnostics_v1.json"
PHASE_CASE_ID = "maker_fringes_phase_matching_like"


def main() -> None:
    mma_payload = json.loads(STAGE_PATH.read_text(encoding="utf-8"))
    mma_case = next(case for case in mma_payload["cases"] if case["id"] == PHASE_CASE_ID)
    py_case = next(case for case in build_cases() if case["id"] == PHASE_CASE_ID)

    point_records = []
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
        minimum_norm_result = solve_multilayer_shg_from_system_polarimetry(
            system,
            mu=1.0,
            eps0=1.0,
            inhomogeneous_source_policy="forward_only",
            inhomogeneous_solution_policy="minimum_norm_if_ill_conditioned",
        )
        epsilon_2omega = _epsilon_lab(system.layers[1].material, omega=False)
        py_fields = result.layer_inhomogeneous_fields[0].as_list()[:3]
        minimum_norm_fields = minimum_norm_result.layer_inhomogeneous_fields[0].as_list()[:3]
        mma_waves = point["inhomogeneous_by_layer"][0][:3]
        source_records = [
            diagnose_source(theta, idx, py_field, minimum_norm_field, mma_wave, epsilon_2omega)
            for idx, (py_field, minimum_norm_field, mma_wave) in enumerate(
                zip(py_fields, minimum_norm_fields, mma_waves)
            )
        ]
        point_records.append(
            {
                "theta_deg": theta,
                "sources": source_records,
                "max_python_residual_norm": max(record["python_residual_norm"] for record in source_records),
                "max_mathematica_residual_norm": max(
                    record["mathematica_residual_norm"] for record in source_records
                ),
                "max_field_delta_norm": max(record["field_delta_norm"] for record in source_records),
                "max_operator_condition": max(record["operator_condition"] for record in source_records),
            }
        )

    payload = {
        "source": "Python diagnostic using Mathematica SHAARP.ml maker_stage_diagnostics_v1.json values",
        "status": "phase_matching_solve_inhom_diagnostic",
        "case_id": PHASE_CASE_ID,
        "interpretation": [
            "This diagnostic checks values, not plot patterns.",
            "Both Python and Mathematica electric fields are evaluated against the same Python operator equation.",
            "Large condition numbers mean multiple large-amplitude particular solutions can be numerically unstable near phase matching.",
            "The minimum_norm_if_ill_conditioned policy removes the near-null direct-solve component using a condition-threshold SVD cutoff.",
        ],
        "points": point_records,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for point in point_records:
        print(
            f"theta={point['theta_deg']:7.2f} "
            f"max_cond={point['max_operator_condition']:.3e} "
            f"max_delta={point['max_field_delta_norm']:.3e} "
            f"max_res_py={point['max_python_residual_norm']:.3e} "
            f"max_res_mma={point['max_mathematica_residual_norm']:.3e}"
        )


def diagnose_source(
    theta: float,
    idx: int,
    py_field,
    minimum_norm_field,
    mma_wave: dict,
    epsilon_2omega: np.ndarray,
) -> dict:
    label = py_field.label
    if not mma_wave.get("present", False):
        return {
            "theta_deg": theta,
            "source_index": idx,
            "label": label,
            "mathematica_present": False,
        }

    alpha = py_field.omega**2
    operator = curl_curl_matrix(py_field.k)
    system = operator - alpha * epsilon_2omega
    rhs = alpha * py_field.polarization
    mma_electric = numeric_array(mma_wave["electric"])
    py_electric = py_field.electric
    minimum_norm_electric = minimum_norm_field.electric
    py_residual = system @ py_electric - rhs
    mma_residual = system @ mma_electric - rhs
    minimum_norm_residual = system @ minimum_norm_electric - rhs
    _, singular_values, vh = np.linalg.svd(system, full_matrices=True)
    rank_default = int(np.linalg.matrix_rank(system))
    rank_strict = int(np.linalg.matrix_rank(system, tol=1e-12))
    lstsq_electric, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    field_delta = py_electric - mma_electric
    near_null_vector = vh[-1, :].conj()
    null_projection = np.vdot(near_null_vector, field_delta) * near_null_vector
    field_delta_norm = float(np.linalg.norm(field_delta))
    null_projection_norm = float(np.linalg.norm(null_projection))

    return {
        "theta_deg": theta,
        "source_index": idx,
        "label": label,
        "mathematica_label": mma_wave.get("label"),
        "mathematica_present": True,
        "operator_condition": float(np.linalg.cond(system)),
        "operator_singular_values": [float(value) for value in singular_values],
        "operator_rank_default": rank_default,
        "operator_rank_tol_1e_minus_12": rank_strict,
        "python_solution_method": py_field.solution_method,
        "python_ill_conditioned": py_field.ill_conditioned,
        "python_residual_norm": float(np.linalg.norm(py_residual)),
        "mathematica_residual_norm": float(np.linalg.norm(mma_residual)),
        "least_squares_residual_norm": float(np.linalg.norm(system @ lstsq_electric - rhs)),
        "python_relative_residual": relative_residual(system, py_electric, rhs, py_residual),
        "mathematica_relative_residual": relative_residual(system, mma_electric, rhs, mma_residual),
        "minimum_norm_policy_solution_method": minimum_norm_field.solution_method,
        "minimum_norm_policy_residual_norm": float(np.linalg.norm(minimum_norm_residual)),
        "minimum_norm_policy_relative_residual": relative_residual(
            system,
            minimum_norm_electric,
            rhs,
            minimum_norm_residual,
        ),
        "least_squares_relative_residual": relative_residual(
            system,
            lstsq_electric,
            rhs,
            system @ lstsq_electric - rhs,
        ),
        "field_delta_norm": field_delta_norm,
        "field_delta_max_abs": float(np.max(np.abs(field_delta))),
        "field_delta_near_null_projection_norm": null_projection_norm,
        "field_delta_near_null_fraction": float(null_projection_norm / field_delta_norm)
        if field_delta_norm
        else 0.0,
        "python_field_norm": float(np.linalg.norm(py_electric)),
        "mathematica_field_norm": float(np.linalg.norm(mma_electric)),
        "minimum_norm_policy_field_norm": float(np.linalg.norm(minimum_norm_electric)),
        "default_to_minimum_norm_delta_norm": float(np.linalg.norm(py_electric - minimum_norm_electric)),
        "least_squares_field_norm": float(np.linalg.norm(lstsq_electric)),
    }


def relative_residual(system: np.ndarray, electric: np.ndarray, rhs: np.ndarray, residual: np.ndarray) -> float:
    scale = np.linalg.norm(system) * np.linalg.norm(electric) + np.linalg.norm(rhs)
    if scale == 0:
        return float(np.linalg.norm(residual))
    return float(np.linalg.norm(residual) / scale)


if __name__ == "__main__":
    main()
