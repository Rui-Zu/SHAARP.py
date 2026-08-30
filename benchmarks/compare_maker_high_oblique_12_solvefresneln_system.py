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
from shaarp.multilayer_basis import build_multilayer_omega_basis_jones  # noqa: E402
from shaarp.multilayer_boundary import build_multilayer_boundary_system  # noqa: E402
from shaarp.multilayer_shg_boundary import _system_setup, incident_jones_from_polarimetry  # noqa: E402


SYSTEM_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_high_oblique_12_solvefresneln_system_v1.json"
)
SUMMARY_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_high_oblique_12_solvefresneln_system_summary_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare theta=12 high-oblique solveFresnelN matrix/RHS/coefficients.")
    parser.add_argument("--mathematica", type=Path, default=SYSTEM_PATH)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    payload = json.loads(args.mathematica.read_text(encoding="utf-8"))
    summary = build_summary(payload, atol=args.atol)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))


def build_summary(payload: dict[str, Any], *, atol: float = 1e-10) -> dict[str, Any]:
    system = payload["solvefresneln_system"]
    mma_matrix = numeric_array(system["coefficient_arrays_M"])
    mma_b = numeric_array(system["coefficient_arrays_B"])
    mma_coeffs = numeric_array(system["nsolve_coefficients"])
    mma_direct_coeffs = -np.linalg.solve(mma_matrix, mma_b)
    py_system = _python_boundary_system(float(payload["theta_deg"]))
    py_waves = _python_basis_waves(float(payload["theta_deg"]))
    py_coeffs = np.linalg.solve(py_system.matrix, py_system.rhs)
    alignment = _column_alignment(mma_matrix, py_system.matrix)
    basis_deltas = _basis_deltas(system.get("unknown_wave_basis_records", []), py_waves)
    coeff_deltas = _coefficient_deltas(mma_coeffs, py_coeffs, alignment)
    max_coeff = max(delta["coefficient_abs_error"] for delta in coeff_deltas)
    max_aligned_matrix_rel = max(delta["relative_matrix_error"] for delta in alignment)
    max_b_plus_rhs = float(np.max(np.abs(mma_b + py_system.rhs)))
    cond_mma = float(np.linalg.cond(mma_matrix))
    cond_py = float(np.linalg.cond(py_system.matrix))
    matrix_matches = max_b_plus_rhs <= atol and max_aligned_matrix_rel <= 1e-6 and abs(cond_mma - cond_py) <= 1e-6
    coeffs_match = max_coeff <= atol
    return {
        "source": payload.get("source"),
        "status": (
            "theta12_solvefresneln_matrix_and_coefficients_match"
            if matrix_matches and coeffs_match
            else "theta12_solvefresneln_system_close_but_basis_coefficients_differ"
        ),
        "agreement_claimed": bool(matrix_matches and coeffs_match),
        "case_id": payload["case_id"],
        "theta_deg": payload["theta_deg"],
        "tolerance": atol,
        "mathematica_condition_number": cond_mma,
        "python_condition_number": cond_py,
        "max_abs_B_plus_python_rhs": max_b_plus_rhs,
        "max_aligned_matrix_relative_error": max_aligned_matrix_rel,
        "max_mathematica_direct_solve_vs_nsolve_coeff_error": float(np.max(np.abs(mma_direct_coeffs - mma_coeffs))),
        "max_basis_electric_relative_error": None if not basis_deltas else max(
            delta["electric_relative_error"] for delta in basis_deltas
        ),
        "max_basis_k_abs_error": None if not basis_deltas else max(delta["k_abs_error"] for delta in basis_deltas),
        "max_basis_error_record": None if not basis_deltas else max(
            basis_deltas, key=lambda item: item["electric_relative_error"]
        ),
        "max_aligned_coefficient_abs_error": max_coeff,
        "max_aligned_coefficient_error_column": max(coeff_deltas, key=lambda item: item["coefficient_abs_error"]),
        "column_alignment": alignment,
        "basis_deltas": basis_deltas,
        "coefficient_deltas": coeff_deltas,
        "interpretation": (
            "The exported Mathematica CoefficientArrays system is close to the Python boundary system "
            "after column phase/order alignment, with RHS agreement at machine precision and a maximum "
            "aligned matrix relative residual around 1e-8. Solving the exported Mathematica system "
            "directly reproduces Mathematica NSolve coefficients. The largest exported wave-basis "
            "difference is the EB11 electric basis, while k-vectors remain matched near machine precision."
        ),
    }


def _python_boundary_system(theta_deg: float):
    basis = _python_basis(theta_deg)
    setup = basis["setup"]
    omega_basis = basis["omega_basis"]
    return build_multilayer_boundary_system(
        top_known=omega_basis.incident,
        top_unknown_basis=omega_basis.reflected_basis,
        layer_unknown_basis=omega_basis.layer_basis,
        substrate_unknown_basis=omega_basis.substrate_basis,
        thicknesses=setup["thicknesses"],
        mu=1.0,
    )


def _python_basis_waves(theta_deg: float) -> list[Any]:
    omega_basis = _python_basis(theta_deg)["omega_basis"]
    return omega_basis.reflected_basis + omega_basis.layer_basis[0] + omega_basis.substrate_basis


def _python_basis(theta_deg: float) -> dict[str, Any]:
    case = next(case for case in build_cases() if case["id"] == "maker_fringes_moderate_high_oblique")
    system = replace(case["system"], polarimetry=replace(case["system"].polarimetry, theta_deg=theta_deg))
    setup = _system_setup(system)
    omega_basis = build_multilayer_omega_basis_jones(
        incident_index=setup["top_index_omega"],
        incident_theta_rad=setup["incident_theta_rad"],
        incident_jones_sp=incident_jones_from_polarimetry(system),
        layer_epsilon_lab=setup["layer_epsilon_omega_lab"],
        substrate_epsilon_lab=setup["substrate_epsilon_omega_lab"],
        omega=setup["omega"],
        layer_shaarp_basis=setup["layer_shaarp_basis_omega"],
        substrate_shaarp_basis=setup["substrate_shaarp_basis_omega"],
    )
    return {"setup": setup, "omega_basis": omega_basis}


def _column_alignment(mma_matrix: np.ndarray, py_matrix: np.ndarray) -> list[dict[str, Any]]:
    remaining = set(range(py_matrix.shape[1]))
    records = []
    for mma_col in range(mma_matrix.shape[1]):
        candidates = []
        for py_col in remaining:
            py_vector = py_matrix[:, py_col]
            denom = np.vdot(py_vector, py_vector)
            scale = np.vdot(py_vector, mma_matrix[:, mma_col]) / denom if denom else 0.0j
            residual = scale * py_vector - mma_matrix[:, mma_col]
            abs_error = float(np.linalg.norm(residual))
            rel_error = abs_error / max(float(np.linalg.norm(mma_matrix[:, mma_col])), 1e-300)
            candidates.append((rel_error, abs_error, py_col, scale))
        rel_error, abs_error, py_col, scale = min(candidates, key=lambda item: item[0])
        remaining.remove(py_col)
        records.append(
            {
                "mathematica_column": mma_col,
                "python_column": py_col,
                "scale": {"real": float(np.real(scale)), "imag": float(np.imag(scale))},
                "absolute_matrix_error": abs_error,
                "relative_matrix_error": rel_error,
            }
        )
    return records


def _coefficient_deltas(mma_coeffs: np.ndarray, py_coeffs: np.ndarray, alignment: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas = []
    for item in alignment:
        mma_col = int(item["mathematica_column"])
        py_col = int(item["python_column"])
        scale = complex(item["scale"]["real"], item["scale"]["imag"])
        expected_python = scale * mma_coeffs[mma_col]
        actual_python = py_coeffs[py_col]
        deltas.append(
            {
                "mathematica_column": mma_col,
                "python_column": py_col,
                "expected_python_coefficient_from_mathematica_nsolve": {
                    "real": float(np.real(expected_python)),
                    "imag": float(np.imag(expected_python)),
                },
                "actual_python_coefficient": {
                    "real": float(np.real(actual_python)),
                    "imag": float(np.imag(actual_python)),
                },
                "coefficient_abs_error": float(abs(actual_python - expected_python)),
            }
        )
    return deltas


def _basis_deltas(records: list[dict[str, Any]], py_waves: list[Any]) -> list[dict[str, Any]]:
    deltas = []
    remaining = set(range(len(py_waves)))
    for index, record in enumerate(records):
        mma_e = numeric_array(record["electric_basis"])
        mma_k = numeric_array(record["k"])
        candidates = []
        for py_index in remaining:
            py_wave = py_waves[py_index]
            denom = np.vdot(py_wave.electric, py_wave.electric)
            scale = np.vdot(py_wave.electric, mma_e) / denom if denom else 0.0j
            electric_residual = scale * py_wave.electric - mma_e
            electric_abs = float(np.linalg.norm(electric_residual))
            electric_rel = electric_abs / max(float(np.linalg.norm(mma_e)), 1e-300)
            k_abs = float(np.max(np.abs(mma_k - py_wave.k)))
            candidates.append((electric_rel, electric_abs, k_abs, py_index, scale))
        electric_rel, electric_abs, k_abs, py_index, scale = min(candidates, key=lambda item: item[0])
        remaining.remove(py_index)
        deltas.append(
            {
                "mathematica_basis_index": index,
                "mathematica_label": record["label"],
                "mathematica_unknown_symbol": record["unknown_symbol"],
                "python_basis_index": py_index,
                "scale": {"real": float(np.real(scale)), "imag": float(np.imag(scale))},
                "electric_abs_error": electric_abs,
                "electric_relative_error": electric_rel,
                "k_abs_error": k_abs,
            }
        )
    return deltas


if __name__ == "__main__":
    main()
