from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import sqrtm

from benchmarks.compare_mathematica_reference import ComparisonResult, numeric_array
from benchmarks.reference_paths import resolve_reference_path
from shaarp.nonlinear import NonlinearSource, compute_pnl_voigt, solve_inhomogeneous_field


def compare_pnl_stage_reference(
    payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Compare SHAARP.si PNL stage reference values against Python formulas."""

    _require_stage_reference(payload, expected_claim="pnl_stage_only_not_full_shaarp_agreement")
    results: list[ComparisonResult] = []
    for case in payload["cases"]:
        case_id = case["id"]
        outputs = case["outputs"]
        d_shg = numeric_array(outputs["dSHG"])
        e_ext = numeric_array(outputs["ETweo"])
        e_ord = numeric_array(outputs["ETwo"])
        python_outputs = {
            "P2eo": compute_pnl_voigt(d_shg, e_ext, e_ext),
            "P2o": compute_pnl_voigt(d_shg, e_ord, e_ord),
            "Peoo": compute_pnl_voigt(d_shg, e_ext, e_ord, factor=2),
        }
        for key, python_value in python_outputs.items():
            results.append(_compare_numpy_array(case_id, key, outputs[key], python_value, atol=atol, rtol=rtol))
    return results


def compare_signal_stage_reference(
    payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Compare SHAARP.si reflected signal/intensity stage values with Python formulas."""

    _require_stage_reference(payload, expected_claim="signal_stage_only_not_full_shaarp_agreement")
    results: list[ComparisonResult] = []
    for case in payload["cases"]:
        case_id = case["id"]
        outputs = case["outputs"]
        signal = numeric_array(outputs["RSignal"])
        signal_crystal = numeric_array(outputs["RSignalCrystal"])
        i2wx = np.abs(signal[0] * np.conjugate(signal[0]))
        i2wy = np.abs(signal[1] * np.conjugate(signal[1]))
        i2w_parallel = np.abs(signal_crystal[0] * np.conjugate(signal_crystal[0]))
        i2w_perpendicular = np.abs(signal_crystal[1] * np.conjugate(signal_crystal[1]))
        max_i2wx = i2wx
        max_i2wy = i2wy
        python_outputs = {
            "I2wx_at_phi_0": i2wx,
            "I2wy_at_phi_0": i2wy,
            "I2wParallel_at_phi_0": i2w_parallel,
            "I2wPerpdicular_at_phi_0": i2w_perpendicular,
            "MaxI2wx": max_i2wx,
            "MaxI2wy": max_i2wy,
            "MaxI2w": 1.1 * max(max_i2wx, max_i2wy),
        }
        for key, python_value in python_outputs.items():
            results.append(_compare_numpy_array(case_id, key, outputs[key], python_value, atol=atol, rtol=rtol))
    return results


def compare_inhomogeneous_stage_reference(
    payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Compare SHAARP.si Croots particular fields against Python solveInhom."""

    _require_stage_reference(payload, expected_claim="inhomogeneous_stage_only_not_full_shaarp_agreement")
    results: list[ComparisonResult] = []
    for case in payload["cases"]:
        case_id = case["id"]
        outputs = case["outputs"]
        epsilon_2w = numeric_array(outputs["elab2wNum"])
        vkt_ext = numeric_array(outputs["Vktweo"])
        vkt_ord = numeric_array(outputs["Vktwo"])
        croots = {rule["symbol"]: numeric_array(rule["value"]).item() for rule in outputs["Croots"]}
        expected_fields = {
            "inhomogeneous_oo": np.array([croots["C11"], croots["C21"], croots["C31"]], dtype=complex),
            "inhomogeneous_ee": np.array([croots["C12"], croots["C22"], croots["C32"]], dtype=complex),
            "inhomogeneous_eo": np.array([croots["C13"], croots["C23"], croots["C33"]], dtype=complex),
        }
        sources = {
            "inhomogeneous_oo": NonlinearSource("oo", 2 * vkt_ord, numeric_array(outputs["P2o"]), omega=2.0),
            "inhomogeneous_ee": NonlinearSource("ee", 2 * vkt_ext, numeric_array(outputs["P2eo"]), omega=2.0),
            "inhomogeneous_eo": NonlinearSource("eo", vkt_ord + vkt_ext, numeric_array(outputs["Peoo"]), omega=2.0),
        }
        for key, source in sources.items():
            actual = solve_inhomogeneous_field(epsilon_2w, source, mu=1.0, eps0=1.0).electric
            results.append(_compare_numpy_array(case_id, key, expected_fields[key], actual, atol=atol, rtol=rtol))
    return results


def compare_fresnel_stage_reference(
    payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Check the SHAARP.si omega Fresnel tangential boundary equations in Python."""

    _require_stage_reference(payload, expected_claim="fresnel_stage_only_not_full_shaarp_agreement")
    zero = {"real": 0.0, "imag": 0.0}
    results: list[ComparisonResult] = []
    for case in payload["cases"]:
        case_id = case["id"]
        outputs = case["outputs"]
        e_i = numeric_array(outputs["EIwNum"])
        e_r = numeric_array(outputs["ERwNum"])
        h_i = numeric_array(outputs["HIwNum"])
        h_r = numeric_array(outputs["HRwNum"])
        e_t_ord = numeric_array(outputs["ETwNumOrd"])
        e_t_ext = numeric_array(outputs["ETwNumExt"])
        h_t_ord = numeric_array(outputs["HTwNumOrd"])
        h_t_ext = numeric_array(outputs["HTwNumExt"])
        residuals = {
            "omega_boundary_e_x": e_i[0] + e_r[0] - e_t_ord[0] - e_t_ext[0],
            "omega_boundary_e_y": e_i[1] + e_r[1] - e_t_ord[1] - e_t_ext[1],
            "omega_boundary_h_x": h_i[0] + h_r[0] - h_t_ord[0] - h_t_ext[0],
            "omega_boundary_h_y": h_i[1] + h_r[1] - h_t_ord[1] - h_t_ext[1],
        }
        for key, residual in residuals.items():
            results.append(_compare_numpy_array(case_id, key, zero, residual, atol=atol, rtol=rtol))
    return results


def compare_angle_stage_reference(
    payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Check exported SHAARP.si angle/index values against scalar Snell closure."""

    _require_stage_reference(payload, expected_claim="angle_stage_only_not_full_shaarp_agreement")
    mapping = _load_case_mapping(payload)
    zero = {"real": 0.0, "imag": 0.0}
    results: list[ComparisonResult] = []
    for case in payload["cases"]:
        case_id = case["id"]
        theta_in = math.radians(float(mapping[case_id]["theta_deg"]))
        target = np.sin(theta_in)
        outputs = case["outputs"]
        residuals = {
            "omega_snell_ord": numeric_array(outputs["nwNumOrd"]).item() * np.sin(numeric_array(outputs["ThetaOrd"]).item()) - target,
            "omega_snell_ext": numeric_array(outputs["nwNumExt"]).item() * np.sin(numeric_array(outputs["ThetaExt"]).item()) - target,
            "two_omega_snell_ord": numeric_array(outputs["n2wNumOrd"]).item() * np.sin(numeric_array(outputs["ThetaOrd2w"]).item()) - target,
            "two_omega_snell_ext": numeric_array(outputs["n2wNumExt"]).item() * np.sin(numeric_array(outputs["ThetaExt2w"]).item()) - target,
        }
        for key, residual in residuals.items():
            results.append(_compare_numpy_array(case_id, key, zero, residual, atol=atol, rtol=rtol))
    return results


def compare_angle_stage_reference_against_shaarp_si_root_equations(
    payload: dict[str, Any],
    *,
    residual_atol: float,
    index_atol: float,
) -> list[ComparisonResult]:
    """Compare SHAARP.si angle labels against the exported anisotropic root equations.

    SHAARP.si does not use the final rounded ``n*sin(theta)`` scalar relation
    as its branch check. It solves ordinary/extraordinary root equations built
    from the ``V``/``W`` invariants of ``Transpose[Cstandard].L.Cstandard`` and
    then stores 6-digit-precision index labels plus branch-flip factors.
    """

    _require_stage_reference(payload, expected_claim="angle_stage_only_not_full_shaarp_agreement")
    mapping = _load_case_mapping(payload)
    results: list[ComparisonResult] = []
    zero = {"real": 0.0, "imag": 0.0}
    for case in payload["cases"]:
        case_id = case["id"]
        mapped_case = mapping[case_id]
        outputs = case["outputs"]
        theta_in = math.radians(float(mapped_case["theta_deg"]))
        omega_ctx = _shaarp_si_angle_context(mapped_case, "\\[CurlyEpsilon]\\[Omega]Cry")
        two_omega_ctx = _shaarp_si_angle_context(mapped_case, "\\[CurlyEpsilon]2\\[Omega]Cry")

        theta_ord = numeric_array(outputs["ThetaOrd"]).item()
        theta_ext = numeric_array(outputs["ThetaExt"]).item()
        theta_ord_2w = numeric_array(outputs["ThetaOrd2w"]).item()
        theta_ext_2w = numeric_array(outputs["ThetaExt2w"]).item()

        residuals = {
            "omega_ord_root_equation_residual": _shaarp_si_angle_root_residual(
                omega_ctx, theta_in, theta_ord, branch_sign=1
            ),
            "omega_ext_root_equation_residual": _shaarp_si_angle_root_residual(
                omega_ctx, theta_in, theta_ext, branch_sign=-1
            ),
            "two_omega_ord_root_equation_residual": _shaarp_si_angle_root_residual(
                two_omega_ctx, theta_in, theta_ord_2w, branch_sign=1
            ),
            "two_omega_ext_root_equation_residual": _shaarp_si_angle_root_residual(
                two_omega_ctx, theta_in, theta_ext_2w, branch_sign=-1
            ),
        }
        for key, residual in residuals.items():
            results.append(_compare_numpy_array(case_id, key, zero, residual, atol=residual_atol, rtol=0.0))

        omega_indices = _shaarp_si_angle_indices_and_factor(omega_ctx, theta_ext, theta_ord)
        two_omega_indices = _shaarp_si_angle_indices_and_factor(two_omega_ctx, theta_ext_2w, theta_ord_2w)
        index_outputs = {
            "nwNumExt_from_root_formula": (outputs["nwNumExt"], omega_indices["extraordinary_index"]),
            "nwNumOrd_from_root_formula": (outputs["nwNumOrd"], omega_indices["ordinary_index"]),
            "n2wNumExt_from_root_formula": (outputs["n2wNumExt"], two_omega_indices["extraordinary_index"]),
            "n2wNumOrd_from_root_formula": (outputs["n2wNumOrd"], two_omega_indices["ordinary_index"]),
        }
        for key, (expected, actual) in index_outputs.items():
            results.append(_compare_numpy_array(case_id, key, expected, actual, atol=index_atol, rtol=0.0))

        factor_outputs = {
            "ExtwFactor_from_branch_heuristic": (outputs["ExtwFactor"], omega_indices["ext_factor"]),
            "Ext2wFactor_from_branch_heuristic": (outputs["Ext2wFactor"], two_omega_indices["ext_factor"]),
        }
        for key, (expected, actual) in factor_outputs.items():
            results.append(_compare_numpy_array(case_id, key, expected, actual, atol=0.0, rtol=0.0))
    return results


def compare_two_omega_boundary_stage_reference(
    payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Check the SHAARP.si 2omega tangential boundary equations in Python."""

    _require_stage_reference(payload, expected_claim="two_omega_boundary_stage_only_not_full_shaarp_agreement")
    zero = {"real": 0.0, "imag": 0.0}
    results: list[ComparisonResult] = []
    for case in payload["cases"]:
        case_id = case["id"]
        outputs = case["outputs"]
        er_2w = numeric_array(outputs["ER2w"])
        hr_2w = numeric_array(outputs["HR2w"])
        et_ext = numeric_array(outputs["ET2wNumExt"])
        et_ord = numeric_array(outputs["ET2wNumOrd"])
        ht_ext = numeric_array(outputs["HT2wNumExt"])
        ht_ord = numeric_array(outputs["HT2wNumOrd"])
        ht_2eo = numeric_array(outputs["HT2w2eo"])
        ht_2o = numeric_array(outputs["HT2w2o"])
        ht_eoo = numeric_array(outputs["HT2weoo"])
        rc = numeric_array(outputs["RC"])
        coeffs = {rule["symbol"]: numeric_array(rule["value"]).item() for rule in outputs["E2wroots"]}
        et_2weo_a = coeffs["ET2weoA"]
        et_2wo_a = coeffs["ET2woA"]
        residuals = {
            "two_omega_boundary_e_x": er_2w[0] - (et_2weo_a * et_ext[0] + et_2wo_a * et_ord[0] + np.sum(rc[0])),
            "two_omega_boundary_e_y": er_2w[1] - (et_2weo_a * et_ext[1] + et_2wo_a * et_ord[1] + np.sum(rc[1])),
            "two_omega_boundary_h_x": hr_2w[0] - (et_2weo_a * ht_ext[0] + et_2wo_a * ht_ord[0] + ht_2eo[0] + ht_2o[0] + ht_eoo[0]),
            "two_omega_boundary_h_y": hr_2w[1] - (et_2weo_a * ht_ext[1] + et_2wo_a * ht_ord[1] + ht_2eo[1] + ht_2o[1] + ht_eoo[1]),
        }
        for key, residual in residuals.items():
            results.append(_compare_numpy_array(case_id, key, zero, residual, atol=atol, rtol=rtol))
    return results


def _require_stage_reference(payload: dict[str, Any], *, expected_claim: str) -> None:
    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("Expected a Mathematica/Wolfram SHAARP.si stage reference payload.")
    if payload.get("status") != "mathematica_stage_reference_exported":
        raise ValueError(f"Unexpected stage reference status: {payload.get('status')!r}.")
    if payload.get("validation_claim") != expected_claim:
        raise ValueError(f"Unexpected validation claim: {payload.get('validation_claim')!r}.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Stage reference payload must contain non-empty cases.")


def _load_case_mapping(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping_path = payload.get("mapping_path")
    if not mapping_path:
        raise ValueError("Stage reference payload is missing mapping_path.")
    path = resolve_reference_path(mapping_path)
    mapping = json.loads(path.read_text(encoding="utf-8"))
    cases = mapping.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Stage reference mapping file must contain non-empty cases.")
    return {str(case["id"]): case for case in cases}


def _shaarp_si_angle_context(mapped_case: dict[str, Any], epsilon_key: str) -> np.ndarray:
    variables = mapped_case["mathematica_variables"]
    a = numeric_array(variables["a"])
    epsilon_crystal = numeric_array(variables[epsilon_key])
    epsilon_sqrt = np.asarray(sqrtm(epsilon_crystal), dtype=complex)
    principal_rotation = _principal_axis_rotation_like_shaarp_si(epsilon_crystal, epsilon_sqrt)
    refractive_index_principal = _chop_small(principal_rotation @ epsilon_sqrt @ principal_rotation.T)
    q_matrix = principal_rotation @ np.linalg.inv(a)
    return np.linalg.inv(refractive_index_principal @ q_matrix)


def _principal_axis_rotation_like_shaarp_si(epsilon_crystal: np.ndarray, epsilon_sqrt: np.ndarray) -> np.ndarray:
    inv = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=complex)
    _, eigenvectors = np.linalg.eig(np.real(epsilon_sqrt))
    rotation = inv @ eigenvectors.T
    reconstructed = rotation @ epsilon_crystal @ rotation.T
    if np.allclose(_six_digit(reconstructed), _six_digit(epsilon_crystal), atol=5e-6, rtol=0.0):
        return np.eye(3, dtype=complex)
    return rotation


def _shaarp_si_l_matrix(theta: complex) -> np.ndarray:
    ux = np.sin(theta)
    uy = 0.0
    uz = -np.cos(theta)
    return np.array(
        [
            [uy**2 + uz**2, -ux * uy, -ux * uz],
            [-ux * uy, ux**2 + uz**2, -uy * uz],
            [-uz * ux, -uz * uy, ux**2 + uy**2],
        ],
        dtype=complex,
    )


def _shaarp_si_vw(context: np.ndarray, theta: complex) -> tuple[complex, complex]:
    matrix = context.T @ _shaarp_si_l_matrix(theta) @ context
    v_value = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2
    w_value = (
        matrix[0, 0] * matrix[1, 1]
        + matrix[0, 0] * matrix[2, 2]
        + matrix[1, 1] * matrix[2, 2]
        - matrix[0, 2] * matrix[2, 0]
        - matrix[0, 1] * matrix[1, 0]
        - matrix[1, 2] * matrix[2, 1]
    )
    return v_value, w_value


def _shaarp_si_angle_root_residual(
    context: np.ndarray,
    theta_in: float,
    theta_out: complex,
    *,
    branch_sign: int,
) -> complex:
    v_value, w_value = _shaarp_si_vw(context, theta_out)
    branch_index = np.sqrt((v_value + branch_sign * np.sqrt(v_value**2 - w_value)) / w_value)
    return np.sin(theta_in) - branch_index * np.sin(theta_out)


def _shaarp_si_angle_indices_and_factor(
    context: np.ndarray,
    theta_ext: complex,
    theta_ord: complex,
) -> dict[str, complex]:
    ext_factor = 1
    v_ext, w_ext = _shaarp_si_vw(context, theta_ext)
    v_ord, w_ord = _shaarp_si_vw(context, theta_ord)
    extraordinary_index = np.sqrt((v_ext + ext_factor * np.sqrt(v_ext**2 - w_ext)) / w_ext)
    ordinary_index = np.sqrt((v_ord - ext_factor * np.sqrt(v_ord**2 - w_ord)) / w_ord)
    if abs(extraordinary_index) > abs(ordinary_index) and abs(theta_ext) > abs(theta_ord):
        ext_factor *= -1
    if abs(extraordinary_index) < abs(ordinary_index) and abs(theta_ext) < abs(theta_ord):
        ext_factor *= -1
    return {
        "extraordinary_index": extraordinary_index,
        "ordinary_index": ordinary_index,
        "ext_factor": complex(ext_factor),
    }


def _six_digit(value: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(value, dtype=complex), 6)


def _chop_small(value: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    result = np.array(value, dtype=complex, copy=True)
    result[np.abs(result) < tol] = 0.0
    return result


def _compare_numpy_array(
    case_id: str,
    key: str,
    mathematica_value: Any,
    python_value: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> ComparisonResult:
    if isinstance(mathematica_value, np.ndarray):
        mma = np.asarray(mathematica_value, dtype=complex)
    else:
        mma = numeric_array(mathematica_value)
    py = np.asarray(python_value, dtype=complex)
    if mma.shape != py.shape:
        return ComparisonResult(case_id, key, False, float("inf"), float("inf"), tuple(py.shape))
    delta = np.abs(py - mma)
    scale = np.maximum(np.abs(mma), np.finfo(float).tiny)
    rel = delta / scale
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(rel)) if rel.size else 0.0
    passed = bool(np.allclose(py, mma, atol=atol, rtol=rtol))
    return ComparisonResult(case_id, key, passed, max_abs, max_rel, tuple(py.shape))
