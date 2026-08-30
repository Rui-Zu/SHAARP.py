from __future__ import annotations

import math
from typing import Any

import numpy as np

from benchmarks.compare_mathematica_reference import ComparisonResult, numeric_array
from benchmarks.reference_paths import resolve_reference_path
from shaarp.anisotropic import shaarp_ordered_generalized_eigensystem
from shaarp.boundary import solve_fresnel_boundary
from shaarp.si_compat import (
    solve_shaarp_si_active_angle_roots,
    solve_shaarp_si_linear_omega,
)
from shaarp.waves import Wave


def compare_linear_field_replay(
    branch_payload: dict[str, Any],
    fresnel_payload: dict[str, Any],
    angle_payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Replay SHAARP.si omega linear fields from exported branch metadata.

    This is not a standalone end-to-end solver: it intentionally uses
    Mathematica-exported angle roots, branch indices, eigenvectors, and
    6-digit index labels. It verifies the Python boundary assembly and SI
    coordinate convention needed before implementing the self-contained path.
    """

    _require_payload(branch_payload, "branch_metadata_stage_only_not_full_shaarp_agreement")
    _require_payload(fresnel_payload, "fresnel_stage_only_not_full_shaarp_agreement")
    _require_payload(angle_payload, "angle_stage_only_not_full_shaarp_agreement")
    branch_by_id = {case["id"]: case for case in branch_payload["cases"]}
    angle_by_id = {case["id"]: case for case in angle_payload["cases"]}
    mapping = _load_mapping(fresnel_payload)
    results: list[ComparisonResult] = []
    for case in fresnel_payload["cases"]:
        case_id = case["id"]
        replay = _replay_linear_case(case, branch_by_id[case_id], angle_by_id[case_id], mapping[case_id])
        results.append(_compare(case_id, "ETweo_replayed_from_branch_metadata", case["outputs"]["ETweo"], replay["ETweo"], atol=atol, rtol=rtol))
        results.append(_compare(case_id, "ETwo_replayed_from_branch_metadata", case["outputs"]["ETwo"], replay["ETwo"], atol=atol, rtol=rtol))
        expected_coeffs = np.array([numeric_array(rule["value"]).item() for rule in case["outputs"]["EwrootsNum"]], dtype=complex)
        results.append(_compare(case_id, "EwrootsNum_replayed_from_branch_metadata", expected_coeffs, replay["EwrootsNum"], atol=atol, rtol=rtol))
    return results


def compare_linear_field_replay_with_python_eigensystem(
    branch_payload: dict[str, Any],
    fresnel_payload: dict[str, Any],
    angle_payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Replay SHAARP.si omega fields with Python-computed eigenvectors.

    This is one step closer to a standalone SI-compatible solver than
    ``compare_linear_field_replay``: it still uses Mathematica-exported angle
    roots and branch indices, but it computes the generalized eigensystems in
    Python and only uses the reference JSON for selecting the SHAARP.si branch.
    """

    _require_payload(branch_payload, "branch_metadata_stage_only_not_full_shaarp_agreement")
    _require_payload(fresnel_payload, "fresnel_stage_only_not_full_shaarp_agreement")
    _require_payload(angle_payload, "angle_stage_only_not_full_shaarp_agreement")
    branch_by_id = {case["id"]: case for case in branch_payload["cases"]}
    angle_by_id = {case["id"]: case for case in angle_payload["cases"]}
    mapping = _load_mapping(fresnel_payload)
    results: list[ComparisonResult] = []
    for case in fresnel_payload["cases"]:
        case_id = case["id"]
        replay = _replay_linear_case_with_python_eigensystem(
            case,
            branch_by_id[case_id],
            angle_by_id[case_id],
            mapping[case_id],
        )
        results.append(_compare(case_id, "ETweo_replayed_from_python_eigensystem", case["outputs"]["ETweo"], replay["ETweo"], atol=atol, rtol=rtol))
        results.append(_compare(case_id, "ETwo_replayed_from_python_eigensystem", case["outputs"]["ETwo"], replay["ETwo"], atol=atol, rtol=rtol))
        expected_coeffs = np.array([numeric_array(rule["value"]).item() for rule in case["outputs"]["EwrootsNum"]], dtype=complex)
        results.append(_compare(case_id, "EwrootsNum_replayed_from_python_eigensystem", expected_coeffs, replay["EwrootsNum"], atol=atol, rtol=rtol))
    return results


def compare_linear_field_replay_with_python_angles_and_eigensystem(
    branch_payload: dict[str, Any],
    fresnel_payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Replay SHAARP.si omega fields with Python angles and eigenvectors.

    This still uses Mathematica branch-index metadata, but computes the active
    SHAARP.si angle roots, 6-digit-style index labels, and generalized
    eigensystems in Python.
    """

    _require_payload(branch_payload, "branch_metadata_stage_only_not_full_shaarp_agreement")
    _require_payload(fresnel_payload, "fresnel_stage_only_not_full_shaarp_agreement")
    branch_by_id = {case["id"]: case for case in branch_payload["cases"]}
    mapping = _load_mapping(fresnel_payload)
    results: list[ComparisonResult] = []
    for case in fresnel_payload["cases"]:
        case_id = case["id"]
        replay = _replay_linear_case_with_python_angles_and_eigensystem(
            case,
            branch_by_id[case_id],
            mapping[case_id],
        )
        results.append(_compare(case_id, "ETweo_replayed_from_python_angles_and_eigensystem", case["outputs"]["ETweo"], replay["ETweo"], atol=atol, rtol=rtol))
        results.append(_compare(case_id, "ETwo_replayed_from_python_angles_and_eigensystem", case["outputs"]["ETwo"], replay["ETwo"], atol=atol, rtol=rtol))
        expected_coeffs = np.array([numeric_array(rule["value"]).item() for rule in case["outputs"]["EwrootsNum"]], dtype=complex)
        results.append(_compare(case_id, "EwrootsNum_replayed_from_python_angles_and_eigensystem", expected_coeffs, replay["EwrootsNum"], atol=atol, rtol=rtol))
    return results


def compare_linear_field_replay_self_contained_python(
    fresnel_payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[ComparisonResult]:
    """Replay SHAARP.si omega fields from original inputs only."""

    _require_payload(fresnel_payload, "fresnel_stage_only_not_full_shaarp_agreement")
    mapping = _load_mapping(fresnel_payload)
    results: list[ComparisonResult] = []
    for case in fresnel_payload["cases"]:
        case_id = case["id"]
        replay = _replay_linear_case_self_contained_python(case, mapping[case_id])
        results.append(_compare(case_id, "ETweo_replayed_self_contained_python", case["outputs"]["ETweo"], replay["ETweo"], atol=atol, rtol=rtol))
        results.append(_compare(case_id, "ETwo_replayed_self_contained_python", case["outputs"]["ETwo"], replay["ETwo"], atol=atol, rtol=rtol))
        expected_coeffs = np.array([numeric_array(rule["value"]).item() for rule in case["outputs"]["EwrootsNum"]], dtype=complex)
        results.append(_compare(case_id, "EwrootsNum_replayed_self_contained_python", expected_coeffs, replay["EwrootsNum"], atol=atol, rtol=rtol))
    return results


def _replay_linear_case(
    fresnel_case: dict[str, Any],
    branch_case: dict[str, Any],
    angle_case: dict[str, Any],
    mapped_case: dict[str, Any],
) -> dict[str, np.ndarray]:
    theta_in = math.radians(float(mapped_case["theta_deg"]))
    incident_polarization = str(mapped_case["incident_polarization"])
    incident, reflected_basis = _shaarp_si_top_waves(theta_in, incident_polarization)
    ext_wave = _wave_from_branch_metadata(branch_case, angle_case, "ext")
    ord_wave = _wave_from_branch_metadata(branch_case, angle_case, "ord")
    _, transmitted, coeffs = solve_fresnel_boundary(
        [incident],
        reflected_basis,
        [ext_wave, ord_wave],
        mu=1.0,
    )
    return {
        "ETweo": transmitted[0].electric,
        "ETwo": transmitted[1].electric,
        "EwrootsNum": coeffs,
    }


def _replay_linear_case_with_python_eigensystem(
    fresnel_case: dict[str, Any],
    branch_case: dict[str, Any],
    angle_case: dict[str, Any],
    mapped_case: dict[str, Any],
) -> dict[str, np.ndarray]:
    theta_in = math.radians(float(mapped_case["theta_deg"]))
    incident_polarization = str(mapped_case["incident_polarization"])
    incident, reflected_basis = _shaarp_si_top_waves(theta_in, incident_polarization)
    ext_wave = _wave_from_python_eigensystem(branch_case, angle_case, mapped_case, "ext")
    ord_wave = _wave_from_python_eigensystem(branch_case, angle_case, mapped_case, "ord")
    _, transmitted, coeffs = solve_fresnel_boundary(
        [incident],
        reflected_basis,
        [ext_wave, ord_wave],
        mu=1.0,
    )
    return {
        "ETweo": transmitted[0].electric,
        "ETwo": transmitted[1].electric,
        "EwrootsNum": coeffs,
    }


def _replay_linear_case_with_python_angles_and_eigensystem(
    fresnel_case: dict[str, Any],
    branch_case: dict[str, Any],
    mapped_case: dict[str, Any],
) -> dict[str, np.ndarray]:
    theta_in = math.radians(float(mapped_case["theta_deg"]))
    incident_polarization = str(mapped_case["incident_polarization"])
    incident, reflected_basis = _shaarp_si_top_waves(theta_in, incident_polarization)
    ext_wave = _wave_from_python_angles_and_eigensystem(branch_case, mapped_case, "ext")
    ord_wave = _wave_from_python_angles_and_eigensystem(branch_case, mapped_case, "ord")
    _, transmitted, coeffs = solve_fresnel_boundary(
        [incident],
        reflected_basis,
        [ext_wave, ord_wave],
        mu=1.0,
    )
    return {
        "ETweo": transmitted[0].electric,
        "ETwo": transmitted[1].electric,
        "EwrootsNum": coeffs,
    }


def _replay_linear_case_self_contained_python(
    fresnel_case: dict[str, Any],
    mapped_case: dict[str, Any],
) -> dict[str, np.ndarray]:
    variables = mapped_case["mathematica_variables"]
    result = solve_shaarp_si_linear_omega(
        epsilon_omega_crystal=numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]),
        epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
        orientation_matrix=numeric_array(variables["a"]),
        theta_in_rad=math.radians(float(mapped_case["theta_deg"])),
        incident_polarization=str(mapped_case["incident_polarization"]),
    )
    return {
        "ETweo": result.transmitted_ext.electric,
        "ETwo": result.transmitted_ord.electric,
        "EwrootsNum": result.coefficients,
    }


def _shaarp_si_top_waves(theta_in: float, incident_polarization: str) -> tuple[Wave, list[Wave]]:
    rotation = _rhtilt(theta_in)
    inverse_rotation = np.linalg.inv(rotation)
    if incident_polarization == "p":
        incident_e = rotation @ np.array([1.0, 0.0, 0.0], dtype=complex)
    elif incident_polarization == "s":
        incident_e = rotation @ np.array([0.0, 1.0, 0.0], dtype=complex)
    else:
        raise ValueError(f"Unsupported incident polarization: {incident_polarization!r}")
    incident = Wave(
        k=rotation @ np.array([0.0, 0.0, -1.0], dtype=complex),
        electric=incident_e,
        omega=1.0,
    )
    reflected_p = Wave(
        k=inverse_rotation @ np.array([0.0, 0.0, 1.0], dtype=complex),
        electric=inverse_rotation @ np.array([-1.0, 0.0, 0.0], dtype=complex),
        omega=1.0,
    )
    reflected_s = Wave(
        k=inverse_rotation @ np.array([0.0, 0.0, 1.0], dtype=complex),
        electric=inverse_rotation @ np.array([0.0, 1.0, 0.0], dtype=complex),
        omega=1.0,
    )
    return incident, [reflected_p, reflected_s]


def _rhtilt(theta_in: float) -> np.ndarray:
    c = np.cos(theta_in)
    s = np.sin(theta_in)
    return np.array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=complex,
    )


def _wave_from_branch_metadata(branch_case: dict[str, Any], angle_case: dict[str, Any], label: str) -> Wave:
    branch_outputs = branch_case["outputs"]
    angle_outputs = angle_case["outputs"]
    if label == "ext":
        index = int(branch_outputs["omega_ext_k_index"]) - 1
        vectors = numeric_array(branch_outputs["EigenwNumExt_vectors"])
        theta = numeric_array(angle_outputs["ThetaExt"]).item()
        refractive_index = numeric_array(angle_outputs["nwNumExt"]).item()
    elif label == "ord":
        index = int(branch_outputs["omega_ord_k_index"]) - 1
        vectors = numeric_array(branch_outputs["EigenwNumOrd_vectors"])
        theta = numeric_array(angle_outputs["ThetaOrd"]).item()
        refractive_index = numeric_array(angle_outputs["nwNumOrd"]).item()
    else:
        raise ValueError(f"Unsupported branch label: {label!r}")
    return Wave(
        k=refractive_index * np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=complex),
        electric=np.asarray(vectors[index], dtype=complex),
        omega=1.0,
    )


def _wave_from_python_eigensystem(
    branch_case: dict[str, Any],
    angle_case: dict[str, Any],
    mapped_case: dict[str, Any],
    label: str,
) -> Wave:
    branch_outputs = branch_case["outputs"]
    angle_outputs = angle_case["outputs"]
    variables = mapped_case["mathematica_variables"]
    a_matrix = numeric_array(variables["a"])
    epsilon_omega_lab = a_matrix @ numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"]) @ a_matrix.T
    if label == "ext":
        index = int(branch_outputs["omega_ext_k_index"]) - 1
        theta = numeric_array(angle_outputs["ThetaExt"]).item()
        refractive_index = numeric_array(angle_outputs["nwNumExt"]).item()
    elif label == "ord":
        index = int(branch_outputs["omega_ord_k_index"]) - 1
        theta = numeric_array(angle_outputs["ThetaOrd"]).item()
        refractive_index = numeric_array(angle_outputs["nwNumOrd"]).item()
    else:
        raise ValueError(f"Unsupported branch label: {label!r}")
    direction = np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=complex)
    _, vectors = shaarp_ordered_generalized_eigensystem(direction, epsilon_omega_lab)
    return Wave(
        k=refractive_index * direction,
        electric=np.asarray(vectors[index], dtype=complex),
        omega=1.0,
    )


def _wave_from_python_angles_and_eigensystem(
    branch_case: dict[str, Any],
    mapped_case: dict[str, Any],
    label: str,
) -> Wave:
    branch_outputs = branch_case["outputs"]
    variables = mapped_case["mathematica_variables"]
    a_matrix = numeric_array(variables["a"])
    epsilon_omega_crystal = numeric_array(variables["\\[CurlyEpsilon]\\[Omega]Cry"])
    epsilon_omega_lab = a_matrix @ epsilon_omega_crystal @ a_matrix.T
    roots = solve_shaarp_si_active_angle_roots(
        epsilon_omega_crystal=epsilon_omega_crystal,
        epsilon_2omega_crystal=numeric_array(variables["\\[CurlyEpsilon]2\\[Omega]Cry"]),
        orientation_matrix=a_matrix,
        theta_in_rad=math.radians(float(mapped_case["theta_deg"])),
    )
    if label == "ext":
        index = int(branch_outputs["omega_ext_k_index"]) - 1
        theta = roots.theta_ext
        refractive_index = roots.n_ext_label
    elif label == "ord":
        index = int(branch_outputs["omega_ord_k_index"]) - 1
        theta = roots.theta_ord
        refractive_index = roots.n_ord_label
    else:
        raise ValueError(f"Unsupported branch label: {label!r}")
    direction = np.array([np.sin(theta), 0.0, -np.cos(theta)], dtype=complex)
    _, vectors = shaarp_ordered_generalized_eigensystem(direction, epsilon_omega_lab)
    return Wave(
        k=refractive_index * direction,
        electric=np.asarray(vectors[index], dtype=complex),
        omega=1.0,
    )


def _require_payload(payload: dict[str, Any], expected_claim: str) -> None:
    if payload.get("status") != "mathematica_stage_reference_exported":
        raise ValueError(f"Unexpected status: {payload.get('status')!r}")
    if payload.get("validation_claim") != expected_claim:
        raise ValueError(f"Unexpected validation claim: {payload.get('validation_claim')!r}")


def _load_mapping(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import json
    from pathlib import Path

    mapping = json.loads(
        resolve_reference_path(payload["mapping_path"]).read_text(encoding="utf-8"))
    return {case["id"]: case for case in mapping["cases"]}


def _compare(
    case_id: str,
    key: str,
    mathematica_value: Any,
    python_value: Any,
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
    return ComparisonResult(
        case_id=case_id,
        key=key,
        passed=bool(np.allclose(py, mma, atol=atol, rtol=rtol)),
        max_abs_error=float(np.max(delta)) if delta.size else 0.0,
        max_rel_error=float(np.max(rel)) if rel.size else 0.0,
        shape=tuple(py.shape),
    )
