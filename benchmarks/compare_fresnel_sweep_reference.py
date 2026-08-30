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

from benchmarks.compare_mathematica_reference import ComparisonResult, numeric_array  # noqa: E402
from benchmarks.generate_maker_fringes_benchmarks import build_cases  # noqa: E402
from shaarp.multilayer_basis import build_multilayer_omega_basis_jones  # noqa: E402
from shaarp.multilayer_boundary import solve_multilayer_boundary  # noqa: E402
from shaarp.multilayer_shg_boundary import _sum_wave_frame_jones_sp, _system_setup  # noqa: E402


SUITE_ID = "fresnel_sweep_gui_workflow"
LABELS = ("Rp", "Rs", "Tp", "Ts")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Mathematica SHAARP.ml GUI Fresnel listFresnel references against Python multilayer boundary values."
    )
    parser.add_argument(
        "--mathematica",
        type=Path,
        default=Path(__file__).resolve().parent / "mathematica_reference" / "fresnel_sweep_reference_v1.json",
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    args = parser.parse_args()

    payload = json.loads(args.mathematica.read_text(encoding="utf-8"))
    results = compare_fresnel_sweep_payload(payload, atol=args.atol, rtol=args.rtol)
    failed = [result for result in results if not result.passed]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.case_id} {result.key}: "
            f"shape={result.shape} max_abs={result.max_abs_error:.3e} max_rel={result.max_rel_error:.3e}"
        )
    if failed:
        raise SystemExit(f"{len(failed)} Fresnel sweep comparisons failed.")
    print(f"All {len(results)} Fresnel sweep comparisons passed.")


def compare_fresnel_sweep_payload(payload: dict[str, Any], *, atol: float, rtol: float) -> list[ComparisonResult]:
    return compare_fresnel_sweep_payload_with_policy(
        payload,
        atol=atol,
        rtol=rtol,
        transmitted_wave_policy="shaarp_ml_selected",
    )


def compare_fresnel_sweep_payload_with_policy(
    payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    transmitted_wave_policy: str,
) -> list[ComparisonResult]:
    _validate_reference(payload)
    if transmitted_wave_policy not in {"shaarp_ml_selected", "physical_sum"}:
        raise ValueError("transmitted_wave_policy must be 'shaarp_ml_selected' or 'physical_sum'.")
    reference_cases = {
        item["id"]: item
        for suite in payload["suites"]
        if suite["suite_id"] == SUITE_ID
        for item in suite["items"]
    }
    python_cases = {case["id"]: case for case in build_cases()}
    missing = sorted(set(reference_cases) - set(python_cases))
    if missing:
        raise ValueError(f"Python Maker/Fresnel input builder is missing case ids: {missing}")

    results: list[ComparisonResult] = []
    for case_id, reference in sorted(reference_cases.items()):
        py_case = python_cases[case_id]
        theta_deg = [float(theta) for theta in reference["inputs"]["theta_deg"]]
        py_curves = _python_list_fresnel(py_case["system"], theta_deg, transmitted_wave_policy=transmitted_wave_policy)
        mma_curves = reference["mathematica_outputs"]["listFresnel"]
        labels = reference["mathematica_outputs"]["listFresnel_labels"]
        if labels != list(LABELS):
            raise ValueError(f"Unexpected listFresnel labels for {case_id}: {labels!r}")
        for label, mma_curve, py_curve in zip(LABELS, mma_curves, py_curves):
            results.append(_compare_numeric(case_id, label, mma_curve, py_curve, atol=atol, rtol=rtol))
    return results


def _validate_reference(payload: dict[str, Any]) -> None:
    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("Strict Fresnel sweep comparison requires Mathematica/Wolfram source metadata.")
    if "python" in source:
        raise ValueError("Strict Fresnel sweep comparison refuses Python-generated payloads as Mathematica references.")
    if payload.get("status") != "mathematica_reference_exported":
        raise ValueError(f"Unexpected Fresnel sweep reference status: {payload.get('status')!r}")


def _python_list_fresnel(system, theta_deg: list[float], *, transmitted_wave_policy: str) -> list[np.ndarray]:
    return [
        _curve(
            system,
            theta_deg,
            incident_jones_sp=(0.0j, 1.0 + 0.0j),
            field="reflected",
            component="p",
            transmitted_wave_policy=transmitted_wave_policy,
        ),
        _curve(
            system,
            theta_deg,
            incident_jones_sp=(1.0 + 0.0j, 0.0j),
            field="reflected",
            component="s",
            transmitted_wave_policy=transmitted_wave_policy,
        ),
        _curve(
            system,
            theta_deg,
            incident_jones_sp=(0.0j, 1.0 + 0.0j),
            field="transmitted",
            component="p",
            transmitted_wave_policy=transmitted_wave_policy,
        ),
        _curve(
            system,
            theta_deg,
            incident_jones_sp=(1.0 + 0.0j, 0.0j),
            field="transmitted",
            component="s",
            transmitted_wave_policy=transmitted_wave_policy,
        ),
    ]


def _curve(
    system,
    theta_deg: list[float],
    *,
    incident_jones_sp: tuple[complex, complex],
    field: str,
    component: str,
    transmitted_wave_policy: str,
) -> np.ndarray:
    rows = []
    for theta in theta_deg:
        point_system = replace(system, polarimetry=replace(system.polarimetry, theta_deg=float(theta)))
        solution = _solve_fundamental(point_system, incident_jones_sp)
        if field == "reflected":
            waves = solution.top_unknown
        elif transmitted_wave_policy == "shaarp_ml_selected":
            waves = solution.substrate_unknown[:1]
        else:
            waves = solution.substrate_unknown
        s_component, p_component = _sum_wave_frame_jones_sp(waves)
        amplitude = p_component if component == "p" else s_component
        rows.append([float(theta), float(abs(amplitude) ** 2)])
    return np.asarray(rows, dtype=float)


def _solve_fundamental(system, incident_jones_sp: tuple[complex, complex]):
    setup = _system_setup(system)
    basis = build_multilayer_omega_basis_jones(
        incident_index=setup["top_index_omega"],
        incident_theta_rad=setup["incident_theta_rad"],
        incident_jones_sp=incident_jones_sp,
        layer_epsilon_lab=setup["layer_epsilon_omega_lab"],
        substrate_epsilon_lab=setup["substrate_epsilon_omega_lab"],
        omega=setup["omega"],
        layer_shaarp_basis=setup["layer_shaarp_basis_omega"],
        substrate_shaarp_basis=setup["substrate_shaarp_basis_omega"],
    )
    return solve_multilayer_boundary(
        top_known=basis.incident,
        top_unknown_basis=basis.reflected_basis,
        layer_unknown_basis=basis.layer_basis,
        substrate_unknown_basis=basis.substrate_basis,
        thicknesses=setup["thicknesses"],
        mu=1.0,
    )


def _compare_numeric(
    case_id: str,
    key: str,
    mathematica_value: Any,
    python_value: Any,
    *,
    atol: float,
    rtol: float,
) -> ComparisonResult:
    mma = numeric_array(mathematica_value)
    py = np.asarray(python_value, dtype=complex)
    if mma.shape != py.shape:
        return ComparisonResult(case_id, key, False, float("inf"), float("inf"), tuple(py.shape))
    delta = np.abs(py - mma)
    scale = np.maximum(np.abs(mma), np.finfo(float).tiny)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    return ComparisonResult(
        case_id=case_id,
        key=key,
        passed=bool(np.allclose(py, mma, atol=atol, rtol=rtol)),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        shape=tuple(py.shape),
    )


if __name__ == "__main__":
    main()
