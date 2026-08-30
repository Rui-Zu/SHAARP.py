from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_multilayer_system_benchmarks import (  # noqa: E402
    build_system,
    complex_array_to_json,
    complex_vector_to_json,
    max_abs_offdiag,
)
from shaarp.config import Polarimetry, with_sample_azimuth_deg  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    shaarp_ml_selected_transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
)


BENCHMARK_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate staged SHAARP.ml Maker fringes benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "maker_fringes_benchmarks_v1.json",
    )
    args = parser.parse_args()

    records = [run_case(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_staged_maker_fringes_regression_not_mathematica_validation",
        "case_count": len(records),
        "total_angle_count": sum(record["angle_count"] for record in records),
        "slow_path_policy": {
            "maker_fringes_can_be_slow": True,
            "default_saved_grid_is_intentionally_small": True,
            "phase_matching_like_case_is_diagnostic_not_small_residual_claim": True,
            "runtime_seconds_not_fingerprinted": True,
            "mathematica_reference_still_required_for_validation_claims": True,
        },
        "notes": [
            "These cases exercise solve_multilayer_maker_fringes_sweep and its listMFpara/listMFperp-shaped outputs.",
            "The saved grids include negative, normal, moderate, and high oblique angles without making routine tests slow.",
            "They use complex non-diagonal dielectric tensors, complex SHG coefficients, nontrivial orientations, polarizer/analyzer angles, ellipticity, and a phase-matching-like epsilon_omega = epsilon_2omega layer case.",
            "These are Python residual/regression benchmarks, not Mathematica-vs-Python validation.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} Maker fringes benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    raw_cases = [
        {
            "id": "maker_fringes_negative_normal_positive",
            "build_index": 0,
            "orientation_index": 1,
            "theta_deg": [-8.0, 0.0, 8.0],
            "phi_deg": 30.0,
            "psi_deg": 45.0,
            "ellipticity_deg": 10.0,
            "sample_azimuth_deg": 25.0,
            "phase_matching_like": False,
        },
        {
            "id": "maker_fringes_moderate_high_oblique",
            "build_index": 7,
            "orientation_index": 2,
            "theta_deg": [0.0, 20.0, 40.0, 60.0],
            "phi_deg": 67.5,
            "psi_deg": 22.5,
            "ellipticity_deg": -25.0,
            "sample_azimuth_deg": 55.0,
            "phase_matching_like": False,
        },
        {
            "id": "maker_fringes_phase_matching_like",
            "build_index": 13,
            "orientation_index": 3,
            "theta_deg": [-15.0, 0.0, 15.0],
            "phi_deg": 20.0,
            "psi_deg": 80.0,
            "ellipticity_deg": 35.0,
            "sample_azimuth_deg": 70.0,
            "phase_matching_like": True,
        },
    ]
    cases = []
    for raw in raw_cases:
        system = build_system(raw["build_index"], raw["theta_deg"][0], raw["orientation_index"])
        if raw["phase_matching_like"]:
            system = _with_active_layer_phase_matching_like(system)
        system = replace(
            system,
            polarimetry=Polarimetry(
                theta_deg=raw["theta_deg"][0],
                phi_deg=raw["phi_deg"],
                psi_deg=raw["psi_deg"],
                ellipticity_deg=raw["ellipticity_deg"],
            ),
        )
        system = with_sample_azimuth_deg(system, raw["sample_azimuth_deg"])
        cases.append({**raw, "system": system})
    return cases


def run_case(case: dict) -> dict:
    sweep = solve_multilayer_maker_fringes_sweep(
        case["system"],
        theta_deg=case["theta_deg"],
        mu=1.0,
        eps0=1.0,
    )
    active = case["system"].layers[1].material
    transmitted_jones = [
        shaarp_ml_selected_transmitted_2omega_jones_sp(result.shg)
        for result in sweep.results
    ]
    list_mf_para, list_mf_perp = sweep.shaarp_ml_copy_lists()
    record = {
        "id": case["id"],
        "theta_deg": [float(theta) for theta in sweep.theta_deg],
        "angle_count": int(len(sweep.theta_deg)),
        "orientation_index": case["orientation_index"],
        "phi_deg": case["phi_deg"],
        "psi_deg": case["psi_deg"],
        "ellipticity_deg": case["ellipticity_deg"],
        "sample_azimuth_deg": case["sample_azimuth_deg"],
        "phase_matching_like": case["phase_matching_like"],
        "inputs": {
            "wavelength_um": case["system"].wavelength_um,
            "thickness_um": [layer.thickness_um for layer in case["system"].layers[1:-1]],
            "orientation": complex_array_to_json(active.orientation.rotation_matrix()),
            "epsilon_omega_layer_crystal": complex_array_to_json(active.eps_w()),
            "epsilon_2omega_layer_crystal": complex_array_to_json(active.eps_2w()),
            "d_voigt_layer_crystal": complex_array_to_json(active.d_voigt()),
        },
        "outputs": {
            "list_mf_para": complex_array_to_json(list_mf_para),
            "list_mf_perp": complex_array_to_json(list_mf_perp),
            "parallel_amplitude": complex_vector_to_json(sweep.parallel_amplitude),
            "perpendicular_amplitude": complex_vector_to_json(sweep.perpendicular_amplitude),
            "transmitted_2omega_jones_sp": [complex_vector_to_json(np.asarray(jones)) for jones in transmitted_jones],
        },
        "metrics": {
            "max_fundamental_residual_l2": round(float(np.max(sweep.fundamental_residual_norm)), 15),
            "max_shg_residual_l2": round(float(np.max(sweep.shg_residual_norm)), 15),
            "small_shg_residual_expected": not case["phase_matching_like"],
            "max_inhomogeneous_operator_condition": round(
                float(
                    max(
                        field.operator_condition
                        for result in sweep.results
                        for layer_fields in result.layer_inhomogeneous_fields
                        for field in layer_fields.as_list()
                    )
                ),
                15,
            ),
            "any_ill_conditioned_inhomogeneous_field": any(
                field.ill_conditioned
                for result in sweep.results
                for layer_fields in result.layer_inhomogeneous_fields
                for field in layer_fields.as_list()
            ),
            "max_parallel_intensity": round(float(np.max(sweep.parallel_intensity)), 15),
            "max_perpendicular_intensity": round(float(np.max(sweep.perpendicular_intensity)), 15),
            "max_abs_epsilon_omega_imag": round(float(np.max(np.abs(np.imag(active.eps_w())))), 15),
            "max_abs_epsilon_2omega_imag": round(float(np.max(np.abs(np.imag(active.eps_2w())))), 15),
            "max_abs_epsilon_omega_offdiag": round(float(max_abs_offdiag(active.eps_w())), 15),
            "max_abs_d_imag": round(float(np.max(np.abs(np.imag(active.d_voigt())))), 15),
            "selected_transmitted_wave_count": sweep.selected_transmitted_wave_count,
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def fingerprint_record(record: dict) -> str:
    comparable = dict(record)
    comparable.pop("fingerprint", None)
    data = json.dumps(comparable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _with_active_layer_phase_matching_like(system):
    layers = list(system.layers)
    active = layers[1].material
    matched = replace(active, epsilon_2omega=active.eps_w())
    layers[1] = replace(layers[1], material=matched)
    return replace(system, layers=layers)


if __name__ == "__main__":
    main()
