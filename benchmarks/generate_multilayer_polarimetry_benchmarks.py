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
    complex_scalar_to_json,
    complex_vector_to_json,
    max_abs_offdiag,
)
from shaarp.config import Polarimetry, with_sample_azimuth_deg  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    analyze_reflected_2omega_polarimetry,
    analyzer_jones_from_polarimetry,
    incident_jones_from_polarimetry,
    reflected_2omega_jones_sp,
    solve_multilayer_shg_from_system_polarimetry,
)


BENCHMARK_VERSION = 1
ANGLES_DEG = [0.0, 8.0, 18.0, 33.0, 52.0]
POLARIMETRY_CONFIGS = [
    {"phi_deg": 0.0, "psi_deg": 0.0, "ellipticity_deg": 0.0, "sample_azimuth_deg": 0.0, "label": "pp_linear"},
    {"phi_deg": 90.0, "psi_deg": 90.0, "ellipticity_deg": 0.0, "sample_azimuth_deg": 15.0, "label": "ss_linear"},
    {
        "phi_deg": 30.0,
        "psi_deg": 45.0,
        "ellipticity_deg": 15.0,
        "sample_azimuth_deg": 37.5,
        "label": "mixed_elliptic_plus",
    },
    {
        "phi_deg": 67.5,
        "psi_deg": 22.5,
        "ellipticity_deg": -35.0,
        "sample_azimuth_deg": 72.0,
        "label": "mixed_elliptic_minus",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multilayer polarimetry/analyzer SHG benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "multilayer_polarimetry_benchmarks_v1.json",
    )
    args = parser.parse_args()

    records = [run_case(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_staged_multilayer_polarimetry_regression_not_mathematica_validation",
        "case_count": len(records),
        "notes": [
            "These cases exercise the staged SHAARP.ml-style polarimetry and reflected-analyzer workflow.",
            "They cover normal and oblique incidence, pure p/p, pure s/s, mixed linear/elliptic incident states, analyzer angles, nontrivial orientations, complex non-diagonal tensors, and ten nonlinear source terms per active layer.",
            "These are Python residual/regression benchmarks, not Mathematica-vs-Python validation.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} multilayer-polarimetry benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    cases = []
    for angle_index, theta_deg in enumerate(ANGLES_DEG):
        for config_index, config in enumerate(POLARIMETRY_CONFIGS):
            idx = len(cases)
            orientation_index = config_index
            system = build_system(idx, theta_deg, orientation_index)
            system = replace(
                system,
                polarimetry=Polarimetry(
                    theta_deg=theta_deg,
                    phi_deg=config["phi_deg"],
                    psi_deg=config["psi_deg"],
                    ellipticity_deg=config["ellipticity_deg"],
                ),
            )
            system = with_sample_azimuth_deg(system, config["sample_azimuth_deg"])
            cases.append(
                {
                    "id": f"multilayer_polarimetry_{idx + 1:02d}",
                    "theta_deg": theta_deg,
                    "phi_deg": config["phi_deg"],
                    "psi_deg": config["psi_deg"],
                    "ellipticity_deg": config["ellipticity_deg"],
                    "sample_azimuth_deg": config["sample_azimuth_deg"],
                    "polarimetry_label": config["label"],
                    "orientation_index": orientation_index,
                    "system": system,
                }
            )
    return cases


def run_case(case: dict) -> dict:
    result = solve_multilayer_shg_from_system_polarimetry(
        case["system"],
        mu=1.0,
        eps0=1.0,
    )
    incident_jones = incident_jones_from_polarimetry(case["system"])
    analyzer_jones = analyzer_jones_from_polarimetry(case["system"])
    reflected_jones = reflected_2omega_jones_sp(result.shg)
    analyzer_amplitude, analyzer_intensity = analyze_reflected_2omega_polarimetry(result.shg, case["system"])
    all_fields = [field for layer in result.layer_inhomogeneous_fields for field in layer.as_list()]
    active = case["system"].layers[1].material
    record = {
        "id": case["id"],
        "theta_deg": case["theta_deg"],
        "phi_deg": case["phi_deg"],
        "psi_deg": case["psi_deg"],
        "ellipticity_deg": case["ellipticity_deg"],
        "sample_azimuth_deg": case["sample_azimuth_deg"],
        "polarimetry_label": case["polarimetry_label"],
        "orientation_index": case["orientation_index"],
        "inputs": {
            "wavelength_um": case["system"].wavelength_um,
            "thickness_um": [layer.thickness_um for layer in case["system"].layers[1:-1]],
            "orientation": complex_array_to_json(active.orientation.rotation_matrix()),
            "epsilon_omega_layer_crystal": complex_array_to_json(active.eps_w()),
            "epsilon_2omega_layer_crystal": complex_array_to_json(active.eps_2w()),
            "d_voigt_layer_crystal": complex_array_to_json(active.d_voigt()),
            "incident_jones_sp": complex_vector_to_json(np.asarray(incident_jones)),
            "analyzer_jones_sp": complex_vector_to_json(np.asarray(analyzer_jones)),
        },
        "outputs": {
            "fundamental_coefficients": complex_vector_to_json(result.fundamental.coefficients),
            "shg_coefficients": complex_vector_to_json(result.shg.coefficients),
            "reflected_2omega_jones_sp": complex_vector_to_json(np.asarray(reflected_jones)),
            "analyzer_amplitude": complex_scalar_to_json(analyzer_amplitude),
            "analyzer_intensity": round(float(analyzer_intensity), 15),
            "first_layer_source_wavevectors": complex_array_to_json(np.array([source.k for source in result.layer_sources[0].as_list()])),
            "first_layer_source_polarizations": complex_array_to_json(
                np.array([source.polarization for source in result.layer_sources[0].as_list()])
            ),
            "first_layer_inhomogeneous_electric": complex_array_to_json(np.array([field.electric for field in result.layer_inhomogeneous_fields[0].as_list()])),
        },
        "metrics": {
            "fundamental_residual_l2": round(float(np.linalg.norm(result.fundamental.residual)), 15),
            "shg_residual_l2": round(float(np.linalg.norm(result.shg.residual)), 15),
            "max_inhomogeneous_residual_l2": round(float(max(np.linalg.norm(field.residual) for field in all_fields)), 15),
            "source_count": len(result.layer_sources[0].as_list()),
            "max_abs_epsilon_omega_imag": round(float(np.max(np.abs(np.imag(active.eps_w())))), 15),
            "max_abs_epsilon_2omega_imag": round(float(np.max(np.abs(np.imag(active.eps_2w())))), 15),
            "max_abs_epsilon_omega_offdiag": round(float(max_abs_offdiag(active.eps_w())), 15),
            "max_abs_d_imag": round(float(np.max(np.abs(np.imag(active.d_voigt())))), 15),
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def fingerprint_record(record: dict) -> str:
    comparable = dict(record)
    comparable.pop("fingerprint", None)
    data = json.dumps(comparable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
