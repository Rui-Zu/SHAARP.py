from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_maker_fringes_benchmarks import build_cases  # noqa: E402
from benchmarks.generate_maker_mathematica_inputs import case_to_record  # noqa: E402
from benchmarks.generate_multilayer_system_benchmarks import complex_vector_to_json  # noqa: E402
from shaarp.config import Polarimetry  # noqa: E402
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_sample_azimuth_sweep  # noqa: E402


BENCHMARK_VERSION = 1
ROTATION_GRID_DEG = [0.0, 15.0, 30.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.ml SampleRotate Python benchmarks and Mathematica inputs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "sample_rotation_benchmarks_v1.json",
    )
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=Path(__file__).resolve().parent / "mathematica_reference" / "sample_rotation_mathematica_inputs_v1.json",
    )
    args = parser.parse_args()

    cases = [run_case(case) for case in build_sample_rotation_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_sample_rotation_benchmark_for_mathematica_reference",
        "case_count": len(cases),
        "notes": [
            "Python SampleRotate-shaped outputs: rotation angle, reflected parallel/perpendicular, transmitted parallel/perpendicular.",
            "The reflected analyzer convention follows SHAARP.ml SampleRotate, where reflected s enters the parallel channel with a minus sign.",
            "The default inhomogeneous source policy is forward_only to match SHAARP.ml f4NL SampleRotate/Maker behavior.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_cases = [case["mathematica_input"] for case in build_sample_rotation_cases()]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "source": "Python explicit SampleRotate input manifest for Mathematica SHAARP.ml export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "rotation_grid_deg": ROTATION_GRID_DEG,
        "notes": [
            "For a uniform rotation grid, SHAARP.ml cumulative step rotation and Python absolute sample azimuth are equivalent.",
            "Mathematica exporter must compute SampleRotate outputs from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} SampleRotate cases to {args.output}")
    print(f"Wrote SampleRotate Mathematica inputs to {args.mathematica_input}")


def build_sample_rotation_cases() -> list[dict]:
    cases = []
    for maker_case in build_cases():
        theta_values = list(maker_case["theta_deg"])
        theta_deg = float(theta_values[len(theta_values) // 2])
        system = replace(
            maker_case["system"],
            polarimetry=replace(maker_case["system"].polarimetry, theta_deg=theta_deg),
        )
        record = case_to_record({**maker_case, "system": system, "theta_deg": [theta_deg]})
        record["theta_deg"] = theta_deg
        record["sample_rotation_deg"] = ROTATION_GRID_DEG
        cases.append(
            {
                "id": maker_case["id"].replace("maker_fringes", "sample_rotation"),
                "source_case_id": maker_case["id"],
                "theta_deg": theta_deg,
                "phi_deg": float(maker_case["phi_deg"]),
                "psi_deg": float(maker_case["psi_deg"]),
                "ellipticity_deg": float(maker_case["ellipticity_deg"]),
                "sample_rotation_deg": ROTATION_GRID_DEG,
                "phase_matching_like": bool(maker_case["phase_matching_like"]),
                "system": system,
                "mathematica_input": {
                    **record,
                    "id": maker_case["id"].replace("maker_fringes", "sample_rotation"),
                    "source_case_id": maker_case["id"],
                },
            }
        )
    return cases


def run_case(case: dict) -> dict:
    sweep = solve_multilayer_shg_sample_azimuth_sweep(
        case["system"],
        sample_azimuth_deg=np.asarray(case["sample_rotation_deg"], dtype=float),
        mu=1.0,
        eps0=1.0,
        inhomogeneous_solution_policy="solve",
    )
    return {
        "id": case["id"],
        "source_case_id": case["source_case_id"],
        "theta_deg": case["theta_deg"],
        "phi_deg": case["phi_deg"],
        "psi_deg": case["psi_deg"],
        "ellipticity_deg": case["ellipticity_deg"],
        "sample_rotation_deg": case["sample_rotation_deg"],
        "phase_matching_like": case["phase_matching_like"],
        "outputs": {
            "sample_rotate_list": _sample_rotate_list(sweep).tolist(),
            "reflected_parallel_amplitude": complex_vector_to_json(sweep.reflected_parallel_amplitude),
            "reflected_perpendicular_amplitude": complex_vector_to_json(sweep.reflected_perpendicular_amplitude),
            "transmitted_parallel_amplitude": complex_vector_to_json(sweep.transmitted_parallel_amplitude),
            "transmitted_perpendicular_amplitude": complex_vector_to_json(sweep.transmitted_perpendicular_amplitude),
        },
    }


def _sample_rotate_list(sweep) -> np.ndarray:
    return np.column_stack(
        [
            sweep.sample_azimuth_deg,
            sweep.reflected_parallel_intensity,
            sweep.reflected_perpendicular_intensity,
            sweep.transmitted_parallel_intensity,
            sweep.transmitted_perpendicular_intensity,
        ]
    )


if __name__ == "__main__":
    main()
