"""Generate an EXTENDED SHAARP.ml SampleRotate benchmark grid (ext1).

This mirrors generate_sample_rotation_benchmarks.py but exercises a broader,
deterministic set of multilayer systems (every orientation index, many
build indices, varied polarimetry) so the SHAARP.ml end-to-end reflected +
transmitted 2omega SHG signal can be value-validated against live Mathematica
over a wider region of parameter space than the original 3 cases.

All cases are intentionally NONSINGULAR (no phase-matching-like epsilon match)
so each is a clean Mathematica-vs-Python agreement target.

Outputs:
  - Python benchmark JSON (sample_rotate_list per case) -> --output
  - Mathematica input manifest JSON -> --mathematica-input

The Wolfram exporter (export_sample_rotation_reference_ext.wl) reads the input
manifest and computes the reference sampleRotateList from SHAARP.ml. It must
never copy these Python output values.
"""

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

from benchmarks.generate_maker_mathematica_inputs import case_to_record  # noqa: E402
from benchmarks.generate_multilayer_system_benchmarks import (  # noqa: E402
    build_system,
    complex_vector_to_json,
)
from shaarp.config import Polarimetry, with_sample_azimuth_deg  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    solve_multilayer_shg_sample_azimuth_sweep,
)


BENCHMARK_VERSION = 1
TAG = "ext1"
ROTATION_GRID_DEG = [0.0, 22.5, 45.0]

# (build_index, orientation_index, theta_deg, phi_deg, psi_deg, ellipticity_deg)
# orientation_index spans all 4 (ORIENTATION_COUNT == 4); build indices are
# deterministic seeds for epsilon/d tensors and thickness; all nonsingular.
EXT_RAW_CASES = [
    (1, 0, 0.0, 35.0, 40.0, 5.0),
    (2, 1, 12.0, 50.0, 25.0, -15.0),
    (4, 2, 25.0, 20.0, 60.0, 18.0),
    (5, 3, -10.0, 75.0, 15.0, -8.0),
    (8, 0, 35.0, 60.0, 50.0, 12.0),
    (9, 1, 18.0, 28.0, 70.0, -20.0),
    (11, 2, -22.0, 45.0, 35.0, 22.0),
    (12, 3, 30.0, 15.0, 80.0, -5.0),
    (15, 0, 8.0, 55.0, 20.0, 28.0),
    (16, 1, -16.0, 38.0, 62.0, -30.0),
    (18, 2, 42.0, 22.0, 48.0, 10.0),
    (19, 3, 5.0, 68.0, 30.0, -18.0),
]


def _case_id(build_index: int, orientation_index: int) -> str:
    return f"sample_rotation_{TAG}_b{build_index}_o{orientation_index}"


def build_ext_cases(limit: int | None = None) -> list[dict]:
    raw = EXT_RAW_CASES if limit is None else EXT_RAW_CASES[:limit]
    cases = []
    for build_index, orientation_index, theta_deg, phi_deg, psi_deg, ellipticity_deg in raw:
        system = build_system(build_index, theta_deg, orientation_index)
        system = replace(
            system,
            polarimetry=Polarimetry(
                theta_deg=theta_deg,
                phi_deg=phi_deg,
                psi_deg=psi_deg,
                ellipticity_deg=ellipticity_deg,
            ),
        )
        # rotation grid is applied by the sweep; start sample azimuth at 0.
        system = with_sample_azimuth_deg(system, 0.0)
        case_id = _case_id(build_index, orientation_index)
        record = case_to_record(
            {
                "id": case_id,
                "system": system,
                "theta_deg": [theta_deg],
                "phi_deg": phi_deg,
                "psi_deg": psi_deg,
                "ellipticity_deg": ellipticity_deg,
                "sample_azimuth_deg": 0.0,
            }
        )
        record["theta_deg"] = theta_deg
        record["sample_rotation_deg"] = ROTATION_GRID_DEG
        record["source_case_id"] = case_id
        cases.append(
            {
                "id": case_id,
                "source_case_id": case_id,
                "build_index": build_index,
                "orientation_index": orientation_index,
                "theta_deg": theta_deg,
                "phi_deg": phi_deg,
                "psi_deg": psi_deg,
                "ellipticity_deg": ellipticity_deg,
                "sample_rotation_deg": ROTATION_GRID_DEG,
                "phase_matching_like": False,
                "system": system,
                "mathematica_input": record,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate extended SHAARP.ml SampleRotate Python benchmarks and Mathematica inputs (ext1)."
    )
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--output",
        type=Path,
        default=here / f"sample_rotation_benchmarks_{TAG}.json",
    )
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"sample_rotation_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="Pilot subset: only the first N ext cases.")
    args = parser.parse_args()

    ext_cases = build_ext_cases(limit=args.limit)
    cases = [run_case(case) for case in ext_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_sample_rotation_ext_benchmark_for_mathematica_reference",
        "case_count": len(cases),
        "rotation_grid_deg": ROTATION_GRID_DEG,
        "notes": [
            "Extended SampleRotate grid: every orientation index, many build indices, varied polarimetry.",
            "All cases are nonsingular (no phase-matching-like epsilon match).",
            "Python SampleRotate-shaped outputs: rotation angle, reflected parallel/perpendicular, transmitted parallel/perpendicular.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_cases = [case["mathematica_input"] for case in ext_cases]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit SampleRotate ext input manifest for Mathematica SHAARP.ml export",
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
    print(f"Wrote {len(cases)} ext SampleRotate cases to {args.output}")
    print(f"Wrote ext SampleRotate Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
