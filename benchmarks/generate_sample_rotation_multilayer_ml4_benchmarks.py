"""Generate a MULTILAYER SHAARP.ml SampleRotate benchmark grid (ml4).

All prior multilayer end-to-end validation (ml1/ml2/ml3) was incidence-swept
Maker fringes. ml4 instead validates a genuine MULTILAYER stack under AZIMUTH
ROTATION -- the SampleRotate path -- which the single-film ext1 grid and the
incidence-swept ml1/ml2/ml3 grids do not exercise together.

Stack (4 layers, 2 internal):
  air / active-film / isotropic-passive-interlayer / substrate

The interlayer is ISOTROPIC (point group "1", scalar epsilon, zero d-tensor).
That is deliberate: under azimuth (sample) rotation a passive isotropic layer is
rotation-invariant, so its orientation convention stays unambiguous while the
active film is rotated about the sample normal. The new content is the
multilayer boundary solve + inter-layer propagation under the SampleRotate
azimuth sweep.

The azimuth-sweep machinery mirrors generate_sample_rotation_ext_benchmarks.py
exactly (solve_multilayer_shg_sample_azimuth_sweep with
inhomogeneous_solution_policy='solve', mu=1.0, eps0=1.0).

Outputs:
  - Python benchmark JSON (sample_rotate_list per case) -> --output
  - Mathematica input manifest JSON -> --mathematica-input

The Wolfram exporter (export_sample_rotation_reference_ml4.wl) reads the input
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
from shaarp.config import (  # noqa: E402
    CrystalOrientation,
    CrystalStructure,
    Layer,
    Material,
    Polarimetry,
    with_sample_azimuth_deg,
)
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    solve_multilayer_shg_sample_azimuth_sweep,
)


BENCHMARK_VERSION = 1
TAG = "ml4"
ROTATION_GRID_DEG = [0.0, 22.5, 45.0]
FIXED_THETA_DEG = 12.0
LAYER_COUNT = 4

# (build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
# Fixed modest incidence (theta ~ 12 deg) for all cases; the SampleRotate sweep
# rotates the sample azimuth. orientation_index spans all 4; build indices are
# deterministic seeds for epsilon/d tensors and thickness; all nonsingular.
RAW_CASES = [
    (1, 0, 35.0, 40.0, 5.0),
    (4, 1, 20.0, 60.0, 18.0),
    (8, 2, 60.0, 50.0, 12.0),
    (15, 3, 55.0, 20.0, 28.0),
]


def _isotropic_interlayer(build_index: int) -> Layer:
    """Deterministic isotropic, rotation-invariant passive dielectric interlayer.

    Mirrors ml1's _isotropic_interlayer: scalar (isotropic) epsilon, point group
    "1", zero d-tensor (passive). Rotation-invariant, so azimuth rotation leaves
    the passive layer's orientation unambiguous."""
    n_w = 1.55 + 0.02 * (build_index % 5)
    n_2w = 1.66 + 0.015 * (build_index % 4)
    eps_w = np.eye(3) * (n_w ** 2)
    eps_2w = np.eye(3) * (n_2w ** 2 + 0.02j)
    mat = Material(
        name="interlayer-iso",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation(),
        epsilon_omega=eps_w,
        epsilon_2omega=eps_2w,
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    thickness = 0.11 + 0.01 * (build_index % 6)
    return Layer("interlayer", mat, thickness_um=thickness, shg_active=False)


def _case_id(build_index: int, orientation_index: int) -> str:
    return f"sample_rotation_{TAG}_b{build_index}_o{orientation_index}"


def build_ml4_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, FIXED_THETA_DEG, orientation_index)
        top, film, substrate = sys3.layers
        interlayer = _isotropic_interlayer(build_index)
        system = replace(sys3, layers=[top, film, interlayer, substrate])
        system = replace(
            system,
            polarimetry=Polarimetry(
                theta_deg=FIXED_THETA_DEG,
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
                "theta_deg": [FIXED_THETA_DEG],
                "phi_deg": phi_deg,
                "psi_deg": psi_deg,
                "ellipticity_deg": ellipticity_deg,
                "sample_azimuth_deg": 0.0,
            }
        )
        record["theta_deg"] = FIXED_THETA_DEG
        record["sample_rotation_deg"] = ROTATION_GRID_DEG
        record["source_case_id"] = case_id
        cases.append(
            {
                "id": case_id,
                "source_case_id": case_id,
                "build_index": build_index,
                "orientation_index": orientation_index,
                "theta_deg": FIXED_THETA_DEG,
                "phi_deg": phi_deg,
                "psi_deg": psi_deg,
                "ellipticity_deg": ellipticity_deg,
                "sample_rotation_deg": ROTATION_GRID_DEG,
                "phase_matching_like": False,
                "layer_count": LAYER_COUNT,
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
        "layer_count": case["layer_count"],
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
        description="Generate multilayer SHAARP.ml SampleRotate Python benchmarks and Mathematica inputs (ml4)."
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
    parser.add_argument("--limit", type=int, default=None, help="Pilot subset: only the first N ml4 cases.")
    args = parser.parse_args()

    ml4_cases = build_ml4_cases(limit=args.limit)
    cases = [run_case(case) for case in ml4_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_sample_rotation_multilayer_benchmark_for_mathematica_reference",
        "case_count": len(cases),
        "rotation_grid_deg": ROTATION_GRID_DEG,
        "fixed_theta_deg": FIXED_THETA_DEG,
        "layer_structure": "air / active-film / isotropic-interlayer / substrate (4 layers, 2 internal)",
        "notes": [
            "Multilayer SampleRotate grid: a 4-layer stack with an isotropic passive interlayer, swept over sample azimuth at fixed modest incidence.",
            "All prior multilayer validation (ml1/ml2/ml3) was incidence-swept Maker fringes; ml4 is the azimuth-rotation (SampleRotate) analogue.",
            "All cases are nonsingular (no phase-matching-like epsilon match).",
            "Isotropic interlayer keeps azimuth rotation orientation-unambiguous for the passive layer.",
            "Python SampleRotate-shaped outputs: rotation angle, reflected parallel/perpendicular, transmitted parallel/perpendicular.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_cases = [case["mathematica_input"] for case in ml4_cases]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit multilayer SampleRotate input manifest (ml4) for Mathematica SHAARP.ml export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "rotation_grid_deg": ROTATION_GRID_DEG,
        "fixed_theta_deg": FIXED_THETA_DEG,
        "notes": [
            "4-layer stacks (air / active-film / isotropic-interlayer / substrate) swept over sample azimuth.",
            "For a uniform rotation grid, SHAARP.ml cumulative step rotation and Python absolute sample azimuth are equivalent.",
            "Mathematica exporter must compute SampleRotate outputs from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} ml4 multilayer SampleRotate cases to {args.output}")
    print(f"Wrote ml4 SampleRotate Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
