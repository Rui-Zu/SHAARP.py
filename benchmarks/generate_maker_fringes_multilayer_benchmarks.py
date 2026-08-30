"""Generate a genuine MULTILAYER SHAARP.ml Maker-fringe benchmark grid (ml1).

Prior end-to-end grids used a single active film (air / film / substrate). This
inserts an extra ISOTROPIC passive dielectric interlayer between the active film
and the substrate (air / active-film / interlayer / substrate), so the
transmitted 2omega SHG Maker fringe exercises SHAARP.ml `solveFresnelN` with two
internal layers and the inter-layer propagation phase Exp[i k.z * thickness]
across an extra interface -- physics the single-film cases do not stress.

The interlayer is isotropic so the (incidence-swept, non-rotated) Maker geometry
has no orientation-convention ambiguity for the passive layer; the new content
is purely the multilayer propagation + boundary solve.

Outputs:
  - Python benchmark JSON (list_mf_para/list_mf_perp etc.) -> --output
  - Mathematica input manifest JSON -> --mathematica-input
The Wolfram exporter (export_maker_fringes_reference.wl, driven via path
overrides) recomputes MFList/listMFpara/listMFperp from SHAARP.ml f4NL over the
4-layer stack; it must not copy Python values.
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
    complex_array_to_json,
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
    shaarp_ml_selected_transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
)


BENCHMARK_VERSION = 1
TAG = "ml1"
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0

# (build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    (1, 0, 35.0, 40.0, 5.0),
    (4, 2, 20.0, 60.0, 18.0),
    (8, 0, 60.0, 50.0, 12.0),
    (11, 2, 45.0, 35.0, 22.0),
    (15, 1, 55.0, 20.0, 28.0),
    (18, 3, 22.0, 48.0, 10.0),
]


def _isotropic_interlayer(build_index: int) -> Layer:
    """Deterministic isotropic, rotation-invariant passive dielectric interlayer."""
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


def build_multilayer_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, THETA_SWEEP_DEG[0], orientation_index)
        top, film, substrate = sys3.layers
        interlayer = _isotropic_interlayer(build_index)
        system = replace(sys3, layers=[top, film, interlayer, substrate])
        system = replace(
            system,
            polarimetry=Polarimetry(
                theta_deg=THETA_SWEEP_DEG[0],
                phi_deg=phi_deg,
                psi_deg=psi_deg,
                ellipticity_deg=ellipticity_deg,
            ),
        )
        system = with_sample_azimuth_deg(system, SAMPLE_AZIMUTH_DEG)
        case_id = f"maker_fringes_{TAG}_b{build_index}_o{orientation_index}"
        cases.append(
            {
                "id": case_id,
                "build_index": build_index,
                "orientation_index": orientation_index,
                "theta_deg": list(THETA_SWEEP_DEG),
                "phi_deg": phi_deg,
                "psi_deg": psi_deg,
                "ellipticity_deg": ellipticity_deg,
                "sample_azimuth_deg": SAMPLE_AZIMUTH_DEG,
                "phase_matching_like": False,
                "layer_count": 4,
                "system": system,
            }
        )
    return cases


def run_case(case: dict) -> dict:
    sweep = solve_multilayer_maker_fringes_sweep(
        case["system"], theta_deg=case["theta_deg"], mu=1.0, eps0=1.0
    )
    transmitted_jones = [
        shaarp_ml_selected_transmitted_2omega_jones_sp(result.shg) for result in sweep.results
    ]
    list_mf_para, list_mf_perp = sweep.shaarp_ml_copy_lists()
    return {
        "id": case["id"],
        "theta_deg": [float(t) for t in sweep.theta_deg],
        "angle_count": int(len(sweep.theta_deg)),
        "orientation_index": case["orientation_index"],
        "phi_deg": case["phi_deg"],
        "psi_deg": case["psi_deg"],
        "ellipticity_deg": case["ellipticity_deg"],
        "sample_azimuth_deg": case["sample_azimuth_deg"],
        "phase_matching_like": case["phase_matching_like"],
        "layer_count": case["layer_count"],
        "outputs": {
            "list_mf_para": complex_array_to_json(list_mf_para),
            "list_mf_perp": complex_array_to_json(list_mf_perp),
            "parallel_amplitude": complex_vector_to_json(sweep.parallel_amplitude),
            "perpendicular_amplitude": complex_vector_to_json(sweep.perpendicular_amplitude),
            "transmitted_2omega_jones_sp": [
                complex_vector_to_json(np.asarray(j)) for j in transmitted_jones
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multilayer SHAARP.ml Maker benchmarks + Mathematica inputs (ml1).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ml_cases = build_multilayer_cases(limit=args.limit)
    records = [run_case(c) for c in ml_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_multilayer_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "layer_structure": "air / active-film / isotropic-interlayer / substrate (4 layers, 2 internal)",
        "notes": [
            "Genuine multilayer: an extra isotropic passive dielectric interlayer between the active film and substrate.",
            "Exercises solveFresnelN with two internal layers and inter-layer propagation; single-film cases do not.",
            "Isotropic interlayer keeps the incidence-swept Maker geometry orientation-unambiguous.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_cases = [
        case_to_record(
            {
                "id": c["id"],
                "system": c["system"],
                "theta_deg": c["theta_deg"],
                "phi_deg": c["phi_deg"],
                "psi_deg": c["psi_deg"],
                "ellipticity_deg": c["ellipticity_deg"],
                "sample_azimuth_deg": c["sample_azimuth_deg"],
            }
        )
        for c in ml_cases
    ]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit multilayer Maker input manifest (ml1) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "4-layer stacks (air / active-film / isotropic-interlayer / substrate).",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} multilayer Maker cases to {args.output}")
    print(f"Wrote multilayer Maker Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
