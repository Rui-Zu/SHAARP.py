"""Generate a TWO-ACTIVE-LAYER SHAARP.ml Maker benchmark grid (ml3).

ml1 (4-layer, 1 active + iso interlayer) and ml2 (5-layer, 1 active + aniso+iso
interlayers) both have exactly ONE SHG-active layer. ml3 validates the genuine
multilayer SHG physics nothing else covers: TWO active nonlinear films stacked
(air / active-film-1 / active-film-2 / substrate), so the transmitted 2omega
signal is the coherent superposition of TWO coupled nonlinear sources with
inter-layer propagation between them. Incidence-swept.

Both active films carry distinct complex SHG d-tensors (different build indices /
orientations). Reuses the existing Maker comparator + exporter (path-override
wrapper); the Wolfram side computes MFList from SHAARP.ml f4NL.
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
from shaarp.config import Layer, Polarimetry, with_sample_azimuth_deg  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    shaarp_ml_selected_transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
)

BENCHMARK_VERSION = 1
TAG = "ml3"
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0

# (build1, orient1, build2, orient2, phi, psi, ellipticity, film2_thickness_um)
RAW_CASES = [
    (4, 2, 9, 1, 40.0, 33.0, 10.0, 0.14),
    (1, 0, 8, 3, 55.0, 25.0, -12.0, 0.12),
    (11, 2, 15, 1, 30.0, 60.0, 18.0, 0.16),
    (18, 3, 5, 0, 48.0, 20.0, 22.0, 0.13),
]


def build_ml3_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for build1, orient1, build2, orient2, phi_deg, psi_deg, ellipticity_deg, film2_h in raw:
        s1 = build_system(build1, THETA_SWEEP_DEG[0], orient1)
        top, film1, substrate = s1.layers
        # second active film from a different build index/orientation
        film2_material = build_system(build2, THETA_SWEEP_DEG[0], orient2).layers[1].material
        film2 = Layer("active-film-2", film2_material, thickness_um=film2_h, shg_active=True)
        system = replace(s1, layers=[top, film1, film2, substrate])
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
        case_id = f"maker_fringes_{TAG}_b{build1}_{build2}_o{orient1}{orient2}"
        cases.append(
            {
                "id": case_id,
                "build_indices": [build1, build2],
                "orientation_indices": [orient1, orient2],
                "theta_deg": list(THETA_SWEEP_DEG),
                "phi_deg": phi_deg,
                "psi_deg": psi_deg,
                "ellipticity_deg": ellipticity_deg,
                "sample_azimuth_deg": SAMPLE_AZIMUTH_DEG,
                "phase_matching_like": False,
                "layer_count": 4,
                "active_layer_count": 2,
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
        "orientation_indices": case["orientation_indices"],
        "phi_deg": case["phi_deg"],
        "psi_deg": case["psi_deg"],
        "ellipticity_deg": case["ellipticity_deg"],
        "sample_azimuth_deg": case["sample_azimuth_deg"],
        "phase_matching_like": case["phase_matching_like"],
        "layer_count": case["layer_count"],
        "active_layer_count": case["active_layer_count"],
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
    parser = argparse.ArgumentParser(description="Generate two-active-layer SHAARP.ml Maker benchmarks + Mathematica inputs (ml3).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ml_cases = build_ml3_cases(limit=args.limit)
    records = [run_case(c) for c in ml_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_multilayer_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "layer_structure": "air / active-film-1 / active-film-2 / substrate (4 layers, TWO active SHG sources)",
        "notes": [
            "ml3 breadth: TWO stacked SHG-active films -> the transmitted 2omega signal is the coherent",
            "superposition of two coupled nonlinear sources with inter-layer propagation between them.",
            "This is the genuine multi-active-source multilayer physics ml1/ml2 (single active layer) do not cover.",
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
        "source": "Python explicit two-active-layer Maker input manifest (ml3) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "4-layer stacks with TWO SHG-active internal films (air / active-film-1 / active-film-2 / substrate).",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} ml3 (two-active-layer) cases to {args.output}")
    print(f"Wrote ml3 Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
