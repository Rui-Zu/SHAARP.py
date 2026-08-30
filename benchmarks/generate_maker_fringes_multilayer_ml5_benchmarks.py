"""Generate a 6-LAYER SHAARP.ml Maker benchmark grid (ml5) for depth.

ml1 (4-layer), ml2 (5-layer), and ml3 (two-active 4-layer) validated multilayer
breadth and the genuine two-active-source physics. ml5 extends the *depth* of the
single-active stack one layer further than ml2:
  air / active-film / ANISOTROPIC passive interlayer / isotropic interlayer-2 /
  isotropic interlayer-3 / substrate
(6 layers, 4 internal). This is ml2's 5-layer stack plus ONE MORE isotropic
passive interlayer. The two isotropic interlayers use slightly different
deterministic indices so they are genuinely distinct media (a third internal
interface beyond ml2). Only the film is SHG-active; all passive layers carry a
zero SHG d-tensor, so the active film remains the only 2omega source (no
multi-active-source coupling -- that is ml3's job). Incidence-swept.

Reuses ml2's _anisotropic_interlayer + _isotropic_interlayer helpers and the
existing Maker comparator + exporter (path-override wrapper); the Wolfram side
computes MFList/listMFpara/listMFperp from SHAARP.ml f4NL. The base exporter's
makeMaterial /@ case["layers"] and f4NL handle any layer count, so no exporter
changes are needed.
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

from benchmarks.generate_maker_fringes_multilayer_ml2_benchmarks import (  # noqa: E402
    _anisotropic_interlayer,
    _isotropic_interlayer,
)
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
TAG = "ml5"
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0

# (build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    (1, 0, 35.0, 40.0, 5.0),
    (4, 2, 20.0, 60.0, 18.0),
    (8, 0, 60.0, 50.0, 12.0),
    (15, 1, 55.0, 20.0, 28.0),
]


def _isotropic_interlayer_2(build_index: int) -> Layer:
    """A SECOND deterministic isotropic passive (zero-SHG) interlayer.

    Mirrors ml2's _isotropic_interlayer but with slightly different index offsets
    and a different thickness so this third internal medium is genuinely distinct
    from the first isotropic interlayer (no accidental degeneracy with it)."""
    n_w = 1.49 + 0.018 * (build_index % 5)
    n_2w = 1.61 + 0.012 * (build_index % 4)
    mat = Material(
        name="interlayer-iso-2",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3) * (n_w ** 2),
        epsilon_2omega=np.eye(3) * (n_2w ** 2 + 0.018j),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    thickness = 0.09 + 0.012 * (build_index % 5)
    return Layer("interlayer-iso-2", mat, thickness_um=thickness, shg_active=False)


def build_ml5_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, THETA_SWEEP_DEG[0], orientation_index)
        top, film, substrate = sys3.layers
        aniso = _anisotropic_interlayer(build_index)
        iso = _isotropic_interlayer(build_index)
        iso2 = _isotropic_interlayer_2(build_index)
        system = replace(sys3, layers=[top, film, aniso, iso, iso2, substrate])
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
                "layer_count": 6,
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
    parser = argparse.ArgumentParser(description="Generate 6-layer SHAARP.ml Maker benchmarks + Mathematica inputs (ml5).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ml_cases = build_ml5_cases(limit=args.limit)
    records = [run_case(c) for c in ml_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_multilayer_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "layer_structure": "air / active-film / anisotropic-interlayer / isotropic-interlayer-2 / isotropic-interlayer-3 / substrate (6 layers, 4 internal)",
        "notes": [
            "6-layer depth: ml2's 5-layer stack plus ONE MORE isotropic passive interlayer (two distinct isotropic interlayers + one anisotropic interlayer between film and substrate).",
            "Exercises a fourth internal layer and a third internal interface beyond the ml2 5-layer grid.",
            "All passive layers have zero SHG tensor; the active film is the only 2omega source.",
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
        "source": "Python explicit 6-layer Maker input manifest (ml5) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "6-layer stacks (air / active-film / anisotropic-interlayer / isotropic-interlayer-2 / isotropic-interlayer-3 / substrate).",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} ml5 (6-layer) cases to {args.output}")
    print(f"Wrote ml5 Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
