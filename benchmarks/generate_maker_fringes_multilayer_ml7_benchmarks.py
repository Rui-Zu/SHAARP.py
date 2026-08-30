"""Generate an 8-LAYER SHAARP.ml Maker benchmark grid (ml7) for depth.

ml1 (4-layer), ml2 (5-layer), ml3 (two-active 4-layer), ml5 (6-layer), and ml6
(7-layer) all live-Wolfram-validated the single-active multilayer stack at
increasing depth. ml7 extends that depth ONE MORE layer:
  air / active-film / ANISOTROPIC passive interlayer / isotropic interlayer /
  isotropic interlayer-2 / isotropic interlayer-3 / ANISOTROPIC passive
  interlayer-2 / substrate
(8 layers, 6 internal). This is ml6's 7-layer stack plus ONE MORE passive
interlayer (a SECOND anisotropic, non-degenerate interlayer placed just before
the substrate), giving five passive interlayers between film and substrate:
2 anisotropic + 3 isotropic, all genuinely distinct media. Only the film is
SHG-active; all passive layers carry a zero SHG d-tensor, so the active film
remains the only 2omega source (no multi-active-source coupling -- that is
ml3's job). Incidence-swept.

Reuses ml2's _anisotropic_interlayer + _isotropic_interlayer, ml5's
_isotropic_interlayer_2, and ml6's _isotropic_interlayer_3 helpers, plus the
existing Maker comparator + exporter (path-override wrapper); the Wolfram side
computes MFList/listMFpara/listMFperp from SHAARP.ml f4NL. The base exporter's
makeMaterial /@ case["layers"] and f4NL handle any layer count, so no exporter
changes are needed. The fifth passive interlayer (a second anisotropic medium)
uses distinct seed offsets and thickness so it does not degenerate with the
first anisotropic interlayer.
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
from benchmarks.generate_maker_fringes_multilayer_ml5_benchmarks import (  # noqa: E402
    _isotropic_interlayer_2,
)
from benchmarks.generate_maker_fringes_multilayer_ml6_benchmarks import (  # noqa: E402
    _isotropic_interlayer_3,
)
from benchmarks.generate_maker_mathematica_inputs import case_to_record  # noqa: E402
from benchmarks.generate_multilayer_system_benchmarks import (  # noqa: E402
    build_system,
    complex_array_to_json,
    complex_symmetric_epsilon,
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
TAG = "ml7"
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0

# (build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    (1, 0, 35.0, 40.0, 5.0),
    (4, 2, 20.0, 60.0, 18.0),
    (8, 0, 60.0, 50.0, 12.0),
    (15, 1, 55.0, 20.0, 28.0),
]


def _anisotropic_interlayer_2(build_index: int) -> Layer:
    """A SECOND deterministic ANISOTROPIC, NON-degenerate passive interlayer.

    Mirrors ml2's _anisotropic_interlayer (same complex_symmetric_epsilon
    generator, zero d-tensor) but with DIFFERENT seed offsets, distinct principal
    indices, and a different thickness so this sixth internal medium is genuinely
    distinct from the first anisotropic interlayer (no accidental degeneracy with
    it, and two distinct transverse modes within itself)."""
    eps_w = complex_symmetric_epsilon(
        build_index + 900, [1.71 ** 2, 1.86 ** 2, 2.04 ** 2], imag_base=0.017
    )
    eps_2w = complex_symmetric_epsilon(
        build_index + 1000, [2.9, 3.3, 3.8], imag_base=0.034
    )
    mat = Material(
        name="interlayer-aniso-2",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation(),
        epsilon_omega=eps_w,
        epsilon_2omega=eps_2w,
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    thickness = 0.15 + 0.011 * (build_index % 5)
    return Layer("interlayer-aniso-2", mat, thickness_um=thickness, shg_active=False)


def build_ml7_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, THETA_SWEEP_DEG[0], orientation_index)
        top, film, substrate = sys3.layers
        aniso = _anisotropic_interlayer(build_index)
        iso = _isotropic_interlayer(build_index)
        iso2 = _isotropic_interlayer_2(build_index)
        iso3 = _isotropic_interlayer_3(build_index)
        aniso2 = _anisotropic_interlayer_2(build_index)
        system = replace(
            sys3, layers=[top, film, aniso, iso, iso2, iso3, aniso2, substrate]
        )
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
                "layer_count": 8,
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
    parser = argparse.ArgumentParser(description="Generate 8-layer SHAARP.ml Maker benchmarks + Mathematica inputs (ml7).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ml_cases = build_ml7_cases(limit=args.limit)
    records = [run_case(c) for c in ml_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_multilayer_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "layer_structure": "air / active-film / anisotropic-interlayer / isotropic-interlayer / isotropic-interlayer-2 / isotropic-interlayer-3 / anisotropic-interlayer-2 / substrate (8 layers, 6 internal)",
        "notes": [
            "8-layer depth: ml6's 7-layer stack plus ONE MORE passive interlayer (a SECOND non-degenerate anisotropic interlayer just before the substrate), giving five distinct passive interlayers between film and substrate: 2 anisotropic + 3 isotropic.",
            "Exercises a sixth internal layer and a fifth internal interface beyond the ml6 7-layer grid.",
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
        "source": "Python explicit 8-layer Maker input manifest (ml7) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "8-layer stacks (air / active-film / anisotropic-interlayer / isotropic-interlayer / isotropic-interlayer-2 / isotropic-interlayer-3 / anisotropic-interlayer-2 / substrate).",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} ml7 (8-layer) cases to {args.output}")
    print(f"Wrote ml7 Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
