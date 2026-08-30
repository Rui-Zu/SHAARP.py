"""Generate a POINT-GROUP TWO-ACTIVE-LAYER Maker-fringe benchmark grid (pg2act).

This validates an intersection never tested before: the COUPLED two-active-source
physics of ml3 (air / active-film-1 / active-film-2 / substrate, the transmitted
2omega signal being the coherent superposition of TWO adjacent nonlinear sources)
where BOTH active films carry a SYMMETRY-CONSTRAINED point-group d-tensor.

  * ml3 validated two coupled active sources with GENERIC 18/18 d.
  * pg1/pg2/pg3 validated each point-group d-pattern in a single film.
  * pgml/pgml2 validated point-group d AT DEPTH (one constrained film + passive
    interlayer).
  * NEVER validated: two ADJACENT, COUPLED active films EACH carrying a
    symmetry-constrained point-group d -- the point-group x coupled-source
    intersection. pg2act closes it.

Construction mirrors ml3 exactly (two active films from different build indices /
orientations, stacked with no passive spacer), except each active film's material
is replaced with one carrying a point-group-constrained d-tensor built by the
validated ``_pointgroup_d_voigt`` substitution (the same routine pg1/pgml use).
Both axes must be present in every case: active_layer_count == 2 AND each film's
crystal d has 0 < nonzero < 18.

Reuses the existing Maker comparator + the base Wolfram exporter (driven via a
path-override wrapper); the Wolfram side recomputes MFList/listMFpara/listMFperp
from SHAARP.ml f4NL over the explicit per-layer d_voigt_crystal -- it must not copy
Python values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_maker_mathematica_inputs import case_to_record  # noqa: E402
from benchmarks.generate_maker_fringes_pgml_benchmarks import _pointgroup_d_voigt  # noqa: E402
from benchmarks.generate_multilayer_system_benchmarks import (  # noqa: E402
    build_system,
    complex_array_to_json,
    complex_vector_to_json,
)
from shaarp.config import (  # noqa: E402
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
TAG = "pg2act"
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0
# Optional second working tree to mirror generated references into, e.g. a checkout of the
# original Mathematica-side project. Unset by default: without it this script writes only inside
# its own repository.
ORIGINAL_REF_DIR = (
    Path(os.environ["SHAARP_ORIGINAL_REF_DIR"])
    if os.environ.get("SHAARP_ORIGINAL_REF_DIR")
    else None
)

# (pg1, build1, orient1, pg2, build2, orient2, phi, psi, ellipticity, film2_thickness_um)
# Diverse point-group PAIRS spanning several distinct patterns; build indices reuse
# ml3's known-valid set.
RAW_CASES = [
    ("3m", 4, 2, "-42m", 9, 1, 40.0, 33.0, 10.0, 0.14),
    ("4mm", 1, 0, "mm2", 8, 3, 55.0, 25.0, -12.0, 0.12),
    ("32", 11, 2, "-6m2", 15, 1, 30.0, 60.0, 18.0, 0.16),
    ("-4", 18, 3, "422", 5, 0, 48.0, 20.0, 22.0, 0.13),
]


def _pointgroup_film(base_film: Layer, point_group: str, seed: int, *, name: str, thickness_um=None) -> tuple[Layer, int]:
    """Replace a base active film's material with one carrying a symmetry-constrained
    point-group d-tensor (keeping its eps tensors / orientation), exactly as pgml does."""
    d_pg = _pointgroup_d_voigt(point_group, seed)
    pg_material = Material(
        name=name,
        structure=CrystalStructure(point_group=point_group),
        orientation=base_film.material.orientation,
        epsilon_omega=base_film.material.eps_w(),
        epsilon_2omega=base_film.material.eps_2w(),
        d_voigt_pm_v=d_pg,
    )
    nz = int(np.sum(np.abs(d_pg) > 1e-12))
    if thickness_um is None:
        film = replace(base_film, material=pg_material)
    else:
        film = Layer(name, pg_material, thickness_um=thickness_um, shg_active=True)
    return film, nz


def build_pg2act_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for pg1, build1, orient1, pg2, build2, orient2, phi_deg, psi_deg, ellipticity_deg, film2_h in raw:
        s1 = build_system(build1, THETA_SWEEP_DEG[0], orient1)
        top, film1_base, substrate = s1.layers
        film1, nz1 = _pointgroup_film(film1_base, pg1, build1, name=f"active-film-1-{pg1}")
        film2_base = build_system(build2, THETA_SWEEP_DEG[0], orient2).layers[1]
        film2, nz2 = _pointgroup_film(film2_base, pg2, build2, name=f"active-film-2-{pg2}", thickness_um=film2_h)
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
        case_id = f"maker_fringes_{TAG}_{pg1.replace('-', 'm')}_{pg2.replace('-', 'm')}_b{build1}_{build2}"
        cases.append(
            {
                "id": case_id,
                "point_groups": [pg1, pg2],
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
                "nonzero_d_entries": [nz1, nz2],
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
        "point_groups": case["point_groups"],
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
        "nonzero_d_entries": case["nonzero_d_entries"],
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
    parser = argparse.ArgumentParser(
        description="Generate point-group two-active-layer SHAARP.ml Maker benchmarks + Mathematica inputs (pg2act)."
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--no-original-tree", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pg2act_cases = build_pg2act_cases(limit=args.limit)
    records = [run_case(c) for c in pg2act_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_pointgroup_two_active_maker_fringes_benchmark_for_mathematica_reference",
        "extension_of": "benchmarks/maker_fringes_benchmarks_ml3.json",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "point_group_pairs": [c["point_groups"] for c in pg2act_cases],
        "layer_structure": "air / point-group-active-film-1 / point-group-active-film-2 / substrate (4 layers, 2 coupled active sources)",
        "notes": [
            "Point-group x coupled-source intersection: TWO adjacent active films, EACH carrying a symmetry-constrained point-group d-tensor.",
            "Generalizes ml3 (two coupled active sources, generic d) and pgml (one point-group film at depth) to two coupled point-group sources.",
            "The transmitted 2omega signal is the coherent superposition of two symmetry-constrained nonlinear sources vs live Wolfram f4NL.",
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
        for c in pg2act_cases
    ]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit point-group two-active-layer Maker input manifest (pg2act) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "4-layer stacks (air / point-group-active-film-1 / point-group-active-film-2 / substrate); BOTH internal layers SHG-active.",
            "Each active film carries a symmetry-constrained point-group d-tensor in its d_voigt_crystal.",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml f4NL; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")

    if not args.no_original_tree and ORIGINAL_REF_DIR is None:
        print("SHAARP_ORIGINAL_REF_DIR is not set -- skipping the second-tree mirror copy.")
    elif not args.no_original_tree:
        ORIGINAL_REF_DIR.mkdir(parents=True, exist_ok=True)
        orig_input_path = ORIGINAL_REF_DIR / f"maker_mathematica_inputs_{TAG}.json"
        orig_input_path.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
        print(f"Mirrored pg2act Mathematica inputs to ORIGINAL tree: {orig_input_path}")

    print(f"Wrote {len(records)} pg2act point-group two-active cases to {args.output}")
    print(f"Wrote pg2act Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
