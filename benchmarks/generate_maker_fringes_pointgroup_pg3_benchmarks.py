"""Generate point-group Maker-fringe benchmark pg3 — closing the last distinct
SHG d-pattern not yet validated end-to-end.

Completeness check (iter21): of the 14 distinct symmetry-constrained d-patterns
among the SHG-active crystallographic point groups, pg1+pg2 covered 11 (+ generic
'1' via the ml depth grids). The one remaining BUILDABLE distinct pattern is the
'4'/'6' class (nonzero at (0,3),(0,4),(1,3),(1,4),(2,0),(2,1),(2,2) -- differs
from '-4' only at (2,2) vs (2,5)). pg3 validates it end-to-end with both '4'
(tetragonal) and '6' (hexagonal), which share this pattern.

NOTE (documented limitation, not built here): point group '432' is SHG-active by
symmetry but its d-tensor vanishes entirely; shaarp.symbolic.d_voigt_symbolic
raises Unsupported for '432' (the all-zero template is not implemented), so it is
not a portable end-to-end target via this path.

Reuses pg1's _pointgroup_d_voigt / run_case / build pattern verbatim.
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

from benchmarks.generate_maker_fringes_pointgroup_pg1_benchmarks import (  # noqa: E402
    THETA_SWEEP_DEG,
    SAMPLE_AZIMUTH_DEG,
    _pointgroup_d_voigt,
    run_case,
)
from benchmarks.generate_maker_mathematica_inputs import case_to_record  # noqa: E402
from benchmarks.generate_multilayer_system_benchmarks import build_system  # noqa: E402
from shaarp.config import CrystalStructure, Material, Polarimetry, with_sample_azimuth_deg  # noqa: E402

BENCHMARK_VERSION = 1
TAG = "pg3"

# (point_group, build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    ("4", 4, 2, 35.0, 40.0, 5.0),
    ("6", 11, 1, 50.0, 25.0, -15.0),
]


def build_pg3_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for point_group, build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, THETA_SWEEP_DEG[0], orientation_index)
        top, film, substrate = sys3.layers
        d_pg = _pointgroup_d_voigt(point_group, build_index)
        pg_material = Material(
            name=f"active-film-{point_group}",
            structure=CrystalStructure(point_group=point_group),
            orientation=film.material.orientation,
            epsilon_omega=film.material.eps_w(),
            epsilon_2omega=film.material.eps_2w(),
            d_voigt_pm_v=d_pg,
        )
        pg_film = replace(film, material=pg_material)
        system = replace(sys3, layers=[top, pg_film, substrate])
        system = replace(
            system,
            polarimetry=Polarimetry(
                theta_deg=THETA_SWEEP_DEG[0], phi_deg=phi_deg, psi_deg=psi_deg, ellipticity_deg=ellipticity_deg
            ),
        )
        system = with_sample_azimuth_deg(system, SAMPLE_AZIMUTH_DEG)
        case_id = f"maker_fringes_{TAG}_pg{point_group}_b{build_index}"
        cases.append(
            {
                "id": case_id,
                "point_group": point_group,
                "build_index": build_index,
                "orientation_index": orientation_index,
                "theta_deg": list(THETA_SWEEP_DEG),
                "phi_deg": phi_deg,
                "psi_deg": psi_deg,
                "ellipticity_deg": ellipticity_deg,
                "sample_azimuth_deg": SAMPLE_AZIMUTH_DEG,
                "phase_matching_like": False,
                "layer_count": 3,
                "nonzero_d_entries": int(np.sum(np.abs(d_pg) > 1e-12)),
                "system": system,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate point-group SHAARP.ml Maker benchmarks (pg3: 4/6 class).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument("--mathematica-input", type=Path, default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pg_cases = build_pg3_cases(limit=args.limit)
    records = [run_case(c) for c in pg_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_pointgroup_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "point_groups": [c["point_group"] for c in pg_cases],
        "notes": [
            "Closes the last distinct buildable SHG d-pattern: the '4'/'6' class (differs from '-4' at one position).",
            "Validates the full incidence-swept Maker end-to-end pipeline for the 4/6 crystal classes vs live Wolfram f4NL.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_cases = [
        case_to_record({
            "id": c["id"], "system": c["system"], "theta_deg": c["theta_deg"],
            "phi_deg": c["phi_deg"], "psi_deg": c["psi_deg"],
            "ellipticity_deg": c["ellipticity_deg"], "sample_azimuth_deg": c["sample_azimuth_deg"],
        })
        for c in pg_cases
    ]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit point-group Maker input manifest (pg3: 4/6 class) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "Active films carry the symmetry-constrained '4'/'6' d-pattern.",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} pg3 point-group cases to {args.output}")
    print(f"Wrote pg3 Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
