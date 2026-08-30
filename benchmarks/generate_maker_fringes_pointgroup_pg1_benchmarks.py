"""Generate a POINT-GROUP Maker-fringe benchmark grid (pg1).

Every prior end-to-end Maker case used point group "1" with a generic
18/18-nonzero d-tensor. SHAARP's purpose is crystallographic SHG, where most
d-components vanish by symmetry. This grid validates the full incidence-swept
Maker end-to-end pipeline driven by SYMMETRY-CONSTRAINED point-group d-tensors
(3m, -42m, mm2, 4mm) against live Wolfram f4NL -- a new physical axis, distinct
from the depth axis (ml1-ml8) and from the isolated symbolic doldExp validation.

Construction: reuse build_system() for the eps tensors / orientation / substrate
(a single active film, 3-layer stack), but REPLACE the active film's d_voigt with
a numeric symmetry-constrained tensor built by substituting deterministic complex
values into shaarp.symbolic.d_voigt_symbolic(pg). All cases incidence-swept,
nonsingular.
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
from shaarp.config import CrystalStructure, Material, Polarimetry, with_sample_azimuth_deg  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    shaarp_ml_selected_transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
)

BENCHMARK_VERSION = 1
TAG = "pg1"
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0

# (point_group, build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    ("3m", 4, 2, 35.0, 40.0, 5.0),
    ("-42m", 8, 0, 50.0, 25.0, -15.0),
    ("mm2", 11, 2, 20.0, 60.0, 18.0),
    ("4mm", 15, 1, 55.0, 20.0, 28.0),
]


def _pointgroup_d_voigt(point_group: str, seed: int) -> np.ndarray:
    """Numeric symmetry-constrained d-tensor: substitute deterministic complex
    values into the symbolic point-group template (the validated doldExp pattern)."""
    import sympy as sp

    from shaarp.symbolic import d_voigt_symbolic

    dsym = d_voigt_symbolic(point_group)
    syms = sorted(dsym.free_symbols, key=str)
    subs = {s: complex(0.30 + 0.07 * (seed % 5) + 0.05 * i, 0.04 * (i + 1)) for i, s in enumerate(syms)}
    return np.array(
        [[complex(sp.N(dsym[r, c].subs(subs))) for c in range(6)] for r in range(3)],
        dtype=complex,
    )


def build_pg1_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for point_group, build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, THETA_SWEEP_DEG[0], orientation_index)
        top, film, substrate = sys3.layers
        # replace the active film's material with one carrying a point-group d-tensor
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
                theta_deg=THETA_SWEEP_DEG[0],
                phi_deg=phi_deg,
                psi_deg=psi_deg,
                ellipticity_deg=ellipticity_deg,
            ),
        )
        system = with_sample_azimuth_deg(system, SAMPLE_AZIMUTH_DEG)
        case_id = f"maker_fringes_{TAG}_{point_group.replace('-','m')}_b{build_index}"
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


def run_case(case: dict) -> dict:
    sweep = solve_multilayer_maker_fringes_sweep(case["system"], theta_deg=case["theta_deg"], mu=1.0, eps0=1.0)
    transmitted_jones = [shaarp_ml_selected_transmitted_2omega_jones_sp(r.shg) for r in sweep.results]
    list_mf_para, list_mf_perp = sweep.shaarp_ml_copy_lists()
    return {
        "id": case["id"],
        "point_group": case["point_group"],
        "theta_deg": [float(t) for t in sweep.theta_deg],
        "angle_count": int(len(sweep.theta_deg)),
        "orientation_index": case["orientation_index"],
        "phi_deg": case["phi_deg"],
        "psi_deg": case["psi_deg"],
        "ellipticity_deg": case["ellipticity_deg"],
        "sample_azimuth_deg": case["sample_azimuth_deg"],
        "phase_matching_like": case["phase_matching_like"],
        "layer_count": case["layer_count"],
        "nonzero_d_entries": case["nonzero_d_entries"],
        "outputs": {
            "list_mf_para": complex_array_to_json(list_mf_para),
            "list_mf_perp": complex_array_to_json(list_mf_perp),
            "parallel_amplitude": complex_vector_to_json(sweep.parallel_amplitude),
            "perpendicular_amplitude": complex_vector_to_json(sweep.perpendicular_amplitude),
            "transmitted_2omega_jones_sp": [complex_vector_to_json(np.asarray(j)) for j in transmitted_jones],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate point-group SHAARP.ml Maker benchmarks + Mathematica inputs (pg1).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument("--mathematica-input", type=Path, default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pg_cases = build_pg1_cases(limit=args.limit)
    records = [run_case(c) for c in pg_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_pointgroup_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "point_groups": [c["point_group"] for c in pg_cases],
        "notes": [
            "Point-group SHG: the active film carries a SYMMETRY-CONSTRAINED d-tensor (3m/-42m/mm2/4mm), not a generic 18/18 tensor.",
            "Validates the full incidence-swept Maker end-to-end pipeline for real crystal classes vs live Wolfram f4NL.",
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
        "source": "Python explicit point-group Maker input manifest (pg1) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "Active films carry symmetry-constrained point-group d-tensors (3m/-42m/mm2/4mm).",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} pg1 point-group cases to {args.output}")
    print(f"Wrote pg1 Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
