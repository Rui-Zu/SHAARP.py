"""Generate a POINT-GROUP-AT-DEPTH Maker-fringe benchmark grid (pgml).

This grid combines two physical axes that have each been validated separately but
NEVER together:

  * the crystallographic point-group axis (pg1/pg2): the active film carries a
    SYMMETRY-CONSTRAINED d-tensor (most components vanish by symmetry), and
  * the multilayer-depth axis (ml1-ml8): a passive isotropic dielectric interlayer
    is inserted between the active film and the substrate, so the transmitted
    2omega SHG fringe exercises solveFresnelN with two internal layers and an
    extra inter-layer propagation phase.

Construction: build the base 3-layer stack with build_system() (eps tensors /
orientation / substrate), REPLACE the active film's material with one carrying a
numeric symmetry-constrained d-tensor (substituted into
shaarp.symbolic.d_voigt_symbolic(pg)), then INSERT an isotropic passive interlayer
between the film and the substrate -> a 4-layer stack
[top, pg_film, interlayer, substrate]. All cases incidence-swept, nonsingular.

The novel coverage is the INTERACTION: a symmetry-constrained nonlinear source
(fewer than 18 nonzero d) driving the SHG boundary solve THROUGH an extra internal
layer. Both axes must be present in every case (layer_count == 4 AND
0 < nonzero_d < 18).

Outputs:
  - Python benchmark JSON (list_mf_para/list_mf_perp etc.) -> --output
  - Mathematica input manifest JSON -> --mathematica-input
The Wolfram exporter (export_maker_fringes_reference.wl, driven via path
overrides by export_maker_fringes_reference_pgml.wl) recomputes
MFList/listMFpara/listMFperp from SHAARP.ml f4NL over the 4-layer stack using the
manifest's explicit symmetry-constrained d_voigt_crystal; it must not copy Python
values.
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
TAG = "pgml"
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0

# (point_group, build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
# Four distinct symmetry-constrained patterns, each in a 4-layer stack.
RAW_CASES = [
    ("3m", 4, 2, 35.0, 40.0, 5.0),
    ("-42m", 8, 0, 50.0, 25.0, -15.0),
    ("32", 11, 2, 20.0, 60.0, 18.0),
    ("-6m2", 15, 1, 55.0, 20.0, 28.0),
]


def _pointgroup_d_voigt(point_group: str, seed: int) -> np.ndarray:
    """Numeric symmetry-constrained d-tensor: substitute deterministic complex
    values into the symbolic point-group template (the validated doldExp pattern).

    Replicated from generate_maker_fringes_pointgroup_pg1_benchmarks.py so the
    point-group axis here is built exactly as in the already-validated pg grids.
    """
    import sympy as sp

    from shaarp.symbolic import d_voigt_symbolic

    dsym = d_voigt_symbolic(point_group)
    syms = sorted(dsym.free_symbols, key=str)
    subs = {
        s: complex(0.30 + 0.07 * (seed % 5) + 0.05 * i, 0.04 * (i + 1))
        for i, s in enumerate(syms)
    }
    return np.array(
        [[complex(sp.N(dsym[r, c].subs(subs))) for c in range(6)] for r in range(3)],
        dtype=complex,
    )


def _isotropic_interlayer(build_index: int) -> Layer:
    """Deterministic isotropic, rotation-invariant passive dielectric interlayer.

    Replicated from generate_maker_fringes_multilayer_ml1_benchmarks.py so the
    depth axis here is built exactly as in the already-validated ml grids.
    """
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


def build_pgml_cases(limit: int | None = None) -> list[dict]:
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for point_group, build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, THETA_SWEEP_DEG[0], orientation_index)
        top, film, substrate = sys3.layers
        # AXIS 1 (point group): replace the active film's material with one carrying
        # a symmetry-constrained d-tensor.
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
        # AXIS 2 (depth): insert a passive isotropic interlayer between film and
        # substrate -> 4-layer stack.
        interlayer = _isotropic_interlayer(build_index)
        system = replace(sys3, layers=[top, pg_film, interlayer, substrate])
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
        case_id = f"maker_fringes_{TAG}_{point_group.replace('-', 'm')}_b{build_index}"
        nonzero_d = int(np.sum(np.abs(d_pg) > 1e-12))
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
                "layer_count": 4,
                "nonzero_d_entries": nonzero_d,
                "system": system,
            }
        )
    return cases


def run_case(case: dict) -> dict:
    sweep = solve_multilayer_maker_fringes_sweep(
        case["system"], theta_deg=case["theta_deg"], mu=1.0, eps0=1.0
    )
    transmitted_jones = [
        shaarp_ml_selected_transmitted_2omega_jones_sp(r.shg) for r in sweep.results
    ]
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
            "transmitted_2omega_jones_sp": [
                complex_vector_to_json(np.asarray(j)) for j in transmitted_jones
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate point-group-at-depth SHAARP.ml Maker benchmarks + Mathematica inputs (pgml)."
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pgml_cases = build_pgml_cases(limit=args.limit)
    records = [run_case(c) for c in pgml_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_pointgroup_at_depth_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "point_groups": [c["point_group"] for c in pgml_cases],
        "layer_structure": "air / point-group-active-film / isotropic-interlayer / substrate (4 layers, 2 internal)",
        "notes": [
            "Point-group-at-depth: the active film carries a SYMMETRY-CONSTRAINED d-tensor (3m/-42m/32/-6m2) AND sits in a 4-layer stack with a passive isotropic interlayer between it and the substrate.",
            "Combines the crystallographic point-group axis (pg1/pg2) with the multilayer-depth axis (ml1-ml8) -- never tested together before.",
            "Exercises the SHG boundary solve with a symmetry-constrained nonlinear source propagating through an extra internal layer vs live Wolfram f4NL.",
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
        for c in pgml_cases
    ]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit point-group-at-depth Maker input manifest (pgml) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "4-layer stacks (air / point-group-active-film / isotropic-interlayer / substrate).",
            "The active film (layer index 1) carries a symmetry-constrained point-group d-tensor (3m/-42m/32/-6m2) in its d_voigt_crystal.",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} pgml point-group-at-depth cases to {args.output}")
    print(f"Wrote pgml Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
