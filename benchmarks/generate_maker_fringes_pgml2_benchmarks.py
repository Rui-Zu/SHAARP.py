"""Generate point-group-at-depth Maker-fringe benchmarks for the REMAINING distinct
SHG d-patterns (pgml2) -- bringing point-group-at-depth coverage to FULL PARITY.

pgml validated point-group-at-depth for 4 patterns (3m/-42m/32/-6m2). The
transmitted single-film Maker path (pg1+pg2+pg3) validated all 13 distinct
buildable patterns, but only 4 of them have been validated AT DEPTH (active film
embedded in a 4-layer stack with a passive isotropic interlayer). pgml2 closes the
remaining 10 distinct representatives -- mm2, 4mm, 2, m, -4, 422, 3, -6, 4, 6 --
so that EVERY distinct buildable symmetry-constrained d-pattern is validated
point-group-AT-DEPTH, matching the single-film and SI-reflected coverage.

It reuses the EXACT pgml machinery verbatim (``_pointgroup_d_voigt``,
``_isotropic_interlayer``, ``run_case`` from the pgml generator; the same
``build_system`` base stack and ``case_to_record`` input manifest). The ONLY
differences are TAG="pgml2" and RAW_CASES (the 10 remaining patterns). Like the
pgml grid, it writes the Python benchmark + the Mathematica input manifest, and it
additionally mirrors the manifest into the ORIGINAL-tree directory (the Wolfram
exporter reads ORIGINAL-tree paths).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace  # noqa: E402

from benchmarks.generate_maker_mathematica_inputs import case_to_record  # noqa: E402
# Reuse pgml's validated helpers verbatim so the only divergence is the case set.
from benchmarks.generate_maker_fringes_pgml_benchmarks import (  # noqa: E402
    THETA_SWEEP_DEG,
    SAMPLE_AZIMUTH_DEG,
    _isotropic_interlayer,
    _pointgroup_d_voigt,
    run_case,
)

from benchmarks.generate_multilayer_system_benchmarks import build_system  # noqa: E402
from shaarp.config import CrystalStructure, Material, Polarimetry, with_sample_azimuth_deg  # noqa: E402

BENCHMARK_VERSION = 1
TAG = "pgml2"
# Optional second working tree to mirror generated references into, e.g. a checkout of the
# original Mathematica-side project. Unset by default: without it this script writes only inside
# its own repository.
ORIGINAL_REF_DIR = (
    Path(os.environ["SHAARP_ORIGINAL_REF_DIR"])
    if os.environ.get("SHAARP_ORIGINAL_REF_DIR")
    else None
)

# The 10 distinct symmetry-constrained patterns NOT in pgml, each in a 4-layer stack.
# (point_group, build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    ("mm2", 0, 0, 30.0, 35.0, 8.0),
    ("4mm", 1, 1, 45.0, 22.0, -12.0),
    ("2", 2, 2, 18.0, 55.0, 16.0),
    ("m", 3, 3, 52.0, 18.0, 24.0),
    ("-4", 5, 4, 27.0, 48.0, -6.0),
    ("422", 6, 5, 40.0, 30.0, 11.0),
    ("3", 7, 6, 33.0, 41.0, -18.0),
    ("-6", 9, 7, 49.0, 26.0, 14.0),
    ("4", 10, 0, 22.0, 52.0, -9.0),
    ("6", 12, 1, 57.0, 19.0, 20.0),
]


def build_pgml2_cases(limit: int | None = None) -> list[dict]:
    """Replicate pgml's build_pgml_cases body verbatim, driven by our RAW_CASES."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate point-group-at-depth SHAARP.ml Maker benchmarks for the remaining 10 distinct patterns (pgml2)."
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument(
        "--no-original-tree",
        action="store_true",
        help="Skip mirroring the Mathematica input manifest into the ORIGINAL-tree directory.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pgml2_cases = build_pgml2_cases(limit=args.limit)
    records = [run_case(c) for c in pgml2_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_pointgroup_at_depth_maker_fringes_benchmark_for_mathematica_reference",
        "extension_of": "benchmarks/maker_fringes_benchmarks_pgml.json",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "point_groups": [c["point_group"] for c in pgml2_cases],
        "layer_structure": "air / point-group-active-film / isotropic-interlayer / substrate (4 layers, 2 internal)",
        "notes": [
            "Point-group-at-depth for the 10 distinct patterns NOT in pgml (mm2/4mm/2/m/-4/422/3/-6/4/6).",
            "Together with pgml (3m/-42m/32/-6m2), point-group-at-depth now covers every distinct buildable SHG d-pattern -- full parity with single-film (pg1+pg2+pg3) and SI-reflected (sipg1+sipg2) coverage.",
            "Each case: a SYMMETRY-CONSTRAINED point-group d film (fewer than 18 nonzero d) inside a 4-layer stack with a passive isotropic interlayer, vs live Wolfram f4NL.",
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
        for c in pgml2_cases
    ]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit point-group-at-depth Maker input manifest (pgml2: remaining 10 patterns) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "4-layer stacks (air / point-group-active-film / isotropic-interlayer / substrate).",
            "The active film (layer index 1) carries a symmetry-constrained point-group d-tensor (mm2/4mm/2/m/-4/422/3/-6/4/6) in its d_voigt_crystal.",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
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
        print(f"Mirrored pgml2 Mathematica inputs to ORIGINAL tree: {orig_input_path}")

    print(f"Wrote {len(records)} pgml2 point-group-at-depth cases to {args.output}")
    print(f"Wrote pgml2 Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
