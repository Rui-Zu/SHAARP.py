"""Generate an EXTENDED SHAARP.ml Maker-fringe (transmitted 2omega SHG) grid (ext1).

Complements the SampleRotate ext1 grid: SampleRotate sweeps azimuth at fixed
incidence, while this sweeps INCIDENCE ANGLE (theta) at fixed azimuth, so it
validates the theta-dependence of the transmitted SHG that SampleRotate does
not cover. It reuses the same 12 deterministic, diverse, nonsingular multilayer
systems (build_index x orientation_index x polarimetry) defined for the
SampleRotate ext grid.

Outputs:
  - Python benchmark JSON (list_mf_para/list_mf_perp etc.) -> --output
  - Mathematica input manifest JSON -> --mathematica-input

The Wolfram exporter (export_maker_fringes_reference.wl, driven via
export_maker_fringes_reference_ext.wl path overrides) recomputes
MFList/listMFpara/listMFperp from SHAARP.ml; it must never copy Python values.
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
from benchmarks.generate_sample_rotation_ext_benchmarks import EXT_RAW_CASES  # noqa: E402
from shaarp.config import Polarimetry, with_sample_azimuth_deg  # noqa: E402
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    shaarp_ml_selected_transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
)


BENCHMARK_VERSION = 1
TAG = "ext1"
# Common incidence-angle sweep for every case (distinct angles per case would
# also be valid; a shared grid exercises theta-dependence uniformly).
THETA_SWEEP_DEG = [0.0, 20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0


def _case_id(build_index: int, orientation_index: int) -> str:
    return f"maker_fringes_{TAG}_b{build_index}_o{orientation_index}"


def build_ext_maker_cases(limit: int | None = None) -> list[dict]:
    raw = EXT_RAW_CASES if limit is None else EXT_RAW_CASES[:limit]
    cases = []
    for build_index, orientation_index, _theta_seed, phi_deg, psi_deg, ellipticity_deg in raw:
        theta0 = THETA_SWEEP_DEG[0]
        system = build_system(build_index, theta0, orientation_index)
        system = replace(
            system,
            polarimetry=Polarimetry(
                theta_deg=theta0,
                phi_deg=phi_deg,
                psi_deg=psi_deg,
                ellipticity_deg=ellipticity_deg,
            ),
        )
        system = with_sample_azimuth_deg(system, SAMPLE_AZIMUTH_DEG)
        case_id = _case_id(build_index, orientation_index)
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
                "system": system,
            }
        )
    return cases


def run_case(case: dict) -> dict:
    sweep = solve_multilayer_maker_fringes_sweep(
        case["system"],
        theta_deg=case["theta_deg"],
        mu=1.0,
        eps0=1.0,
    )
    transmitted_jones = [
        shaarp_ml_selected_transmitted_2omega_jones_sp(result.shg) for result in sweep.results
    ]
    list_mf_para, list_mf_perp = sweep.shaarp_ml_copy_lists()
    return {
        "id": case["id"],
        "theta_deg": [float(theta) for theta in sweep.theta_deg],
        "angle_count": int(len(sweep.theta_deg)),
        "orientation_index": case["orientation_index"],
        "phi_deg": case["phi_deg"],
        "psi_deg": case["psi_deg"],
        "ellipticity_deg": case["ellipticity_deg"],
        "sample_azimuth_deg": case["sample_azimuth_deg"],
        "phase_matching_like": case["phase_matching_like"],
        "outputs": {
            "list_mf_para": complex_array_to_json(list_mf_para),
            "list_mf_perp": complex_array_to_json(list_mf_perp),
            "parallel_amplitude": complex_vector_to_json(sweep.parallel_amplitude),
            "perpendicular_amplitude": complex_vector_to_json(sweep.perpendicular_amplitude),
            "transmitted_2omega_jones_sp": [
                complex_vector_to_json(np.asarray(jones)) for jones in transmitted_jones
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate extended SHAARP.ml Maker fringes Python benchmarks and Mathematica inputs (ext1)."
    )
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="Pilot subset: only the first N ext cases.")
    args = parser.parse_args()

    ext_cases = build_ext_maker_cases(limit=args.limit)
    records = [run_case(case) for case in ext_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_maker_fringes_ext_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "Extended Maker-fringe grid: every orientation index, many build indices, varied polarimetry, swept over incidence angle.",
            "All cases are nonsingular (no phase-matching-like epsilon match).",
            "Outputs are transmitted 2omega SHG listMFpara/listMFperp and analyzer amplitudes.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    input_cases = [
        case_to_record(
            {
                "id": case["id"],
                "system": case["system"],
                "theta_deg": case["theta_deg"],
                "phi_deg": case["phi_deg"],
                "psi_deg": case["psi_deg"],
                "ellipticity_deg": case["ellipticity_deg"],
                "sample_azimuth_deg": case["sample_azimuth_deg"],
            }
        )
        for case in ext_cases
    ]
    input_payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit Maker input manifest (ext1) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "These are inputs only, not validation results.",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} ext Maker fringes cases to {args.output}")
    print(f"Wrote ext Maker Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
