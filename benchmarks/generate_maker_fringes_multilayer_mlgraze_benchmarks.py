"""9-LAYER SHAARP.ml Maker cases at GRAZING INCIDENCE (tag: mlgraze) — R1 breadth, ML side.

WHY THIS SET EXISTS. Residual risk R1: live-Mathematica agreement is certified only at the
EXPORTED reference points. After the SI side's grazing hole was closed (sigraze, theta 75..89.5,
agreement 3.6e-15), enumerating the MULTILAYER side showed the same class of hole, wider:

    certified theta (deg), read from the frozen benchmark JSONs (theta_deg per case):
      ml1/ml2/ml3/ml5/ml6/ml7/ml8 0 20 40 60
      pgml / pgml2 / pg2act 0 20 40 60
      pg1 / pg2 / pg3 / ext1 <= 60
      multilayer_system_benchmarks <= 52 sample_rotation <= 42
    -> MULTILAYER ceiling = 60.0 deg. (The single-film fine-Maker chunks reach 80 deg
       — maker_fine_chunk_high_oblique_* — but nothing MULTILAYER goes past 60.)

Above 60 deg the multilayer path is exactly where grazing is hardest: cos(theta_T) shrinks in
EVERY layer, the exp(i k_z h) propagation phases chain across seven internal interfaces, and the
per-interface boundary blocks are worst-conditioned. The GUI exposes the full range and its matrix
sweep drives theta = 89 deg on the ML tab.

CONSTRUCTION. A deliberate clone of the ml8 generator (the DEEPEST live-validated stack: 9 layers,
air / active-film / 2 anisotropic + 4 isotropic passive interlayers / substrate) with ONLY the
incidence angles changed — the same RAW_CASES, the same interlayer helpers imported from
ml2/ml5/ml6/ml7/ml8, the same build_system seeds. build_system's theta argument seeds only the
initial Polarimetry (immediately overwritten below); material constants derive from build_index and
orientation_index alone — verified before writing this file — so the stacks here are byte-identical
to ml8's validated ones and any disagreement is attributable to the ANGLE. The film eps is complex
(imag_base 0.012/0.045) with three distinct principal values, so these cases are absorbing AND
biaxial as well as grazing.

SCHEMA. Mirrors ml8 EXACTLY (same payload keys, same case_to_record manifest). The sigraze lesson
is standing policy: when cloning a validated pipeline, match the SCHEMA exactly — a renamed payload
key cost two >90-minute kernel spins on the SI side ("equivalent" is not "identical", and the
consumer is not forgiving).

This file produces Python-side outputs and a Mathematica input manifest. It is NOT validation
evidence on its own — the evidence is the live SHAARP.ml f4NL export compared against it.
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
from benchmarks.generate_maker_fringes_multilayer_ml7_benchmarks import (  # noqa: E402
    _anisotropic_interlayer_2,
)
from benchmarks.generate_maker_fringes_multilayer_ml8_benchmarks import (  # noqa: E402
    _isotropic_interlayer_4,
)
from benchmarks.generate_maker_mathematica_inputs import case_to_record  # noqa: E402
from benchmarks.generate_multilayer_system_benchmarks import (  # noqa: E402
    build_system,
    complex_array_to_json,
    complex_vector_to_json,
)
from shaarp.config import (  # noqa: E402
    Polarimetry,
    with_sample_azimuth_deg,
)
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    shaarp_ml_selected_transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
)

BENCHMARK_VERSION = 1
TAG = "mlgraze"
# The UNCERTIFIED multilayer band only (>60 deg), walked to the limit the ORIGINAL tool can
# actually export FOR EVERY CASE. The first draft used the SI sigraze grid (75..89.5); per-angle
# mapping runs on (single L solve per kernel session, 2-4 attempts per angle)
# showed the live SHAARP.ml f4NL kills the WolframKernel — silent exit 0, no output, no stderr —
# above a CASE-DEPENDENT onset angle:
# case 1 (b1_o0): 75..87.5 pass | 88, 89, 89.5 die -> onset 88.0
# case 2 (b4_o2): 75..83 pass | 84 dies -> onset 84.0 <- binds the grid
# case 3 (b8_o0): 75..87.5 pass | none found <=87.5 -> onset > 87.5
# case 4 (b15_o1): 75..86 pass | 87 dies -> onset 87.0
# A breadcrumb-instrumented exporter localized the death INSIDE f4NL itself (the 'entering
# L' crumb lands, the 'returned' crumb never does). The onset varying 84 -> >=88 across
# stacks that differ only in deterministic seeds/orientations shows this is a numerical
# boundary of the original's grazing solve, not a single pathological input. The grid below is
# the largest common band every case exports: certified multilayer ceiling 60 -> 83 deg, with
# the per-case onsets above recorded as a documented original-tool limitation (see the mlgraze
# gated test + residual-risk register R1). The Python port solves the full 75..89.5 range on
# these same inputs with ~1e-15 boundary residuals; certification above 83 deg is blocked by
# the ORACLE, not the port.
THETA_SWEEP_DEG = [75.0, 78.0, 80.0, 81.0, 82.0, 83.0]
SAMPLE_AZIMUTH_DEG = 0.0

# Identical to ml8's RAW_CASES — the point is that ONLY the angle differs.
# (build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    (1, 0, 35.0, 40.0, 5.0),
    (4, 2, 20.0, 60.0, 18.0),
    (8, 0, 60.0, 50.0, 12.0),
    (15, 1, 55.0, 20.0, 28.0),
]


def build_mlgraze_cases(limit: int | None = None) -> list[dict]:
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
        iso4 = _isotropic_interlayer_4(build_index)
        system = replace(
            sys3,
            layers=[top, film, aniso, iso, iso2, iso3, aniso2, iso4, substrate],
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
                "layer_count": 9,
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
    parser = argparse.ArgumentParser(
        description="Generate GRAZING-INCIDENCE 9-layer SHAARP.ml Maker benchmarks + Mathematica inputs (mlgraze).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ml_cases = build_mlgraze_cases(limit=args.limit)
    records = [run_case(c) for c in ml_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_multilayer_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "layer_structure": "air / active-film / anisotropic-interlayer / isotropic-interlayer / isotropic-interlayer-2 / isotropic-interlayer-3 / anisotropic-interlayer-2 / isotropic-interlayer-4 / substrate (9 layers, 7 internal)",
        "notes": [
            "GRAZING-INCIDENCE clone of the ml8 9-layer grid (theta 75..83 deg vs ml8's 0..60).",
            "The grid stops at 83 deg: the live SHAARP.ml f4NL deterministically kills the "
            "WolframKernel (silent exit 0, no stderr) above a CASE-DEPENDENT onset angle — mapped "
            "per-angle 2026-08-15: case 1 onset 88.0, case 2 onset 84.0 (binds this grid), case 3 "
            "onset > 87.5, case 4 onset 87.0 — and localized INSIDE f4NL via a "
            "breadcrumb-instrumented exporter. 83 deg is the largest common band all four cases "
            "export. The Python port solves the full 75..89.5 range on the same inputs with "
            "~1e-15 boundary residuals; certification above 83 deg is blocked by the oracle, not "
            "the port.",
            "Motivation: residual risk R1. Every pre-existing MULTILAYER live reference tops out at "
            "theta = 60 deg (ml1..ml8, pgml/pgml2/pg2act all sweep 0/20/40/60; polarimetry <= 52; "
            "sample rotation <= 42), while the GUI exposes the full range and its matrix sweep "
            "drives theta = 89. The single-film fine-Maker chunks reach 80 deg, but nothing "
            "multilayer goes past 60.",
            "Stacks are BYTE-IDENTICAL to ml8's validated ones (same RAW_CASES, same interlayer "
            "helpers, same build_system seeds; build_system's theta argument seeds only the initial "
            "polarimetry, which is overwritten) — so a disagreement is attributable to the angle.",
            "Only the film is SHG-active; all passive layers carry a zero SHG tensor.",
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
        "source": "Python explicit 9-layer GRAZING-INCIDENCE Maker input manifest (mlgraze) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "notes": [
            "9-layer stacks identical to ml8's; only the incidence angles differ (75..87.5 deg; "
            ">= 88 deg excluded — the live f4NL kernel-death band, see the benchmark notes).",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} mlgraze (9-layer grazing) cases to {args.output}")
    print(f"Wrote mlgraze Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
