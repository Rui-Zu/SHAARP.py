"""9-LAYER SHAARP.ml Maker cases with a NON-AIR INCIDENT MEDIUM (tag: mlamb) — certification.

WHY THIS SET EXISTS. The SHAARP.py GUI lets the user set BOTH half-spaces of a stack. The
RELEASED Mathematica ♯SHAARP.ml GUI does not: `SHAARP.ml.nb:666-716` forces `m1 = setMater@Air[]`
and `:425-439` overwrites the exit slot (`allmaterials[[materialnumber+2 ;; ...]] = Table[mbot,...]`,
mbot = air), while the layer selector `Range[2, materialnumber+1]` cannot reach either. So
settable half-spaces are an EXTENSION of that tool — and an extension must be certified, not
assumed (this is "a fundamentally different capability than what we have in
mathematica where we hard coded both top and bottom surface to be air").

Certification status BEFORE this set:
  * NON-AIR EXIT medium: already covered. Every multilayer manifest (ml1..ml8, pg*, pgml*, ext1,
    mlgraze, polarimetry, sample_rotation) carries an ABSORBING, ANISOTROPIC, ROTATED substrate
    (generate_multilayer_system_benchmarks.py build_system) — the engine has always accepted an
    arbitrary `mSub` (setup.nb:6421 documents wSub as "waves into Substrate (can be Air)").
  * NON-AIR ENTRANCE medium: NEVER exercised. All 27 input manifests set layer 0 to eps = I, so
    the Mathematica global n0 = Sqrt[Mean[Eigenvalues[mats[[1]][epsOmegaC]]]]
    (export_maker_fringes_reference.wl:119) was identically 1 in every live reference.

THIS SET closes that hole. n0 is NOT a literal on the Mathematica side: it is derived per case
from layer 0's crystal-frame eps at omega, then consumed by setwInc as the true ambient index
(wInc = setWave[{omega, n0 (omega/c0) {Sin, 0, Cos}, n0 omega/c0, theta, Einc}]), so it scales
k-vector and k0 and therefore the tangential k_x that every Snell/Fresnel step chains off.
Swapping layer 0's material HERE makes the original compute a genuinely non-air ambient with NO
Wolfram-side edit at all — the wrapper is the usual 4-line path-override clone.

CONSTRUCTION. A deliberate clone of the mlgraze generator (itself ml8's stacks) with exactly TWO
deviations, so a disagreement is attributable to the ambient:
  1. layer 0 "top-air" (eps = I) -> "top-water" with eps = n^2 I, n(omega) = 1.33, n(2omega) =
     1.34. ISOTROPIC on purpose: Mathematica averages eigenvalues (tolerant of anisotropy) while
     the port's _isotropic_index RAISES unless eps = scalar*I (multilayer_shg_boundary.py:1199),
     so only an isotropic ambient is a like-for-like comparison. Real (non-absorbing) so n0 stays
     a clean positive real.
  2. incidence angles moved back into the ordinary band (20..60 deg): the point here is the
     ambient, not grazing, and the band avoids the mlgraze f4NL kernel-death onsets entirely.
Everything else — RAW_CASES, interlayer helpers, build_system seeds, schema, payload keys — is
byte-identical to mlgraze/ml8 (the sigraze lesson: when cloning a validated pipeline, match the
SCHEMA exactly).

PHYSICS NOTE (why this is a real test, not a relabel): with n0 = 1.33 the incident k_x =
n0 (omega/c) sin(theta) differs from the air case at every angle, so the transmitted angles in
EVERY layer, all seven internal propagation phases and the whole boundary solve shift. The
companion air-ambient twin (`--twin`) is generated from the same stacks so the gated test can
assert the two differ materially — a dead n0 knob would make them identical.

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
    CrystalOrientation,
    CrystalStructure,
    Material,
    Polarimetry,
    with_sample_azimuth_deg,
)
from shaarp.multilayer_shg_boundary import (  # noqa: E402
    shaarp_ml_selected_transmitted_2omega_jones_sp,
    solve_multilayer_maker_fringes_sweep,
)

BENCHMARK_VERSION = 1
TAG = "mlamb"

# The ambient under test: water-like, isotropic, non-absorbing.
AMBIENT_N_OMEGA = 1.33
AMBIENT_N_2OMEGA = 1.34

# Ordinary incidence band (the ambient is the variable here, not grazing); also clear of the
# mlgraze f4NL kernel-death onsets (>= 84 deg for the binding case).
THETA_SWEEP_DEG = [20.0, 40.0, 60.0]
SAMPLE_AZIMUTH_DEG = 0.0

# A 2-case subset of mlgraze/ml8's RAW_CASES (one kernel run; the ambient is a global scaling of
# the incident wave, so two independent stacks are ample to catch a convention/wiring error).
# (build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg)
RAW_CASES = [
    (1, 0, 35.0, 40.0, 5.0),
    (4, 2, 20.0, 60.0, 18.0),
]


def ambient_material(n_omega: float = AMBIENT_N_OMEGA,
                     n_2omega: float = AMBIENT_N_2OMEGA) -> Material:
    """The isotropic non-air incident medium: eps = n^2 * I at both frequencies.

    Isotropic BY CONSTRUCTION so the two sides agree exactly: Mathematica takes
    n0 = Sqrt[Mean[Eigenvalues[eps]]] (which equals n for eps = n^2 I) and the port takes
    _isotropic_index (which requires eps = scalar * I). Zero d — a half-space is passive.
    """
    return Material(
        name=f"top-ambient-n{n_omega:g}",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3) * float(n_omega) ** 2,
        epsilon_2omega=np.eye(3) * float(n_2omega) ** 2,
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )


def build_mlamb_cases(limit: int | None = None, *, air_twin: bool = False) -> list[dict]:
    """The mlgraze/ml8 stacks with layer 0 swapped for the non-air ambient.

    ``air_twin=True`` keeps the original air ambient on the SAME stacks — the twin used to prove
    the n0 knob is alive (identical curves would mean the ambient never reached the solve).
    """
    raw = RAW_CASES if limit is None else RAW_CASES[:limit]
    cases = []
    for build_index, orientation_index, phi_deg, psi_deg, ellipticity_deg in raw:
        sys3 = build_system(build_index, THETA_SWEEP_DEG[0], orientation_index)
        top, film, substrate = sys3.layers
        if not air_twin:  # the ONLY structural deviation from mlgraze/ml8
            top = replace(top, material=ambient_material())
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
        description="Generate NON-AIR-AMBIENT 9-layer SHAARP.ml Maker benchmarks + Mathematica "
                    "inputs (mlamb).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / f"maker_fringes_benchmarks_{TAG}.json")
    parser.add_argument(
        "--mathematica-input",
        type=Path,
        default=here / "mathematica_reference" / f"maker_mathematica_inputs_{TAG}.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--twin", action="store_true",
        help="also write the AIR-ambient twin (same stacks, layer 0 = air) next to the output, "
             "for the n0-is-alive check")
    args = parser.parse_args()

    ml_cases = build_mlamb_cases(limit=args.limit)
    records = [run_case(c) for c in ml_cases]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "status": "python_multilayer_maker_fringes_benchmark_for_mathematica_reference",
        "case_count": len(records),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "ambient_index_omega": AMBIENT_N_OMEGA,
        "ambient_index_2omega": AMBIENT_N_2OMEGA,
        "layer_structure": "NON-AIR ambient (isotropic n=1.33/1.34) / active-film / anisotropic-interlayer / isotropic-interlayer / isotropic-interlayer-2 / isotropic-interlayer-3 / anisotropic-interlayer-2 / isotropic-interlayer-4 / substrate (9 layers, 7 internal)",
        "notes": [
            "Certification of the SETTABLE INCIDENT MEDIUM. The released Mathematica .ml GUI "
            "hardcodes BOTH half-spaces to air (SHAARP.ml.nb:666-716 forces m1 = Air; :425-439 "
            "overwrites the exit slot with mbot = Air; the layer selector Range[2, "
            "materialnumber+1] reaches neither), so a user-set ambient is an EXTENSION of that "
            "tool and needs its own live evidence.",
            "The ENGINE was always general: setup.nb:6421 documents wSub as 'waves into Substrate "
            "(can be Air)', and n0 is computed per case from layer 0's eps "
            "(export_maker_fringes_reference.wl:119) then consumed by setwInc as the ambient "
            "index — so swapping layer 0 in THIS manifest makes the original compute a non-air "
            "ambient with no Wolfram-side edit.",
            "A NON-AIR EXIT medium was already certified: every multilayer reference carries an "
            "absorbing anisotropic rotated substrate. Only the ENTRANCE medium was unexercised "
            "(all 27 prior manifests set layer 0 to eps = I -> n0 == 1).",
            "Stacks are byte-identical to mlgraze/ml8's validated ones except layer 0 (and the "
            "ordinary 20/40/60 deg band) — so a disagreement is attributable to the ambient.",
            "The ambient is ISOTROPIC on purpose: Mathematica averages eigenvalues for n0 while "
            "the port requires eps = scalar*I, so only an isotropic ambient is like-for-like.",
            "Only the film is SHG-active; all passive layers carry a zero SHG tensor.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.twin:
        twin_cases = build_mlamb_cases(limit=args.limit, air_twin=True)
        twin_records = [run_case(c) for c in twin_cases]
        twin_path = args.output.with_name(f"maker_fringes_benchmarks_{TAG}_airtwin.json")
        twin_path.write_text(json.dumps({
            "benchmark_version": BENCHMARK_VERSION,
            "tag": f"{TAG}_airtwin",
            "status": "python_air_ambient_twin_for_n0_liveness_check",
            "case_count": len(twin_records),
            "theta_sweep_deg": THETA_SWEEP_DEG,
            "notes": ["Same stacks as mlamb with layer 0 = air (eps = I). Used ONLY to prove the "
                      "ambient index reaches the solve: identical curves would mean a dead knob."],
            "cases": twin_records,
        }, indent=2), encoding="utf-8")
        print(f"Wrote air-ambient twin to {twin_path}")

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
        "source": "Python explicit 9-layer NON-AIR-AMBIENT Maker input manifest (mlamb) for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "theta_sweep_deg": THETA_SWEEP_DEG,
        "ambient_index_omega": AMBIENT_N_OMEGA,
        "ambient_index_2omega": AMBIENT_N_2OMEGA,
        "notes": [
            "Layer 0 carries eps = n^2 I with n(omega) = 1.33, n(2omega) = 1.34 — the Mathematica "
            "side derives n0 = Sqrt[Mean[Eigenvalues[eps_omega]]] = 1.33 from this record and "
            "feeds it to setwInc. No exporter edit is required.",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": input_cases,
    }
    args.mathematica_input.parent.mkdir(parents=True, exist_ok=True)
    args.mathematica_input.write_text(json.dumps(input_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} mlamb (9-layer non-air-ambient) cases to {args.output}")
    print(f"Wrote mlamb Mathematica inputs to {args.mathematica_input}")


if __name__ == "__main__":
    main()
