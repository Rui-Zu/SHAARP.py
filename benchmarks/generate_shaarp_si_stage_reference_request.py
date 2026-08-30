from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
REF_DIR = BENCHMARK_DIR / "mathematica_reference"
DEFAULT_OUTPUT = REF_DIR / "shaarp_si_stage_reference_request_v1.json"


REQUESTED_MATHEMATICA_SYMBOLS = [
    "ThetaExt",
    "ThetaOrd",
    "nwNumExt",
    "nwNumOrd",
    "EwrootsNum",
    "E2wroots",
    "ERpNum",
    "ERsNum",
    "ER2w",
    "HR2w",
    "P2e",
    "P2o",
    "Peo",
    "P2eo",
    "Peoo",
    "Croots",
    "ETweo",
    "ETwo",
    "Vktweo",
    "Vktwo",
    "RSignalCrystal",
    "I2",
    "Fresnel",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.si stage-level Mathematica reference request.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_request()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si stage reference request to {args.output}")


def build_request() -> dict:
    solver_stage = _load_json(BENCHMARK_DIR / "solver_stage_benchmarks_v1.json")
    cases = solver_stage["cases"]
    request_items = [_case_request(case) for case in cases]
    angles = sorted({float(case["theta_deg"]) for case in cases})
    orientations = sorted({int(case["orientation_index"]) for case in cases})
    return {
        "schema_version": 1,
        "source": "SHAARP.py request for Mathematica SHAARP.si stage-level reference export",
        "status": "reference_values_requested_not_exported",
        "validation_claim": False,
        "source_benchmark_file": "benchmarks/solver_stage_benchmarks_v1.json",
        "source_case_count": len(cases),
        "case_count": len(request_items),
        "coverage_summary": {
            "angles_deg": angles,
            "has_normal_incidence": 0.0 in angles,
            "max_angle_deg": max(angles),
            "orientation_indices": orientations,
            "orientation_count": len(orientations),
            "has_s_polarization": any(case["incident_polarization"] == "s" for case in cases),
            "has_p_polarization": any(case["incident_polarization"] == "p" for case in cases),
            "all_cases_have_complex_epsilon": all(_has_complex_matrix(case["inputs"]["epsilon_omega_lab"]) for case in cases),
            "all_cases_have_complex_2omega_epsilon": all(
                _has_complex_matrix(case["inputs"]["epsilon_2omega_lab"]) for case in cases
            ),
            "all_cases_have_nondiagonal_epsilon": all(
                _has_nondiagonal_matrix(case["inputs"]["epsilon_omega_lab"]) for case in cases
            ),
            "all_cases_have_complex_d": all(_has_complex_matrix(case["inputs"]["d_voigt_lab"]) for case in cases),
            "phase_matching_like_cases_in_this_batch": False,
            "trivial_real_isotropic_cases_in_this_batch": False,
        },
        "mathematica_notebook_source": "Github/SHAARP/SHAARP_V1.03/SHAARP_V1.03.nb",
        "extraction_evidence": [
            "benchmarks/mathematica_reference/shaarp_si_extraction_plan_v1.json",
            "benchmarks/mathematica_reference/shaarp_si_candidate_region_symbol_audit.json",
            "benchmarks/mathematica_reference/shaarp_si_numerical_stage_snippets.json",
            "benchmarks/mathematica_reference/shaarp_si_refined_extraction_map_v1.json",
            "benchmarks/mathematica_reference/shaarp_si_refined_region_texts_v1.json",
            "benchmarks/mathematica_reference/shaarp_si_stage_symbol_sources_v1.json",
            "benchmarks/mathematica_reference/shaarp_si_symbol_gap_audit_v1.json",
        ],
        "requested_symbol_resolution": _requested_symbol_resolution(),
        "requested_mathematica_symbols": REQUESTED_MATHEMATICA_SYMBOLS,
        "requested_outputs": [
            {
                "name": symbol,
                "source": "SHAARP.si cell-4 numerical candidate symbol",
                "required": True,
            }
            for symbol in REQUESTED_MATHEMATICA_SYMBOLS
        ],
        "python_comparison_targets": [
            "coefficients",
            "reflected_total_2omega_electric",
            "transmitted_total_2omega_electric",
            "boundary_residual",
        ],
        "required_follow_up_case_families": [
            {
                "family_id": "trivial_real_isotropic_normal_incidence",
                "reason": "Required baseline sanity check with simple normal-incidence real tensors and zero/nonzero SHG.",
                "status": "queued_not_in_solver_stage_batch",
            },
            {
                "family_id": "simple_uniaxial_ordinary_extraordinary_identity",
                "reason": "Required to catch ordinary/extraordinary recognition and nonlinear-polarization branch swaps.",
                "status": "queued_not_in_solver_stage_batch",
            },
            {
                "family_id": "phase_matching_like_epsilon_omega_equals_epsilon_2omega",
                "reason": "Required extreme case for singular or near-singular inhomogeneous-field behavior.",
                "status": "queued_not_in_solver_stage_batch",
            },
            {
                "family_id": "biaxial_branch_ambiguity",
                "reason": "Required to validate branch identity without forcing ordinary/extraordinary labels where not physical.",
                "status": "queued_not_in_solver_stage_batch",
            },
            {
                "family_id": "nontrivial_reciprocal_miller_all_indices_nonzero",
                "reason": "Required to cover reciprocal-space Miller conversion with lattice-aware orientation.",
                "status": "queued_not_in_solver_stage_batch",
            },
        ],
        "notes": [
            "This is a request/contract for Mathematica export, not exported Mathematica data.",
            "The requested symbols come from SHAARP.si front-end text symbol audits and numerical stage snippets.",
            "Each exported case must preserve exact SHAARP.si symbol meanings and include enough metadata to map symbols to Python stage outputs.",
            "Agreement requires value-by-value comparison after Mathematica exports are produced.",
        ],
        "cases": request_items,
    }


def _case_request(case: dict) -> dict:
    inputs = case["inputs"]
    return {
        "id": case["id"],
        "theta_deg": case["theta_deg"],
        "incident_polarization": case["incident_polarization"],
        "orientation_index": case["orientation_index"],
        "input_summary": {
            "has_complex_epsilon_omega": _has_complex_matrix(inputs["epsilon_omega_lab"]),
            "has_complex_epsilon_2omega": _has_complex_matrix(inputs["epsilon_2omega_lab"]),
            "has_nondiagonal_epsilon_omega": _has_nondiagonal_matrix(inputs["epsilon_omega_lab"]),
            "has_nondiagonal_epsilon_2omega": _has_nondiagonal_matrix(inputs["epsilon_2omega_lab"]),
            "has_complex_d_voigt": _has_complex_matrix(inputs["d_voigt_lab"]),
        },
        "requested_mathematica_symbols": REQUESTED_MATHEMATICA_SYMBOLS,
        "python_output_keys_available_now": list(case["outputs"].keys()),
        "mathematica_outputs_status": "not_exported",
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _requested_symbol_resolution() -> dict:
    symbol_sources_path = REF_DIR / "shaarp_si_stage_symbol_sources_v1.json"
    gap_audit_path = REF_DIR / "shaarp_si_symbol_gap_audit_v1.json"
    symbol_sources = _load_json(symbol_sources_path)
    gap_audit = _load_json(gap_audit_path)
    gap_by_symbol = {record["symbol"]: record for record in gap_audit["records"]}
    resolution = {}
    for symbol in REQUESTED_MATHEMATICA_SYMBOLS:
        sources = symbol_sources["symbol_sources"].get(symbol, [])
        if sources:
            resolution[symbol] = {
                "status": "mapped_to_refined_region_sources",
                "source_count": len(sources),
                "regions": sorted({source["region_id"] for source in sources}),
            }
        else:
            gap = gap_by_symbol.get(symbol)
            resolution[symbol] = {
                "status": "not_mapped_to_exact_refined_assignment",
                "classification": None if gap is None else gap["classification"],
                "assigned_symbols_containing_name": [] if gap is None else gap["assigned_symbols_containing_name"],
            }
    return resolution


def _has_complex_matrix(value: Any) -> bool:
    if isinstance(value, dict):
        return abs(float(value.get("imag", 0.0))) > 0.0
    if isinstance(value, list):
        return any(_has_complex_matrix(item) for item in value)
    return False


def _has_nondiagonal_matrix(matrix: list) -> bool:
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i == j:
                continue
            if _complex_abs(value) > 0.0:
                return True
    return False


def _complex_abs(value: Any) -> float:
    if isinstance(value, dict):
        return abs(float(value.get("real", 0.0))) + abs(float(value.get("imag", 0.0)))
    return abs(float(value))


if __name__ == "__main__":
    main()
