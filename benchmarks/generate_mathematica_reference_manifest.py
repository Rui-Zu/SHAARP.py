from __future__ import annotations

import argparse
import json
from pathlib import Path


MANIFEST_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
DEFAULT_OUTPUT = BENCHMARK_DIR / "mathematica_reference" / "reference_manifest_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehensive Mathematica reference manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote Mathematica reference manifest to {args.output}")


def build_manifest() -> dict:
    coverage_report = _load("coverage_report_v1.json")
    suites = [
        _suite(
            suite_id="reduced_numeric_si_ml",
            source_file="numeric_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "single_interface_intensity",
                "multilayer_reflected_intensity",
                "multilayer_transmitted_intensity",
            ],
            notes=[
                "Reduced-model Python regression cases. Useful for legacy SI/ML intensity comparisons only.",
                "Do not use these to claim full anisotropic SHAARP equivalence.",
            ],
        ),
        _suite(
            suite_id="single_interface_solver_stage",
            source_file="solver_stage_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "rotated_epsilon_omega_lab",
                "rotated_epsilon_2omega_lab",
                "omega_reflected_wavevectors",
                "omega_transmitted_wavevectors",
                "omega_transmitted_electric_fields",
                "P2eo",
                "P2o",
                "Peoo",
                "inhomogeneous_ee_field",
                "inhomogeneous_oo_field",
                "inhomogeneous_eo_field",
                "two_omega_boundary_coefficients",
                "reflected_total_2omega_electric",
                "transmitted_total_2omega_electric",
            ],
            notes=[
                "Primary current full-port solver-stage target.",
                "Includes complex anisotropic non-diagonal epsilon, complex SHG coefficients, s/p polarization, and cubic/non-cubic Miller orientations.",
            ],
        ),
        _suite(
            suite_id="shaarp_si_stage_reference_request",
            source_file="mathematica_reference/shaarp_si_stage_reference_request_v1.json",
            case_key="cases",
            requested_outputs=[
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
            ],
            notes=[
                "Targets original SHAARP.si notebook stage-symbol export, not Python surrogate stage names.",
                "Request is based on front-end text extraction, symbol audit, and numerical stage snippets from SHAARP_V1.03.nb cell 4.",
                "The request contains 36 complex, non-diagonal, angle/orientation-varied cases plus an explicit follow-up queue for trivial and phase-matching-like extremes.",
                "This suite is not validation until Mathematica exports are produced and compared value-by-value.",
            ],
        ),
        _suite(
            suite_id="complex_inhomogeneous_field",
            source_file="complex_inhomogeneous_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "source_wavevector",
                "source_polarization",
                "inhomogeneous_electric_field",
                "solveInhom_residual",
            ],
            notes=[
                "Isolates Mathematica solveInhom against Python solve_inhomogeneous_field.",
                "All cases use complex non-diagonal anisotropic epsilon and complex source polarizations.",
            ],
        ),
        _suite(
            suite_id="reflected_complex_snell",
            source_file="reflected_snell_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "fast_theta",
                "slow_theta",
                "fast_refractive_index",
                "slow_refractive_index",
                "fast_electric_direction",
                "slow_electric_direction",
                "fast_wavevector_direction",
                "slow_wavevector_direction",
            ],
            notes=[
                "Isolates reflected/backward complex anisotropic solveSnell roots.",
                "Use Mathematica branch labels exactly as SHAARP emits them.",
            ],
        ),
        _suite(
            suite_id="biaxial_branch_sweeps",
            source_file="biaxial_branch_benchmarks_v1.json",
            case_key="sequences",
            requested_outputs=[
                "fast_slow_theta_by_angle",
                "fast_slow_refractive_index_by_angle",
                "fast_slow_electric_direction_by_angle",
                "mathematica_branch_labels_by_angle",
            ],
            notes=[
                "Tracks branch continuity across angle sweeps for biaxial/non-uniaxial cases.",
                "Do not force ordinary/extraordinary labels unless Mathematica explicitly does so.",
            ],
        ),
        _suite(
            suite_id="multilayer_system_workflow",
            source_file="multilayer_system_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "fundamental_boundary_coefficients",
                "fundamental_layer_wavevectors",
                "fundamental_layer_electric_fields",
                "ten_source_wavevectors",
                "ten_source_polarizations",
                "ten_inhomogeneous_electric_fields",
                "two_omega_boundary_coefficients",
                "reflected_two_omega_electric_fields",
                "substrate_two_omega_electric_fields",
            ],
            notes=[
                "Targets the staged SHAARP.ml-style system workflow.",
                "Includes complex non-diagonal tensors, nontrivial orientations, s/p polarization, and all ten forward/backward nonlinear source terms.",
            ],
        ),
        _suite(
            suite_id="multilayer_boundary_system",
            source_file="multilayer_boundary_system_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "solveFresnelN_matrix",
                "solveFresnelN_rhs",
                "solveFresnelN_known_residual",
                "solveFresnelN_coefficients",
                "solveFresnelN_residual",
                "equation_labels",
                "equation_metadata",
                "unknown_labels",
                "unknown_metadata",
                "unknown_basis_waves",
                "known_waves",
            ],
            notes=[
                "Targets direct solveFresnelN matrix/RHS/unknown-order validation.",
                "Includes one-, two-, and three-layer systems, normal and oblique incidence, complex indices, and known in-layer source waves.",
                "Use this suite to catch equation ordering, layer propagation phase, and unknown ordering mismatches.",
                "Equation metadata is Python-side row bookkeeping for preserving the exact E_x, E_y, H_x, H_y interface order.",
                "Unknown metadata is Python-side column bookkeeping for mapping each column to exact Mathematica symbols during export.",
                "Unknown and known wave records expose k, E, H, and tangential E/H values used to construct the matrix and RHS.",
            ],
        ),
        _suite(
            suite_id="multilayer_polarimetry_workflow",
            source_file="multilayer_polarimetry_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "incident_jones_sp",
                "analyzer_jones_sp",
                "sample_azimuth_deg",
                "fundamental_boundary_coefficients",
                "fundamental_layer_wavevectors",
                "fundamental_layer_electric_fields",
                "ten_source_wavevectors",
                "ten_source_polarizations",
                "ten_inhomogeneous_electric_fields",
                "two_omega_boundary_coefficients",
                "reflected_two_omega_jones_sp",
                "analyzer_amplitude",
                "analyzer_intensity",
            ],
            notes=[
                "Targets polarizer/analyzer convention validation for the staged SHAARP.ml-style workflow.",
                "Includes phi, psi, ellipticity, normal incidence, oblique incidence, nontrivial orientations, complex non-diagonal tensors, and all ten source terms.",
                "Use this suite to catch value-level polarimetry sign/order/conjugation mismatches against Mathematica.",
            ],
        ),
        _suite(
            suite_id="maker_fringes_workflow",
            source_file="maker_fringes_benchmarks_v1.json",
            case_key="cases",
            requested_outputs=[
                "MFList",
                "listMFpara",
                "listMFperp",
                "transmitted_two_omega_jones_sp",
                "parallel_analyzer_amplitude",
                "perpendicular_analyzer_amplitude",
                "fundamental_boundary_residual",
                "two_omega_boundary_residual",
            ],
            notes=[
                "Targets SHAARP.ml Maker fringes value validation for transmitted SHG angle scans.",
                "The saved Python grid is intentionally small because Maker fringes can be slow.",
                "Includes negative, normal, moderate, high-angle, complex anisotropic, and phase-matching-like cases.",
                "Use this suite to confirm listMFpara/listMFperp analyzer and normalization conventions.",
            ],
        ),
    ]
    return {
        "manifest_version": MANIFEST_VERSION,
        "status": "mathematica_reference_values_requested_not_yet_available",
        "coverage_report": "coverage_report_v1.json",
        "coverage_snapshot": coverage_report,
        "suite_count": len(suites),
        "total_reference_items": sum(suite["item_count"] for suite in suites),
        "validated_artifacts": _validated_artifacts(),
        "source_reference_artifacts": _source_reference_artifacts(),
        "diagnostic_artifacts": _diagnostic_artifacts(),
        "notes": [
            "This manifest is the comprehensive request list for Mathematica SHAARP reference exports.",
            "It does not contain Mathematica values and is not validation by itself.",
            "After exporting Mathematica values, compare value-by-value with explicit tolerances.",
        ],
        "suites": suites,
    }


def _suite(*, suite_id: str, source_file: str, case_key: str, requested_outputs: list[str], notes: list[str]) -> dict:
    payload = _load(source_file)
    items = payload[case_key]
    return {
        "suite_id": suite_id,
        "source_benchmark_file": source_file,
        "source_key": case_key,
        "item_count": len(items),
        "item_ids": [item["id"] for item in items],
        "requested_outputs": requested_outputs,
        "notes": notes,
    }


def _validated_artifacts() -> list[dict]:
    return [
        _artifact(
            "mathematica_reference/wolfram_live_compute_pnl_smoke_v1.json",
            "numeric_stage_value_comparison",
            ["tests/test_wolfram_live_compute_pnl_smoke.py"],
            "Live Wolfram 14.3 computePNL smoke reference.",
        ),
        _artifact(
            "mathematica_reference/wolfram_live_solve_inhom_smoke_v1.json",
            "numeric_stage_value_comparison",
            ["tests/test_wolfram_live_solve_inhom_smoke.py"],
            "Live Wolfram 14.3 complex non-diagonal solveInhom smoke reference.",
        ),
        _artifact(
            "mathematica_reference/solve_fresnel_smoke_output.json",
            "numeric_stage_value_comparison",
            ["tests/test_wolfram_live_solve_fresnel_smoke.py"],
            "Live Wolfram 14.3 SHAARP.ml solveFresnel smoke reference.",
        ),
        _artifact(
            "mathematica_reference/solve_snell_smoke_output.json",
            "numeric_stage_value_comparison",
            ["tests/test_wolfram_live_solve_snell_smoke.py"],
            "Live Wolfram 14.3 SHAARP.ml solveSnell smoke reference.",
        ),
        _artifact(
            "mathematica_reference/solve_fresnel_n_reference_v1.json",
            "matrix_rhs_coefficient_value_comparison",
            ["tests/test_mathematica_solve_fresnel_n_reference.py", "tests/test_multilayer_boundary.py"],
            "solveFresnelN-style multilayer boundary matrix/RHS/coefficient reference.",
        ),
        _artifact(
            "mathematica_reference/fresnel_sweep_reference_v1.json",
            "gui_list_fresnel_value_comparison_with_transmitted_wave_policy_caveat",
            ["tests/test_fresnel_sweep_reference.py", "tests/test_fresnel_sweep_python_comparison.py"],
            "SHAARP.ml GUI-shaped FresnelList Rp/Rs/Tp/Ts reference.",
        ),
        _artifact(
            "mathematica_reference/maker_fringes_reference_v1.json",
            "maker_mflist_value_comparison_with_phase_matching_diagnostic",
            ["tests/test_maker_fringes_reference_comparison.py"],
            "SHAARP.ml coarse Maker MFList/listMFpara/listMFperp reference.",
        ),
        _artifact(
            "mathematica_reference/maker_fine_reference_v1.json",
            "maker_mflist_value_comparison",
            ["tests/test_maker_fine_reference_comparison.py"],
            "SHAARP.ml three-case fine Maker reference including dense fringe-resolution grid.",
        ),
        _artifact(
            "mathematica_reference/maker_fine_fringe_resolution_reference_v1.json",
            "maker_mflist_value_comparison",
            ["tests/test_maker_fine_fringe_resolution_reference_comparison.py"],
            "SHAARP.ml standalone 481-angle dense Maker fringe-resolution reference.",
        ),
        _artifact(
            "mathematica_reference/maker_fine_phase_matching_local_reference_v1.json",
            "phase_matching_diagnostic_value_comparison",
            ["tests/test_maker_fine_phase_matching_local_reference_comparison.py"],
            "SHAARP.ml local phase-matching-like Maker diagnostic reference.",
        ),
        _artifact(
            "mathematica_reference/sample_rotation_reference_v1.json",
            "sample_rotation_value_comparison_with_phase_matching_diagnostic",
            ["tests/test_sample_rotation_reference_comparison.py"],
            "SHAARP.ml SampleRotate reference after JSON export bug fix.",
        ),
        _artifact(
            "mathematica_reference/wolfram_live_symbolic_compute_pnl_reference_v1.json",
            "symbolic_residual_and_complex_substitution_grid",
            ["tests/test_wolfram_live_symbolic_compute_pnl_reference.py"],
            "Live Wolfram 14.3 symbolic computePNL same-wave and mixed-wave reference.",
        ),
        _artifact(
            "mathematica_reference/wolfram_live_symbolic_ml_partial_pnl_reference_v1.json",
            "symbolic_residual_and_complex_substitution_grid",
            ["tests/test_wolfram_live_symbolic_ml_partial_pnl_reference.py"],
            "Live Wolfram 14.3 symbolic SHAARP.ml ten-source partial PNL reference.",
        ),
        _artifact(
            "mathematica_reference/shaarp_si_compat_signal_agreement_v1.json",
            "si_compat_reflected_er2w_36_case_value_comparison",
            ["tests/test_shaarp_si_compat_signal_agreement.py"],
            "Self-contained SHAARP.si compatibility reflected ER2w agreement diagnostic.",
        ),
    ]


def _artifact(path: str, validation_level: str, tests: list[str], notes: str) -> dict:
    return {
        "path": path,
        "exists": (BENCHMARK_DIR / path).exists(),
        "validation_level": validation_level,
        "tests": tests,
        "notes": notes,
    }


def _diagnostic_artifacts() -> list[dict]:
    return [
        _artifact(
            "mathematica_reference/wolfram_live_symbolic_point_group_d_probe_v1.json",
            "diagnostic_only_not_point_group_tensor_validation",
            ["tests/test_coverage_report.py"],
            "Live Wolfram 14.3 probe showing SHAARP.ml material extension does not rewrite dC from point group; GUI point-group tensor patterns still need a separate source-branch validation.",
        )
    ]


def _source_reference_artifacts() -> list[dict]:
    return [
        _artifact(
            "mathematica_reference/point_group_tensor_patterns_from_si_source_v1.json",
            "source_text_pattern_extraction_not_live_symbolic_execution",
            ["tests/test_point_group_tensor_source_reference.py"],
            "Machine-readable point-group d tensor table parsed from the SHAARP.si Partial Analytical Expressions source branch.",
        )
    ]


def _load(name: str) -> dict:
    return json.loads((BENCHMARK_DIR / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
