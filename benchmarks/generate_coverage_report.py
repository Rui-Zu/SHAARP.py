from __future__ import annotations

import argparse
import json
from pathlib import Path

BENCHMARK_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.py benchmark coverage report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DIR / "coverage_report_v1.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote coverage report to {args.output}")


def build_report() -> dict:
    solver = _load("solver_stage_benchmarks_v1.json")
    complex_inhom = _load("complex_inhomogeneous_benchmarks_v1.json")
    reflected = _load("reflected_snell_benchmarks_v1.json")
    numeric = _load("numeric_benchmarks_v1.json")
    biaxial = _load("biaxial_branch_benchmarks_v1.json")
    multilayer_boundary_system = _load("multilayer_boundary_system_benchmarks_v1.json")
    multilayer_system = _load("multilayer_system_benchmarks_v1.json")
    multilayer_polarimetry = _load("multilayer_polarimetry_benchmarks_v1.json")
    maker_fringes = _load("maker_fringes_benchmarks_v1.json")
    matrix = _load("comprehensive_coverage_matrix_v1.json")
    compute_pnl_reference = _load("mathematica_reference/compute_pnl_reference_v1.json")
    solve_inhom_reference = _load("mathematica_reference/solve_inhom_reference_v1.json")
    solve_snell_reference = _load("mathematica_reference/solve_snell_reference_v1.json")
    solve_snell_2omega_reference = _load("mathematica_reference/solve_snell_2omega_reference_v1.json")
    solve_fresnel_reference = _load("mathematica_reference/solve_fresnel_reference_v1.json")
    fresnel_sweep_reference = _load("mathematica_reference/fresnel_sweep_reference_v1.json")
    sample_rotation_reference = _load("mathematica_reference/sample_rotation_reference_v1.json")
    symbolic_ml_partial_pnl_reference = _load("mathematica_reference/wolfram_live_symbolic_ml_partial_pnl_reference_v1.json")
    symbolic_point_group_d_probe = _load("mathematica_reference/wolfram_live_symbolic_point_group_d_probe_v1.json")
    point_group_tensor_source_reference = _load("mathematica_reference/point_group_tensor_patterns_from_si_source_v1.json")

    test_files = {path.name: path.read_text(encoding="utf-8") for path in (ROOT / "tests").glob("test_*.py")}
    return {
        "coverage_report_version": BENCHMARK_VERSION,
        "status": "generated_python_coverage_report_not_mathematica_validation",
        "coverage_matrix_version": matrix["coverage_matrix_version"],
        "benchmark_counts": {
            "numeric_reduced_model_cases": numeric["case_count"],
            "solver_stage_cases": solver["case_count"],
            "complex_inhomogeneous_cases": complex_inhom["case_count"],
            "reflected_snell_cases": reflected["case_count"],
            "biaxial_branch_sequences": biaxial["sequence_count"],
            "biaxial_branch_angles_per_sequence": biaxial["angle_count_per_sequence"],
            "multilayer_boundary_system_cases": multilayer_boundary_system["case_count"],
            "multilayer_system_cases": multilayer_system["case_count"],
            "multilayer_polarimetry_cases": multilayer_polarimetry["case_count"],
            "maker_fringes_cases": maker_fringes["case_count"],
            "maker_fringes_total_angles": maker_fringes["total_angle_count"],
        },
        "incidence_angle": _angle_summary(solver, complex_inhom, reflected, biaxial),
        "polarization": {
            "solver_stage_polarizations": sorted({case["incident_polarization"] for case in solver["cases"]}),
        },
        "orientation": _orientation_summary(solver, biaxial),
        "tensor_complexity": _tensor_summary(solver, complex_inhom, numeric),
        "mode_identity": {
            "has_uniaxial_identity_test": "test_positive_uniaxial_identity_is_not_fast_slow_order" in test_files["test_anisotropic.py"],
            "has_shg_source_identity_test": "test_uniaxial_sources_use_extraordinary_ordinary_identity" in test_files["test_shg.py"],
            "has_biaxial_branch_tracking_test": "test_biaxial_branch_tracking_preserves_continuity_through_swapped_order"
            in test_files["test_anisotropic.py"],
            "has_ambiguous_branch_test": "test_branch_tracking_marks_ambiguous_polarization_assignment"
            in test_files["test_anisotropic.py"],
        },
        "phase_matching_and_extremes": {
            "has_numeric_phase_matching_test": "test_phase_matched_same_complex_dielectric_at_both_frequencies_is_ill_conditioned"
            in test_files["test_shg.py"],
            "has_exact_singular_phase_matching_diagnostic_test": "test_exact_phase_matching_singular_operator_uses_explicit_least_squares_diagnostic"
            in test_files["test_nonlinear.py"],
            "has_symbolic_phase_matching_test": "test_symbolic_isotropic_shg_phase_matched_same_complex_dielectric"
            in test_files["test_symbolic.py"],
            "has_normal_incidence_all_complex_test": "test_normal_incidence_all_complex_properties_solves_boundary_residual"
            in test_files["test_shg.py"],
            "has_reflected_complex_anisotropic_test": "test_reflected_complex_anisotropic_snell_satisfies_equation"
            in test_files["test_anisotropic.py"],
        },
        "multilayer_nonlinear": {
            "has_tensor_basis_generation_test": "test_generated_2omega_basis_matches_shg_tangential_target"
            in test_files["test_multilayer_basis.py"],
            "has_boundary_system_matrix_benchmark_test": "test_boundary_system_benchmarks_store_matrix_rhs_and_labels"
            in test_files["test_multilayer_boundary_system_benchmarks.py"],
            "has_ten_source_terms_test": "test_multilayer_sources_have_all_shaarp_ml_wavevectors_and_mix_factors"
            in test_files["test_nonlinear.py"],
            "has_ten_source_inhomogeneous_residual_test": "test_multilayer_inhomogeneous_fields_solve_all_ten_complex_nondiagonal_sources"
            in test_files["test_nonlinear.py"],
            "has_ten_source_boundary_bridge_test": "test_single_layer_shg_boundary_accepts_ten_solved_multilayer_sources"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_automatic_fundamental_to_shg_workflow_test": "test_workflow_solves_fundamental_then_ten_source_shg_boundary"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_tensor_driven_complex_nondiagonal_workflow_test": "test_tensor_workflow_generates_bases_and_solves_complex_nondiagonal_case"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_manual_anisotropic_top_workflow_test": "test_manual_basis_workflow_accepts_anisotropic_top_medium"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_system_config_to_tensor_workflow_test": "test_system_workflow_rotates_configured_material_tensors"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_system_angle_sweep_test": "test_system_sweep_runs_multiple_incidence_angles"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_jones_incidence_test": "test_system_jones_pure_p_matches_polarization_workflow"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_polarimetry_incidence_test": "test_system_polarimetry_uses_phi_and_ellipticity_for_incident_jones"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_reflected_analyzer_projection_test": "test_reflected_2omega_jones_components_and_analyzer_intensity"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_phi_psi_polarimetry_sweep_test": "test_polarimetry_sweep_broadcasts_phi_psi_and_returns_intensity"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_sample_azimuth_sweep_test": "test_sample_azimuth_sweep_rotates_sample_separately_from_phi"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_maker_fringes_sweep_test": "test_maker_fringes_sweep_returns_shaarp_ml_copy_lists"
            in test_files["test_multilayer_shg_boundary.py"],
            "has_transmitted_analyzer_projection_test": "test_transmitted_2omega_jones_components_and_analyzer_intensity"
            in test_files["test_multilayer_shg_boundary.py"],
            "saved_multilayer_boundary_system_case_count": multilayer_boundary_system["case_count"],
            "saved_multilayer_boundary_system_layer_counts": sorted({case["layer_count"] for case in multilayer_boundary_system["cases"]}),
            "saved_multilayer_boundary_system_has_known_sources": any(
                case["known_source"] for case in multilayer_boundary_system["cases"]
            ),
            "saved_multilayer_system_case_count": multilayer_system["case_count"],
            "saved_multilayer_system_angles_deg": sorted({case["theta_deg"] for case in multilayer_system["cases"]}),
            "saved_multilayer_system_orientation_count": len({case["orientation_index"] for case in multilayer_system["cases"]}),
            "saved_multilayer_polarimetry_case_count": multilayer_polarimetry["case_count"],
            "saved_multilayer_polarimetry_angles_deg": sorted({case["theta_deg"] for case in multilayer_polarimetry["cases"]}),
            "saved_multilayer_polarimetry_phi_deg": sorted({case["phi_deg"] for case in multilayer_polarimetry["cases"]}),
            "saved_multilayer_polarimetry_psi_deg": sorted({case["psi_deg"] for case in multilayer_polarimetry["cases"]}),
            "saved_multilayer_polarimetry_ellipticity_deg": sorted(
                {case["ellipticity_deg"] for case in multilayer_polarimetry["cases"]}
            ),
            "saved_multilayer_polarimetry_sample_azimuth_deg": sorted(
                {case["sample_azimuth_deg"] for case in multilayer_polarimetry["cases"]}
            ),
            "saved_multilayer_polarimetry_orientation_count": len(
                {case["orientation_index"] for case in multilayer_polarimetry["cases"]}
            ),
            "saved_maker_fringes_case_count": maker_fringes["case_count"],
            "saved_maker_fringes_total_angle_count": maker_fringes["total_angle_count"],
            "saved_maker_fringes_angles_deg": sorted({theta for case in maker_fringes["cases"] for theta in case["theta_deg"]}),
            "saved_maker_fringes_has_negative_angle": any(
                theta < 0.0 for case in maker_fringes["cases"] for theta in case["theta_deg"]
            ),
            "saved_maker_fringes_has_normal_incidence": any(
                theta == 0.0 for case in maker_fringes["cases"] for theta in case["theta_deg"]
            ),
            "saved_maker_fringes_max_angle_deg": max(theta for case in maker_fringes["cases"] for theta in case["theta_deg"]),
            "saved_maker_fringes_has_phase_matching_like_case": any(
                case["phase_matching_like"] for case in maker_fringes["cases"]
            ),
            "saved_maker_fringes_slow_path_small_grid": maker_fringes["slow_path_policy"][
                "default_saved_grid_is_intentionally_small"
            ],
        },
        "symbolic": {
            "symbolic_test_count_markers": sum(1 for line in test_files["test_symbolic.py"].splitlines() if line.strip().startswith("def test_")),
            "has_dense_complex_symbolic_known_waves": "test_symbolic_single_interface_shg_known_waves_dense_complex_basis_matches_numeric_pipeline"
            in test_files["test_symbolic.py"],
            "has_nondiagonal_complex_symbolic_inhomogeneous": "test_symbolic_inhomogeneous_field_nondiagonal_complex_epsilon_matches_numeric"
            in test_files["test_symbolic.py"],
            "has_mathematica_point_group_label_aliases": "test_mathematica_point_group_labels_match_ascii_aliases"
            in test_files["test_symbolic.py"],
            "has_point_group_source_reference_test": "test_source_derived_point_group_patterns_match_python_symbolic_tensors"
            in test_files["test_point_group_tensor_source_reference.py"],
        },
        "mathematica_validation": {
            "comparator_present": (BENCHMARK_DIR / "compare_mathematica_reference.py").exists(),
            "wolfram_batch_detector_present": (BENCHMARK_DIR / "detect_wolfram_batch.py").exists(),
            "wolfram_batch_detector_test_present": "test_default_candidates_include_new_player_version_pattern"
            in test_files["test_wolfram_batch_detection.py"],
            "strict_comparator_rejects_missing_source": "test_strict_compare_requires_mathematica_source_metadata"
            in test_files["test_mathematica_comparison.py"],
            "strict_comparator_rejects_python_reference": "test_strict_compare_rejects_python_regression_as_reference"
            in test_files["test_mathematica_comparison.py"],
            "strict_comparator_rejects_unfilled_placeholders": "test_strict_compare_rejects_unfilled_reference_placeholders"
            in test_files["test_mathematica_comparison.py"],
            "strict_comparator_accepts_filled_mathematica_reference": "test_strict_compare_accepts_filled_mathematica_reference"
            in test_files["test_mathematica_comparison.py"],
            "suite_template_comparator_present": "test_compare_payloads_accepts_suite_template_shape_with_suite_id"
            in test_files["test_mathematica_comparison.py"],
            "suite_template_requires_suite_id": "test_suite_template_shape_requires_suite_id_when_ambiguous"
            in test_files["test_mathematica_comparison.py"],
            "suite_template_rejects_unfilled_outputs": "test_strict_suite_template_rejects_unfilled_mathematica_outputs"
            in test_files["test_mathematica_comparison.py"],
            "compute_pnl_reference_present": (BENCHMARK_DIR / "mathematica_reference" / "compute_pnl_reference_v1.json").exists(),
            "compute_pnl_reference_case_count": len(compute_pnl_reference["cases"]),
            "compute_pnl_reference_source_is_mathematica": "Mathematica" in compute_pnl_reference["source"],
            "compute_pnl_reference_test_present": "test_mathematica_compute_pnl_reference_matches_python_values"
            in test_files["test_mathematica_compute_pnl_reference.py"],
            "solve_inhom_reference_present": (BENCHMARK_DIR / "mathematica_reference" / "solve_inhom_reference_v1.json").exists(),
            "solve_inhom_reference_case_count": len(solve_inhom_reference["cases"]),
            "solve_inhom_reference_source_is_mathematica": "Mathematica" in solve_inhom_reference["source"],
            "solve_inhom_reference_test_present": "test_mathematica_solve_inhom_reference_matches_python_values"
            in test_files["test_mathematica_solve_inhom_reference.py"],
            "solve_snell_reference_present": (BENCHMARK_DIR / "mathematica_reference" / "solve_snell_reference_v1.json").exists(),
            "solve_snell_reference_case_count": len(solve_snell_reference["cases"]),
            "solve_snell_reference_source_is_mathematica": "Mathematica" in solve_snell_reference["source"],
            "solve_snell_reference_test_present": "test_mathematica_solve_snell_reference_matches_python_values"
            in test_files["test_mathematica_solve_snell_reference.py"],
            "solve_snell_2omega_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "solve_snell_2omega_reference_v1.json"
            ).exists(),
            "solve_snell_2omega_reference_case_count": len(solve_snell_2omega_reference["cases"]),
            "solve_snell_2omega_reference_source_is_mathematica": "Mathematica"
            in solve_snell_2omega_reference["source"],
            "solve_snell_2omega_reference_test_present": "test_mathematica_solve_snell_2omega_reference_matches_python_values"
            in test_files["test_mathematica_solve_snell_2omega_reference.py"],
            "solve_fresnel_reference_present": (BENCHMARK_DIR / "mathematica_reference" / "solve_fresnel_reference_v1.json").exists(),
            "solve_fresnel_reference_case_count": len(solve_fresnel_reference["cases"]),
            "solve_fresnel_reference_source_is_mathematica": "Mathematica" in solve_fresnel_reference["source"],
            "solve_fresnel_reference_test_present": "test_mathematica_solve_fresnel_reference_matches_python_values"
            in test_files["test_mathematica_solve_fresnel_reference.py"],
            "fresnel_sweep_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "fresnel_sweep_reference_v1.json"
            ).exists(),
            "fresnel_sweep_reference_case_count": fresnel_sweep_reference["case_count"],
            "fresnel_sweep_reference_test_present": "test_fresnel_sweep_reference_matches_python_gui_shaped_values"
            in test_files["test_fresnel_sweep_python_comparison.py"],
            "maker_fringes_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "maker_fringes_reference_v1.json"
            ).exists(),
            "maker_fine_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "maker_fine_reference_v1.json"
            ).exists(),
            "maker_fine_fringe_resolution_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "maker_fine_fringe_resolution_reference_v1.json"
            ).exists(),
            "maker_fine_phase_matching_local_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "maker_fine_phase_matching_local_reference_v1.json"
            ).exists(),
            "sample_rotation_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "sample_rotation_reference_v1.json"
            ).exists(),
            "sample_rotation_reference_case_count": sample_rotation_reference["case_count"],
            "solve_fresnel_n_reference_present": (
                BENCHMARK_DIR / "mathematica_reference" / "solve_fresnel_n_reference_v1.json"
            ).exists(),
            "live_symbolic_compute_pnl_reference_present": (
                BENCHMARK_DIR
                / "mathematica_reference"
                / "wolfram_live_symbolic_compute_pnl_reference_v1.json"
            ).exists(),
            "live_symbolic_ml_partial_pnl_reference_present": (
                BENCHMARK_DIR
                / "mathematica_reference"
                / "wolfram_live_symbolic_ml_partial_pnl_reference_v1.json"
            ).exists(),
            "live_symbolic_ml_partial_pnl_expression_count": len(symbolic_ml_partial_pnl_reference["expressions"]),
            "point_group_tensor_source_reference_present": (
                BENCHMARK_DIR
                / "mathematica_reference"
                / "point_group_tensor_patterns_from_si_source_v1.json"
            ).exists(),
            "point_group_tensor_source_reference_pattern_count": point_group_tensor_source_reference["pattern_count"],
            "live_symbolic_point_group_d_probe_present": (
                BENCHMARK_DIR
                / "mathematica_reference"
                / "wolfram_live_symbolic_point_group_d_probe_v1.json"
            ).exists(),
            "live_symbolic_point_group_d_probe_status": symbolic_point_group_d_probe["status"],
            "status": "partial_reference_values_plus_live_gui_and_symbolic_validation_artifacts",
        },
    }


def _angle_summary(solver: dict, complex_inhom: dict, reflected: dict, biaxial: dict) -> dict:
    solver_angles = sorted({case["theta_deg"] for case in solver["cases"]})
    complex_angles = sorted({case["theta_deg"] for case in complex_inhom["cases"]})
    reflected_angles = sorted({case["theta_deg"] for case in reflected["cases"]})
    branch_angles = sorted({step["theta_deg"] for sequence in biaxial["sequences"] for step in sequence["steps"]})
    return {
        "solver_stage_deg": solver_angles,
        "complex_inhomogeneous_deg": complex_angles,
        "reflected_snell_deg": reflected_angles,
        "biaxial_branch_deg": branch_angles,
        "has_normal_incidence": 0.0 in solver_angles,
        "max_solver_angle_deg": max(solver_angles),
        "max_any_angle_deg": max(solver_angles + complex_angles + reflected_angles + branch_angles),
    }


def _orientation_summary(solver: dict, biaxial: dict) -> dict:
    solver_indices = sorted({case["orientation_index"] for case in solver["cases"]})
    branch_indices = sorted({sequence["orientation_index"] for sequence in biaxial["sequences"]})
    solver_last_orientation = solver["cases"][-1]["inputs"]["orientation"]
    branch_last_orientation = biaxial["sequences"][-1]["inputs"]["orientation"]
    return {
        "solver_stage_orientation_indices": solver_indices,
        "biaxial_branch_orientation_indices": branch_indices,
        "solver_stage_orientation_count": len(solver_indices),
        "biaxial_branch_orientation_count": len(branch_indices),
        "has_nonzero_cubic_miller_orientation": 4 in solver_indices and 4 in branch_indices,
        "has_noncubic_reciprocal_miller_orientation": 5 in solver_indices and 5 in branch_indices,
        "has_mathematica_point_group_axis_alias_test": (
            "test_shaarp_physics_axes_accept_mathematica_point_group_label_aliases"
            in (ROOT / "tests" / "test_orientation.py").read_text(encoding="utf-8")
        ),
        "last_solver_surface_normal": [_json_complex_real(solver_last_orientation[row][2]) for row in range(3)],
        "last_branch_surface_normal": [_json_complex_real(branch_last_orientation[row][2]) for row in range(3)],
    }


def _tensor_summary(solver: dict, complex_inhom: dict, numeric: dict) -> dict:
    return {
        "solver_stage_has_complex_epsilon_omega": any(case["metrics"]["max_abs_epsilon_omega_imag"] > 0 for case in solver["cases"]),
        "solver_stage_has_complex_epsilon_2omega": any(case["metrics"]["max_abs_epsilon_2omega_imag"] > 0 for case in solver["cases"]),
        "solver_stage_has_nondiagonal_epsilon": any(case["metrics"]["max_abs_offdiag_epsilon_omega_lab"] > 0 for case in solver["cases"]),
        "complex_inhomogeneous_has_complex_epsilon": all(
            case["metrics"]["max_abs_epsilon_imag"] > 0 for case in complex_inhom["cases"]
        ),
        "complex_inhomogeneous_has_nondiagonal_epsilon": all(
            case["metrics"]["max_abs_epsilon_offdiag"] > 0 for case in complex_inhom["cases"]
        ),
        "numeric_reduced_cases_have_complex_d": all(
            case["material_summary"]["max_abs_d_voigt_imag"] > 0 for case in numeric["cases"]
        ),
        "numeric_reduced_cases_have_nondiagonal_epsilon": all(
            case["material_summary"]["max_abs_offdiag_epsilon_omega"] > 0 for case in numeric["cases"]
        ),
    }


def _load(name: str) -> dict:
    return json.loads((BENCHMARK_DIR / name).read_text(encoding="utf-8"))


def _json_complex_real(value: dict) -> float:
    return round(float(value["real"]), 15)


if __name__ == "__main__":
    main()
