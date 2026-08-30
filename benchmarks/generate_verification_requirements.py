from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BENCHMARK_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_coverage_report import build_report


def build_manifest() -> dict:
    coverage = build_report()
    fine_sampling_plan = json.loads((BENCHMARK_DIR / "maker_fringes_fine_sampling_plan_v1.json").read_text(encoding="utf-8"))
    counts = coverage["benchmark_counts"]
    validation = coverage["mathematica_validation"]

    requirements = [
        _requirement(
            "value_level_mathematica_comparison",
            "Every accepted SHAARP agreement claim must compare numerical values against Mathematica/Wolfram exports, not only shapes or patterns.",
            "partially_met_stage_level_only",
            "mathematica_value_validated",
            [
                "benchmarks/mathematica_reference/compute_pnl_reference_v1.json",
                "benchmarks/mathematica_reference/solve_inhom_reference_v1.json",
                "benchmarks/mathematica_reference/solve_snell_reference_v1.json",
                "benchmarks/mathematica_reference/solve_snell_2omega_reference_v1.json",
                "benchmarks/mathematica_reference/solve_fresnel_reference_v1.json",
            ],
            "Export full SHAARP.si and full SHAARP.ml end-to-end reference cases and compare values stage by stage.",
            {
                "stage_reference_case_counts": {
                    "compute_pnl": validation["compute_pnl_reference_case_count"],
                    "solve_inhom": validation["solve_inhom_reference_case_count"],
                    "solve_snell_omega": validation["solve_snell_reference_case_count"],
                    "solve_snell_2omega": validation["solve_snell_2omega_reference_case_count"],
                    "solve_fresnel": validation["solve_fresnel_reference_case_count"],
                },
                "full_end_to_end_agreement_proven": False,
            },
        ),
        _requirement(
            "at_least_20_complex_numerical_cases",
            "Maintain at least 20 numerical cases with anisotropy, complex dielectric permittivity, and complex SHG coefficients.",
            "met_for_python_and_stage_references_not_full_end_to_end",
            "mixed_python_regression_and_mathematica_stage_validation",
            [
                "benchmarks/numeric_benchmarks_v1.json",
                "benchmarks/solver_stage_benchmarks_v1.json",
                "benchmarks/mathematica_reference/compute_pnl_reference_v1.json",
                "benchmarks/mathematica_reference/solve_inhom_reference_v1.json",
            ],
            "Carry the same level of complexity into full SHAARP.si/SHAARP.ml end-to-end exports.",
            {
                "numeric_reduced_model_cases": counts["numeric_reduced_model_cases"],
                "solver_stage_cases": counts["solver_stage_cases"],
                "solve_inhom_mathematica_cases": validation["solve_inhom_reference_case_count"],
                "complex_d_covered": coverage["tensor_complexity"]["numeric_reduced_cases_have_complex_d"],
                "complex_epsilon_covered": coverage["tensor_complexity"]["solver_stage_has_complex_epsilon_omega"],
                "nondiagonal_epsilon_covered": coverage["tensor_complexity"]["solver_stage_has_nondiagonal_epsilon"],
            },
        ),
        _requirement(
            "incidence_angle_sweeps",
            "Check across multiple incidence angles, including normal incidence and high oblique angles.",
            "met_for_python_and_stage_references_not_full_end_to_end",
            "mixed_python_regression_and_mathematica_stage_validation",
            [
                "benchmarks/solver_stage_benchmarks_v1.json",
                "benchmarks/reflected_snell_benchmarks_v1.json",
                "benchmarks/multilayer_system_benchmarks_v1.json",
                "benchmarks/mathematica_reference/solve_snell_reference_v1.json",
                "benchmarks/mathematica_reference/solve_snell_2omega_reference_v1.json",
            ],
            "Add the same angle grid to full SHAARP.si/SHAARP.ml reference exports.",
            {
                "has_normal_incidence": coverage["incidence_angle"]["has_normal_incidence"],
                "max_any_angle_deg": coverage["incidence_angle"]["max_any_angle_deg"],
                "solver_stage_angles_deg": coverage["incidence_angle"]["solver_stage_deg"],
                "reflected_snell_angles_deg": coverage["incidence_angle"]["reflected_snell_deg"],
            },
        ),
        _requirement(
            "fresnel_coefficient_sweep_plots",
            "Port and validate SHAARP.ml Fresnel coefficient angle sweeps, not only one-off solveFresnel boundary equations.",
            "mathematica_value_validated_for_exported_gui_list_fresnel_cases",
            "mathematica_gui_list_fresnel_value_validated_with_transmitted_wave_policy_caveat",
            [
                "Github/SHAARP.ml/docs/examples.md",
                "benchmarks/mathematica_reference/WORKAROUND.md",
                "benchmarks/mathematica_reference/solve_fresnel_reference_v1.json",
                "benchmarks/mathematica_reference/fresnel_sweep_reference_v1.json",
                "benchmarks/compare_fresnel_sweep_reference.py",
                "tests/test_fresnel_sweep_reference.py",
                "tests/test_fresnel_sweep_python_comparison.py",
                "shaarp/linear.py",
                "tests/test_linear.py",
            ],
            "Expand SHAARP.ml FresnelList convention coverage beyond the current three exported cases and keep the transmitted-wave policy caveat visible.",
            {
                "basic_solve_fresnel_reference_cases": validation["solve_fresnel_reference_case_count"],
                "linear_interface_sweep_exists": True,
                "fresnel_coefficient_sweep_api_present": True,
                "fresnel_list_mathematica_export_present": True,
                "first_class_fresnel_coefficient_plot_workflow_present": True,
            },
        ),
        _requirement(
            "maker_fringes_multilayer_angle_scan",
            "Port and validate SHAARP.ml Maker fringes angle scans with full multiple reflection, backward waves, standing waves, and thickness-dependent fringes.",
            "partially_mathematica_value_validated_not_full_ml_end_to_end",
            "mathematica_mf_list_value_validated_with_phase_matching_diagnostic",
            [
                "Github/SHAARP.ml/docs/examples.md",
                "benchmarks/mathematica_reference/WORKAROUND.md",
                "shaarp/multilayer_shg_boundary.py",
                "tests/test_multilayer_shg_boundary.py",
                "benchmarks/maker_fringes_fine_sampling_plan_v1.json",
                "benchmarks/run_maker_fringes_fine_sampling.py",
                "benchmarks/compare_maker_fine_to_coarse.py",
                "benchmarks/compare_maker_fringes_reference.py",
                "benchmarks/build_maker_fringes_manual_reference.py",
                "benchmarks/mathematica_reference/maker_fringes_reference_v1.json",
                "benchmarks/mathematica_reference/maker_fine_reference_v1.json",
                "benchmarks/mathematica_reference/maker_fine_fringe_resolution_reference_v1.json",
                "benchmarks/mathematica_reference/maker_fine_phase_matching_local_reference_v1.json",
                "tests/test_maker_fine_to_coarse_comparison.py",
                "tests/test_maker_fringes_reference_comparison.py",
                "tests/test_maker_fringes_manual_reference.py",
            ],
            "Expand Maker fringes validation into broader full SHAARP.ml end-to-end cases, including additional GUI assumptions and documented singular diagnostics.",
            {
                "multilayer_system_angle_sweep_exists": coverage["multilayer_nonlinear"]["has_system_angle_sweep_test"],
                "multilayer_polarimetry_sweep_exists": coverage["multilayer_nonlinear"]["has_phi_psi_polarimetry_sweep_test"],
                "maker_fringes_sweep_api_present": True,
                "maker_fringes_copy_lists_present": True,
                "maker_fringes_saved_python_case_count": coverage["benchmark_counts"]["maker_fringes_cases"],
                "maker_fringes_saved_python_total_angles": coverage["benchmark_counts"]["maker_fringes_total_angles"],
                "maker_fringes_saved_angles_deg": coverage["multilayer_nonlinear"]["saved_maker_fringes_angles_deg"],
                "maker_fringes_has_negative_angle": coverage["multilayer_nonlinear"]["saved_maker_fringes_has_negative_angle"],
                "maker_fringes_has_normal_incidence": coverage["multilayer_nonlinear"][
                    "saved_maker_fringes_has_normal_incidence"
                ],
                "maker_fringes_max_angle_deg": coverage["multilayer_nonlinear"]["saved_maker_fringes_max_angle_deg"],
                "maker_fringes_has_phase_matching_like_case": coverage["multilayer_nonlinear"][
                    "saved_maker_fringes_has_phase_matching_like_case"
                ],
                "maker_fringes_slow_path_small_grid": coverage["multilayer_nonlinear"][
                    "saved_maker_fringes_slow_path_small_grid"
                ],
                "maker_fringes_fine_sampling_plan_present": True,
                "maker_fringes_fine_sampling_total_planned_angles": fine_sampling_plan["total_planned_angle_count"],
                "maker_fringes_fine_sampling_requires_confirm_slow": "--confirm-slow"
                in fine_sampling_plan["slow_runner"],
                "maker_fringes_fine_sampling_not_routine_suite": fine_sampling_plan[
                    "routine_suite_should_not_run_fine_grid"
                ],
                "maker_fringes_agreement_requires_mathematica_mf_values": fine_sampling_plan[
                    "mathematica_reference_required_for_agreement_claim"
                ],
                "maker_fringes_required_mathematica_outputs": [
                    "MFList",
                    "listMFpara",
                    "listMFperp",
                ],
                "maker_fringes_python_only_regression_is_not_agreement": False,
                "maker_fringes_comparator_present": (BENCHMARK_DIR / "compare_maker_fringes_reference.py").exists(),
                "maker_fringes_manual_reference_builder_present": (
                    BENCHMARK_DIR / "build_maker_fringes_manual_reference.py"
                ).exists(),
                "maker_fringes_fine_to_coarse_comparator_present": (
                    BENCHMARK_DIR / "compare_maker_fine_to_coarse.py"
                ).exists(),
                "full_multiple_reflection_standing_wave_claim_validated": False,
                "maker_fringes_reference_export_present": True,
            },
        ),
        _requirement(
            "phase_matching_and_singular_limits",
            "Include extreme phase-matching cases such as same dielectric permittivity at omega and 2omega, and treat singular systems explicitly.",
            "met_for_python_diagnostics_not_mathematica_value_validated",
            "python_residual_and_symbolic_regression",
            [
                "tests/test_shg.py",
                "tests/test_nonlinear.py",
                "tests/test_symbolic.py",
                "benchmarks/compare_symbolic_reference.py",
            ],
            "Export Mathematica singular diagnostics or limiting expressions before claiming agreement for phase-matched analytical cases.",
            coverage["phase_matching_and_extremes"],
        ),
        _requirement(
            "all_complex_properties",
            "Exercise simultaneous complex epsilon, complex nonlinear coefficients, complex fields, and complex wavevectors.",
            "met_for_python_and_selected_stage_references_not_full_end_to_end",
            "mixed_python_regression_and_mathematica_stage_validation",
            [
                "benchmarks/complex_inhomogeneous_benchmarks_v1.json",
                "benchmarks/mathematica_reference/solve_inhom_reference_v1.json",
                "tests/test_shg.py",
                "tests/test_multilayer_shg_boundary.py",
            ],
            "Promote these all-complex cases into the full interface and multilayer Mathematica export set.",
            {
                "complex_inhomogeneous_cases": counts["complex_inhomogeneous_cases"],
                "all_complex_normal_incidence_test": coverage["phase_matching_and_extremes"]["has_normal_incidence_all_complex_test"],
                "complex_inhomogeneous_has_complex_epsilon": coverage["tensor_complexity"][
                    "complex_inhomogeneous_has_complex_epsilon"
                ],
                "complex_inhomogeneous_has_nondiagonal_epsilon": coverage["tensor_complexity"][
                    "complex_inhomogeneous_has_nondiagonal_epsilon"
                ],
            },
        ),
        _requirement(
            "nondiagonal_dielectric_permittivity",
            "Cover non-diagonal dielectric tensors, including rotated anisotropic tensors and explicitly complex off-diagonal terms.",
            "met_for_python_and_selected_stage_references_not_full_end_to_end",
            "mixed_python_regression_and_mathematica_stage_validation",
            [
                "benchmarks/solver_stage_benchmarks_v1.json",
                "benchmarks/complex_inhomogeneous_benchmarks_v1.json",
                "benchmarks/mathematica_reference/solve_inhom_reference_v1.json",
                "tests/test_multilayer_shg_boundary.py",
            ],
            "Export non-diagonal dielectric full SHAARP.si/SHAARP.ml cases, including stage intermediates.",
            {
                "solver_stage_has_nondiagonal_epsilon": coverage["tensor_complexity"][
                    "solver_stage_has_nondiagonal_epsilon"
                ],
                "complex_inhomogeneous_has_nondiagonal_epsilon": coverage["tensor_complexity"][
                    "complex_inhomogeneous_has_nondiagonal_epsilon"
                ],
            },
        ),
        _requirement(
            "biaxial_and_branch_identity",
            "Cover biaxial/non-uniaxial branch tracking and prevent ordinary/extraordinary wave identity swaps.",
            "met_for_python_regression_not_mathematica_value_validated",
            "python_residual_and_regression",
            [
                "benchmarks/biaxial_branch_benchmarks_v1.json",
                "tests/test_anisotropic.py",
                "tests/test_shg.py",
            ],
            "Export Mathematica branch identities/eigenvectors for the biaxial sweeps and compare phase-aligned modes.",
            {
                "biaxial_branch_sequences": counts["biaxial_branch_sequences"],
                "angles_per_sequence": counts["biaxial_branch_angles_per_sequence"],
                "has_mode_identity_tests": all(coverage["mode_identity"].values()),
            },
        ),
        _requirement(
            "miller_indices_and_lattice_orientation",
            "Treat Miller indices as reciprocal-space plane indices requiring a crystal lattice, not as direct Cartesian vectors.",
            "met_for_python_regression_not_mathematica_value_validated",
            "python_regression",
            [
                "tests/test_orientation.py",
                "benchmarks/mathematica_reference/export_orientation_reference.wl",
                "benchmarks/mathematica_reference/EXACT_SYMBOL_AUDIT_2026_05_17.md",
            ],
            "Generate accepted orientation_reference_v1.json from Mathematica hklConvert/QC/QP outputs.",
            {
                "has_nonzero_cubic_miller_orientation": coverage["orientation"]["has_nonzero_cubic_miller_orientation"],
                "has_noncubic_reciprocal_miller_orientation": coverage["orientation"][
                    "has_noncubic_reciprocal_miller_orientation"
                ],
                "solver_stage_orientation_count": coverage["orientation"]["solver_stage_orientation_count"],
                "biaxial_branch_orientation_count": coverage["orientation"]["biaxial_branch_orientation_count"],
                "complex_orientation_matrix_rejected": True,
            },
        ),
        _requirement(
            "multilayer_solve_fresnel_n_boundary",
            "Compare SHAARP.ml solveFresnelN-style multilayer matrix/RHS/coefficients/residuals with explicit row and column metadata.",
            "met_for_python_benchmark_not_mathematica_value_validated",
            "python_residual_and_regression",
            [
                "benchmarks/multilayer_boundary_system_benchmarks_v1.json",
                "benchmarks/compare_multilayer_boundary_reference.py",
                "tests/test_multilayer_boundary_reference_comparison.py",
            ],
            "Export solveFresnelN matrix/RHS/known residual/coefficients from Mathematica and compare values.",
            {
                "saved_case_count": counts["multilayer_boundary_system_cases"],
                "layer_counts": coverage["multilayer_nonlinear"]["saved_multilayer_boundary_system_layer_counts"],
                "has_known_sources": coverage["multilayer_nonlinear"]["saved_multilayer_boundary_system_has_known_sources"],
            },
        ),
        _requirement(
            "symbolic_si_full_and_ml_partial",
            "Validate full SHAARP.si analytical expressions and SHAARP.ml partial analytical expressions by symbolic residuals and deterministic substitutions.",
            "scaffolded_not_mathematica_value_validated",
            "symbolic_comparator_scaffold",
            [
                "benchmarks/compare_symbolic_reference.py",
                "tests/test_symbolic_reference_comparison.py",
            ],
            "Export real Mathematica InputForm expressions with exact symbol metadata, then compare canonical residuals and numeric substitutions.",
            {
                "has_dense_complex_symbolic_known_waves": coverage["symbolic"]["has_dense_complex_symbolic_known_waves"],
                "has_nondiagonal_complex_symbolic_inhomogeneous": coverage["symbolic"][
                    "has_nondiagonal_complex_symbolic_inhomogeneous"
                ],
                "has_phase_matching_singular_diagnostics": coverage["phase_matching_and_extremes"][
                    "has_symbolic_phase_matching_test"
                ],
            },
        ),
        _requirement(
            "shaarp_ml_gui_feature_parity",
            "Track SHAARP.ml GUI capabilities from the updated notebook/docs so GUI-only features are not lost during the Python port.",
            "merged_gui_implemented_complete",
            "gui_capability_audit_not_physics_validation",
            [
                "benchmarks/gui_capability_audit_v1.json",
                "Github/SHAARP.ml/SHAARP.ml.nb",
                "Github/SHAARP.ml/docs/input.md",
                "Github/SHAARP.ml/docs/examples.md",
            ],
            "Convert each GUI audit item marked missing or partial into explicit Python API, notebook, or documentation work, then validate numerical outputs against Mathematica.",
            {
                "full_gui_feature_parity_claimed": False,
                "gui_audit_present": (BENCHMARK_DIR / "gui_capability_audit_v1.json").exists(),
                "gui_capabilities_audited": 16,
            },
        ),
    ]

    return {
        "verification_requirements_version": BENCHMARK_VERSION,
        "status": "user_requested_requirements_queue_not_full_validation_claim",
        "full_end_to_end_agreement_proven": False,
        "acceptance_standard": {
            "compare_values_not_patterns": True,
            "strict_mathematica_source_required_for_validation_claims": True,
            "document_discrepancies_instead_of_guessing": True,
            "do_not_skip_complex_symbolic_cases_due_to_runtime": True,
        },
        "requirements": requirements,
    }


def _requirement(
    requirement_id: str,
    description: str,
    status: str,
    validation_level: str,
    evidence: list[str],
    next_action: str,
    measured: dict,
) -> dict:
    return {
        "requirement_id": requirement_id,
        "description": description,
        "status": status,
        "validation_level": validation_level,
        "evidence": evidence,
        "next_action": next_action,
        "measured": measured,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.py verification requirements manifest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DIR / "verification_requirements_v1.json",
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_manifest(), indent=2), encoding="utf-8")
    print(f"Wrote verification requirements manifest to {args.output}")


if __name__ == "__main__":
    main()
