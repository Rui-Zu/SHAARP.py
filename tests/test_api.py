import tempfile
import unittest
from pathlib import Path

import numpy as np

from shaarp import (
    CrystalOrientation,
    CrystalStructure,
    Layer,
    Material,
    MultilayerSystem,
    Polarimetry,
    compare_with_mathematica,
    default_interactive_material,
    export_result,
    presets,
    run_fresnel_sweep,
    run_maker_fringes,
    run_ml_numeric,
    run_ml_partial_analytical,
    run_sample_rotation,
    run_si_full_analytical,
    run_si_numeric,
    solve_shaarp_si_reflected_shg,
)


class PublicAPITests(unittest.TestCase):
    def test_top_level_exports_include_si_compat_helpers(self):
        self.assertTrue(callable(default_interactive_material))
        self.assertTrue(callable(solve_shaarp_si_reflected_shg))

    def test_si_numeric_facade_returns_metadata_and_exports(self):
        result = run_si_numeric(
            presets.blank_nonlinear(),
            {"polarimetry": Polarimetry(theta_deg=20.0, phi_deg=[0.0, 45.0], psi_deg=0.0)},
        )

        self.assertEqual(result.kind, "si_numeric")
        self.assertIn("intensity", result.numeric)
        self.assertEqual(result.validation.status, "reduced_model_not_full_shaarp_si")
        self.assertEqual(result.conventions.field_convention, "exp(i(omega*t - k.r))")

        with tempfile.TemporaryDirectory() as tmp:
            json_path = export_result(result, Path(tmp) / "si.json")
            csv_path = export_result(result, Path(tmp) / "si.csv", format="csv")
            self.assertGreater(json_path.stat().st_size, 0)
            self.assertGreater(csv_path.stat().st_size, 0)

    def test_si_numeric_facade_can_include_stage_validation_summary(self):
        result = run_si_numeric(
            presets.blank_nonlinear(),
            {
                "polarimetry": Polarimetry(theta_deg=0.0, phi_deg=[0.0], psi_deg=0.0),
                "include_validation_summary": True,
            },
        )

        summary = result.stages["mathematica_stage_summary"]
        self.assertEqual(summary["status"], "partial_stage_value_comparison_not_full_shaarp_si_agreement")
        self.assertFalse(summary["full_shaarp_si_agreement_claimed"])
        self.assertEqual(summary["passed_value_comparison_count"], 1116)
        stages = {stage["stage_id"]: stage for stage in summary["stages"]}
        self.assertEqual(stages["pnl"]["comparison_status"], "passed")
        self.assertEqual(stages["inhomogeneous_croots"]["comparison_status"], "passed")
        self.assertEqual(stages["angle_root_equations"]["comparison_status"], "passed")
        self.assertEqual(stages["angle_root_labels"]["comparison_status"], "diagnostic_only")
        self.assertEqual(result.validation.status, "reduced_model_not_full_shaarp_si")

    def test_si_numeric_facade_exposes_validated_shaarp_si_compat_workflow(self):
        result = run_si_numeric(
            _si_compat_api_material(),
            {
                "workflow": "shaarp_si_compat",
                "polarimetry": Polarimetry(theta_deg=0.0),
                "incident_polarization": "s",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )

        self.assertEqual(result.kind, "si_numeric_shaarp_si_compat")
        self.assertIn("reflected_er2w", result.numeric)
        self.assertIn("boundary_residual_norm", result.numeric)
        self.assertEqual(
            result.validation.status,
            "mathematica_stage_validated_for_exported_36_case_si_reflected_er2w",
        )
        self.assertIn("normal_incidence_2omega_branch_policy", result.stages)

    def test_compat_frame_bridge_taas_my_mirror_and_gaas_referee(self):
        """F42 fence: the shaarp_si_compat adapter's FRAME BRIDGE.

        The compat stack is SHAARP.si-V1.04-faithful: internally DOWNWARD-propagating, and its
        ``orientation_matrix`` input is V1.04's ``a`` (COLUMNS = crystal Z axes in lab; the
        36-case validation fed live-Mathematica ``a`` matrices, incl. 30 tilted+anisotropic
        oblique cases). The port's Material orientation is the ROWS convention consumed by the
        UPWARD (ml-frame) solvers. The old adapter passed rows-M as ``a`` directly, so eps was
        consumed at the transposed orientation while d was rotated with rows-M — eps and d
        described two different crystals, breaking the m_y mirror of the tilted TaAs (112)
        (nonzero s-out for s-in). The bridge is rows_si = M·diag(-1,-1,1), a = rows_si.T, with d
        rotated by the SAME rows_si (exact for intensities: global z-mirror ≡ 180° azimuth up to
        a global d sign).

        Discriminators: (1) TaAs s-in must give s-out ~ 0 (m_y); the broken adapter gave
        s/p ~ 0.5. (2) TaAs's compat eo Snell root at 45 deg must sit ON the published rising
        Fig 7(b) curve (~4.336 + 2.080j; the broken adapter's diagonal-lean crystal gave
        4.315 + 2.092j). (3) GaAs (isotropic eps — immune to the V1.04 numeric-branch
        birefringent k-par quirk documented in the validation notes) is the end-to-end referee:
        compat must equal the generic solver's channel intensities to ~1e-6."""
        import math

        from shaarp.casestudy_materials import build_casestudy_material
        from shaarp.shg import solve_single_interface_shg
        from shaarp.tensors import rotate_d_voigt_crystal_to_lab, rotate_rank2_crystal_to_lab

        taas = build_casestudy_material("TaAs (112)", wavelength_um=0.8)
        res = run_si_numeric(taas, {"workflow": "shaarp_si_compat",
                                    "polarimetry": Polarimetry(theta_deg=45.0),
                                    "incident_polarization": "s"})
        e = np.asarray(res.numeric["reflected_er2w"])
        i_s = abs(e[1]) ** 2
        i_p = abs(e[0]) ** 2 + abs(e[2]) ** 2
        self.assertGreater(i_p, 1e-6, "vacuous TaAs compat result")
        self.assertLess(i_s / i_p, 1e-20,
                        f"compat TaAs s-in gives s-out/p-out = {i_s / i_p:.3e}: the m_y mirror is "
                        "broken again — eps/d frame inconsistency in the adapter (regression)")
        roots = res.stages["linear_omega"].roots
        n_eo = math.sin(math.radians(45.0)) / np.sin(complex(roots.theta_ext))
        self.assertAlmostEqual(n_eo.real, 4.3356, delta=5e-3,
                               msg=f"compat TaAs eo root n = {n_eo.real:.4f} off the published "
                                   "rising Fig 7(b) curve at 45 deg (~4.336)")
        self.assertAlmostEqual(n_eo.imag, 2.0796, delta=5e-3,
                               msg=f"compat TaAs eo root k = {n_eo.imag:.4f} off the published "
                                   "curve at 45 deg (~2.080)")

        gaas = build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)
        rows = gaas.orientation.rotation_matrix()
        ew = rotate_rank2_crystal_to_lab(gaas.epsilon_omega, rows)
        e2 = rotate_rank2_crystal_to_lab(gaas.epsilon_2omega, rows)
        dl = rotate_d_voigt_crystal_to_lab(gaas.d_voigt_pm_v, rows)
        for pol in ("s", "p"):
            g = solve_single_interface_shg(ew, e2, dl, incident_index_omega=1.0 + 0j,
                                           incident_index_2omega=1.0 + 0j,
                                           incident_theta_rad=math.radians(45.0),
                                           incident_polarization=pol, omega=1.0, mu=1.0, eps0=1.0)
            c = np.asarray(g.coefficients)  # validated convention: [0]=reflected s, [1]=p
            gs, gp = abs(complex(c[0])) ** 2, abs(complex(c[1])) ** 2
            r = run_si_numeric(gaas, {"workflow": "shaarp_si_compat",
                                      "polarimetry": Polarimetry(theta_deg=45.0),
                                      "incident_polarization": pol})
            ec = np.asarray(r.numeric["reflected_er2w"])
            cs, cp = abs(ec[1]) ** 2, abs(ec[0]) ** 2 + abs(ec[2]) ** 2
            for got, want, ch in ((cs, gs, "s"), (cp, gp, "p")):
                self.assertLess(abs(got - want) / max(want, 1e-300), 1e-5,
                                f"GaAs compat {pol}-in {ch}-out {got:.6e} != generic {want:.6e} — "
                                "the compat frame bridge no longer matches the validated solver")

    def test_fresnel_sweep_facade_runs_small_grid(self):
        result = run_fresnel_sweep(
            presets.blank_linear(),
            angle_grid=[0.0, 10.0],
            options={"incident_index": 1.0},
        )

        self.assertEqual(result.kind, "fresnel_sweep")
        np.testing.assert_allclose(result.numeric["theta_deg"], [0.0, 10.0])
        self.assertIn("Poynting-flux power", result.conventions.intensity_normalization)

    def test_fresnel_sweep_facade_can_include_component_validation_summary(self):
        result = run_fresnel_sweep(
            presets.blank_linear(),
            angle_grid=[0.0],
            options={"incident_index": 1.0, "include_validation_summary": True},
        )

        summary = result.stages["mathematica_validation_artifacts"]
        self.assertEqual(
            summary["status"],
            "gui_fresnel_list_selected_policy_mathematica_validated_with_transmitted_sum_caveat",
        )
        self.assertTrue(summary["gui_fresnel_list_agreement_claimed"])
        self.assertEqual(summary["component_references"]["solve_fresnel"]["case_count"], 20)
        self.assertEqual(summary["component_references"]["solve_fresnel_n"]["case_count"], 6)
        self.assertEqual(summary["gui_fresnel_reference"]["case_count"], 3)
        self.assertEqual(
            summary["gui_fresnel_reference"]["comparison_policy"],
            "SHAARP-compatible selected transmitted wave matches Mathematica; physical transmitted sum is a documented caveat.",
        )
        self.assertEqual(
            summary["discrepancy_reports"][0]["classification"],
            "mathematica_shaarp_bug_or_ambiguity",
        )
        self.assertEqual(result.validation.status, "staged_python_not_fully_mathematica_validated")

    def test_compare_with_mathematica_uses_values_not_patterns(self):
        result = run_si_numeric(
            presets.blank_nonlinear(),
            {"polarimetry": Polarimetry(theta_deg=0.0, phi_deg=[0.0, 90.0], psi_deg=0.0)},
        )
        reference = {
            "keys": ["intensity"],
            "intensity": [
                {"real": float(result.numeric["intensity"][0]), "imag": 0.0},
                {"real": float(result.numeric["intensity"][1]) + 0.1, "imag": 0.0},
            ],
        }

        comparisons = compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

        self.assertEqual(len(comparisons), 1)
        self.assertFalse(comparisons[0].passed)
        self.assertGreater(comparisons[0].max_abs_error, 0.01)

    def test_compare_with_mathematica_can_compare_stage_paths(self):
        result = run_si_numeric(
            _si_compat_api_material(),
            {
                "workflow": "shaarp_si_compat",
                "polarimetry": Polarimetry(theta_deg=30.0),
                "incident_polarization": "s",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )
        residual = float(result.numeric["boundary_residual_norm"])
        reference = {
            "keys": ["numeric.boundary_residual_norm", "stages.normal_incidence_2omega_branch_policy"],
            "numeric": {"boundary_residual_norm": {"real": residual, "imag": 0.0}},
            "stages": {"normal_incidence_2omega_branch_policy": "shaarp_reference_like"},
        }

        comparisons = compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(comparison.passed for comparison in comparisons))
        self.assertEqual(comparisons[1].shape, ())

    def test_compare_with_mathematica_can_compare_validation_and_convention_paths(self):
        result = run_si_numeric(
            _si_compat_api_material(),
            {
                "workflow": "shaarp_si_compat",
                "polarimetry": Polarimetry(theta_deg=30.0),
                "incident_polarization": "s",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )
        reference = {
            "keys": ["validation.status", "conventions.field_convention"],
            "validation": {"status": "mathematica_stage_validated_for_exported_36_case_si_reflected_er2w"},
            "conventions": {"field_convention": "exp(i(omega*t - k.r))"},
        }

        comparisons = compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

        self.assertEqual(len(comparisons), 2)
        self.assertTrue(all(comparison.passed for comparison in comparisons))
        self.assertTrue(all(comparison.shape == () for comparison in comparisons))

    def test_compare_with_mathematica_can_compare_validation_note_sequences(self):
        result = run_si_numeric(
            _si_compat_api_material(),
            {
                "workflow": "shaarp_si_compat",
                "polarimetry": Polarimetry(theta_deg=30.0),
                "incident_polarization": "s",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )
        reference = {
            "keys": ["validation.notes"],
            "validation": {"notes": list(result.validation.notes)},
        }

        comparisons = compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

        self.assertEqual(len(comparisons), 1)
        self.assertTrue(comparisons[0].passed)
        self.assertEqual(comparisons[0].shape, ())

    def test_compare_with_mathematica_missing_python_path_lists_available_keys(self):
        result = run_si_numeric(
            _si_compat_api_material(),
            {
                "workflow": "shaarp_si_compat",
                "polarimetry": Polarimetry(theta_deg=30.0),
                "incident_polarization": "s",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )
        reference = {
            "keys": ["stages.missing_policy"],
            "stages": {"missing_policy": "shaarp_reference_like"},
        }

        with self.assertRaisesRegex(ValueError, "available keys.*normal_incidence_2omega_branch_policy"):
            compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

    def test_compare_with_mathematica_missing_reference_path_lists_available_keys(self):
        result = run_si_numeric(
            _si_compat_api_material(),
            {
                "workflow": "shaarp_si_compat",
                "polarimetry": Polarimetry(theta_deg=30.0),
                "incident_polarization": "s",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )
        reference = {
            "keys": ["stages.normal_incidence_2omega_branch_policy"],
            "stages": {"other_policy": "shaarp_reference_like"},
        }

        with self.assertRaisesRegex(ValueError, "available keys.*other_policy"):
            compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

    def test_compare_with_mathematica_stage_path_string_mismatch_fails(self):
        result = run_si_numeric(
            _si_compat_api_material(),
            {
                "workflow": "shaarp_si_compat",
                "polarimetry": Polarimetry(theta_deg=30.0),
                "incident_polarization": "s",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )
        reference = {
            "keys": ["stages.normal_incidence_2omega_branch_policy"],
            "stages": {"normal_incidence_2omega_branch_policy": "python_exact_zero"},
        }

        comparisons = compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

        self.assertEqual(len(comparisons), 1)
        self.assertFalse(comparisons[0].passed)
        self.assertEqual(comparisons[0].max_abs_error, float("inf"))

    def test_compare_with_mathematica_stage_path_boolean_mismatch_fails_as_metadata(self):
        result = run_ml_numeric(_maker_api_system(), {"include_validation_summary": True})
        reference = {
            "keys": ["stages.mathematica_validation_artifacts.full_shaarp_ml_agreement_claimed"],
            "stages": {"mathematica_validation_artifacts": {"full_shaarp_ml_agreement_claimed": True}},
        }

        comparisons = compare_with_mathematica(reference, result, {"atol": 1e-12, "rtol": 1e-12})

        self.assertEqual(len(comparisons), 1)
        self.assertFalse(comparisons[0].passed)
        self.assertEqual(comparisons[0].max_abs_error, float("inf"))
        self.assertEqual(comparisons[0].shape, ())

    def test_analytical_facades_return_scoped_symbolic_results_with_validation_caveats(self):
        si = run_si_full_analytical(None, {"workflow": "isotropic_symbolic_scaffold", "simplify": False})

        self.assertEqual(si.kind, "si_full_analytical_isotropic_scaffold")
        self.assertEqual(si.numeric, {})
        self.assertIn("not_full_mathematica_validated", si.validation.status)
        self.assertIn("linear_coefficients", si.stages)
        self.assertIn("shg_boundary_coefficients", si.stages)
        self.assertIn("boundary_residual", si.stages)
        self.assertIn("symbol_metadata", si.stages)

        ml = run_ml_partial_analytical({"point_group": "3m"}, {"simplify": True})

        self.assertEqual(ml.kind, "ml_partial_analytical_pnl_scaffold")
        self.assertEqual(ml.numeric, {})
        self.assertIn("not_full_mathematica_validated", ml.validation.status)
        self.assertEqual(ml.stages["point_group"], "3m")
        self.assertIn("d_voigt", ml.stages)
        self.assertIn("partial_sources", ml.stages)

        self.assertIn("h", ml.stages["symbol_metadata"])
        self.assertEqual(
            ml.stages["symbol_metadata"]["point_group_tensor_source_reference"],
            "benchmarks/mathematica_reference/point_group_tensor_patterns_from_si_source_v1.json",
        )
        self.assertEqual(
            ml.stages["symbol_metadata"]["ml_partial_pnl_symbolic_reference"],
            "benchmarks/mathematica_reference/wolfram_live_symbolic_ml_partial_pnl_reference_v1.json",
        )

        with tempfile.TemporaryDirectory() as tmp:
            exported = export_result(si, Path(tmp) / "si_symbolic.json", format="json")
            self.assertIn("linear_coefficients", exported.read_text(encoding="utf-8"))

    def test_polarimetry_facade_workflows_faithfully_expose_validated_solvers(self):
        """The 'polarimetry' workflows expose the d-extraction polarimetry through the public
        API. They are validated by asserting the facade output is byte-identical to the
        directly-called (and separately, rigorously validated) symbolic solvers -- so the
        facade faithfully surfaces the validated capability -- plus the scaffold remains the
        DEFAULT and an unknown workflow is rejected."""
        import numpy as np
        import sympy as sp

        from shaarp.multilayer_basis import build_multilayer_2omega_basis, build_multilayer_omega_basis
        from shaarp.multilayer_shg_symbolic import solve_single_film_shg_symbolic_polarimetry
        from shaarp.symbolic import solve_si_shg_full_analytical_symbolic
        from shaarp.waves import EPS0, MU0

        # ---- SI full-analytical polarimetry facade == direct solver ----
        d = sp.zeros(3, 6)
        d[0, 3] = d[1, 4] = d[2, 5] = sp.Symbol("d14")
        phi, psi = sp.Symbol("phi", real=True), sp.Symbol("psi", real=True)
        si = run_si_full_analytical(
            {"point_group": "-43m", "d_voigt_lab": d, "phi_symbol": phi, "analyzer_symbol": psi},
            {"workflow": "polarimetry"},
        )
        self.assertEqual(si.kind, "si_full_analytical_polarimetry")
        self.assertIn("not_full_mathematica_validated", si.validation.status)
        self.assertFalse(si.validation.is_mathematica_validated)
        direct = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=2.2**2, eps_y_omega=2.2**2, eps_z_omega=2.5**2,
            eps_x_2omega=2.4**2, eps_y_2omega=2.4**2, eps_z_2omega=2.7**2,
            d_voigt_lab=d, incident_theta_rad=0.5, phi_symbol=phi, analyzer_symbol=psi, simplify=False,
        )
        self.assertEqual(si.stages["reflected_p_2omega"], str(direct.reflected_p))
        self.assertEqual(si.stages["analyzed_intensity"], str(direct.intensity))

        # ---- ML partial-analytical polarimetry facade == direct solver ----
        dml = sp.zeros(3, 6)
        dml[0, 3] = dml[1, 4] = dml[2, 5] = sp.Symbol("d14")
        h = sp.Symbol("h", positive=True)
        ml = run_ml_partial_analytical(
            {"point_group": "-43m", "d_voigt_symbolic": dml, "phi_symbol": phi, "thickness_symbol": h},
            {"workflow": "polarimetry"},
        )
        self.assertEqual(ml.kind, "ml_partial_analytical_polarimetry")
        self.assertIn("not_full_mathematica_validated", ml.validation.status)
        common = dict(
            incident_index=1.0, incident_theta_rad=0.5,
            layer_epsilon_lab=[np.diag([2.1**2] * 3).astype(complex)],
            substrate_epsilon_lab=np.diag([1.5**2] * 3).astype(complex), omega=1.0,
        )
        bs = build_multilayer_omega_basis(incident_polarization="s", **common)
        bp = build_multilayer_omega_basis(incident_polarization="p", **common)
        eps2 = [np.diag([2.25**2] * 3).astype(complex)]
        b2 = build_multilayer_2omega_basis(
            top_index_2omega=1.0, tangential_index_omega=1.0, incident_theta_rad=0.5,
            layer_epsilon_2omega_lab=eps2, substrate_epsilon_2omega_lab=np.diag([1.55**2] * 3).astype(complex), omega_2=2.0,
        )
        dsol = solve_single_film_shg_symbolic_polarimetry(
            omega_basis_s=bs, omega_basis_p=bp, twoomega_basis=b2, d_voigt_symbolic=dml,
            # natural units -- the convention tests/test_ml_units_convention.py locked
            # for every multilayer entry point. This reference call was the last
            # holdout still passing SI constants; because the facade passed them too, this
            # equivalence fence stayed GREEN THROUGH the SI-units bug it should have caught.
            eps_2omega_lab=eps2[0], thickness_symbol=h, phi_symbol=phi, mu=1.0, eps0=1.0,
        )
        self.assertEqual(ml.stages["reflected_p_2omega"], str(dsol.reflected_p))

        # scaffold stays the DEFAULT; unknown workflow rejected
        self.assertEqual(run_si_full_analytical({}).kind, "si_full_analytical_isotropic_scaffold")
        self.assertEqual(run_ml_partial_analytical({}).kind, "ml_partial_analytical_pnl_scaffold")
        with self.assertRaises(ValueError):
            run_si_full_analytical({}, {"workflow": "nope"})
        with self.assertRaises(ValueError):
            run_ml_partial_analytical({}, {"workflow": "nope"})

    def test_multilayer_system_can_be_passed_to_fresnel_facade(self):
        system = MultilayerSystem(
            wavelength_um=1.064,
            polarimetry=Polarimetry(theta_deg=0.0, phi_deg=0.0, psi_deg=0.0),
            layers=[
                Layer("air in", presets.air(), shg_active=False),
                Layer("film", presets.blank_linear(), thickness_um=0.1, shg_active=False),
                Layer("air out", presets.air(), shg_active=False),
            ],
        )

        result = run_fresnel_sweep(system, angle_grid=[0.0])

        self.assertEqual(result.kind, "fresnel_sweep")
        np.testing.assert_allclose(result.numeric["rp"], [0.0], atol=1e-12)

    def test_maker_fringes_facade_can_include_validation_caveats(self):
        result = run_maker_fringes(
            _maker_api_system(),
            angle_grid=[0.0],
            options={"include_validation_summary": True},
        )

        artifacts = result.stages["mathematica_validation_artifacts"]
        self.assertEqual(
            artifacts["status"],
            "maker_outputs_nonsingular_mathematica_validated_with_phase_matching_diagnostic_and_transmitted_sum_caveat",
        )
        self.assertEqual(
            artifacts["mathematica_reference"],
            "benchmarks/mathematica_reference/maker_fringes_reference_v1.json",
        )
        self.assertEqual(
            artifacts["fine_grid_summary"]["status"],
            "maker_outputs_match_all_compared_cases",
        )
        self.assertEqual(artifacts["fine_grid_summary"]["nonsingular_case_count"], 3)
        self.assertEqual(
            artifacts["fringe_resolution_summary"]["status"],
            "maker_outputs_match_all_compared_cases",
        )
        self.assertEqual(artifacts["fringe_resolution_summary"]["nonsingular_case_count"], 1)
        self.assertEqual(
            artifacts["phase_matching_local_summary"]["status"],
            "nonsingular_maker_outputs_match_with_phase_matching_diagnostic",
        )
        self.assertEqual(artifacts["phase_matching_local_summary"]["diagnostic_fail_count"], 7)
        self.assertEqual(
            artifacts["high_oblique_stage_summary"]["status"],
            "high_oblique_12_stage_passes",
        )
        self.assertEqual(
            artifacts["high_oblique_solvefresneln_summary"]["status"],
            "theta12_solvefresneln_matrix_and_coefficients_match",
        )
        self.assertEqual(
            artifacts["high_oblique_linear_stage_summary"]["status"],
            "python_direct_linear_stage_matches_mathematica",
        )
        self.assertLess(artifacts["high_oblique_linear_stage_summary"]["max_abs_error"], 1e-10)
        reports = {report["classification"]: report for report in artifacts["discrepancy_reports"]}
        self.assertIn("mathematica_shaarp_bug_or_ambiguity", reports)
        self.assertIn("singular_or_phase_matched_case", reports)
        self.assertIn("transmitted 2omega wave selection", reports["mathematica_shaarp_bug_or_ambiguity"]["first_failing_stage"])
        self.assertIn("phase-matching-like solveInhom", reports["singular_or_phase_matched_case"]["first_failing_stage"])
        self.assertEqual(
            result.validation.status,
            "maker_outputs_nonsingular_mathematica_validated_with_phase_matching_diagnostic_and_transmitted_sum_caveat",
        )

    def test_maker_fringes_physical_sum_policy_is_not_labeled_as_mathematica_gui_default(self):
        result = run_maker_fringes(
            _maker_api_system(),
            angle_grid=[0.0],
            options={"transmitted_wave_policy": "physical_sum", "include_validation_summary": True},
        )

        self.assertEqual(result.validation.status, "physically_motivated_not_mathematica_gui_default")
        self.assertEqual(result.conventions.maker_transmitted_policy, "physical_sum")
        self.assertIn("mathematica_validation_artifacts", result.stages)

    def test_ml_numeric_facade_can_include_validation_coverage(self):
        result = run_ml_numeric(_maker_api_system(), {"include_validation_summary": True})

        artifacts = result.stages["mathematica_validation_artifacts"]
        self.assertEqual(
            artifacts["status"],
            "python_regression_with_component_references_not_full_ml_agreement",
        )
        self.assertFalse(artifacts["full_shaarp_ml_agreement_claimed"])
        self.assertEqual(artifacts["python_regression_benchmarks"]["multilayer_system"]["case_count"], 20)
        self.assertEqual(artifacts["python_regression_benchmarks"]["multilayer_polarimetry"]["case_count"], 20)
        self.assertEqual(artifacts["component_mathematica_references"]["solve_fresnel_n"]["case_count"], 6)
        self.assertEqual(
            artifacts["gui_mathematica_workflows"]["maker_fringes"]["fine_grid_summary"]["status"],
            "maker_outputs_match_all_compared_cases",
        )
        self.assertEqual(
            artifacts["gui_mathematica_workflows"]["maker_fringes"]["phase_matching_local_summary"]["diagnostic_fail_count"],
            7,
        )
        self.assertEqual(
            artifacts["gui_mathematica_workflows"]["fresnel_sweep"]["status"],
            "gui_fresnel_list_selected_policy_mathematica_validated_with_transmitted_sum_caveat",
        )
        self.assertTrue(artifacts["gui_mathematica_workflows"]["fresnel_sweep"]["gui_fresnel_list_agreement_claimed"])
        self.assertEqual(
            artifacts["gui_mathematica_workflows"]["sample_rotation"]["comparison_summary"]["status"],
            "sample_rotation_nonsingular_match_with_phase_matching_diagnostic",
        )
        self.assertEqual(
            artifacts["gui_mathematica_workflows"]["sample_rotation"]["comparison_summary"]["diagnostic_fail_count"],
            1,
        )
        self.assertEqual(result.validation.status, "staged_python_not_fully_mathematica_validated")

    def test_sample_rotation_facade_can_include_python_regression_coverage(self):
        result = run_sample_rotation(
            _maker_api_system(),
            [0.0, 15.0],
            {"include_validation_summary": True},
        )

        artifacts = result.stages["validation_artifacts"]
        self.assertIn("reflected_parallel_intensity", result.numeric)
        self.assertIn("reflected_perpendicular_intensity", result.numeric)
        self.assertIn("transmitted_parallel_intensity", result.numeric)
        self.assertIn("transmitted_perpendicular_intensity", result.numeric)
        np.testing.assert_allclose(result.numeric["intensity"], result.numeric["reflected_parallel_intensity"])
        self.assertEqual(artifacts["status"], "mathematica_sample_rotation_nonsingular_match_with_phase_matching_diagnostic")
        self.assertFalse(artifacts["mathematica_sample_rotation_agreement_claimed"])
        self.assertEqual(artifacts["mathematica_reference"]["case_count"], 3)
        self.assertEqual(artifacts["comparison_summary"]["nonsingular_fail_count"], 0)
        self.assertEqual(artifacts["comparison_summary"]["diagnostic_fail_count"], 1)
        self.assertEqual(artifacts["python_regression_benchmark"]["case_count"], 20)
        self.assertIn(15.0, artifacts["python_regression_benchmark"]["sample_azimuth_deg"])
        self.assertIn(72.0, artifacts["python_regression_benchmark"]["sample_azimuth_deg"])
        self.assertEqual(result.validation.status, "mathematica_nonsingular_match_with_phase_matching_diagnostic")


def _maker_api_system():
    structure = CrystalStructure(point_group="1")
    orientation = CrystalOrientation.from_cubic_miller_surface((1, 2, 3))
    top = Material(
        name="air",
        structure=structure,
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3),
        epsilon_2omega=np.eye(3) * (1.04**2),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    film = Material(
        name="maker-film",
        structure=structure,
        orientation=orientation,
        epsilon_omega=np.diag([1.58**2, 1.76**2, 2.01**2]),
        epsilon_2omega=np.diag([2.65 + 0.05j, 3.05 + 0.04j, 3.55 + 0.03j]),
        d_voigt_pm_v=np.array(
            [
                [0.35 + 0.08j, -0.12j, 0.5, 0.11, 0.04j, -0.27],
                [0.18j, 0.42, -0.09 + 0.04j, 0.31j, 0.63, -0.22],
                [-0.08, 0.17j, 0.41 + 0.09j, -0.34, 0.48j, 0.21],
            ],
            dtype=complex,
        ),
    )
    substrate = Material(
        name="substrate",
        structure=structure,
        orientation=CrystalOrientation(),
        epsilon_omega=np.diag([1.32**2, 1.41**2, 1.49**2]),
        epsilon_2omega=np.diag([1.8 + 0.02j, 2.05 + 0.02j, 2.3 + 0.01j]),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )
    return MultilayerSystem(
        wavelength_um=1.064,
        polarimetry=Polarimetry(theta_deg=0.0, phi_deg=30.0, psi_deg=45.0, ellipticity_deg=10.0),
        layers=[
            Layer("top", top, shg_active=False),
            Layer("film", film, thickness_um=0.18, shg_active=True),
            Layer("substrate", substrate, shg_active=False),
        ],
    )


def _si_compat_api_material():
    return Material(
        name="si-compat-api",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation.from_cubic_miller_surface((1, 2, 3)),
        epsilon_omega=np.diag([1.58**2, 1.76**2, 2.01**2]),
        epsilon_2omega=np.diag([2.65 + 0.05j, 3.05 + 0.04j, 3.55 + 0.03j]),
        d_voigt_pm_v=np.array(
            [
                [0.35 + 0.08j, -0.12j, 0.5, 0.11, 0.04j, -0.27],
                [0.18j, 0.42, -0.09 + 0.04j, 0.31j, 0.63, -0.22],
                [-0.08, 0.17j, 0.41 + 0.09j, -0.34, 0.48j, 0.21],
            ],
            dtype=complex,
        ),
    )


if __name__ == "__main__":
    unittest.main()
