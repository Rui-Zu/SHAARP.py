"""Headless material-property validation (gui_set_material_properties).

These exercise the pure, side-effect-free validator: every real preset material
must pass with NO hard issues (the no-false-positive guarantee -- including the
LiNbO3 3m mirror-perpendicular-x1 axis setting, which differs from the symbolic
tensor's setting), while physically invalid input (non-symmetric / non-positive
/ non-finite dielectric tensors, malformed or symmetry-forbidden d-tensors) is
flagged. The d-tensor checks are convention-robust: a valid crystal in any
standard axis setting is never falsely rejected.
"""

import unittest
from dataclasses import replace

import numpy as np

from shaarp import layer_presets as lp
from shaarp import presets
from shaarp.material_validation import (
    MaterialValidationReport,
    check_material_properties,
    d_tensor_issues_and_diagnostics,
    dielectric_tensor_issues,
    validate_material_properties,
)


class PresetMaterialsHaveNoHardIssuesTests(unittest.TestCase):
    """No real preset material may produce a hard validation issue."""

    def test_all_material_presets_valid(self):
        names = presets.list_materials()
        self.assertGreaterEqual(len(names), 6)
        for name in names:
            material = presets.build_material(name)
            report = check_material_properties(material)
            self.assertIsInstance(report, MaterialValidationReport)
            self.assertTrue(report.ok, f"{name} unexpectedly flagged: {report.issues}")
            self.assertEqual(report.issues, [])

    def test_all_layer_preset_materials_valid(self):
        for name in lp.list_layer_presets():
            material = lp.build_layer_preset(name).material
            report = check_material_properties(material)
            self.assertTrue(report.ok, f"{name} unexpectedly flagged: {report.issues}")

    def test_linbo3_3m_alternate_axis_setting_not_falsely_flagged(self):
        # LiNbO3 carries the d22-family in row 1 (mirror _|_ x1); the symbolic 3m
        # tensor uses row 2 (mirror _|_ x2). The validator must not reject it.
        material = presets.build_material("linbo3_zcut_1064")
        report = check_material_properties(material)
        self.assertTrue(report.ok)
        self.assertEqual(report.diagnostics, [])  # count matches (8 == 8), no diagnostic either


class DielectricTensorTests(unittest.TestCase):
    def _eps(self):
        return presets.build_material("linbo3_zcut_1064").eps_w()

    def test_non_symmetric_is_flagged(self):
        eps = self._eps().copy()
        eps[0, 1] += 5.0
        issues = dielectric_tensor_issues(eps, "eps_w")
        self.assertTrue(any("not symmetric" in m for m in issues))

    def test_non_positive_definite_is_flagged(self):
        eps = self._eps().copy()
        eps[0, 0] = -2.0
        issues = dielectric_tensor_issues(eps, "eps_w")
        self.assertTrue(any("positive-definite" in m for m in issues))

    def test_non_finite_is_flagged_and_short_circuits(self):
        eps = self._eps().copy()
        eps[2, 2] = np.inf
        issues = dielectric_tensor_issues(eps, "eps_w")
        self.assertEqual(len(issues), 1)
        self.assertIn("non-finite", issues[0])

    def test_wrong_shape_is_flagged(self):
        issues = dielectric_tensor_issues(np.zeros((2, 2)), "eps_w")
        self.assertTrue(any("3x3" in m for m in issues))

    def test_positive_definite_check_can_be_relaxed(self):
        eps = self._eps().copy()
        eps[0, 0] = -2.0  # still symmetric, just not PD
        self.assertEqual(dielectric_tensor_issues(eps, "eps_w", require_positive_definite=False), [])


class DTensorPointGroupTests(unittest.TestCase):
    def test_wrong_shape_is_hard_issue(self):
        issues, _ = d_tensor_issues_and_diagnostics(np.zeros((3, 5)), "3m")
        self.assertTrue(any("3x6" in m for m in issues))

    def test_nonzero_d_under_centrosymmetric_group_is_hard_issue(self):
        d = presets.build_material("gaas_111_800").d_voigt()  # non-zero d
        issues, _ = d_tensor_issues_and_diagnostics(d, "m-3m")  # centrosymmetric -> SHG forbidden
        self.assertTrue(any("non-centrosymmetric" in m for m in issues))

    def test_zero_d_under_centrosymmetric_group_is_consistent(self):
        issues, diags = d_tensor_issues_and_diagnostics(np.zeros((3, 6)), "m-3m")
        self.assertEqual(issues, [])

    def test_nonzero_count_mismatch_is_soft_diagnostic_only(self):
        # LiNbO3 d-tensor (8 non-zero) declared under -43m (3 non-zero, still SHG-active):
        # not a hard issue, but a soft diagnostic about the count.
        d = presets.build_material("linbo3_zcut_1064").d_voigt()
        issues, diags = d_tensor_issues_and_diagnostics(d, "-43m")
        self.assertEqual(issues, [])
        self.assertTrue(any("non-zero Voigt entries" in m for m in diags))


class ValidateRaisesTests(unittest.TestCase):
    def test_validate_passes_on_real_material(self):
        validate_material_properties(presets.build_material("gaas_111_800"))  # must not raise

    def test_validate_raises_on_non_symmetric_eps(self):
        material = presets.build_material("linbo3_zcut_1064")
        bad_eps = material.eps_w().copy()
        bad_eps[0, 1] += 5.0
        bad = replace(material, epsilon_omega=bad_eps)
        with self.assertRaises(ValueError):
            validate_material_properties(bad)

    def test_validate_raises_on_symmetry_forbidden_d(self):
        material = presets.build_material("gaas_111_800")
        forbidden = replace(material, structure=replace(material.structure, point_group="m-3m"))
        with self.assertRaises(ValueError):
            validate_material_properties(forbidden)


if __name__ == "__main__":
    unittest.main()
