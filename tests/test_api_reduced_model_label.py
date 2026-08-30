"""Lock in the explicit reduced-model labeling on the public API (modular_user_api
'label/retire reduced paths' item). ValidationStatus/SHAARPResult now expose
queryable is_reduced_model / is_mathematica_validated instead of forcing callers
to string-match status text."""

import unittest

from shaarp.api import ValidationStatus, run_maker_fringes, run_ml_numeric
from shaarp.interactive_multilayer import build_demo_multilayer_system


class ReducedModelLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = build_demo_multilayer_system()

    def test_validationstatus_is_reduced_model_flag(self):
        self.assertTrue(ValidationStatus("reduced_model_not_full_shaarp_ml").is_reduced_model)
        self.assertFalse(ValidationStatus("maker_outputs_nonsingular_mathematica_validated").is_reduced_model)

    def test_validationstatus_is_mathematica_validated_flag(self):
        self.assertTrue(ValidationStatus("maker_outputs_nonsingular_mathematica_validated_with_phase_matching_diagnostic").is_mathematica_validated)
        self.assertFalse(ValidationStatus("reduced_model_not_full_shaarp_ml").is_mathematica_validated)
        self.assertFalse(ValidationStatus("staged_python_not_fully_mathematica_validated").is_mathematica_validated)
        self.assertFalse(ValidationStatus("maker_outputs_have_unresolved_nonsingular_mismatch").is_mathematica_validated)

    def test_genuine_reduced_path_is_labeled_reduced(self):
        """The actual reduced effective-index path (single_interface_intensity,
        model='reduced_effective_index') must be flagged reduced via its warnings."""
        import warnings as _w

        from shaarp import presets
        from shaarp.config import Polarimetry
        from shaarp.single_interface import single_interface_intensity

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            r = single_interface_intensity(presets.linbo3_1120_xcut(), Polarimetry(theta_deg=20.0))
        self.assertEqual(r.model, "reduced_effective_index")
        self.assertTrue(r.warnings)
        # A ValidationStatus carrying those warnings is correctly classified reduced.
        vs = ValidationStatus("reduced_effective_index", notes=tuple(r.warnings))
        self.assertTrue(vs.is_reduced_model)
        self.assertFalse(vs.is_mathematica_validated)

    def test_run_ml_numeric_is_staged_not_validated_and_not_reduced(self):
        """run_ml_numeric is the STAGED numeric path: not Mathematica-validated, and
        not the reduced effective-index model (distinct categories)."""
        result = run_ml_numeric(self.system)
        self.assertFalse(result.validation.is_mathematica_validated,
                         f"ml_numeric is staged-not-validated; status={result.validation.status}")
        self.assertIn("staged", result.validation.status.lower())

    def test_run_maker_fringes_is_not_reduced_and_is_validated(self):
        """The validated Maker facade must NOT be reduced and SHOULD read validated."""
        result = run_maker_fringes(self.system, [0.0, 20.0])
        self.assertFalse(result.is_reduced_model, f"maker should not be reduced; status={result.validation.status}")
        self.assertTrue(result.validation.is_mathematica_validated, f"status={result.validation.status}")


if __name__ == "__main__":
    unittest.main()
