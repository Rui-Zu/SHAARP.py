import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shaarp.api import run_si_numeric
from shaarp.api import SHAARPResult, ValidationStatus
from shaarp.interactive import _closed_grid, default_interactive_material, default_interactive_system


class InteractiveSessionTests(unittest.TestCase):
    def test_default_interactive_system_is_small_and_valid(self):
        system = default_interactive_system()

        system.validate()
        self.assertEqual(len(system.layers), 3)
        self.assertEqual(float(np.asarray(system.polarimetry.theta_deg)), 20.0)
        active_layers = [layer for layer in system.layers if layer.shg_active]
        self.assertEqual(len(active_layers), 1)
        self.assertGreater(np.linalg.norm(active_layers[0].material.d_voigt()), 0.0)
        self.assertGreater(np.linalg.norm(active_layers[0].material.eps_2w().imag), 0.0)

    def test_default_interactive_material_runs_si_compat_workflow(self):
        material = default_interactive_material()

        result = run_si_numeric(
            material,
            {
                "workflow": "shaarp_si_compat",
                "normal_incidence_2omega_branch_policy": "shaarp_reference_like",
            },
        )

        self.assertEqual(result.kind, "si_numeric_shaarp_si_compat")
        self.assertEqual(
            result.validation.status,
            "mathematica_stage_validated_for_exported_36_case_si_reflected_er2w",
        )

    def test_closed_grid_includes_endpoint_when_on_grid(self):
        np.testing.assert_allclose(_closed_grid(-10.0, 10.0, 10.0), [-10.0, 0.0, 10.0])

    def test_make_interactive_session_imports_when_ipywidgets_available(self):
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            self.skipTest("ipywidgets is not installed in this environment.")

        from shaarp.interactive import make_interactive_session

        widget = make_interactive_session(default_interactive_system())

        self.assertTrue(hasattr(widget, "children"))

    def test_make_interactive_session_exposes_analytical_modes(self):
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            self.skipTest("ipywidgets is not installed in this environment.")

        from shaarp.interactive import make_interactive_session

        widget = make_interactive_session(default_interactive_system())
        mode_dropdown = widget.children[0].children[0]
        option_labels = [label for label, _value in mode_dropdown.options]

        self.assertIn("SI analytical", option_labels)
        self.assertIn("ML partial analytical", option_labels)

    def test_make_interactive_session_exposes_validation_reference_toggle(self):
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            self.skipTest("ipywidgets is not installed in this environment.")

        from shaarp.interactive import make_interactive_session

        widget = make_interactive_session(default_interactive_system())
        policy_row = widget.children[4]
        controls = {child.description: child for child in policy_row.children}

        self.assertIn("Refs", controls)
        self.assertTrue(controls["Refs"].value)

    def test_interactive_fresnel_mode_uses_gui_multilayer_workflow(self):
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            self.skipTest("ipywidgets is not installed in this environment.")

        from shaarp.interactive import make_interactive_session

        calls = []

        def fake_run_fresnel_sweep(case, angle_grid, options=None):
            calls.append((case, np.asarray(angle_grid), dict(options or {})))
            return SHAARPResult(
                kind="fresnel_sweep_gui_multilayer",
                numeric={"theta_deg": np.asarray(angle_grid), "rp": np.zeros_like(np.asarray(angle_grid, dtype=float))},
                validation=ValidationStatus("test_validation"),
            )

        with patch("shaarp.interactive.run_fresnel_sweep", fake_run_fresnel_sweep):
            widget = make_interactive_session(default_interactive_system())
            mode_dropdown = widget.children[0].children[0]
            run_button = widget.children[0].children[1]
            policy_row = widget.children[4]
            transmitted_policy = policy_row.children[0]
            validation_refs = policy_row.children[2]

            mode_dropdown.value = "fresnel_sweep"
            transmitted_policy.value = "physical_sum"
            validation_refs.value = True
            run_button.click()

        self.assertEqual(len(calls), 1)
        _case, _grid, options = calls[0]
        self.assertEqual(options["workflow"], "gui_multilayer")
        self.assertEqual(options["transmitted_wave_policy"], "physical_sum")
        self.assertTrue(options["include_validation_summary"])

    def test_interactive_notebook_includes_si_compat_direct_example(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks" / "SHAARP_py_interactive_session.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("workflow\": \"shaarp_si_compat", source)
        self.assertIn("default_interactive_material", source)
        self.assertIn("reflected_er2w", source)

    def test_interactive_notebook_includes_analytical_scaffold_example(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks" / "SHAARP_py_interactive_session.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("Direct Analytical Scaffold Example", source)
        self.assertIn("run_si_full_analytical", source)
        self.assertIn("run_ml_partial_analytical", source)
        self.assertIn("isotropic_symbolic_scaffold", source)

    def test_step_by_step_notebook_includes_si_compat_section(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks" / "SHAARP_py_step_by_step.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("Validated SHAARP.si compatibility calculation", source)
        self.assertIn("workflow\": \"shaarp_si_compat", source)
        self.assertIn("normal_incidence_2omega_branch_policy", source)

    def test_step_by_step_notebook_opening_status_is_not_stale_reduced_only_wording(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks" / "SHAARP_py_step_by_step.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("validated compatibility workflows", source)
        self.assertIn("legacy reduced examples", source)
        self.assertNotIn("The current Python code uses effective-index approximations", source)
        self.assertIn("continue expanding end-to-end Mathematica value comparisons", source)
        self.assertNotIn("the next implementation work is to port the Mathematica eigenmode", source)

    def test_step_by_step_notebook_includes_analytical_scaffold_section(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks" / "SHAARP_py_step_by_step.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("Scoped analytical scaffold calculations", source)
        self.assertIn("run_si_full_analytical", source)
        self.assertIn("run_ml_partial_analytical", source)
        self.assertIn("pnl_symbolic_scaffold", source)


if __name__ == "__main__":
    unittest.main()
