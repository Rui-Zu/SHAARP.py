import unittest

from shaarp.config import MultilayerSystem
from shaarp.interactive_multilayer import build_demo_multilayer_system
from shaarp.layer_editor_model import LayerEditorModel, LayerSpec, ValidationReport


class LayerEditorModelTests(unittest.TestCase):
    def setUp(self):
        self.system = build_demo_multilayer_system()
        self.model = LayerEditorModel.from_system(self.system)

    def test_roundtrip_from_and_build_system(self):
        report = self.model.validate()
        self.assertTrue(report.ok, msg=str(report.errors))
        rebuilt = self.model.build_system()
        self.assertIsInstance(rebuilt, MultilayerSystem)
        self.assertEqual(len(rebuilt.layers), len(self.system.layers))
        # rebuilt system must actually solve (data model produces a valid input)
        from dataclasses import replace

        from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_system_polarimetry
        ps = replace(rebuilt, polarimetry=replace(rebuilt.polarimetry, theta_deg=20.0))
        result = solve_multilayer_shg_from_system_polarimetry(ps, mu=1.0, eps0=1.0,
                                                              inhomogeneous_source_policy="forward_only",
                                                              inhomogeneous_solution_policy="solve")
        import numpy as np
        self.assertLess(float(np.linalg.norm(result.fundamental.residual)), 1e-9)

    def test_add_remove_move_and_set_ops(self):
        film_material = self.system.layers[1].material
        n0 = len(self.model.layers)
        # insert a new interior passive layer before the substrate
        self.model.add_layer(LayerSpec("extra", film_material, thickness_um=0.15, shg_active=False), index=2)
        self.assertEqual(len(self.model.layers), n0 + 1)
        self.assertTrue(self.model.validate().ok)
        # reorder the two interior layers
        self.model.move_layer(1, 2)
        self.assertTrue(self.model.validate().ok)
        # edit thickness + active flag
        self.model.set_thickness(1, 0.3).set_active(1, True)
        self.assertTrue(self.model.validate().ok)
        # remove the extra layer
        self.model.remove_layer(self.model.layers.index(next(l for l in self.model.layers if l.name == "extra")))
        self.assertEqual(len(self.model.layers), n0)

    def test_validation_catches_constraint_violations(self):
        # no active layer
        m = LayerEditorModel.from_system(self.system)
        for i in range(len(m.layers)):
            m.set_active(i, False)
        r = m.validate()
        self.assertFalse(r.ok)
        self.assertTrue(any("active" in e for e in r.errors))

        # interior layer missing thickness
        m2 = LayerEditorModel.from_system(self.system)
        m2.set_thickness(1, None)
        r2 = m2.validate()
        self.assertFalse(r2.ok)
        self.assertTrue(any("finite thickness" in e for e in r2.errors))

        # substrate given a finite thickness (must be semi-infinite)
        m3 = LayerEditorModel.from_system(self.system)
        m3.set_thickness(len(m3.layers) - 1, 0.2)
        r3 = m3.validate()
        self.assertFalse(r3.ok)
        self.assertTrue(any("substrate" in e for e in r3.errors))

        # too few layers
        m4 = LayerEditorModel(layers=[LayerSpec("only", self.system.layers[1].material)])
        self.assertFalse(m4.validate().ok)

    def test_build_raises_on_invalid_state(self):
        m = LayerEditorModel.from_system(self.system)
        m.set_active(1, False)  # removes the only active layer
        with self.assertRaises(ValueError):
            m.build_system()

    def test_report_type(self):
        self.assertIsInstance(self.model.validate(), ValidationReport)


if __name__ == "__main__":
    unittest.main()
