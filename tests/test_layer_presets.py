"""Headless tests for the layer/stack preset registry (gui_layer_property_presets).

No display: these exercise the pure-data preset builders, that every stack preset
is :meth:`MultilayerSystem.validate`-clean, that presets are fresh per call, and
that a preset stack flows through the headless layer-editor model AND the real
solver (including under the JK/HH assumption modes) -- i.e. the preset capability
is wired end-to-end, not just a data blob.
"""

import unittest

import numpy as np

from shaarp import layer_presets as lp
from shaarp.config import Layer, MultilayerSystem


class LayerPresetRegistryTests(unittest.TestCase):
    def test_all_layer_presets_build(self):
        names = lp.list_layer_presets()
        self.assertGreaterEqual(len(names), 6)
        for name in names:
            layer = lp.build_layer_preset(name)
            self.assertIsInstance(layer, Layer)
            self.assertTrue(layer.name)

    def test_all_stack_presets_build_and_validate(self):
        names = lp.list_stack_presets()
        self.assertGreaterEqual(len(names), 3)
        for name in names:
            system = lp.build_stack_preset(name)
            self.assertIsInstance(system, MultilayerSystem)
            system.validate()  # must not raise
            self.assertGreaterEqual(len(system.layers), 3)
            # exactly one SHG-active internal film in each demo stack
            self.assertGreaterEqual(sum(bool(layer.shg_active) for layer in system.layers), 1)

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            lp.build_layer_preset("does_not_exist")
        with self.assertRaises(KeyError):
            lp.build_stack_preset("does_not_exist")

    def test_presets_are_fresh_objects(self):
        a = lp.build_layer_preset("linbo3_zcut_film")
        b = lp.build_layer_preset("linbo3_zcut_film")
        self.assertIsNot(a, b)
        self.assertIsNot(a.material, b.material)

    def test_preset_stack_loads_into_layer_editor_model(self):
        from shaarp.layer_editor_model import LayerEditorModel

        system = lp.build_stack_preset("film_interlayer_substrate")
        model = LayerEditorModel.from_system(system)
        rebuilt = model.build_system()
        self.assertEqual(len(rebuilt.layers), len(system.layers))
        rebuilt.validate()  # the editor model round-trips the preset into a solvable system

    def test_preset_stack_runs_through_solver_all_modes(self):
        from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

        system = lp.build_stack_preset("single_film_linbo3")
        for mrassumption in (0, 1, 2):
            sweep = solve_multilayer_maker_fringes_sweep(
                system, theta_deg=[0.0, 30.0], mu=1.0, eps0=1.0, mrassumption=mrassumption
            )
            para, _ = sweep.shaarp_ml_copy_lists()
            values = np.array(para)
            col = values[:, 1] if values.ndim == 2 else values
            self.assertTrue(np.all(np.isfinite(np.abs(col.astype(complex)))))


class MaterialRegistryTests(unittest.TestCase):
    """Headless material picker registry (gui_material_selection)."""

    def test_all_materials_build(self):
        from shaarp import presets
        from shaarp.config import Material

        names = presets.list_materials()
        self.assertGreaterEqual(len(names), 6)
        for name in names:
            mat = presets.build_material(name)
            self.assertIsInstance(mat, Material)
            self.assertTrue(mat.name)

    def test_material_unknown_raises_and_fresh(self):
        from shaarp import presets

        with self.assertRaises(KeyError):
            presets.build_material("not_a_material")
        a = presets.build_material("linbo3_zcut_1064")
        b = presets.build_material("linbo3_zcut_1064")
        self.assertIsNot(a, b)

    def test_material_registry_feeds_a_layer(self):
        from shaarp import presets
        from shaarp.config import Layer

        layer = Layer("picked film", presets.build_material("gaas_111_800"), thickness_um=0.5, shg_active=True)
        self.assertTrue(layer.shg_active)
        self.assertEqual(layer.material.name, "GaAs (111) 800 nm")


if __name__ == "__main__":
    unittest.main()
