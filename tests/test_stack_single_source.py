"""F55 pure fences: the layer editor as the SINGLE stack source.

the author's structural finding: the GUI carried TWO thickness inputs (the editor's per-layer row and a
standalone "film thickness" row for the simple modes), and the layer editor showed stale defaults
while a named preset computed a completely different stack. makes `stack_state` the single
truth: presets LOAD their real stack into the editor via :func:`layer_stack.stack_from_system`,
the simple modes are 3-layer templates whose film row IS the thickness input, and the standalone
spin is deleted.

These fences are PURE (no Qt): they pin the preset->spec mapping so it can never drift from the
factory systems the validated references certify.

  * MAPPING EQUIVALENCE — for every named ML preset, rebuilding a system from
    `stack_from_system(...)` reproduces the factory system layer-by-layer: crystal-frame eps/d,
    orientation rows, thicknesses, SHG flags. The Fig-4 preset's materials are NOT registry-named
    ("Z-cut quartz (docs)" / "Au coating (docs)" — transcribed setup.nb constants), so its rows
    map through PRESET_MATERIAL_LABELS to palette materials that are numerically identical at the
    preset wavelength (Au exactly; quartz same Sellmeier) — the tolerance below is the guard that
    keeps that alias table honest against any future registry edit.
  * CUSTOM-SNAPSHOT ROUND-TRIP — custom_spec_snapshot_from_material must reproduce an arbitrary
    anisotropic material exactly (the frame trap: build_custom_si_material treats eps/d as
    CRYSTAL-frame and rotates by the orientation itself; snapshotting lab-frame accessors would
    double-rotate — this fence fails if anyone "simplifies" the snapshot that way).
  * ISOTROPIC HALF-SPACES — the simple templates' ambient/substrate rows are REAL
    isotropic-n layers (the deleted Substrate group relocated into the editor); the editor path
    and the builder path must construct the identical eps = n^2 * I media.
"""

from __future__ import annotations

import unittest

import numpy as np

from shaarp.layer_stack import (
    CUSTOM_LAYER_CHOICE,
    ISOTROPIC_LAYER_CHOICE,
    build_system_from_stack,
    custom_spec_snapshot_from_material,
    default_layer_spec,
    isotropic_layer_spec,
    set_halfspace_n,
    simple_film_stack,
    stack_film_thickness_um,
    stack_from_system,
    stack_halfspace_n,
)
from shaarp.shaarp_gui import ML_SYSTEM_PRESETS, resolve_ml_system_preset

# Fig 6/7 build every layer from registry keys -> exact; Fig 4 maps docs-material aliases to the
# palette (identical constants at 0.8 um: Au exact, quartz same Sellmeier evaluated at a grid node).
MAPPING_ATOL = 1e-9


class PresetMappingEquivalence(unittest.TestCase):
    def test_every_named_preset_round_trips_through_the_editor_specs(self):
        self.assertTrue(ML_SYSTEM_PRESETS, "no named ML presets found")
        for preset_name in ML_SYSTEM_PRESETS:
            with self.subTest(preset=preset_name):
                factory = resolve_ml_system_preset(preset_name)
                specs = stack_from_system(factory)
                rebuilt = build_system_from_stack(
                    specs, wavelength_um=float(factory.wavelength_um))

                self.assertEqual(len(rebuilt.layers), len(factory.layers))
                for i, (fl, rl) in enumerate(zip(factory.layers, rebuilt.layers)):
                    tag = f"{preset_name} layer {i} ({getattr(fl.material, 'name', '?')})"
                    self.assertEqual(bool(rl.shg_active), bool(fl.shg_active), tag)
                    f_th = fl.thickness_um
                    r_th = rl.thickness_um
                    if f_th is None:
                        self.assertIn(r_th, (None, 0.0), tag)
                    else:
                        self.assertAlmostEqual(float(r_th), float(f_th), places=12, msg=tag)
                    np.testing.assert_allclose(
                        np.asarray(rl.material.epsilon_omega, dtype=complex),
                        np.asarray(fl.material.epsilon_omega, dtype=complex),
                        atol=MAPPING_ATOL, err_msg=tag + " eps(w)")
                    np.testing.assert_allclose(
                        np.asarray(rl.material.epsilon_2omega, dtype=complex),
                        np.asarray(fl.material.epsilon_2omega, dtype=complex),
                        atol=MAPPING_ATOL, err_msg=tag + " eps(2w)")
                    if fl.shg_active:
                        # d participates in the compute only for SHG-ACTIVE layers. Passive layers
                        # may legitimately differ (the factory zeroes passive d, while e.g. the
                        # registry "Au coating (800 nm)" carries a faithful-to-original stray
                        # d11 = 0.3 despite point group ∞∞m — never used, never "fixed").
                        np.testing.assert_allclose(
                            np.asarray(rl.material.d_voigt_pm_v, dtype=complex),
                            np.asarray(fl.material.d_voigt_pm_v, dtype=complex),
                            atol=MAPPING_ATOL, err_msg=tag + " d")
                    np.testing.assert_allclose(
                        np.asarray(rl.material.orientation.z_axes_in_lab, dtype=float),
                        np.asarray(fl.material.orientation.z_axes_in_lab, dtype=float),
                        atol=1e-12, err_msg=tag + " orientation")

    def test_preset_specs_carry_display_labels_where_the_palette_has_them(self):
        """The editor should SHOW palette labels (not 'Custom (fields)') for preset layers the
        palette can express — that is what makes the mirror readable."""
        fig4 = next((n for n in ML_SYSTEM_PRESETS if "Fig 4" in n), None)
        self.assertIsNotNone(fig4, "Fig-4 preset missing from ML_SYSTEM_PRESETS")
        specs = stack_from_system(resolve_ml_system_preset(fig4))
        materials = [s["material"] for s in specs]
        self.assertEqual(materials[0], "air", materials)
        self.assertNotIn(CUSTOM_LAYER_CHOICE, materials,
                         f"Fig-4 rows must map to palette labels, got {materials}")
        self.assertEqual([s["thickness_um"] for s in specs[1:-1]], [121.2, 0.0139])


class CustomSnapshotRoundTrip(unittest.TestCase):
    def test_snapshot_reproduces_an_anisotropic_material_exactly(self):
        from shaarp.casestudy_materials import build_casestudy_material
        from shaarp.layer_stack import _material_from_custom_spec

        # a genuinely anisotropic, non-identity-oriented registry material
        mat = build_casestudy_material("TaAs (112)", wavelength_um=0.8)
        spec = custom_spec_snapshot_from_material(mat)
        rebuilt = _material_from_custom_spec(spec, wavelength_um=0.8)
        np.testing.assert_allclose(np.asarray(rebuilt.epsilon_omega, dtype=complex),
                                   np.asarray(mat.epsilon_omega, dtype=complex), atol=1e-12)
        np.testing.assert_allclose(np.asarray(rebuilt.epsilon_2omega, dtype=complex),
                                   np.asarray(mat.epsilon_2omega, dtype=complex), atol=1e-12)
        np.testing.assert_allclose(np.asarray(rebuilt.d_voigt_pm_v, dtype=complex),
                                   np.asarray(mat.d_voigt_pm_v, dtype=complex), atol=1e-12)
        np.testing.assert_allclose(np.asarray(rebuilt.orientation.z_axes_in_lab, dtype=float),
                                   np.asarray(mat.orientation.z_axes_in_lab, dtype=float),
                                   atol=1e-12)


class IsotropicHalfspaces(unittest.TestCase):
    """The requirement: "remove the substrate group and blend both top and substrate to the layer
    editor ... as isotropic layers"): the simple templates carry REAL isotropic half-space rows,
    directly buildable, numerically identical to the old Substrate-group construction."""

    def test_simple_template_shape_and_defaults(self):
        stack = simple_film_stack(CUSTOM_LAYER_CHOICE, 2.0)
        self.assertEqual([s["material"] for s in stack],
                         [ISOTROPIC_LAYER_CHOICE, CUSTOM_LAYER_CHOICE, ISOTROPIC_LAYER_CHOICE])
        self.assertEqual(stack_halfspace_n(stack, "top"), (1.0, 1.0))       # air
        self.assertEqual(stack_halfspace_n(stack, "bottom"), (1.45, 1.46))  # old group defaults

    def test_iso_rows_build_the_exact_substrate_construction(self):
        """build_system_from_stack's isotropic layer == the builders' eps = n^2 * I substrate —
        the two compute paths (simple-mode builder vs editor) can never drift apart."""
        stack = [isotropic_layer_spec(1.2, 1.25, name="ambient"),
                 default_layer_spec("Quartz z-cut (800 nm)", 5.0, True),
                 isotropic_layer_spec(1.7, 1.72, name="substrate")]
        sys_ = build_system_from_stack(stack, wavelength_um=0.8)
        np.testing.assert_array_equal(
            np.asarray(sys_.layers[0].material.eps_w()), np.eye(3) * 1.2 ** 2)
        np.testing.assert_array_equal(
            np.asarray(sys_.layers[0].material.eps_2w()), np.eye(3) * 1.25 ** 2)
        np.testing.assert_array_equal(
            np.asarray(sys_.layers[-1].material.eps_w()), np.eye(3) * 1.7 ** 2)
        np.testing.assert_array_equal(
            np.asarray(sys_.layers[-1].material.eps_2w()), np.eye(3) * 1.72 ** 2)

    def test_builders_take_the_stack_halfspace_values(self):
        """Equal-results seam: build_casestudy_ml_system fed the template's half-space pairs
        produces layer eps identical to the editor building the same template directly."""
        from shaarp.casestudy_materials import build_casestudy_ml_system

        stack = simple_film_stack("Quartz z-cut · 800 nm", 5.0,
                                  top_n=(1.1, 1.12), bottom_n=(1.6, 1.62))
        top = stack_halfspace_n(stack, "top")
        bot = stack_halfspace_n(stack, "bottom")
        via_builder = build_casestudy_ml_system(
            "Quartz z-cut (800 nm)", thickness_um=5.0, wavelength_um=0.8,
            ambient_n_omega=top[0], ambient_n_2omega=top[1],
            substrate_n_omega=bot[0], substrate_n_2omega=bot[1])
        via_editor = build_system_from_stack(stack, wavelength_um=0.8)
        for i in (0, -1):
            np.testing.assert_allclose(
                np.asarray(via_builder.layers[i].material.eps_w()),
                np.asarray(via_editor.layers[i].material.eps_w()), atol=0)
            np.testing.assert_allclose(
                np.asarray(via_builder.layers[i].material.eps_2w()),
                np.asarray(via_editor.layers[i].material.eps_2w()), atol=0)

    def test_builder_air_default_is_exactly_presets_air(self):
        """ambient defaults (1.0, 1.0) keep the presets.air() object — the equal-results contract
        for every pre-F56 result."""
        from shaarp import presets
        from shaarp.casestudy_materials import build_casestudy_ml_system
        from shaarp.shaarp_gui import build_custom_ml_system

        for sys_ in (build_custom_ml_system("-43m"),
                     build_casestudy_ml_system("Quartz z-cut (800 nm)")):
            np.testing.assert_array_equal(np.asarray(sys_.layers[0].material.eps_w()),
                                          np.asarray(presets.air().eps_w()))
            self.assertEqual(sys_.layers[0].name, "air in")

    def test_set_halfspace_n_converts_in_place(self):
        stack = [default_layer_spec("air", 0.0, False),
                 default_layer_spec("Quartz z-cut (800 nm)", 7.0, True),
                 default_layer_spec("air", 0.0, False)]
        self.assertEqual(stack_halfspace_n(stack, "bottom"), (1.0, 1.0))  # 'air' reads as (1,1)
        set_halfspace_n(stack, "bottom", 1.5, 1.52)
        self.assertEqual(stack[-1]["material"], ISOTROPIC_LAYER_CHOICE)
        self.assertEqual(stack_halfspace_n(stack, "bottom"), (1.5, 1.52))
        # a case-material half-space (e.g. Fig-6's Al2O3) is NOT a plain isotropic pair
        stack[-1] = default_layer_spec("Al2O3 (0001) (1550 nm)", 0.0, False)
        self.assertIsNone(stack_halfspace_n(stack, "bottom"))

    def test_film_thickness_helper_reads_the_first_interior_layer(self):
        self.assertEqual(stack_film_thickness_um(simple_film_stack("air", 3.25)), 3.25)
        four = [default_layer_spec("air", 0.0, False),
                default_layer_spec("Quartz z-cut (800 nm)", 7.0, True),
                default_layer_spec("Au coating (800 nm)", 0.0139, False),
                default_layer_spec("air", 0.0, False)]
        self.assertEqual(stack_film_thickness_um(four), 7.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
