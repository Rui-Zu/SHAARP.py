"""N-layer stack model (the original SHAARP.ml Layer Selection) — pure/headless tests, plus a
GUI smoke that the editor drives an arbitrary stack through the validated multilayer Maker path.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from shaarp.layer_stack import (
    LAYER_MATERIAL_CHOICES,
    build_system_from_stack,
    default_stack,
    set_layer_count,
)


class LayerStackModelTests(unittest.TestCase):
    def test_default_stack_is_three_layers(self):
        s = default_stack()
        self.assertEqual(len(s), 3)
        self.assertEqual(s[0]["material"], "air")
        self.assertEqual(s[-1]["material"], "air")
        self.assertNotIn("shg_active", s[1], "F63: activity is derived, never stored")
        from shaarp.layer_stack import build_system_from_stack
        self.assertTrue(build_system_from_stack(s, wavelength_um=1.55).layers[1].shg_active)

    def test_material_choices_include_air_and_casestudy(self):
        # case-study fidelity audit: layer materials = the ORIGINAL SHAARP.ml
        # 16-button palette (display labels), not the full registry -- .si-only cases (TaAs,
        # LBO, ...) and the setup.nb dead-code LiNbO3@1064 do NOT appear.
        self.assertEqual(LAYER_MATERIAL_CHOICES[0], "air")
        self.assertIn("GaAs (111) (800 nm)", LAYER_MATERIAL_CHOICES)
        self.assertIn("Quartz z-cut · 800 nm", LAYER_MATERIAL_CHOICES)  # wavelength-labelled form
        self.assertNotIn("TaAs (112)", LAYER_MATERIAL_CHOICES)  # SHAARP.si-only case
        self.assertNotIn("LiNbO3 z-cut (1064 nm)", LAYER_MATERIAL_CHOICES)  # setup.nb dead code
        self.assertIn("Custom (fields)", LAYER_MATERIAL_CHOICES)  # per-layer custom crystal entry
        self.assertIn("isotropic n (set below)", LAYER_MATERIAL_CHOICES)  # F56 half-space entry
        # air + isotropic-n + 16 palette films + Custom
        self.assertEqual(len(LAYER_MATERIAL_CHOICES), 19)

    def test_per_layer_custom_crystal(self):
        """A layer marked 'Custom (fields)' builds its Material from the per-layer crystal snapshot
        (the original 'Set Material Properties' per-layer entry), not a case-study material."""
        from shaarp.layer_stack import CUSTOM_LAYER_CHOICE, build_system_from_stack

        stack = [
            {"material": "air", "thickness_um": 0.0},
            {"material": CUSTOM_LAYER_CHOICE, "thickness_um": 2.0,
             "custom": {"point_group": "3m", "n_w": 2.2, "n_2w": 2.3, "ne_w": 2.26, "ne_2w": 2.36,
                        "lattice": [5.1, 5.1, 13.8, 90, 90, 120], "orientation_mode": "z-cut (identity)",
                        "surface_hkl": [0, 0, 1], "in_plane_uvw": [1, 0, 0], "d_free": {"2,2": 33.0}}},
            {"material": "air", "thickness_um": 0.0, "shg_active": False},
        ]
        s = build_system_from_stack(stack, wavelength_um=1.064)
        film = s.layers[1].material
        self.assertEqual(film.structure.point_group, "3m")
        self.assertAlmostEqual(float(film.d_voigt_pm_v[2, 2].real), 33.0, places=6)
        self.assertAlmostEqual(float(film.epsilon_omega[2, 2].real), 2.26 ** 2, places=3)

    def test_grow_and_shrink_preserve_halfspaces(self):
        s = set_layer_count(default_stack(), 5)
        self.assertEqual(len(s), 5)
        self.assertEqual(s[0]["material"], "air")
        self.assertEqual(s[-1]["material"], "air")
        s2 = set_layer_count(s, 2)
        self.assertEqual(len(s2), 2)
        with self.assertRaises(ValueError):
            set_layer_count(s, 1)

    def test_build_system_assigns_thickness_to_interior_only(self):
        s = set_layer_count(default_stack(), 4)
        s[2]["material"] = "Quartz z-cut · 800 nm"
        s[2]["thickness_um"] = 5.0
        sysm = build_system_from_stack(s, wavelength_um=1.064, theta_deg=20.0)
        self.assertEqual(len(sysm.layers), 4)
        self.assertAlmostEqual(sysm.layers[2].thickness_um, 5.0)
        self.assertAlmostEqual(sysm.wavelength_um, 1.064)

    def test_arbitrary_stack_runs_validated_maker_path(self):
        from shaarp.shaarp_gui import compute_ml_gui_result

        s = set_layer_count(default_stack(), 4)
        sysm = build_system_from_stack(s, wavelength_um=1.064, theta_deg=20.0)
        r = compute_ml_gui_result("Maker Fringes", theta_min_deg=0, theta_max_deg=30,
                                  theta_step_deg=10, system=sysm)
        self.assertEqual(r.kind, "maker_fringes")
        self.assertIn("parallel_intensity", r.numeric)


@unittest.skipUnless(__import__("importlib").util.find_spec("PySide6"), "PySide6 not installed")
class LayerEditorGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_ml_tab_has_number_of_layers_and_runs(self):
        from PySide6 import QtWidgets

        from shaarp.desktop_app import build_main_window

        win = build_main_window()

        def _all(w):
            yield w
            for ch in (w.children() if hasattr(w, "children") else []):
                yield from _all(ch)

        ml = win.findChild(QtWidgets.QTabWidget).widget(1)
        nl = next((s for s in _all(ml) if isinstance(s, QtWidgets.QSpinBox)
                   and s.toolTip().startswith("Number of Layers")), None)
        self.assertIsNotNone(nl, "Number-of-Layers control missing on ML tab")
        sysd = next(c for c in _all(ml) if isinstance(c, QtWidgets.QComboBox)
                    and any("N-layer" in c.itemText(i) for i in range(c.count())))
        self.assertTrue(any(c.itemText(i) == "N-layer stack (editor)" for i in range(sysd.count())
                            for c in [sysd]))



class LayerNumberingConvention(unittest.TestCase):
    """EVERY medium is a numbered layer, 1..N.

    This SUPERSEDES the interior-only convention decided a day earlier. F59 followed the
    released .ml, which numbers only the interior films (`SHAARP.ml.nb:3707-3712`) -- but it
    does so because it hardcodes air at both ends (`:666-716`, `:425-439`), so its half-spaces
    are not editable entries at all. This port made both half-spaces user-settable media
    so the author's call is that they are first-class rows: "in this, I would consider call
    it 4 layers" for air/quartz/Au/air. They are marked semi-infinite, which is the original's
    own description ("thickness of first and last material is infinite", `:5097-5104`).

    The rule that survives: ONE number per row -- the bug that started it read "2: layer 1",
    with the tensor-panel title compounding it into "layer 2: layer 1"."""

    def test_every_medium_is_numbered_and_half_spaces_say_semi_infinite(self):
        from shaarp.layer_stack import layer_role_label

        total = 5  # ambient + 3 films + substrate
        self.assertEqual(layer_role_label(0, total), "1: ambient (semi-infinite)")
        self.assertEqual(layer_role_label(total - 1, total), "5: substrate (semi-infinite)")
        for k in (1, 2, 3):
            self.assertEqual(layer_role_label(k, total), f"{k + 1}: film",
                             "an unnamed interior row carries its stack number ONCE")

    def test_named_rows_keep_their_number_and_no_row_shows_two(self):
        from shaarp.layer_stack import layer_role_label

        total = 4  # the Fig-4 shape: air / quartz / Au / air
        self.assertEqual(layer_role_label(0, total, "air in"), "1: air in")
        self.assertEqual(layer_role_label(1, total, "Z-cut quartz"), "2: Z-cut quartz")
        self.assertEqual(layer_role_label(2, total, "Au coating"), "3: Au coating")
        self.assertEqual(layer_role_label(3, total, "air out"), "4: air out")
        for i in range(total):
            for nm in (None, "named"):
                label = layer_role_label(i, total, nm)
                self.assertNotRegex(label, r"\d.*(layer|film) \d",
                                    f"row {i} shows two numbers for one layer: {label!r}")

    def test_the_count_is_every_medium(self):
        from shaarp.layer_stack import default_stack, set_layer_count

        stack = default_stack()  # air / film / air
        self.assertEqual(len(stack), 3, "a 3-medium stack reads as 3 layers")
        grown = set_layer_count(stack, 5)
        self.assertEqual(len(grown), 5)
        self.assertEqual(grown[0]["material"], stack[0]["material"], "ambient preserved")
        self.assertEqual(grown[-1]["material"], stack[-1]["material"], "substrate preserved")
        shrunk = set_layer_count(grown, 3)
        self.assertEqual(len(shrunk), 3)

    def test_half_spaces_can_never_be_sources_or_symbolic(self):
        """Model guarantee: the half-spaces "are semi-infinite and always SHG
        inactive"): even a spec that asks for it is refused at build time, so a stale session
        or preset cannot smuggle a source into a half-space."""
        from shaarp.layer_stack import build_system_from_stack, default_layer_spec

        stack = [default_layer_spec("air", 0.0, True, analytic_h=True, analytic_d=True),
                 default_layer_spec("LiNbO3 z-cut · 1550 nm", 2.0, True, analytic_h=True),
                 default_layer_spec("air", 0.0, True, analytic_h=True)]
        sysm = build_system_from_stack(stack, wavelength_um=1.55)
        for idx in (0, -1):
            self.assertFalse(sysm.layers[idx].shg_active)
            self.assertFalse(sysm.layers[idx].analytic_h)
            self.assertFalse(sysm.layers[idx].analytic_d)
        self.assertTrue(sysm.layers[1].shg_active)
        self.assertTrue(sysm.layers[1].analytic_h, "an interior layer keeps its flag")

    def test_shg_activity_is_decided_by_the_point_group(self):
        """No SHG-active switch -- the original's two point-group popups
        decide. Interior palette/Custom layers with an active group are sources; centrosymmetric,
        isotropic, air and half-space rows never are; a legacy spec key is ignored."""
        from shaarp.layer_stack import (CUSTOM_LAYER_CHOICE, build_system_from_stack,
                                        default_layer_spec, isotropic_layer_spec, spec_shg_active)

        def custom(pg, **extra):
            return {"material": CUSTOM_LAYER_CHOICE, "thickness_um": 1.0, "analytic_d": True,
                    "custom": {"point_group": pg, "n_w": 2.2, "n_2w": 2.3, "ne_w": 2.2, "ne_2w": 2.3,
                               "lattice": [5, 5, 5, 90, 90, 90],
                               "orientation_mode": "z-cut (identity)",
                               "surface_hkl": [0, 0, 1], "in_plane_uvw": [1, 0, 0], "d_free": {}},
                    **extra}

        stack = [default_layer_spec("air", 0.0),
                 default_layer_spec("ZnO (001)", 1.0),                     # 6mm -> active
                 default_layer_spec("Pt (111) (1550 nm)", 0.05),           # m3m -> inactive
                 default_layer_spec("Au coating (800 nm)", 0.01),          # inf inf m -> inactive
                 custom("3m"),                                             # active
                 custom("m3m", shg_active=True),                           # legacy key ignored
                 isotropic_layer_spec(1.5, 1.5),                           # iso interior -> inactive
                 default_layer_spec("LiNbO3 z-cut · 1550 nm", 1.0)]        # substrate half-space
        n = len(stack)
        expect = [False, True, False, False, True, False, False, False]
        sysm = build_system_from_stack(stack, wavelength_um=1.55)
        self.assertEqual([L.shg_active for L in sysm.layers], expect)
        self.assertEqual([spec_shg_active(s, i, n, sysm.layers[i].material) for i, s in enumerate(stack)],
                         expect)
        self.assertFalse(sysm.layers[5].analytic_d, "an inactive layer cannot keep analytical d")
        self.assertTrue(sysm.layers[4].analytic_d)

    def test_preset_stacks_reproduce_the_factory_activity_pattern(self):
        """Fig 6 (Air/ZnO/Pt/Al2O3) and Fig 7 (air/LNO/quartz/air) derived from the point groups
        equal the validated presets' explicit flags."""
        from shaarp.layer_stack import build_system_from_stack, stack_from_system
        from shaarp.shaarp_gui import ML_SYSTEM_PRESETS

        for name, factory in ML_SYSTEM_PRESETS.items():
            ref = factory()
            rebuilt = build_system_from_stack(stack_from_system(ref), wavelength_um=ref.wavelength_um)
            self.assertEqual([L.shg_active for L in rebuilt.layers],
                             [bool(L.shg_active) for L in ref.layers], name)


if __name__ == "__main__":
    unittest.main()
