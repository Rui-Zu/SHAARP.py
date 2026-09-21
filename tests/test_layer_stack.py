"""N-layer stack model (the original SHAARP.ml Layer Selection) — pure/headless tests, plus a
GUI smoke that the editor drives an arbitrary stack through the validated multilayer Maker path.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from shaarp.layer_stack import (
    ISOTROPIC_LAYER_CHOICE,
    LAYER_MATERIAL_CHOICES,
    build_system_from_stack,
    default_layer_spec,
    default_stack,
    layer_material_choices,
    set_layer_count,
    simple_film_stack,
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
        # ...plus the DISPERSIVE variants, between the palette and Custom. These are not further
        # original examples -- they are palette crystals whose linear optics come from a published
        # index table -- but the editor is the single stack truth, so a material the case combo can
        # select has to be a material a layer row can hold. While they were missing, choosing a
        # dispersive film silently degraded the row to Custom and the run then failed on a
        # half-filled snapshot. Counted from the registry so shipping a new table cannot break this.
        from shaarp.dispersion import dispersive_material_names

        dispersive = dispersive_material_names()
        self.assertEqual(LAYER_MATERIAL_CHOICES[-1 - len(dispersive):-1], dispersive,
                         "the dispersive variants must sit between the palette and Custom")
        # air + isotropic-n + 16 palette films + the dispersive variants + Custom
        self.assertEqual(len(LAYER_MATERIAL_CHOICES), 19 + len(dispersive))

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

    def test_half_space_rows_are_offered_only_isotropic_materials(self):
        """The other half of the isotropy rule: the option is never OFFERED, not just refused.

        `layer_stack._require_isotropic_halfspace` guarantees it in the model; this checks the
        editor does not put a birefringent crystal in front of the user on the ambient/substrate
        rows in the first place. Driven through the REAL window, because the palette depends on
        which row is selected and the half-spaces MOVE when the layer count changes -- a per-row
        rule cannot be verified from the combo's initial contents.
        """
        from PySide6 import QtWidgets

        from shaarp.desktop_app import build_main_window
        from shaarp.layer_stack import ISOTROPIC_LAYER_CHOICE

        win = build_main_window()

        def _all(w):
            yield w
            for ch in (w.children() if hasattr(w, "children") else []):
                yield from _all(ch)

        ml = win.findChild(QtWidgets.QTabWidget).widget(1)
        combos = [c for c in _all(ml) if isinstance(c, QtWidgets.QComboBox)]
        row_sel = next(c for c in combos if (c.toolTip() or "").startswith("Select the layer"))
        material = next(c for c in combos if (c.toolTip() or "").startswith("Assign a material"))

        rows = row_sel.count()
        self.assertGreaterEqual(rows, 3, "need at least ambient + film + substrate")
        interior_seen = False
        for i in range(rows):
            row_sel.setCurrentIndex(i)
            offered = [material.itemText(k) for k in range(material.count())]
            if i in (0, rows - 1):
                self.assertEqual(offered, ["air", ISOTROPIC_LAYER_CHOICE],
                                 f"row {i + 1} is a half-space and must offer isotropic media only")
            else:
                interior_seen = True
                self.assertGreater(len(offered), 2,
                                   f"row {i + 1} is interior and must keep the full palette")
                self.assertIn("Custom (fields)", offered)
        self.assertTrue(interior_seen, "no interior row exercised -- the contrast is the test")

        # returning to a half-space must restrict again (the combo is ONE widget, repopulated)
        row_sel.setCurrentIndex(0)
        self.assertEqual([material.itemText(k) for k in range(material.count())],
                         ["air", ISOTROPIC_LAYER_CHOICE])



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
                 # substrate half-space: isotropic by rule -- a crystal here is now REFUSED by
                 # build_system_from_stack (HalfSpacesAreIsotropic), so the old "a crystal in a
                 # half-space is still inactive" case is superseded by "it cannot be one at all".
                 default_layer_spec("air", 0.0)]
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


class PresetRowsRoundTrip(unittest.TestCase):
    """Every shipped preset must load into the editor as PALETTE rows carrying its true thickness.

    Found by the fidelity auditor (2026-09-18): the Fig 6 and Fig 7 presets build from the
    registry, whose materials are named 'ZnO', 'Pt', 'Al₂O₃', 'LiNbO₃' -- none a palette label --
    so `stack_from_system` sent those rows to Custom (fields) with a snapshot that carried no
    thickness, and the editor then displayed h = 1.0 um for the 159 nm ZnO film. Pressing Update
    "settled" that 1.0 into the stack and computed the wrong sample: shg coefficients differed from
    the factory preset by a relative 1.00. Fig 4 was immune only because its materials happen to
    be named in PRESET_MATERIAL_LABELS. Rows are now matched by tensor equality.
    """

    def test_every_preset_row_is_a_palette_entry_with_its_true_thickness(self):
        from shaarp.layer_stack import CUSTOM_LAYER_CHOICE, stack_from_system
        from shaarp.shaarp_gui import ML_SYSTEM_PRESETS

        for name, factory in ML_SYSTEM_PRESETS.items():
            reference = factory()
            stack = stack_from_system(reference)
            for row, (spec, layer) in enumerate(zip(stack, reference.layers), start=1):
                with self.subTest(preset=name, row=row):
                    self.assertNotEqual(spec["material"], CUSTOM_LAYER_CHOICE,
                                        f"row {row} ({layer.material.name!r}) fell to Custom")
                    self.assertAlmostEqual(spec["thickness_um"], float(layer.thickness_um or 0.0))

    def test_rebuilding_a_preset_from_its_rows_reproduces_the_physics(self):
        """Not just the labels: the rebuilt stack must carry the same tensors and thicknesses."""
        import numpy as np

        from shaarp.layer_stack import stack_from_system
        from shaarp.shaarp_gui import ML_SYSTEM_PRESETS

        for name, factory in ML_SYSTEM_PRESETS.items():
            reference = factory()
            rebuilt = build_system_from_stack(stack_from_system(reference),
                                              wavelength_um=reference.wavelength_um)
            with self.subTest(preset=name):
                self.assertEqual(len(rebuilt.layers), len(reference.layers))
                for a, b in zip(rebuilt.layers, reference.layers):
                    self.assertEqual(a.thickness_um, b.thickness_um)
                    self.assertEqual(a.shg_active, b.shg_active)
                    np.testing.assert_allclose(np.asarray(a.material.eps_w(), complex),
                                               np.asarray(b.material.eps_w(), complex))
                    np.testing.assert_allclose(np.asarray(a.material.eps_2w(), complex),
                                               np.asarray(b.material.eps_2w(), complex))
                    if b.shg_active:
                        # d only where it is READ. The registry's "Au Coating" (∞∞m) carries a
                        # stray d11 = 0.3 pm/V that the Fig 4 docs material does not; gold is
                        # SHG-inactive by point group, so the tensor never enters a computation
                        # (register row on the data quirk). Comparing it here would fail the
                        # round-trip on a number nothing uses.
                        np.testing.assert_allclose(np.asarray(a.material.d_voigt_pm_v, complex),
                                                   np.asarray(b.material.d_voigt_pm_v, complex))


class SchematicIndicesAreFiniteForMetals(unittest.TestCase):
    """The stack schematic drew Pt with n = 1 and warned on every Fig 6 Update.

    `schematic_indices_for` took sqrt(Re(eps)); a metal has Re(eps) < 0, so that is NaN, the
    '> 0.05' guard then fell back to 1.0, and numpy printed a RuntimeWarning each time the ML tab
    mirrored or ran the Fig 6 preset (found in a GUI review, 2026-09-18). The schematic must use
    Re(sqrt(eps)) -- 1.77 for Pt at 1550 nm -- and stay silent.
    """

    def test_fig6_pt_layer_has_a_finite_index_and_no_warning(self):
        import warnings

        from shaarp.shaarp_gui import ML_SYSTEM_PRESETS, schematic_indices_for

        system = ML_SYSTEM_PRESETS["ZnO / Pt / Al2O3 (Fig 6, 1550 nm)"]()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RuntimeWarning fails the test
            pairs = schematic_indices_for(system)
        self.assertIsNotNone(pairs)
        self.assertEqual(len(pairs), len(system.layers))
        pt_w, pt_2w = pairs[2]
        self.assertGreater(pt_w, 1.05, "Pt drawn with the n=1 fallback (NaN index)")
        self.assertAlmostEqual(pt_w, 1.7719, places=3)
        self.assertAlmostEqual(pt_2w, 0.5557, places=3)
        for n_w, n_2w in pairs:
            self.assertTrue(n_w > 0 and n_2w > 0)


class HalfSpacesAreIsotropic(unittest.TestCase):
    """The ambient and substrate media must be isotropic -- nx = ny = nz, kx = ky = kz.

    The two semi-infinite media set the tangential wavevector for every mode in the stack and the
    solvers read ONE scalar index from each; a birefringent half-space would split the incoming and
    outgoing beams before the stack is even reached. So the combo does not offer the choice, and the
    model refuses it anyway -- the same belt-and-braces this file already applies to shg_active on a
    half-space ("guaranteed in the MODEL, not just by hiding the checkboxes"), because a saved
    session or a stack built in code never goes near the combo.

    The incident side was already enforced deeper in (`multilayer_shg_boundary._isotropic_index`);
    the exit side was not, which is the hole this closes.
    """

    BIREFRINGENT = "LiNbO3 z-cut · 1550 nm"

    def test_the_combo_offers_only_isotropic_media_for_half_spaces(self):
        choices = layer_material_choices(halfspace=True)
        self.assertEqual(choices, ["air", ISOTROPIC_LAYER_CHOICE])
        self.assertNotIn(self.BIREFRINGENT, choices)
        # ...while interior rows keep the full palette, or this is a regression, not a rule.
        self.assertIn(self.BIREFRINGENT, layer_material_choices())

    def test_a_birefringent_ambient_or_substrate_is_refused_by_the_builder(self):
        """Enforced in the MODEL, so a saved session or a code-built stack cannot smuggle one in."""
        for role, index in (("ambient", 0), ("substrate", -1)):
            with self.subTest(role=role):
                stack = default_stack()
                stack[index] = default_layer_spec(self.BIREFRINGENT, 0.0, False)
                with self.assertRaises(ValueError) as ctx:
                    build_system_from_stack(stack)
                message = str(ctx.exception)
                self.assertIn("must be isotropic", message)
                self.assertIn(role, message)
                # the message has to say what to do, not merely that it refused
                self.assertIn(ISOTROPIC_LAYER_CHOICE, message)

    def test_every_shipped_preset_obeys_the_rule(self):
        """The rule has to hold for the paper stacks, or it is not a rule.

        Fig 6 used to be the sole violator: its sapphire was modelled as a semi-infinite exit
        medium. Rui confirmed (2026-09-12) the sample was sapphire sitting on AIR, so it is now a
        finite 100 um wafer (the released .ml button's value) above an air half-space -- which is also the only shape the released
        Mathematica .ml GUI could express. Measured impact of the correction: 5.3e-13 relative,
        correlation 1.000000000000, because the 200 nm Pt in front of it is opaque.
        """
        from shaarp.layer_stack import stack_from_system
        from shaarp.shaarp_gui import ML_SYSTEM_PRESETS

        for name, factory in ML_SYSTEM_PRESETS.items():
            with self.subTest(preset=name):
                reference = factory()
                build_system_from_stack(stack_from_system(reference),
                                        wavelength_um=reference.wavelength_um)

    def test_fig6_models_sapphire_as_a_finite_wafer_on_air(self):
        """Pin the corrected architecture in BOTH copies -- they are edited by hand, separately."""
        from benchmarks.paper_cases import ml_fig6_system
        from shaarp.shaarp_gui import ML_SYSTEM_PRESETS

        for label, system in (("gui preset", ML_SYSTEM_PRESETS["ZnO / Pt / Al2O3 (Fig 6, 1550 nm)"]()),
                              ("paper_cases", ml_fig6_system())):
            with self.subTest(copy=label):
                self.assertEqual(len(system.layers), 5, "air / ZnO / Pt / Al2O3 / air")
                self.assertAlmostEqual(system.layers[3].thickness_um, 100.0,
                                       msg="the released original's Al2O3(0001) button sets 100 um "
                                           "(setup.nb:3537)")
                self.assertIsNone(system.layers[-1].thickness_um, "exit medium is semi-infinite air")
                self.assertEqual(system.layers[-1].material.name, "Air")

    def test_the_same_crystal_is_fine_in_an_interior_layer(self):
        """The rule is about the half-spaces only -- it must not disarm the stack's actual film."""
        stack = default_stack()
        stack[1] = default_layer_spec(self.BIREFRINGENT, 10.0, True)
        self.assertEqual(len(build_system_from_stack(stack).layers), 3)

    def test_the_shipped_defaults_still_build(self):
        """A rule that breaks the app's own starting stack is not a rule, it is a bug."""
        self.assertEqual(len(build_system_from_stack(default_stack()).layers), 3)
        self.assertEqual(
            len(build_system_from_stack(simple_film_stack(self.BIREFRINGENT, 10.0)).layers), 3)


class APresetNameDoesNotOutliveItsMaterial(unittest.TestCase):
    """A preset ships its rows with names describing the material they came with -- "Z-cut Quartz"
    on the Quartz + Au film -- and `spec["name"]` wins over the auto label in
    build_system_from_stack. Swapping that row's material therefore left the layer selector, the
    schematic caption and the sweep's notes all naming a material that is no longer in the stack
    (found in a GUI review, 2026-09-19: an undersampling note read "Z-cut Quartz (121.2 um thick)"
    about a LiNbO3 row).

    A name the USER typed is theirs and must survive the same swap, which is what makes this a
    rule about provenance rather than a blanket clear.
    """

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _editor(self):
        from PySide6 import QtWidgets

        from shaarp.desktop_app import TOOLTIPS, build_main_window
        from tests.gui_harness import ml_case_combo

        win = build_main_window()
        page = win.findChild(QtWidgets.QTabWidget).widget(1)
        preset = ml_case_combo(page)
        preset.setCurrentIndex(preset.findText("Quartz + Au (Fig 4, 800 nm)"))
        self.app.processEvents()
        select = next(c for c in page.findChildren(QtWidgets.QComboBox)
                      if c.toolTip() == TOOLTIPS["layer_select"])
        material = next(c for c in page.findChildren(QtWidgets.QComboBox)
                        if c.toolTip() == TOOLTIPS["layer_material"])
        name = next(e for e in page.findChildren(QtWidgets.QLineEdit)
                    if "layer name" in (e.toolTip() or "").lower())
        select.setCurrentIndex(1)                     # the quartz film
        self.app.processEvents()
        return page, select, material, name

    def _row_name(self, page, row=1):
        from shaarp.layer_stack import decode_stack

        return decode_stack(page._ml_stack_payload()["stack"])[row].get("name", "")

    def test_a_preset_name_goes_when_the_user_changes_that_rows_material(self):
        from shaarp.layer_stack import decode_stack

        page, _select, material, name = self._editor()
        self.assertIn("uartz", self._row_name(page), "the preset row should start out named")

        other = "LiNbO3 z-cut · 1550 nm"
        material.setCurrentText(other)
        material.textActivated.emit(other)             # what a user's own pick fires
        self.app.processEvents()

        self.assertEqual(name.text().strip(), "")
        self.assertNotIn("uartz", self._row_name(page),
                         "the row still carries the name of the material it no longer holds")
        # ...and the built system labels the row by what is actually in it
        label = build_system_from_stack(
            decode_stack(page._ml_stack_payload()["stack"]), wavelength_um=1.55).layers[1].name
        self.assertNotIn("uartz", label)
        self.assertIn("LiNbO3", label)

    def test_a_name_the_user_typed_survives_the_swap(self):
        page, _select, material, name = self._editor()
        name.setText("my film")
        name.textEdited.emit("my film")                # typing, not a programmatic write
        name.editingFinished.emit()
        self.app.processEvents()
        self.assertEqual(self._row_name(page), "my film")

        other = "LiNbO3 z-cut · 1550 nm"
        material.setCurrentText(other)
        material.textActivated.emit(other)
        self.app.processEvents()
        self.assertEqual(name.text().strip(), "my film")
        self.assertEqual(self._row_name(page), "my film")


if __name__ == "__main__":
    unittest.main()
