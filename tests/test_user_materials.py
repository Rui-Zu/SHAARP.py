"""The user's "My Materials" JSON store (pure) and the material-resolution seam."""
import json
import os
import tempfile
import unittest

import numpy as np

from shaarp import user_materials as um

_REAL_ENV = os.environ.get(um.ENV_OVERRIDE)


class _TempStore(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "user_materials.json")
        os.environ[um.ENV_OVERRIDE] = self.path
        # guard: the suite must never touch the real per-user store
        self.assertTrue(um.store_path().startswith(self._dir.name))

    def tearDown(self):
        if _REAL_ENV is None:
            os.environ.pop(um.ENV_OVERRIDE, None)
        else:
            os.environ[um.ENV_OVERRIDE] = _REAL_ENV
        self._dir.cleanup()


def _ktp_spec():
    from shaarp.casestudy_materials import build_casestudy_material
    from shaarp.layer_stack import custom_spec_snapshot_from_material
    return custom_spec_snapshot_from_material(build_casestudy_material("KTP (100)"))


class StoreContract(_TempStore):
    def test_empty_store(self):
        self.assertEqual(um.load(), {})
        self.assertEqual(um.list_names(), [])
        self.assertIsNone(um.get("nothing"))
        self.assertFalse(um.is_user_material("nothing"))

    def test_save_get_round_trip_is_exact_with_complex_grids(self):
        spec = _ktp_spec()
        spec["eps_omega_full"][0][0] = complex(3.1, 0.25)
        spec["thickness"] = 12.0        # layer property: must be dropped
        spec["wavelength"] = 1.55       # global property: must be dropped
        name = um.save("  my KTP  ", spec, 1.55)
        self.assertEqual(name, "my KTP")
        e = um.get("my KTP")
        self.assertEqual(e["wavelength_um"], 1.55)
        self.assertNotIn("thickness", e["spec"])
        self.assertNotIn("wavelength", e["spec"])
        self.assertEqual(e["spec"]["eps_omega_full"][0][0], complex(3.1, 0.25))
        self.assertEqual(e["spec"]["point_group"], spec["point_group"])
        self.assertTrue(um.is_user_material("my KTP"))
        self.assertEqual(um.list_names(), ["my KTP"])
        self.assertFalse(os.path.exists(self.path + ".tmp"), "atomic write leaves no temp file")
        with open(self.path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.assertEqual(raw["kind"], um.KIND)

    def test_overwrite_is_update_rename_and_delete(self):
        um.save("A", _ktp_spec(), 1.0)
        spec2 = _ktp_spec()
        spec2["point_group"] = "3m"
        um.save("A", spec2, 0.8)
        self.assertEqual(um.get("A")["spec"]["point_group"], "3m")
        self.assertEqual(um.get("A")["wavelength_um"], 0.8)
        self.assertEqual(um.rename("A", "B"), "B")
        self.assertEqual(um.list_names(), ["B"])
        um.save("C", _ktp_spec(), 1.0)
        with self.assertRaises(ValueError):
            um.rename("B", "C")  # would collide
        with self.assertRaises(KeyError):
            um.rename("missing", "D")
        self.assertTrue(um.delete("B"))
        self.assertFalse(um.delete("B"))
        self.assertEqual(um.list_names(), ["C"])

    def test_name_validation_protects_the_built_ins(self):
        from shaarp.casestudy_materials import CASE_STUDY_ORDER, GUI_ML_GROUPS, GUI_SI_GROUPS
        from shaarp.layer_stack import CUSTOM_LAYER_CHOICE, ISOTROPIC_LAYER_CHOICE

        for bad in ("", "   ", "—  header  —", "air", "Air", "Custom (use fields)",
                    "N-layer stack (editor)", CUSTOM_LAYER_CHOICE, ISOTROPIC_LAYER_CHOICE,
                    "gaas (111) (800 nm)", "Quartz + Au (Fig 4, 800 nm)"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                um.validate_name(bad)
        for key in CASE_STUDY_ORDER:
            with self.assertRaises(ValueError, msg=key):
                um.validate_name(key)
        for _h, entries in GUI_ML_GROUPS + GUI_SI_GROUPS:
            for label, _k in entries:
                with self.assertRaises(ValueError, msg=label):
                    um.validate_name(label)
        self.assertEqual(um.validate_name(" my LiNbO3 sample #2 "), "my LiNbO3 sample #2")

    def test_corrupt_or_foreign_file_reads_as_empty(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertEqual(um.load(), {})
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"kind": "shaarp_gui_session", "materials": {"x": {}}}, fh)
        self.assertEqual(um.load(), {})

    def test_built_material_reproduces_the_source_tensors(self):
        from shaarp.casestudy_materials import build_casestudy_material

        ref = build_casestudy_material("KTP (100)")
        um.save("KTP copy", _ktp_spec(), float(ref.structure.a) if False else 1.064)
        mat = um.build_user_material("KTP copy", 1.064)
        self.assertEqual(mat.name, "KTP copy")
        self.assertEqual(str(mat.structure.point_group), str(ref.structure.point_group))
        for got, exp in ((mat.eps_w(), ref.eps_w()), (mat.eps_2w(), ref.eps_2w()),
                         (mat.d_voigt(), ref.d_voigt())):
            self.assertLess(np.max(np.abs(np.asarray(got) - np.asarray(exp))), 1e-12)


class ResolutionSeam(_TempStore):
    def test_every_built_in_label_resolves_identically(self):
        """The seam must not change a single built-in lookup."""
        from shaarp.casestudy_materials import (CASE_STUDY_ORDER, GUI_ML_CASES, GUI_SI_GROUPS,
                                                build_casestudy_material, resolve_case_label)
        from shaarp.layer_stack import material_for_label

        labels = list(CASE_STUDY_ORDER) + [lbl for lbl, _k in GUI_ML_CASES]
        for _h, entries in GUI_SI_GROUPS:
            labels += [lbl for lbl, _k in entries]
        for label in labels:
            ref = build_casestudy_material(resolve_case_label(label), wavelength_um=1.064)
            got = material_for_label(label, 1.064)
            self.assertEqual(got.name, ref.name, label)
            self.assertLess(np.max(np.abs(np.asarray(got.d_voigt()) - np.asarray(ref.d_voigt()))), 0.0 + 1e-15)
            self.assertLess(np.max(np.abs(np.asarray(got.eps_w()) - np.asarray(ref.eps_w()))), 1e-15)
        self.assertEqual(material_for_label("air", 1.0).name, material_for_label("air", 1.0).name)
        with self.assertRaises(ValueError):
            material_for_label("no such material", 1.0)

    def test_user_material_resolves_and_choices_follow_the_store(self):
        from shaarp.layer_stack import (CUSTOM_LAYER_CHOICE, LAYER_MATERIAL_CHOICES,
                                        layer_material_choices, material_for_label)

        self.assertEqual(layer_material_choices(), LAYER_MATERIAL_CHOICES)
        um.save("mine", _ktp_spec(), 1.064)
        choices = layer_material_choices()
        self.assertEqual(choices[-1], CUSTOM_LAYER_CHOICE)
        self.assertEqual(choices[-3:-1], [um.USER_SECTION_HEADER, "mine"])
        self.assertEqual(len(LAYER_MATERIAL_CHOICES), 19, "the pinned palette constant is untouched")
        self.assertEqual(material_for_label("mine", 1.064).name, "mine")


if __name__ == "__main__":
    unittest.main()
