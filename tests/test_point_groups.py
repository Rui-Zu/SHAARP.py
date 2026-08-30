"""Point groups, SHG activity and crystal-system lattice rules -- parity with the
original ♯SHAARP.ml GUI's two popups (SHAARP.ml.nb:5191 Noncentrosymmetric:5630 Centrosymmetric)."""
import unittest

from shaarp.point_groups import (
    ALL_POINT_GROUPS,
    POINT_GROUP_SECTIONS,
    SHG_ACTIVE_GROUPS,
    SHG_INACTIVE_GROUPS,
    apply_lattice_constraints,
    canonical_point_group,
    crystal_system,
    is_known_point_group,
    is_shg_active,
    lattice_constraints,
    symbolic_key,
)

ORIGINAL_ACTIVE = ("1", "2", "m", "mm2", "222", "3", "32", "3m", "4", "6", "-4", "4mm", "6mm",
                   "422", "622", "-42m", "-6", "-6m2", "-43m", "23", "∞", "∞m", "∞2")
ORIGINAL_INACTIVE = ("-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m3", "m3m",
                     "432", "∞/m", "∞/mm", "∞∞", "∞∞m")


class OriginalLists(unittest.TestCase):
    def test_lists_match_the_original_popups_in_order(self):
        self.assertEqual(SHG_ACTIVE_GROUPS, ORIGINAL_ACTIVE)
        self.assertEqual(SHG_INACTIVE_GROUPS, ORIGINAL_INACTIVE)
        self.assertEqual(len(ALL_POINT_GROUPS), 39)
        self.assertEqual(len(set(ALL_POINT_GROUPS)), 39)
        self.assertEqual([s[0] for s in POINT_GROUP_SECTIONS],
                         ["Noncentrosymmetric (SHG-active)", "Centrosymmetric (SHG-inactive)"])

    def test_activity_is_the_list_the_group_came_from(self):
        for pg in ORIGINAL_ACTIVE:
            self.assertTrue(is_shg_active(pg), pg)
        for pg in ORIGINAL_INACTIVE:
            self.assertFalse(is_shg_active(pg), pg)
        self.assertFalse(is_shg_active("432"), "432 is filed under Centrosymmetric by the original")
        self.assertFalse(is_shg_active("nonsense"))
        self.assertFalse(is_known_point_group("nonsense"))

    def test_aliases(self):
        cases = {
            "-62m": "-6m2", "6bar2m": "-6m2",                       # the .si popup spelling
            "Overscript[4,_]3m": "-43m", "43m": "-43m", "Td": "-43m",
            r"\!\(\*OverscriptBox[\(6\), \(_\)]\)m2": "-6m2",
            r"\!\(\*OverscriptBox[\(3\), \(_\)]\)m": "-3m",
            r"\!\(\*OverscriptBox[\(1\), \(_\)]\)": "-1",
            "m-3m": "m3m", "m-3": "m3", "Oh": "m3m",
            "inf": "∞", "infm": "∞m", "inf2": "∞2", r"\[Infinity]": "∞", r"\[Infinity]m": "∞m",
            "inf/mm": "∞/mm", "infinfm": "∞∞m", r"\[Infinity]\[Infinity]m": "∞∞m",
            "4̄": "-4", "6̄m2": "-6m2", "C3v": "3m",
        }
        for raw, canon in cases.items():
            self.assertEqual(canonical_point_group(raw), canon, raw)
        self.assertEqual(symbolic_key("∞m"), "infm")
        self.assertEqual(symbolic_key("-43m"), "-43m")


class CrystalSystems(unittest.TestCase):
    def test_every_group_has_a_system(self):
        expect = {"1": "triclinic", "-1": "triclinic", "2/m": "monoclinic", "mm2": "orthorhombic",
                  "-42m": "tetragonal", "4/mmm": "tetragonal", "32": "trigonal", "-3m": "trigonal",
                  "-6m2": "hexagonal", "6/mmm": "hexagonal", "-43m": "cubic", "432": "cubic",
                  "m3m": "cubic", "∞m": "curie_axial", "∞/mm": "curie_axial", "∞∞m": "curie_isotropic"}
        for pg, sysname in expect.items():
            self.assertEqual(crystal_system(pg), sysname, pg)
        for pg in ALL_POINT_GROUPS:
            crystal_system(pg)  # must not raise
        with self.assertRaises(ValueError):
            crystal_system("nonsense")

    def test_lock_rules(self):
        six = (5.0, 6.0, 7.0, 80.0, 85.0, 95.0)
        self.assertEqual(apply_lattice_constraints("1", six), six)
        self.assertEqual(apply_lattice_constraints("2", six), (5.0, 6.0, 7.0, 90.0, 85.0, 90.0))
        self.assertEqual(apply_lattice_constraints("mm2", six), (5.0, 6.0, 7.0, 90.0, 90.0, 90.0))
        self.assertEqual(apply_lattice_constraints("4mm", six), (5.0, 5.0, 7.0, 90.0, 90.0, 90.0))
        self.assertEqual(apply_lattice_constraints("3m", six), (5.0, 5.0, 7.0, 90.0, 90.0, 120.0))
        self.assertEqual(apply_lattice_constraints("6/mmm", six), (5.0, 5.0, 7.0, 90.0, 90.0, 120.0))
        self.assertEqual(apply_lattice_constraints("-43m", six), (5.0, 5.0, 5.0, 90.0, 90.0, 90.0))
        self.assertEqual(apply_lattice_constraints("∞m", six), (5.0, 5.0, 7.0, 90.0, 90.0, 120.0))
        self.assertEqual(apply_lattice_constraints("∞∞m", six), (5.0, 5.0, 5.0, 90.0, 90.0, 90.0))
        self.assertEqual(lattice_constraints("-43m").locked_indices, (1, 2, 3, 4, 5))
        self.assertEqual(lattice_constraints("6mm").locked_indices, (1, 3, 4, 5))
        self.assertEqual(lattice_constraints("1").locked_indices, ())
        self.assertIn("a = b = c", lattice_constraints("m3m").describe())

    def test_a_propagates_to_b_and_c_under_cubic(self):
        self.assertEqual(apply_lattice_constraints("-43m", (5.5, 1, 1, 90, 90, 90))[:3], (5.5, 5.5, 5.5))

    def test_every_palette_lattice_is_a_fixed_point(self):
        """Locking never rewrites a validated preset material's cell."""
        from shaarp.casestudy_materials import CASE_STUDY_ORDER, build_casestudy_material

        for name in CASE_STUDY_ORDER:
            st = build_casestudy_material(name).structure
            six = (st.a, st.b, st.c, st.alpha_deg, st.beta_deg, st.gamma_deg)
            got = apply_lattice_constraints(st.point_group, six)
            for x, y in zip(got, six):
                self.assertAlmostEqual(x, y, places=9, msg=f"{name} ({st.point_group}): {six} -> {got}")


class TensorPatterns(unittest.TestCase):
    def test_inactive_groups_have_an_all_zero_d_and_active_groups_do_not(self):
        import sympy as sp

        from shaarp.shaarp_gui import point_group_free_components
        from shaarp.symbolic import d_voigt_symbolic

        for pg in SHG_INACTIVE_GROUPS:
            m = d_voigt_symbolic(pg)
            self.assertEqual(m, sp.zeros(3, 6), pg)
            self.assertEqual(point_group_free_components(pg), [], pg)
        for pg in SHG_ACTIVE_GROUPS:
            m = d_voigt_symbolic(pg)
            self.assertTrue(m.free_symbols, pg)
            self.assertTrue(point_group_free_components(pg), pg)
        self.assertEqual(d_voigt_symbolic("-62m"), d_voigt_symbolic("-6m2"))
        with self.assertRaises(ValueError):
            d_voigt_symbolic("nonsense")


if __name__ == "__main__":
    unittest.main()
