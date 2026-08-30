"""Case Study materials registry — the original SHAARP.ml material set, exported live from
Mathematica (benchmarks/generate_casestudy_registry.py). These tests pin that every material
builds, that the optical constants match the original Mathematica values, and that the registry
agrees to machine precision with the INDEPENDENTLY-validated quartz+Au docs case.
"""

import unittest

import numpy as np

from shaarp.casestudy_materials import (
    CASE_STUDY_ORDER,
    build_casestudy_material,
    build_casestudy_ml_system,
    casestudy_native_wavelength,
)


class CaseStudyRegistryTests(unittest.TestCase):
    def test_full_original_material_set_present(self):
        # the 17 original setup.nb Case Study materials remain present (plus the SHAARP.si materials
        # added from the SI notebook's exact inline constants -> the registry only grows).
        # 'LiNbO3 z-cut (1064 nm)' is setup.nb dead code (never exposed by the original palette);
        # it stays in the REGISTRY as a numeric-fence anchor but is hidden from the GUI lists
        # (case-study fidelity audit).
        self.assertGreaterEqual(len(CASE_STUDY_ORDER), 24)
        for need in ("GaAs (111) (800 nm)", "Quartz z-cut (800 nm)", "KTP x-cut",
                     "LiNbO3 z-cut (1064 nm)", "Au coating (800 nm)", "MoS2", "BaTiO3",
                     "LiNbO3 (11-20) MTI X-cut", "KTP (100)"):
            self.assertIn(need, CASE_STUDY_ORDER)

    def test_gui_curation_matches_original_palettes(self):
        # GUI-visible lists = the ORIGINAL palettes exactly: 16 SHAARP.ml buttons (original order)
        # and the SHAARP.si 5-group panel (all 800 nm in the DOI group); every display label
        # resolves to a registry key; the ONLY hidden registry key is the setup.nb dead code.
        from shaarp.casestudy_materials import (GUI_ML_CASES, GUI_ML_GROUPS, GUI_SI_GROUPS,
                                                resolve_case_label)

        self.assertEqual([key for _label, key in GUI_ML_CASES], [
            "Blank linear", "Blank nonlinear", "BaTiO3", "Air",
            "LiNbO3 x-cut (1550 nm)", "LiNbO3 z-cut (1550 nm)", "KTP x-cut", "KTP y-cut",
            "ZnO (001)", "Pt (111) (1550 nm)", "Al2O3 (0001) (1550 nm)", "GaAs (111) (800 nm)",
            "Quartz x-cut (1064 nm)", "Quartz z-cut (800 nm)", "Au coating (800 nm)", "MoS2"])
        si_keys = [key for _hdr, entries in GUI_SI_GROUPS for _label, key in entries]
        self.assertEqual(si_keys, [
            "GaAs (111) (800 nm)", "LiNbO3 (11-20) MTI X-cut", "KTP (100)", "TaAs (112)",
            "GaAs (111) (1064 nm)", "LiB3O5 (LBO)", "KBBF", "LiOsO3"])
        for label, key in GUI_ML_CASES:
            self.assertEqual(resolve_case_label(label), key)
            self.assertIn(key, CASE_STUDY_ORDER)
        for _hdr, entries in GUI_ML_GROUPS:
            for label, key in entries:  # the indented multi-wavelength child rows resolve too
                self.assertEqual(resolve_case_label(label), key)
        hidden = set(CASE_STUDY_ORDER) - {k for _l, k in GUI_ML_CASES} - set(si_keys)
        self.assertEqual(hidden, {"LiNbO3 z-cut (1064 nm)"})

    def test_si_doi_cases_match_notebook_constants(self):
        # the two 'Cases in DOI' buttons transcribed (SHAARP_V1.04.nb ~40088-40320):
        # exact-constant fences. LNO-MTI stores n = sqrt(eps) in the button; the radicands are
        # asserted here verbatim. KTP (100) is epsilon-mode; both native lambda = 0.8 um.
        m = build_casestudy_material("LiNbO3 (11-20) MTI X-cut")
        self.assertEqual(m.structure.point_group, "3m")
        np.testing.assert_allclose(np.diag(m.epsilon_omega).real, [5.0756, 5.0756, 4.6673],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.diag(m.epsilon_2omega).real, [5.9248, 5.9248, 5.3319],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(m.d_voigt_pm_v.real, [
            [0, 0, 0, 0, 5.95, -13.07],
            [-13.07, 13.07, 0, 5.95, 0, 0],
            [5.95, 5.95, 34.45, 0, 0, 0]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(m.orientation.z_axes_in_lab, dtype=float),
                                   [[0, 0, 1], [0, -1, 0], [1, 0, 0]], rtol=0, atol=1e-12)
        self.assertAlmostEqual(casestudy_native_wavelength("LiNbO3 (11-20) MTI X-cut"), 0.8)

        k = build_casestudy_material("KTP (100)")
        self.assertEqual(k.structure.point_group, "mm2")
        np.testing.assert_allclose(np.diag(k.epsilon_omega).real, [3.00, 3.04, 3.34],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.diag(k.epsilon_2omega).real, [3.16, 3.19, 3.56],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(k.d_voigt_pm_v.real, [
            [0, 0, 0, 0, 1.9, 0],
            [0, 0, 0, 3.6, 0, 0],
            [2.5, 4.4, 16.9, 0, 0, 0]], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(k.orientation.z_axes_in_lab, dtype=float),
                                   [[0, 0, 1], [1, 0, 0], [0, 1, 0]], rtol=0, atol=1e-12)
        self.assertAlmostEqual(casestudy_native_wavelength("KTP (100)"), 0.8)

    def test_every_material_builds_with_valid_tensors(self):
        for name in CASE_STUDY_ORDER:
            m = build_casestudy_material(name)
            self.assertEqual(m.epsilon_omega.shape, (3, 3))
            self.assertEqual(m.epsilon_2omega.shape, (3, 3))
            self.assertEqual(m.d_voigt_pm_v.shape, (3, 6))
            R = m.orientation.rotation_matrix()
            self.assertLess(float(np.max(np.abs(R @ R.T - np.eye(3)))), 1e-9, f"{name} orientation not a rotation")

    def test_gaas_matches_mathematica_values(self):
        m = build_casestudy_material("GaAs (111) (800 nm)")
        self.assertEqual(m.structure.point_group, "-43m")
        self.assertAlmostEqual(m.structure.a, 5.65, places=6)
        self.assertAlmostEqual(m.d_voigt_pm_v[0, 3].real, 270.0, places=6)  # d14 = 270 pm/V
        self.assertAlmostEqual(float(np.sqrt(m.epsilon_omega[0, 0]).real), 3.666, places=3)
        # (111) plane normal maps onto the lab surface normal
        v = m.orientation.rotation_matrix().T @ (np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0))
        self.assertLess(min(float(np.max(np.abs(v - [0, 0, 1]))), float(np.max(np.abs(v + [0, 0, 1])))), 1e-12)

    def test_registry_quartz_au_match_validated_docs_case(self):
        # the strongest provenance check: registry vs the independently live-validated docs case
        from benchmarks.quartz_au_docs_case import AU_N_OMEGA, AU_N_2OMEGA, quartz_sellmeier_index

        q = build_casestudy_material("Quartz z-cut (800 nm)")
        self.assertAlmostEqual(float(np.sqrt(q.epsilon_omega[0, 0]).real),
                               quartz_sellmeier_index(0.8, extraordinary=False), places=10)
        self.assertAlmostEqual(float(np.sqrt(q.epsilon_omega[2, 2]).real),
                               quartz_sellmeier_index(0.8, extraordinary=True), places=10)
        self.assertAlmostEqual(q.d_voigt_pm_v[0, 0].real, 0.3, places=10)  # quartz d11
        au = build_casestudy_material("Au coating (800 nm)")
        self.assertLess(abs(complex(np.sqrt(au.epsilon_omega[0, 0])) - AU_N_OMEGA), 1e-9)
        self.assertLess(abs(complex(np.sqrt(au.epsilon_2omega[0, 0])) - AU_N_2OMEGA), 1e-9)

    def test_zcut_quartz_orientation_is_identity(self):
        q = build_casestudy_material("Quartz z-cut (800 nm)")
        self.assertTrue(np.allclose(q.orientation.rotation_matrix(), np.eye(3), atol=1e-12))

    def test_ml_system_from_casestudy_film_builds(self):
        sysm = build_casestudy_ml_system("GaAs (111) (800 nm)", thickness_um=2.0, wavelength_um=0.8)
        self.assertEqual(len(sysm.layers), 3)
        self.assertTrue(sysm.layers[1].shg_active)
        self.assertAlmostEqual(sysm.layers[1].thickness_um, 2.0)

    def test_dielectric_tensors_update_with_wavelength(self):
        # the original GUI behaviour: a dispersive material's eps tracks the fundamental wavelength
        import numpy as np

        k08 = build_casestudy_material("KTP x-cut", wavelength_um=0.8)
        k15 = build_casestudy_material("KTP x-cut", wavelength_um=1.55)
        n08 = float(np.sqrt(k08.epsilon_omega[0, 0]).real)
        n15 = float(np.sqrt(k15.epsilon_omega[0, 0]).real)
        self.assertGreater(abs(n08 - n15), 0.02, "KTP index must change with wavelength (dispersion)")
        # d tensor is wavelength-independent (Kleinman) -> unchanged
        self.assertTrue(np.allclose(k08.d_voigt_pm_v, k15.d_voigt_pm_v))

    def test_native_wavelength_reproduces_validated_quartz(self):
        import numpy as np

        from benchmarks.quartz_au_docs_case import quartz_sellmeier_index

        self.assertAlmostEqual(casestudy_native_wavelength("Quartz z-cut (800 nm)"), 0.8)
        q = build_casestudy_material("Quartz z-cut (800 nm)")  # native lambda
        self.assertAlmostEqual(float(np.sqrt(q.epsilon_omega[0, 0]).real),
                               quartz_sellmeier_index(0.8, extraordinary=False), places=9)


if __name__ == "__main__":
    unittest.main()
