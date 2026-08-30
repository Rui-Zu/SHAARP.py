import unittest

import numpy as np

from shaarp.config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry, with_sample_azimuth_deg
from shaarp.tensors import rotate_rank2_crystal_to_lab, rotate_rank3_crystal_to_lab


class OrientationTests(unittest.TestCase):
    def test_all_nonzero_cubic_miller_surface_orientation_is_orthonormal(self):
        orientation = CrystalOrientation.from_cubic_miller_surface((1, 2, 3), in_plane_hint=(1, -1, 0.5))
        rotation = orientation.rotation_matrix()
        expected_normal = np.array([1.0, 2.0, 3.0]) / np.linalg.norm([1.0, 2.0, 3.0])

        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotation[:, 2], expected_normal, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
        self.assertTrue(np.all(np.abs(rotation[:, 2]) > 0.0))

    def test_complex_orientation_matrix_is_rejected_not_silently_cast(self):
        orientation = np.eye(3, dtype=complex)
        orientation[0, 1] = 0.1j

        with self.assertRaisesRegex(ValueError, "complex spatial rotations"):
            CrystalOrientation(orientation).rotation_matrix()
        with self.assertRaisesRegex(ValueError, "complex spatial rotations"):
            rotate_rank2_crystal_to_lab(np.eye(3), orientation)
        with self.assertRaisesRegex(ValueError, "complex spatial rotations"):
            rotate_rank3_crystal_to_lab(np.ones((3, 3, 3)), orientation)

    def test_general_miller_surface_uses_reciprocal_lattice_metric(self):
        structure = CrystalStructure(a=4.1, b=5.2, c=6.3, alpha_deg=76.0, beta_deg=88.0, gamma_deg=103.0)
        hkl = np.array([1.0, 2.0, 3.0])
        orientation = CrystalOrientation.from_miller_surface(
            structure,
            hkl,
            in_plane_direction_uvw=(2.0, -1.0, 1.0),
        )
        rotation = orientation.rotation_matrix()
        reciprocal_normal = structure.miller_plane_normal(hkl)
        naive_cartesian = hkl / np.linalg.norm(hkl)

        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotation[:, 2], reciprocal_normal, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
        self.assertGreater(np.linalg.norm(reciprocal_normal - naive_cartesian), 0.1)

    def test_shaarp_hkl_orientation_uses_reciprocal_plane_and_direct_vertical(self):
        structure = CrystalStructure(point_group="1", a=4.1, b=5.2, c=6.3, alpha_deg=76.0, beta_deg=88.0, gamma_deg=103.0)
        plane_hkl = np.array([1.0, 2.0, 3.0])
        vertical_uvw = np.array([1.0, 1.0, -1.0])
        orientation = CrystalOrientation.from_shaarp_miller_surface(structure, plane_hkl, vertical_uvw)
        rotation = orientation.rotation_matrix()

        reciprocal_normal = structure.miller_plane_normal(plane_hkl)
        direct_vertical = structure.direct_lattice_direction(vertical_uvw)
        target_rows = np.vstack(
            [
                np.cross(direct_vertical, reciprocal_normal) / np.linalg.norm(np.cross(direct_vertical, reciprocal_normal)),
                direct_vertical,
                reciprocal_normal,
            ]
        )
        shaarp_axes = structure.shaarp_physics_axes()
        expected = (target_rows @ np.linalg.inv(shaarp_axes)).T
        expected = expected / np.linalg.norm(expected, axis=1)[:, None]

        np.testing.assert_allclose(rotation, expected, atol=1e-12)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertGreater(np.linalg.norm(rotation[:, 2] - reciprocal_normal), 0.1)

    def test_shaarp_hkl_orientation_rejects_vertical_not_in_plane(self):
        structure = CrystalStructure(a=4.1, b=5.2, c=6.3, alpha_deg=76.0, beta_deg=88.0, gamma_deg=103.0)
        with self.assertRaisesRegex(ValueError, "vertical_uvw must lie"):
            CrystalOrientation.from_shaarp_miller_surface(structure, (1, 2, 3), (2, -1, 1))

    def test_shaarp_physics_axes_accept_mathematica_point_group_label_aliases(self):
        tetragonal = CrystalStructure(a=3.2, b=3.2, c=5.1, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)
        hexagonal = CrystalStructure(a=3.2, b=3.2, c=5.1, alpha_deg=90.0, beta_deg=90.0, gamma_deg=120.0)

        aliases = {
            "Overscript[4, _]": (tetragonal, r"\!\(\*OverscriptBox[\(4\), \(_\)]\)"),
            "Overscript[4, _]2m": (tetragonal, r"\!\(\*OverscriptBox[\(4\), \(_\)]\)2m"),
            "Overscript[4, _]3m": (hexagonal, r"\!\(\*OverscriptBox[\(4\), \(_\)]\)3m"),
            "Overscript[6, _]": (hexagonal, r"\!\(\*OverscriptBox[\(6\), \(_\)]\)"),
            "Overscript[6, _]m2": (hexagonal, r"\!\(\*OverscriptBox[\(6\), \(_\)]\)m2"),
            r"\[Infinity]": (hexagonal, "3"),
            r"\[Infinity]m": (hexagonal, "3m"),
            r"\[Infinity]2": (hexagonal, "32"),
        }
        for alias, (structure, canonical) in aliases.items():
            with self.subTest(alias=alias):
                np.testing.assert_allclose(
                    structure.shaarp_physics_axes(alias),
                    structure.shaarp_physics_axes(canonical),
                    atol=1e-12,
                )

    def test_cubic_miller_surface_matches_general_reciprocal_result(self):
        structure = CrystalStructure(a=3.0, b=3.0, c=3.0, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)
        hkl = (1, 2, 3)
        cubic = CrystalOrientation.from_cubic_miller_surface(hkl, in_plane_hint=(1, -1, 0.5)).rotation_matrix()
        general = CrystalOrientation.from_miller_surface(
            structure,
            hkl,
            in_plane_direction_uvw=(1, -1, 0.5),
        ).rotation_matrix()

        np.testing.assert_allclose(general[:, 2], cubic[:, 2], atol=1e-12)
        np.testing.assert_allclose(general @ general.T, np.eye(3), atol=1e-12)

    def test_lab_azimuth_rotates_tensor_about_surface_normal(self):
        orientation = CrystalOrientation.lab_azimuth_deg(30.0)
        eps_crystal = np.diag([2.0, 3.0, 4.0])
        eps_lab = rotate_rank2_crystal_to_lab(eps_crystal, orientation.rotation_matrix())
        angle = np.deg2rad(30.0)
        lab_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(eps_lab, lab_rotation @ eps_crystal @ lab_rotation.T, atol=1e-12)

    def test_lab_azimuth_is_distinct_from_miller_surface_normal(self):
        base = CrystalOrientation.from_cubic_miller_surface((1, 2, 3), in_plane_hint=(1, -1, 0.5))
        rotated = base.with_lab_azimuth_deg(45.0)
        base_tensor = rotate_rank2_crystal_to_lab(np.diag([2.0, 3.0, 4.0]), base.rotation_matrix())
        rotated_tensor = rotate_rank2_crystal_to_lab(np.diag([2.0, 3.0, 4.0]), rotated.rotation_matrix())
        angle = np.deg2rad(45.0)
        lab_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        np.testing.assert_allclose(rotated.rotation_matrix() @ rotated.rotation_matrix().T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotated_tensor, lab_rotation @ base_tensor @ lab_rotation.T, atol=1e-12)

    def test_system_sample_azimuth_rotates_material_orientation_not_polarizer_angle(self):
        structure = CrystalStructure(point_group="1")
        material = Material(
            name="m",
            structure=structure,
            orientation=CrystalOrientation(),
            epsilon_omega=np.diag([2.0, 3.0, 4.0]),
            epsilon_2omega=np.diag([2.1, 3.1, 4.1]),
            d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
        )
        system = MultilayerSystem(
            wavelength_um=1.0,
            polarimetry=Polarimetry(theta_deg=12.0, phi_deg=33.0, psi_deg=44.0),
            layers=[
                Layer("top", material, shg_active=False),
                Layer("film", material, thickness_um=0.2, shg_active=True),
                Layer("substrate", material, shg_active=False),
            ],
        )

        rotated = with_sample_azimuth_deg(system, 25.0)

        self.assertEqual(rotated.polarimetry.phi_deg, 33.0)
        self.assertEqual(rotated.polarimetry.psi_deg, 44.0)
        np.testing.assert_allclose(rotated.layers[0].material.orientation.rotation_matrix(), np.eye(3), atol=1e-12)
        self.assertFalse(
            np.allclose(rotated.layers[1].material.orientation.rotation_matrix(), system.layers[1].material.orientation.rotation_matrix())
        )
        self.assertFalse(
            np.allclose(rotated.layers[2].material.orientation.rotation_matrix(), system.layers[2].material.orientation.rotation_matrix())
        )


if __name__ == "__main__":
    unittest.main()
