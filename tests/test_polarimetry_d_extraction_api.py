"""Public d-extraction API (shaarp.polarimetry_extraction.extract_si_d_voigt): recover the
SHG coefficient tensor from a reflected-SHG polarimetry scan, via the closed-form model.

Validated by recovering a KNOWN d-tensor (the "measurements" are produced by the independent
numeric arbitrary-Jones solver) through BOTH extraction methods:
  * field amplitude -> exact d (up to identifiability),
  * intensity only -> d up to overall sign (the physical ambiguity),
plus the honest identifiability report (a single geometry is NOT identifiable).
"""

import math
import unittest

import numpy as np

from shaarp.polarimetry_extraction import extract_si_d_voigt
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

EPS_W = [2.2**2, 2.2**2, 2.5**2]      # uniaxial: azimuth about z rotates only d (consistent)
EPS_2W = [2.4**2, 2.4**2, 2.7**2]
POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
TRUE = np.array([0.30, 0.50, 0.60, 0.80, 0.40, 0.20])
GEOMETRIES = [(0.35, 0.0), (0.35, 0.6), (0.35, 1.2), (0.75, 0.0), (0.75, 0.6), (0.75, 1.2)]
PHIS = list(np.linspace(0.25, math.pi - 0.25, 6))


def _rz(az):
    c, s = math.cos(az), math.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _measure(theta, azimuth, phi):
    """Forward model 'measurement': complex reflected (E_s, E_p) for the sample rotated by
    `azimuth`, from the independent numeric arbitrary-Jones solver."""
    d_num = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, TRUE):
        d_num[m, ell] = v
    d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d_num, _rz(azimuth)), dtype=complex))
    r = solve_single_interface_shg(
        np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), d_rot,
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
    )
    c = np.asarray(r.coefficients)
    return complex(c[0]), complex(c[1])


def _measure_transmitted(theta, azimuth, phi):
    """Transmitted (Maker) forward 'measurement': the total transmitted SHG field 3-vector for
    the sample rotated by Rz(azimuth), from the numeric arbitrary-Jones SI solver."""
    d_num = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, TRUE):
        d_num[m, ell] = v
    d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d_num, _rz(azimuth)), dtype=complex))
    r = solve_single_interface_shg(
        np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), d_rot,
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
    )
    return np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)


class ExtractSiDVoigtApiTests(unittest.TestCase):
    def test_field_method_recovers_full_tensor(self):
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES, phi_values=PHIS, measure=_measure, method="field", mu=1, eps0=1,
        )
        self.assertEqual(res.method, "field")
        self.assertFalse(res.sign_ambiguous)
        self.assertTrue(res.identifiable, f"not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-10, "model inconsistent with data")
        self.assertLess(np.max(np.abs(res.values - TRUE)), 1e-9, "field d-extraction failed")
        # the placed Voigt tensor matches at the requested positions
        for (m, ell), v in zip(POSITIONS, TRUE):
            self.assertAlmostEqual(res.d_voigt[m, ell], v, places=8)

    def test_intensity_method_recovers_tensor_up_to_sign(self):
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=GEOMETRIES, phi_values=PHIS, measure=_measure, method="intensity", mu=1, eps0=1,
        )
        self.assertEqual(res.method, "intensity")
        self.assertTrue(res.sign_ambiguous)
        self.assertLess(res.residual, 1e-9, "Gram model inconsistent with measured intensity")
        # determined up to overall sign -> compare both signs
        err = min(np.max(np.abs(res.values - TRUE)), np.max(np.abs(res.values + TRUE)))
        self.assertLess(err, 1e-8, "intensity d-extraction failed (beyond the sign ambiguity)")

    def test_single_geometry_reports_not_identifiable(self):
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=[(0.5, 0.0)], phi_values=list(np.linspace(0.15, math.pi - 0.15, 9)),
            measure=_measure, method="field", mu=1, eps0=1,
        )
        # one scan cannot determine the whole tensor -> honest report
        self.assertFalse(res.identifiable)
        self.assertLess(res.rank, res.n_components)

    def test_transmitted_observable_recovers_tensor(self):
        # transmitted (Maker) channel: 3 observables per scan point -> full rank with fewer
        # geometries than reflected (here 3).
        res = extract_si_d_voigt(
            eps_omega_principal=EPS_W, eps_2omega_principal=EPS_2W, d_positions=POSITIONS,
            geometries=[(0.35, 0.0), (0.35, 0.9), (0.75, 0.0)],
            phi_values=list(np.linspace(0.25, math.pi - 0.25, 5)),
            measure=_measure_transmitted, method="field", observable="transmitted", mu=1, eps0=1,
        )
        self.assertTrue(res.identifiable, f"transmitted not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-10, "transmitted model inconsistent with data")
        self.assertLess(np.max(np.abs(res.values - TRUE)), 1e-9, "transmitted d-extraction failed")


# ----------------------------- ML thin-film extractor -----------------------------
from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_tensors_jones  # noqa: E402
from shaarp.polarimetry_extraction import extract_ml_film_d_voigt  # noqa: E402

ML_NF, ML_NS, ML_NF2, ML_NS2, ML_H = 2.1, 1.5, 2.25, 1.55, 1.3
ML_GEOMETRIES = [(0.35, 0.0), (0.35, 0.6), (0.35, 1.2), (0.75, 0.0), (0.75, 0.6), (0.75, 1.2)]


def _ml_measure(theta, azimuth, phi):
    """Forward 'measurement' for a film whose sample is rotated by Rz(azimuth): complex
    reflected (E_s, E_p) from the independent numeric arbitrary-Jones multilayer workflow."""
    d_num = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, TRUE):
        d_num[m, ell] = v
    d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d_num, _rz(azimuth)), dtype=complex))
    r = solve_multilayer_shg_from_tensors_jones(
        incident_index_omega=1.0, top_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones_sp=(math.sin(phi), math.cos(phi)),
        layer_epsilon_omega_lab=[np.diag([ML_NF**2] * 3).astype(complex)],
        substrate_epsilon_omega_lab=np.diag([ML_NS**2] * 3).astype(complex),
        layer_d_voigt_lab=[d_rot], layer_epsilon_2omega_lab=[np.diag([ML_NF2**2] * 3).astype(complex)],
        substrate_epsilon_2omega_lab=np.diag([ML_NS2**2] * 3).astype(complex),
        thicknesses=[ML_H], omega=1.0, mu=1.0, eps0=1.0,
    )
    c = np.asarray(r.shg.coefficients)
    return complex(c[0]), complex(c[1])


class ExtractMlFilmDVoigtApiTests(unittest.TestCase):
    def test_field_method_recovers_film_tensor(self):
        res = extract_ml_film_d_voigt(
            film_index_omega=ML_NF, substrate_index_omega=ML_NS, film_index_2omega=ML_NF2,
            substrate_index_2omega=ML_NS2, thickness=ML_H, d_positions=POSITIONS,
            geometries=ML_GEOMETRIES, phi_values=list(np.linspace(0.25, math.pi - 0.25, 5)),
            measure=_ml_measure, method="field", mu=1.0, eps0=1.0,  # match the measure's units
        )
        self.assertTrue(res.identifiable, f"film tensor not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(res.residual, 1e-10, "ML model inconsistent with data")
        self.assertLess(np.max(np.abs(res.values - TRUE)), 1e-9, "ML field d-extraction failed")

    def test_intensity_method_recovers_film_tensor_up_to_sign(self):
        res = extract_ml_film_d_voigt(
            film_index_omega=ML_NF, substrate_index_omega=ML_NS, film_index_2omega=ML_NF2,
            substrate_index_2omega=ML_NS2, thickness=ML_H, d_positions=POSITIONS,
            geometries=ML_GEOMETRIES, phi_values=list(np.linspace(0.2, math.pi - 0.2, 6)),
            measure=_ml_measure, method="intensity", mu=1.0, eps0=1.0,  # match the measure's units
        )
        self.assertTrue(res.sign_ambiguous)
        self.assertLess(res.residual, 1e-9, "ML Gram model inconsistent with measured intensity")
        err = min(np.max(np.abs(res.values - TRUE)), np.max(np.abs(res.values + TRUE)))
        self.assertLess(err, 1e-8, "ML intensity d-extraction failed (beyond the sign ambiguity)")

    def test_transmitted_observable_end_to_end(self):
        """observable='transmitted' (the thin-film Maker geometry) recovers d from the
        substrate-transmitted SHG. A single d component is identifiable from incidence-angle
        diversity alone (no slow symbolic azimuth needed)."""
        pos = [(0, 3)]  # one component -> rank-1 identifiable from theta-only

        def measure_t(theta, az, phi):
            d = np.zeros((3, 6))
            d[0, 3] = 0.7
            r = solve_multilayer_shg_from_tensors_jones(
                incident_index_omega=1.0, top_index_2omega=1.0, incident_theta_rad=theta,
                incident_jones_sp=(math.sin(phi), math.cos(phi)),
                layer_epsilon_omega_lab=[np.diag([ML_NF**2] * 3).astype(complex)],
                substrate_epsilon_omega_lab=np.diag([ML_NS**2] * 3).astype(complex),
                layer_d_voigt_lab=[d], layer_epsilon_2omega_lab=[np.diag([ML_NF2**2] * 3).astype(complex)],
                substrate_epsilon_2omega_lab=np.diag([ML_NS2**2] * 3).astype(complex),
                thicknesses=[ML_H], omega=1.0, mu=1.0, eps0=1.0,
            )
            c = np.asarray(r.shg.coefficients)
            return complex(c[6]), complex(c[7])

        res = extract_ml_film_d_voigt(
            film_index_omega=ML_NF, substrate_index_omega=ML_NS, film_index_2omega=ML_NF2,
            substrate_index_2omega=ML_NS2, thickness=ML_H, d_positions=pos,
            geometries=[(0.3, 0.0), (0.6, 0.0)], phi_values=list(np.linspace(0.3, 2.8, 5)),
            measure=measure_t, method="field", observable="transmitted", mu=1.0, eps0=1.0,  # match the measure's units
        )
        self.assertTrue(res.identifiable, f"transmitted not identifiable (rank {res.rank}/{res.n_components})")
        self.assertLess(abs(float(np.real(res.values[0])) - 0.7), 1e-9, "ML transmitted d-extraction failed")


class PolarimetryPublicExportsTests(unittest.TestCase):
    """The polarimetry + d-extraction capability is discoverable from the package top level."""

    def test_top_level_exports_present(self):
        import shaarp

        expected = [
            "solve_si_shg_full_analytical_symbolic", "SymbolicSIFullPolarimetry",
            "solve_single_film_shg_symbolic_polarimetry", "SymbolicThicknessSHGSolution",
            "solve_single_film_shg_symbolic_thickness", "solve_single_film_shg_symbolic_thickness_and_d",
            "extract_si_d_voigt", "extract_ml_film_d_voigt", "DExtractionResult",
            "make_shaarp_gui", "compute_si_gui_result", "compute_ml_gui_result",
            "analytical_expression_text",
        ]
        for name in expected:
            self.assertTrue(hasattr(shaarp, name), f"shaarp.{name} not exported")
            self.assertIn(name, shaarp.__all__, f"{name} missing from shaarp.__all__")
        self.assertTrue(callable(shaarp.extract_si_d_voigt))
        self.assertTrue(callable(shaarp.extract_ml_film_d_voigt))


if __name__ == "__main__":
    unittest.main()
