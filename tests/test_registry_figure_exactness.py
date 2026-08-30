"""REGISTRY-WIDE FIGURE EXACTNESS (post-mortem gate).

Why this file exists -- a second lesson: si_figure_kwargs_from_material's orientation
branch covered only lab-PRINCIPAL-ALIGNED cases, and the one general-orientation registry
case (TaAs (112)) fell through SILENTLY to the z-cut approximation. The paper fence for that case
asserted only nonzero/lobe-family -- which the wrong-but-plausible curve also satisfied. Two
structural rules follow, both enforced here:

1. ITERATE THE REGISTRY, don't hand-pick cases: every SI case material (including any added
   later) must produce figure kwargs whose curve equals the independent validated numeric
   arbitrary-Jones solver at spot polarizations. A new tilted material is covered the day it
   lands in the registry -- no enumeration gap.
2. Zero-SHG materials must be CONSISTENTLY zero through both routes (a silent fallback that
   invents signal would also fail).
"""

import math
import unittest

import numpy as np

from shaarp.casestudy_materials import build_casestudy_material, casestudy_native_wavelength
from shaarp.shaarp_gui import _epsilon_lab_of, si_figure_kwargs_from_material, si_polarimetry_curve
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab


def _registry_names():
    from shaarp.casestudy_materials import _data

    return list(_data()["materials"].keys())


class RegistryFigureExactness(unittest.TestCase):
    THETA = 40.0

    def _reference_pair(self, mat, phi_rad):
        """(I_p, I_s) from the independent validated numeric solver (lab tensors via the same
        helpers the figure path uses -- the equality under test is figure-kwargs -> curve)."""
        eps_w = _epsilon_lab_of(mat, omega=True)
        eps_2 = _epsilon_lab_of(mat, omega=False)
        rot = np.asarray(mat.orientation.rotation_matrix(), dtype=float)
        d_lab = np.asarray(rotate_d_voigt_crystal_to_lab(
            np.asarray(mat.d_voigt_pm_v, dtype=complex), rot), dtype=complex)
        r = solve_single_interface_shg(
            eps_w, eps_2, d_lab, incident_index_omega=1.0, incident_index_2omega=1.0,
            incident_theta_rad=math.radians(self.THETA),
            incident_jones=(math.sin(phi_rad), math.cos(phi_rad)), omega=1.0, mu=1.0, eps0=1.0)
        ip = float(np.sum(np.abs(np.asarray(r.reflected_p_2omega.electric)) ** 2))
        is_ = float(np.sum(np.abs(np.asarray(r.reflected_s_2omega.electric)) ** 2))
        return ip, is_

    def test_every_registry_case_figure_matches_the_numeric_solver(self):
        n_phi = 13
        phi_grid = np.linspace(0.0, 360.0, n_phi)
        probe_idx = [2, 5]  # 60 and 150 deg -- off the s/p endpoints, cross terms active
        checked = zero_cases = 0
        for name in _registry_names():
            with self.subTest(case=name):
                try:
                    mat = build_casestudy_material(
                        name, wavelength_um=casestudy_native_wavelength(name))
                except Exception as exc:  # registry entries must all build
                    self.fail(f"{name}: material failed to build: {exc}")
                kw = si_figure_kwargs_from_material(mat)
                pg = kw.pop("point_group")
                try:
                    cur = si_polarimetry_curve(pg, theta_deg=self.THETA, n_phi=n_phi, **kw)
                except Exception as exc:
                    self.fail(f"{name}: figure curve raised: {exc}")
                peak = max(float(np.max(np.asarray(cur["intensity_p"], dtype=float))),
                           float(np.max(np.asarray(cur["intensity_s"], dtype=float))))
                refs = [self._reference_pair(mat, math.radians(float(phi_grid[k])))
                        for k in probe_idx]
                ref_peak = max(max(r) for r in refs)
                if ref_peak < 1e-20:
                    # zero-SHG material (centrosymmetric / linear / vacuum): the figure route
                    # must agree that there is no signal -- inventing one would be the silent-
                    # fallback failure mode in reverse.
                    self.assertLess(peak, 1e-16,
                                    f"{name}: solver says zero SHG but the figure curve is not")
                    zero_cases += 1
                    continue
                for k, (ip_ref, is_ref) in zip(probe_idx, refs):
                    ip_fig = float(cur["intensity_p"][k])
                    is_fig = float(cur["intensity_s"][k])
                    scale = max(ip_ref, is_ref, 1e-300)
                    self.assertLess(abs(ip_fig - ip_ref) / scale, 1e-6,
                                    f"{name}: figure I_p deviates from the validated solver at "
                                    f"phi={phi_grid[k]:.0f} deg (silent-approximation class)")
                    self.assertLess(abs(is_fig - is_ref) / scale, 1e-6,
                                    f"{name}: figure I_s deviates from the validated solver at "
                                    f"phi={phi_grid[k]:.0f} deg (silent-approximation class)")
                checked += 1
        self.assertGreaterEqual(checked, 10, "registry iteration went vacuous")
        self.assertGreaterEqual(zero_cases, 2, "expected the linear/centrosymmetric cases")


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
