"""Isotropic (fully degenerate) crystals through the SHAARP.si compat path -- the merged GUI's
DEFAULT tab state (point group -43m, n_o == n_e), which previously crashed with
``LinAlgError: Singular matrix``.

Three degenerate flavors were fixed (all collision/isotropy-gated so anisotropic validated
configs are untouched):
1. branch-index label collision -> same eigenvector picked twice (omega selector);
2. duplicated eigenvectors at the degenerate eigenvalue despite distinct indices (2omega) ->
   ordinary rebuilt as the orthogonal transverse polarization k x E (exact at isotropy);
3. the V/W angle-root iteration stalling on the degenerate landscape -> exact Snell shortcut.

Validation here is physics, not run-doesn't-crash: classical Fresnel at the omega boundary
(exact), and the reflected SHG p-in theta sweep against the PUBLISHED-GaAs(111)-validated
full-analytical closed form (absolute agreement, no scale fudge), plus the continuity lens.
"""

import unittest

import numpy as np

from shaarp.shaarp_gui import build_custom_si_material, compute_si_gui_result, count_discontinuities
from shaarp.si_compat import solve_shaarp_si_linear_omega, solve_shaarp_si_reflected_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab


def _gui_default_material():
    # the merged GUI's default SI field values: -43m, n_w=2.0, n_2w=2.2, d14=1, z-cut
    return build_custom_si_material("-43m")


class IsotropicOmegaBoundaryTests(unittest.TestCase):
    def test_omega_boundary_matches_classical_fresnel(self):
        m = _gui_default_material()
        n1, n2, th = 1.0, 2.0, np.deg2rad(45.0)
        tht = np.arcsin(n1 * np.sin(th) / n2)
        rs_ref = (n1 * np.cos(th) - n2 * np.cos(tht)) / (n1 * np.cos(th) + n2 * np.cos(tht))
        rp_ref = (n2 * np.cos(th) - n1 * np.cos(tht)) / (n2 * np.cos(th) + n1 * np.cos(tht))
        ts_ref = 2 * n1 * np.cos(th) / (n1 * np.cos(th) + n2 * np.cos(tht))
        tp_ref = 2 * n1 * np.cos(th) / (n2 * np.cos(th) + n1 * np.cos(tht))
        for pol, r_ref, t_ref in (("s", rs_ref, ts_ref), ("p", rp_ref, tp_ref)):
            res = solve_shaarp_si_linear_omega(
                epsilon_omega_crystal=m.epsilon_omega,
                epsilon_2omega_crystal=m.epsilon_2omega,
                orientation_matrix=m.orientation.rotation_matrix(),
                theta_in_rad=th,
                incident_polarization=pol,
            )
            r_s = np.linalg.norm(res.reflected_s.electric)
            r_p = np.linalg.norm(res.reflected_p.electric)
            t_tot = np.linalg.norm(res.transmitted_ext.electric + res.transmitted_ord.electric)
            same = r_s if pol == "s" else r_p
            cross = r_p if pol == "s" else r_s
            self.assertGreater(abs(r_ref), 0.1, "nonzero-reference guard")
            self.assertLess(abs(same - abs(r_ref)), 1e-9, f"{pol}: |r| != classical Fresnel")
            self.assertLess(cross, 1e-12, f"{pol}: isotropic cross-polarized reflection must vanish")
            self.assertLess(abs(t_tot - abs(t_ref)), 1e-9, f"{pol}: |t_total| != classical Fresnel")


class IsotropicReflectedSHGTests(unittest.TestCase):
    def test_gui_default_material_shg_runs(self):
        # regression for the original crash: the merged GUI's default SI tab Run
        res = compute_si_gui_result("SHG Simulation", theta_deg=45.0, material=_gui_default_material())
        self.assertEqual(res.kind, "si_numeric_shaarp_si_compat")

    def test_s_incidence_selection_rule_zero_is_exact(self):
        # z-cut -43m with s-incidence has E = (0, Ey, 0) -> ALL d14/d25/d36 source products
        # vanish -> the SHG is EXACTLY zero. Pin it as intentional physics, not a solver hole.
        m = _gui_default_material()
        d_lab = rotate_d_voigt_crystal_to_lab(m.d_voigt_pm_v, m.orientation.rotation_matrix())
        res = solve_shaarp_si_reflected_shg(
            epsilon_omega_crystal=m.epsilon_omega, epsilon_2omega_crystal=m.epsilon_2omega,
            orientation_matrix=m.orientation.rotation_matrix(), d_voigt_lab=d_lab,
            theta_in_rad=np.deg2rad(45.0), incident_polarization="s",
            normal_incidence_2omega_branch_policy="shaarp_reference_like",
        )
        self.assertEqual(float(np.linalg.norm(res.reflected_total.electric)), 0.0)

    def test_p_in_sweep_matches_validated_closed_form_and_is_continuous(self):
        import sympy as sp

        from shaarp.symbolic import solve_si_shg_full_analytical_symbolic

        m = _gui_default_material()
        R = m.orientation.rotation_matrix()
        d_lab = rotate_d_voigt_crystal_to_lab(m.d_voigt_pm_v, R)
        thetas = np.array([5.0, 20.0, 35.0, 50.0, 65.0, 75.0])
        py_vals = []
        for th in thetas:
            res = solve_shaarp_si_reflected_shg(
                epsilon_omega_crystal=m.epsilon_omega, epsilon_2omega_crystal=m.epsilon_2omega,
                orientation_matrix=R, d_voigt_lab=d_lab, theta_in_rad=float(np.deg2rad(th)),
                incident_polarization="p",
                normal_incidence_2omega_branch_policy="shaarp_reference_like",
            )
            e = np.asarray(res.reflected_total.electric)
            py_vals.append(float(np.vdot(e, e).real))
        py_vals = np.array(py_vals)
        self.assertTrue(np.all(py_vals > 0), "nonzero-output guard (p-in is NOT a selection zero)")
        self.assertEqual(count_discontinuities(py_vals), 0, "continuity lens")

        phi = sp.Symbol("phi", real=True)
        dm = sp.zeros(3, 6)
        dm[0, 3] = 1
        dm[1, 4] = 1
        dm[2, 5] = 1
        cf_vals = []
        for th in thetas:
            sol = solve_si_shg_full_analytical_symbolic(
                eps_x_omega=4.0, eps_y_omega=4.0, eps_z_omega=4.0,
                eps_x_2omega=4.84, eps_y_2omega=4.84, eps_z_2omega=4.84,
                d_voigt_lab=dm, incident_theta_rad=float(np.deg2rad(th)),
                phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
            )
            e_s = complex(sol.reflected_s.subs(phi, 0).evalf())
            e_p = complex(sol.reflected_p.subs(phi, 0).evalf())
            cf_vals.append(abs(e_s) ** 2 + abs(e_p) ** 2)
        cf_vals = np.array(cf_vals)
        self.assertTrue(np.all(cf_vals > 0), "closed-form nonzero guard")
        # ABSOLUTE agreement (same natural units end to end) -- no normalization fudge
        self.assertLess(float(np.max(np.abs(py_vals / cf_vals - 1.0))), 1e-5,
                        "compat solver must match the GaAs(111)-validated closed form absolutely")


if __name__ == "__main__":
    unittest.main()
