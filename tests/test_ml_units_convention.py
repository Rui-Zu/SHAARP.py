"""Multilayer units convention -- the SI-units 'telegraph' ROOT CAUSE, locked.

The multilayer bases are built with the c=1 convention (omega = 2*pi/lambda_um, k = n*omega), so
the inhomogeneous operator curl_curl(k) - omega^2*mu*eps0*eps is dispersion-consistent ONLY at
natural units (mu*eps0 = 1). At physical SI constants the eps term is ~1e17 smaller than
curl_curl, leaving a near-singular operator whose per-angle branchy solves produced the telegraph
artifact (the transmitted field hopping between exact E/n fractions) -- NOT an eigen-normalization
bug and NOT genuine ill-conditioning (those were earlier misdiagnoses; the cond>=1e12 flag was a
symptom of the wrong equation).

These tests pin the resolution: (1) the multilayer entry points DEFAULT to natural units (the
live-Mathematica-validated convention); (2) passing SI constants emits a loud RuntimeWarning;
(3) the exact window that telegraphed is natively smooth at the defaults.
"""

import inspect
import unittest
import warnings
from dataclasses import replace

import numpy as np

from shaarp.multilayer_shg_boundary import (
    sample_rotate_transmitted_2omega_amplitudes,  # noqa: F401 (import sanity)
    solve_multilayer_maker_fringes_sweep,
    solve_multilayer_shg_from_fundamental,
    solve_multilayer_shg_from_system_polarimetry,
)
from shaarp.waves import EPS0, MU0


def _system():
    from examples.maker_fringes_dense import _system as build

    return build()


class MlUnitsConventionTests(unittest.TestCase):
    def test_multilayer_defaults_are_natural_units(self):
        for fn in (solve_multilayer_maker_fringes_sweep, solve_multilayer_shg_from_system_polarimetry,
                   solve_multilayer_shg_from_fundamental):
            params = inspect.signature(fn).parameters
            self.assertEqual(float(params["mu"].default), 1.0, f"{fn.__name__} mu default must be natural")
            self.assertEqual(float(params["eps0"].default), 1.0, f"{fn.__name__} eps0 default must be natural")

    def test_symbolic_multilayer_defaults_are_natural_units(self):
        """The SYMBOLIC multilayer entry points must default to natural units too.

        June locked this for the boundary path but missed this module, so all six of its entry
        points defaulted to SI -- any caller omitting mu/eps0 silently got the
        dispersion-inconsistent operator. That is the bug's mechanism."""
        from shaarp import multilayer_shg_symbolic as MS

        entries = [getattr(MS, n) for n in dir(MS)
                   if n.startswith(("solve_multilayer_shg_symbolic", "solve_single_film_shg_symbolic"))
                   and callable(getattr(MS, n))]
        self.assertGreaterEqual(len(entries), 6, "expected the symbolic entry points to be found")
        for fn in entries:
            params = inspect.signature(fn).parameters
            for name in ("mu", "eps0"):
                if name in params and params[name].default is not inspect.Parameter.empty:
                    self.assertEqual(float(params[name].default), 1.0,
                                     f"{fn.__name__} {name} default must be natural units")

    def test_si_constants_emit_dispersion_warning(self):
        sysm = _system()
        s = replace(sysm, polarimetry=replace(sysm.polarimetry, theta_deg=26.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_multilayer_shg_from_system_polarimetry(s, mu=MU0, eps0=EPS0)
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(any("dispersion-INCONSISTENT" in m for m in msgs),
                        "SI constants must trigger the dispersion-inconsistency warning")

    def test_symbolic_path_si_constants_emit_dispersion_warning(self):
        """The SYMBOLIC (Partial Analytical) path must warn too.

        It did not, for months: `_warn_if_units_break_dispersion` was defined by the June lock but
        had ZERO callers package-wide, and every fence above drives the BOUNDARY path. So
        `_run_ml_partial_analytical_polarimetry` quietly ran at SI constants -- operator condition
        7.7e17, 30/30 solves flagging ill_conditioned, and Partial Analytical breaking the
        crystal's own point-group symmetry by 1.1e-2 (tests/test_pa_crystal_symmetry).
        Without this fence, deleting the new call site would leave the suite green."""
        from shaarp.multilayer_shg_symbolic import solve_multilayer_shg_symbolic_polarimetry

        kwargs = self._symbolic_polarimetry_kwargs()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_multilayer_shg_symbolic_polarimetry(mu=MU0, eps0=EPS0, **kwargs)
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(any("dispersion-INCONSISTENT" in m for m in msgs),
                        "SI constants on the SYMBOLIC path must trigger the warning; "
                        f"got {msgs}")

    def test_symbolic_path_natural_units_do_not_warn(self):
        from shaarp.multilayer_shg_symbolic import solve_multilayer_shg_symbolic_polarimetry

        kwargs = self._symbolic_polarimetry_kwargs()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_multilayer_shg_symbolic_polarimetry(mu=1.0, eps0=1.0, **kwargs)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, RuntimeWarning) and "dispersion" in str(w.message)]
        self.assertFalse(msgs, f"natural units must not warn on the symbolic path; got {msgs}")

    def _symbolic_polarimetry_kwargs(self):
        """A minimal single-film symbolic-polarimetry call, built the way the GUI builds one."""
        import sympy as sp

        from shaarp.multilayer_basis import (build_multilayer_2omega_basis,
                                             build_multilayer_omega_basis)
        from shaarp.symbolic import d_voigt_symbolic

        common = dict(incident_index=1.0, incident_theta_rad=0.5,
                      layer_epsilon_lab=[np.diag([2.1 ** 2] * 3).astype(complex)],
                      substrate_epsilon_lab=np.diag([1.5 ** 2] * 3).astype(complex), omega=1.0)
        eps2 = [np.diag([2.25 ** 2] * 3).astype(complex)]
        return dict(
            omega_basis_s=build_multilayer_omega_basis(incident_polarization="s", **common),
            omega_basis_p=build_multilayer_omega_basis(incident_polarization="p", **common),
            twoomega_basis=build_multilayer_2omega_basis(
                top_index_2omega=1.0, tangential_index_omega=1.0, incident_theta_rad=0.5,
                layer_epsilon_2omega_lab=eps2,
                substrate_epsilon_2omega_lab=np.diag([1.55 ** 2] * 3).astype(complex), omega_2=2.0),
            layer_d_voigt_symbolic=[sp.Matrix(d_voigt_symbolic("-43m"))],
            layer_epsilon_2omega_lab=eps2,
            thickness_symbols=[sp.Symbol("h", positive=True)],
            phi_symbol=sp.Symbol("phi", real=True),
        )

    def test_natural_units_do_not_warn(self):
        sysm = _system()
        s = replace(sysm, polarimetry=replace(sysm.polarimetry, theta_deg=26.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solve_multilayer_shg_from_system_polarimetry(s)
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertFalse(any("dispersion-INCONSISTENT" in m for m in msgs))

    def test_sweep_vs_point_offset_is_the_source_policy_default(self):
        """PINS the flagged ~1.4% sweep-vs-point offset: it is NOT a projection or
        solver discrepancy -- the Maker sweep defaults to inhomogeneous_source_policy=
        'forward_only' (the Mathematica-validated MFList convention) while the point entry
        defaults to 'all' (backward sources included). With the policy ALIGNED the two paths agree
        to machine precision; with the defaults they differ by the backward-source contribution."""
        from shaarp.multilayer_shg_boundary import sample_rotate_transmitted_2omega_amplitudes as proj_t

        sysm = _system()
        theta = 26.0
        sw = solve_multilayer_maker_fringes_sweep(sysm, theta_deg=[theta], mrassumption=0)
        sweep_val = float(np.abs(np.asarray(sw.parallel_intensity))[0])
        s = replace(sysm, polarimetry=replace(sysm.polarimetry, theta_deg=theta))
        aligned = solve_multilayer_shg_from_system_polarimetry(s, inhomogeneous_source_policy="forward_only")
        point_aligned = float(abs(proj_t(aligned.shg, 0.0)[0]) ** 2)
        default = solve_multilayer_shg_from_system_polarimetry(s)  # default 'all'
        point_default = float(abs(proj_t(default.shg, 0.0)[0]) ** 2)
        self.assertAlmostEqual(sweep_val, point_aligned, places=9,
                               msg="with aligned source policy the paths must agree exactly")
        self.assertGreater(abs(point_default - sweep_val) / sweep_val, 1e-3,
                           "the default divergence (backward sources) is real and pinned")

    def test_former_telegraph_window_is_smooth_at_defaults(self):
        # the exact window that telegraphed between E/n levels at SI constants: at the (natural)
        # defaults it must be a smooth physical curve -- no zeros, tiny steps.
        grid = [round(26.0 + 0.002 * i, 6) for i in range(11)]
        sw = solve_multilayer_maker_fringes_sweep(_system(), theta_deg=grid, mrassumption=0)
        i_par = np.abs(np.asarray(sw.parallel_intensity))
        self.assertEqual(int(np.sum(i_par == 0.0)), 0)
        self.assertLess(float(np.max(np.abs(np.diff(i_par)))), 0.01 * float(i_par.mean()),
                        "the former telegraph window must be smooth at natural-unit defaults")


if __name__ == "__main__":
    unittest.main()
