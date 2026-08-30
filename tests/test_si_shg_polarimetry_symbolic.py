"""FULL ANALYTICAL SHG POLARIMETRY (SHAARP.si): the reflected SHG as a closed form in the
INPUT POLARIZATION angle phi with the d_ijk tensor as a symbolic coefficient -- i.e. the
expression an experimentalist fits to a polarimetry scan to EXTRACT d. This is the project's
key analytical capability ("you are missing analytical shg coefficient d which is the key").

``solve_si_shg_full_analytical_symbolic`` is symbolic in EVERYTHING SHAARP.si supports:
input polarization phi, analyzer psi, sample azimuth, incidence theta, refractive indices,
and d_ijk. Validated here without any vacuous (0==0) shortcut:

  1. s/p ENDPOINTS (phi = 90 / 0 deg) reduce EXACTLY to the already-validated numeric
     single-interface SHG ``solve_single_interface_shg`` (~1e-12), nonzero-guarded -- this
     pins the absolute normalization against the trusted numeric path.
  2. The FULL phi-curve equals the PUBLISHED GaAs(111) full-analytical closed form,
     composed independently from the paper's eq 29-32 (fundamental fields) -> eq 20-28
     (P_NL) -> eq 11-13 (C_Li) -> eq 9-10 (signal), over a phi sweep, in BOTH channels,
     to the published 5-decimal rounding floor -- this validates the WHOLE curve including
     the eo cross term and the symbolic d14 coefficient against an EXTERNAL reference.
     (npj Comput. Mater. 8, 246 (2022), Supplementary Note 5, GaAs(111).)
  3. 3-FOLD sample-azimuth symmetry: GaAs is -43m, so the (111) surface is C3v and the
     polarimetry must be EXACTLY 120-deg periodic in sample azimuth (and NOT 60-deg) --
     a paper-independent physics signature that exercises the cross term + azimuth handling.
  4. Genuinely SYMBOLIC in phi, psi, theta, refractive index, and d14, with a nonzero,
     phi-varying analyzed intensity I(phi,psi)=|sin psi E_s + cos psi E_p|^2.

Convention note (documented, load-bearing): the extraordinary driving field is the physically
TRANSVERSE p-field (cos thetaT, 0, -sin thetaT); the published eq 29-31 prints +sin thetaT
with the L3 axis oriented into the crystal. Both compose to the same |E|^2 (the measured
polarimetry), and the ratio Python/published is 1.000 to the rounding floor (no scale fudge).
"""

import math
import unittest

import numpy as np
import sympy as sp

from shaarp.config import CrystalOrientation
from shaarp.shg import solve_single_interface_shg
from shaarp.symbolic import solve_si_shg_full_analytical_symbolic
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

IN_PLANE_HINT = np.array([1.0, -1.0, 0.0])  # published GaAs(111) lab frame: L1 along [1,-1,0]


def _gaas111_geometric_d():
    """Real geometric GaAs(111) lab-frame Voigt d (d14=d25=d36=1 rotated to (111))."""
    d = np.zeros((3, 6))
    d[0, 3] = d[1, 4] = d[2, 5] = 1.0
    ori = CrystalOrientation.from_cubic_miller_surface(np.array([1.0, 1.0, 1.0]), in_plane_hint=IN_PLANE_HINT)
    R = np.asarray(ori.rotation_matrix(), dtype=float)
    return np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d, R), dtype=complex))


def _rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _published_gaas111_intensities(nw, n2, thi, phi):
    """The published GaAs(111) full-analytical reflected SHG |E_s|^2, |E_p|^2 at input
    polarization angle phi, composed ONLY from the paper's closed-form equations (an
    external reference, independent of the Python solver under test). d14 = 1.

    eq 29-32 fundamental fields (SHAARP input convention s = sin phi, p = cos phi; the
    extraordinary field is the transverse (cos thT, 0, -sin thT)) -> eq 20-28 P_NL ->
    eq 11-13 isotropic C_Li -> eq 9-10 reflected signal.
    """
    thTw = math.asin(math.sin(thi) / nw)
    thT2 = math.asin(math.sin(thi) / n2)
    cTw, sTw, cT2, ci = math.cos(thTw), math.sin(thTw), math.cos(thT2), math.cos(thi)
    tp = 2 * ci / (nw * ci + cTw)   # Fresnel t_p (eq 29-31)
    ts = 2 * ci / (ci + nw * cTw)   # Fresnel t_s (eq 32)
    cphi, sphi = math.cos(phi), math.sin(phi)
    # driving fields: extraordinary from p (cos phi), ordinary from s (sin phi)
    e1, e2, e3 = cphi * tp * cTw, 0.0, -cphi * tp * sTw   # transverse p-field
    o1, o2, o3 = 0.0, sphi * ts, 0.0
    # eq 20-28 P_NL (ee + oo + eo), d14 = 1 geometric coefficients
    Pee = [1.63299 * e1 * e2 - 1.15470 * e1 * e3,
           0.81650 * e1**2 - 0.81650 * e2**2 - 1.15470 * e2 * e3,
           -0.57735 * e1**2 - 0.57735 * e2**2 + 1.15470 * e3**2]
    Poo = [1.63299 * o1 * o2 - 1.15470 * o1 * o3,
           0.81650 * o1**2 - 0.81650 * o2**2 - 1.15470 * o2 * o3,
           -0.57735 * o1**2 - 0.57735 * o2**2 + 1.15470 * o3**2]
    Peo = [1.63299 * (o1 * e2 + e1 * o2) - 1.15470 * (o1 * e3 + e1 * o3),
           1.63299 * e1 * o1 - 1.63299 * e2 * o2 - 1.15470 * (o2 * e3 + e2 * o3),
           -1.15470 * e1 * o1 - 1.15470 * e2 * o2 + 2.30940 * e3 * o3]
    P = [Pee[i] + Poo[i] + Peo[i] for i in range(3)]
    # eq 11-13 isotropic 2omega inhomogeneous map (thetaT = thTw)
    den = n2**2 * (n2**2 - nw**2)
    C1 = (-n2**2 * P[0] + nw**2 * P[2] * cTw * sTw + nw**2 * P[0] * sTw**2) / den
    C2 = -P[1] / (n2**2 - nw**2)
    C3 = (-n2**2 * P[2] + nw**2 * P[2] * cTw**2 + nw**2 * P[0] * cTw * sTw) / den
    # eq 9-10 reflected SHG signal
    Es = C2 * (n2 * cT2 - nw * cTw) / (ci + n2 * cT2)
    Ep = -(C1 * (n2 - nw * cT2 * cTw) + C3 * (nw * cT2 * sTw)) / (n2 * ci + cT2)
    return abs(Es) ** 2, abs(Ep) ** 2


class SIEndpointsVsNumericTests(unittest.TestCase):
    """The phi = 90/0 endpoints reduce to the trusted numeric single-interface SHG."""

    def test_endpoints_match_numeric_single_interface(self):
        geom = _gaas111_geometric_d()
        phi = sp.Symbol("phi", real=True)
        nw, n2, thi = 3.4, 3.7, 0.5
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=nw**2, eps_y_omega=nw**2, eps_z_omega=nw**2,
            eps_x_2omega=n2**2, eps_y_2omega=n2**2, eps_z_2omega=n2**2,
            d_voigt_lab=sp.Matrix(geom.tolist()), incident_theta_rad=thi,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )

        def numeric(pol):
            r = solve_single_interface_shg(
                np.diag([nw**2] * 3).astype(complex), np.diag([n2**2] * 3).astype(complex),
                geom.astype(complex), incident_index_omega=1.0, incident_index_2omega=1.0,
                incident_theta_rad=thi, incident_polarization=pol, omega=1.0, mu=1.0, eps0=1.0,
            )
            return complex(np.asarray(r.coefficients)[0]), complex(np.asarray(r.coefficients)[1])

        ns_s, np_s = numeric("s")
        ns_p, np_p = numeric("p")
        # phi = pi/2 -> pure s; phi = 0 -> pure p
        es_s = complex(sol.reflected_s.subs(phi, sp.pi / 2).evalf())
        ep_s = complex(sol.reflected_p.subs(phi, sp.pi / 2).evalf())
        es_p = complex(sol.reflected_s.subs(phi, 0).evalf())
        ep_p = complex(sol.reflected_p.subs(phi, 0).evalf())
        # nonzero guards (avoid vacuous 0==0)
        self.assertGreater(abs(ns_s), 1e-5, "s-incidence E_s ~ 0 -> vacuous")
        self.assertGreater(abs(np_p), 1e-5, "p-incidence E_p ~ 0 -> vacuous")
        self.assertLess(abs(es_s - ns_s), 1e-12, "s-endpoint E_s != numeric")
        self.assertLess(abs(ep_s - np_s), 1e-12, "s-endpoint E_p != numeric")
        self.assertLess(abs(es_p - ns_p), 1e-12, "p-endpoint E_s != numeric")
        self.assertLess(abs(ep_p - np_p), 1e-12, "p-endpoint E_p != numeric")


class SIFullCurveVsPublishedGaAs111Tests(unittest.TestCase):
    """The full phi-curve equals the PUBLISHED GaAs(111) closed form (eq 9-32 composed)."""

    def test_phi_curve_matches_published_both_channels(self):
        geom = _gaas111_geometric_d()
        d14 = sp.Symbol("d14")
        phi = sp.Symbol("phi", real=True)
        nw, n2, thi = 3.4, 3.7, 0.5
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=nw**2, eps_y_omega=nw**2, eps_z_omega=nw**2,
            eps_x_2omega=n2**2, eps_y_2omega=n2**2, eps_z_2omega=n2**2,
            d_voigt_lab=sp.Matrix(geom.tolist()) * d14, incident_theta_rad=thi,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )
        # the result must be a genuine function of phi and d14
        self.assertIn(phi, sol.reflected_p.free_symbols)
        self.assertIn(d14, sol.reflected_p.free_symbols)

        py_s, py_p, ratios_s, ratios_p = [], [], [], []
        for v in np.linspace(0.05, math.pi - 0.05, 19):
            ref_s, ref_p = _published_gaas111_intensities(nw, n2, thi, v)
            qs = abs(complex(sol.reflected_s.subs({phi: v, d14: 1.0}).evalf())) ** 2
            qp = abs(complex(sol.reflected_p.subs({phi: v, d14: 1.0}).evalf())) ** 2
            py_s.append(qs)
            py_p.append(qp)
            if ref_p > 1e-18:
                ratios_p.append(qp / ref_p)
            if ref_s > 1e-18:
                ratios_s.append(qs / ref_s)
        # genuine, nonzero, phi-varying polarimetry curves
        self.assertGreater(max(py_p), 1e-8, "p-channel ~ 0 -> vacuous")
        self.assertGreater(max(py_s), 1e-8, "s-channel ~ 0 -> vacuous")
        self.assertGreater(max(py_p) - min(py_p), 0.3 * max(py_p), "p-curve barely varies with phi")
        self.assertGreater(max(py_s) - min(py_s), 0.3 * max(py_s), "s-curve barely varies with phi")
        # equals published to the 5-decimal rounding floor, with ratio == 1 (no scale fudge)
        for ratios, name in ((ratios_p, "p"), (ratios_s, "s")):
            mean = sum(ratios) / len(ratios)
            spread = max(ratios) - min(ratios)
            self.assertAlmostEqual(mean, 1.0, places=3, msg=f"{name}-channel scale != published")
            self.assertLess(spread / mean, 1e-3, f"{name}-channel SHAPE deviates from published")

    def test_complex_index_endpoints(self):
        """SHAARP targets absorbing media: validate the curve endpoints with a COMPLEX
        2omega index against the numeric path (the building-block tests cover the full
        complex chain; this guards the polarimetry entry in the complex domain)."""
        geom = _gaas111_geometric_d()
        phi = sp.Symbol("phi", real=True)
        nw, thi = 3.4, 0.5
        n2 = complex(3.7, 0.25)
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=nw**2, eps_y_omega=nw**2, eps_z_omega=nw**2,
            eps_x_2omega=n2**2, eps_y_2omega=n2**2, eps_z_2omega=n2**2,
            d_voigt_lab=sp.Matrix(geom.tolist()), incident_theta_rad=thi,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )
        r = solve_single_interface_shg(
            np.diag([nw**2] * 3).astype(complex), np.diag([n2**2] * 3).astype(complex),
            geom.astype(complex), incident_index_omega=1.0, incident_index_2omega=1.0,
            incident_theta_rad=thi, incident_polarization="p", omega=1.0, mu=1.0, eps0=1.0,
        )
        ep_p = complex(sol.reflected_p.subs(phi, 0).evalf())
        self.assertGreater(abs(complex(np.asarray(r.coefficients)[1])), 1e-5, "vacuous")
        self.assertLess(abs(ep_p - complex(np.asarray(r.coefficients)[1])), 1e-11)


class SIThreefoldAzimuthTests(unittest.TestCase):
    """GaAs(111) is C3v -> the polarimetry must be EXACTLY 120-deg periodic in sample
    azimuth (and not 60-deg). Paper-independent physics that exercises the full curve."""

    def setUp(self):
        self.geom0 = _gaas111_geometric_d()
        self.nw, self.n2, self.thi = 3.4, 3.7, 0.5
        self.phi0 = math.radians(40.0)  # fixed input polarization

    def _power_p(self, az):
        geom = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(self.geom0, _rz(az)), dtype=complex))
        phi = sp.Symbol("phi", real=True)
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=self.nw**2, eps_y_omega=self.nw**2, eps_z_omega=self.nw**2,
            eps_x_2omega=self.n2**2, eps_y_2omega=self.n2**2, eps_z_2omega=self.n2**2,
            d_voigt_lab=sp.Matrix(geom.tolist()), incident_theta_rad=self.thi,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )
        return abs(complex(sol.reflected_p.subs(phi, self.phi0).evalf())) ** 2

    def test_polarimetry_is_120deg_periodic_not_60(self):
        base = [0.0, math.radians(35.0), math.radians(80.0)]
        worst120 = max(abs(self._power_p(a) - self._power_p(a + math.radians(120.0))) for a in base)
        scale = max(self._power_p(a) for a in base)
        self.assertGreater(scale, 1e-8, "vacuous (zero SHG)")
        self.assertLess(worst120 / scale, 1e-6, "not 120-deg periodic about [111]")
        diff60 = max(abs(self._power_p(a) - self._power_p(a + math.radians(60.0))) for a in base)
        self.assertGreater(diff60 / scale, 0.05, "60-deg unexpectedly a symmetry (should be 3-fold)")

    def test_sample_azimuth_symbol_is_symbolic_and_consistent(self):
        """sample_azimuth_symbol threads a SYMBOLIC azimuth through the chain; substituting
        a value reproduces the explicit numeric-rotation path."""
        az = sp.Symbol("az", real=True)
        phi = sp.Symbol("phi", real=True)
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=self.nw**2, eps_y_omega=self.nw**2, eps_z_omega=self.nw**2,
            eps_x_2omega=self.n2**2, eps_y_2omega=self.n2**2, eps_z_2omega=self.n2**2,
            d_voigt_lab=sp.Matrix(self.geom0.tolist()), incident_theta_rad=self.thi,
            phi_symbol=phi, sample_azimuth_symbol=az, omega=1, mu=1, eps0=1, simplify=False,
        )
        self.assertIn(az, sol.reflected_p.free_symbols)
        azv = math.radians(25.0)
        sym_val = abs(complex(sol.reflected_p.subs({az: azv, phi: self.phi0}).evalf())) ** 2
        # explicit numeric-rotation reference
        geom = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(self.geom0, _rz(azv)), dtype=complex))
        sol2 = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=self.nw**2, eps_y_omega=self.nw**2, eps_z_omega=self.nw**2,
            eps_x_2omega=self.n2**2, eps_y_2omega=self.n2**2, eps_z_2omega=self.n2**2,
            d_voigt_lab=sp.Matrix(geom.tolist()), incident_theta_rad=self.thi,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )
        ref_val = abs(complex(sol2.reflected_p.subs(phi, self.phi0).evalf())) ** 2
        self.assertGreater(ref_val, 1e-8, "vacuous")
        self.assertLess(abs(sym_val - ref_val) / ref_val, 1e-9, "symbolic azimuth != numeric rotation")


class SISymbolicEverythingTests(unittest.TestCase):
    """The full analytical form is genuinely symbolic in phi, psi, theta, index, and d14,
    with a nonzero, phi-varying analyzed intensity I(phi,psi)."""

    def test_intensity_is_symbolic_in_phi_psi_and_varies(self):
        geom = _gaas111_geometric_d()
        phi, psi, d14 = sp.symbols("phi psi d14", real=True)
        nw, n2, thi = 3.4, 3.7, 0.5
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=nw**2, eps_y_omega=nw**2, eps_z_omega=nw**2,
            eps_x_2omega=n2**2, eps_y_2omega=n2**2, eps_z_2omega=n2**2,
            d_voigt_lab=sp.Matrix(geom.tolist()) * d14, incident_theta_rad=thi,
            phi_symbol=phi, analyzer_symbol=psi, omega=1, mu=1, eps0=1, simplify=False,
        )
        self.assertIn(phi, sol.intensity.free_symbols)
        self.assertIn(psi, sol.intensity.free_symbols)
        # analyzer at psi = 0 picks E_p, psi = 90 picks E_s (sanity of I = |sin psi E_s + cos psi E_p|^2)
        ip0 = sp.simplify(sol.intensity.subs(psi, 0) - sol.intensity_p)
        self.assertEqual(ip0, 0, "I(psi=0) != |E_p|^2")
        # nonzero + varies over a (phi, psi) grid
        vals = []
        for pv in np.linspace(0.1, math.pi - 0.1, 7):
            for sv in (0.0, math.pi / 4, math.pi / 2):
                vals.append(float(sol.intensity.subs({phi: pv, psi: sv, d14: 1.0}).evalf()))
        self.assertGreater(max(vals), 1e-8, "intensity ~ 0 -> vacuous")
        self.assertGreater(max(vals) - min(vals), 0.3 * max(vals), "intensity does not vary")

    def test_symbolic_refractive_index_runs(self):
        """Full analytical means the refractive indices may be SYMBOLIC too (the eigenmode
        selection is symbolic-safe). Build with a symbolic index and confirm it appears."""
        geom = _gaas111_geometric_d()
        n_w, n_2w, phi, d14 = sp.symbols("n_w n_2w phi d14", positive=True)
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=n_w**2, eps_y_omega=n_w**2, eps_z_omega=n_w**2,
            eps_x_2omega=n_2w**2, eps_y_2omega=n_2w**2, eps_z_2omega=n_2w**2,
            d_voigt_lab=sp.Matrix(geom.tolist()) * d14, incident_theta_rad=0.5,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )
        syms = {str(s) for s in sol.reflected_p.free_symbols}
        self.assertTrue({"n_w", "n_2w"} & syms, "output not symbolic in refractive index")
        self.assertIn("phi", syms)
        self.assertIn("d14", syms)
        # subbing reduces to the validated numeric chain (nonzero)
        val = abs(complex(sol.reflected_p.subs({n_w: 3.4, n_2w: 3.7, phi: 0.7, d14: 1.0}).evalf()))
        self.assertGreater(val, 1e-6, "symbolic-index curve vanished -> vacuous")


class SIAnisotropicPolarimetryVsNumericJonesTests(unittest.TestCase):
    """The ANISOTROPIC (uniaxial + biaxial) full-curve polarimetry -- including the eo CROSS
    term, not just the s/p endpoints -- equals the INDEPENDENT numeric arbitrary-Jones SI
    solver (``solve_single_interface_shg(incident_jones=...)``) over a phi sweep. The
    published-GaAs test above covers the isotropic class; this closes the cross-term
    validation for genuinely birefringent crystals (where the published closed form does
    not apply). The numeric Jones path is itself first shown to reduce EXACTLY to the
    discrete s/p path at the endpoints (so it is a faithful reference)."""

    THETA = 0.5

    @staticmethod
    def _rich_d():
        # nonzero ordinary-driven (Voigt col 1 = yy, reached by s), extraordinary-driven
        # (xz, reached by p), and mixing components -> nonzero s, p, AND cross.
        d = np.zeros((3, 6))
        d[0, 0], d[1, 1], d[2, 2] = 0.3, 0.5, 0.6
        d[0, 3], d[1, 4], d[2, 0], d[0, 4] = 0.8, 0.4, 0.2, 0.25
        return d

    def _numeric(self, epsw, eps2, d, jones=None, pol=None):
        kw = dict(incident_index_omega=1.0, incident_index_2omega=1.0,
                  incident_theta_rad=self.THETA, omega=1.0, mu=1.0, eps0=1.0)
        if jones is not None:
            kw["incident_jones"] = jones
        else:
            kw["incident_polarization"] = pol
        r = solve_single_interface_shg(np.diag(epsw).astype(complex), np.diag(eps2).astype(complex), d, **kw)
        return complex(np.asarray(r.coefficients)[0]), complex(np.asarray(r.coefficients)[1])

    def test_numeric_jones_reduces_to_sp_at_endpoints(self):
        """The new numeric incident_jones must reduce EXACTLY to the discrete s/p path:
        (J_s,J_p)=(0,1) == 'p' and (1,0) == 's'. Otherwise it is not a valid reference."""
        d = self._rich_d()
        for epsw, eps2 in (([2.2**2, 2.2**2, 2.5**2], [2.4**2, 2.4**2, 2.7**2]),
                           ([2.1**2, 2.3**2, 2.5**2], [2.35**2, 2.5**2, 2.7**2])):
            sp_p = self._numeric(epsw, eps2, d, pol="p")
            j_p = self._numeric(epsw, eps2, d, jones=(0.0, 1.0))
            sp_s = self._numeric(epsw, eps2, d, pol="s")
            j_s = self._numeric(epsw, eps2, d, jones=(1.0, 0.0))
            self.assertGreater(max(abs(sp_p[1]), abs(sp_s[0])), 1e-4, "vacuous endpoints")
            self.assertLess(max(abs(sp_p[0] - j_p[0]), abs(sp_p[1] - j_p[1])), 1e-13, "jones(0,1) != 'p'")
            self.assertLess(max(abs(sp_s[0] - j_s[0]), abs(sp_s[1] - j_s[1])), 1e-13, "jones(1,0) != 's'")

    def _check_class(self, epsw, eps2):
        d = self._rich_d()
        phi = sp.Symbol("phi", real=True)
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=epsw[0], eps_y_omega=epsw[1], eps_z_omega=epsw[2],
            eps_x_2omega=eps2[0], eps_y_2omega=eps2[1], eps_z_2omega=eps2[2],
            d_voigt_lab=sp.Matrix(d.tolist()), incident_theta_rad=self.THETA,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )
        worst, max_out, curve = 0.0, 0.0, []
        for v in np.linspace(0.05, math.pi - 0.05, 13):
            ns, npp = self._numeric(epsw, eps2, d, jones=(math.sin(v), math.cos(v)))
            ss = complex(sol.reflected_s.subs(phi, v).evalf())
            spp = complex(sol.reflected_p.subs(phi, v).evalf())
            worst = max(worst, abs(ss - ns), abs(spp - npp))
            max_out = max(max_out, abs(ns), abs(npp))
            curve.append(abs(spp) ** 2)
        self.assertGreater(max_out, 1e-4, "near-zero output -> vacuous 0==0 check")
        self.assertGreater(max(curve) - min(curve), 0.3 * max(curve), "polarimetry barely varies with phi")
        self.assertLess(worst, 1e-12, f"anisotropic phi-curve deviates from numeric Jones: {worst:.2e}")

    def test_uniaxial_phi_curve_matches_numeric_jones(self):
        self._check_class([2.2**2, 2.2**2, 2.5**2], [2.4**2, 2.4**2, 2.7**2])

    def test_biaxial_phi_curve_matches_numeric_jones(self):
        self._check_class([2.1**2, 2.3**2, 2.5**2], [2.35**2, 2.5**2, 2.7**2])

    def test_complex_absorbing_biaxial_phi_curve_matches_numeric_jones(self):
        """SHAARP targets absorbing media: the anisotropic polarimetry curve must also agree
        in the complex domain (absorbing indices)."""
        epsw = [complex(2.1, 0.05) ** 2, complex(2.3, 0.06) ** 2, complex(2.5, 0.04) ** 2]
        eps2 = [complex(2.35, 0.07) ** 2, complex(2.5, 0.05) ** 2, complex(2.7, 0.06) ** 2]
        self._check_class(epsw, eps2)


class SITransmittedPolarimetryTests(unittest.TestCase):
    """The TRANSMITTED SHG channel (the Maker-fringe observable) as a polarimetry curve:
    the total transmitted SHG field E_t(phi) = sum of the two homogeneous transmitted 2 omega
    eigenmodes, validated vs the numeric total transmitted field over a phi sweep. (Everything
    else in this file is the reflected channel; this adds the transmitted channel.)"""

    THETA = 0.5

    @staticmethod
    def _rich_d():
        d = np.zeros((3, 6))
        d[0, 0], d[1, 1], d[2, 2] = 0.3, 0.5, 0.6
        d[0, 3], d[1, 4], d[2, 0], d[0, 4] = 0.8, 0.4, 0.2, 0.25
        return d

    def _numeric_transmitted(self, epsw, eps2, d, phi_val):
        r = solve_single_interface_shg(
            np.diag(epsw).astype(complex), np.diag(eps2).astype(complex), d,
            incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=self.THETA,
            incident_jones=(math.sin(phi_val), math.cos(phi_val)), omega=1.0, mu=1.0, eps0=1.0,
        )
        return np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)

    def _check(self, epsw, eps2):
        d = self._rich_d()
        phi = sp.Symbol("phi", real=True)
        sol = solve_si_shg_full_analytical_symbolic(
            eps_x_omega=epsw[0], eps_y_omega=epsw[1], eps_z_omega=epsw[2],
            eps_x_2omega=eps2[0], eps_y_2omega=eps2[1], eps_z_2omega=eps2[2],
            d_voigt_lab=sp.Matrix(d.tolist()), incident_theta_rad=self.THETA,
            phi_symbol=phi, omega=1, mu=1, eps0=1, simplify=False,
        )
        e_t = sol.transmitted_field  # symbolic total transmitted field, function of phi
        self.assertIn(phi, e_t.free_symbols, "transmitted field is not a function of phi")
        worst, max_out, curve = 0.0, 0.0, []
        for v in np.linspace(0.05, math.pi - 0.05, 13):
            ref = self._numeric_transmitted(epsw, eps2, d, v)
            got = np.array([complex(c.subs(phi, v).evalf()) for c in e_t])
            worst = max(worst, float(np.max(np.abs(got - ref))))
            max_out = max(max_out, float(np.linalg.norm(ref)))
            curve.append(float(np.linalg.norm(got)) ** 2)
        self.assertGreater(max_out, 1e-3, "near-zero transmitted field -> vacuous")
        self.assertGreater(max(curve) - min(curve), 0.3 * max(curve), "transmitted curve barely varies with phi")
        self.assertLess(worst, 1e-12, f"transmitted polarimetry deviates from numeric: {worst:.2e}")

    def test_uniaxial_transmitted_polarimetry_matches_numeric(self):
        self._check([2.2**2, 2.2**2, 2.5**2], [2.4**2, 2.4**2, 2.7**2])

    def test_biaxial_transmitted_polarimetry_matches_numeric(self):
        self._check([2.1**2, 2.3**2, 2.5**2], [2.35**2, 2.5**2, 2.7**2])


if __name__ == "__main__":
    unittest.main()
