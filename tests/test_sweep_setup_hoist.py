"""Regression: hoisting the angle-independent `_system_setup` out of the
incidence-angle sweep loops (Maker fringes + GUI Fresnel) must be BYTE-IDENTICAL
to recomputing it at every angle.

Only `incident_theta_rad` depends on the incidence angle; every other entry of
`_system_setup` (eps tensors, eigen-bases, d-tensor, indices, thicknesses, omega)
is angle-independent and is now built ONCE per sweep, with `incident_theta_rad`
overridden per angle. The downstream tensor solvers receive every quantity --
including the angle -- as explicit kwargs and never read the system, so the
optimization changes performance only, never the numbers. These tests pin that:
the fast (hoisted) path equals the slow (per-angle recompute) path exactly.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

import shaarp.shaarp_gui as gui
from shaarp.api import _gui_fresnel_pair, _run_gui_multilayer_fresnel_sweep
from shaarp.multilayer_shg_boundary import (
    analyze_transmitted_2omega_waves,
    analyzer_jones_from_polarimetry,
    analyzer_jones_from_psi_deg,
    solve_multilayer_maker_fringes_sweep,
    solve_multilayer_shg_from_system_polarimetry,
    _transmitted_waves_for_maker_policy,
)

# A spread of angles incl. negative, zero, and non-grid values -- nothing should
# depend on the angle entering only via the hoisted incident_theta_rad override.
THETAS = [-30.0, -7.5, 0.0, 13.25, 28.0, 41.6]


def _maker_reference(system, thetas):
    """Per-angle recompute (setup=None default => full _system_setup each angle),
    extracting parallel/perpendicular transmitted-2omega intensities exactly the
    way solve_multilayer_maker_fringes_sweep does."""
    par, perp = [], []
    for t in thetas:
        pol = replace(system.polarimetry, theta_deg=float(t))
        point = replace(system, polarimetry=pol)
        res = solve_multilayer_shg_from_system_polarimetry(
            point, inhomogeneous_source_policy="forward_only"
        )
        tw = _transmitted_waves_for_maker_policy(res.shg, "shaarp_ml_selected")
        _, pi = analyze_transmitted_2omega_waves(tw, analyzer_jones_from_polarimetry(point))
        _, pe = analyze_transmitted_2omega_waves(
            tw, analyzer_jones_from_psi_deg(float(pol.psi_deg) + 90.0)
        )
        par.append(pi)
        perp.append(pe)
    return np.asarray(par, dtype=float), np.asarray(perp, dtype=float)


class SweepSetupHoistTests(unittest.TestCase):
    def test_maker_sweep_hoist_is_byte_identical(self):
        system = gui._quartz_au_docs_system()
        sweep = solve_multilayer_maker_fringes_sweep(
            system, theta_deg=THETAS, inhomogeneous_source_policy="forward_only"
        )
        ref_par, ref_perp = _maker_reference(system, THETAS)
        # Exact equality -- the hoist must not perturb a single bit.
        self.assertTrue(np.array_equal(sweep.parallel_intensity, ref_par))
        self.assertTrue(np.array_equal(sweep.perpendicular_intensity, ref_perp))
        # Guard against a vacuous all-zero pass: the parallel channel must carry signal.
        self.assertGreater(float(np.max(sweep.parallel_intensity)), 0.0)

    def test_gui_fresnel_sweep_hoist_is_byte_identical(self):
        system = gui._quartz_au_docs_system()
        thetas = np.asarray(THETAS, dtype=float)
        fast = _run_gui_multilayer_fresnel_sweep(
            system, thetas, transmitted_wave_policy="shaarp_ml_selected", include_validation_summary=False
        )
        # base_setup=None forces the per-angle recompute inside the pair (old behavior).
        rp, tp = _gui_fresnel_pair(
            system, thetas, (0.0j, 1.0 + 0.0j), component="p",
            transmitted_wave_policy="shaarp_ml_selected", base_setup=None,
        )
        rs, ts = _gui_fresnel_pair(
            system, thetas, (1.0 + 0.0j, 0.0j), component="s",
            transmitted_wave_policy="shaarp_ml_selected", base_setup=None,
        )
        self.assertTrue(np.array_equal(fast.numeric["rp"], rp))
        self.assertTrue(np.array_equal(fast.numeric["rs"], rs))
        self.assertTrue(np.array_equal(fast.numeric["tp"], tp))
        self.assertTrue(np.array_equal(fast.numeric["ts"], ts))
        # Non-vacuous: nonzero, non-degenerate output.
        self.assertGreater(float(np.max(fast.numeric["rp"])), 0.0)
        self.assertGreater(float(np.max(fast.numeric["ts"])), 0.0)


if __name__ == "__main__":
    unittest.main()
