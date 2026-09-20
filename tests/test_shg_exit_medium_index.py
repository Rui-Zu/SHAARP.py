"""SHG intensities carry the exit medium's index at 2omega.

A plane wave carries ``I = (1/2) c eps0 n |E|^2`` -- proportional to the index of the medium it
travels in. So a reflected SHG intensity takes the INCIDENT medium's index and a transmitted one
takes the SUBSTRATE's, both at 2omega. Every intensity in the package used to be bare ``|E|^2``,
which is right only when the exit medium is vacuum.

What is NOT here is a ``cos(theta)`` factor, and that asymmetry with the linear Fresnel curves is
the thing most likely to look like a mistake, so it is pinned by a test below. A cos belongs to
reflectance/transmittance, which are ratios between two beams of different widths sharing one patch
of interface; it does not belong to a single beam's intensity. Measured, not argued: SHAARP.py(HH)
reproduces the analytic Herman-Hayden expression with a pointwise ratio spread of 1.4e-10 across
5-60 degrees, and inserting a trial cos(theta_t) blows that up to 6.1e-01 with the residual
tracking cos(theta) at correlation +1.0000.

Every published SHAARP case exits into air, where the index factor is exactly 1 -- which is both
why the omission survived and why the published-figure fences must be untouched by the fix.
"""

from __future__ import annotations

import unittest

import numpy as np

from shaarp.multilayer_shg_boundary import (
    analyze_reflected_2omega,
    analyze_transmitted_2omega_waves,
    exit_medium_index,
)
from shaarp.waves import Wave


def _wave(index, omega=2.0, sign=1.0, amplitude=1.0):
    """An isotropic plane wave of effective index ``index`` travelling along +-z."""
    return Wave(
        k=np.array([0.0, 0.0, sign * complex(index) * omega], dtype=complex),
        electric=np.array([amplitude, 0.0, 0.0], dtype=complex),
        omega=omega,
    )


class ExitMediumIndexTests(unittest.TestCase):

    def test_vacuum_is_exactly_one(self):
        """The whole published corpus exits into air; this must be 1.0 to the bit."""
        self.assertEqual(exit_medium_index([_wave(1.0)]), 1.0)

    def test_reads_the_index_back_from_the_wavevector(self):
        for index in (1.0, 1.4533, 1.52, 2.35, 3.67):
            self.assertAlmostEqual(exit_medium_index([_wave(index)]), index, places=12)

    def test_backward_wave_gives_a_positive_index(self):
        """Reflected waves travel along -z; an index is a magnitude, never negative."""
        self.assertAlmostEqual(exit_medium_index([_wave(1.5, sign=-1.0)]), 1.5, places=12)

    def test_absorbing_medium_uses_the_real_part(self):
        self.assertAlmostEqual(exit_medium_index([_wave(2.0 + 1.0j)]), 2.0, places=12)

    def test_no_field_leaves_intensity_alone(self):
        self.assertEqual(exit_medium_index([]), 1.0)

    def test_dominant_wave_wins_when_several_modes_are_present(self):
        waves = [_wave(1.0, amplitude=1e-12), _wave(2.35, amplitude=1.0)]
        self.assertAlmostEqual(exit_medium_index(waves), 2.35, places=12)

    def test_index_is_independent_of_field_amplitude(self):
        """It is a property of the medium, so scaling E must not move it."""
        self.assertAlmostEqual(exit_medium_index([_wave(1.52, amplitude=1.0)]),
                               exit_medium_index([_wave(1.52, amplitude=137.0)]), places=12)

    def test_index_is_independent_of_the_omega_scale(self):
        self.assertAlmostEqual(exit_medium_index([_wave(1.52, omega=2.0)]),
                               exit_medium_index([_wave(1.52, omega=9.7)]), places=12)

    def test_no_cos_theta_factor_is_folded_in(self):
        """An OBLIQUE wave of the same index must return the same value.

        This is the asymmetry with the linear Fresnel path stated explicitly. If someone later
        "fixes" the inconsistency by adding cos(theta) here, this test fails -- and the Herman-Hayden
        fences fail with it, which is the physics reason.
        """
        omega, index = 2.0, 1.52
        for theta_deg in (0.0, 20.0, 45.0, 70.0):
            theta = np.deg2rad(theta_deg)
            oblique = Wave(
                k=np.array([index * omega * np.sin(theta), 0.0, index * omega * np.cos(theta)], dtype=complex),
                electric=np.array([0.0, 1.0, 0.0], dtype=complex),
                omega=omega,
            )
            self.assertAlmostEqual(exit_medium_index([oblique]), index, places=12,
                                   msg="exit_medium_index picked up a theta dependence at %.0f deg" % theta_deg)


class AnalyzerAppliesTheIndexTests(unittest.TestCase):
    """The analyzers must actually apply the factor, not merely have it available."""

    def test_transmitted_intensity_scales_with_the_exit_index(self):
        analyzer = (0.0 + 0.0j, 1.0 + 0.0j)  # pure p
        amp_air, i_air = analyze_transmitted_2omega_waves([_wave(1.0)], analyzer)
        amp_glass, i_glass = analyze_transmitted_2omega_waves([_wave(1.52)], analyzer)

        # Same field amplitude in both, so the ONLY difference is the medium.
        self.assertAlmostEqual(abs(amp_air), abs(amp_glass), places=12)
        self.assertAlmostEqual(i_air, float(abs(amp_air) ** 2), places=12)
        self.assertAlmostEqual(i_glass / i_air, 1.52, places=10)

    def test_air_exit_intensity_is_bit_identical_to_bare_amplitude_squared(self):
        """Why no published fence moves: n_exit = 1 makes the fix a no-op for every paper case."""
        analyzer = (1.0 + 0.0j, 0.0 + 0.0j)
        amplitude, intensity = analyze_transmitted_2omega_waves([_wave(1.0, amplitude=0.37)], analyzer)
        self.assertEqual(intensity, float(abs(amplitude) ** 2))

    def test_zero_field_stays_zero(self):
        analyzer = (1.0 + 0.0j, 0.0 + 0.0j)
        _, intensity = analyze_transmitted_2omega_waves([_wave(2.35, amplitude=0.0)], analyzer)
        self.assertEqual(intensity, 0.0)


class ReflectedAnalyzerAppliesTheIndexTests(unittest.TestCase):

    def test_reflected_intensity_uses_the_incident_medium_index(self):
        class _Stub:
            """Minimal stand-in exposing only what analyze_reflected_2omega reads."""

            def __init__(self, waves):
                self.reflected_2omega = waves

        analyzer = (1.0 + 0.0j, 0.0 + 0.0j)

        import shaarp.multilayer_shg_boundary as mlb

        original = mlb.reflected_2omega_jones_sp
        try:
            # Hold the projected amplitude fixed so the ONLY variable is the medium index.
            mlb.reflected_2omega_jones_sp = lambda shg: (1.0 + 0.0j, 0.0 + 0.0j)
            _, i_air = analyze_reflected_2omega(_Stub([_wave(1.0, sign=-1.0)]), analyzer)
            _, i_water = analyze_reflected_2omega(_Stub([_wave(1.33, sign=-1.0)]), analyzer)
        finally:
            mlb.reflected_2omega_jones_sp = original

        self.assertAlmostEqual(i_air, 1.0, places=12)
        self.assertAlmostEqual(i_water / i_air, 1.33, places=10)


class WhereTheDominantModeChoiceIsExact(unittest.TestCase):
    """Pin the boundary of the one approximation inside `exit_medium_index`.

    The helper takes the index of the transmitted mode carrying the most field and applies it to the
    COHERENTLY SUMMED 2omega field. That is EXACT, not a tie-break, because both semi-infinite media
    are isotropic by rule (`shaarp/layer_stack.py::_require_isotropic_halfspace`) and the two
    transmitted 2omega eigenmodes are therefore degenerate — there is no choice to get wrong.

    This test is what makes that reasoning checkable rather than asserted. It also guards the
    reverse direction: were a birefringent exit medium ever allowed back in, the choice WOULD start
    to matter — on the ml/ext stacks the two modes differ by 5-11% in index with both carrying
    comparable field (register row R18, closed by the isotropy rule rather than by picking a
    weighting).
    """

    def test_isotropic_exit_medium_makes_the_two_modes_degenerate(self):
        import numpy as np

        from shaarp.config import Layer, MultilayerSystem
        from shaarp.layer_stack import _material_from_iso
        from shaarp.multilayer_shg_boundary import solve_multilayer_shg_from_system
        from shaarp.presets import gaas_111_800

        n_sub = 1.6
        system = MultilayerSystem(
            wavelength_um=0.8,
            layers=[
                Layer(name="air", material=_material_from_iso(1.0, 1.0, "air"),
                      thickness_um=None, shg_active=False),
                Layer(name="film", material=gaas_111_800(), thickness_um=0.2, shg_active=True),
                Layer(name="substrate", material=_material_from_iso(n_sub, n_sub, "substrate"),
                      thickness_um=None, shg_active=False),
            ],
        )
        result = solve_multilayer_shg_from_system(
            system, incident_polarization="p", mu=1.0, eps0=1.0)
        waves = list(result.shg.substrate_2omega)
        self.assertGreaterEqual(len(waves), 2,
                                "need both transmitted eigenmodes for this to mean anything")

        def mode_index(wave):
            k = np.asarray(wave.k, dtype=complex)
            return abs(float(np.real(np.sqrt(complex(np.dot(k, k))) / float(np.real(wave.omega)))))

        indices = [mode_index(w) for w in waves]
        self.assertLess(abs(indices[0] - indices[1]), 1e-10,
                        f"isotropic substrate should give degenerate 2omega modes, got {indices}")
        # ...and the helper returns that shared index, not something else entirely.
        self.assertAlmostEqual(exit_medium_index(waves), n_sub, places=10)


if __name__ == "__main__":
    unittest.main()
