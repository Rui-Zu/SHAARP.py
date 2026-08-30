import unittest
from dataclasses import replace
from math import pi

import numpy as np

from benchmarks.generate_maker_fringes_benchmarks import build_cases
from shaarp.anisotropic import solve_snell_modes
from shaarp.linear import mode_to_wave
from shaarp.multilayer_shg_boundary import _system_setup


class ComplexSnellBranchSelectionTests(unittest.TestCase):
    def test_reflected_complex_slow_mode_stays_on_reflected_branch(self):
        case = next(case for case in build_cases() if case["id"] == "maker_fringes_moderate_high_oblique")
        system = replace(case["system"], polarimetry=replace(case["system"].polarimetry, theta_deg=80.0))
        setup = _system_setup(system)
        epsilon = setup["layer_epsilon_omega_lab"][0]

        transmitted = solve_snell_modes(
            epsilon,
            incident_index=setup["top_index_omega"],
            incident_theta_rad=setup["incident_theta_rad"],
            reflected=False,
        )
        reflected = solve_snell_modes(
            epsilon,
            incident_index=setup["top_index_omega"],
            incident_theta_rad=setup["incident_theta_rad"],
            reflected=True,
        )

        reflected_slow_wave = mode_to_wave(reflected.slow, omega=setup["omega"])
        transmitted_slow_wave = mode_to_wave(transmitted.slow, omega=setup["omega"])

        self.assertGreater(float(np.real(reflected.slow.theta_rad)), pi / 2)
        self.assertLess(float(np.real(reflected_slow_wave.k[2])), 0.0)
        self.assertGreater(abs(reflected_slow_wave.k[2] - transmitted_slow_wave.k[2]), 1.0)


if __name__ == "__main__":
    unittest.main()
