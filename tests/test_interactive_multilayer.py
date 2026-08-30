"""Fast tests for the multilayer interactive scaffold (shaarp.interactive_multilayer).

These exercise the new step functions end-to-end over the validated public
facades, using a tiny 2-3 point grid so the (potentially slow) Maker solve
stays fast. They assert dict shape, finite numeric arrays, and non-empty
validation strings -- they do NOT re-validate the physics (that is covered by
the dedicated Mathematica-comparison suites).
"""

import unittest

import numpy as np

from shaarp.config import MultilayerSystem
from shaarp.interactive_multilayer import (
    DEFAULT_AZIMUTH_GRID_DEG,
    DEFAULT_THETA_GRID_DEG,
    build_demo_multilayer_system,
    build_demo_substrate_material,
    run_maker_demo,
    run_ml_numeric_demo,
    run_sample_rotation_demo,
    summarize,
)


def _assert_finite_1d(test: unittest.TestCase, array, expected_len: int) -> None:
    arr = np.asarray(array)
    test.assertEqual(arr.ndim, 1)
    test.assertEqual(arr.shape[0], expected_len)
    test.assertTrue(np.all(np.isfinite(arr)), f"non-finite values in array: {arr!r}")


def _assert_nonempty_status(test: unittest.TestCase, value) -> None:
    test.assertIsInstance(value, str)
    test.assertGreater(len(value), 0)


class BuildDemoSystemTests(unittest.TestCase):
    def test_demo_system_is_small_valid_and_active(self):
        system = build_demo_multilayer_system()

        self.assertIsInstance(system, MultilayerSystem)
        system.validate()  # raises if invalid
        self.assertEqual(len(system.layers), 3)
        active = [layer for layer in system.layers if layer.shg_active]
        self.assertEqual(len(active), 1)
        # The active film must carry a non-zero SHG tensor.
        self.assertGreater(np.linalg.norm(active[0].material.d_voigt()), 0.0)

    def test_demo_substrate_is_passive_nondegenerate_anisotropic(self):
        substrate = build_demo_substrate_material()

        # Passive: no SHG tensor.
        self.assertEqual(np.linalg.norm(substrate.d_voigt()), 0.0)
        # Non-degenerate: three distinct principal omega indices.
        eps_w_diag = np.diag(substrate.eps_w())
        self.assertEqual(len({round(complex(v).real, 6) for v in eps_w_diag}), 3)

    def test_theta_argument_threads_through_to_polarimetry(self):
        system = build_demo_multilayer_system(theta_deg=12.5)
        self.assertEqual(float(np.asarray(system.polarimetry.theta_deg)), 12.5)


class MakerDemoTests(unittest.TestCase):
    def test_run_maker_demo_returns_finite_arrays_and_status(self):
        system = build_demo_multilayer_system()
        grid = [-10.0, 0.0, 10.0]

        out = run_maker_demo(system, theta_deg=grid)

        for key in (
            "kind",
            "validation_status",
            "theta_deg",
            "parallel_intensity",
            "perpendicular_intensity",
            "parallel_amplitude",
            "perpendicular_amplitude",
            "result",
        ):
            self.assertIn(key, out)
        _assert_nonempty_status(self, out["validation_status"])
        np.testing.assert_allclose(out["theta_deg"], grid)
        _assert_finite_1d(self, out["parallel_intensity"], len(grid))
        _assert_finite_1d(self, out["perpendicular_intensity"], len(grid))
        # Intensities are non-negative.
        self.assertTrue(np.all(out["parallel_intensity"] >= 0.0))
        self.assertTrue(np.all(out["perpendicular_intensity"] >= 0.0))

    def test_run_maker_demo_uses_default_grid_when_unspecified(self):
        system = build_demo_multilayer_system()

        out = run_maker_demo(system)

        np.testing.assert_allclose(out["theta_deg"], np.asarray(DEFAULT_THETA_GRID_DEG))


class SampleRotationDemoTests(unittest.TestCase):
    def test_run_sample_rotation_demo_returns_finite_arrays_and_status(self):
        system = build_demo_multilayer_system()
        azimuths = [0.0, 45.0, 90.0]

        out = run_sample_rotation_demo(system, azimuths_deg=azimuths)

        for key in ("kind", "validation_status", "sample_azimuth_deg", "intensity", "analyzer_amplitude", "result"):
            self.assertIn(key, out)
        _assert_nonempty_status(self, out["validation_status"])
        np.testing.assert_allclose(out["sample_azimuth_deg"], azimuths)
        _assert_finite_1d(self, out["intensity"], len(azimuths))
        _assert_finite_1d(self, out["analyzer_amplitude"], len(azimuths))
        self.assertTrue(np.all(out["intensity"] >= 0.0))

    def test_run_sample_rotation_demo_uses_default_grid_when_unspecified(self):
        system = build_demo_multilayer_system()

        out = run_sample_rotation_demo(system)

        np.testing.assert_allclose(out["sample_azimuth_deg"], np.asarray(DEFAULT_AZIMUTH_GRID_DEG))


class MlNumericDemoTests(unittest.TestCase):
    def test_run_ml_numeric_demo_returns_finite_scalars_and_status(self):
        system = build_demo_multilayer_system()

        out = run_ml_numeric_demo(system)

        for key in (
            "kind",
            "validation_status",
            "reflected_intensity",
            "reflected_analyzer_amplitude",
            "fundamental_residual_norm",
            "shg_residual_norm",
            "result",
        ):
            self.assertIn(key, out)
        _assert_nonempty_status(self, out["validation_status"])
        self.assertTrue(np.isfinite(out["reflected_intensity"]))
        self.assertGreaterEqual(out["reflected_intensity"], 0.0)
        self.assertTrue(np.isfinite(out["reflected_analyzer_amplitude"]))
        self.assertTrue(np.isfinite(out["fundamental_residual_norm"]))
        self.assertTrue(np.isfinite(out["shg_residual_norm"]))


class SummarizeTests(unittest.TestCase):
    def test_summarize_accepts_demo_dict_and_extracts_keys(self):
        system = build_demo_multilayer_system()
        maker = run_maker_demo(system, theta_deg=[0.0, 10.0])

        summary = summarize(maker)

        for key in ("kind", "validation_status", "validation_notes", "conventions", "numeric_shapes", "numeric_keys", "stage_keys"):
            self.assertIn(key, summary)
        _assert_nonempty_status(self, summary["validation_status"])
        self.assertEqual(summary["kind"], "maker_fringes")
        self.assertIn("theta_deg", summary["numeric_keys"])
        self.assertEqual(summary["numeric_shapes"]["theta_deg"], [2])
        self.assertIsInstance(summary["validation_notes"], list)

    def test_summarize_accepts_raw_result_object(self):
        system = build_demo_multilayer_system()
        maker = run_maker_demo(system, theta_deg=[0.0, 10.0])

        summary = summarize(maker["result"])

        self.assertEqual(summary["kind"], "maker_fringes")
        _assert_nonempty_status(self, summary["validation_status"])


if __name__ == "__main__":
    unittest.main()
