"""R8 fence: the d-extraction NOISE-ROBUSTNESS characterization must stay true (and stay pinned).

benchmarks/dextraction_noise_benchmark.py established (seeded Monte-Carlo):
  * clean recovery through the shared _recover_d_from_design is exact (~1e-12);
  * the FIELD (phase-resolved) method degrades gracefully under multiplicative intensity
    noise (median err ~ sigma at these condition numbers);
  * the INTENSITY (Gram + rank-1, phase-less) method amplifies catastrophically on the same
    scans (median err > 30% at sigma = 2% on the baseline design) -- its output on noisy data
    is an initial guess, not a quantitative estimate.

This fence pins all three facts on the baseline design. If a future estimator improvement
makes the intensity method robust, this test FAILS ON PURPOSE -- update the benchmark and these
bounds together (the characterization is versioned evidence, not a loose doc).
"""

from __future__ import annotations

import unittest

import numpy as np

from benchmarks.dextraction_noise_benchmark import (
    DESIGNS,
    POSITIONS,
    build_design_and_clean_measurements,
    relative_error,
)
from shaarp.polarimetry_extraction import _recover_d_from_design


class DExtractionNoiseRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geoms, phis = DESIGNS["baseline 6geom x 6phi"]
        cls.M, cls.e_clean = build_design_and_clean_measurements(geoms, phis)

    def _mc(self, method: str, sigma: float, n: int = 40, seed: int = 20260704) -> float:
        rng = np.random.default_rng(seed)
        errs = []
        for _ in range(n):
            eps = rng.normal(0.0, sigma, size=self.e_clean.shape)
            if method == "intensity":
                e_noisy = self.e_clean * np.sqrt(np.maximum(1.0 + eps, 0.0))
            else:
                e_noisy = self.e_clean * (1.0 + eps / 2.0)
            errs.append(relative_error(
                _recover_d_from_design(self.M, e_noisy, POSITIONS, method).values))
        return float(np.median(np.asarray(errs)))

    def test_clean_recovery_is_exact_for_both_methods(self):
        for method in ("field", "intensity"):
            res = _recover_d_from_design(self.M, self.e_clean, POSITIONS, method)
            self.assertTrue(res.identifiable, f"{method}: benchmark scan not identifiable")
            self.assertLess(relative_error(res.values), 1e-8,
                            f"{method}: clean recovery no longer exact")

    def test_field_method_degrades_gracefully(self):
        med = self._mc("field", sigma=0.02)
        self.assertLess(med, 0.20,
                        f"field-method noise response worsened (median {med:.3f} at sigma=2%)")

    def test_intensity_method_fragility_is_documented_truth(self):
        med = self._mc("intensity", sigma=0.02)
        self.assertGreater(
            med, 0.25,
            f"intensity-method median err {med:.3f} at sigma=2% is BETTER than the documented "
            "characterization -- if the estimator improved, update the benchmark, the "
            "RESIDUAL_RISKS R8 row, and this bound together")


if __name__ == "__main__":
    unittest.main()
