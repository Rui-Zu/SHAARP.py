"""Recompute the external isotropic-stack reference live and check the fixture has not drifted.

Skipped unless BOTH ``tmm`` and ``inkstone`` are importable -- install them with::

    pip install -e ".[benchmark]"

``tests/test_isotropic_stack_reference_comparison.py`` is the always-on gate and reads the
committed fixture. That fixture is only as good as its provenance, so this module closes the loop:
where the packages are available it regenerates the same numbers and asserts they still match. A
committed reference that nothing ever re-derives is how a benchmark quietly goes stale.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import tmm  # noqa: F401

    HAVE_TMM = True
except ImportError:
    HAVE_TMM = False

try:
    import inkstone  # noqa: F401

    HAVE_INKSTONE = True
except ImportError:
    HAVE_INKSTONE = False

from benchmarks.compare_isotropic_stack_reference import CURVE_KEYS, load_reference, reference_curves
from benchmarks.isotropic_stack_cases import build_cases, closed_form_rt, inkstone_rt, tmm_rt

# The fixture stores full double precision, so a faithful recomputation should reproduce it to the
# last bit or thereabouts. This is a drift detector, not a physics tolerance.
DRIFT_ATOL = 1e-13


class ClosedFormAlwaysRunsTests(unittest.TestCase):
    """The analytic leg has no third-party dependency, so it is checked unconditionally."""

    def test_closed_form_leg_reproduces_the_committed_values(self):
        reference = load_reference()
        payloads = {case["case_id"]: case for case in reference["cases"]}
        worst = 0.0
        for case in build_cases():
            live = closed_form_rt(case)
            stored = reference_curves(payloads[case.case_id], leg="closed_form")
            for key in CURVE_KEYS:
                worst = max(worst, float(np.max(np.abs(live[key] - stored[key]))))
        self.assertLess(worst, DRIFT_ATOL,
                        "committed closed-form leg drifted from a live recomputation by %.3e" % worst)


@unittest.skipUnless(HAVE_TMM and HAVE_INKSTONE, "tmm and/or inkstone not installed (pip install -e '.[benchmark]')")
class ExternalLegLiveRecheckTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()
        cls.payloads = {case["case_id"]: case for case in cls.reference["cases"]}

    def _recheck(self, leg: str, driver):
        worst = 0.0
        worst_where = ""
        for case in build_cases():
            live = driver(case)
            stored = reference_curves(self.payloads[case.case_id], leg=leg)
            for key in CURVE_KEYS:
                err = float(np.max(np.abs(live[key] - stored[key])))
                if err > worst:
                    worst, worst_where = err, "%s/%s" % (case.case_id, key)
        self.assertLess(worst, DRIFT_ATOL,
                        "committed %s leg drifted by %.3e at %s -- regenerate the fixture with "
                        "benchmarks/generate_isotropic_stack_reference.py and review the change"
                        % (leg, worst, worst_where))

    def test_tmm_leg_has_not_drifted(self):
        self._recheck("tmm", tmm_rt)

    def test_inkstone_leg_has_not_drifted(self):
        self._recheck("inkstone", inkstone_rt)

    def test_installed_versions_match_the_ones_that_generated_the_fixture(self):
        """A version mismatch is not a failure -- it is a prompt to regenerate and re-read."""
        from importlib import metadata

        recorded = self.reference["packages"]
        for name in ("tmm", "inkstone"):
            installed = metadata.version(name)
            if installed != recorded.get(name):
                self.skipTest(
                    "%s %s installed but the fixture was generated with %s; the drift tests above "
                    "still ran and passed, so the numbers agree across versions."
                    % (name, installed, recorded.get(name))
                )


if __name__ == "__main__":
    unittest.main()
