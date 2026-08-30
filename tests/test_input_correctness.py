"""Gate the polarization / complex-eps / d_ijk correctness checks.

What must be verified -- not just that the GUI doesn't crash, but that the physics USES the
three core inputs correctly: input POLARIZATION (Jones phi/analyzer), the COMPLEX permittivity tensor
(symmetry + imaginary part + off-diagonals), and the SHG d_ijk tensor (point-group patterns, intrinsic
jk-symmetry, P_NL contraction, complex preservation), plus the end-to-end d-extraction round-trip.

The checks live in ``scripts/verify_inputs_correctness.py`` (9 checks, each prints PASS/FAIL and the
script exits non-zero if any fail). This test runs it as a subprocess so the whole battery is part of
the suite / release gate and can't silently regress.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_inputs_correctness.py"


class InputCorrectnessGate(unittest.TestCase):
    def test_polarization_complex_eps_and_d_tensor_are_used_correctly(self):
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8",
               "QT_QPA_PLATFORM": "offscreen"}
        r = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT),
                           capture_output=True, text=True, env=env, timeout=600)
        self.assertEqual(r.returncode, 0,
                         msg="input-correctness checks failed:\n" + (r.stdout or "")[-3000:]
                             + "\n--- stderr ---\n" + (r.stderr or "")[-1000:])
        self.assertIn("9/9 checks passed", r.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
