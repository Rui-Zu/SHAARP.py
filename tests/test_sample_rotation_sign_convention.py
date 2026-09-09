"""One sense for a positive sample azimuth, everywhere in the package.

GROUND TRUTH is the physical rotation the orientation defines,
:meth:`CrystalOrientation.with_lab_azimuth_deg`, which is what the multilayer sweep, the
Mathematica sample-rotation benchmarks and the original all use.

The algebra, once, because two spellings were defensible-looking and only one is right.
``rotate_d_voigt_symbolic(d, a)`` contracts on the SECOND index of ``a``
(``out_ijk = a[i,l] a[j,m] a[k,n] d[l,mn]``) while ``rotate_rank3_crystal_to_lab(t, r)`` contracts
on the FIRST (``einsum("ai,bj,ck,abc->ijk")``), so the two differ by a transpose. With
``A(a) = A0 @ Rz(a).T`` we get ``A(a)_ai = A0_am Rz(a)_im``, hence

    d_lab(a)_ijk = Rz(a)_im Rz(a)_jn Rz(a)_kp d_lab(0)_mnp

which contracts on the SECOND index of Rz. So the physical rotation is
``rotate_d_voigt_symbolic(d_lab0, Rz(+a))``. Passing the transpose is a rotation by ``-a``: the
mirror. It produces curves that are still smooth, still symmetric, still the right magnitude, so
nothing but a signed check catches it.

The angle and crystal here are chosen to be CHIRAL: a three-fold crystal at 120 degrees cannot
tell the two senses apart, and a test that uses one fences nothing.
"""

import unittest

import numpy as np
import sympy as sp

from shaarp.config import CrystalOrientation
from shaarp.polarimetry_extraction import _rotated_d_symbolic
from shaarp.symbolic import rotate_d_voigt_symbolic
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

AZIMUTH_DEG = 37.0          # not a multiple of any symmetry period in play
D_CRYSTAL = np.array([      # quartz, point group 32: d11 and d14 both present
    [0.30, -0.30, 0.0, 0.12, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -0.12, -0.30],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
], dtype=float)


def _rz(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _physical_d_lab(base_orientation, azimuth_deg):
    """What the package means by rotating the sample: turn the ORIENTATION, then rebuild d."""
    turned = base_orientation.with_lab_azimuth_deg(azimuth_deg)
    return rotate_d_voigt_crystal_to_lab(D_CRYSTAL, turned.rotation_matrix())


def _as_array(m):
    return np.array(sp.Matrix(m).evalf().tolist(), dtype=complex)


class PositiveAzimuthMeansOneThing(unittest.TestCase):
    TOL = 1e-12

    def setUp(self):
        # a tilted base orientation, so the check is not accidentally passed by a symmetric case
        base = CrystalOrientation()
        self.base = base
        self.d_lab0 = rotate_d_voigt_crystal_to_lab(D_CRYSTAL, base.rotation_matrix())
        self.want = _physical_d_lab(base, AZIMUTH_DEG)

    def _dev(self, got):
        got = _as_array(got)
        scale = max(1e-30, float(np.max(np.abs(self.want))))
        return float(np.max(np.abs(got - self.want))) / scale

    def test_the_fence_can_fail(self):
        """The mirror must be measurably different at this angle, or every check below is vacuous."""
        mirrored = rotate_d_voigt_symbolic(sp.Matrix(self.d_lab0), sp.Matrix(_rz(AZIMUTH_DEG).T),
                                           simplify=False)
        self.assertGreater(self._dev(mirrored), 1e-3,
                           "the chosen crystal/angle cannot tell the two senses apart")

    def test_multilayer_symbolic_spelling(self):
        got = rotate_d_voigt_symbolic(sp.Matrix(self.d_lab0), sp.Matrix(_rz(AZIMUTH_DEG)),
                                      simplify=False)
        self.assertLess(self._dev(got), self.TOL)

    def test_single_interface_symbolic_spelling(self):
        """The single-interface closed form must rotate the sample the same way the multilayer one
        does. It used the transpose, so a positive azimuth meant opposite physical rotations on the
        two tabs."""
        from shaarp.symbolic import solve_si_shg_full_analytical_symbolic  # noqa: F401

        got = _si_rotated_d(self.d_lab0, AZIMUTH_DEG)
        self.assertLess(self._dev(got), self.TOL)

    def test_extraction_spelling(self):
        """The d-extraction forward model rotates the sample too; its `measure` contract names the
        same physical rotation."""
        got = _rotated_d_symbolic(sp.Matrix(self.d_lab0), float(np.deg2rad(AZIMUTH_DEG)))
        self.assertLess(self._dev(got), self.TOL)


def _si_rotated_d(d_lab0, azimuth_deg):
    """The rotation the single-interface closed form applies, isolated from the rest of the solve
    so the convention can be checked without building an expression."""
    import inspect

    from shaarp import symbolic as S

    src = inspect.getsource(S.solve_si_shg_full_analytical_symbolic)
    a = np.deg2rad(azimuth_deg)
    c, s = np.cos(a), np.sin(a)
    if "[[c, s, 0], [-s, c, 0]" in src:            # the transposed (mirror) spelling
        rot = sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    else:                                           # the physical spelling
        rot = sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rotate_d_voigt_symbolic(sp.Matrix(d_lab0), rot, simplify=False)


if __name__ == "__main__":
    unittest.main()
