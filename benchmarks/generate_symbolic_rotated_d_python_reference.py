"""Python symbolic reference for the SHAARP crystal->lab Voigt-d rotation.

SHAARP.ml `extMater` computes the lab-frame SHG Voigt tensor as
    dL = t2m3[ TensorContract[ QC QC QC. m2t3[dC], {{2,7},{4,8},{6,9}} ] ]
i.e. a full rank-3 rotation of the crystal SHG tensor dC by the crystal->lab
matrix QC. SHAARP.si's analytical branch uses the same `dnewExp` loop. Python's
`shaarp.symbolic.rotate_d_voigt_symbolic` mirrors that loop.

This builds the Python symbolic dL (a 3x6 Voigt tensor with symbolic crystal
components d11..d36) for a FIXED, EXACT-RATIONAL orientation matrix QC, so the
live-Wolfram comparator can check both an exact symbolic residual and numeric
substitution. The rational QC (composed from 3-4-5 and 5-12-13 Pythagorean
rotations) is orthonormal with exact rational entries, which keeps the rotated
coefficients exact rationals on both sides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "python_symbolic_rotated_d_reference_v1.json"


def rational_crystal_to_lab():
    """Exact-rational, orthonormal crystal->lab rotation QC = Rz(3-4-5) . Rx(5-12-13)."""
    import sympy as sp

    c1, s1 = sp.Rational(3, 5), sp.Rational(4, 5)
    c2, s2 = sp.Rational(5, 13), sp.Rational(12, 13)
    rz = sp.Matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    rx = sp.Matrix([[1, 0, 0], [0, c2, -s2], [0, s2, c2]])
    return rz * rx


def rational_crystal_to_lab_b():
    """Second independent exact-rational rotation QC = Rz(8-15-17) . Ry(20-21-29)."""
    import sympy as sp

    c1, s1 = sp.Rational(15, 17), sp.Rational(8, 17)
    c2, s2 = sp.Rational(21, 29), sp.Rational(20, 29)
    rz = sp.Matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    ry = sp.Matrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    return rz * ry


# expression_id -> rotation builder. The first id is kept stable across versions.
ROTATIONS = {
    "rotated_d_voigt_symbolic": rational_crystal_to_lab,
    "rotated_d_voigt_symbolic_b": rational_crystal_to_lab_b,
}


def build_payload() -> dict:
    import sympy as sp

    from shaarp.symbolic import rotate_d_voigt_symbolic

    d_symbols = sp.Matrix([[sp.Symbol(f"d{i}{j}") for j in range(1, 7)] for i in range(1, 4)])
    symbol_roles = {f"d{i}{j}": f"Subscript[d,{i}{j}]" for i in range(1, 4) for j in range(1, 7)}

    expressions = []
    rotations_rational = {}
    for expression_id, builder in ROTATIONS.items():
        qc = builder()
        d_lab = rotate_d_voigt_symbolic(d_symbols, qc, simplify=True)
        flat = [str(d_lab[i, j]) for i in range(3) for j in range(6)]
        rotations_rational[expression_id] = [[str(qc[i, j]) for j in range(3)] for i in range(3)]
        expressions.append(
            {
                "expression_id": expression_id,
                "source_stage": "extMater_dL_dnewExp",
                "output_component": "lab_frame_voigt_d_3x6_row_major_flat",
                "expression_format": "sympy",
                "expression": flat,
                "symbol_roles": symbol_roles,
                "numeric_substitutions": _numeric_substitutions(),
            }
        )

    return {
        "source": "Python SymPy symbolic crystal-to-lab Voigt-d rotation (dnewExp / extMater dL)",
        "status": "python_symbolic_output",
        "rotation_crystal_to_lab_rational": rotations_rational,
        "expressions": expressions,
    }


def _numeric_substitutions() -> list[dict]:
    first, second = {}, {}
    for i in range(1, 4):
        for j in range(1, 7):
            first[f"d{i}{j}"] = {"real": 0.07 * i - 0.03 * j, "imag": 0.011 * i * j}
            second[f"d{i}{j}"] = {"real": -0.02 * i + 0.05 * j, "imag": -0.013 * (i + j)}
    return [first, second]


def write_payload(path: Path = DEFAULT_OUTPUT) -> dict:
    payload = build_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Python symbolic rotated Voigt-d comparison payload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_payload(args.output)
    print(f"Wrote Python symbolic rotated-d payload to {args.output}")


if __name__ == "__main__":
    main()
