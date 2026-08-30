"""Python symbolic reference matching SHAARP.si's live GUI dnewExp For-loop.

Python's rotate_d_voigt_symbolic mirrors that loop; here we evaluate it for the
SAME exact-rational crystal->lab matrix `a` = Rz(3-4-5) . Rx(5-12-13) and a
symbolic crystal tensor d11..d36, keyed to the loop reference's expression_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WOLFRAM_REFERENCE = (
    ROOT / "benchmarks" / "mathematica_reference" / "wolfram_live_symbolic_dnewexp_loop_reference_v1.json"
)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "python_symbolic_dnewexp_loop_reference_v1.json"


def build_payload() -> dict:
    import sympy as sp

    from shaarp.symbolic import d_voigt_symbolic, rotate_d_voigt_symbolic

    c1, s1 = sp.Rational(3, 5), sp.Rational(4, 5)
    c2, s2 = sp.Rational(5, 13), sp.Rational(12, 13)
    rz = sp.Matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    rx = sp.Matrix([[1, 0, 0], [0, c2, -s2], [0, s2, c2]])
    a = rz * rx

    d_full = d_voigt_symbolic("1")  # generic 3x6 d11..d36
    d_lab = rotate_d_voigt_symbolic(d_full, a, simplify=True)
    flat = [str(d_lab[i, j]) for i in range(3) for j in range(6)]

    reference = json.loads(WOLFRAM_REFERENCE.read_text(encoding="utf-8"))
    ref_record = reference["expressions"][0]
    return {
        "source": "Python SymPy rotate_d_voigt_symbolic (mirror of SHAARP.si dnewExp GUI for-loop)",
        "status": "python_symbolic_output",
        "expressions": [
            {
                "expression_id": ref_record["expression_id"],
                "source_stage": ref_record["source_stage"],
                "output_component": ref_record["output_component"],
                "expression_format": "sympy",
                "expression": flat,
                "symbol_roles": ref_record["symbol_roles"],
                "numeric_substitutions": ref_record["numeric_substitutions"],
            }
        ],
    }


def write_payload(path: Path = DEFAULT_OUTPUT) -> dict:
    payload = build_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Python symbolic dnewExp-loop comparison payload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_payload(args.output)
    print(f"Wrote Python symbolic dnewExp-loop payload to {args.output}")


if __name__ == "__main__":
    main()
