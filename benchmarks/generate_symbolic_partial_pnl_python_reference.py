"""Python symbolic reference for SHAARP.si partial-analytical nonlinear sources.

SHAARP.si's GUI cell builds the partial-analytical nonlinear-source polarization
as `e0 * Simplify[{...}]` using the Voigt computePNL formula on a lab-frame
d-tensor (dnewExp) and the eigenmode 2omega fields:

    P2o = computePNL(d, ETwo) (same-mode, ordinary field)
    Peoo = 2 * computePNL(d, ETweo, ETwo) (mixed e-o term, factor 2)

This emits Python's `compute_pnl_voigt_symbolic` for the SAME symbolic d-tensor
and field symbols, keyed by the identical expression_id, reusing the
live-Wolfram reference's symbol_roles and numeric substitutions so the symbolic
comparator can confirm SHAARP.si's hand-written assembly equals Python exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WOLFRAM_REFERENCE = (
    ROOT / "benchmarks" / "mathematica_reference" / "wolfram_live_symbolic_partial_pnl_reference_v1.json"
)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "python_symbolic_partial_pnl_reference_v1.json"


def build_payload() -> dict:
    import sympy as sp

    from shaarp.symbolic import compute_pnl_voigt_symbolic, d_voigt_symbolic

    d_full = d_voigt_symbolic("1")  # generic 3x6 tensor with symbols d11..d36
    eo = sp.Matrix([sp.Symbol("eo1"), sp.Symbol("eo2"), sp.Symbol("eo3")])
    eeo = sp.Matrix([sp.Symbol("eeo1"), sp.Symbol("eeo2"), sp.Symbol("eeo3")])

    # shaarp_symbol -> python symbolic nonlinear-source polarization
    builders = {
        "P2o": lambda: compute_pnl_voigt_symbolic(d_full, eo),
        "P2eo": lambda: compute_pnl_voigt_symbolic(d_full, eeo),
        "Peoo": lambda: compute_pnl_voigt_symbolic(d_full, eeo, eo, factor=2),
    }

    reference = json.loads(WOLFRAM_REFERENCE.read_text(encoding="utf-8"))
    expressions = []
    for record in reference["expressions"]:
        symbol = record["shaarp_symbol"]
        if symbol not in builders:
            raise ValueError(f"No Python builder for SHAARP partial-PNL symbol {symbol!r}.")
        vec = builders[symbol]()
        flat = [str(vec[i]) for i in range(3)]
        expressions.append(
            {
                "expression_id": record["expression_id"],
                "shaarp_symbol": symbol,
                "source_stage": record["source_stage"],
                "output_component": record["output_component"],
                "expression_format": "sympy",
                "expression": flat,
                # reuse the live reference's roles/subs so they match exactly
                "symbol_roles": record["symbol_roles"],
                "numeric_substitutions": record["numeric_substitutions"],
            }
        )

    return {
        "source": "Python SymPy compute_pnl_voigt_symbolic partial-analytical nonlinear sources (mirror of SHAARP.si P2o/Peoo)",
        "status": "python_symbolic_output",
        "record_count": len(expressions),
        "expressions": expressions,
    }


def write_payload(path: Path = DEFAULT_OUTPUT) -> dict:
    payload = build_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Python symbolic partial-PNL comparison payload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_payload(args.output)
    print(f"Wrote Python symbolic partial-PNL payload to {args.output}")


if __name__ == "__main__":
    main()
