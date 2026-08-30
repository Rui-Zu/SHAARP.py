from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "python_symbolic_ml_partial_pnl_reference_v1.json"

SOURCE_SPECS = [
    ("PNLF1F1", "F1F1", "F1", "F1", 1),
    ("PNLF2F2", "F2F2", "F2", "F2", 1),
    ("PNLF1F2", "F1F2", "F1", "F2", 2),
    ("PNLB1B1", "B1B1", "B1", "B1", 1),
    ("PNLB2B2", "B2B2", "B2", "B2", 1),
    ("PNLB1B2", "B1B2", "B1", "B2", 2),
    ("PNLF1B1", "F1B1", "F1", "B1", 2),
    ("PNLF2B2", "F2B2", "F2", "B2", 2),
    ("PNLF1B2", "F1B2", "F1", "B2", 2),
    ("PNLF2B1", "F2B1", "F2", "B1", 2),
]


def build_payload() -> dict:
    import sympy as sp

    from shaarp.symbolic import compute_pnl_voigt_symbolic

    d_symbols = sp.Matrix([[sp.Symbol(f"d{i}{j}") for j in range(1, 7)] for i in range(1, 4)])
    fields = {
        label: sp.Matrix([sp.Symbol(f"{label}x"), sp.Symbol(f"{label}y"), sp.Symbol(f"{label}z")])
        for label in ("F1", "F2", "B1", "B2")
    }
    symbol_roles = _symbol_roles()
    substitutions = _numeric_substitutions()
    expressions = []
    for output_component, short_id, left, right, factor in SOURCE_SPECS:
        expr = compute_pnl_voigt_symbolic(d_symbols, fields[left], fields[right], factor=factor)
        expressions.append(
            {
                "expression_id": f"ml_partial_{short_id}",
                "source_stage": "SHAARP.ml_partial_computePNL",
                "output_component": output_component,
                "expression_format": "sympy",
                "expression": [str(item) for item in expr],
                "symbol_roles": symbol_roles,
                "numeric_substitutions": substitutions,
            }
        )
    return {
        "source": "Python SymPy SHAARP.ml ten-source partial PNL output",
        "status": "python_symbolic_output",
        "expressions": expressions,
    }


def write_payload(path: Path = DEFAULT_OUTPUT) -> dict:
    payload = build_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _symbol_roles() -> dict[str, str]:
    roles = {f"d{i}{j}": f"Subscript[d,{i}{j}]" for i in range(1, 4) for j in range(1, 7)}
    for label in ("F1", "F2", "B1", "B2"):
        for axis in ("x", "y", "z"):
            roles[f"{label}{axis}"] = f"Subscript[Superscript[E,omega],{label},{axis}]"
    return roles


def _numeric_substitutions() -> list[dict]:
    first = {}
    second = {}
    labels = ("F1", "F2", "B1", "B2")
    axes = ("x", "y", "z")
    for i in range(1, 4):
        for j in range(1, 7):
            first[f"d{i}{j}"] = {"real": 0.031 * i - 0.017 * j, "imag": 0.007 * i * j}
            second[f"d{i}{j}"] = {"real": -0.041 * i + 0.019 * j, "imag": -0.005 * (i + 2 * j)}
    for idx, label in enumerate(labels, start=1):
        for axis_idx, axis in enumerate(axes, start=1):
            first[f"{label}{axis}"] = {
                "real": 0.09 * len(label) + 0.03 * ord(axis) / 120 - 0.04 * idx,
                "imag": -0.05 * idx + 0.02 * axis_idx,
            }
            second[f"{label}{axis}"] = {
                "real": -0.07 * idx + 0.025 * ord(axis) / 130,
                "imag": 0.04 * idx - 0.015 * len(label) - 0.01 * axis_idx,
            }
    return [first, second]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Python symbolic SHAARP.ml partial PNL payload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_payload(args.output)
    print(f"Wrote Python symbolic ML partial PNL payload to {args.output}")


if __name__ == "__main__":
    main()
