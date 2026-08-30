from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "python_symbolic_compute_pnl_reference_v1.json"


def build_payload() -> dict:
    import sympy as sp

    from shaarp.symbolic import compute_pnl_voigt_symbolic

    d_symbols = sp.Matrix([[sp.Symbol(f"d{i}{j}") for j in range(1, 7)] for i in range(1, 4)])
    e1_symbols = sp.Matrix([sp.Symbol("e1x"), sp.Symbol("e1y"), sp.Symbol("e1z")])
    e2_symbols = sp.Matrix([sp.Symbol("e2x"), sp.Symbol("e2y"), sp.Symbol("e2z")])
    same_source = compute_pnl_voigt_symbolic(d_symbols, e1_symbols, e1_symbols)
    mixed_source = compute_pnl_voigt_symbolic(d_symbols, e1_symbols, e2_symbols, factor=2)
    symbol_roles = _symbol_roles()
    substitutions = _numeric_substitutions()
    return {
        "source": "Python SymPy symbolic computePNL output",
        "status": "python_symbolic_output",
        "expressions": [
            _record(
                "compute_pnl_same_symbolic",
                "same_wave_vector",
                same_source,
                symbol_roles=symbol_roles,
                numeric_substitutions=substitutions,
            ),
            _record(
                "compute_pnl_mixed_symbolic",
                "mixed_wave_vector_factor_2",
                mixed_source,
                symbol_roles=symbol_roles,
                numeric_substitutions=substitutions,
            ),
        ],
    }


def write_payload(path: Path = DEFAULT_OUTPUT) -> dict:
    payload = build_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _record(
    expression_id: str,
    output_component: str,
    expression,
    *,
    symbol_roles: dict[str, str],
    numeric_substitutions: list[dict],
) -> dict:
    return {
        "expression_id": expression_id,
        "source_stage": "computePNL",
        "output_component": output_component,
        "expression_format": "sympy",
        "expression": [str(item) for item in expression],
        "symbol_roles": symbol_roles,
        "numeric_substitutions": numeric_substitutions,
    }


def _symbol_roles() -> dict[str, str]:
    roles = {f"d{i}{j}": f"Subscript[d,{i}{j}]" for i in range(1, 4) for j in range(1, 7)}
    roles.update(
        {
            "e1x": "Subscript[Superscript[E,omega],1,x]",
            "e1y": "Subscript[Superscript[E,omega],1,y]",
            "e1z": "Subscript[Superscript[E,omega],1,z]",
            "e2x": "Subscript[Superscript[E,omega],2,x]",
            "e2y": "Subscript[Superscript[E,omega],2,y]",
            "e2z": "Subscript[Superscript[E,omega],2,z]",
        }
    )
    return roles


def _numeric_substitutions() -> list[dict]:
    first = {}
    second = {}
    for i in range(1, 4):
        for j in range(1, 7):
            first[f"d{i}{j}"] = {"real": 0.07 * i - 0.03 * j, "imag": 0.011 * i * j}
            second[f"d{i}{j}"] = {"real": -0.02 * i + 0.05 * j, "imag": -0.013 * (i + j)}
    first.update(
        {
            "e1x": {"real": 0.23, "imag": -0.17},
            "e1y": {"real": -0.41, "imag": 0.09},
            "e1z": {"real": 0.13, "imag": 0.31},
            "e2x": {"real": -0.29, "imag": 0.19},
            "e2y": {"real": 0.37, "imag": -0.23},
            "e2z": {"real": -0.11, "imag": -0.07},
        }
    )
    second.update(
        {
            "e1x": {"real": 0.08, "imag": 0.44},
            "e1y": {"real": 0.51, "imag": -0.12},
            "e1z": {"real": -0.33, "imag": 0.27},
            "e2x": {"real": 0.19, "imag": -0.38},
            "e2y": {"real": -0.47, "imag": -0.16},
            "e2z": {"real": 0.22, "imag": 0.29},
        }
    )
    return [first, second]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Python symbolic computePNL comparison payload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_payload(args.output)
    print(f"Wrote Python symbolic computePNL payload to {args.output}")


if __name__ == "__main__":
    main()
