"""Python symbolic reference for SHAARP.si point-group SHG Voigt templates (doldExp).

For each point group that the live-Wolfram extractor pulled from SHAARP_V1.03.nb,
this emits Python's `shaarp.symbolic.d_voigt_symbolic` for the SAME point group,
keyed by the IDENTICAL `expression_id` so the symbolic comparator can check that
Python's point-group zero/equality/sign pattern matches SHAARP.si exactly
(symbolic residual zero) plus numeric substitution.

The Wolfram point-group labels use Mathematica linear-syntax for barred groups
(OverscriptBox) and the infinity glyph for Curie groups; `_wolfram_pg_to_alias`
normalizes them to the ASCII aliases `d_voigt_symbolic` accepts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WOLFRAM_REFERENCE = (
    ROOT / "benchmarks" / "mathematica_reference" / "wolfram_live_symbolic_doldexp_reference_v1.json"
)
DEFAULT_OUTPUT = ROOT / "benchmarks" / "python_symbolic_doldexp_reference_v1.json"


def _wolfram_pg_to_alias(pg: str) -> str:
    """Map a Wolfram point-group label string to a d_voigt_symbolic alias."""
    plain = {"1", "2", "m", "mm2", "222", "3", "32", "3m", "23", "4", "6",
             "4mm", "6mm", "422", "622", "-4", "-42m", "-6", "-6m2"}
    if pg in plain:
        return pg
    # Curie (infinite) groups use the infinity glyph U+221E.
    if "∞" in pg:
        tail = "".join(c for c in pg if c in "m2")
        if tail == "":
            return "inf"
        if tail == "m":
            return "infm"
        if tail == "2":
            return "inf2"
    # Barred groups: Mathematica OverscriptBox linear syntax. Recover the base
    # digit and any ASCII suffix (e.g. "2m" -> -42m, "m2" -> -6m2).
    ascii_alnum = [c for c in pg if c in "0123456789m"]
    if ascii_alnum:
        base = ascii_alnum[0]
        suffix = "".join(ascii_alnum[1:])
        if base == "4":
            return "-42m" if "2m" in suffix else "-4"
        if base == "6":
            return "-6m2" if "m2" in suffix else "-6"
    raise ValueError(f"Cannot map Wolfram point-group label to alias: {pg!r}")


def _numeric_substitutions() -> list[dict]:
    first, second = {}, {}
    for i in range(1, 4):
        for j in range(1, 7):
            first[f"d{i}{j}"] = {"real": 0.07 * i - 0.03 * j, "imag": 0.011 * i * j}
            second[f"d{i}{j}"] = {"real": -0.02 * i + 0.05 * j, "imag": -0.013 * (i + j)}
    return [first, second]


def build_payload() -> dict:
    from shaarp.symbolic import d_voigt_symbolic

    reference = json.loads(WOLFRAM_REFERENCE.read_text(encoding="utf-8"))
    symbol_roles = {f"d{i}{j}": f"Subscript[d,{i}{j}]" for i in range(1, 4) for j in range(1, 7)}
    subs = _numeric_substitutions()

    expressions = []
    for record in reference["expressions"]:
        pg_label = record["point_group"]
        expression_id = record["expression_id"]
        alias = _wolfram_pg_to_alias(pg_label)
        mat = d_voigt_symbolic(alias)
        flat = [str(mat[i, j]) for i in range(3) for j in range(6)]
        expressions.append(
            {
                "expression_id": expression_id,
                "point_group": pg_label,
                "python_alias": alias,
                "source_stage": "SHAARP_si_doldExp_point_group_template",
                "output_component": "crystal_frame_voigt_d_3x6_row_major_flat",
                "expression_format": "sympy",
                "expression": flat,
                "symbol_roles": symbol_roles,
                "numeric_substitutions": subs,
            }
        )

    return {
        "source": "Python SymPy d_voigt_symbolic point-group templates (mirror of SHAARP.si doldExp)",
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
    parser = argparse.ArgumentParser(description="Generate Python symbolic doldExp point-group comparison payload.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_payload(args.output)
    print(f"Wrote Python symbolic doldExp payload to {args.output}")


if __name__ == "__main__":
    main()
