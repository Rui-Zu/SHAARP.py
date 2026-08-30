"""Validate the SYMBOLIC single-interface solveInhom against LIVE Wolfram values.

`tests/test_symbolic.py` already checks the symbolic solveInhom against the
PYTHON numeric solveInhom (self-consistency). This closes the Standing-Rule-4 gap:
it evaluates the symbolic ee/oo/eo inhomogeneous 2omega fields at the exact inputs
of the 20 LIVE Wolfram `solveInhom` reference cases
(`solve_inhom_reference_v1.json`, status=mathematica_reference_exported) and
compares the symbolic-evaluated electric fields against the Wolfram numeric
ee/oo/eo. No new Wolfram export is needed -- it reuses the existing live reference.

Returns a machine-readable summary; gated by
`tests/test_symbolic_solve_inhom_live_comparison.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks" / "mathematica_reference" / "solve_inhom_reference_v1.json"
SUMMARY_OUT = ROOT / "benchmarks" / "mathematica_reference" / "symbolic_solve_inhom_live_comparison_summary_v1.json"


def _cx(v: Any) -> Any:
    if isinstance(v, dict):
        return complex(v["real"], v["imag"])
    if isinstance(v, list):
        return np.array([_cx(x) for x in v])
    return v


def build_summary(*, atol: float = 1e-9, rtol: float = 1e-9) -> dict[str, Any]:
    import sympy as sp

    from shaarp.symbolic import (
        SymbolicWave,
        single_interface_sources_symbolic,
        solve_single_interface_inhomogeneous_fields_symbolic,
    )

    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("solveInhom reference is not Mathematica/Wolfram sourced.")

    results = []
    for case in payload["cases"]:
        inp = case["inputs"]
        eps = _cx(inp["epsilon_2omega_lab"])
        d_voigt = _cx(inp["d_voigt_lab"])
        e1 = _cx(inp["e1"])
        e2 = _cx(inp["e2"])
        k1 = _cx(inp["k1"])
        k2 = _cx(inp["k2"])
        omega = _cx(inp["omega"])

        fast = SymbolicWave(k=sp.Matrix(k1), electric=sp.Matrix(e1), omega=complex(omega))
        slow = SymbolicWave(k=sp.Matrix(k2), electric=sp.Matrix(e2), omega=complex(omega))
        sources = single_interface_sources_symbolic(sp.Matrix(d_voigt), fast, slow)
        fields = solve_single_interface_inhomogeneous_fields_symbolic(
            sp.Matrix(eps), sources, mu=1, eps0=1, simplify=False
        )

        case_result = {"id": case["id"], "components": {}}
        worst = 0.0
        for label in ("ee", "oo", "eo"):
            sym_field = getattr(fields, label)
            sym_vec = np.array([complex(val.evalf()) for val in sym_field.electric], dtype=complex)
            wol_vec = _cx(case["outputs"][label])
            delta = float(np.max(np.abs(sym_vec - wol_vec)))
            scale = np.maximum(np.abs(wol_vec), np.finfo(float).tiny)
            rel = float(np.max(np.abs(sym_vec - wol_vec) / scale))
            passed = bool(np.allclose(sym_vec, wol_vec, atol=atol, rtol=rtol))
            worst = max(worst, delta)
            case_result["components"][label] = {
                "max_abs_error": delta,
                "max_rel_error": rel,
                "passed": passed,
            }
        case_result["max_abs_error"] = worst
        case_result["passed"] = all(c["passed"] for c in case_result["components"].values())
        results.append(case_result)

    n_pass = sum(1 for r in results if r["passed"])
    summary = {
        "source": payload.get("source"),
        "comparison": "symbolic_single_interface_solveInhom_vs_live_wolfram_numeric",
        "status": (
            "symbolic_solve_inhom_matches_live_wolfram"
            if n_pass == len(results)
            else "symbolic_solve_inhom_has_mismatch"
        ),
        "full_symbolic_solve_inhom_agreement_claimed": n_pass == len(results),
        "atol": atol,
        "rtol": rtol,
        "case_count": len(results),
        "passing_case_count": n_pass,
        "failing_case_count": len(results) - n_pass,
        "max_abs_error_over_all_cases": max((r["max_abs_error"] for r in results), default=0.0),
        "results": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare symbolic single-interface solveInhom vs live Wolfram numeric.")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()
    summary = build_summary(atol=args.atol, rtol=args.rtol)
    if args.summary_out is not None:
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))
        return
    for r in summary["results"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'} {r['id']}: max_abs={r['max_abs_error']:.3e}")
    print(f"{summary['passing_case_count']}/{summary['case_count']} symbolic solveInhom cases match live Wolfram "
          f"(max_abs={summary['max_abs_error_over_all_cases']:.3e})")
    if summary["failing_case_count"]:
        raise SystemExit(f"{summary['failing_case_count']} symbolic solveInhom mismatches.")


if __name__ == "__main__":
    main()
