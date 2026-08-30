"""Validate Python's symbolic Voigt computePNL against LIVE Wolfram SHAARP.si
PNL-stage source polarizations (P2o, P2eo, Peoo) over all 36 cases.

The live `shaarp_si_pnl_stage_reference_v1.json` exposes, per case, the lab-frame
SHG tensor `dSHG`, the eigenmode 2omega fields `ETwo`/`ETweo`, and the numeric
source polarizations `P2o`/`P2eo`/`Peoo` that SHAARP.si computes. This feeds the
SAME d and fields into Python's `compute_pnl_voigt_symbolic` and checks the
symbolic result (evaluated numerically) against the live Wolfram values:

    P2o = computePNL(dSHG, ETwo) (same-mode, ordinary field)
    P2eo = computePNL(dSHG, ETweo) (same-mode, eo-combined field)
    Peoo = 2 * computePNL(dSHG, ETweo, ETwo) (mixed, factor 2)

This is a live-Wolfram validation of the symbolic PNL building block over a much
broader (36-case, biaxial-anisotropic) grid than the existing partial-PNL symbolic
test. Reuses the existing live reference; no new Wolfram export needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks" / "mathematica_reference" / "shaarp_si_pnl_stage_reference_v1.json"


def _cx(v: Any) -> Any:
    if isinstance(v, dict):
        return complex(v["real"], v["imag"])
    if isinstance(v, list):
        return np.array([_cx(x) for x in v])
    return v


def _sym_pnl(d_voigt, e1, e2=None, factor=1):
    import sympy as sp

    from shaarp.symbolic import compute_pnl_voigt_symbolic

    dM = sp.Matrix(np.asarray(d_voigt).tolist())
    E1 = sp.Matrix(np.asarray(e1).reshape(3, 1).tolist())
    E2 = sp.Matrix(np.asarray(e2 if e2 is not None else e1).reshape(3, 1).tolist())
    res = compute_pnl_voigt_symbolic(dM, E1, E2, factor=factor)
    return np.array([complex(v) for v in res], dtype=complex)


def build_summary(*, atol: float = 1e-9, rtol: float = 1e-9) -> dict[str, Any]:
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    src = str(payload.get("source", "")).lower()
    if "mathematica" not in src and "wolfram" not in src:
        raise ValueError("PNL-stage reference is not Mathematica/Wolfram sourced.")

    results = []
    for case in payload["cases"]:
        out = case.get("outputs", {})
        if not all(k in out for k in ("dSHG", "ETwo", "ETweo", "P2o", "P2eo", "Peoo")):
            continue  # skip cases lacking the PNL-stage fields
        dshg = _cx(out["dSHG"])
        etwo = _cx(out["ETwo"])
        etweo = _cx(out["ETweo"])
        comps = {}
        worst = 0.0
        for name, ref_key, args in (
            ("P2o", "P2o", (etwo,)),
            ("P2eo", "P2eo", (etweo,)),
            ("Peoo", "Peoo", (etweo, etwo, 2)),
        ):
            ref = _cx(out[ref_key])
            if len(args) == 1:
                got = _sym_pnl(dshg, args[0])
            else:
                got = _sym_pnl(dshg, args[0], args[1], factor=args[2])
            delta = float(np.max(np.abs(got - ref)))
            worst = max(worst, delta)
            comps[name] = {"max_abs_error": delta, "passed": bool(np.allclose(got, ref, atol=atol, rtol=rtol))}
        results.append(
            {
                "id": case["id"],
                "components": comps,
                "max_abs_error": worst,
                "passed": all(c["passed"] for c in comps.values()),
            }
        )

    n_pass = sum(1 for r in results if r["passed"])
    return {
        "source": payload.get("source"),
        "comparison": "symbolic_computePNL_vs_live_wolfram_si_pnl_stage_P2o_P2eo_Peoo",
        "status": (
            "symbolic_pnl_matches_live_wolfram"
            if results and n_pass == len(results)
            else "symbolic_pnl_has_mismatch"
        ),
        "full_symbolic_pnl_agreement_claimed": bool(results) and n_pass == len(results),
        "atol": atol,
        "rtol": rtol,
        "case_count": len(results),
        "passing_case_count": n_pass,
        "failing_case_count": len(results) - n_pass,
        "max_abs_error_over_all_cases": max((r["max_abs_error"] for r in results), default=0.0),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare symbolic computePNL vs live Wolfram SI PNL stage.")
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
    print(f"{summary['passing_case_count']}/{summary['case_count']} symbolic PNL cases match live Wolfram "
          f"(max_abs={summary['max_abs_error_over_all_cases']:.3e})")
    if summary["failing_case_count"]:
        raise SystemExit(f"{summary['failing_case_count']} symbolic PNL mismatches.")


if __name__ == "__main__":
    main()
