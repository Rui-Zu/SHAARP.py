from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_maker_stage_diagnostics import build_stage_deltas  # noqa: E402


STAGE_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_high_oblique_12_stage_diagnostics_v1.json"
SUMMARY_PATH = (
    ROOT / "benchmarks" / "mathematica_reference" / "maker_high_oblique_12_stage_comparison_summary_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize high-oblique theta=12 Maker f4NL stage residuals.")
    parser.add_argument("--mathematica", type=Path, default=STAGE_PATH)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    payload = json.loads(args.mathematica.read_text(encoding="utf-8"))
    summary = build_summary(payload, atol=args.atol)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.summary_out), "status": summary["status"]}, sort_keys=True))


def build_summary(payload: dict, *, atol: float = 1e-10) -> dict:
    deltas = build_stage_deltas(payload)
    finite = [delta for delta in deltas if math.isfinite(delta.max_abs)]
    understood = [
        delta
        for delta in finite
        if (
            ".best_" in delta.stage
            or delta.stage in {"linear_transmitted[0].electric", "linear_transmitted[0].k"}
        )
    ]
    failing = [delta for delta in understood if delta.max_abs > atol]
    top = sorted(finite, key=lambda item: item.max_abs, reverse=True)[:25]
    top_understood = sorted(understood, key=lambda item: item.max_abs, reverse=True)[:25]
    first_residual = min(failing, key=_stage_order_key) if failing else None
    return {
        "source": payload.get("source"),
        "status": "high_oblique_12_stage_has_residual_above_strict_tolerance" if failing else "high_oblique_12_stage_passes",
        "agreement_claimed": not failing,
        "case_id": payload["cases"][0]["id"],
        "theta_deg": payload["cases"][0]["points"][0]["theta_deg"],
        "tolerance": atol,
        "first_understood_residual_stage": None if first_residual is None else first_residual.stage,
        "first_understood_residual_max_abs": None if first_residual is None else first_residual.max_abs,
        "max_understood_residual_stage": None if not top_understood else top_understood[0].stage,
        "max_understood_residual_abs": None if not top_understood else top_understood[0].max_abs,
        "top_finite_deltas": [_delta_record(delta) for delta in top],
        "top_understood_deltas": [_delta_record(delta) for delta in top_understood],
        "known_caveats": [
            "Direct reflected-wave indices include Mathematica/Python ordering differences; use best-match reflected records.",
            "Mathematica emits absent selected-wave slots for some transmitted waves; those are selection-policy diagnostics.",
        ],
    }


def _stage_order_key(delta) -> tuple[int, str]:
    if delta.stage.startswith("linear_"):
        group = 0
    elif delta.stage.startswith("inhomogeneous_"):
        group = 1
    elif "homogeneous_layer" in delta.stage:
        group = 2
    elif delta.stage.startswith("nonlinear_"):
        group = 3
    else:
        group = 4
    return (group, delta.stage)


def _delta_record(delta) -> dict:
    return {
        "case_id": delta.case_id,
        "theta_deg": delta.theta_deg,
        "stage": delta.stage,
        "max_abs": delta.max_abs,
    }


if __name__ == "__main__":
    main()
