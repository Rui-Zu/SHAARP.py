from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_maker_fringes_benchmarks import build_cases, run_case  # noqa: E402
from benchmarks.generate_maker_fringes_fine_sampling_plan import build_plan  # noqa: E402


DEFAULT_OUTPUT = ROOT / "benchmarks" / "maker_fringes_fine_sampling_output_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optional slow Maker fringes fine-sampling verification.")
    parser.add_argument("--confirm-slow", action="store_true", help="Required because dense Maker fringes can be slow.")
    parser.add_argument("--case-id", action="append", help="Optional fine-sampling case id to run. May be repeated.")
    parser.add_argument("--max-points", type=int, default=None, help="Optional guard on total angle points.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.confirm_slow:
        raise SystemExit("Refusing to run dense Maker fringes without --confirm-slow.")

    plan = build_plan()
    selected = plan["cases"]
    if args.case_id:
        wanted = set(args.case_id)
        selected = [case for case in selected if case["id"] in wanted]
        missing = wanted - {case["id"] for case in selected}
        if missing:
            raise SystemExit(f"Unknown fine-sampling case id(s): {sorted(missing)}")

    total_points = sum(case["theta_count"] for case in selected)
    if args.max_points is not None and total_points > args.max_points:
        raise SystemExit(f"Selected fine-sampling grid has {total_points} points, above --max-points={args.max_points}.")

    source_cases = {case["id"]: case for case in build_cases()}
    records = []
    start = time.perf_counter()
    for fine_case in selected:
        source = dict(source_cases[fine_case["source_case_id"]])
        source["id"] = fine_case["id"]
        source["theta_deg"] = fine_case["theta_deg"]
        record = run_case(source)
        record["fine_sampling"] = {
            "source_case_id": fine_case["source_case_id"],
            "theta_step_deg": fine_case["theta_step_deg"],
            "expected_residual_behavior": fine_case["expected_residual_behavior"],
        }
        records.append(record)
    runtime_seconds = time.perf_counter() - start

    payload = {
        "status": "python_slow_path_maker_fringes_fine_sampling_not_mathematica_validation",
        "plan_version": plan["plan_version"],
        "case_count": len(records),
        "total_angle_count": total_points,
        "runtime_seconds": round(float(runtime_seconds), 6),
        "runtime_seconds_not_fingerprinted": True,
        "mathematica_reference_required_for_agreement_claim": True,
        "cases": records,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote slow Maker fringes fine-sampling output to {args.output}")


if __name__ == "__main__":
    main()
