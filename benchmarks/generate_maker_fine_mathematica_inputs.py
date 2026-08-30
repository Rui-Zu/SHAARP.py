from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_maker_fringes_benchmarks import build_cases  # noqa: E402
from benchmarks.generate_maker_fringes_fine_sampling_plan import build_plan  # noqa: E402
from benchmarks.generate_maker_mathematica_inputs import BENCHMARK_VERSION, case_to_record  # noqa: E402


DEFAULT_OUTPUT = ROOT / "benchmarks" / "mathematica_reference" / "maker_fine_mathematica_inputs_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dense Maker fine-sampling cases for Mathematica MF generation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-id", action="append", help="Optional fine-sampling case id. May be repeated.")
    args = parser.parse_args()

    payload = write_fine_input_payload(args.output, case_ids=args.case_id)
    print(f"Wrote {payload['case_count']} fine Maker Mathematica input cases to {args.output}")


def write_fine_input_payload(path: Path, *, case_ids: list[str] | None = None) -> dict:
    payload = build_fine_input_payload(case_ids=case_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_fine_input_payload(*, case_ids: list[str] | None = None) -> dict:
    source_cases = {case["id"]: case for case in build_cases()}
    selected = build_plan()["cases"]
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case["id"] in wanted]
        missing = wanted - {case["id"] for case in selected}
        if missing:
            raise ValueError(f"Unknown fine-sampling case id(s): {sorted(missing)}")

    records = []
    for fine_case in selected:
        source = dict(source_cases[fine_case["source_case_id"]])
        source["id"] = fine_case["id"]
        source["theta_deg"] = fine_case["theta_deg"]
        record = case_to_record(source)
        record["source_case_id"] = fine_case["source_case_id"]
        record["fine_sampling"] = {
            "theta_step_deg": fine_case["theta_step_deg"],
            "theta_count": fine_case["theta_count"],
            "expected_residual_behavior": fine_case["expected_residual_behavior"],
        }
        records.append(record)

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "source": "Python explicit fine-sampling Maker input manifest for Mathematica SHAARP.ml MF export",
        "status": "python_fine_sampling_input_manifest_not_mathematica_validation",
        "case_count": len(records),
        "total_angle_count": sum(len(case["theta_deg"]) for case in records),
        "notes": [
            "These are dense-grid inputs only, not validation results.",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
            "Fine-grid Mathematica runs can be slow; run selected cases before all cases if needed.",
        ],
        "cases": records,
    }


if __name__ == "__main__":
    main()
