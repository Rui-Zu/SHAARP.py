from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
REFERENCE_DIR = BENCHMARK_DIR / "mathematica_reference"
DEFAULT_MANIFEST = REFERENCE_DIR / "reference_manifest_v1.json"
DEFAULT_OUTPUT = REFERENCE_DIR / "reference_template_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fillable Mathematica reference JSON template.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    template = build_template(json.loads(args.manifest.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"Wrote Mathematica reference template to {args.output}")


def build_template(manifest: dict) -> dict:
    suites = []
    for suite in manifest["suites"]:
        source_payload = _load_source(suite["source_benchmark_file"])
        source_items = {item["id"]: item for item in source_payload[suite["source_key"]]}
        suites.append(
            {
                "suite_id": suite["suite_id"],
                "source_benchmark_file": suite["source_benchmark_file"],
                "item_count": suite["item_count"],
                "requested_outputs": suite["requested_outputs"],
                "items": [
                    {
                        "id": item_id,
                        "inputs": source_items[item_id].get("inputs", {}),
                        "mathematica_outputs": {key: None for key in suite["requested_outputs"]},
                    }
                    for item_id in suite["item_ids"]
                ],
            }
        )
    return {
        "template_version": TEMPLATE_VERSION,
        "status": "fillable_mathematica_reference_template_no_values_yet",
        "manifest_version": manifest["manifest_version"],
        "instructions": [
            "Fill each mathematica_outputs value with numeric data exported from Mathematica SHAARP.",
            "Keep suite_id and item id values unchanged.",
            "Use {'real': x, 'imag': y} objects for complex numbers.",
            "After filling values, compare against Python benchmark outputs with the comparison tooling.",
        ],
        "suite_count": len(suites),
        "total_items": sum(suite["item_count"] for suite in suites),
        "suites": suites,
    }


def _load_source(source_file: str) -> dict:
    return json.loads((BENCHMARK_DIR / source_file).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
