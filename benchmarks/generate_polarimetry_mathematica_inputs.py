"""Build the SHAARP.ml Mathematica input manifest for the polarimetry-sweep validation.

A polarimetry point is a single-azimuth SampleRotate evaluation at a fixed
(theta, phi, psi, ellipticity): the original SHAARP.ml SampleRotate machinery
computes the reflected/transmitted 2-omega field and projects it onto the
analyzer. We therefore reuse the SampleRotate input format (per-layer eps / d /
orientation / qc via ``case_to_record``) but with a single-element rotation grid
equal to each case's sample azimuth.

This file ONLY produces the Mathematica INPUT manifest (the case inputs). The
Wolfram exporter computes the reference outputs from SHAARP.ml; it must not copy
Python output values. Run: python -m benchmarks.generate_polarimetry_mathematica_inputs
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path

from benchmarks.generate_maker_mathematica_inputs import case_to_record
from benchmarks.generate_multilayer_polarimetry_benchmarks import build_cases

TAG = "v1"
BENCHMARK_VERSION = 1


def build_input_cases() -> list[dict]:
    cases = build_cases()
    out = []
    for c in cases:
        # case_to_record is maker-oriented (expects an iterable theta grid); wrap the
        # scalar polarimetry theta, then restore a scalar theta for SampleRotate and
        # attach the single-azimuth rotation grid.
        rec = case_to_record({**c, "theta_deg": [c["theta_deg"]]})
        rec["theta_deg"] = float(c["theta_deg"])
        # build_cases already bakes sample_azimuth into each layer's orientation
        # (with_sample_azimuth_deg), so SampleRotate must NOT rotate again: use a
        # single-element zero rotation grid (evaluate at the already-oriented crystal).
        rec["sample_rotation_deg"] = [0.0]
        rec["id"] = c["id"]
        rec["source_case_id"] = c["id"]
        rec["polarimetry_label"] = c.get("polarimetry_label")
        rec["orientation_index"] = c.get("orientation_index")
        out.append(rec)
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate the SHAARP.ml Mathematica input manifest for the polarimetry-sweep validation."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "mathematica_reference" / f"polarimetry_mathematica_inputs_{TAG}.json",
    )
    args = parser.parse_args()

    input_cases = build_input_cases()
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "tag": TAG,
        "source": "Python explicit polarimetry input manifest for Mathematica SHAARP.ml SampleRotate export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(input_cases),
        "notes": [
            "A polarimetry point = SampleRotate at a single azimuth with fixed (theta, phi, psi, ellipticity).",
            "The Wolfram exporter computes reflected/transmitted analyzer outputs from SHAARP.ml; it must NOT copy Python values.",
            "Each case's sample_rotation_deg is a single-element grid equal to its sample_azimuth_deg.",
        ],
        "cases": input_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(input_cases)} polarimetry input cases to {args.output}")

    # Optionally mirror the manifest into a second working tree (e.g. a checkout alongside the
    # original Mathematica package), so an exporter run there picks it up. Opt in by setting
    # SHAARP_ORIGINAL_REF_DIR; unset, this script writes only inside its own repository.
    mirror_root = os.environ.get("SHAARP_ORIGINAL_REF_DIR")
    if mirror_root:
        original = Path(mirror_root) / f"polarimetry_mathematica_inputs_{TAG}.json"
        if original.parent.exists():
            original.write_text(text, encoding="utf-8")
            print(f"Mirrored input manifest to {original}")


if __name__ == "__main__":
    main()
