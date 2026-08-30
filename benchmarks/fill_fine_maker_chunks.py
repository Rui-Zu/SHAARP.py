"""Tile the remaining fine-Maker angles into chunks and export live Wolfram references.

Computes a complete non-overlapping partition of each fine-Maker case that abuts
the already-exported count=5 chunks, generates the matched input/python/script
files for every missing segment, runs the live Wolfram 14.3 export for each, and
verifies each reference against the matching Python slice. Writes a machine
spec at benchmarks/.fill_chunks_spec.json for downstream registration.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.generate_maker_fine_chunk_manifest import (
    build_chunk_pair,
    wolfram_chunk_script_text,
    wolfram_path_expression,
)
from benchmarks.run_wolfram_export_script import run_wolfram_export_script
from benchmarks.compare_maker_fringes_reference import compare_maker_reference_payloads


ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "benchmarks" / "mathematica_reference"
WOLFRAM = Path(r"C:\Program Files\Wolfram Research\Wolfram\14.3\WolframKernel.exe")
MAX_COUNT = 60

# Already-exported count=5 chunks (start indices) per case.
OCCUPIED = {
    "fine_maker_negative_normal_positive": [4, 24, 48, 70, 92],
    "fine_maker_high_oblique": [0, 40, 80, 120, 156],
    "fine_maker_fringe_resolution": [0, 120, 240, 360, 476],
}
SHORT = {
    "fine_maker_negative_normal_positive": "neg",
    "fine_maker_high_oblique": "hob",
    "fine_maker_fringe_resolution": "frng",
}


def _wp(path: Path) -> str:
    return str(path).replace("\\", "/")


def plan_fill_specs(case_lengths: dict[str, int]) -> list[tuple[str, int, int, str]]:
    specs: list[tuple[str, int, int, str]] = []
    for case, n in case_lengths.items():
        covered: set[int] = set()
        for start in OCCUPIED[case]:
            covered.update(range(start, start + 5))
        gaps: list[tuple[int, int]] = []
        idx = 0
        while idx < n:
            if idx in covered:
                idx += 1
                continue
            j = idx
            while j < n and j not in covered:
                j += 1
            gaps.append((idx, j - idx))
            idx = j
        for gap_start, gap_count in gaps:
            offset = 0
            while offset < gap_count:
                count = min(MAX_COUNT, gap_count - offset)
                start = gap_start + offset
                slug = f"fill_{SHORT[case]}_{start}_{count}"
                specs.append((case, start, count, slug))
                offset += count
    return specs


def main() -> None:
    inp = json.loads((MR / "maker_fine_mathematica_inputs_v1.json").read_text(encoding="utf-8"))
    py = json.loads((ROOT / "benchmarks" / "maker_fringes_fine_sampling_output_v1.json").read_text(encoding="utf-8"))
    case_lengths = {case["id"]: len(case["theta_deg"]) for case in inp["cases"]}

    specs = plan_fill_specs(case_lengths)
    print(f"Planned {len(specs)} fill chunks covering {sum(s[2] for s in specs)} angles", flush=True)

    results = []
    for case, start, count, slug in specs:
        input_chunk, python_chunk = build_chunk_pair(inp, py, case_id=case, start=start, count=count)
        input_out = MR / f"maker_fine_chunk_{slug}_input_v1.json"
        python_out = ROOT / "benchmarks" / f"maker_fine_chunk_{slug}_python_output_v1.json"
        script_out = MR / f"export_maker_fine_chunk_{slug}_reference.wl"
        ref_out = MR / f"maker_fine_chunk_{slug}_reference_v1.json"
        diag_out = MR / f"maker_fine_chunk_{slug}_diagnostics.txt"

        input_out.write_text(json.dumps(input_chunk, indent=2), encoding="utf-8")
        python_out.write_text(json.dumps(python_chunk, indent=2), encoding="utf-8")
        script_out.write_text(
            wolfram_chunk_script_text(
                input_path=_wp(input_out.resolve()),
                output_path=_wp(ref_out.resolve()),
                diagnostic_path=_wp(diag_out.resolve()),
            ),
            encoding="utf-8",
        )

        run = run_wolfram_export_script(
            script_out,
            ref_out,
            executable=WOLFRAM,
            original_export_expression=wolfram_path_expression(ref_out.resolve()),
            timeout=1800.0,
            temp_root=MR / ".wolfram_tmp",
            output_wait=5.0,
        )
        if not run.output_created:
            print(f"FAIL export {slug} (rc={run.returncode})", flush=True)
            results.append({"slug": slug, "case": case, "start": start, "count": count, "ok": False})
            continue

        reference = json.loads(ref_out.read_text(encoding="utf-8"))
        python_payload = json.loads(python_out.read_text(encoding="utf-8"))
        comparisons = compare_maker_reference_payloads(reference, python_payload, atol=1e-10, rtol=1e-10)
        failed = [c for c in comparisons if not c.passed]
        max_abs = max((c.max_abs_error for c in comparisons), default=0.0)
        ok = not failed
        print(f"{'OK ' if ok else 'BAD'} {slug} start={start} count={count} failed={len(failed)} max_abs={max_abs}", flush=True)
        results.append({"slug": slug, "case": case, "start": start, "count": count, "ok": ok})

    (ROOT / "benchmarks" / ".fill_chunks_spec.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"DONE {sum(1 for r in results if r['ok'])}/{len(results)} ok", flush=True)


if __name__ == "__main__":
    main()
