"""Validate the Python FMR backward / standing-wave source policies against LIVE SHAARP.ml.

The original FMR mode (mrassumption=0) has three winhAssumption sub-variants: 0 = no backward
waves, 1 = backward (no standing), 2 = backward + standing. The Python multilayer solver exposes
these as inhomogeneous_source_policy = forward_only / forward_backward / all. This script proves
that correspondence by driving the SAME quartz+Au docs system through the live SHAARP.ml MF
function (via export_maker_fringes_reference.wl, which now maps winhAssumption -> flagBackward /
flagStandingWave) at winhAssumption 0/1/2, and comparing to the Python policies.

SANITY GATE: winhAssumption=0 must reproduce the Python forward_only result (the already-validated
docs case to ~1e-9). If that holds, the harness is trustworthy and the winh=1/2 comparisons are
meaningful. Run: python -m benchmarks.verify_fmr_backward_vs_mathematica
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.generate_maker_mathematica_inputs import case_to_record
from benchmarks.quartz_au_docs_case import build_quartz_au_case, build_quartz_au_system
from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

MR = Path(__file__).resolve().parent / "mathematica_reference"
EXPORTER = MR / "export_maker_fringes_reference.wl"
WOLFRAM = r"C:\Program Files\Wolfram Research\WolframScript\wolframscript"

WINH_TO_POLICY = {0: "forward_only", 1: "forward_backward", 2: "all"}


def _chunk_script(input_path: str, output_path: str, diag_path: str) -> str:
    # like wolfram_chunk_script_text but Get's THIS tree's (winh-aware) exporter
    return "\n".join([
        'BeginPackage["SHAARPPathOverrides`"];',
        f'MakerInputPath = "{input_path}";',
        f'MakerOutputPath = "{output_path}";',
        f'MakerDiagnosticPath = "{diag_path}";',
        "EndPackage[];",
        f'Get["{str(EXPORTER).replace(chr(92), "/")}"];',
        "",
    ])


def _find_mflist(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "MFList":
                return v
            f = _find_mflist(v)
            if f is not None:
                return f
    if isinstance(o, list):
        for x in o:
            f = _find_mflist(x)
            if f is not None:
                return f
    return None


def _cval(x):
    return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)


def run_winh(window, winh: int):
    case = build_quartz_au_case(window)
    rec = case_to_record(case)
    rec["mrassumption"] = 0
    rec["winhAssumption"] = winh
    inp = {"benchmark_version": 1, "source": "FMR winh validation", "status": "python_input_manifest",
           "case_count": 1, "total_angle_count": len(window),
           "chunk": {"case_id": rec["id"], "start": 0, "count": len(window), "stop": len(window)},
           "cases": [rec]}
    in_path = MR / f"fmr_winh{winh}_input.json"
    out_path = MR / f"fmr_winh{winh}_reference.json"
    diag = MR / f"fmr_winh{winh}_diag.txt"
    in_path.write_text(json.dumps(inp, indent=2), encoding="utf-8")
    script = MR / f"export_fmr_winh{winh}.wl"
    script.write_text(_chunk_script(str(in_path).replace("\\", "/"),
                                    str(out_path).replace("\\", "/"),
                                    str(diag).replace("\\", "/")), encoding="utf-8")
    r = subprocess.run([WOLFRAM, "-file", str(script)], capture_output=True, text=True, timeout=3600)
    if not out_path.exists():
        print(f"  winh={winh}: NO OUTPUT rc={r.returncode} stderr={r.stderr[-300:]}")
        return None
    mf = _find_mflist(json.loads(out_path.read_text(encoding="utf-8")))
    return np.abs([_cval(row[1]) for row in mf])  # parallel intensity


def main():
    window = [round(20.0 + 1.0 * i, 4) for i in range(11)]  # 20..30 deg, 1 deg, 11 pts (fast)
    sysm = build_quartz_au_system()
    py = {}
    for winh, pol in WINH_TO_POLICY.items():
        sw = solve_multilayer_maker_fringes_sweep(sysm, theta_deg=window, mrassumption=0,
                                                  inhomogeneous_source_policy=pol, mu=1.0, eps0=1.0)
        py[winh] = np.abs(np.asarray(sw.parallel_intensity))
    print(f"FMR winh validation, quartz+Au docs, {len(window)} angles (20-30 deg)")
    results = {}
    for winh, pol in WINH_TO_POLICY.items():
        mma = run_winh(window, winh)
        if mma is None:
            continue
        err = float(np.max(np.abs(mma - py[winh])))
        rel = err / max(float(np.max(mma)), 1e-300)
        tag = "SANITY" if winh == 0 else "compare"
        print(f"  [{tag}] winh={winh} ({pol:16s}): max|SHAARP.ml - py| = {err:.3e}  (rel {rel:.3e})")
        results[winh] = {"policy": pol, "max_abs_err": err, "rel_err": rel}
    (MR / "fmr_winh_validation_summary.json").write_text(
        json.dumps({"window_deg": window, "results": results,
                    "winh_to_policy": WINH_TO_POLICY}, indent=2), encoding="utf-8")
    print("summary ->", MR / "fmr_winh_validation_summary.json")


if __name__ == "__main__":
    main()
