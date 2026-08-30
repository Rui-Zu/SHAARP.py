"""Validate the SHAARP.ml docs' canonical Maker example (Z-cut quartz 121.2 um + Au 13.9 nm,
lambda=0.8 um, p-in/p-out) point-by-point against LIVE SHAARP.ml (wolframscript), over a
fine-fringe window at the paper's 0.1 deg sampling.

Same pipeline as verify_linbo3_swap_vs_mathematica.py: case_to_record -> SHAARP.ml engine ->
MFList, compared to the Python Maker sweep (natural units mu=1, eps0=1, mrassumption=0/Full).
Reuses the exported reference JSON when present (compare-only re-runs).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_maker_fine_chunk_manifest import wolfram_chunk_script_text
from benchmarks.generate_maker_mathematica_inputs import case_to_record
from benchmarks.quartz_au_docs_case import build_quartz_au_case, build_quartz_au_system
from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

MR = ROOT / "benchmarks" / "mathematica_reference"
WOLFRAM = r"C:\Program Files\Wolfram Research\WolframScript\wolframscript"


def main():
    window = [round(20.0 + 0.1 * i, 4) for i in range(101)]  # 20.0 .. 30.0 deg, 0.1 deg (paper rigor)
    case = build_quartz_au_case(window)
    sysm = build_quartz_au_system()

    sw = solve_multilayer_maker_fringes_sweep(sysm, theta_deg=window, mrassumption=0, mu=1.0, eps0=1.0)
    py_par = np.abs(np.asarray(sw.parallel_intensity))
    py_perp = np.abs(np.asarray(sw.perpendicular_intensity))

    rec = case_to_record(case)
    inp = {"benchmark_version": 1, "source": "SHAARP.ml docs quartz+Au Maker example", "status": "python_input_manifest",
           "case_count": 1, "total_angle_count": len(window),
           "chunk": {"case_id": case["id"], "start": 0, "count": len(window), "stop": len(window)}, "cases": [rec]}
    in_path = MR / "quartz_au_docs_input.json"
    out_path = MR / "quartz_au_docs_reference.json"
    diag = MR / "quartz_au_docs_diag.txt"
    in_path.write_text(json.dumps(inp, indent=2), encoding="utf-8")
    script = MR / "export_quartz_au_docs.wl"
    script.write_text(wolfram_chunk_script_text(input_path=str(in_path).replace("\\", "/"),
                                                output_path=str(out_path).replace("\\", "/"),
                                                diagnostic_path=str(diag).replace("\\", "/")), encoding="utf-8")
    if not out_path.exists():
        print("running live wolframscript (SHAARP.ml) on the quartz+Au docs window (101 angles)...")
        r = subprocess.run([WOLFRAM, "-file", str(script)], capture_output=True, text=True, timeout=3600)
        print("wolfram rc=", r.returncode, "| stderr tail:", (r.stderr or "")[-400:])
        if not out_path.exists():
            print("NO OUTPUT -- stdout tail:", (r.stdout or "")[-600:])
            return

    mma = json.loads(out_path.read_text(encoding="utf-8"))

    def find_mflist(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "MFList":
                    return v
                f = find_mflist(v)
                if f is not None:
                    return f
        if isinstance(o, list):
            for x in o:
                f = find_mflist(x)
                if f is not None:
                    return f
        return None

    def cval(x):
        return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)

    mf = find_mflist(mma)
    mma_par = np.abs([cval(row[1]) for row in mf])
    mma_perp = np.abs([cval(row[2]) for row in mf])
    err_par = float(np.max(np.abs(mma_par - py_par)))
    err_perp = float(np.max(np.abs(mma_perp - py_perp)))
    rel = err_par / max(float(np.max(mma_par)), 1e-300)
    print(f"window 20-30 deg @0.1 ({len(window)} angles)")
    print(f"max|SHAARP.ml - py| parallel = {err_par:.3e}  (rel {rel:.3e})")
    print(f"max|SHAARP.ml - py| perp     = {err_perp:.3e}")
    print(f"SHAARP.ml parallel range [{mma_par.min():.4e}, {mma_par.max():.4e}], exact-zeros: {int(np.sum(mma_par == 0.0))}")
    print(f"python    parallel range [{py_par.min():.4e}, {py_par.max():.4e}], exact-zeros: {int(np.sum(py_par == 0.0))}")


if __name__ == "__main__":
    main()
