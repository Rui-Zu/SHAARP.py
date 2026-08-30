"""DECISIVE verification (request): does live SHAARP.ml (Mathematica) exhibit the same
transmitted-channel exact-0 'swap' that Python does for a LiNbO3 z-cut Maker sweep, and which
Python transmitted_wave_policy matches it at the swap angles?

Runs the SAME validated pipeline as run_maker_0p1_pilot.py (case_to_record -> SHAARP.ml engine ->
MFList) on the LiNbO3 single-film 10um config over a swap-angle window, and compares SHAARP.ml's
parallel/perp Maker intensity to BOTH Python policies (shaarp_ml_selected = mode[0]; physical_sum
= total transmitted field).
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_maker_fine_chunk_manifest import wolfram_chunk_script_text
from benchmarks.generate_maker_mathematica_inputs import case_to_record
from shaarp.layer_presets import build_stack_preset
from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

MR = ROOT / "benchmarks" / "mathematica_reference"
WOLFRAM = r"C:\Program Files\Wolfram Research\WolframScript\wolframscript"


def main():
    base = build_stack_preset("single_film_linbo3")
    layers = list(base.layers)
    layers[1] = replace(layers[1], thickness_um=10.0)
    sysm = replace(base, layers=layers)
    window = [round(0.90 + 0.01 * i, 4) for i in range(21)]  # 0.90..1.10 deg (covers a swap)

    case = {
        "id": "linbo3_swap_verify", "theta_deg": window, "phi_deg": 0.0, "psi_deg": 0.0,
        "ellipticity_deg": 0.0, "sample_azimuth_deg": 0.0, "system": sysm,
    }

    py = {}
    for pol in ("shaarp_ml_selected", "physical_sum"):
        sw = solve_multilayer_maker_fringes_sweep(sysm, theta_deg=window, mrassumption=0, mu=1.0, eps0=1.0,
                                                  transmitted_wave_policy=pol)
        py[pol] = np.abs(np.asarray(sw.parallel_intensity))

    rec = case_to_record(case)
    inp = {"benchmark_version": 1, "source": "LiNbO3 swap verification", "status": "python_input_manifest",
           "case_count": 1, "total_angle_count": len(window),
           "chunk": {"case_id": case["id"], "start": 0, "count": len(window), "stop": len(window)}, "cases": [rec]}
    in_path = MR / "linbo3_swap_input.json"
    out_path = MR / "linbo3_swap_reference.json"
    diag = MR / "linbo3_swap_diag.txt"
    in_path.write_text(json.dumps(inp, indent=2), encoding="utf-8")
    script = MR / "export_linbo3_swap.wl"
    script.write_text(wolfram_chunk_script_text(input_path=str(in_path).replace("\\", "/"),
                                                output_path=str(out_path).replace("\\", "/"),
                                                diagnostic_path=str(diag).replace("\\", "/")), encoding="utf-8")
    if not out_path.exists():
        print("running live wolframscript (SHAARP.ml) on LiNbO3 swap window...")
        r = subprocess.run([WOLFRAM, "-file", str(script)], capture_output=True, text=True, timeout=1800)
        print("wolfram rc=", r.returncode, "| stderr tail:", (r.stderr or "")[-400:])
        if not out_path.exists():
            print("NO OUTPUT -- stdout tail:", (r.stdout or "")[-600:]); return

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
    print("\n theta   SHAARP.ml   py_selected[mode0]  py_physical_sum")
    for i, t in enumerate(window):
        print(f" {t:5.2f}  {mma_par[i]:9.4f}   {py['shaarp_ml_selected'][i]:9.4f}        {py['physical_sum'][i]:9.4f}")
    print("\nmax|SHAARP.ml - py_selected|   =", float(np.max(np.abs(mma_par - py['shaarp_ml_selected']))))
    print("max|SHAARP.ml - py_physical_sum| =", float(np.max(np.abs(mma_par - py['physical_sum']))))
    print("SHAARP.ml exact-zeros in window:", int(np.sum(mma_par == 0.0)))


if __name__ == "__main__":
    main()
