"""0.1deg Maker-fringe live-Mathematica PILOT (rigor per Zu et al. npj 10,64 (2024)).

Reuses the EXACT validated pair the existing Maker benchmark uses -- SHAARP.py =
solve_multilayer_maker_fringes_sweep(case['system']); Mathematica = the SHAARP.ml engine on
case_to_record(case) -- but at a 0.1deg theta window (the paper's angular rigor) instead of the
current 0.25deg grid. A bounded window keeps the live Wolfram run tractable; agreement here
proves SHAARP.py == live Mathematica at 0.1deg sampling, extensible to the full grid.

Run: python benchmarks/run_maker_0p1_pilot.py
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

from benchmarks.generate_maker_fringes_benchmarks import build_cases
from benchmarks.generate_maker_mathematica_inputs import case_to_record
from benchmarks.generate_maker_fine_chunk_manifest import wolfram_chunk_script_text
from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

MR = ROOT / "benchmarks" / "mathematica_reference"
WOLFRAM = r"C:\Program Files\Wolfram Research\WolframScript\wolframscript"
THETA0, STEP, N = 20.0, 0.1, 100   # 0.1deg window: 20.0 .. 29.9 deg (broad fine-fringe range)


def main():
    case = build_cases()[0]
    window = [round(THETA0 + STEP * i, 4) for i in range(N)]
    print(f"case id={case['id']}  azimuth={case['sample_azimuth_deg']}  window={window[0]}..{window[-1]} @ {STEP} deg ({N} pts)")

    # --- SHAARP.py side: the SAME call the validated benchmark uses (run_case), at 0.1deg ---
    # NOTE: run_case uses mu=1, eps0=1 (NOT the MU0/EPS0 sweep defaults). Mixing them rescales
    # the inhomogeneous SHG by MU0*EPS0=1/c^2 -> a ~300x false mismatch. Must match the engine.
    sweep = solve_multilayer_maker_fringes_sweep(case["system"], theta_deg=window, mrassumption=0, mu=1.0, eps0=1.0)
    py_para = np.asarray(sweep.parallel_intensity)
    py_perp = np.asarray(sweep.perpendicular_intensity)

    # --- Mathematica input: case_to_record at the 0.1deg window ---
    rec_case = dict(case); rec_case["theta_deg"] = window
    rec = case_to_record(rec_case)
    inp = {"benchmark_version": 1, "source": "0.1deg Maker pilot", "status": "python_fine_sampling_chunk_input_manifest_not_mathematica_validation",
           "case_count": 1, "total_angle_count": N, "chunk": {"case_id": case["id"], "start": 0, "count": N, "stop": N}, "cases": [rec]}
    in_path = MR / "maker_0p1_pilot_input_v1.json"
    out_path = MR / "maker_0p1_pilot_reference_v1.json"
    diag_path = MR / "maker_0p1_pilot_diagnostics.txt"
    in_path.write_text(json.dumps(inp, indent=2), encoding="utf-8")
    script = MR / "export_maker_0p1_pilot_reference.wl"
    script.write_text(wolfram_chunk_script_text(input_path=str(in_path).replace("\\", "/"),
                                                output_path=str(out_path).replace("\\", "/"),
                                                diagnostic_path=str(diag_path).replace("\\", "/")), encoding="utf-8")

    # --- run live Wolfram (reuse existing reference if present) ---
    if out_path.exists():
        print("reusing existing live-Wolfram reference:", out_path.name)
    else:
        print("running wolframscript (live SHAARP.ml Maker export at 0.1deg)...")
        r = subprocess.run([WOLFRAM, "-file", str(script)], capture_output=True, text=True, timeout=1800)
        print("wolfram rc=", r.returncode, "| stderr tail:", (r.stderr or "")[-300:])
        if not out_path.exists():
            print("NO OUTPUT JSON -- wolfram export failed; stdout tail:", (r.stdout or "")[-500:]); return

    # --- compare MFList (theta, para, perp) to SHAARP.py ---
    mma = json.loads(out_path.read_text(encoding="utf-8"))
    def find_mflist(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "MFList": return v
                f = find_mflist(v)
                if f is not None: return f
        if isinstance(o, list):
            for x in o:
                f = find_mflist(x)
                if f is not None: return f
        return None
    mf = find_mflist(mma)
    def cval(x): return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)
    mma_para = np.array([cval(row[1]) for row in mf]); mma_perp = np.array([cval(row[2]) for row in mf])
    para_err = float(np.max(np.abs(mma_para - py_para))); perp_err = float(np.max(np.abs(mma_perp - py_perp)))
    scale = max(float(np.max(np.abs(mma_para))), float(np.max(np.abs(mma_perp))), 1e-300)
    print(f"0.1deg agreement vs live Mathematica:  max|para err|={para_err:.2e}  max|perp err|={perp_err:.2e}  (scale={scale:.2e}, rel={max(para_err,perp_err)/scale:.2e})")
    json.dump({"status": "maker_0p1_pilot", "window_deg": window, "max_para_err": para_err, "max_perp_err": perp_err,
               "scale": scale, "passed": bool(max(para_err, perp_err) < 1e-9 + 1e-6 * scale)},
              open(MR / "maker_0p1_pilot_comparison_summary_v1.json", "w", encoding="utf-8"), indent=2)
    print("PASS" if max(para_err, perp_err) < 1e-9 + 1e-6 * scale else "CHECK")


if __name__ == "__main__":
    main()
