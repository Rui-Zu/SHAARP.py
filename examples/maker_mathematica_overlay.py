"""Visual proof of the 0.1-degree Maker-fringe agreement vs LIVE Mathematica (SHAARP.ml).

Overlays the SHAARP.py transmitted-SHG Maker intensities (lines) on the live-Wolfram
SHAARP.ml MFList points (markers) at 0.1-degree incidence-angle sampling, plus a residual
panel showing the per-angle |SHAARP.py - Mathematica| (machine precision, ~1e-15). Reads the
pilot reference produced by `benchmarks/run_maker_0p1_pilot.py`.

Run: python examples/maker_mathematica_overlay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "benchmarks" / "mathematica_reference" / "maker_0p1_pilot_reference_v1.json"


def _find_mflist(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "MFList":
                return v
            f = _find_mflist(v)
            if f is not None:
                return f
    if isinstance(obj, list):
        for x in obj:
            f = _find_mflist(x)
            if f is not None:
                return f
    return None


# `python examples/<name>.py` puts examples/ on sys.path, NOT the repo root, so the
# `benchmarks` import inside compute() fails in a fresh clone -- which is exactly how
# this file's own docstring says to run it. Make the root importable first.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


def _c(x):
    return complex(x["real"], x["imag"]) if isinstance(x, dict) else complex(x)


def compute():
    from benchmarks.generate_maker_fringes_benchmarks import build_cases
    from shaarp.multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

    mf = _find_mflist(json.loads(REF.read_text(encoding="utf-8")))
    theta = [float(_c(r[0]).real) for r in mf]
    mma_para = np.abs(np.array([_c(r[1]) for r in mf]))
    mma_perp = np.abs(np.array([_c(r[2]) for r in mf]))
    sweep = solve_multilayer_maker_fringes_sweep(build_cases()[0]["system"], theta_deg=theta, mrassumption=0, mu=1.0, eps0=1.0)
    py_para = np.abs(np.asarray(sweep.parallel_intensity))
    py_perp = np.abs(np.asarray(sweep.perpendicular_intensity))
    return np.array(theta), mma_para, mma_perp, py_para, py_perp


def build_figure(data=None):
    import matplotlib.pyplot as plt

    theta, mma_para, mma_perp, py_para, py_perp = data or compute()
    res = np.maximum(np.abs(py_para - mma_para), np.abs(py_perp - mma_perp))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(theta, py_para, "-", color="navy", lw=2, label="SHAARP.py  |MF$_\\parallel$|")
    ax1.plot(theta[::3], mma_para[::3], "o", color="navy", mfc="none", ms=7, label="♯SHAARP.ml (Mathematica)")
    ax1.plot(theta, py_perp, "-", color="crimson", lw=2, label="SHAARP.py  |MF$_\\perp$|")
    ax1.plot(theta[::3], mma_perp[::3], "s", color="crimson", mfc="none", ms=6, label="♯SHAARP.ml (Mathematica)")
    ax1.set_xlabel(r"incidence angle $\theta_i$ (deg)")
    ax1.set_ylabel("transmitted SHG intensity")
    ax1.set_title("(a) 0.1° Maker scan: SHAARP.py (lines) on live Mathematica (markers)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.semilogy(theta, np.maximum(res, 1e-18), color="seagreen", lw=1.5)
    ax2.axhline(1e-12, color="grey", ls="--", lw=1, label="1e-12 gate")
    ax2.set_xlabel(r"incidence angle $\theta_i$ (deg)")
    ax2.set_ylabel(r"$|$SHAARP.py $-$ Mathematica$|$")
    ax2.set_title(f"(b) per-angle residual (max = {np.max(res):.1e})", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Maker fringes at 0.1° — SHAARP.py vs live ♯SHAARP.ml, machine-precision agreement", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, mp, mq, pp, pq = compute()
    print(f"angles={len(t)} range=[{t.min():.1f},{t.max():.1f}] step={np.median(np.diff(t)):.2f}deg "
          f"max|resid|={np.max(np.maximum(np.abs(pp-mp), np.abs(pq-mq))):.2e}")
    build_figure((t, mp, mq, pp, pq))
    plt.show()
