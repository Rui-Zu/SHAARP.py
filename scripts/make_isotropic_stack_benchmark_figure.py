"""Render the isotropic-stack benchmark overlay for the docs.

    python scripts/make_isotropic_stack_benchmark_figure.py

Writes ``docs/_static/isotropic_stack_benchmark.png``: SHAARP.py against ``tmm`` and ``inkstone``
on the Fabry-Perot fringe case, with a residual panel underneath. A raster, so it can be opened and
checked by eye before it ships -- never a live chart object.

Needs the ``benchmark`` extra only if the committed fixture is missing; by default it reads
``benchmarks/isotropic_stack_reference_v1.json`` and computes just the SHAARP leg.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_isotropic_stack_reference import load_reference, reference_curves  # noqa: E402
from benchmarks.isotropic_stack_cases import build_cases, shaarp_rt  # noqa: E402

CASE_ID = "single_film_spectral_sweep"
OUTPUT = ROOT / "docs" / "_static" / "isotropic_stack_benchmark.png"

NAVY = "#1f3b73"
ORANGE = "#e08214"
TEAL = "#2a9d8f"


def main() -> int:
    reference = load_reference()
    payload = {c["case_id"]: c for c in reference["cases"]}[CASE_ID]
    case = {c.case_id: c for c in build_cases()}[CASE_ID]

    grid = np.asarray(payload["grid"], dtype=float) * 1000.0  # um -> nm for the axis
    tmm_curves = reference_curves(payload, leg="tmm")
    ink_curves = reference_curves(payload, leg="inkstone")
    shaarp_curves = shaarp_rt(case)

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7.4, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.12},
    )

    ax.plot(grid, tmm_curves["R_s"], color=NAVY, lw=3.2, alpha=0.30, label="tmm (transfer matrix)")
    ax.plot(grid, ink_curves["R_s"], color=TEAL, lw=1.8, ls="--", label="inkstone (RCWA)")
    ax.plot(grid, shaarp_curves["R_s"], color=ORANGE, lw=1.1, label="SHAARP.py")
    ax.plot(grid, tmm_curves["T_s"], color=NAVY, lw=3.2, alpha=0.30)
    ax.plot(grid, ink_curves["T_s"], color=TEAL, lw=1.8, ls="--")
    ax.plot(grid, shaarp_curves["T_s"], color=ORANGE, lw=1.1)

    ax.annotate("$T$", xy=(grid[8], tmm_curves["T_s"][8]), xytext=(0, 8),
                textcoords="offset points", fontsize=12, color=NAVY, ha="center")
    ax.annotate("$R$", xy=(grid[8], tmm_curves["R_s"][8]), xytext=(0, -16),
                textcoords="offset points", fontsize=12, color=NAVY, ha="center")

    ax.set_ylabel("power coefficient")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("air / $n$=2.35 $\\times$ 0.5 µm / $n$=1.52, normal incidence, $s$-polarized",
                 fontsize=10.5)
    ax.legend(loc="center right", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.25, lw=0.6)

    res_r = np.abs(shaarp_curves["R_s"] - tmm_curves["R_s"])
    res_t = np.abs(shaarp_curves["T_s"] - tmm_curves["T_s"])
    floor = 1e-17
    ax_res.semilogy(grid, np.maximum(res_r, floor), color=ORANGE, lw=1.0, label="|$\\Delta R$| vs tmm")
    ax_res.semilogy(grid, np.maximum(res_t, floor), color=NAVY, lw=1.0, label="|$\\Delta T$| vs tmm")
    ax_res.axhline(1e-12, color="0.35", ls=":", lw=1.0)
    ax_res.annotate("test tolerance $10^{-12}$", xy=(grid[2], 1e-12), xytext=(0, 5),
                    textcoords="offset points", fontsize=8, color="0.30")
    ax_res.set_xlabel("wavelength (nm)")
    ax_res.set_ylabel("|residual|")
    ax_res.set_ylim(1e-17, 1e-9)
    ax_res.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)
    ax_res.grid(alpha=0.25, lw=0.6, which="both")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print("wrote %s" % OUTPUT.relative_to(ROOT).as_posix())
    print("  max |dR| vs tmm = %.3e" % float(res_r.max()))
    print("  max |dT| vs tmm = %.3e" % float(res_t.max()))
    print("  points          = %d" % grid.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
