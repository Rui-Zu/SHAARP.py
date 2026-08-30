"""Generate shaarp/casestudy_dispersion.json from the Mathematica-exported raw grids.

The raw grid (benchmarks/mathematica_reference/casestudy_dispersion_raw.json) was produced by
EVALUATING each original SHAARP.ml setup.nb material function across a wavelength grid in a live
Mathematica kernel (wolframscript) and exporting its returned crystal-physics-frame dielectric
tensors at omega and 2omega, the (wavelength-independent) SHG d tensor, lattice, point group,
Miller indices, and orientation matrix. NO hand-transcription of optical constants.

This produces a NORMALIZED, display-name-keyed JSON shipped as package data
(shaarp/casestudy_dispersion.json); shaarp/casestudy_materials.py loads it and interpolates the
dielectric tensors at the user's wavelength, reproducing the original GUI's behaviour that the
case-study dielectric tensors update with the fundamental wavelength.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks" / "mathematica_reference" / "casestudy_dispersion_raw.json"
OUT = ROOT / "shaarp" / "casestudy_dispersion.json"

# raw setup.nb symbol key -> (display name, native/standard wavelength um)
DISPLAY = [
    ("Air", "Air", 1.064),
    ("LiNbO3", "LiNbO3 z-cut (1064 nm)", 1.064),
    ("LiNbO3zCutλ1550", "LiNbO3 z-cut (1550 nm)", 1.55),
    ("LiNbO3xCutλ1550", "LiNbO3 x-cut (1550 nm)", 1.55),
    ("KTPxCut", "KTP x-cut", 1.064),
    ("KTPyCut", "KTP y-cut", 1.064),
    ("Quartz$zCutλ800", "Quartz z-cut (800 nm)", 0.8),
    ("Quartz$xCutλ1064", "Quartz x-cut (1064 nm)", 1.064),
    ("ZnO001", "ZnO (001)", 1.064),
    ("GaAs111λ800", "GaAs (111) (800 nm)", 0.8),
    ("BTO", "BaTiO3", 1.064),
    ("MoS2", "MoS2", 1.064),
    ("Al2O3$0001λ1550", "Al2O3 (0001) (1550 nm)", 1.55),
    ("Pt111λ1550", "Pt (111) (1550 nm)", 1.55),
    ("Auλ800", "Au coating (800 nm)", 0.8),
    ("blankMaterL", "Blank linear", 1.064),
    ("blankMaterNL", "Blank nonlinear", 1.064),
]


def _norm_pg(pg: str) -> str:
    return pg.replace("_\n", "-").strip()


def _clean_grid(grid, *re_im_pairs):
    """Drop wavelength points where any tensor entry failed to evaluate (Null in the raw export)."""
    keep = []
    for i in range(len(grid)):
        ok = True
        for arr in re_im_pairs:
            blk = arr[i]
            if blk is None or any(v is None for row in blk for v in row):
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def build():
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    out = {"order": [], "materials": {}}
    for key, disp, native in DISPLAY:
        m = raw[key]
        grid = m["grid"]
        keep = _clean_grid(grid, m["epsW_re"], m["epsW_im"], m["eps2W_re"], m["eps2W_im"])
        g = [grid[i] for i in keep]

        def pick(arr):
            return [arr[i] for i in keep]

        out["order"].append(disp)
        out["materials"][disp] = {
            "name": m["name"], "source_key": key, "native_lambda_um": native,
            "point_group": _norm_pg(m["pg"]), "lattice": [float(x) for x in m["lattice"]],
            "miller": m["miller"], "orientation": m["orientation"],
            "grid_um": g,
            "epsW_re": pick(m["epsW_re"]), "epsW_im": pick(m["epsW_im"]),
            "eps2W_re": pick(m["eps2W_re"]), "eps2W_im": pick(m["eps2W_im"]),
            "d_re": m["d_re"], "d_im": m["d_im"],
        }
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) with {len(out['order'])} materials")


if __name__ == "__main__":
    build()
