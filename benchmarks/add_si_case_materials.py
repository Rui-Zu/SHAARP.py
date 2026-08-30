"""Add the SHAARP.si notebook's extra Case-Study materials to casestudy_dispersion.json.

Constants are transcribed VERBATIM from the SHAARP.si V1.04 notebook's Case-Study button
SetValue blocks (`SHAARP.si/Code/SHAARP_V1.04/SHAARP_V1.04.nb`, ~line 45095-45376). Each block sets
the full state tuple; ``LinearInput == 2`` means the dielectric is entered DIRECTLY as the complex
epsilon tensor (crystal-physics frame), so the epsW/eps2W below are epsilon (NOT refractive indices).
The d_ij Voigt rows (d1j/d2j/d3j) are the original's pm/V values. NO value is invented.

Orientation: materials with UseDirection==1 store the crystal-physics Z-axes-in-lab MATRIX directly
(cr11..cr33, row-major); TaAs uses UseDirection==2 with Miller indices, so its orientation is computed
with the package's validated CrystalOrientation.from_shaarp_miller_surface (same path the GUI uses).

The SI tool is single-wavelength; these epsilon are at the SI default fundamental lambda = 0.8 um
(GaAs(111)@1064nm is the labelled exception at 1.064 um). Single-grid -> the loader returns the fixed
epsilon at any requested wavelength (np.interp on one point), so the value is exact; only the displayed
default wavelength carries the SI default.

BiBO is intentionally NOT added here: its block carries non-zero phiw/phi2w (=-47.1/-45.38) and an
off-diagonal monoclinic epsilon, which require the SHAARP.si in-plane dielectric-rotation semantics
(not yet verified) -- deferred rather than guessed.

Run: python -m benchmarks.add_si_case_materials (idempotent; backs up the JSON once)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shaarp.config import CrystalOrientation, CrystalStructure  # noqa: E402

JSON_PATH = ROOT / "shaarp" / "casestudy_dispersion.json"

# (a,b,c,alpha,beta,gamma); point group; orientation spec; epsW 3x3; eps2W 3x3; d 3x6; native lambda.
# epsW/eps2W given as the 9 row-major values exactly as in the notebook; reshaped below.
SI_MATERIALS = [
    {
        "name": "TaAs (112)",
        "lattice": [3.4348, 3.4348, 11.641, 90, 90, 90],
        "point_group": "4mm",
        # UseDirection==2: Miller surface (1,1,2), in-plane [1,-1,0]
        "miller": [(1, 1, 2), (1, -1, 0)],
        "epsW": [6.2591 + 18.8118j, 0, 0, 0, 6.2591 + 18.8118j, 0, 0, 0, 15.2221 + 18.0536j],
        "eps2W": [3.6748 + 13.3883j, 0, 0, 0, 3.6748 + 13.3883j, 0, 0, 0, 3.4997 + 14.6893j],
        "d": [[0, 0, 0, 0, -92.2879, 0], [0, 0, 0, -92.2879, 0, 0], [21.3876, 21.3876, 746.8152, 0, 0, 0]],
        "lambda_um": 0.8,
    },
    {
        "name": "GaAs (111) (1064 nm)",
        "lattice": [5.65, 5.65, 5.65, 90, 90, 90],
        "point_group": "-43m",
        # UseDirection==1: orientation matrix given directly (the (111) Z-axes-in-lab)
        "orientation": [[-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)],
                        [0.0, -np.sqrt(2 / 3), 1 / np.sqrt(3)],
                        [1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)]],
        "epsW": [12.03, 0, 0, 0, 12.03, 0, 0, 0, 12.03],
        "eps2W": [16.94 + 2.73j, 0, 0, 0, 16.94 + 2.73j, 0, 0, 0, 16.94 + 2.73j],
        "d": [[0, 0, 0, 165 - 475j, 0, 0], [0, 0, 0, 0, 165 - 475j, 0], [0, 0, 0, 0, 0, 165 - 475j]],
        "lambda_um": 1.064,
    },
    {
        "name": "LiB3O5 (LBO)",
        "lattice": [8.75, 7.48, 5.01, 90, 90, 90],
        "point_group": "mm2",
        "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "epsW": [2.45, 0, 0, 0, 2.58, 0, 0, 0, 2.53],
        "eps2W": [2.49, 0, 0, 0, 2.63, 0, 0, 0, 2.57],
        "d": [[0, 0, 0, 0, -2.3, 0], [0, 0, 0, 2.52, 0, 0], [-2.51, 2.69, 0.51, 0, 0, 0]],
        "lambda_um": 0.8,
    },
    {
        "name": "KBBF",
        "lattice": [4.43, 4.43, 18.74, 90, 90, 120],
        "point_group": "32",
        "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "epsW": [2.17, 0, 0, 0, 2.17, 0, 0, 0, 1.94],
        "eps2W": [2.2, 0, 0, 0, 2.2, 0, 0, 0, 1.95],
        "d": [[0.49, -0.49, 0, 0, 0, 0], [0, 0, 0, 0, 0, -0.49], [0, 0, 0, 0, 0, 0]],
        "lambda_um": 0.8,
    },
    {
        "name": "LiOsO3",
        "lattice": [5.05, 5.05, 13.24, 90, 90, 120],
        "point_group": "3m",
        "orientation": [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
        "epsW": [1.7269 + 4.9604j, 0, 0, 0, 1.7269 + 4.9604j, 0, 0, 0, 2.5727 + 5.6211j],
        "eps2W": [2.7317 + 4.0352j, 0, 0, 0, 2.7317 + 4.0352j, 0, 0, 0, 1.8678 + 3.9295j],
        "d": [[0, 0, 0, 0, 2.8, 2.3], [2.3, -2.3, 0, 2.8, 0, 0], [-1.9, -1.9, 0.93, 0, 0, 0]],
        "lambda_um": 0.8,
    },
    # --- the two "Cases in DOI" buttons that were missing from the first transcription pass
    # (case-study fidelity audit, ; V1.04 buttons at ~lines 40088-40320) ---
    {
        # Button "LiNbO3 (11-2bar-0), MTI X-cut". LinearInput==1 (refractive-index mode): the button
        # writes n = Sqrt[eps] literally (SuperscriptBox[5.0756, 0.5] etc.), so the EPSILON values
        # below are the exact radicands — no rounding introduced. UseDirection==2 (Miller (1,1,0),
        # in-plane [1,-1,0]) but the button's stored cr matrix is used verbatim here (same treatment
        # as GaAs(111)@1064) to bypass the si-frame Miller azimuth subtlety.
        "name": "LiNbO3 (11-20) MTI X-cut",
        "lattice": [5.14, 5.14, 13.50, 90, 90, 120],
        "point_group": "3m",
        "orientation": [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
        "epsW": [5.0756, 0, 0, 0, 5.0756, 0, 0, 0, 4.6673],
        "eps2W": [5.9248, 0, 0, 0, 5.9248, 0, 0, 0, 5.3319],
        "d": [[0, 0, 0, 0, 5.95, -13.07], [-13.07, 13.07, 0, 5.95, 0, 0], [5.95, 5.95, 34.45, 0, 0, 0]],
        "lambda_um": 0.8,
    },
    {
        # Button "KTP (100)". LinearInput==2: entered directly as epsilon. UseDirection==1: the cr
        # matrix is the orientation used by the original, verbatim.
        "name": "KTP (100)",
        "lattice": [6.40, 10.62, 12.81, 90, 90, 90],
        "point_group": "mm2",
        "orientation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        "epsW": [3.00, 0, 0, 0, 3.04, 0, 0, 0, 3.34],
        "eps2W": [3.16, 0, 0, 0, 3.19, 0, 0, 0, 3.56],
        "d": [[0, 0, 0, 0, 1.9, 0], [0, 0, 0, 3.6, 0, 0], [2.5, 4.4, 16.9, 0, 0, 0]],
        "lambda_um": 0.8,
    },
]


def _orientation_matrix(spec) -> list[list[float]]:
    if "orientation" in spec:
        return [[float(x) for x in row] for row in spec["orientation"]]
    a, b, c, al, be, ga = spec["lattice"]
    structure = CrystalStructure(point_group=spec["point_group"], a=a, b=b, c=c,
                                 alpha_deg=al, beta_deg=be, gamma_deg=ga)
    (hkl, uvw) = spec["miller"]
    orient = CrystalOrientation.from_shaarp_miller_surface(structure, hkl, uvw)
    return [[float(x) for x in row] for row in np.asarray(orient.z_axes_in_lab, dtype=float)]


def _entry(spec) -> dict:
    epsW = np.asarray(spec["epsW"], dtype=complex).reshape(3, 3)
    eps2W = np.asarray(spec["eps2W"], dtype=complex).reshape(3, 3)
    d = np.asarray(spec["d"], dtype=complex)
    return {
        "name": spec["name"],
        "source_key": "SHAARP.si V1.04 case study (notebook inline constants)",
        "native_lambda_um": float(spec["lambda_um"]),
        "point_group": spec["point_group"],
        "lattice": [float(x) for x in spec["lattice"]],
        "orientation": _orientation_matrix(spec),
        "grid_um": [float(spec["lambda_um"])],
        "epsW_re": [epsW.real.tolist()],
        "epsW_im": [epsW.imag.tolist()],
        "eps2W_re": [eps2W.real.tolist()],
        "eps2W_im": [eps2W.imag.tolist()],
        "d_re": d.real.tolist(),
        "d_im": d.imag.tolist(),
    }


def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    # (the one-time pre-SI backup step was retired: the .bak_pre_si artifact served its
    # migration purpose and would trip the release gate's bundle-hygiene *.bak* ban if recreated)
    added = []
    for spec in SI_MATERIALS:
        name = spec["name"]
        if name in data["materials"]:
            continue
        data["materials"][name] = _entry(spec)
        data["order"].append(name)
        added.append(name)
    JSON_PATH.write_text(json.dumps(data), encoding="utf-8")
    print("added:", added or "(all already present)")
    print("total materials:", len(data["materials"]))


if __name__ == "__main__":
    main()
