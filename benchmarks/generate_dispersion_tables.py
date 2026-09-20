"""Generate the shipped refractive-index tables in shaarp/dispersion_data/.

WHERE THE NUMBERS COME FROM. Every coefficient set below was read from the refractiveindex.info
database's own YAML source files (github.com/polyanskiy/refractiveindex.info-database, released
CC0), NOT retyped from a rendered web page, and each carries the primary literature reference that
database cites. The dispersion formulas are that database's numbered forms:

  formula 2   n^2 - 1 = c1 + sum_i c_{2i} lam^2 / (lam^2 - c_{2i+1})
  formula 4   n^2 = c1 + c2 lam^c3/(lam^2 - c4^c5) + c6 lam^c7/(lam^2 - c8^c9)
                       + c10 lam^c11 + c12 lam^c13 + ...

The tables are SAMPLED from those formulas rather than shipping the formulas themselves, because
the runtime reads tables and a table is what a user can supply for their own crystal. Sampling is
finer in the blue, where Sellmeier dispersion is steep; the generator measures the resulting
linear-interpolation error against the closed form and refuses to write a table that exceeds
INTERP_TOLERANCE, so the shipped file is provably as good as the formula it came from.

Run:  python benchmarks/generate_dispersion_tables.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "shaarp" / "dispersion_data"

# A sampled table must reproduce its own closed form this well, or the generator refuses to write.
# 1e-5 is not arbitrary: the published Sellmeier coefficients are quoted to about five significant
# figures, so the closed form's own accuracy in n is of that order. Sampling tighter than the
# source data's precision buys nothing but file size.
INTERP_TOLERANCE = 1e-5

RII = "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main"


def formula_2(lam: np.ndarray, c: list[float]) -> np.ndarray:
    total = c[0] + 1.0
    for i in range(1, len(c) - 1, 2):
        total = total + c[i] * lam**2 / (lam**2 - c[i + 1])
    return np.sqrt(total)


def formula_4(lam: np.ndarray, c: list[float]) -> np.ndarray:
    c = list(c) + [0.0] * (9 - len(c))
    total = (c[0]
             + c[1] * lam**c[2] / (lam**2 - c[3]**c[4])
             + c[5] * lam**c[6] / (lam**2 - c[7]**c[8]))
    for i in range(9, len(c) - 1, 2):
        total = total + c[i] * lam**c[i + 1]
    return np.sqrt(total)


FORMULAS = {2: formula_2, 4: formula_4}


def sampled_grid(low: float, high: float) -> np.ndarray:
    """Fine in the blue where Sellmeier curvature lives, coarser in the infrared.

    Linear-interpolation error goes as h^2 times the curvature, and the curvature is concentrated
    near the ultraviolet resonance -- LiNbO3 at 2 nm sampling misses its own formula by 8.8e-6,
    which a uniform grid can only fix by paying that resolution across five micrometres of flat
    infrared. Three bands instead."""
    steps = ((0.6, 0.001), (1.5, 0.002), (float("inf"), 0.02))
    pieces, start = [], low
    for edge, step in steps:
        stop = min(edge, high)
        if stop > start:
            pieces.append(np.arange(start, stop, step))
            start = stop
    grid = np.concatenate(pieces + [np.array([high])])
    # ROUND BEFORE uniquifying. np.arange accumulates float error, so neighbours can differ by
    # 1e-16 -- distinct to np.unique, identical once written at six significant figures, and the
    # loader then rejects the file for repeated wavelengths. Round to the precision actually
    # written and the two agree by construction.
    return np.unique(np.round(grid, 6))


# --------------------------------------------------------------------------------------------
# The materials. `axes` maps each principal direction to (formula number, coefficients), read
# verbatim from the refractiveindex.info YAML named in `files`.
# --------------------------------------------------------------------------------------------
MATERIALS = [
    {
        "slug": "linbo3_zelmon_1997",
        "display": "LiNbO3 (dispersive)",
        "symmetry": "uniaxial",
        "range_um": (0.4, 5.0),
        "columns": ["no", "ne"],
        "axes": [(2, [0, 2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60]),
                 (2, [0, 2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08])],
        "files": ["LiNbO3/nk/Zelmon-o.yml", "LiNbO3/nk/Zelmon-e.yml"],
        "reference": ("D. E. Zelmon, D. L. Small, D. Jundt. Infrared corrected Sellmeier "
                      "coefficients for congruently grown lithium niobate and 5 mol.% magnesium "
                      "oxide-doped lithium niobate. J. Opt. Soc. Am. B 14, 3319-3322 (1997)."),
        "doi": "10.1364/JOSAB.14.003319",
        "conditions": "congruently grown LiNbO3, 21 C",
        "template": "LiNbO3 (11-20) MTI X-cut",
    },
    {
        "slug": "ktp_kato_2002",
        "display": "KTP (dispersive)",
        "symmetry": "biaxial",
        "range_um": (0.43, 3.54),
        "columns": ["nx", "ny", "nz"],
        "axes": [(4, [3.29100, 0.04140, 0, 0.03978, 1, 9.35522, 0, 31.45571, 1]),
                 (4, [3.45018, 0.04341, 0, 0.04597, 1, 16.98825, 0, 39.43799, 1]),
                 (4, [4.59423, 0.06206, 0, 0.04763, 1, 110.80672, 0, 86.12171, 1])],
        "files": ["KTiOPO4/nk/Kato-alpha.yml", "KTiOPO4/nk/Kato-beta.yml",
                  "KTiOPO4/nk/Kato-gamma.yml"],
        "reference": ("K. Kato and E. Takaoka. Sellmeier and thermo-optic dispersion formulas for "
                      "KTP. Appl. Opt. 41, 5040-5044 (2002)."),
        "doi": "10.1364/AO.41.005040",
        "conditions": "20 C",
        "template": "KTP (100)",
    },
    {
        "slug": "lbo_chen_1989",
        "display": "LiB3O5 / LBO (dispersive)",
        "symmetry": "biaxial",
        "range_um": (0.2894, 1.064),
        "columns": ["nx", "ny", "nz"],
        "axes": [(4, [2.45768, 0.0098877, 0, 0.026095, 1, 0, 0, 0, 1, -0.013847, 2]),
                 (4, [2.52500, 0.017123, 0, -0.0060517, 1, 0, 0, 0, 1, -0.0087838, 2]),
                 (4, [2.58488, 0.012737, 0, 0.021414, 1, 0, 0, 0, 1, -0.016293, 2])],
        "files": ["LiB3O5/nk/Chen-alpha.yml", "LiB3O5/nk/Chen-beta.yml",
                  "LiB3O5/nk/Chen-gamma.yml"],
        "reference": ("C. Chen, Y. Wu, A. Jiang, B. Wu, G. You, R. Li, S. Lin. New nonlinear-optical "
                      "crystal: LiB3O5. J. Opt. Soc. Am. B 6, 616-621 (1989); dispersion formula "
                      "from F. Hanson and D. Dick, Opt. Lett. 16, 205-207 (1991)."),
        "doi": "10.1364/JOSAB.6.000616",
        "conditions": "room temperature",
        "template": "LiB3O5 (LBO)",
    },
]

# GaAs is TABULATED n and k rather than a transparent-region formula, because its second harmonic
# sits above the band edge where the crystal absorbs -- a Sellmeier fit has nothing to say there.
TABULATED = [
    {
        "slug": "gaas_rakic_1996",
        "display": "GaAs (dispersive)",
        "symmetry": "isotropic",
        "columns": ["n", "k"],
        "files": ["GaAs/nk/Rakic.yml"],
        "reference": ("A. D. Rakic and M. L. Majewski. Modeling the optical dielectric function of "
                      "GaAs and AlAs: Extension of Adachi's model. J. Appl. Phys. 80, 5909-5914 "
                      "(1996)."),
        "doi": "10.1063/1.363586",
        "conditions": "room temperature",
        "template": "GaAs (111) (1064 nm)",
    },
]



# --------------------------------------------------------------------------------------------
# Materials given as a LORENTZ-OSCILLATOR model rather than a Sellmeier formula. TaAs is a Weyl
# semimetal: it absorbs across the whole visible, so there is no transparent region for a
# Sellmeier fit to describe, and its optical constants come from spectroscopic ellipsometry.
#
# THE MODEL is Eq. 1 of Zu et al., Phys. Rev. B 103, 165137 (2021):
#
#     eps = eps_inf + sum_n A_n G_n E_n / (E_n^2 - E^2 - i E G_n)
#           + A_UV / (E_UV^2 - E^2)  -  A_IR / E^2
#
# SIGN OF THE INFRARED POLE. The paper prints that last term with a PLUS. A plus cannot be right:
# it puts eps_1 of the ordinary axis at 14.5 at 1.55 eV, where the paper's own Fig. 1(a) reads
# about 6, and where the exported case-study registry -- built from the author's setup.nb -- holds
# 6.2591. With a MINUS the model gives 6.19, agreeing with both to better than half a percent.
# The minus is also what the pole function A/(E_pole^2 - E^2) gives as E_pole goes to zero, which
# is the form the UV term above is written in. So the printed plus is a typographical slip and the
# minus is used here; the registry cross-check in the test suite pins it.
#
# THE PARAMETERS are the author's ellipsometry fit record (Ellipsometry/Fitting/Detailed Model
# Parameter.xlsx in the TaAs and NbAs project), which is what Fig. 1 was drawn from; the printed
# paper plots the oscillators (Fig. 2) but does not tabulate them.
# --------------------------------------------------------------------------------------------
OSCILLATOR = [
    {
        "slug": "taas_zu_2021",
        "display": "TaAs (dispersive)",
        "symmetry": "uniaxial",
        "range_um": (0.2066, 1.0332),          # the paper's measured window, 1.2-6 eV
        "columns": ["no", "ne", "ko", "ke"],
        "reference": ("R. Zu, M. Gu, L. Min, C. Hu, N. Ni, Z. Mao, J. M. Rondinelli, V. Gopalan. "
                      "Comprehensive anisotropic linear optical properties of the Weyl semimetals "
                      "TaAs and NbAs. Phys. Rev. B 103, 165137 (2021)."),
        "doi": "10.1103/PhysRevB.103.165137",
        "conditions": "spectroscopic ellipsometry, room temperature, Lorentz model of Eq. 1",
        "template": "TaAs (112)",
        "axes": {
            # ordinary (eps_11) and extraordinary (eps_33); columns R4 and R32 of the fit record
            "ordinary": dict(einf=1.833, uv=(0.001175, 5.786), ir=9.9950, osc=[
                (21.320227, 0.8441, 1.019), (0.992923, 0.4831, 1.811),
                (11.384437, 4.0886, 2.911), (0.844580, 0.4012, 4.125),
                (1.440048, 0.5097, 3.736), (4.675134, 2.1295, 4.815),
                (1.363416, 1.0745, 5.698), (4.627214, 1.9852, 6.846)]),
            "extraordinary": dict(einf=1.748, uv=(0.1115, 6.497), ir=6.5604, osc=[
                (7.973320, 0.4650, 1.169), (4.568836, 0.8307, 1.840),
                (1.597659, 0.2236, 2.201), (0.776836, 0.3022, 2.810),
                (12.419482, 4.4101, 3.056), (2.745969, 0.7513, 4.147),
                (4.986716, 4.1050, 6.248)]),
        },
    },
]

HC_EV_UM = 1.23984193           # photon energy in eV times wavelength in um


def oscillator_epsilon(lam_um: np.ndarray, p: dict) -> np.ndarray:
    """Eq. 1 of the paper, evaluated at a wavelength grid. See the sign note above."""
    energy = HC_EV_UM / np.asarray(lam_um, dtype=float)
    amp_uv, en_uv = p["uv"]
    eps = (complex(p["einf"])
           + amp_uv / (en_uv**2 - energy**2)
           - p["ir"] / energy**2)
    for amp, broad, en in p["osc"]:
        eps = eps + amp * broad * en / (en**2 - energy**2 - 1j * energy * broad)
    return eps


def fetch_tabulated(rel: str) -> np.ndarray:
    with urllib.request.urlopen(f"{RII}/{rel}", timeout=60) as handle:
        text = handle.read().decode("utf-8")
    rows, started = [], False
    for line in text.splitlines():
        if "data: |" in line:
            started = True
            continue
        if started:
            parts = line.split()
            if len(parts) == 3:
                rows.append([float(x) for x in parts])
            elif parts:
                break
    if not rows:
        raise SystemExit(f"no tabulated rows parsed from {rel}")
    return np.asarray(rows, dtype=float)


def write_csv(path: Path, header: list[str], grid: np.ndarray, columns: list[np.ndarray],
              banner: list[str]) -> None:
    lines = [f"# {b}" for b in banner]
    lines.append(",".join(["wavelength_um"] + header))
    for row in range(grid.size):
        cells = [f"{grid[row]:.6g}"] + [f"{col[row]:.8g}" for col in columns]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = {}

    for spec in MATERIALS:
        low, high = spec["range_um"]
        grid = sampled_grid(low, high)
        columns, worst = [], 0.0
        for kind, coefficients in spec["axes"]:
            evaluate = FORMULAS[kind]
            columns.append(evaluate(grid, coefficients))
            # does linear interpolation of what we are about to write reproduce the formula?
            dense = np.linspace(low, high, 20001)
            error = np.abs(np.interp(dense, grid, columns[-1]) - evaluate(dense, coefficients))
            worst = max(worst, float(error.max()))
        if worst > INTERP_TOLERANCE:
            raise SystemExit(f"{spec['slug']}: interpolation error {worst:.2e} exceeds tolerance")
        banner = [f"{spec['display']} -- principal refractive index",
                  f"source: {spec['reference']}",
                  f"doi: {spec['doi']}   conditions: {spec['conditions']}",
                  f"validity: {low:g}-{high:g} um",
                  "via refractiveindex.info database (CC0): " + ", ".join(spec["files"]),
                  f"generated by benchmarks/generate_dispersion_tables.py; max interpolation "
                  f"error vs the closed form {worst:.1e}"]
        write_csv(OUT / f"{spec['slug']}.csv", spec["columns"], grid, columns, banner)
        registry[spec["slug"]] = {k: spec[k] for k in
                                  ("display", "symmetry", "reference", "doi", "conditions",
                                   "template")}
        registry[spec["slug"]]["range_um"] = [low, high]
        registry[spec["slug"]]["source_files"] = spec["files"]
        print(f"  {spec['slug']:24s} {grid.size:5d} pts  {low:g}-{high:g} um  "
              f"interp err {worst:.1e}")

    for spec in TABULATED:
        raw = fetch_tabulated(spec["files"][0])
        grid, n, k = raw[:, 0], raw[:, 1], raw[:, 2]
        low, high = float(grid[0]), float(grid[-1])
        banner = [f"{spec['display']} -- refractive index and extinction coefficient",
                  f"source: {spec['reference']}",
                  f"doi: {spec['doi']}   conditions: {spec['conditions']}",
                  f"validity: {low:g}-{high:g} um",
                  "via refractiveindex.info database (CC0): " + ", ".join(spec["files"]),
                  "generated by benchmarks/generate_dispersion_tables.py; tabulated values copied "
                  "verbatim, not resampled"]
        write_csv(OUT / f"{spec['slug']}.csv", spec["columns"], grid, [n, k], banner)
        registry[spec["slug"]] = {k2: spec[k2] for k2 in
                                  ("display", "symmetry", "reference", "doi", "conditions",
                                   "template")}
        registry[spec["slug"]]["range_um"] = [low, high]
        registry[spec["slug"]]["source_files"] = spec["files"]
        print(f"  {spec['slug']:24s} {grid.size:5d} pts  {low:g}-{high:g} um  (tabulated n,k)")

    for spec in OSCILLATOR:
        low, high = spec["range_um"]
        grid = sampled_grid(low, high)
        eps_o = oscillator_epsilon(grid, spec["axes"]["ordinary"])
        eps_e = oscillator_epsilon(grid, spec["axes"]["extraordinary"])
        n_o, n_e = np.sqrt(eps_o), np.sqrt(eps_e)
        if np.any(n_o.imag < 0) or np.any(n_e.imag < 0):
            raise SystemExit(f"{spec['slug']}: negative extinction -- wrong square-root branch")
        banner = [f"{spec['display']} -- principal refractive index and extinction coefficient",
                  f"source: {spec['reference']}",
                  f"doi: {spec['doi']}   conditions: {spec['conditions']}",
                  f"validity: {low:g}-{high:g} um (the measured window, 1.2-6 eV)",
                  "Lorentz-oscillator model of Eq. 1 with the INFRARED POLE TAKEN NEGATIVE -- the "
                  "printed plus disagrees with the paper's own Fig. 1 and with the exported "
                  "case-study registry; see benchmarks/generate_dispersion_tables.py",
                  "generated by benchmarks/generate_dispersion_tables.py"]
        write_csv(OUT / f"{spec['slug']}.csv", spec["columns"], grid,
                  [n_o.real, n_e.real, n_o.imag, n_e.imag], banner)
        registry[spec["slug"]] = {k: spec[k] for k in
                                  ("display", "symmetry", "reference", "doi", "conditions",
                                   "template")}
        registry[spec["slug"]]["range_um"] = [low, high]
        registry[spec["slug"]]["source_files"] = ["author ellipsometry fit record (Lorentz model)"]
        print(f"  {spec['slug']:24s} {grid.size:5d} pts  {low:g}-{high:g} um  (oscillator model)")

    (OUT / "index.json").write_text(
        json.dumps({"note": ("Refractive-index tables for the dispersive material variants. Every "
                             "entry names the primary literature it came from. The Sellmeier and "
                             "tabulated entries were read from the refractiveindex.info database's "
                             "YAML sources, which are released CC0; TaAs was evaluated from the "
                             "Lorentz-oscillator model of its source paper, with the oscillator "
                             "parameters of that paper's ellipsometry fit."),
                    "materials": registry}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(registry)} tables + index.json to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
