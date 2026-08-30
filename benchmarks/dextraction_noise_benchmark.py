"""R8: Monte-Carlo NOISE-ROBUSTNESS characterization of d-extraction (field vs intensity).

Every d-extraction gate so far verifies the exact inverse on noise-free synthetic scans
(docs/.md R8). A real polarimetry experiment measures with noise; this benchmark
characterizes how multiplicative intensity noise propagates into the recovered d-components,
across scan designs and both extraction methods.

Method (mirrors the validated ``basis="numeric"`` path of ``extract_si_d_voigt`` exactly --
see ``polarimetry_extraction._extract_numeric_basis``):
  * build the numeric design matrix M once per scan design (unit-d responses of the validated
    ``solve_single_interface_shg``);
  * cross-check: recovery from CLEAN measurements through the shared ``_recover_d_from_design``
    reproduces the true d to ~1e-12 (pins that this loop IS the public API's model);
  * per trial, inject multiplicative intensity noise I -> I*(1+eps), eps ~ N(0, sigma):
    realized on the fields as E -> E*sqrt(max(1+eps,0)) for the intensity method, and as
    E -> E*(1+eps/2) (amplitude noise, phase kept) for the field method;
  * error metric: err = min(max|d^-d|, max|d^+d|) / max|d| (min over the intensity method's
    inherent global-sign ambiguity).

HEADLINE RESULT (N=100/level, seeded): the two methods are qualitatively different
under noise --
  * FIELD (phase-resolved) degrades gracefully: dense 16-geometry x 24-phi scan gives median
    err ~ 1.7% at sigma = 2% intensity noise (~ err = sigma/2 propagation);
  * INTENSITY (Gram + rank-1, phase-less) amplifies catastrophically at this problem's
    conditioning (kappa(M) ~ 1.6e4-8.3e4; the Gram system effectively squares it): median err
    is 40-400% at sigma = 2% depending on design, and never below ~8% even in the best trial.
    Denser geometry helps (kappa 83k -> 16k, median 4.0 -> 0.6) but does not rescue it.
GUIDANCE: treat intensity-method output on noisy data as an initial guess / structure detector
(which components are large), not a quantitative estimate; quantitative extraction from noisy
data needs the phase-resolved field method (or a future nonlinear refinement on |E|^2).

Output: build/dextraction_noise/ (PNG + JSON summary). Seeded -> reproducible.

Run: python benchmarks/dextraction_noise_benchmark.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from shaarp.polarimetry_extraction import _recover_d_from_design
from shaarp.shg import solve_single_interface_shg
from shaarp.tensors import rotate_d_voigt_crystal_to_lab

EPS_W = [2.2 ** 2, 2.2 ** 2, 2.5 ** 2]  # uniaxial about z: azimuth rotates only d
EPS_2W = [2.4 ** 2, 2.4 ** 2, 2.7 ** 2]
POSITIONS = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 0)]
TRUE = np.array([0.30, 0.50, 0.60, 0.80, 0.40, 0.20])

# scan designs: (label, geometries, phi values)
_P6 = list(np.linspace(0.25, math.pi - 0.25, 6))
_P24 = list(np.linspace(0.05, math.pi - 0.05, 24))
DESIGNS = {
    "baseline 6geom x 6phi": ([(0.35, 0.0), (0.35, 0.6), (0.35, 1.2),
                               (0.75, 0.0), (0.75, 0.6), (0.75, 1.2)], _P6),
    "dense 16geom x 24phi": ([(t, a) for t in (0.25, 0.5, 0.75, 1.0)
                              for a in (0.0, 0.4, 0.8, 1.2)], _P24),
}

SIGMAS = (0.005, 0.02, 0.05)
N_TRIALS = 100
SEED = 20260704


def _rz(az: float) -> np.ndarray:
    c, s = math.cos(az), math.sin(az)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _reflected_coeffs(d_voigt: np.ndarray, theta: float, az: float, phi: float):
    """(E_s, E_p) reflected-SHG coefficients for a crystal with Voigt d rotated by Rz(az)."""
    d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(d_voigt, _rz(az)), dtype=complex))
    r = solve_single_interface_shg(
        np.diag(EPS_W).astype(complex), np.diag(EPS_2W).astype(complex), d_rot,
        incident_index_omega=1.0, incident_index_2omega=1.0, incident_theta_rad=theta,
        incident_jones=(math.sin(phi), math.cos(phi)), omega=1.0, mu=1.0, eps0=1.0,
    )
    c = np.asarray(r.coefficients)
    return complex(c[0]), complex(c[1])


def build_design_and_clean_measurements(geometries, phi_values):
    """Numeric design matrix M (rows: geometry x phi x channel; cols: d-components) + the clean
    measurement vector -- the same loop as _extract_numeric_basis, hoisted so the expensive
    part runs ONCE for all Monte-Carlo trials of a design."""
    unit_ds = []
    for pos in POSITIONS:
        du = np.zeros((3, 6))
        du[pos] = 1.0
        unit_ds.append(du)
    d_true = np.zeros((3, 6))
    for (m, ell), v in zip(POSITIONS, TRUE):
        d_true[m, ell] = v

    rows, e_clean = [], []
    for theta, az in geometries:
        for phi in phi_values:
            unit_resp = [_reflected_coeffs(du, theta, az, phi) for du in unit_ds]
            meas = _reflected_coeffs(d_true, theta, az, phi)
            for ch in (0, 1):  # s, p
                rows.append([resp[ch] for resp in unit_resp])
                e_clean.append(meas[ch])
    return np.asarray(rows, dtype=complex), np.asarray(e_clean, dtype=complex)


def relative_error(recovered: np.ndarray) -> float:
    r = np.real(np.asarray(recovered))
    return float(min(np.max(np.abs(r - TRUE)), np.max(np.abs(r + TRUE))) / np.max(np.abs(TRUE)))


def run(n_trials: int = N_TRIALS, sigmas=SIGMAS, seed: int = SEED, designs=None) -> dict:
    out: dict = {"n_trials": n_trials, "seed": seed, "designs": {}}
    for label, (geoms, phis) in (designs or DESIGNS).items():
        M, e_clean = build_design_and_clean_measurements(geoms, phis)
        clean = _recover_d_from_design(M, e_clean, POSITIONS, "intensity")
        clean_err = relative_error(clean.values)
        assert clean.identifiable, f"{label}: scan lost identifiability (rank {clean.rank})"
        assert clean_err < 1e-8, f"{label}: clean recovery off ({clean_err:.2e})"
        entry = {"rows": int(M.shape[0]), "condition_number": float(clean.condition_number),
                 "clean_error": clean_err, "methods": {}}
        for method in ("intensity", "field"):
            rng = np.random.default_rng(seed)  # same noise stream per method -> paired comparison
            per_sigma = {}
            for sigma in sigmas:
                errs = []
                for _ in range(n_trials):
                    eps = rng.normal(0.0, sigma, size=e_clean.shape)
                    if method == "intensity":
                        e_noisy = e_clean * np.sqrt(np.maximum(1.0 + eps, 0.0))
                    else:  # amplitude noise equivalent to the same intensity noise, phase kept
                        e_noisy = e_clean * (1.0 + eps / 2.0)
                    res = _recover_d_from_design(M, e_noisy, POSITIONS, method)
                    errs.append(relative_error(res.values))
                e = np.asarray(errs)
                per_sigma[f"{sigma:g}"] = {"median": float(np.median(e)),
                                           "p90": float(np.quantile(e, 0.90)),
                                           "max": float(np.max(e))}
            entry["methods"][method] = per_sigma
        out["designs"][label] = entry
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = ROOT / "build" / "dextraction_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run()

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    styles = {"intensity": dict(ls="-", marker="o"), "field": dict(ls="--", marker="s")}
    colors = {"baseline 6geom x 6phi": "crimson", "dense 16geom x 24phi": "navy"}
    for label, entry in result["designs"].items():
        for method, per_sigma in entry["methods"].items():
            sig = [float(s) for s in per_sigma]
            med = [v["median"] for v in per_sigma.values()]
            ax.loglog(sig, med, color=colors[label], **styles[method],
                      label=f"{method}, {label} (k={entry['condition_number']:.0f})")
    sig_all = sorted(float(s) for s in SIGMAS)
    ax.loglog(sig_all, [s / 2 for s in sig_all], color="0.6", lw=1, label="err = sigma/2 (reference)")
    ax.set_xlabel(r"intensity noise level $\sigma$ (multiplicative)")
    ax.set_ylabel("median relative d error (min over sign)")
    ax.set_title("d-extraction noise robustness: field vs intensity method\n"
                 "(Monte-Carlo N=%d/level; clean recovery ~1e-12 for all)" % result["n_trials"],
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / "dextraction_noise_robustness.png", dpi=130)
    plt.close(fig)

    (out_dir / "dextraction_noise_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"figure + summary in {out_dir}")


if __name__ == "__main__":
    main()
