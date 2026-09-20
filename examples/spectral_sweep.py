"""Reflected SHG across a wavelength band, and a wavelength-by-angle map.

Run it:

    python examples/spectral_sweep.py

Writes two PNGs into build/ (untracked). The single-interface panel sweeps quartz across its
usable band; the multilayer panel maps Maker fringes against both wavelength and angle.

A spectral sweep passes a FACTORY of wavelength rather than a material or a system, because the
dielectric tensors have to be rebuilt at every point. On a layer stack the wavelength also sets
each layer's optical thickness, so the whole system is rebuilt, not just its wavelength field --
changing only the wavelength while the tensors stay put is a thickness sweep, not a spectrum.

A layer stack also has interference fringes along the wavelength axis, closer together the
thicker the layer. The map below uses a 2 um film and a 0.02 um step, which resolves them; a
3 um film at that step would not, and the sweep would warn that the curve is undersampled.

The SHG tensor is held constant across the sweep; every result says so in
``res.stages["assumptions"]``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# `python examples/<name>.py` puts examples/ on sys.path, NOT the repo root, so `shaarp` would
# resolve to whatever is installed rather than to this checkout. Make the root importable first.
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from shaarp import run_si_spectrum, run_spectral_map
from shaarp.spectral import casestudy_ml_spectrum, casestudy_spectrum

OUT = _ROOT / "build"
MATERIAL = "Quartz z-cut (800 nm)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- a spectrum from one crystal surface ------------------------------------------------
    res = run_si_spectrum(
        casestudy_spectrum(MATERIAL),
        options={"lambda_min_um": 0.60, "lambda_max_um": 1.60, "lambda_step_um": 0.02,
                 "theta_deg": 45.0, "phi_deg": 30.0},
    )
    lam = res.numeric["wavelength_um"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(lam, res.numeric["intensity_s"], label=r"$I_s^{2\omega}$")
    ax.plot(lam, res.numeric["intensity_p"], label=r"$I_p^{2\omega}$")
    ax.set_xlabel(r"fundamental wavelength $\lambda$ (µm)")
    ax.set_ylabel(r"reflected $I^{2\omega}$")
    ax.set_title(f"{MATERIAL}: reflected SHG vs wavelength\n"
                 f"SHG tensor {res.stages['assumptions']['d_voigt']}", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "spectral_sweep_si.png", dpi=150)
    print(f"single interface: {lam.size} wavelengths, "
          f"I_s {res.numeric['intensity_s'].min():.3e} to {res.numeric['intensity_s'].max():.3e}")

    # --- a wavelength-by-angle Maker map from a 2 um film ------------------------------------
    mapped = run_spectral_map(
        casestudy_ml_spectrum(MATERIAL, thickness_um=2.0),
        options={"lambda_min_um": 0.80, "lambda_max_um": 1.40, "lambda_step_um": 0.02,
                 "theta_min_deg": 0.0, "theta_max_deg": 45.0, "theta_step_deg": 1.5,
                 "kind": "maker", "phi_deg": 30.0},
    )
    rows, cols = mapped.stages["shape"]
    # the map is stored LONG (one row per wavelength-angle pair, so it writes straight to CSV);
    # reshape it to wavelengths x angles to draw it
    grid = np.asarray(mapped.numeric["parallel_intensity"]).reshape(rows, cols)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    mesh = ax.pcolormesh(mapped.stages["theta_axis"], mapped.stages["wavelength_axis"], grid,
                         shading="nearest")
    fig.colorbar(mesh, ax=ax, label=r"transmitted $I_\parallel^{2\omega}$")
    ax.set_xlabel(r"incidence angle $\theta_i$ (deg)")
    ax.set_ylabel(r"fundamental wavelength $\lambda$ (µm)")
    ax.set_title(f"{MATERIAL}, 2 µm film: Maker fringes vs wavelength", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "spectral_sweep_map.png", dpi=150)
    print(f"multilayer map: {rows} wavelengths x {cols} angles")
    print(f"  wrote {OUT / 'spectral_sweep_si.png'}")
    print(f"  wrote {OUT / 'spectral_sweep_map.png'}")


if __name__ == "__main__":
    main()
