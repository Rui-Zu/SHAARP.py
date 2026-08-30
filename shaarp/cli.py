from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import Layer, MultilayerSystem, Polarimetry, multilayer_shg, presets, single_interface_intensity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAARP.py demos.")
    parser.add_argument("--mode", choices=["si", "ml"], default="si")
    parser.add_argument("--theta", type=float, default=45.0)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--points", type=int, default=181)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    phi = np.linspace(0, 360, args.points)
    pol = Polarimetry(theta_deg=args.theta, phi_deg=phi, psi_deg=args.psi)

    if args.mode == "si":
        result = single_interface_intensity(presets.linbo3_1120_xcut(), pol)
    else:
        system = MultilayerSystem(
            wavelength_um=1.064,
            polarimetry=pol,
            layers=[
                Layer("air in", presets.air(), shg_active=False),
                Layer("LNO slab", presets.linbo3_zcut_1064(), thickness_um=1.0),
                Layer("air out", presets.air(), shg_active=False),
            ],
        )
        result = multilayer_shg(system)

    result.plot()
    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
