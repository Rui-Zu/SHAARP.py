"""The SHAARP.ml docs' canonical Maker-fringe example, replicated FAITHFULLY from the source:
Z-cut quartz (121.2 um) with a gold coating (13.9 nm) at the back, lambda = 0.8 um, p-in/p-out,
theta 0-45 deg at 0.1 deg (Github/SHAARP.ml/docs/examples.md "Maker fringes" + the case-study
presets in setup.nb: Quartz$zCut\\[Lambda]800 and Au\\[Lambda]800).

Every constant below is transcribed from setup.nb:
  * quartz: point group 32 with d11 = 0.3 pm/V, d14 = 0.0; z-cut (surface (001), in-plane (120),
    crystal-physics axes = lab axes -> identity orientation); ordinary/extraordinary indices from
    the 5-term Sellmeier (Ghosh) evaluated at lambda and lambda/2 -- the .nb's commented fixed
    values (no_w=1.538360, ne_w=1.5473, no_2w=1.5577, ne_2w=1.5673) are used as a transcription
    cross-check;
  * gold: isotropic COMPLEX n_w = 3.479500+2.921145i, n_2w = 2.251755+1.596650i (the preset's own
    values -- replicated verbatim for faithfulness, whatever their provenance), no SHG, 0.0139 um.

This is the THICK-slab case where genuine sub-degree fine fringes physically exist (phase budget
~6e3 rad -> fringe period ~0.5 deg, well resolved at 0.1 deg) -- the honest subject for the dense
fine-fringe benchmark panel.
"""

from __future__ import annotations

import numpy as np

from shaarp import presets
from shaarp.config import CrystalOrientation, CrystalStructure, Layer, Material, MultilayerSystem, Polarimetry

WAVELENGTH_UM = 0.8
QUARTZ_THICKNESS_UM = 121.2  # docs example value (preset default is 121.1; the tutorial enters 121.2)
AU_THICKNESS_UM = 0.0139

# setup.nb Sellmeier coefficient pairs (B_i, C_i) with n^2 = 1 + sum B_i lam^2/(lam^2 - C_i^2)
_QUARTZ_ORDINARY = ((0.663044, 0.060), (0.517852, 0.106), (0.175912, 0.119), (0.565380, 8.844), (1.675299, 20.742))
_QUARTZ_EXTRAORDINARY = ((0.665721, 0.060), (0.503511, 0.106), (0.214792, 0.119), (0.539173, 8.792), (1.807613, 19.70))

AU_N_OMEGA = 3.479500 + 2.921145j
AU_N_2OMEGA = 2.251755 + 1.596650j

D11_PM_V = 0.3
D14_PM_V = 0.0


def quartz_sellmeier_index(lambda_um: float, *, extraordinary: bool) -> float:
    coeffs = _QUARTZ_EXTRAORDINARY if extraordinary else _QUARTZ_ORDINARY
    lam2 = float(lambda_um) ** 2
    n2 = 1.0 + sum(b * lam2 / (lam2 - c * c) for b, c in coeffs)
    return float(np.sqrt(n2))


def _quartz_material() -> Material:
    no_w = quartz_sellmeier_index(WAVELENGTH_UM, extraordinary=False)
    ne_w = quartz_sellmeier_index(WAVELENGTH_UM, extraordinary=True)
    no_2 = quartz_sellmeier_index(WAVELENGTH_UM / 2, extraordinary=False)
    ne_2 = quartz_sellmeier_index(WAVELENGTH_UM / 2, extraordinary=True)
    # 32 pattern (setup.nb): row1 {d11,-d11,0,d14,0,0}; row2 {0,0,0,0,-d14,-d11}; row3 zeros.
    d = np.zeros((3, 6), dtype=complex)
    d[0, 0], d[0, 1], d[0, 3] = D11_PM_V, -D11_PM_V, D14_PM_V
    d[1, 4], d[1, 5] = -D14_PM_V, -D11_PM_V
    return Material(
        name="Z-cut quartz (docs)", structure=CrystalStructure(point_group="32"),
        orientation=CrystalOrientation(),  # z-cut, crystal-physics axes = lab axes (setup.nb identity)
        epsilon_omega=np.diag([no_w**2, no_w**2, ne_w**2]).astype(complex),
        epsilon_2omega=np.diag([no_2**2, no_2**2, ne_2**2]).astype(complex),
        d_voigt_pm_v=d,
    )


def _gold_material() -> Material:
    return Material(
        name="Au coating (docs)", structure=CrystalStructure(point_group="∞∞m"),
        orientation=CrystalOrientation(),
        epsilon_omega=(np.eye(3) * AU_N_OMEGA**2).astype(complex),
        epsilon_2omega=(np.eye(3) * AU_N_2OMEGA**2).astype(complex),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )


def build_quartz_au_system() -> MultilayerSystem:
    return MultilayerSystem(
        wavelength_um=WAVELENGTH_UM,
        polarimetry=Polarimetry(theta_deg=0.0, phi_deg=0.0, psi_deg=0.0),  # p-in / p-out
        layers=[
            Layer("air in", presets.air(), shg_active=False),
            Layer("Z-cut quartz", _quartz_material(), thickness_um=QUARTZ_THICKNESS_UM, shg_active=True),
            Layer("Au coating", _gold_material(), thickness_um=AU_THICKNESS_UM, shg_active=False),
            Layer("air out", presets.air(), shg_active=False),
        ],
    )


def build_quartz_au_case(theta_deg) -> dict:
    """Case dict in the shape consumed by benchmarks.generate_maker_mathematica_inputs.case_to_record."""
    return {
        "id": "quartz_au_docs_maker", "theta_deg": list(theta_deg), "phi_deg": 0.0, "psi_deg": 0.0,
        "ellipticity_deg": 0.0, "sample_azimuth_deg": 0.0, "system": build_quartz_au_system(),
    }
