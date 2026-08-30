"""Reusable per-case builders that reproduce the published figures of BOTH SHAARP papers.

Every parameter (crystal orientation, point group, SHG ``d`` tensor, dielectric ``eps(w)``/``eps(2w)``,
wavelength, layer thicknesses) is sourced from the author's own Mathematica case-study examples -- the
``shaarp/casestudy_dispersion.json`` registry was transcribed verbatim from the original ``SHAARP.si``
V1.04 case buttons and the ``SHAARP.ml`` V1.3x case-study notebooks (provenance-audited: memory
``shaarp-case-materials-provenance``) -- plus the per-figure geometry stated in the manuscripts. Nothing
here is read off the papers' prose tables or invented; the registry IS the extracted-from-Mathematica
store, and every case cites the ``source_key`` (the Mathematica button/variable name) it came from.

The compute calls are exactly the ones the desktop GUI's Update button makes
(``si_polarimetry_curve`` / ``si_effective_index_curve`` / ``compute_ml_gui_result``), so a shape
mismatch against a paper panel is a product defect, not a harness artifact (T3 precedent: /).

PROVENANCE MAP (verified R1):

  paper fig case registry entry (source_key) lambda geometry
  ----- --- --------------------------- -------------------------------------- ------- -----------------------
  si 4 GaAs (111) GaAs (111) (800 nm) [GaAs111lam800] 0.800 semi-infinite, pg -43m
  si 5 LiNbO3 (11-20) x-cut LiNbO3 x-cut (1550 nm) [LiNbO3xCut..] 0.800 semi-infinite, pg 3m
  si 6 KTP (100) x-cut KTP x-cut [KTPxCut] 0.800 semi-infinite, pg mm2
  si 7 TaAs (112) Weyl semimetal TaAs (112) [V1.04 inline, complex d] 0.800 semi-infinite, pg 4mm
  ml 3 300 um X-cut quartz Quartz x-cut (1064 nm) 1.064 300 um film, HH/JK
  ml 4 123.6 um Z-cut quartz + Au Quartz z-cut (800 nm) + Au coating 0.800 123.6 um + 13.9 nm Au
  ml 5 LiNbO3/KTP single crystals LiNbO3 x/z-cut (1550) + KTP x-cut 1.550 thick single crystals
  ml 6 ZnO//Pt//Al2O3 heterostr. ZnO (001) / Pt (111) / Al2O3 (0001) 1.550 159 nm / 200 nm / sub
  ml 7 LiNbO3//quartz interference LiNbO3 x-cut (1550) // Quartz 1.550 50 um // 35 um

The SI paper used a single Ti:Sapphire source centred at 800 nm for ALL four crystals (verified in the
manuscript methods -- zero "1550 nm" occurrences), so Fig 5/6/7 interpolate the registry dispersion
grids (all span 0.4-2.0 um, 81 pts) to 800 nm; the SHG ``d`` ratios and orientation carry the crystal's
identity. ML Fig 6/7 layer thicknesses (159/200 nm; 50/35 um) are the manuscript's stated values.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shaarp.casestudy_materials import build_casestudy_material, build_casestudy_ml_system
from shaarp.config import Layer, MultilayerSystem, Polarimetry, with_sample_azimuth_deg
from shaarp.shaarp_gui import (
    compute_ml_gui_result,
    ml_polarimetry_curve,
    si_effective_index_curve,
    si_figure_kwargs_from_material,
    si_polarimetry_curve,
)

# Assumption labels -- must match the strings compute_ml_gui_result dispatches on (see shaarp_gui).
ASSUMPTION_FMR = "Full Multiple Reflections (FMR)"
ASSUMPTION_JK = "Jerphagnon & Kurtz Assumption (No MR)"
ASSUMPTION_HH = "Herman & Hayden Assumption (MR only for 2ω Homo Waves)"


# ======================================================================================
# SI paper (single air/crystal interface, reflected SHG polarimetry)
# ======================================================================================

# THE PAPER'S OWN EXTRACTED d TENSORS AT 800 nm (the strict benchmark inputs).
# Source: SHAARP.si 2022 main-text Table 1 ("Comparison of ratios of SHG coefficients from
# #SHAARP and literature", all in pm/V):
# GaAs |d36| = 267 +- 20
# LiNbO3 |d33| = 28.5 +- 0.2, d33/d31 = 5.2 +- 0.1, d22/d31 = -0.23 +- 0.02
# (d15/d31 = 1.25 from the fit notebook 'Plot of Polar Plots_xLNO.nb' -- the final
# NonlinearModelFit call restarts from the converged values {d33 5.209, d15 1.25,
# d22 -0.25} with d31 normalized to 1; Table 1 omits the d15 row)
# KTP |d33| = 17.0 +- 0.2, d33/d31 = 6.0, d32/d31 = 1.6, d24/d31 = 1.5, d15/d31 = 1.3
# TaAs d33 = -827 +- 39, d31 = -12 +- 15, d15 = -113 +- 20
# These OVERRIDE the registry case-study d sets (which carry other-wavelength/starting values,
# e.g. the LNO 1550-nm set with d33/d31 = 5.45 -- outside Table 1's error bar).
def _voigt_3m(d15, d22, d31, d33):
    return np.array([[0, 0, 0, 0, d15, -d22],
                     [-d22, d22, 0, d15, 0, 0],
                     [d31, d31, d33, 0, 0, 0]], dtype=complex)


def _voigt_mm2(d15, d24, d31, d32, d33):
    return np.array([[0, 0, 0, 0, d15, 0],
                     [0, 0, 0, d24, 0, 0],
                     [d31, d32, d33, 0, 0, 0]], dtype=complex)


def _voigt_4mm(d15, d31, d33):
    return np.array([[0, 0, 0, 0, d15, 0],
                     [0, 0, 0, d15, 0, 0],
                     [d31, d31, d33, 0, 0, 0]], dtype=complex)


def _voigt_43m(d36):
    return np.array([[0, 0, 0, d36, 0, 0],
                     [0, 0, 0, 0, d36, 0],
                     [0, 0, 0, 0, 0, d36]], dtype=complex)


_LNO_D31 = 28.5 / 5.2                        # 5.4808 pm/V (|d33| / (d33/d31))
_KTP_D31 = 17.0 / 6.0                        # 2.8333 pm/V
PAPER_TABLE1_D = {
    "fig4": _voigt_43m(267.0),
    "fig5": _voigt_3m(d15=1.25 * _LNO_D31, d22=-0.23 * _LNO_D31, d31=_LNO_D31, d33=28.5),
    "fig6": _voigt_mm2(d15=1.3 * _KTP_D31, d24=1.5 * _KTP_D31, d31=_KTP_D31,
                       d32=1.6 * _KTP_D31, d33=17.0),
    "fig7": _voigt_4mm(d15=-113.0, d31=-12.0, d33=-827.0),
}


@dataclass(frozen=True)
class SICase:
    """A single-interface reflected-SHG case from the SHAARP.si 2022 paper."""

    key: str
    fig: str
    title: str
    registry_name: str          # case-study material in casestudy_dispersion.json
    wavelength_um: float
    point_group: str
    surface: str                # human label of the cut (e.g. "(112̄0) x-cut")
    provenance: str             # one-line cite to the Mathematica source
    thetas_deg: tuple = (0.0, 15.0, 30.0, 45.0)

    def material(self, *, paper_d: bool = True):
        """The case material: registry dispersion (epsilon at this wavelength) + the PAPER'S OWN
        Table-1 extracted d tensor at 800 nm (``paper_d=False`` keeps the registry d for
        comparison). The paper values are the strict benchmark inputs (see PAPER_TABLE1_D)."""
        mat = build_casestudy_material(self.registry_name, wavelength_um=self.wavelength_um)
        if paper_d and self.key in PAPER_TABLE1_D:
            mat = replace(mat, d_voigt_pm_v=PAPER_TABLE1_D[self.key])
        return mat

    def _kwargs(self) -> dict:
        kw = si_figure_kwargs_from_material(self.material())
        kw.pop("point_group", None)
        return kw

    def polar_curves(self, n_phi: int = 361) -> dict:
        """{theta_deg: {phi_deg, intensity_p, intensity_s}} -- panels (c)/(d) of each SI figure."""
        kw = self._kwargs()
        return {th: si_polarimetry_curve(self.point_group, theta_deg=th, n_phi=n_phi, **kw)
                for th in self.thetas_deg}

    def effective_index(self):
        """Panel (b): effective index of the transmitted mode vs incident angle (or None if N/A)."""
        kw = self._kwargs()
        keys = ("n_omega", "n_2omega", "n_omega_e", "n_2omega_e")
        try:
            return si_effective_index_curve(self.point_group, **{k: kw[k] for k in keys if k in kw})
        except Exception:
            return None

    def params_text(self) -> str:
        """Human-readable dump of the case parameters, for printing in the notebook with provenance."""
        m = self.material()
        d = np.asarray(m.d_voigt_pm_v, dtype=complex)
        nz = [f"d[{i+1}{j+1}]={_fmt_c(d[i, j])}" for i in range(3) for j in range(6)
              if abs(d[i, j]) > 1e-9]
        ori = np.asarray(m.orientation.z_axes_in_lab, dtype=float)
        ew = np.asarray(m.epsilon_omega, dtype=complex)
        e2 = np.asarray(m.epsilon_2omega, dtype=complex)
        lines = [
            f"case          : {self.fig} — {self.title}",
            f"crystal/pg    : {m.name}   point group {self.point_group}   surface {self.surface}",
            f"wavelength    : {self.wavelength_um:.3f} µm (fundamental); ε interpolated to this λ",
            f"orientation Z (crystal axes in lab, rows Z1/Z2/Z3):",
            f"                {np.array2string(ori, precision=4, suppress_small=True)}",
            f"ε(ω) diag     : {np.array2string(np.diag(ew), precision=4, suppress_small=True)}",
            f"ε(2ω) diag    : {np.array2string(np.diag(e2), precision=4, suppress_small=True)}",
            f"d (pm/V)      : {', '.join(nz)}",
            f"provenance    : {self.provenance}",
        ]
        return "\n".join(lines)


def _fmt_c(z: complex) -> str:
    z = complex(z)
    if abs(z.imag) < 1e-9:
        return f"{z.real:g}"
    return f"{z.real:g}{'+' if z.imag >= 0 else ''}{z.imag:g}i"


SI_CASES = {
    "fig4": SICase(
        key="fig4", fig="Fig. 4", title="GaAs (111), 800 nm",
        registry_name="GaAs (111) (800 nm)", wavelength_um=0.800,
        point_group="-43m", surface="(111)",
        provenance="SHAARP.si V1.04 case button 'GaAs (111)' (source_key GaAs111λ800)",
    ),
    "fig5": SICase(
        key="fig5", fig="Fig. 5", title="LiNbO₃ (11̄20), 800 nm",
        registry_name="LiNbO3 x-cut (1550 nm)", wavelength_um=0.800,
        point_group="3m", surface="(11̄20) x-cut",
        provenance="registry 'LiNbO3 x-cut' (source_key LiNbO3xCutλ1550); ε interpolated to 800 nm",
        thetas_deg=(15.0, 30.0, 45.0),   # paper Fig 5(c,d) angles; I_s panel is scaled x15
    ),
    "fig6": SICase(
        key="fig6", fig="Fig. 6", title="KTP (100), 800 nm",
        registry_name="KTP x-cut", wavelength_um=0.800,
        point_group="mm2", surface="(100) x-cut",
        provenance="SHAARP.si V1.04 case button 'KTP (100)' (source_key KTPxCut); ε to 800 nm",
        thetas_deg=(20.0, 30.0, 40.0),   # paper Fig 6(c,d) angles; I_p panel is scaled x10
    ),
    "fig7": SICase(
        key="fig7", fig="Fig. 7", title="TaAs (112) Weyl semimetal, 800 nm",
        registry_name="TaAs (112)", wavelength_um=0.800,
        point_group="4mm", surface="(112)",
        provenance="SHAARP.si V1.04 case study 'TaAs (112)' inline constants (complex d, complex ε)",
        thetas_deg=(45.0,),              # paper Fig 7(c): single representative angle, I_L2 scaled x20
    ),
}


# ======================================================================================
# ML paper (multilayer / Maker fringes / interference)
# ======================================================================================

def ml_fig3_system(thickness_um: float = 300.0):
    """300 um X-cut quartz, 1064 nm, air both sides (Fig. 3 Maker fringes, HH vs JK)."""
    return build_casestudy_ml_system(
        "Quartz x-cut (1064 nm)", thickness_um=thickness_um, wavelength_um=1.064,
        substrate_n_omega=1.0, substrate_n_2omega=1.0)


def ml_fig4_system(*, au: bool):
    """123.6 um Z-cut quartz, 800 nm; optionally + backside Au (Fig. 4 FMR effect).

    NOTE: this is the UNCOATED panel-(b) geometry (h = 123.6 um). The published panel (d) was
    computed at a DIFFERENT spot of the sample — use :func:`ml_fig4d_system` for it.
    """
    s_bare = build_casestudy_ml_system(
        "Quartz z-cut (800 nm)", thickness_um=123.6, wavelength_um=0.8,
        substrate_n_omega=1.0, substrate_n_2omega=1.0)
    if not au:
        return s_bare
    au_mat = build_casestudy_material("Au coating (800 nm)", wavelength_um=0.8)
    layers = list(s_bare.layers)
    au_layer = Layer(name="Au coating", material=au_mat, thickness_um=0.0139, shg_active=False)
    return replace(s_bare, layers=layers[:-1] + [au_layer, layers[-1]])


def ml_fig4d_system(*, h_quartz_um: float = 121.0, au_um: float = 0.0139,
                    wavelength_um: float = 0.8):
    """The PUBLISHED Fig-4(d) system: Z-cut quartz (121 um spot) + 13.9 nm backside Au.

    Provenance: h1 = 121 um is the Au-coated SPOT thickness, from the figure archive itself
    (NAMING NOTE: the archive folder 'fig3/' and its notebooks' internal 'Fig. 3d' labels predate
    the final figure numbering — they hold the material of PUBLISHED Figures 3 AND 4; always cite
    the published label. Sources: fig3/dataRui100AuFiner121um.mx and the generating cell
    ``/. {h1 -> 121, ...}``; the
    SI documents ~10 um thickness variation across the 10 mm sample — the uncoated panel-(b) spot
    fitted 123.6 um). Au = 13.9 nm per the caption AND the ellipsometry fit ('Au layer_13p9nm.txt',
    whose n,k == the registry preset): the generating cell's saved state substitutes h2 = 0.00139 um
    (1.39 nm), but that value is DISPROVEN by the published panel itself — it gives fringe
    modulation 0.12 in the 30-40 deg inset vs the published ~0.5 (13.9 nm gives 0.40); the saved
    cell state is a stale exploration. At (121 um, 13.9 nm) the published ~0.35-0.4 central bump
    at normal incidence and the +-38-40 deg experiment peaks are reproduced.
    ``wavelength_um`` is exposed for the FMR+theta+h+lambda averaging (SI: +-5 nm bandwidth).
    """
    s = build_casestudy_ml_system(
        "Quartz z-cut (800 nm)", thickness_um=h_quartz_um, wavelength_um=wavelength_um,
        substrate_n_omega=1.0, substrate_n_2omega=1.0)
    au_mat = build_casestudy_material("Au coating (800 nm)", wavelength_um=wavelength_um)
    layers = list(s.layers)
    au_layer = Layer(name="Au coating", material=au_mat, thickness_um=au_um, shg_active=False)
    return replace(s, layers=layers[:-1] + [au_layer, layers[-1]])


def ml_fig4d_bare_system(*, h_quartz_um: float = 121.0, wavelength_um: float = 0.8):
    """Fig 4(d)'s quartz slab WITHOUT the backside Au (air / quartz / air) — the single-slab HH/JK
    BASELINE that the full-MR Au curve departs from.

    Provenance (from fig3/setup.nb signatures): the 4-layer Au stack is only ever run
    through ``fMLNL`` (which has NO assumption variant → it is the full solve = FMR). The JK/HH/FMR
    assumption curves come from the SINGLE-SLAB functions ``fs0NL``/``fsNLHH``/``fsNL`` (2 media). So the
    published panel-(d) "HH" is the bare-quartz single-slab HH, NOT a multilayer-HH-with-Au — reproducing
    it on the Au stack (the old benchmark bug) inflated HH to ~0.88·FMR instead of sitting below it."""
    return build_casestudy_ml_system(
        "Quartz z-cut (800 nm)", thickness_um=h_quartz_um, wavelength_um=wavelength_um,
        substrate_n_omega=1.0, substrate_n_2omega=1.0)


def ml_fig6_system():
    """ZnO (159 nm) // Pt (200 nm) // Al₂O₃ substrate, 1550 nm (Fig. 6 heterostructure).

    Thicknesses are the manuscript's stated values ("159 nm ZnO // 200 nm Pt"); Al₂O₃ is the
    substrate half-space. Only ZnO is SHG-active (Pt metal + Al₂O₃ centrosymmetric are inactive).

    ARCHITECTURE-FIDELITY NOTE. This builder models Al₂O₃ as a SEMI-INFINITE
    exit medium (thickness_um=None). The RELEASED Mathematica .ml GUI could not express that:
    it forces the exit half-space to air (`SHAARP.ml.nb:425-439` overwrites slot
    materialnumber+2 with `mbot = setMater@Air[]`; the layer selector `Range[2,
    materialnumber+1]` cannot reach it), so the original's Fig-6 run must have had Al₂O₃ as a
    FINITE numbered layer above air — the same shape the MoS₂ notebook uses ("500um for
    Al2O3"). No Fig-6 generator notebook survives in the archive to pin the thickness, and the
    panel is REFLECTED SHG at 45 deg behind ~200 nm of Pt, so the bottom boundary is optically
    almost irrelevant here. Recorded rather than silently reproduced: the numbers agree, the
    architecture differs. (SHAARP.py deliberately allows an arbitrary exit medium — the ENGINE
    always did, `setup.nb:6421`: "wSub - a list of waves into Substrate (can be Air)".)
    """
    lam = 1.55
    air = build_casestudy_material("Air", wavelength_um=lam)
    zno = build_casestudy_material("ZnO (001)", wavelength_um=lam)
    pt = build_casestudy_material("Pt (111) (1550 nm)", wavelength_um=lam)
    al2o3 = build_casestudy_material("Al2O3 (0001) (1550 nm)", wavelength_um=lam)
    layers = [
        Layer(name="air", material=air, thickness_um=None, shg_active=False),
        Layer(name="ZnO", material=zno, thickness_um=0.159, shg_active=True),
        Layer(name="Pt", material=pt, thickness_um=0.200, shg_active=False),
        Layer(name="Al₂O₃ substrate", material=al2o3, thickness_um=None, shg_active=False),
    ]
    return MultilayerSystem(wavelength_um=lam, layers=layers)


def ml_fig7_case(case: int):
    """Fig 7: SHG interference in LiNbO₃(21̄1̄0) // quartz(001) heterostructures, 1550 nm.

    The paper's four stacking cases (panel b), probed by TRANSMITTED SHG polarimetry near normal
    incidence (panels c, d — I_L1(φ) ×1 and I_L2(φ), plus the I_L1(φ=0°) bar comparison):

      Case 1 — LNO alone, polar axis **P ∥ −L1**;
      Case 2 — LNO (P ∥ −L1) // quartz(001) with [100] ∥ +L1 (constructive);
      Case 3 — LNO (P ∥ +L1) // quartz(001) with [100] ∥ +L1 (destructive);
      Case 4 — LNO alone, P ∥ +L1.

    The registry 'LiNbO3 x-cut (1550 nm)' natively has c = P along −L1 (verified from its
    orientation matrix), so Cases 3/4 flip P with a 180° sample azimuth about L3. The registry
    'Quartz z-cut (800 nm)' natively has [100] ∥ +L1 and c ∥ L3 — the paper's quartz(001)[100].
    Thicknesses: LNO 50 µm, quartz 35 µm (manuscript values).
    """
    lam = 1.55
    air = build_casestudy_material("Air", wavelength_um=lam)
    lno = build_casestudy_material("LiNbO3 x-cut (1550 nm)", wavelength_um=lam)
    if case in (3, 4):  # flip the in-plane polar axis P: −L1 → +L1
        lno = replace(lno, orientation=lno.orientation.with_lab_azimuth_deg(180.0))
    quartz = build_casestudy_material("Quartz z-cut (800 nm)", wavelength_um=lam)
    layers = [Layer(name="air", material=air, thickness_um=None, shg_active=False),
              Layer(name="LiNbO₃ (21̄1̄0)", material=lno, thickness_um=50.0, shg_active=True)]
    if case in (2, 3):
        layers.append(Layer(name="quartz (001)", material=quartz, thickness_um=35.0, shg_active=True))
    layers.append(Layer(name="air out", material=air, thickness_um=None, shg_active=False))
    return MultilayerSystem(wavelength_um=lam, layers=layers,
                            polarimetry=Polarimetry(theta_deg=0.5, phi_deg=0.0))


def ml_maker(system, assumption: str, *, th_min: float = 0.0, th_max: float = 65.0,
             step: float = 0.1, fmr_submode: str = "Forward + Backward waves"):
    """Transmitted p-p Maker fringes I(theta_i) for a stack, via the GUI compute path."""
    r = compute_ml_gui_result(
        "Maker Fringes", system=system, theta_min_deg=th_min, theta_max_deg=th_max,
        theta_step_deg=step, assumption=assumption, fmr_submode=fmr_submode)
    return (np.asarray(r.numeric["theta_deg"], dtype=float),
            np.asarray(r.numeric["parallel_intensity"], dtype=float))


# --- Fig 5 (transmitted polarimetry, single crystals) + Fig 6 (reflected polarimetry, heterostructure) ---

# Fig 5 single crystals: LiNbO3 (11-20)=x-cut, LiNbO3 (0001)=z-cut, KTP x-cut and y-cut, all @ 1550 nm.
FIG5_CRYSTALS = {
    "LiNbO₃ (11̄20)": "LiNbO3 x-cut (1550 nm)",
    "LiNbO₃ (0001)": "LiNbO3 z-cut (1550 nm)",
    "KTP (x-cut)": "KTP x-cut",
    "KTP (y-cut)": "KTP y-cut",
}


def ml_single_crystal_system(registry_name: str, *, thickness_um: float = 500.0,
                             wavelength_um: float = 1.55):
    """A thick single-crystal slab in air (air / crystal / air) for transmitted-SHG polarimetry."""
    lam = wavelength_um
    air = build_casestudy_material("Air", wavelength_um=lam)
    xtal = build_casestudy_material(registry_name, wavelength_um=lam)
    layers = [
        Layer(name="air", material=air, thickness_um=None, shg_active=False),
        Layer(name=registry_name, material=xtal, thickness_um=thickness_um, shg_active=True),
        Layer(name="air out", material=air, thickness_um=None, shg_active=False),
    ]
    return MultilayerSystem(wavelength_um=lam, layers=layers)


def ml_polar(system, *, theta_deg: float, channel: str, n_phi: int = 181, mrassumption: int = 0):
    """(phi_deg, I_p, I_s) reflected- or transmitted-SHG polarimetry vs incident polarization phi.

    ``channel`` = 'reflected' (Fig 6 heterostructure) or 'transmitted' (Fig 5 single crystals).
    """
    cur = ml_polarimetry_curve(system, n_phi=n_phi, theta_deg=theta_deg, mrassumption=mrassumption)
    if channel == "reflected":
        return cur["phi_deg"], cur["intensity_p"], cur["intensity_s"]
    if channel == "transmitted":
        return cur["phi_deg"], cur["intensity_p_trans"], cur["intensity_s_trans"]
    raise ValueError("channel must be 'reflected' or 'transmitted'")


# --- Fig S7 (twist bilayer MoS2, rotational-anisotropy reflected SHG) ---

# The registry MoS2 is a MONOLAYER (point group -6m2, d22-dominant); a twist bilayer stacks two of
# them with a relative azimuth. Monolayer thickness ~0.65 nm keeps the two layers in the coherent-
# addition regime, so the twist bilayer's SHG is LARGER than either monolayer (as the paper shows).
MOS2_MONOLAYER_UM = 0.00065


def ml_mos2_system(twists_deg, *, monolayer_um: float = MOS2_MONOLAYER_UM,
                   wavelength_um: float = 1.064, theta_deg: float = 45.0):
    """Air / stacked MoS2 monolayers / air. ``twists_deg`` lists the layers TOP->BOTTOM, each rotated
    about the surface normal (e.g. [0.0] = single monolayer; [25.0, 0.0] = 25° twist bilayer)."""
    lam = wavelength_um
    air = build_casestudy_material("Air", wavelength_um=lam)
    layers = [Layer(name="air", material=air, thickness_um=None, shg_active=False)]
    for tw in twists_deg:
        m = build_casestudy_material("MoS2", wavelength_um=lam)
        m = replace(m, orientation=m.orientation.with_lab_azimuth_deg(float(tw)))
        layers.append(Layer(name=f"MoS₂ {tw:g}°", material=m, thickness_um=monolayer_um, shg_active=True))
    layers.append(Layer(name="substrate", material=air, thickness_um=None, shg_active=False))
    return MultilayerSystem(wavelength_um=lam, layers=layers,
                            polarimetry=Polarimetry(theta_deg=theta_deg, phi_deg=0.0))


def ml_ra_reflected_polar(system, *, n_azimuth: int = 181):
    """Rotational-anisotropy (RA) reflected SHG: sweep the SAMPLE azimuth (fixed p-polarized
    incidence along L1) and read the reflected p (fixed analyzer 0°) and s (fixed analyzer 90°)
    2ω intensities -- the observable behind SI Fig S7(b,c). Returns (azimuth_deg, I_p, I_s)."""
    from shaarp.multilayer_shg_boundary import (
        reflected_2omega_jones_sp, solve_multilayer_shg_from_system_polarimetry)
    az = np.linspace(0.0, 360.0, int(n_azimuth))
    ip = np.empty(az.shape, dtype=float)
    is_ = np.empty(az.shape, dtype=float)
    for k, a in enumerate(az):
        s = with_sample_azimuth_deg(system, float(a), rotate_substrate=False)
        res = solve_multilayer_shg_from_system_polarimetry(s)
        r_s, r_p = reflected_2omega_jones_sp(res.shg)
        ip[k] = float(abs(r_p) ** 2)
        is_[k] = float(abs(r_s) ** 2)
    return az, ip, is_


# Twist-bilayer MoS2 cases for SI Fig S7(b,c): monolayer at 0°, monolayer at 25°, and the bilayer.
MOS2_CASES = (
    ("MoS₂ (0°)", [0.0], "#3465c0"),
    ("MoS₂ (25°)", [25.0], "#e8a33d"),
    ("MoS₂(25°)//MoS₂(0°)", [25.0, 0.0], "#7cb342"),
)


def ml_mos2_on_sapphire(twists_deg, *, substrate_um: float, wavelength_um: float = 1.064,
                        theta_deg: float = 45.0):
    """MoS2 monolayer(s) on a FINITE Al2O3 (sapphire) wafer: air / MoS2 x N / Al2O3(t) / air.

    Published SI Fig S7(a) sweeps the SUBSTRATE thickness t (480-520 um): the sapphire slab's
    Fabry-Perot phase modulates the coherence between the layers' SHG. (Substrate identity from
    the author's own kappa notebooks — `minmaxkappa.nb` labels the sweep variable t_Al2O3.)
    """
    lam = wavelength_um
    air = build_casestudy_material("Air", wavelength_um=lam)
    al2o3 = build_casestudy_material("Al2O3 (0001) (1550 nm)", wavelength_um=lam)
    layers = [Layer(name="air", material=air, thickness_um=None, shg_active=False)]
    for tw in twists_deg:
        m = build_casestudy_material("MoS2", wavelength_um=lam)
        m = replace(m, orientation=m.orientation.with_lab_azimuth_deg(float(tw)))
        layers.append(Layer(name=f"MoS₂ {tw:g}°", material=m, thickness_um=MOS2_MONOLAYER_UM,
                            shg_active=True))
    layers.append(Layer(name="Al₂O₃ wafer", material=al2o3, thickness_um=float(substrate_um),
                        shg_active=False))
    layers.append(Layer(name="air out", material=air, thickness_um=None, shg_active=False))
    return MultilayerSystem(wavelength_um=lam, layers=layers,
                            polarimetry=Polarimetry(theta_deg=theta_deg, phi_deg=0.0))


def ml_mos2_kappa_sweep(t_um, *, theta_deg: float = 45.0):
    """Published SI Fig S7(a): kappa(dTheta=0, t) = I_bilayer/(2 I_monolayer) - 1 vs substrate
    thickness t — the coherent interference cross-term of the two monolayers' SHG (his
    `Heterostructure Test.nb` definition kappa = (I_bi - I_top - I_bot)/(2 sqrt(I_top I_bot)),
    which for identical layers reduces to I_bi/(2 I_mono) - 1; kappa = 1 = fully coherent).

    Reflected SHG at fixed azimuth/polarization (phi = 0), matching the notebooks' Is0/I10 usage.
    """
    from shaarp.multilayer_shg_boundary import (
        reflected_2omega_jones_sp, solve_multilayer_shg_from_system_polarimetry)

    def refl_ip(system, azimuth):
        s = with_sample_azimuth_deg(system, float(azimuth), rotate_substrate=False)
        res = solve_multilayer_shg_from_system_polarimetry(s)
        r_s, r_p = reflected_2omega_jones_sp(res.shg)
        return float(abs(r_p) ** 2)

    t_um = np.asarray(t_um, dtype=float)
    # kappa is azimuth-independent for dTheta = 0 (mono and bilayer share the same six-fold
    # pattern), but the registry orientation puts a symmetry NULL at azimuth 0 -- evaluate at the
    # monolayer's lobe azimuth, found once (the pattern does not rotate with t).
    probe = ml_mos2_on_sapphire([0.0], substrate_um=float(t_um[0]), theta_deg=theta_deg)
    az_star = max((az for az in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)),
                  key=lambda az: refl_ip(probe, az))
    kappa = np.empty(t_um.shape)
    for k, t in enumerate(t_um):
        i_mono = refl_ip(ml_mos2_on_sapphire([0.0], substrate_um=float(t), theta_deg=theta_deg), az_star)
        i_bi = refl_ip(ml_mos2_on_sapphire([0.0, 0.0], substrate_um=float(t), theta_deg=theta_deg), az_star)
        kappa[k] = i_bi / (2.0 * i_mono) - 1.0
    return t_um, kappa


def ml_mos2_kappa_twist_sweep(twists_deg, *, substrate_um: float, theta_deg: float = 45.0,
                              n_azimuth: int = 61):
    """Published Fig 8c: SHG coherence kappa vs TWIST angle dTheta between two MoS2 monolayers, at a
    FIXED Al2O3 substrate thickness t (the `minmaxkappa.nb`, twist panel).

    kappa(dTheta) = I_bi/(2 I_mono) - 1 -- the identical-monolayer limit of
    kappa = (I_bi - I_a - I_b)/(2 sqrt(I_a I_b)) -- where I_bi is the PEAK-over-azimuth reflected SHG
    of the [dTheta, 0] twisted bilayer and I_mono the peak-over-azimuth of the [0] monolayer (Rui reads
    `MaxValue[...]` over the polarization angle; the peak lobe is azimuth-swept here, matching
    :func:`ml_mos2_kappa_sweep`). The ideal fully-coherent limit is cos(3 dTheta) (the three-fold
    monolayer symmetry); the substrate Fabry-Perot phase pushes the real kappa off it (the red band
    over t is that thickness sensitivity)."""
    from shaarp.multilayer_shg_boundary import (
        reflected_2omega_jones_sp, solve_multilayer_shg_from_system_polarimetry)

    def peak_ip(twists):
        sysm = ml_mos2_on_sapphire(list(twists), substrate_um=float(substrate_um), theta_deg=theta_deg)
        best = 0.0
        for az in np.linspace(0.0, 60.0, n_azimuth):   # 3-fold: one 60-deg azimuth period holds the peak
            s = with_sample_azimuth_deg(sysm, float(az), rotate_substrate=False)
            _r_s, r_p = reflected_2omega_jones_sp(solve_multilayer_shg_from_system_polarimetry(s).shg)
            best = max(best, float(abs(r_p) ** 2))
        return best

    i_mono = peak_ip([0.0])
    twists = np.asarray(twists_deg, dtype=float)
    kappa = np.empty(twists.shape)
    for k, dt in enumerate(twists):
        i_bi = peak_ip([float(dt), 0.0])
        kappa[k] = (i_bi / (2.0 * i_mono) - 1.0) if i_mono else 0.0
    return twists, kappa


@dataclass(frozen=True)
class MLCase:
    """A multilayer/Maker/interference case from the SHAARP.ml 2024 paper."""

    key: str
    fig: str
    title: str
    wavelength_um: float
    provenance: str
    system_builder: object = field(repr=False, default=None)


ML_CASES = {
    "fig3": MLCase(
        key="fig3", fig="Fig. 3", title="300 µm X-cut quartz, 1064 nm — Maker fringes HH/JK",
        wavelength_um=1.064,
        provenance="registry 'Quartz x-cut (1064 nm)'; 300 µm from fig3/run.nb; Herman-Hayden benchmark",
        system_builder=ml_fig3_system,
    ),
    "fig4": MLCase(
        key="fig4", fig="Fig. 4", title="123.6 µm Z-cut quartz + 13.9 nm Au, 800 nm — FMR",
        wavelength_um=0.800,
        provenance="registry 'Quartz z-cut (800 nm)' + 'Au coating (800 nm)'; 123.6 µm/13.9 nm from paper",
        system_builder=ml_fig4_system,
    ),
    "fig6": MLCase(
        key="fig6", fig="Fig. 6", title="ZnO//Pt//Al₂O₃ heterostructure, 1550 nm",
        wavelength_um=1.550,
        provenance="registry ZnO (001)/Pt (111)/Al₂O₃ (0001); 159 nm ZnO // 200 nm Pt from paper",
        system_builder=ml_fig6_system,
    ),
    "fig7": MLCase(
        key="fig7", fig="Fig. 7", title="LiNbO₃(21̄1̄0)//quartz(001) SHG interference (4 cases), 1550 nm",
        wavelength_um=1.550,
        provenance="registry LiNbO3 x-cut (1550, P∥−L1 native) + Quartz z-cut ([100]∥+L1 native); "
                   "50 µm LNO // 35 µm quartz from paper; P flip = 180° sample azimuth",
        system_builder=ml_fig7_case,
    ),
}


if __name__ == "__main__":
    # Smoke: build every case and report a one-line shape summary (no plotting).
    print("SI cases:")
    for c in SI_CASES.values():
        cur = c.polar_curves(n_phi=181)
        pk = max(float(np.max(cur[t]["intensity_p"])) for t in c.thetas_deg)
        print(f"  {c.fig:8s} {c.title:34s} peak I_p={pk:.3e}  [{c.point_group}]")
    print("ML cases:")
    for c in ML_CASES.values():
        sysobj = c.system_builder() if c.key != "fig4" else c.system_builder(au=True)
        th, i = ml_maker(sysobj, ASSUMPTION_HH, th_max=45.0, step=1.0)
        print(f"  {c.fig:8s} {c.title:44s} maker peak={float(np.max(i)):.3e}  ({len(sysobj.layers)} layers)")
