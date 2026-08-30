"""T3: replicate the PUBLISHED figures of both SHAARP papers through the GUI compute path.

The tightened-evaluation ladder (T1 sensitivity -> T2 fidelity -> T3 replication): T1/T2 verify
enumerable dimensions; T3 interrogates the product against ground truth the papers already
published. Every curve here is produced by the SAME pure compute calls the desktop GUI's Update
button makes (compute_*/si_polarimetry_curve/si_effective_index_curve) with the case-study
materials that came verbatim from the original notebooks -- so a shape mismatch against the
paper panel is a product defect, not a harness artifact.

Targets (verbatim from the manuscripts on disk):
  * SHAARP.si 2022 (npj Comput. Mater. 8, 246) Figure 4(b-d): GaAs (111), 800 nm --
    effective complex indices vs theta_i (flat: cubic), and I_p/I_s reflected-SHG
    polarimetry vs polarizer angle phi at theta_i = 0/15/30/45 deg. Panel (e)'s claim
    ("eps_R approximation underestimates d36 by nearly 20%") is checked numerically.
  * SHAARP.ml 2024 (npj Comput. Mater. 10, 64) Figure 3(b,c): 300 um X-cut quartz,
    1064 nm, p-p Maker fringes -- SHAARP(HH) vs SHAARP(JK) (near-identical envelopes,
    fine-fringe differences in the zoom).
  * SHAARP.ml 2024 Figure 4(b,d): 123.6 um Z-cut quartz at 800 nm, uncoated (JK/HH/FMR
    agree in envelope) vs 13.9 nm backside-Au-coated (FMR departs strongly -- the paper's
    central point about multiple reflections).

FINDINGS this replication already produced: F31 (rotated-isotropic eps float noise
-> NaN polarimetry for GaAs(111)) and F32 (Im(eps)/Im(d) silently dropped by the curve path --
the paper's own eps_R approximation applied without consent). Both fixed in shaarp_gui and
verified against solve_single_interface_shg to 5e-14.

KNOWN paper-vs-software deviation (documented, NOT a defect): the PAPER's Fig. 4(d) shows a
central bump (~30% of the wing peak) near theta_i = 0 for the Au-coated slab. The released
software's own preset Au (setup.nb Au-lambda-800: n_w = 3.4795+2.9211i, n_2w = 2.2518+1.5966i,
transcribed verbatim in benchmarks/quartz_au_docs_case.py and VALIDATED against live SHAARP.ml
to 3.6e-9 over 0-45 deg including theta ~ 0) produces only a small central feature (~1% of the
wing peak) -- so the figure's bump stems from the authors' journal-SI experimental parameters /
normalization pipeline (ellipsometry note absent from the arXiv SI on disk), not from anything
in the released notebook presets. The fringe-amplification signature of FMR (the panel's actual
point) IS replicated.

Run: python scripts/replicate_paper_figures.py -> PNGs in build/paper_replication/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shaarp.casestudy_materials import (
    build_casestudy_material,
    build_casestudy_ml_system,
)
from shaarp.config import Layer
from shaarp.shaarp_gui import (
    compute_ml_gui_result,
    si_effective_index_curve,
    si_figure_kwargs_from_material,
    si_polarimetry_curve,
)

OUT = ROOT / "build" / "paper_replication"
OUT.mkdir(parents=True, exist_ok=True)

ASSUMPTION_FMR = "Full Multiple Reflections (FMR)"
ASSUMPTION_JK = "Jerphagnon & Kurtz Assumption (No MR)"
ASSUMPTION_HH = "Herman & Hayden Assumption (MR only for 2ω Homo Waves)"


# --------------------------------------------------------------------------------------
# SI paper Figure 4(b-d): GaAs (111), 800 nm
# --------------------------------------------------------------------------------------

def replicate_si_fig4() -> None:
    mat = build_casestudy_material("GaAs (111) (800 nm)", wavelength_um=0.8)
    kw = si_figure_kwargs_from_material(mat)
    pg = kw.pop("point_group")
    thetas = [0.0, 15.0, 30.0, 45.0]
    colors = ["#3465c0", "#e8a33d", "#7cb342", "#e0532f"]
    curves = {th: si_polarimetry_curve(pg, theta_deg=th, n_phi=361, **kw) for th in thetas}
    idx = si_effective_index_curve(
        pg, **{k: kw[k] for k in ("n_omega", "n_2omega", "n_omega_e", "n_2omega_e") if k in kw})

    fig = plt.figure(figsize=(13.0, 4.1))
    ax_b = fig.add_subplot(1, 3, 1)
    ax_b.plot(idx["theta_deg"], idx["n_ordinary"], "b-", lw=1.7)
    n_im = float(np.imag(np.sqrt(complex(np.asarray(mat.epsilon_omega, dtype=complex)[0, 0]))))
    ax_i = ax_b.twinx()
    ax_i.plot(idx["theta_deg"], np.full_like(idx["theta_deg"], n_im), color="crimson", ls="--", lw=1.4)
    ax_i.set_ylim(0, 0.20)
    ax_i.set_ylabel(r"$\tilde{n}_I$", color="crimson")
    ax_b.set_ylim(0, 8)
    ax_b.set_xlim(0, 89)
    ax_b.set_xlabel(r"Incident Angle, $\theta_i$ ($^\circ$)")
    ax_b.set_ylabel(r"$\tilde{n}_R$", color="b")
    ax_b.set_title("(b) effective index of the transmitted mode")

    for j, key, lab in ((2, "intensity_p", r"(c) $I_p^{2\omega}(\varphi)$"),
                        (3, "intensity_s", r"(d) $I_s^{2\omega}(\varphi)$")):
        ax = fig.add_subplot(1, 3, j, projection="polar")
        mx = max(float(np.max(np.asarray(curves[th][key]))) for th in thetas)
        for th, col in zip(thetas, colors):
            c = curves[th]
            ph = np.radians(np.asarray(c["phi_deg"], dtype=float))
            ax.plot(ph, np.asarray(c[key], dtype=float) / mx, color=col, lw=1.5,
                    label=rf"$\theta_i$={th:g}$^\circ$")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["0.5", "1.0"], fontsize=7)
        ax.set_title(lab, pad=16)
        if j == 3:
            ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=8)

    fig.suptitle("SHAARP.py GUI-path replication -- SHAARP.si 2022 Fig. 4(b-d), GaAs (111) at 800 nm",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "si2022_fig4_gaas111.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # panel (e) quantitative claim: fitting with the eps_R (real-eps) expression underestimates
    # d36 by ~20% (paper bars: 216/267 = 0.81). Same-d intensity ratio -> fit-equivalent d ratio.
    kw_real = dict(kw)
    kw_real["eps_omega_principal"] = tuple(float(np.real(x)) for x in kw["eps_omega_principal"])
    kw_real["eps_2omega_principal"] = tuple(float(np.real(x)) for x in kw["eps_2omega_principal"])
    c_full = si_polarimetry_curve(pg, theta_deg=30.0, n_phi=361, **kw)
    c_real = si_polarimetry_curve(pg, theta_deg=30.0, n_phi=361, **kw_real)
    ratio = float(np.sqrt(np.max(c_full["intensity_p"]) / np.max(c_real["intensity_p"])))
    print(f"SI Fig4(e) check: eps_R-fit d36 / full d36 = {ratio:.3f}  (paper: 216/267 = 0.809)")
    assert 0.75 < ratio < 0.90, "eps_R approximation effect no longer matches the paper's ~20%"


# --------------------------------------------------------------------------------------
# ML paper Figure 3(b,c): 300 um X-cut quartz, 1064 nm, p-p Maker fringes, HH vs JK
# --------------------------------------------------------------------------------------

def _maker(system, assumption: str, th_max: float = 65.0, step: float = 0.1,
           fmr_submode: str = "Forward + Backward waves"):
    # full FMR includes the BACKWARD nonlinear waves -- with the Au backside mirror those are
    # exactly what builds Fig. 4(d)'s central bump (the default forward-only submode misses it)
    r = compute_ml_gui_result("Maker Fringes", system=system, theta_min_deg=0.0,
                              theta_max_deg=th_max, theta_step_deg=step, assumption=assumption,
                              fmr_submode=fmr_submode)
    return (np.asarray(r.numeric["theta_deg"], dtype=float),
            np.asarray(r.numeric["parallel_intensity"], dtype=float))


def replicate_ml_fig3() -> None:
    s = build_casestudy_ml_system("Quartz x-cut (1064 nm)", thickness_um=300.0,
                                  wavelength_um=1.064,
                                  substrate_n_omega=1.0, substrate_n_2omega=1.0)
    th_hh, i_hh = _maker(s, ASSUMPTION_HH)
    th_jk, i_jk = _maker(s, ASSUMPTION_JK)
    mx = max(float(np.max(i_hh)), float(np.max(i_jk)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.0), width_ratios=[2.1, 1.0])
    ax1.plot(th_hh, i_hh / mx, color="#3465c0", lw=0.9, label="SHAARP.py (HH)")
    ax1.plot(th_jk, i_jk / mx, color="#e0532f", lw=0.9, alpha=0.75, label="SHAARP.py (JK)")
    ax1.set_xlabel(r"Incident Angle, $\theta_i$ ($^\circ$)")
    ax1.set_ylabel(r"$I_{pp}^{2\omega}$ (a.u.)")
    ax1.set_title("(b) Maker fringes, 300 $\\mu$m X-cut quartz, 1064 nm, p-p")
    ax1.legend(fontsize=8)
    zoom = (22.0, 30.0)  # the paper's dashed box: HH fine fringes at 20-30 deg, absent for JK
    m = (th_hh >= zoom[0]) & (th_hh <= zoom[1])
    ax2.plot(th_hh[m], (i_hh / mx)[m], color="#3465c0", lw=1.1, label="HH")
    ax2.plot(th_jk[m], (i_jk / mx)[m], color="#e0532f", lw=1.1, alpha=0.75, label="JK")
    ax2.set_xlim(*zoom)
    ax2.set_xlabel(r"$\theta_i$ ($^\circ$)")
    ax2.set_title("(c) magnified region")
    fig.suptitle("SHAARP.py GUI-path replication -- SHAARP.ml 2024 Fig. 3(b,c)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "ml2024_fig3_xcut_quartz.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"ML Fig3: HH/JK peak ratio = {float(np.max(i_hh) / np.max(i_jk)):.4f} "
          f"(near-identical envelopes expected)")


# --------------------------------------------------------------------------------------
# ML paper Figure 4(b,d): 123.6 um Z-cut quartz at 800 nm, uncoated vs 13.9 nm backside Au
# --------------------------------------------------------------------------------------

def replicate_ml_fig4() -> None:
    s_bare = build_casestudy_ml_system("Quartz z-cut (800 nm)", thickness_um=123.6,
                                       wavelength_um=0.8,
                                       substrate_n_omega=1.0, substrate_n_2omega=1.0)
    au = build_casestudy_material("Au coating (800 nm)", wavelength_um=0.8)
    from dataclasses import replace
    layers = list(s_bare.layers)
    au_layer = Layer(name="Au coating", material=au, thickness_um=0.0139, shg_active=False)
    s_au = replace(s_bare, layers=layers[:-1] + [au_layer, layers[-1]])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.0))
    for ax, system, ttl in ((axes[0], s_bare, "(b) uncoated Z-cut quartz, 123.6 $\\mu$m, 800 nm"),
                            (axes[1], s_au, "(d) + 13.9 nm backside Au coating")):
        th_f, i_f = _maker(system, ASSUMPTION_FMR, th_max=45.0)
        th_h, i_h = _maker(system, ASSUMPTION_HH, th_max=45.0)
        th_j, i_j = _maker(system, ASSUMPTION_JK, th_max=45.0)
        mx = float(np.max(i_f))
        ax.plot(th_j, i_j / mx, color="#7cb342", lw=0.8, alpha=0.8, label="JK")
        ax.plot(th_h, i_h / mx, color="#3465c0", lw=0.8, alpha=0.8, label="HH")
        ax.plot(th_f, i_f / mx, color="#e0532f", lw=0.9, label="FMR")
        ax.set_xlabel(r"Incident Angle, $\theta_i$ ($^\circ$)")
        ax.set_ylabel(r"$I_{pp}^{2\omega}$ (a.u.)")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
        # report the FMR-vs-single-pass departure (the paper's central claim: small when
        # uncoated, large with the Au mirror behind the slab)
        n = min(len(i_f), len(i_h))
        dev = float(np.max(np.abs(i_f[:n] - i_h[:n])) / mx)
        print(f"ML Fig4 {ttl[:4]}: max |FMR - HH| / max(FMR) = {dev:.3f}")
    fig.suptitle("SHAARP.py GUI-path replication -- SHAARP.ml 2024 Fig. 4(b,d)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "ml2024_fig4_zcut_quartz_au.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    replicate_si_fig4()
    replicate_ml_fig3()
    replicate_ml_fig4()
    print(f"replication figures written to {OUT}")
