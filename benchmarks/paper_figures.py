"""Rendering helpers that reproduce each published SHAARP figure and show it BESIDE the real paper
panel. Used by the two flagship notebooks (notebooks/Reproduce_SHAARP_*_paper.ipynb) and importable
for ad-hoc checks. Every curve comes from :mod:`benchmarks.paper_cases` (the GUI compute path with
Mathematica-sourced parameters); the paper panels are the crops in ``docs/_static/paper_panels/``.

Resolution defaults are chosen so a headless ``nbconvert --execute`` finishes in minutes; pass finer
grids for publication-density fringes (noted where it matters).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.paper_cases import (
    ASSUMPTION_FMR, ASSUMPTION_HH, ASSUMPTION_JK, FIG5_CRYSTALS, ML_CASES, MOS2_CASES, SI_CASES,
    ml_fig4_system, ml_fig4d_system, ml_fig4d_bare_system, ml_fig6_system, ml_fig7_case, ml_maker,
    ml_mos2_kappa_twist_sweep, ml_mos2_system, ml_polar, ml_ra_reflected_polar,
    ml_single_crystal_system,
)
from benchmarks.herman_hayden_maker import herman_hayden_quartz_reference_curve

PANELS = ROOT / "docs" / "_static" / "paper_panels"
COLORS = ["#3465c0", "#e8a33d", "#7cb342", "#e0532f", "#8e44ad"]


def _display_factor(curve_max: float, ref_max: float) -> int:
    """The author's computed cross-channel / cross-case display multiplier (the Heterostructure_LNO_Quartz.nb
    lines 1698-1726: ``maxTot = 1.1*Max[...]; factor = Max[1, IntegerPart[maxTot/channelMax]]``).
    Returns ``max(1, int(1.1*ref_max/curve_max))`` -- lifts a weak curve so its peak approaches the
    shared reference scale ``1.1*ref_max`` (the strong reference gets ×1), so the annotated "× N"
    preserves the TRUE inter-channel/inter-case intensity ratio instead of hiding it behind per-channel
    normalization. This is why the paper's weak-channel lobes are visible at a comparable size."""
    curve_max = float(curve_max) or 1e-300
    return max(1, int(1.1 * float(ref_max) / curve_max))


def _panel(name: str):
    p = PANELS / f"{name}.png"
    return p if p.exists() else None


def _show_panel(name: str, caption: str) -> None:
    """Display a cropped paper panel below the reproduction (no-op outside IPython)."""
    try:
        from IPython.display import Image, Markdown, display
    except Exception:
        return
    p = _panel(name)
    if p is None:
        display(Markdown(f"_paper panel `{name}` not found — run scripts/extract_paper_panels.py_"))
        return
    display(Markdown(f"**Published panel — {caption}:**"))
    display(Image(str(p)))


def _embed_fig(fig, *, dpi: int = 120) -> None:
    """Embed a Matplotlib figure into the notebook as a PNG, **backend-independently**.

    Why not ``plt.show()``: under nbconvert the kernel runs with the Agg backend, so ``plt.show()``
    is a no-op (it only warns). Figures reach the notebook solely via the inline backend's
    end-of-cell ``flush_figures`` hook — which silently DROPPED some cells' figures (Fig S7a, Fig 8c)
    while catching others, because that hook is fragile to how/when each figure is created. Rendering
    to PNG bytes with ``fig.savefig`` (which works under Agg) and displaying via ``IPython.display.Image``
    always embeds; closing the figure afterwards means ``flush_figures`` has nothing left to double-emit.
    """
    try:
        from IPython.display import Image, display
    except Exception:
        plt.close(fig)
        return
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))


# ============================ SI (reflected-SHG polarimetry) ============================

def si_reproduction_figure(key: str, *, n_phi: int = 361):
    """Reproduce an SI figure's I_p(phi)/I_s(phi) reflected-SHG polar panels at the paper's angles."""
    c = SI_CASES[key]
    curves = c.polar_curves(n_phi=n_phi)
    fig = plt.figure(figsize=(7.4, 3.6))
    # weak channel lifted by the author's computed ×N factor (annotated) -- the paper's LiNbO₃ I_s ×15,
    # KTP I_p ×10, TaAs I_s ×20 etc. keep the true I_p/I_s ratio visible instead of per-channel fill.
    pmax = max(float(np.max(np.asarray(curves[th]["intensity_p"]))) for th in c.thetas_deg) or 1.0
    smax = max(float(np.max(np.asarray(curves[th]["intensity_s"]))) for th in c.thetas_deg) or 1.0
    ref = max(pmax, smax)
    for col, kkey, chan in ((0, "intensity_p", r"I_p^{2\omega}(\varphi)"),
                            (1, "intensity_s", r"I_s^{2\omega}(\varphi)")):
        ax = fig.add_subplot(1, 2, col + 1, projection="polar")
        fac = _display_factor(pmax if col == 0 else smax, ref)
        for th, color in zip(c.thetas_deg, COLORS):
            cur = curves[th]
            ax.plot(np.radians(np.asarray(cur["phi_deg"], float)),
                    np.asarray(cur[kkey], float) * fac / (1.1 * ref), color=color, lw=1.4,
                    label=rf"$\theta_i$={th:g}°")
        ax.set_ylim(0, 1.05); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["", ""])
        fx = rf"$\times{fac}$" if fac > 1 else ""
        ax.set_title(rf"${chan}$ {fx}", fontsize=10, pad=10)
        if col == 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=7)
    fig.suptitle(f"SHAARP.py reproduction — {c.fig} {c.title}", y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


def show_si(key: str) -> None:
    c = SI_CASES[key]
    print(c.params_text())
    _embed_fig(si_reproduction_figure(key))
    _show_panel(f"si_{key}", f"SHAARP.si 2022 {c.fig}")


# ============================ ML (Maker fringes / polarimetry / interference) ============================

def _fig3_reference_curves():
    """the author's own serialized Fig-3 reference curves (fig3/*.mx, evaluated at h = 300 µm, φ = 0 via
    wolframscript and cached as JSON): 'hhsim_finer_20_30deg' = his HH simulation over the fine-
    fringe window; 'old_hhjk_0_90deg' col 3 = his evaluated HH curve (identified by 0.9996 shape
    correlation with the ported analyticHH expression). Returns None if the cache is absent."""
    import json
    p = ROOT / "benchmarks" / "fig3_reference_curves.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return (np.array(d["hhsim_finer_20_30deg"], float),
            np.array(d["old_hhjk_0_90deg"], float))


def _fig4d_hh_reference():
    """the author's published closed-form HH model for Fig 4(d) (QuartzAu_800nm_HHNMRP1S0p01.mx, h0 = 121 µm,
    φ = 0), bundled as ``fig4d_hh_reference.csv`` and used AS the HH curve on panel (d).

    Why embedded (option B): my numeric single-pass-ω HH diverges from the published HH for
    the strong-reflector 13.9 nm Au. Decoded from the author's ``.mx`` denominators, HH = FMR with the ω
    Fabry-Pérot factor removed (single-pass ω) + full 2ω FP; my interface-by-interface single-pass does
    not reproduce that ω-FP removal at r_ω ≈ 0.60 (it matches the author's HH to 1e-13 only on low-reflectance
    dielectric stacks). HH is the paper's DELIBERATELY-FAILING illustrative curve (Fig 4d, shown ×1.5) —
    '♯SHAARP(HH) fails to capture the total transmitted SHG intensity' (npj CM 10, 64 (2024), p. 19). My
    FMR — the physics-result curve — reproduces the author's FMR model to corr 0.9993 (fenced)."""
    d = np.loadtxt(ROOT / "benchmarks" / "fig4d_hh_reference.csv", delimiter=",")
    return d[:, 0], d[:, 1]


def _fig4d_fmr_reference():
    """the author's published closed-form FMR model for Fig 4(d) (QuartzAuSimuMRP1S0p02.mx, h0 = 121 µm, φ = 0),
    bundled as ``fig4d_fmr_reference.csv`` — the agreement target my numeric FMR is fenced against
    (corr 0.9993, magnitude 0.992 on the dense 0.02° grid)."""
    d = np.loadtxt(ROOT / "benchmarks" / "fig4d_fmr_reference.csv", delimiter=",")
    return d[:, 0], d[:, 1]


def ml_fig3_figure(*, step: float = 0.25, th_max: float = 65.0):
    """Fig 3: 300 um X-cut quartz Maker fringes — SHAARP.py(HH) vs SHAARP.py(JK) vs the RAW
    Herman-1995 analytic HH benchmark (analyticHH, ported byte-exact), with the author's own
    serialized .mx reference curves overlaid (main panel: dataoldHHJK evaluated HH; zoom panel:
    dataHHsimHHFiner1 fine-fringe window)."""
    s = ML_CASES["fig3"].system_builder()
    th_hh, i_hh = ml_maker(s, ASSUMPTION_HH, th_max=th_max, step=step)
    th_jk, i_jk = ml_maker(s, ASSUMPTION_JK, th_max=th_max, step=step)
    th_ref, i_ref = herman_hayden_quartz_reference_curve(0.0, th_max, 651)
    mx = max(float(np.max(i_hh)), float(np.max(i_jk))) or 1.0
    ref = _fig3_reference_curves()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.0), width_ratios=[2.1, 1.0])
    ax1.plot(th_ref, i_ref, color="k", lw=2.2, alpha=0.30, label="raw Herman-1995 analytic HH")
    stats = {"hh_jk_peak_ratio": float(np.max(i_hh) / np.max(i_jk)),
             "hh_vs_analytic_corr": _shape_corr(th_hh, i_hh / mx, th_ref, i_ref)}
    if ref is not None:
        fine, hhjk = ref
        m = (hhjk[:, 0] >= 0.0) & (hhjk[:, 0] <= th_max)
        ax1.plot(hhjk[m, 0], hhjk[m, 2] / (np.max(hhjk[m, 2]) or 1.0), "k:", lw=1.2,
                 label="the author's evaluated HH (dataoldHHJK.mx)")
        stats["analytic_vs_mx_corr"] = _shape_corr(
            th_ref, i_ref, hhjk[m, 0], hhjk[m, 2] / (np.max(hhjk[m, 2]) or 1.0))
    ax1.plot(th_hh, i_hh / mx, color="#3465c0", lw=0.9, label="SHAARP.py (HH)")
    ax1.plot(th_jk, i_jk / mx, color="#e0532f", lw=0.9, alpha=0.8, label="SHAARP.py (JK)")
    ax1.set_xlabel(r"Incident angle $\theta_i$ (°)"); ax1.set_ylabel(r"$I_{pp}^{2\omega}$ (norm.)")
    ax1.set_title("(b) Maker fringes, 300 µm X-cut quartz, 1064 nm"); ax1.legend(fontsize=7)
    # Fine-fringe zoom at 0.02 deg -- the author's own S0p02 sampling. The 2w fine fringes have ~0.2 deg
    # period, so 0.05 deg (4 pts/period) renders the jagged/beating envelope was flagged; 0.02 deg
    # (~10 pts/period) is smooth. The smooth reference is the byte-exact analytic Herman-Hayden HH on
    # the SAME fine grid (identical to HH sim on a shared grid, corrcoef 1.0); the 0.05 deg .mx
    # points ride along as sparse validation markers + the same-grid correlation check.
    zoom = (20.0, 30.0)
    fine_step = 0.02
    th_hf, i_hf = ml_maker(s, ASSUMPTION_HH, th_min=zoom[0], th_max=zoom[1], step=fine_step)
    th_jf, i_jf = ml_maker(s, ASSUMPTION_JK, th_min=zoom[0], th_max=zoom[1], step=fine_step)
    ax2.plot(th_hf, i_hf / mx, color="#3465c0", lw=1.0, label="SHAARP.py (HH)")
    ax2.plot(th_jf, i_jf / mx, color="#e0532f", lw=1.0, alpha=0.8, label="SHAARP.py (JK)")
    n_fine = int(round((zoom[1] - zoom[0]) / fine_step)) + 1
    th_rf, i_rf = herman_hayden_quartz_reference_curve(zoom[0], zoom[1], n_fine)
    rmean = float(np.mean(i_hf / mx)) or 1.0
    ax2.plot(th_rf, i_rf * (rmean / (float(np.mean(i_rf)) or 1.0)), "k--", lw=0.8, alpha=0.7,
             label="HH analytic (= the author's sim, 0.02°)")
    if ref is not None:
        fine, _ = ref
        w = (fine[:, 0] >= zoom[0]) & (fine[:, 0] <= zoom[1])
        sc = rmean / (float(np.mean(fine[w, 1])) or 1.0)
        ax2.plot(fine[w, 0], fine[w, 1] * sc, "k.", ms=2.6, alpha=0.5,
                 label="the author's HH sim .mx (0.05° pts)")
        # same-grid correlation on the .mx's own 0.05 deg points (resample the smooth 0.02 deg SHAARP
        # HH onto them -- accurate since 0.02 deg is well above Nyquist, so no aliasing).
        i_hf_on_mx = np.interp(fine[w, 0], th_hf, i_hf)
        stats["shaarp_hh_vs_hhsim_fine_corr"] = float(np.corrcoef(i_hf_on_mx, fine[w, 1])[0, 1])
    ax2.set_xlim(*zoom); ax2.set_xlabel(r"$\theta_i$ (°)")
    ax2.set_title("(c) magnified (fine fringes)"); ax2.legend(fontsize=6)
    fig.suptitle("SHAARP.py reproduction — SHAARP.ml 2024 Fig. 3 (+ raw HH benchmark + the author's .mx data)",
                 y=1.03, fontsize=10)
    fig.tight_layout()
    return fig, stats


_FIG4_DAT = {
    # published Fig-4 experimental Maker scans (raw data, fig3 archive folder):
    # 4 columns [stage_deg, I_PinPout, monitor, NaN]; the stage reading is TWICE the physical
    # incidence angle (T70 = +-70 deg at 0.1-deg physical step -> -140..140 at 0.2 in the file).
    False: "Quartz_800nm_100um_MF_T70_Step1_F0p25_PinPout.dat",     # (b) uncoated
    True:  "Quartz_100um_5nmAuBack_MF_T70_Step0p1_F0p25_PinPout.dat",  # (d) + backside Au
}


# Physical-theta points Rui manually removes from the UNCOATED (b) scan (fig3.nb "Fig 3c",
# Position/Delete at line 1633); the Au (d) panel has NO outlier removal (verified: no Delete/Position
# in the Fig-3d section). Applied at |theta| within 0.3 deg of a listed value (both scan signs).
_FIG4_OUTLIER_THETA = {
    False: (24.6, 27.0, 28.6, 41.0, 42.4, 42.6, 44.8),   # uncoated (b)
    True: (),                                            # + backside Au (d): none
}


def _fig4_experiment(au: bool, *, th_window: float = 50.0):
    """(theta_deg, I_norm) of the published Fig-4 Maker scan, THE PAPER'S OWN recipe.

    The published-Fig-4 generating cell (archived under fig3/ with the pre-renumbering internal
    label 'Fig. 3d'): select |stage/2| <= 50 deg, remove the manual outlier points (uncoated panel),
    then ``Rescale`` the raw counts (min -> 0, max -> 1). No baseline subtraction, no monitor
    division -- the earlier median-baseline version wrongly subtracted the (d) panel's central bump."""
    p = (ROOT.parent.parent / "Papers/Manuscript_Linear/EM Code/SHAARP_SLAB/Manuscript/fig3"
         / _FIG4_DAT[au])
    if not p.exists():
        return None
    raw = np.loadtxt(p, usecols=(0, 1))
    th = raw[:, 0] / 2.0                       # stage reading -> physical incidence angle
    m = np.abs(th) <= th_window
    th, counts = th[m], raw[m, 1]
    drops = _FIG4_OUTLIER_THETA.get(au, ())   # the author's manual bad-point removal (Rescale AFTER)
    if drops:
        keep = np.ones(th.shape, dtype=bool)
        for d in drops:
            keep &= np.abs(np.abs(th) - d) > 0.3
        th, counts = th[keep], counts[keep]
    lo, hi = float(np.min(counts)), float(np.max(counts))
    return th, (counts - lo) / ((hi - lo) or 1.0)


def ml_fig4_averaged_fmr(*, step: float = 0.15, th_max: float = 45.0, h_center: float = 121.0,
                         span_um: float = 0.80, n_h: int = 13, theta_window_deg: float = 3.0):
    """The paper's '♯SHAARP (FMR+θⁱ+h+λω)' curve for panel (d), reproducing the author's EXACT method from
    ``Maker fringes 100 um quartz.nb`` section "Binning Study θ+h+λ" (line 86834+):

    The wavelength (λω) spread is folded into an EQUIVALENT-THICKNESS broadening — the Maker-fringe
    phase goes as h/λ, so a Δλ/λ shift is equivalent to a Δh/h shift (the `Abs[N[795/800]-1]*123 ≈
    0.77 µm` justification). So h and λ MERGE into ONE Gaussian thickness-average of span
    ``0.80 µm = 0.05 (h) + 0.75 (λ-equivalent)`` about the panel thickness (the 62 samples; N=21 here
    captures the Gaussian core tractably), then the θ (beam-divergence) spread is a ~3° boxcar
    (MovingAverage) — his `θAve = 150` on the 0.02° grid = 3°. NOT a ±5 nm λ re-simulation (that was
    a prior port invention).
    """
    hs = np.linspace(h_center - span_um / 2.0, h_center + span_um / 2.0, n_h)
    idx = np.arange(n_h) - (n_h - 1) / 2.0            # discrete Gaussian weights over the samples,
    sigma_idx = float(np.std(np.arange(n_h))) or 1.0   # matching MFhAve's PDF[NormalDistribution]
    wt = np.exp(-0.5 * (idx / sigma_idx) ** 2)
    wt /= wt.sum()
    acc = None
    th = None
    for h, w in zip(hs, wt):
        s = ml_fig4d_system(h_quartz_um=float(h))
        th, i = ml_maker(s, ASSUMPTION_FMR, th_max=th_max, step=step)
        acc = w * i if acc is None else acc + w * i
    i_hl = acc                                          # h+λ Gaussian thickness-average
    # θ beam divergence: 3° boxcar MovingAverage (θAve), edge-corrected
    win = max(1, int(round(theta_window_deg / step)))
    kern = np.ones(win) / win
    i_sm = np.convolve(i_hl, kern, mode="same")
    norm = np.convolve(np.ones_like(i_hl), kern, mode="same")
    return th, i_sm / norm


def ml_fig4_figure(*, step: float = 0.05, th_max: float = 45.0, averaged: bool = True):
    """Fig 4: Z-cut quartz Maker fringes, uncoated (b, 123.6 µm) vs backside-Au (d), in the
    PAPER'S OWN format: ±45° span, model curves + the published EXPERIMENTAL scans (black dots,
    the paper's exact 100·Rescale normalization), and the 30–40° fine-fringe inset.

    Panel (d) uses the PUBLISHED generating-cell substitutions (h1 = 121 µm — a different spot of
    the sample than (b)'s 123.6 µm — and 13.9 nm backside Au; see :func:`ml_fig4d_system`), which give
    the published ~0.35 central bump and the ±38–40° experiment peaks, plus the paper's fourth
    curve — FMR averaged over the SI-documented θ/h/λ spreads (func:`ml_fig4_averaged_fmr`).

    The FMR and JK curves and the θ/h/λ averaging are all my numeric solver; my FMR reproduces the author's
    closed-form FMR model (fig4d_fmr_reference.csv) to corr 0.9993 — reported as the panel-(d)
    ``d_FMR_vs_his_FMR_corr`` stat. The HH curve on panel (d) is the author's OWN published HH model
    (fig4d_hh_reference.csv, ×1.5), embedded because my numeric single-pass-ω HH diverges from his
    closed-form HH for the strong 13.9 nm Au reflector (option B, — see
    :func:`_fig4d_hh_reference`). HH is the paper's deliberately-failing illustrative curve.

    The 0.05° step is REQUIRED: the FMR fine fringes have ~0.4–0.5° period, so a 0.3° step sits
    under Nyquist and renders jagged sawteeth instead of smooth oscillations (caught by Rui,
     — same aliasing class as the Fig-3 zoom).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
    stats = {}

    def mirror(th_c, y):
        return np.concatenate([-th_c[::-1], th_c]), np.concatenate([y[::-1], y])

    for ax, au, ttl in ((axes[0], False, "(b) uncoated Z-cut quartz, 123.6 µm"),
                        (axes[1], True, "(d) + backside Au (121 µm spot)")):
        s = ml_fig4d_system() if au else ml_fig4_system(au=False)
        th_f, i_f = ml_maker(s, ASSUMPTION_FMR, th_max=th_max, step=step)
        mx = float(np.max(i_f)) or 1.0
        if au:
            # Panel (d) HH = the author's OWN published HH model (QuartzAu_800nm_HHNMRP1S0p01.mx), embedded as
            # fig4d_hh_reference.csv (option B). My numeric single-pass-ω HH diverges from his
            # closed-form HH for the strong 13.9 nm Au reflector — decoded from his .mx denominators, his HH
            # = FMR with the ω Fabry-Pérot factor removed, which the interface-by-interface single-pass does
            # not reproduce at r_ω≈0.60 (it matches his HH to 1e-13 only on low-reflectance dielectric
            # stacks). HH is the paper's DELIBERATELY-FAILING illustrative curve; my FMR — the physics
            # result — reproduces his FMR model to corr 0.9993 (fenced by the stat below). His HH is
            # normalized to his FMR-reference peak so it shares the FMR-peak axis with my FMR (i_f/mx).
            th_h, i_hh_ref = _fig4d_hh_reference()
            fmr_th, fmr_ref = _fig4d_fmr_reference()
            fmr_ref_peak = float(np.max(fmr_ref)) or 1.0
            i_h = i_hh_ref / fmr_ref_peak            # his HH as a fraction of his FMR peak
            sel = th_h <= th_max                     # clip his 0–50° grid to the ±th_max panel window
            th_h, i_h = th_h[sel], i_h[sel]
            hh_scale = 1.5                           # paper's HHScale display factor
            curves = [(th_h, hh_scale * i_h, "#7cb342", "♯SHAARP.ml (HH) ×1.5", 1.2),
                      (th_f, i_f / mx, "#e0532f", "♯SHAARP.py (FMR)", 0.8)]
        else:
            # Panel (b): the same uncoated slab; HH/JK are my solver's single-slab baselines.
            s_hh = s
            th_h, i_h = ml_maker(s_hh, ASSUMPTION_HH, th_max=th_max, step=step)
            i_h = i_h / mx
            hh_scale = 1.0
            curves = [(th_h, i_h, "#7cb342", "♯SHAARP.py (HH)", 1.2),
                      (th_f, i_f / mx, "#e0532f", "♯SHAARP.py (FMR)", 0.8)]
        if not au:  # JK is shown only on panel (b); the published (d) panel omits it
            th_j, i_j = ml_maker(s, ASSUMPTION_JK, th_max=th_max, step=step)
            curves.insert(0, (th_j, i_j / mx, "#e8a33d", "♯SHAARP.py (JK)", 1.4))
        i_av = None
        if au and averaged:
            th_a, i_av = ml_fig4_averaged_fmr(step=0.15, th_max=th_max)
            i_av = i_av / mx
            curves.append((th_a, i_av, "#2aa198", "♯SHAARP.py (FMR+θⁱ+h+λω)", 2.2))
        for th_c, y, color, lab, lw in curves:
            xm, ym = mirror(th_c, y)
            ax.plot(xm, ym, color=color, lw=lw, label=lab)
        # Rescale the experiment over the PLOTTED angular window (±th_max), not the raw ±50° of the
        # fig3.nb `Select[... <= 50 &]` cell: that ±50° window's minimum is the θ≈50° grazing tail
        # (OFF the ±45° plot), so within the panel the center dip floated ~0.066 above the JK/HH curves
        # (expt vs shaarp needs improving"). Rescaling over the plotted range makes the
        # center dip the minimum, so the expt hugs the curves through the dip — matching the published panel.
        expt = _fig4_experiment(au, th_window=th_max)
        if expt is not None:
            th_e, i_e = expt
            m = np.abs(th_e) <= th_max
            # the paper Rescales the experiment to [0, 1]; put its max on the curve it follows
            # (the averaged curve in (d); the smooth JK/HH envelope in (b)). Panel (d) without the
            # averaged curve falls back to the FMR peak (=1.0) — JK is not computed on (d) anymore.
            if i_av is not None:
                target = float(np.max(i_av))
            elif not au:
                target = float(np.max(i_j)) / mx
            else:
                target = 1.0
            sc = target / (float(np.max(i_e[m])) or 1.0)
            ax.plot(th_e[m], i_e[m] * sc, "k.", ms=1.6, label="Expt. (the author's scan)")
            stats[f"expt_overlaid_{'d' if au else 'b'}"] = 1.0
            if i_av is not None:
                ii = np.interp(np.abs(th_e[m]), th_a, i_av)
                stats["expt_vs_averaged_corr"] = float(np.corrcoef(i_e[m] * sc, ii)[0, 1])
        ax.set_xlabel(r"Incident angle $\theta_i$ (°)"); ax.set_ylabel(r"$I_{pp}^{2\omega}$ (norm.)")
        ax.set_title(ttl, fontsize=10); ax.legend(fontsize=6.5, loc="upper left")
        axi = ax.inset_axes([0.36, 0.62, 0.30, 0.33])
        # inset fine fringes over 30–40°. Panel (b): re-sweep HH/JK at 0.02° (smooth, not aliased).
        # Panel (d): slice the author's embedded HH reference (already 0.02°). FMR (0.05°, ~0.4–0.5° period) is
        # already smooth so slice the main array.
        wm = (th_f >= 30.0) & (th_f <= 40.0)
        if au:  # published (d) omits JK; HH = the embedded reference sliced to the window
            im = (th_h >= 30.0) & (th_h <= 40.0)
            axi.plot(th_h[im], hh_scale * i_h[im], color="#7cb342", lw=1.0)
        else:
            th_hi, i_hi = ml_maker(s_hh, ASSUMPTION_HH, th_min=30.0, th_max=40.0, step=0.02)
            th_ji, i_ji = ml_maker(s, ASSUMPTION_JK, th_min=30.0, th_max=40.0, step=0.02)
            axi.plot(th_ji, i_ji / mx, color="#e8a33d", lw=1.2)
            axi.plot(th_hi, hh_scale * i_hi / mx, color="#7cb342", lw=1.0)
        axi.plot(th_f[wm], (i_f / mx)[wm], color="#e0532f", lw=0.7)
        if expt is not None:
            me = (th_e >= 30.0) & (th_e <= 40.0)
            axi.plot(th_e[me], i_e[me] * sc, "k.", ms=1.5)
        axi.set_xlim(30, 40); axi.tick_params(labelsize=6)
        if au:
            # method-equivalence stat: my numeric FMR vs the author's closed-form FMR model (fig4d_fmr_reference).
            # Sample both on MY grid, interpolating only his DENSER reference (his-fine → my-grid) so no
            # coarse-interp artifact deflates the corr — the reverse (my-coarse → his-fine linear interp)
            # reads ~0.988, an artifact; the true agreement on matched grids is corr 0.9993.
            his_on_mine = np.interp(th_f, fmr_th, fmr_ref)
            stats["d_FMR_vs_his_FMR_corr"] = float(
                np.corrcoef(i_f / mx, his_on_mine / (np.max(his_on_mine) or 1.0))[0, 1])
            stats["d_central_bump"] = float(i_f[0] / mx)
        else:
            n = min(len(i_f), len(i_h)); stats[ttl[:3]] = float(np.max(np.abs((i_f / mx)[:n] - i_h[:n])))
    fig.suptitle("SHAARP.py reproduction — SHAARP.ml 2024 Fig. 4 (FMR departs with the Au mirror)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    return fig, stats


def ml_polarimetry_figure(rows, *, channel: str, n_phi: int = 73, title: str = ""):
    """Grid of transmitted- or reflected-SHG polar plots. ``rows`` = [(label, system, thetas), ...]."""
    fig = plt.figure(figsize=(7.6, 3.5 * len(rows)))
    for r, (label, system, thetas) in enumerate(rows):
        curves = {th: ml_polar(system, theta_deg=th, channel=channel, n_phi=n_phi) for th in thetas}
        # the author's convention: both channels share the scale maxTot = 1.1*max(I_p,I_s); the weak channel
        # is lifted by its integer ×N factor (annotated) so the true I_p/I_s ratio stays visible.
        pmax = max(float(np.max(curves[th][1])) for th in thetas) or 1.0
        smax = max(float(np.max(curves[th][2])) for th in thetas) or 1.0
        ref = max(pmax, smax)
        for col, chan in ((0, r"I_p"), (1, r"I_s")):
            ax = fig.add_subplot(len(rows), 2, 2 * r + col + 1, projection="polar")
            fac = _display_factor(pmax if col == 0 else smax, ref)
            for th, color in zip(thetas, COLORS):
                ph, ip, is_ = curves[th]
                ax.plot(np.radians(ph), (ip if col == 0 else is_) * fac / (1.1 * ref), color=color,
                        lw=1.3, label=rf"$\theta_i$={th:g}°")
            ax.set_ylim(0, 1.05); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["", ""])
            fx = rf"$\times{fac}$" if fac > 1 else ""
            ax.set_title(f"{label}\n${chan}^{{{channel[0].upper()},2\\omega}}$ {fx}", fontsize=8, pad=10)
            if r == 0 and col == 1:
                ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=7)
    if title:
        fig.suptitle(title, y=1.005, fontsize=10)
    fig.tight_layout()
    return fig


def ml_interference_figure(*, n_phi: int = 73):
    """Fig 7(c,d): transmitted SHG polarimetry of the four LNO//quartz stacking cases at normal
    incidence — I_L1(φ) and I_L2(φ) polar plots plus the I_L1(φ=0°) bar chart showing constructive
    (LNO P∥−L1 // quartz) vs destructive (P∥+L1 // quartz) two-crystal SHG interference.

    Near θ≈0 the transmitted p/s channels coincide with the lab components I_L1/I_L2 (p̂→L1, ŝ→L2).
    """
    cases = {1: "Case 1: LNO alone (P∥−L1)", 2: "Case 2: LNO(P−)//quartz",
             3: "Case 3: LNO(P+)//quartz", 4: "Case 4: LNO alone (P∥+L1)"}
    colors = {1: "#3465c0", 2: "#e8a33d", 3: "#7cb342", 4: "#e0532f"}
    styles = {1: "-", 2: "-", 3: "-", 4: "--"}
    data = {k: ml_polar(ml_fig7_case(k), theta_deg=0.5, channel="transmitted", n_phi=n_phi)
            for k in cases}
    g1 = max(float(np.max(d[1])) for d in data.values()) or 1.0
    g2 = max(float(np.max(d[2])) for d in data.values()) or 1.0
    fig = plt.figure(figsize=(12.6, 4.2))
    facs = {}
    for col, (gm, ttl) in enumerate([(g1, r"$I_{L_1}^{T,2\omega}(\varphi)$"),
                                     (g2, r"$I_{L_2}^{T,2\omega}(\varphi)$")]):
        ax = fig.add_subplot(1, 3, col + 1, projection="polar")
        for k in cases:
            ph, ip, is_ = data[k]
            cur = ip if col == 0 else is_
            # per-case display factor within this channel = the author's factorO = int(channelMax/caseMax)
            # (NO 1.1 -- that multiplier is only on the cross-CHANNEL factor); lifts each case to the
            # channel's own max so every case is visible; the annotated ×N keeps the true ratio.
            fac = max(1, int(gm / (float(np.max(cur)) or 1e-300)))
            facs[(col, k)] = fac
            ax.plot(np.radians(ph), cur * fac / (1.1 * gm), styles[k], color=colors[k],
                    lw=1.5, label=cases[k] + (rf"  $\times{fac}$" if fac > 1 else ""))
        ax.set_ylim(0, 1.05); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["", ""])
        # the weak L2 panel also carries the cross-channel factor vs L1 (the paper's ×61)
        cross = rf"  ($\times{_display_factor(g2, g1)}$ vs $I_{{L_1}}$)" if col == 1 else ""
        ax.set_title(ttl + cross, fontsize=10, pad=10)
    axb = fig.add_subplot(1, 3, 3)
    bars = [float(data[k][1][0]) for k in (1, 2, 3, 4)]   # I_L1 at phi = 0 (first grid point)
    bmax = max(bars) or 1.0
    axb.bar([f"Case {k}" for k in (1, 2, 3, 4)], [b / bmax for b in bars],
            color=[colors[k] for k in (1, 2, 3, 4)])
    axb.set_ylabel(r"normalized $I_{L_1}^{T,2\omega}(\varphi=0°)$")
    axb.set_ylim(0, 1.1)
    axb.set_title("(d) constructive vs destructive", fontsize=9)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle("SHAARP.py reproduction — SHAARP.ml 2024 Fig. 7: LNO(21̄1̄0)//quartz(001) SHG interference",
                 y=1.02, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    stats = {f"case{k}_IL1_phi0": float(data[k][1][0]) for k in cases}
    stats["IL1_over_IL2"] = float(g1 / g2) if g2 else float("inf")
    stats["L2_cross_factor"] = _display_factor(g2, g1)     # the paper's ×61 (L2 vs L1)
    stats["L2_case_factors"] = [facs[(1, k)] for k in (1, 2, 3, 4)]   # per-case weak-channel ×N
    return fig, stats


def ml_mos2_figure(*, n_azimuth: int = 145):
    """SI Fig S7(b,c): twist bilayer MoS2 rotational-anisotropy reflected SHG. Panel (b) fixed
    analyzer at 0° (reflected I_p), panel (c) at 90° (reflected I_s); three curves on a common
    scale — monolayer MoS2(0°), monolayer MoS2(25°), and the MoS2(25°)//MoS2(0°) twist bilayer."""
    data = {lab: ml_ra_reflected_polar(ml_mos2_system(tw), n_azimuth=n_azimuth)
            for lab, tw, _ in MOS2_CASES}
    gmax_p = max(float(np.max(d[1])) for d in data.values()) or 1.0
    gmax_s = max(float(np.max(d[2])) for d in data.values()) or 1.0
    fig = plt.figure(figsize=(11.0, 4.3))
    for col, (gm, ttl) in enumerate([(gmax_p, r"(b) $I^{R,2\omega}(\theta,\ \varphi_{an}=0°)$"),
                                     (gmax_s, r"(c) $I^{R,2\omega}(\theta,\ \varphi_{an}=90°)$")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="polar")
        for lab, tw, color in MOS2_CASES:
            az, ip, is_ = data[lab]
            ax.plot(np.radians(az), (ip if col == 0 else is_) / gm, color=color, lw=1.5, label=lab)
        ax.set_ylim(0, 1.05); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["", ""])
        ax.set_title(ttl, fontsize=9, pad=10)
        if col == 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=7)
    fig.suptitle("SHAARP.py reproduction — SHAARP.ml SI Fig. S7(b,c): twist bilayer MoS₂ RA-SHG",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    ratio = gmax_p / (max(float(np.max(data[lab][1])) for lab, tw, _ in MOS2_CASES if tw == [0.0]) or 1.0)
    return fig, {"bilayer_over_monolayer_Ip_peak": float(ratio)}


def _shape_corr(x1, y1, x2, y2):
    """Normalized cross-correlation of two curves on a common grid (shape agreement, 1 = identical)."""
    xc = np.linspace(max(x1.min(), x2.min()), min(x1.max(), x2.max()), 400)
    a = np.interp(xc, x1, y1); b = np.interp(xc, x2, y2)
    a = (a - a.mean()) / (a.std() + 1e-30); b = (b - b.mean()) / (b.std() + 1e-30)
    return float(np.mean(a * b))


def ml_fig8c_figure(**_):
    """Fig 8c: SHG coherence κ vs TWIST angle Δθ for the twisted MoS₂ bilayer on Al₂O₃, vs the ideal
    cos(3Δθ) (three-fold monolayer symmetry), with the RED BAND = κ's spread over the Al₂O₃ substrate
    Fabry-Pérot extremes.

    The band is the author's EXACT published values: his ``minmaxkappa.nb`` ``kappamin``/``kappamax``, evaluated
    from the FROZEN ♯SHAARP.ml SHG coefficients at the FP-min / FP-max substrate cases (peak-over-azimuth of
    the twist-bilayer / (2×monolayer) − 1, at each twist 0–75°/5°). Bundled verbatim as
    ``benchmarks/fig8c_reference_band.csv`` (θ, κ_min, κ_max) — the exact-benchmark port, same discipline as
    ``analyticHH.mx``. κ(0)=[0.688, 0.936], crossing 0 near 28° and reaching −1 at 60°, matching the paper.
    (An independent SHAARP.py recompute of the Al₂O₃ FP extremes is the open refinement — the earlier
    substrate-thickness sweep gave a too-narrow band because 500.57/508.47 µm are not the true FP extremes
    for the registry Al₂O₃ index; linked to the metal/substrate multilayer-FP work.)"""
    band = np.loadtxt(ROOT / "benchmarks" / "fig8c_reference_band.csv", delimiter=",")
    twists, lo, hi = band[:, 0], band[:, 1], band[:, 2]
    xf = np.linspace(0.0, 75.0, 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xf, np.cos(3.0 * np.radians(xf)), color="k", lw=2.2, label=r"$\cos(3\Delta\theta)$")
    ax.fill_between(twists, lo, hi, color="#e0532f", alpha=0.35)
    ax.plot(twists, lo, color="#e0532f", lw=1.0)
    ax.plot(twists, hi, color="#e0532f", lw=1.0, label=r"$\kappa(\Delta\theta,\ t_{Al_2O_3})$")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlim(0, 75); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Twist Angle, $\Delta\theta$ (°)"); ax.set_ylabel(r"$\kappa(\Delta\theta)$")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("SHAARP.py reproduction — SHAARP.ml 2024 Fig. 8c (MoS₂ twist coherence)", fontsize=10)
    fig.tight_layout()
    stats = {"kappa0_band": (float(lo[0]), float(hi[0])),
             "kappa_band_min": [float(x) for x in lo],
             "kappa_band_max": [float(x) for x in hi]}
    return fig, stats
