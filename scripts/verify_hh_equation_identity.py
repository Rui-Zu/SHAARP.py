"""Verify that the archived `analyticHH.mx` expression IS the Herman & Hayden (1995) Maker-fringe
equation -- not merely that curves agree with each other (which would be circular).

Method: every numeric constant inside the ported expression must reconstruct EXACTLY from the
physical parameters of the Fig-3 case (L = 300 um, lambda = 1.064 um -> lambda_2w = 0.532 um,
quartz n_w = 1.53413, n_2w = 1.54702, quartz d11 = 0.30 pm/V) through the Herman-Hayden equation
structure -- and a from-physics reconstruction of the full I(theta) must equal the archived
expression point-by-point.

HH-1995 structure (transmitted p-p SHG of a slab, multiple reflections kept ONLY for the free /
homogeneous 2w wave):

    I(theta) = A * cos^4(theta) * P_f^2
               * [ sinc^2(Psi-) - 2 r_p cos(delta_2) (P_b/P_f) sinc(Psi-) sinc(Psi+)
                                + r_p^2 (P_b/P_f)^2 sinc^2(Psi+) ]
               / [ (n_w cos + w_w/n_w)^4 (n_2w cos + w_2/n_2w)^2 * |1 - r_p^2 e^{2 i delta_2}|^2 ]

with w_w = sqrt(n_w^2 - sin^2), w_2 = sqrt(n_2w^2 - sin^2) (n cos(theta_internal))
      Psi-+ = (pi L / lambda_2w) (w_w -+ w_2) (coherence phases)
      delta_2 = (2 pi L / lambda_2w) w_2 (free-wave one-way phase)
      r_p = (n_2w^2 cos - w_2) / (n_2w^2 cos + w_2) (air/quartz p Fresnel at 2w)
      P_f/b = d11 s [ 2 a b / n_w -+ ... ] (forward/backward bound-wave
                                                                        d-projections, quartz d11 p-p)

Run: python scripts/verify_hh_equation_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks.herman_hayden_maker import herman_hayden_quartz_300um_intensity

# ---- the physical parameters of the ML Fig-3 case (nothing else is allowed in) ----------------
L_UM = 300.0            # slab thickness
LAM_2W_UM = 1.064 / 2   # SHG wavelength 532 nm
N_W = 1.53413           # quartz n at 1064 nm (ordinary, isotropic approx)
N_2W = 1.54702          # quartz n at 532 nm
D11 = 0.30              # quartz d11 in pm/V (the case-study value)

CHECKS: list[tuple[str, float, float]] = []


def check(name: str, archived: float, physical: float) -> None:
    CHECKS.append((name, archived, physical))


def main() -> None:
    # ---- 1. constant-by-constant identity against the archived expression's literals ----------
    check("pi L / lambda_2w            (coherence phase scale)", 1771.574804655898, np.pi * L_UM / LAM_2W_UM)
    check("(lambda_2w / pi L)^2        (sinc^2 normaliser)", 3.186258519909891e-7, (LAM_2W_UM / (np.pi * L_UM)) ** 2)
    check("2 (lambda_2w / pi L)^2      (cross-term x2)", 6.372517039819782e-7, 2 * (LAM_2W_UM / (np.pi * L_UM)) ** 2)
    check("delta_2 scale = 2 pi L n_2w / lambda_2w / n_2w -> coeff of sqrt(1-s^2/n_2w^2)",
          5481.323308597535, 2 * np.pi * L_UM * N_2W / LAM_2W_UM)
    check("2 delta_2 scale             (FP round-trip)", 10962.64661719507, 2 * 2 * np.pi * L_UM * N_2W / LAM_2W_UM)
    check("1 / n_w^2                   (omega refraction sqrt)", 0.4248891828751154, 1 / N_W ** 2)
    check("1 / n_2w^2                  (2-omega refraction sqrt)", 0.41783820134596067, 1 / N_2W ** 2)
    check("n_w  (bound-wave index literal)", 1.53413, N_W)
    check("n_2w (free-wave index literal)", 1.54702, N_2W)
    check("2 d11 / n_w                 (projection coeff)", 0.39110114527452045, 2 * D11 / N_W)
    check("1 / n_2w                    (projection coeff)", 0.6464040542462282, 1 / N_2W)
    check("d11 / n_w^2                 (projection coeff)", 0.12746675486253461, D11 / N_W ** 2)
    check("d11                         (quartz d11, pm/V)", 0.3, D11)

    worst = 0.0
    print("constant-by-constant identity (archived literal vs physical formula):")
    for name, arch, phys in CHECKS:
        rel = abs(arch - phys) / max(abs(phys), 1e-300)
        worst = max(worst, rel)
        print(f"  {'OK ' if rel < 1e-12 else 'FAIL'}  {name:58s} {arch!r:>22} vs {phys!r:<22} rel={rel:.2e}")
    if worst >= 1e-12:
        raise SystemExit("CONSTANT IDENTITY: FAIL")

    # ---- 2. full-curve reconstruction from physics (no literals from the .mx) ------------------
    def hh_from_physics(theta_deg):
        t = np.deg2rad(np.asarray(theta_deg, dtype=float))
        s, c = np.sin(t), np.cos(t)
        a = np.sqrt(1 - s**2 / N_2W**2)          # w_2 / n_2w
        b = np.sqrt(1 - s**2 / N_W**2)           # w_w / n_w
        w_w, w_2 = N_W * b, N_2W * a
        psi_m = (np.pi * L_UM / LAM_2W_UM) * (w_w - w_2)
        psi_p = (np.pi * L_UM / LAM_2W_UM) * (w_w + w_2)
        delta2 = (2 * np.pi * L_UM / LAM_2W_UM) * w_2
        # forward / backward bound-wave d-projections (quartz d11, p-p)
        pf = a * (2 * D11 / N_W) * s * b - (1 / N_2W) * s * (D11 / N_W**2 * s**2 - D11 * (1 - s**2 / N_W**2))
        pb = -a * (2 * D11 / N_W) * s * b - (1 / N_2W) * s * (D11 / N_W**2 * s**2 - D11 * (1 - s**2 / N_W**2))
        # air/quartz p-polarised Fresnel numerators at 2w (r_p = fm / fp)
        fm = N_2W * c - a
        fp = N_2W * c + a
        k2 = (LAM_2W_UM / (np.pi * L_UM)) ** 2
        bracket = (k2 * np.sin(psi_m) ** 2 / (w_w - w_2) ** 2
                   - 2 * k2 * np.cos(delta2) * (fm / fp) * (pb / pf)
                     * np.sin(psi_m) * np.sin(psi_p) / ((w_w - w_2) * (w_w + w_2))
                   + k2 * (fm / fp) ** 2 * (pb / pf) ** 2 * np.sin(psi_p) ** 2 / (w_w + w_2) ** 2)
        fabry_perot = 1 + (fm / fp) ** 4 - 2 * np.cos(2 * delta2) * (fm / fp) ** 2
        denom = (N_W * c + b) ** 4 * (N_2W * c + a) ** 2 * fabry_perot
        return c**4 * pf**2 * bracket / denom

    th = np.linspace(1.0, 65.0, 1281)                              # 0.05 deg
    i_port = herman_hayden_quartz_300um_intensity(np.deg2rad(th))  # the byte-exact .mx port
    i_phys = hh_from_physics(th)
    ratio = i_port / i_phys
    spread = float(np.std(ratio) / np.mean(ratio))
    print(f"\nfull-curve reconstruction from physics vs archived expression (1-65 deg, 0.05 deg):")
    print(f"  overall amplitude constant A = {float(np.mean(ratio)):.10g}   (pure units prefactor)")
    print(f"  relative spread of the ratio  = {spread:.3e}   (0 = identical up to units)")
    corr = float(np.corrcoef(i_port, i_phys)[0, 1])
    print(f"  same-grid corrcoef            = {corr:.9f}")
    if spread > 1e-10 or corr < 0.999999999:
        raise SystemExit("EQUATION RECONSTRUCTION: FAIL")
    print("\nHH EQUATION IDENTITY: PASS -- the archived analyticHH expression is exactly the")
    print("Herman & Hayden (1995) Maker-fringe equation (sinc^2 coherence + free-wave Fabry-Perot")
    print("MR factor + backward-wave cross term) for 300 um / 1064 nm / quartz d11 = 0.30 pm/V.")


if __name__ == "__main__":
    main()
