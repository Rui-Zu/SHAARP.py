"""Raw Herman & Hayden (1995) analytic Maker-fringe intensity -- an INDEPENDENT benchmark for
SHAARP.py's Herman-Hayden (HH) multilayer SHG mode, for the SHAARP.ml 2024 Fig-3 case
(300 um X-cut quartz, fundamental 1064 nm).

PROVENANCE. This is a byte-exact Python translation of the author's own reference expression
``analyticHH`` stored in ``.../SHAARP_SLAB/Manuscript/fig3/analyticHH.mx`` -- the serialized output
of the ``run.nb`` cell captioned *"Compare with Herman 1995 - Fig. 2 - 300 um X-cut Quartz under
isotropic approx."* The Wolfram closed form (Head = Times, pure Cos/Sin/Sqrt arithmetic in the
incidence angle theta) was exported with ``ToString[expr, InputForm]`` and mechanically converted
(Cos[..]->np.cos(..), Sqrt->np.sqrt, ^->**, *^->e); the translated value agrees with the Mathematica
value to 1e-15 relative (verified at theta = 30 deg: 4.875968756693962e-05).

DECODED STRUCTURE (Herman & Hayden, J. Opt. Soc. Am. B 12, 416 (1995)). The intensity is
``I(theta) = [Fresnel/projection prefactor] * [ sinc^2 coherence term + HH multiple-reflection
cos-terms ]``:
  * prefactor Cos[theta]^4 * (d-projection)^2 over the transmission denominators
    (n_w Cos theta + sqrt(1 - sin^2/n_w^2))^4 (n_2w Cos theta + sqrt(1 - sin^2/n_2w^2))^2, with the
    substituted quartz indices n_w = 1.53413 (1064 nm) and n_2w = 1.54702 (532 nm) -- quartz normal
    dispersion; the 4th-power transmission factor carries the fundamental (enters squared in E^2)
    and the 2nd-power the exiting SHG (the constants 0.42489 = 1/n_w^2 and
    0.41784 = 1/n_2w^2 appear inside the refraction sqrt factors);
  * coherence Sin[1771.574 (n_2w cos_2w - n_w cos_w)]^2 / (...)^2 = sinc^2(dk L / 2), the classic
    Maker fringe, with phase constant 1771.574 = pi L / lambda_2w (L = 300 um, lambda_2w = 532 nm);
  * MR terms Cos[5481.32 sqrt(..)] and Cos[10962.65 sqrt(..)] (2x and 4x the single-pass 2w phase)
    = multiple reflections kept ONLY for the homogeneous 2w waves -- exactly the SHAARP.ml
    "Herman & Hayden (MR only for 2w homogeneous waves)" assumption.

USE. ``herman_hayden_quartz_reference_curve`` returns the (theta_deg, I) reference curve to overlay
on the ML Fig-3 reproduction; the gate test ``tests/test_herman_hayden_benchmark.py`` pins that
SHAARP.py's own HH Maker sweep agrees with this raw analytic expression to a stated tolerance --
proving the HH mode against an independent analytic model, not just against SHAARP itself.
"""

from __future__ import annotations

import numpy as np

# Substituted parameters of the Fig-3 case (read out of the analyticHH constants; see module docstring).
# Quartz normal dispersion: the LOWER index belongs to the fundamental (1064 nm).
QUARTZ_N_OMEGA = 1.53413      # n_w at 1064 nm (1/n_w^2 = 0.4248891828751154)
QUARTZ_N_2OMEGA = 1.54702     # n_2w at 532 nm (1/n_2w^2 = 0.41783820134596067)
QUARTZ_THICKNESS_UM = 300.0
FUNDAMENTAL_UM = 1.064


def herman_hayden_quartz_300um_intensity(theta_rad):
    """The author's raw Herman-1995 analytic HH transmitted-SHG Maker-fringe intensity I(theta) for the
    ML Fig-3 case (300 um X-cut quartz, 1064 nm). ``theta_rad`` = incidence angle in radians
    (scalar or array). Byte-exact translation of fig3/analyticHH.mx (see module docstring)."""
    t = np.asarray(theta_rad, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
      return (
        (153.16933634560002*np.cos(t)**4*(0. + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)*(0. + 0.39110114527452045*np.sin(t)*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2)) - 0.6464040542462282*np.sin(t)*(0. + 0.12746675486253461*np.sin(t)**2 - 0.3*(1 - 0.4248891828751154*np.sin(t)**2)))**2*((3.186258519909891e-7*np.sin(1771.574804655898*(1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) - 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)))**2)/(1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) - 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2 - (6.372517039819782e-7*np.cos(5481.323308597535*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))*(1.54702*np.cos(t) - np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))*(0. - np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)*(0. + 0.39110114527452045*np.sin(t)*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2)) - 0.6464040542462282*np.sin(t)*(0. + 0.12746675486253461*np.sin(t)**2 - 0.3*(1 - 0.4248891828751154*np.sin(t)**2)))*np.sin(1771.574804655898*(1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) - 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)))*np.sin(1771.574804655898*(1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) + 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))))/((1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) - 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))*(1.54702*np.cos(t) + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))*(1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) + 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))*(0. + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)*(0. + 0.39110114527452045*np.sin(t)*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2)) - 0.6464040542462282*np.sin(t)*(0. + 0.12746675486253461*np.sin(t)**2 - 0.3*(1 - 0.4248891828751154*np.sin(t)**2)))) + (3.186258519909891e-7*(1.54702*np.cos(t) - np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2*(0. - np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)*(0. + 0.39110114527452045*np.sin(t)*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2)) - 0.6464040542462282*np.sin(t)*(0. + 0.12746675486253461*np.sin(t)**2 - 0.3*(1 - 0.4248891828751154*np.sin(t)**2)))**2*np.sin(1771.574804655898*(1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) + 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)))**2)/((1.54702*np.cos(t) + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2*(1.53413*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2) + 1.54702*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2*(0. + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)*(0. + 0.39110114527452045*np.sin(t)*np.sqrt(1 - 0.4248891828751154*np.sin(t)**2)) - 0.6464040542462282*np.sin(t)*(0. + 0.12746675486253461*np.sin(t)**2 - 0.3*(1 - 0.4248891828751154*np.sin(t)**2)))**2)))/((1.53413*np.cos(t) + np.sqrt(1 - 0.4248891828751154*np.sin(t)**2))**4*(1.54702*np.cos(t) + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2*(1 + ((1.54702*np.cos(t) - np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2*(-1.54702*np.cos(t) + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2)/(1.54702*np.cos(t) + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**4 + (2*np.cos(10962.64661719507*np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))*(1.54702*np.cos(t) - np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))*(-1.54702*np.cos(t) + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2)))/(1.54702*np.cos(t) + np.sqrt(1 - 0.41783820134596067*np.sin(t)**2))**2))
    )


def herman_hayden_quartz_reference_curve(theta_min_deg: float = 0.0, theta_max_deg: float = 65.0,
                                         n: int = 651, *, normalize: bool = True):
    """(theta_deg, I) reference Maker-fringe curve from the raw Herman-1995 analytic HH expression.

    Set ``normalize`` to divide by the peak (the overlay in the notebook compares shapes/fringe
    positions; the absolute scale carries the case's unit conventions)."""
    th = np.linspace(float(theta_min_deg), float(theta_max_deg), int(n))
    inten = herman_hayden_quartz_300um_intensity(np.deg2rad(th))
    # theta = 0 (normal incidence) is a removable 0/0 singularity of this analytic form (the
    # p-projection factor vanishes); interpolate any non-finite sample from its finite neighbours,
    # exactly as the SHAARP.ml Maker sweep interpolates its removable eigenmode singularities.
    bad = ~np.isfinite(inten)
    if bad.any() and (~bad).sum() >= 2:
        inten[bad] = np.interp(th[bad], th[~bad], inten[~bad])
    if normalize:
        peak = float(np.nanmax(inten))
        if peak > 0:
            inten = inten / peak
    return th, inten


if __name__ == "__main__":
    th, i = herman_hayden_quartz_reference_curve()
    print(f"Herman-Hayden analytic reference: {len(th)} pts over {th[0]:.0f}-{th[-1]:.0f} deg, "
          f"peak at theta={th[int(np.argmax(i))]:.2f} deg")
    print(f"  I(30 deg) = {float(herman_hayden_quartz_300um_intensity(np.deg2rad(30.0))):.6e} "
          f"(Mathematica: 4.875969e-05)")
