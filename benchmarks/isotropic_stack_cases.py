"""Case definitions and the four drivers for the isotropic-stack Fresnel benchmark.

The benchmark cross-checks SHAARP.py's linear (fundamental-frequency) multilayer stage against two
independent external codes and one analytic expression:

===========  ==========================================================================
leg          what it is
===========  ==========================================================================
``shaarp``   ``shaarp.api.run_fresnel_sweep`` -- the simultaneous boundary-matrix solve
``tmm``      Steven Byrnes' ``tmm`` package -- transfer-matrix method
``inkstone`` Song/Catrysse/Fan ``inkstone`` -- RCWA truncated to the zeroth order
``closed``   :mod:`benchmarks.isotropic_stack_closed_form` -- Abeles characteristic matrix
===========  ==========================================================================

``tmm`` and ``closed`` are two spellings of the same transfer-matrix idea, so on their own they
could share a convention error. ``inkstone`` is a genuinely different formalism -- a Fourier-space
eigenmode expansion which, for an unpatterned stack with a single retained order, must collapse to
the same answer. Agreement across all three is what makes the comparison load-bearing.

All stacks are ISOTROPIC and non-magnetic. Thicknesses and wavelengths are in micrometres. The
complex-index convention is ``n = n' + i k`` with ``exp(-i omega t)`` (loss is a POSITIVE imaginary
part) in every one of the four legs.

A note on the two half-space cases
----------------------------------
``inkstone`` cannot assemble a stack with no interior layer -- ``GetPowerFlux`` raises inside its
S-matrix routine. Both bare-interface cases therefore carry an interior layer whose index EQUALS
the substrate's, which is physically identical to a bare interface (the internal boundary has zero
index contrast) and is accepted by all four legs. This was verified explicitly: the emulated
air/glass interface returns R = 0.040000000 at normal incidence, the textbook value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["IsotropicStackCase", "CASES", "build_cases", "shaarp_rt", "tmm_rt", "inkstone_rt", "closed_form_rt"]

# Gold at 800 nm. Johnson & Christy (1972), Phys. Rev. B 6, 4370, interpolated to 800 nm; the same
# film thickness as the SHAARP.ml paper's Fig. 4(d) Au-coated quartz geometry.
AU_800NM = 0.1541 + 4.9096j
QUARTZ_800NM = 1.4533


@dataclass(frozen=True)
class IsotropicStackCase:
    """One benchmark stack plus the grid it is evaluated on.

    Exactly one of ``theta_deg`` / ``wavelength_um`` is the swept axis; the other is a scalar.
    ``n_list`` includes both semi-infinite media; ``thickness_um`` carries interior layers only.
    """

    case_id: str
    description: str
    n_list: tuple
    thickness_um: tuple
    sweep: str                       # "theta" or "wavelength"
    theta_deg: np.ndarray
    wavelength_um: np.ndarray
    notes: str = ""
    fringe_axis_period: float | None = None   # expected fringe period on the swept axis, if any
    metadata: dict = field(default_factory=dict)

    def grid(self) -> np.ndarray:
        return self.theta_deg if self.sweep == "theta" else self.wavelength_um


def _theta_grid(stop: float = 85.0, step: float = 2.5) -> np.ndarray:
    return np.round(np.arange(0.0, stop + 0.5 * step, step), 10)


def build_cases() -> list[IsotropicStackCase]:
    """The six benchmark cases. Kept as a function so callers cannot mutate shared arrays."""
    glass = 1.52
    film = 2.35
    low = 1.46
    lam_vis = 0.633

    # Case 3 fringe period on the wavelength axis. Transmission maxima sit at 2 n d = m lam, so
    # consecutive maxima are separated by roughly lam^2 / (2 n d) -- about 0.085 um at 400 nm and
    # 0.34 um at 800 nm. The grid below puts >5 samples across even the narrowest of those.
    spectral_grid = np.round(np.linspace(0.40, 0.80, 121), 10)
    narrowest_fringe_um = 0.40 ** 2 / (2.0 * film * 0.5)

    # Case 6: an absentee layer -- optical thickness exactly lam/2 at normal incidence, where the
    # film becomes invisible and R must equal the bare substrate's.
    absentee_thickness = lam_vis / (2.0 * film)

    return [
        IsotropicStackCase(
            case_id="bare_interface_air_glass",
            description="Bare air/glass interface (n=1.5), angle sweep. Brewster zero in p.",
            n_list=(1.0, 1.5, 1.5),
            thickness_um=(0.1,),
            sweep="theta",
            theta_deg=_theta_grid(),
            wavelength_um=np.array([lam_vis]),
            notes=(
                "Interior layer index equals the substrate index, so this is physically a bare "
                "interface; the layer exists only because inkstone cannot build a zero-layer stack."
            ),
            metadata={"brewster_deg": float(np.rad2deg(np.arctan(1.5)))},
        ),
        IsotropicStackCase(
            case_id="single_film_angle_sweep",
            description="air / n=2.35 x 0.5 um / n=1.52 at 633 nm, angle sweep.",
            n_list=(1.0, film, glass),
            thickness_um=(0.5,),
            sweep="theta",
            theta_deg=_theta_grid(),
            wavelength_um=np.array([lam_vis]),
        ),
        IsotropicStackCase(
            case_id="single_film_spectral_sweep",
            description="Same film at normal incidence, 400-800 nm. Fabry-Perot fringes.",
            n_list=(1.0, film, glass),
            thickness_um=(0.5,),
            sweep="wavelength",
            theta_deg=np.array([0.0]),
            wavelength_um=spectral_grid,
            notes="The interference leg proper: several full fringes across the sweep.",
            fringe_axis_period=narrowest_fringe_um,
        ),
        IsotropicStackCase(
            case_id="quarter_wave_stack_spectral",
            description="4x (n=2.35 / n=1.46) quarter-wave pairs on glass, 400-800 nm, normal incidence.",
            n_list=(1.0,) + (film, low) * 4 + (glass,),
            thickness_um=(lam_vis / (4.0 * film), lam_vis / (4.0 * low)) * 4,
            sweep="wavelength",
            theta_deg=np.array([0.0]),
            wavelength_um=spectral_grid,
            notes=(
                "Eight interior layers with a high-reflectance stopband centred on 633 nm. Layer "
                "chaining is exercised where a single-interface error cannot hide."
            ),
            fringe_axis_period=narrowest_fringe_um,
        ),
        IsotropicStackCase(
            case_id="absorbing_gold_film",
            description="air / 13.9 nm Au / quartz at 800 nm, angle sweep. Complex index; R+T < 1.",
            n_list=(1.0, AU_800NM, QUARTZ_800NM),
            thickness_um=(0.0139,),
            sweep="theta",
            theta_deg=_theta_grid(),
            wavelength_um=np.array([0.800]),
            notes="Mirrors the SHAARP.ml paper Fig. 4(d) Au-on-quartz geometry.",
            metadata={"au_index_source": "Johnson & Christy (1972) Phys. Rev. B 6, 4370, at 800 nm"},
        ),
        IsotropicStackCase(
            case_id="half_wave_absentee",
            description="Absentee film: optical thickness lam/2 at 633 nm, normal incidence only.",
            n_list=(1.0, film, glass),
            thickness_um=(absentee_thickness,),
            sweep="theta",
            theta_deg=np.array([0.0]),
            wavelength_um=np.array([lam_vis]),
            notes=(
                "Discriminating anchor: a half-wave layer is optically absent at normal incidence, "
                "so R must equal the BARE air/glass value ((1-1.52)/(1+1.52))^2, independent of the "
                "film index. A solver that merely looks plausible will not reproduce this."
            ),
            metadata={"expected_bare_R": float(((1.0 - glass) / (1.0 + glass)) ** 2)},
        ),
    ]


CASES = build_cases()


# --------------------------------------------------------------------------------------------
# Drivers. Each returns a dict of four float arrays keyed "R_s", "R_p", "T_s", "T_p", evaluated
# over case.grid(). Keeping the signatures identical is what lets the comparator stay generic.
# --------------------------------------------------------------------------------------------


def _sweep_points(case: IsotropicStackCase):
    """Yield ``(theta_deg, wavelength_um)`` for each point of the case's swept axis."""
    if case.sweep == "theta":
        lam = float(case.wavelength_um[0])
        for th in case.theta_deg:
            yield float(th), lam
    else:
        th = float(case.theta_deg[0])
        for lam in case.wavelength_um:
            yield th, float(lam)


def closed_form_rt(case: IsotropicStackCase) -> dict:
    from .isotropic_stack_closed_form import stack_rt

    out = {k: [] for k in ("R_s", "R_p", "T_s", "T_p")}
    for th, lam in _sweep_points(case):
        for pol, tag in (("s", "s"), ("p", "p")):
            r_val, t_val = stack_rt(case.n_list, case.thickness_um, np.deg2rad(th), lam, pol)
            out["R_" + tag].append(r_val)
            out["T_" + tag].append(t_val)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def tmm_rt(case: IsotropicStackCase) -> dict:
    import tmm as _tmm

    d_list = [np.inf] + [float(d) for d in case.thickness_um] + [np.inf]
    n_list = [complex(n) for n in case.n_list]
    out = {k: [] for k in ("R_s", "R_p", "T_s", "T_p")}
    for th, lam in _sweep_points(case):
        for pol in ("s", "p"):
            res = _tmm.coh_tmm(pol, n_list, d_list, np.deg2rad(th), lam)
            out["R_" + pol].append(float(res["R"]))
            out["T_" + pol].append(float(res["T"]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def inkstone_rt(case: IsotropicStackCase) -> dict:
    from inkstone import Inkstone

    out = {k: [] for k in ("R_s", "R_p", "T_s", "T_p")}
    for th, lam in _sweep_points(case):
        for pol in ("s", "p"):
            sim = Inkstone()
            # A deeply sub-wavelength cell with a single retained order leaves only the specular
            # channel propagating, which is the TMM limit of RCWA.
            sim.SetLattice(((1e-3, 0.0), (0.0, 1e-3)))
            sim.SetNumG(1)
            sim.SetFrequency(1.0 / lam)
            names = []
            for idx, n_val in enumerate(case.n_list):
                name = "m%d" % idx
                sim.AddMaterial(name=name, epsilon=complex(n_val) ** 2)
                names.append(name)
            sim.AddLayer(name="in", thickness=0.0, material_background=names[0])
            for idx, d_val in enumerate(case.thickness_um):
                sim.AddLayer(name="L%d" % idx, thickness=float(d_val), material_background=names[idx + 1])
            sim.AddLayer(name="out", thickness=0.0, material_background=names[-1])
            s_amp, p_amp = (1.0, 0.0) if pol == "s" else (0.0, 1.0)
            sim.SetExcitation(theta=float(th), phi=0.0, s_amplitude=s_amp, p_amplitude=p_amp)
            incident, backward = sim.GetPowerFlux("in")
            forward, _ = sim.GetPowerFlux("out")
            out["R_" + pol].append(float(-backward / incident))
            out["T_" + pol].append(float(forward / incident))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _shaarp_system(case: IsotropicStackCase, wavelength_um: float):
    from shaarp.config import Layer, MultilayerSystem
    from shaarp.layer_stack import _material_from_iso

    layers = [
        Layer(
            name="ambient",
            material=_material_from_iso(complex(case.n_list[0]).real, complex(case.n_list[0]).real, "ambient"),
            thickness_um=None,
            shg_active=False,
        )
    ]
    for idx, (n_val, d_val) in enumerate(zip(case.n_list[1:-1], case.thickness_um)):
        layers.append(
            Layer(
                name="layer%d" % idx,
                material=_iso_complex(n_val, "layer%d" % idx),
                thickness_um=float(d_val),
                shg_active=False,
            )
        )
    layers.append(
        Layer(
            name="substrate",
            material=_iso_complex(case.n_list[-1], "substrate"),
            thickness_um=None,
            shg_active=False,
        )
    )
    return MultilayerSystem(wavelength_um=float(wavelength_um), layers=layers)


def _iso_complex(n_value, name: str):
    """Isotropic Material carrying a possibly COMPLEX index.

    ``layer_stack._material_from_iso`` casts through ``float``, so it cannot express an absorbing
    layer; the gold case needs one. Same shape otherwise -- identity orientation, zero d tensor.
    """
    from shaarp.config import CrystalOrientation, CrystalStructure, Material

    eps = (np.eye(3) * complex(n_value) ** 2).astype(complex)
    return Material(
        name=name,
        structure=CrystalStructure(point_group="∞∞m"),
        orientation=CrystalOrientation(),
        epsilon_omega=eps,
        epsilon_2omega=eps,
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )


def shaarp_rt(case: IsotropicStackCase, *, transmittance: str = "power") -> dict:
    """SHAARP.py's linear multilayer Fresnel curves for a case.

    ``transmittance`` is passed straight through to :func:`shaarp.api.run_fresnel_sweep`:
    ``"power"`` (the shipped default) or ``"amplitude"`` for the bare ``|t|**2`` that SHAARP.ml's
    ``listFresnel`` emits. Exposing both lets the benchmark demonstrate WHICH one the external
    codes agree with rather than asserting it.
    """
    from shaarp.api import run_fresnel_sweep

    out = {k: [] for k in ("R_s", "R_p", "T_s", "T_p")}
    options = {
        "workflow": "gui_multilayer",
        "transmitted_wave_policy": "physical_sum",
        "mrassumption": 0,
        "transmittance": transmittance,
    }
    for th, lam in _sweep_points(case):
        system = _shaarp_system(case, lam)
        res = run_fresnel_sweep(system, np.array([th], dtype=float), options=options)
        for pol in ("s", "p"):
            out["R_" + pol].append(float(np.asarray(res.numeric["r" + pol], float)[0]))
            out["T_" + pol].append(float(np.asarray(res.numeric["t" + pol], float)[0]))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def flux_factor(n_incident, n_exit, theta_deg: float) -> float:
    """``Re(n_exit cos theta_exit) / Re(n_inc cos theta_inc)`` -- the obliquity factor.

    This is what converts ``|t|^2`` into a power transmittance: refraction changes the beam's
    width, so the two beams do not share a cross-section even though they share an interface area.
    """
    n_i = complex(n_incident)
    n_t = complex(n_exit)
    theta = np.deg2rad(float(theta_deg))
    cos_i = np.cos(theta) + 0j
    sin_t = n_i * np.sin(theta) / n_t
    cos_t = np.sqrt(1.0 - sin_t * sin_t + 0j)
    if np.imag(n_t * cos_t) < 0.0:
        cos_t = -cos_t
    return float(np.real(n_t * cos_t) / np.real(n_i * cos_i))
