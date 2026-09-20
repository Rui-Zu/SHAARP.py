"""Wavelength-range (spectral) sweeps.

WHERE THE WAVELENGTH ACTUALLY ENTERS. The two solvers answer to the wavelength differently, and
the difference is not a modelling choice -- it follows from what each problem contains.

A SINGLE INTERFACE has no length in it. There is no layer thickness, so there is no propagation
phase and no scale to compare the wavelength against. With mu = eps0 = 1 both sides of the driven
wave equation, curl_curl(k) - omega^2 mu eps0 (eps E + P), are homogeneous of degree 2 in omega,
so the frequency cancels exactly: the same geometry evaluated at omega = 1, at 2*pi/0.8 and at
2*pi/1.55 returns the same reflected second-harmonic field to 5e-15. The wavelength therefore
reaches a single-interface answer through ONE channel only -- the dispersion of the permittivity,
eps(w) and eps(2w). That is why this module sweeps the MATERIAL and leaves the solver's omega
alone; the solver keeps its dimensionless 1 and 2.

A MULTILAYER does have a length, the layer thicknesses, so the wavelength reaches the answer
through two channels: the same permittivity dispersion, and the propagation phase exp(i k_z h).
At frozen permittivity the second channel is exactly a rescaling of every thickness -- lambda 0.8
with h = 5 um and lambda 1.0 with h = 6.25 um give the same intensity to 3e-14. The corollary is
worth stating plainly: a multilayer wavelength sweep at frozen permittivity is not a spectrum at
all, it is a thickness sweep wearing a wavelength label. A spectrum has to move eps or it is
telling the user something untrue, which is why the support check below is not optional decoration.

WHAT IS HELD FIXED. The SHG tensor d is held at its tabulated value across a sweep: the exported
registry stores one d per material with no wavelength axis, and no dispersion model for it ships
today. So a computed spectrum carries the dispersion of the LINEAR optics and the
wavelength-independence of the nonlinearity. Away from an electronic resonance d varies slowly and
the linear dispersion sets the spectral shape; approaching a resonance at w or 2w the real d(lam)
rises and a spectrum computed this way understates the variation. Every result carries this in its
``assumptions`` so it travels into an export rather than living only in the documentation.
"""
from __future__ import annotations

import contextvars
import math
import warnings
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from .casestudy_materials import (
    build_casestudy_material,
    build_casestudy_ml_system,
    casestudy_spectral_support,
)
from .config import Material, MultilayerSystem
from .shg import solve_single_interface_shg
from .tensors import rotate_d_voigt_crystal_to_lab, rotate_rank2_crystal_to_lab

# A spectrum is a factory, not a widened Material. `Material` is frozen and `dataclasses.replace`
# is used on it throughout the package, so a wavelength field on it would survive a
# replace(mat, epsilon_omega=...) and produce a Material that lies about its own wavelength.
# Provenance belongs on the producer: build_casestudy_material already returns a fresh object per
# call, so the factory protocol is a name for machinery that exists rather than new plumbing.
MaterialSpectrum = Callable[[float], Material]
SystemSpectrum = Callable[[float], MultilayerSystem]

# What the assumptions dict says on every spectral result. Carried with the data so an exported
# CSV or JSON states them too.
SPECTRAL_ASSUMPTIONS = {
    "d_voigt": "wavelength-independent (held at the tabulated value across the sweep)",
    "epsilon_2omega": "taken from the material's own second-harmonic data at each fundamental",
    "units": "mu = eps0 = 1",
}


def wavelength_grid(start_um: float, stop_um: float, step_um: float) -> np.ndarray:
    """Inclusive wavelength grid, in um.

    Same semantics as the incidence-angle grids the app already sweeps: the endpoint is included,
    a non-positive step and a reversed range both raise, and ``start == stop`` yields a ONE-point
    grid rather than an error -- a sweep collapsed to a single wavelength is a legitimate request
    and has to render as the ordinary single-wavelength result."""

    if float(start_um) <= 0:
        raise ValueError("Wavelength start must be positive.")
    return inclusive_grid(start_um, stop_um, step_um, what="Wavelength")


def inclusive_grid(start: float, stop: float, step: float, *, what: str = "Grid") -> np.ndarray:
    """Endpoint-inclusive grid; ``start == stop`` gives one point.

    The same construction the incidence-angle sweeps already use, kept here so the angle axis of a
    wavelength-by-angle map is built exactly like the angle axis of an ordinary scan -- an angle
    grid legitimately starts at zero, which the wavelength wrapper above forbids."""

    start, stop, step = float(start), float(stop), float(step)
    if step <= 0:
        raise ValueError(f"{what} step must be positive.")
    if stop < start:
        raise ValueError(f"{what} stop must be >= start.")
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return start + step * np.arange(count, dtype=float)


def casestudy_spectrum(display_name: str) -> MaterialSpectrum:
    """A :data:`MaterialSpectrum` for a Case Study material, tagged with its registry name so the
    support check can name it and consult its usable window."""

    # a dispersive variant is table-backed, not registry-backed; route it rather than let the
    # registry lookup fail on a name it has never heard of. Any name the variant answers to works
    # here -- including the short one the documentation prints.
    from .casestudy_materials import CASE_STUDY_ORDER
    from .dispersion import dispersive_material_names, dispersive_spectrum, resolve_dispersive_name

    full = resolve_dispersive_name(display_name)
    if full is not None:
        return dispersive_spectrum(full)
    if display_name not in CASE_STUDY_ORDER:
        # Checked HERE rather than at the first wavelength, and listing both kinds of name: the
        # registry's own error lists only its own materials, which left anyone reaching for a
        # dispersive one with a list that did not contain it.
        raise ValueError(
            f"unknown material {display_name!r}. Case Study materials: {list(CASE_STUDY_ORDER)}. "
            f"Materials with published index data: {dispersive_material_names()}.")

    def _at(lambda_um: float) -> Material:
        return build_casestudy_material(display_name, wavelength_um=float(lambda_um))

    _at.material_names = (display_name,)          # type: ignore[attr-defined]
    return _at


def casestudy_ml_spectrum(display_name: str, *, thickness_um: float = 1.0,
                          **system_kwargs) -> SystemSpectrum:
    """A :data:`SystemSpectrum` for the ambient / case-study film / substrate stack.

    Rebuilds the WHOLE system at each wavelength, which is the point: both the layer
    permittivities and omega = 2*pi/lambda have to move together. Setting only the system's
    wavelength and leaving its materials frozen is a silent bug -- it changes the propagation
    phase while the dispersion stays behind, which is a thickness sweep, not a spectrum."""

    # The same names casestudy_spectrum answers to, including a dispersive variant's short name --
    # this factory used to accept only the full "... 0.40-5.00 um" form while its sibling took both.
    from .casestudy_materials import CASE_STUDY_ORDER
    from .dispersion import dispersive_material_names, resolve_dispersive_name

    name = resolve_dispersive_name(display_name) or display_name
    if name not in CASE_STUDY_ORDER and name not in dispersive_material_names():
        raise ValueError(
            f"unknown material {display_name!r}. Case Study materials: {list(CASE_STUDY_ORDER)}. "
            f"Materials with published index data: {dispersive_material_names()}.")

    def _at(lambda_um: float) -> MultilayerSystem:
        return build_casestudy_ml_system(name, thickness_um=float(thickness_um),
                                         wavelength_um=float(lambda_um), **system_kwargs)

    _at.material_names = (name,)                  # type: ignore[attr-defined]
    return _at


def constant_spectrum(material: Material) -> MaterialSpectrum:
    """A :data:`MaterialSpectrum` that returns the SAME material at every wavelength.

    Explicit rather than implicit: a sweep over this is flat by construction, and the support
    check says so rather than letting a flat line pass for a spectrum."""

    def _at(_lambda_um: float) -> Material:
        return material

    _at.material_names = ()                        # type: ignore[attr-defined]
    return _at


def _permittivity_signature(built) -> list[np.ndarray]:
    """The permittivities a spectrum returns at one wavelength, for the flatness comparison.

    A MultilayerSystem contributes its LAYERS' tensors and deliberately NOT its wavelength_um.
    Wavelength changes a multilayer answer through two channels -- the permittivity and the
    propagation phase -- and at frozen permittivity only the second is left, which makes the
    result identical to a thickness sweep rather than a spectrum. Including wavelength_um in the
    signature would report that case as varying and hide exactly the situation worth warning
    about."""

    if isinstance(built, MultilayerSystem):
        out: list[np.ndarray] = []
        for layer in built.layers:
            out.extend((layer.material.eps_w(), layer.material.eps_2w()))
        return out
    return [built.eps_w(), built.eps_2w()]


def _lab_tensors(material: Material):
    rotation = material.orientation.rotation_matrix()
    return (rotate_rank2_crystal_to_lab(material.eps_w(), rotation),
            rotate_rank2_crystal_to_lab(material.eps_2w(), rotation),
            rotate_d_voigt_crystal_to_lab(np.asarray(material.d_voigt_pm_v, dtype=complex), rotation))


# WHO IS READING. The desktop app shows these messages verbatim in its amber note; a Python caller
# reads them in a warnings log or a traceback. Written once for both, the app's note told GUI users
# to "give it an index table" and to pass a keyword argument, in ASCII units beside the app's own
# µm. The app wraps its call in speaking_to("app"); everything else gets the Python wording.
_AUDIENCE: contextvars.ContextVar[str] = contextvars.ContextVar("shaarp_spectral_audience",
                                                                default="python")


class speaking_to:
    """``with speaking_to("app"):`` -- the support messages raised inside are worded for the app."""

    def __init__(self, audience: str):
        self.audience = audience

    def __enter__(self):
        self._token = _AUDIENCE.set(self.audience)
        return self

    def __exit__(self, *_exc):
        _AUDIENCE.reset(self._token)
        return False


def _in_app() -> bool:
    return _AUDIENCE.get() == "app"


def _um() -> str:
    return "µm" if _in_app() else "um"


def _eps2w() -> str:
    return "ε(2ω)" if _in_app() else "eps(2w)"


def _span(lo: float, hi: float) -> str:
    """A range as each audience writes it: an en dash in the app, like every other range there,
    and a plain hyphen for a Python caller, whose wording stays ASCII."""
    return f"{lo:g}{'–' if _in_app() else '-'}{hi:g}"


def _table_span(lo: float, hi: float) -> str:
    """An index table's span. In the app, to two decimals, as the material's name prints it --
    "covers 0.2894-1.064" beside a name reading 0.29-1.06 looked like a second table."""
    if _in_app():
        return f"{lo:.2f}–{hi:.2f}"
    return _span(lo, hi)


def _inward(value: float) -> float:
    """Rounded UP to 0.01 in the app, so a stated start of coverage is inside it (as the app's
    scan-range fit rounds it); exact for a Python caller."""
    return math.ceil(value * 100 - 1e-9) / 100 if _in_app() else value


# Air is exactly n = 1 at every wavelength: its "one index" is physics, not a limitation of its
# data, so it is never named as a material that fails to disperse.
_NOT_A_LIMITATION = {"Air"}


@dataclass(frozen=True)
class SpectralSupportReport:
    """Whether a grid is one this spectrum can honestly answer over."""

    grid_um: np.ndarray
    varies: bool | None          # None = a one-point grid, where flatness is not a question
    material_names: tuple[str, ...]
    flat_reason: str
    unusable: tuple[tuple[str, tuple[float, float]], ...]
    clamped: tuple[tuple[str, tuple[float, float]], ...]
    # index TABLES the sweep runs past, each with its own data range -- reported once per table
    # here instead of once per wavelength point by the table itself
    table_clamped: tuple[tuple[str, tuple[float, float]], ...] = ()
    # interior layers of a stack whose permittivity stays put while other layers disperse
    frozen_layers: tuple[str, ...] = ()
    # the registry materials whose data does not move, for naming them and their dispersive twin
    flat_names: tuple[str, ...] = ()
    # False for a FRESNEL map: linear optics at the fundamental, so the half-wavelength rules for
    # eps(2w) -- the pole screen and the table's lambda/2 coverage -- do not apply to it
    harmonic: bool = True
    # (layer, thickness_um, fringe period_um, step_um, near_um) when the wavelength step is too
    # coarse for the stack's interference fringes, else None
    fringe_alias: tuple | None = None

    def raise_if_unusable(self) -> None:
        """A wavelength where eps(2w) is not physical is a WRONG answer, not a degraded one, so it
        refuses rather than warns. Contrast :meth:`emit_warnings`, which covers the cases that are
        merely uninformative."""
        if not self.unusable:
            return
        um = _um()
        detail = "; ".join(f"{name} is physical over {_span(lo, hi)} {um}"
                           for name, (lo, hi) in self.unusable)
        raise ValueError(
            f"the requested wavelength range {_span(self.grid_um[0], self.grid_um[-1])} {um} "
            f"reaches outside the physical range of its material data ({detail}). Outside that "
            "window the material's index data runs into its ultraviolet pole at half the "
            f"wavelength and {_eps2w()} comes back unphysical, so a spectrum there would be wrong "
            "rather than approximate. Narrow the range, or pick a material whose index data "
            "covers it.")

    def _flat_message(self) -> str:
        """Which materials stand still, what that does on each path, and what to pick instead.

        The advice names the dispersive twin when there is one. "Pick the same crystal from the
        Dispersive group" was offered for materials that have no such crystal (quartz, a custom
        material), which sent a reader looking for something that is not there."""
        from .dispersion import dispersive_twin

        um = _um()
        reasons = []
        for name in self.flat_names:
            support = casestudy_spectral_support(name)
            span = support.tabulated_um
            if span and abs(span[1] - span[0]) < 1e-12:
                reasons.append(f"{name} is defined at one wavelength ({span[0]:g} {um})")
            elif span:
                reasons.append(f"{name} has the same index everywhere across {span[0]:g}-"
                               f"{span[1]:g} {um}")
            else:
                reasons.append(f"{name} has one index at every wavelength")
        if not reasons:
            reasons.append("the material in the fields has one index at every wavelength"
                           if _in_app() else
                           "the supplied material spectrum returns the same permittivity at every "
                           "wavelength")
        twins = []
        for name in self.flat_names:
            found = dispersive_twin(name)
            if found and found[0] not in [t for t, _ in twins]:
                # same template: just the name. Same crystal, another cut or wavelength: say which
                # cut it is, so nobody swaps a z-cut for an x-cut without knowing.
                twins.append((found[0], "" if found[1] == name else found[1]))
        named = " or ".join(t if not cut else f"{t} (cut as {cut})" for t, cut in twins)
        head = (f"The permittivity does not move with wavelength: {'; '.join(reasons)}. On a single "
                "interface the spectrum is flat; on a layer stack the curve still moves, but only "
                "through each layer's optical thickness, so it is a thickness sweep rather than a "
                "spectrum.")
        if _in_app():
            advice = (f"For a real spectrum pick {named}." if twins else
                      "For a real spectrum pick a crystal from the Dispersive group.")
            return f"{head} {advice}"
        advice = (f"For a real spectrum use {named}, a table-backed variant of the same crystal."
                  if twins else
                  "For a real spectrum use a material with index data "
                  "(shaarp.dispersion.dispersive_material_names(); the app's Dispersive group), or "
                  "give it an index table.")
        return f"{head} {advice} allow_constant_dispersion=True silences this warning."

    def emit_warnings(self, *, include_flat: bool = True) -> None:
        """Say everything the sweep cannot show, most consequential first.

        ``include_flat=False`` is the ``allow_constant_dispersion`` opt-in. It silences the
        flatness warning ONLY; the others are separate findings about the data and are still said.
        """
        um = _um()
        if include_flat and self.varies is False:
            # The consequence differs by path and both have to be stated: on a single interface a
            # frozen permittivity gives a flat line, which is obviously wrong. On a multilayer it
            # gives a curve that MOVES -- the propagation phase still responds to omega -- so it
            # looks like a spectrum while actually being a thickness sweep. The second case is the
            # dangerous one, and naming only the first would leave it unsaid.
            warnings.warn(self._flat_message(), RuntimeWarning, stacklevel=3)
        if self.fringe_alias:
            # See _fringe_alias for the measured basis of the period. The advice asks for five
            # samples per fringe. In the app it is rounded DOWN to what the 4-decimal step field can
            # hold -- a suggested 0.00057 used to round UP to 0.0006, still too coarse.
            layer, h, period, step, near = self.fringe_alias
            advice = period / 5.0
            if _in_app():
                fits = math.floor(advice * 1e4) / 1e4
                how = (f"Use a step of {fits:g} {um} or less over a narrower range." if fits > 0
                       else f"The fringes are finer than the step field can resolve here; narrow "
                            f"the range to a few fringes and use the smallest step (0.0001 {um}).")
            else:
                how = f"Use a step below about {advice:.2g} {um} over a narrower range."
            warnings.warn(
                f"The wavelength step ({step:g} {um}) is coarser than this stack's interference "
                f"fringes: {layer} ({h:g} {um} thick) puts them roughly {period:.2g} {um} apart "
                f"near {near:g} {um}, so the curve is undersampled and can show spikes that are "
                f"not spectral features. {how}",
                RuntimeWarning, stacklevel=3)
        if self.frozen_layers:
            # Not flat -- something disperses -- but a reader will take the curve as the whole
            # stack's spectrum, so say which layers are standing still inside it.
            one = len(self.frozen_layers) == 1
            warnings.warn(
                f"Only part of this stack disperses: {', '.join(self.frozen_layers)} "
                f"{'keeps' if one else 'keep'} a single index at every wavelength, so the spectrum "
                f"carries the other layers' dispersion but not {'its own' if one else 'their own'}.",
                RuntimeWarning, stacklevel=3)
        for name, (lo, hi) in self.table_clamped:
            if self.harmonic:
                # The half-wavelength rule is the part people miss: a name says 0.40-5.00 um, and
                # a sweep from 0.55 um still clamps, because eps(2w) is the table read at lambda/2.
                start = min(_inward(2.0 * lo), hi)
                answer = (f"{start:.2f}–{hi:.2f}" if _in_app() else _span(start, hi))
                warnings.warn(
                    f"{name}: its index data covers {_table_span(lo, hi)} {um}, and the second "
                    f"harmonic reads it at half the wavelength, so it can answer a sweep over "
                    f"{answer} {um}. Outside that the index is held at the nearest tabulated value "
                    "rather than extrapolated.",
                    RuntimeWarning, stacklevel=3)
            else:
                warnings.warn(
                    f"{name}: its index data covers {_table_span(lo, hi)} {um}; outside that range "
                    "the index is held at the nearest tabulated value rather than extrapolated.",
                    RuntimeWarning, stacklevel=3)
        for name, (lo, hi) in self.clamped:
            warnings.warn(
                f"Part of the requested range falls outside the tabulated grid of {name} "
                f"({_span(lo, hi)} {um}); its tensors clamp to the nearest tabulated value there, so "
                "the spectrum flattens at that end rather than extrapolating.",
                RuntimeWarning, stacklevel=3)


def check_spectral_support(spectrum: MaterialSpectrum, grid_um: Sequence[float],
                           *, allow_constant: bool = False,
                           harmonic: bool = True) -> SpectralSupportReport:
    """Decide whether ``spectrum`` can answer honestly over ``grid_um``.

    The flatness test is EMPIRICAL -- it evaluates the spectrum and compares the tensors it
    actually returns -- so it covers any factory, including one built from a user's own index
    table, rather than trusting a registry label. Named materials additionally contribute their
    usable and tabulated windows, which is what lets the message name the offender.

    ``harmonic=False`` is for linear optics at the fundamental (a Fresnel map): the ultraviolet pole
    at lambda/2 and a table's lambda/2 coverage belong to eps(2w), which a Fresnel map never reads.
    Applying them refused a KTP Fresnel sweep over a range its linear optics answer perfectly."""

    grid = np.asarray(grid_um, dtype=float)
    names = tuple(getattr(spectrum, "material_names", ()) or ())
    # Only REGISTRY materials have registry support to consult. A stack's row labels also include
    # half-space entries ("air", "isotropic n (set below)") and custom layers, which used to reach
    # the note as "'isotropic n (set below)' is not a Case Study material".
    from .casestudy_materials import CASE_STUDY_ORDER

    registry_names = tuple(n for n in names if n in CASE_STUDY_ORDER)

    # does eps actually move? compare the ends against the middle, so a spectrum that happens to
    # return to its starting value is not mistaken for a flat one. A ONE-POINT grid is left
    # undetermined rather than called flat: a sweep collapsed to a single wavelength is a
    # legitimate request (it has to render as the ordinary single-wavelength result), and there is
    # nothing for it to be flat over -- warning there would fire on every degenerate range.
    probes = sorted({float(grid[0]), float(grid[len(grid) // 2]), float(grid[-1])})
    varies: bool | None
    frozen_layers: tuple[str, ...] = ()
    fringe_alias = None
    flat_names = tuple(n for n in registry_names
                       if n not in _NOT_A_LIMITATION and not casestudy_spectral_support(n).varies)
    if len(probes) < 2:
        varies, flat_reason = None, ""
    else:
        with quiet_table_clamps():         # summarized once below, not once per probe
            built = [spectrum(p) for p in probes]
        signatures = [_permittivity_signature(b) for b in built]
        varies = any(not all(np.allclose(x, y) for x, y in zip(a, b))
                     for a, b in zip(signatures, signatures[1:]))
        if isinstance(built[0], MultilayerSystem):
            if varies:
                frozen_layers = _frozen_interior_layers(built)
            # every probe, not only the shortest wavelength: a layer can be opaque at the blue end
            # and fringe further out (GaAs above its band gap)
            fringe_alias = _fringe_alias(built, float(grid[1] - grid[0]), harmonic=harmonic)
        flat_reason = "" if varies else "; ".join(
            casestudy_spectral_support(n).reason for n in flat_names)

    unusable: list[tuple[str, tuple[float, float]]] = []
    clamped: list[tuple[str, tuple[float, float]]] = []
    for name in registry_names:
        support = casestudy_spectral_support(name)
        # A material whose data does not move has nothing to clamp: holding a constant at its end
        # value changes nothing, and saying it "flattens at that end" right after saying the whole
        # curve is flat contradicted the note above it.
        if not support.varies:
            continue
        window = support.usable_um
        if window is None:
            continue
        lo, hi = window
        span = support.tabulated_um
        # The pole region is the part of the TABULATED span that is not physical. Reaching past a
        # grid END is a different thing entirely -- the data simply stops and the tensors clamp --
        # so the overlap is taken against the span first. Conflating the two refused a perfectly
        # ordinary sweep off the red end of the grid. The pole is an eps(2w) property, so a linear
        # (Fresnel) map is never refused for it.
        if harmonic and span:
            probe_lo = max(float(grid[0]), span[0])
            probe_hi = min(float(grid[-1]), span[1])
            if probe_lo <= probe_hi and (probe_lo < lo - 1e-12 or probe_hi > hi + 1e-12):
                unusable.append((name, window))
        if span and (grid[0] < span[0] - 1e-12 or grid[-1] > span[1] + 1e-12):
            clamped.append((name, span))

    # Index tables: the sweep needs eps(w) over the grid, and for SHG eps(2w) at half of it too, so
    # the span a table has to cover is [grid_min / 2, grid_max] (just [grid_min, grid_max] for a
    # Fresnel map). Reported once per table, with the range it can actually answer, instead of the
    # table's own once-per-point warning.
    need_lo = float(grid[0]) / 2.0 if harmonic else float(grid[0])
    table_clamped: list[tuple[str, tuple[float, float]]] = []
    for table in _index_tables(spectrum, names):
        lo, hi = table.range_um
        if need_lo < lo - 1e-12 or float(grid[-1]) > hi + 1e-12:
            table_clamped.append((table.name, (float(lo), float(hi))))

    report = SpectralSupportReport(grid_um=grid, varies=varies, material_names=names,
                                   flat_reason=flat_reason, unusable=tuple(unusable),
                                   clamped=tuple(clamped), table_clamped=tuple(table_clamped),
                                   frozen_layers=frozen_layers, flat_names=flat_names,
                                   harmonic=harmonic, fringe_alias=fringe_alias)
    report.raise_if_unusable()
    # the opt-in silences the FLATNESS warning only; every other finding is about the data and is
    # still said
    report.emit_warnings(include_flat=not allow_constant)
    return report


# The table's own warning, raised once per wavelength it is read outside its data. Inside a sweep
# that is one line per grid point saying the same thing, and it pushed the more important notes
# out of sight, so the sweep reports each table once (SpectralSupportReport.table_clamped) and
# silences the per-point form while it runs. A single-wavelength read keeps it.
_TABLE_CLAMP_PATTERN = r".*has index data over "


class quiet_table_clamps(warnings.catch_warnings):
    """Silence an index table's per-point clamp warning for the duration of a sweep."""

    def __enter__(self):
        entered = super().__enter__()
        warnings.filterwarnings("ignore", message=_TABLE_CLAMP_PATTERN, category=RuntimeWarning)
        return entered


def _quietly(spectrum):
    """The same factory, with the per-point table clamp silenced while it builds."""
    def _at(lambda_um: float):
        with quiet_table_clamps():
            return spectrum(lambda_um)
    return _at


def _index_tables(spectrum, names: Sequence[str]) -> list:
    """Every index table feeding ``spectrum``: one attached to the factory itself (a user's own
    table), and any shipped dispersive material named among its materials (a stack in the app)."""
    from .dispersion import dispersive_material_names, load_shipped_table

    tables = []
    own = getattr(spectrum, "index_table", None)
    if own is not None:
        tables.append(own)
    shipped = set(dispersive_material_names())
    for name in names:
        if name in shipped and all(t.name != name for t in tables):
            tables.append(load_shipped_table(name))
    return tables


def _fringe_alias(built: Sequence[MultilayerSystem], step_um: float, *, harmonic: bool = True):
    """(layer, thickness, period, step, near) when the wavelength step cannot resolve the stack's
    interference fringes; None otherwise.

    The period estimate is lambda^2 / (4 n h), n the larger real index of the layer at the two
    harmonics, taken at each probe wavelength and minimised over the interior layers. MEASURED
    against it (2026-09-19), dominant oscillation period by FFT of a fine sweep:
        quartz 121.2 um   0.0009 um   vs 0.00085      LiNbO3 1 um   0.153 um  vs 0.131
        GaAs   1 um       0.068 um    vs 0.065        TaAs   1 um   no fringes (opaque)
    The fundamental's own Fabry-Perot period lambda^2 / (2 n h), used first, was 2x too coarse: it
    stayed silent on the GaAs film while a 0.04 um step drew one sample per fringe. A layer is left
    out only when BOTH harmonics die on a round trip -- that is what makes TaAs fringe-free, while
    GaAs, opaque at 2w alone, still fringes at 31%.

    ``harmonic=False`` (a Fresnel map) is linear optics at the fundamental: only the fundamental's
    Fabry-Perot period lambda^2 / (2 n_w h) is there, and only its own absorption can hide it -- the
    second-harmonic estimate over-warned by 2x. Measured at 40 deg by the spacing of maxima along a
    fine row: quartz 121.2 um 0.0019 um (lambda^2 / (2 n_w h) = 0.0017 near 0.8 um), 1 um quartz film
    0.25-0.33 um (0.23-0.32 across 0.6-1.4 um). A Maker map keeps the second-harmonic estimate: its
    transmitted rows fringe at 0.139 um on the 1 um film and 0.00168 um on the 121.2 um plate, both
    inside what lambda^2 / (4 n h) warns about.

    The warning fires at fewer than two samples per fringe (true aliasing: the curve can show
    features that are not there); the advice asks for five, which draws the fringes faithfully."""
    worst = None
    for system in built:
        lam = float(system.wavelength_um)
        for layer in system.layers[1:-1]:
            h = layer.thickness_um
            if not h or h <= 0:
                continue
            n_w = np.sqrt(np.diag(np.asarray(layer.material.eps_w(), dtype=complex)))
            n_2w = np.sqrt(np.diag(np.asarray(layer.material.eps_2w(), dtype=complex)))
            k_w, k_2w = float(np.max(np.abs(n_w.imag))), float(np.max(np.abs(n_2w.imag)))
            round_trip_w = math.exp(-2.0 * (4.0 * math.pi * k_w / lam) * float(h))
            round_trip_2w = math.exp(-2.0 * (4.0 * math.pi * k_2w / (lam / 2.0)) * float(h))
            if harmonic:
                if round_trip_w < 1e-3 and round_trip_2w < 1e-3:
                    continue                                 # opaque at both: no fringes
                n = float(max(np.max(np.abs(n_w.real)), np.max(np.abs(n_2w.real))))
                optical = 4.0                                # the 2w round trip sets the period
            else:
                if round_trip_w < 1e-3:
                    continue                                 # opaque at the fundamental
                n = float(np.max(np.abs(n_w.real)))
                optical = 2.0                                # the fundamental's own round trip
            if n <= 0:
                continue
            period = lam ** 2 / (optical * n * float(h))
            if worst is None or period < worst[2]:
                worst = (layer.name, float(h), period, lam)
    if worst is not None and step_um > worst[2] / 2.0:
        return (worst[0], worst[1], worst[2], float(step_um), worst[3])
    return None


def _frozen_interior_layers(built: Sequence[MultilayerSystem]) -> tuple[str, ...]:
    """Interior layers whose permittivity is the same at every probe while the stack as a whole
    disperses. The two half-spaces are left out: they are isotropic media the user sets by index
    (air, a substrate), and naming them on every sweep would bury the layer that matters."""
    first = built[0].layers
    out = []
    for i in range(1, len(first) - 1):
        tensors = [(b.layers[i].material.eps_w(), b.layers[i].material.eps_2w()) for b in built]
        if all(np.allclose(t[0], tensors[0][0]) and np.allclose(t[1], tensors[0][1])
               for t in tensors[1:]):
            out.append(first[i].name)
    return tuple(out)


@dataclass(frozen=True)
class SiSpectralSweepResult:
    """Reflected single-interface SHG across a wavelength grid, at one fixed geometry.

    Mirrors the shape of the incidence-angle sweep results: the observable arrays, the per-point
    diagnostics, and the full per-point solver results, with ``wavelength_um`` as the abscissa."""

    wavelength_um: np.ndarray
    reflected_s: np.ndarray
    reflected_p: np.ndarray
    intensity_s: np.ndarray
    intensity_p: np.ndarray
    intensity_analyzed: np.ndarray
    n_2omega_fast: np.ndarray
    n_2omega_slow: np.ndarray
    boundary_residual_norm: np.ndarray
    operator_condition: np.ndarray
    ill_conditioned: np.ndarray
    theta_deg: float
    phi_deg: float
    psi_deg: float
    ellipticity_deg: float
    support: SpectralSupportReport
    assumptions: dict
    results: tuple

    @property
    def list_spectrum(self) -> list[np.ndarray]:
        """(wavelength, value) column pairs, matching the copy payloads the angle sweeps return."""
        return [np.column_stack([self.wavelength_um, v])
                for v in (self.intensity_s, self.intensity_p, self.intensity_analyzed)]


def solve_si_spectral_sweep(
    spectrum: MaterialSpectrum,
    *,
    wavelength_um: Sequence[float] | None = None,
    lambda_min_um: float = 0.55,
    lambda_max_um: float = 1.60,
    lambda_step_um: float = 0.025,
    theta_deg: float = 45.0,
    phi_deg: float = 0.0,
    psi_deg: float = 0.0,
    ellipticity_deg: float = 0.0,
    incident_index_omega: complex = 1.0,
    incident_index_2omega: complex = 1.0,
    allow_constant_dispersion: bool = False,
) -> SiSpectralSweepResult:
    """Reflected SHG from one interface across a wavelength grid, at a FIXED geometry.

    The geometry is fixed on purpose. The observable has to be one scalar per wavelength for the
    result to be a spectrum, so the polarizer, the analyzer and the incidence angle are single
    values here rather than the swept quantities they are elsewhere.

    Per wavelength this runs the same arbitrary-Jones solve the app's own polarimetry curve runs
    at that angle, with ``incident_jones = (sin(phi) e^{i.ell}, cos(phi))``, so a spectrum and a
    polar plot agree where they meet. Note the generic solver's coefficient order: ``[0]`` is the
    reflected s amplitude and ``[1]`` the reflected p -- the compat workflow's order is the
    OPPOSITE, and conflating them silently swaps the two channels.

    ``psi_deg`` is the analyzer, with 0 deg detecting p and 90 deg detecting s, matching the app's
    stated convention.
    """

    grid = (wavelength_grid(lambda_min_um, lambda_max_um, lambda_step_um)
            if wavelength_um is None else np.asarray(wavelength_um, dtype=float))
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("wavelength_um must be a non-empty one-dimensional grid.")
    support = check_spectral_support(spectrum, grid, allow_constant=allow_constant_dispersion)
    spectrum = _quietly(spectrum)       # its clamps are in the report; say them once

    # the singular-incidence nudge lives with the GUI facade; imported lazily because that module
    # imports the api facade, which imports this one.
    from .shaarp_gui import _desingularize_theta_deg

    theta_rad = math.radians(_desingularize_theta_deg(float(theta_deg)))
    phi_rad = math.radians(float(phi_deg))
    psi_rad = math.radians(float(psi_deg))
    ell = complex(math.cos(math.radians(float(ellipticity_deg))),
                  math.sin(math.radians(float(ellipticity_deg))))
    jones = (math.sin(phi_rad) * ell, complex(math.cos(phi_rad)))

    n = grid.size
    e_s = np.empty(n, dtype=complex)
    e_p = np.empty(n, dtype=complex)
    n_fast = np.empty(n, dtype=complex)
    n_slow = np.empty(n, dtype=complex)
    residual = np.empty(n, dtype=float)
    condition = np.empty(n, dtype=float)
    ill = np.zeros(n, dtype=bool)
    results = []

    for i, lam in enumerate(grid):
        eps_w_lab, eps_2w_lab, d_lab = _lab_tensors(spectrum(float(lam)))
        result = solve_single_interface_shg(
            eps_w_lab, eps_2w_lab, d_lab,
            incident_index_omega=incident_index_omega,
            incident_index_2omega=incident_index_2omega,
            incident_theta_rad=theta_rad, incident_jones=jones,
            omega=1.0, mu=1.0, eps0=1.0)
        coefficients = np.asarray(result.coefficients)
        e_s[i] = complex(coefficients[0])
        e_p[i] = complex(coefficients[1])
        n_fast[i] = complex(result.transmitted_2omega_modes.fast.refractive_index)
        n_slow[i] = complex(result.transmitted_2omega_modes.slow.refractive_index)
        residual[i] = float(np.linalg.norm(np.asarray(result.boundary_residual)))
        fields = (result.inhomogeneous.ee, result.inhomogeneous.oo, result.inhomogeneous.eo)
        condition[i] = max(float(f.operator_condition) for f in fields)
        ill[i] = any(bool(f.ill_conditioned) for f in fields)
        results.append(result)

    if ill.any():
        # a wavelength where n(w) approaches n(2w) puts the bound source on a free-wave shell --
        # genuine phase matching, and a genuinely near-singular solve. Name the wavelengths rather
        # than letting a spike in the curve pass for structure.
        bad = ", ".join(f"{v:g}" for v in grid[ill])
        warnings.warn(
            f"the inhomogeneous solve is ill-conditioned at {int(ill.sum())} of {n} wavelengths "
            f"({bad} um): n(w) approaches n(2w) there, which is phase matching rather than a "
            "numerical fault. Those points are reported in ill_conditioned.",
            RuntimeWarning, stacklevel=2)

    intensity_s = (np.abs(e_s) ** 2).astype(float)
    intensity_p = (np.abs(e_p) ** 2).astype(float)
    analyzed = (np.abs(math.sin(psi_rad) * e_s + math.cos(psi_rad) * e_p) ** 2).astype(float)

    return SiSpectralSweepResult(
        wavelength_um=grid, reflected_s=e_s, reflected_p=e_p,
        intensity_s=intensity_s, intensity_p=intensity_p, intensity_analyzed=analyzed,
        n_2omega_fast=n_fast, n_2omega_slow=n_slow,
        boundary_residual_norm=residual, operator_condition=condition, ill_conditioned=ill,
        theta_deg=float(theta_deg), phi_deg=float(phi_deg), psi_deg=float(psi_deg),
        ellipticity_deg=float(ellipticity_deg),
        support=support, assumptions=dict(SPECTRAL_ASSUMPTIONS), results=tuple(results))


@dataclass(frozen=True)
class MlSpectralSweepResult:
    """Multilayer SHG across a wavelength grid, at one fixed geometry.

    Mirrors the Maker-fringe sweep result field for field, with ``wavelength_um`` in place of
    ``theta_deg`` and ``omega`` carried alongside it because on this path the wavelength reaches
    the answer through the propagation phase as well as through the permittivity."""

    wavelength_um: np.ndarray
    omega: np.ndarray
    channel: str
    analyzer_amplitude: np.ndarray
    intensity: np.ndarray
    fundamental_residual_norm: np.ndarray
    shg_residual_norm: np.ndarray
    theta_deg: float
    phi_deg: float
    psi_deg: float
    ellipticity_deg: float
    support: SpectralSupportReport
    assumptions: dict
    results: tuple

    @property
    def list_spectrum(self) -> list[np.ndarray]:
        return [np.column_stack([self.wavelength_um, self.intensity])]


def solve_ml_spectral_sweep(
    spectrum: SystemSpectrum,
    *,
    wavelength_um: Sequence[float] | None = None,
    lambda_min_um: float = 0.55,
    lambda_max_um: float = 1.60,
    lambda_step_um: float = 0.025,
    theta_deg: float | None = None,
    phi_deg: float | None = None,
    psi_deg: float | None = None,
    ellipticity_deg: float | None = None,
    channel: str = "reflected",
    mrassumption: int = 0,
    inhomogeneous_source_policy: str = "all",
    inhomogeneous_solution_policy: str = "solve",
    condition_threshold: float = 1e12,
    allow_constant_dispersion: bool = False,
) -> MlSpectralSweepResult:
    """Multilayer SHG across a wavelength grid, at a FIXED geometry.

    The system is rebuilt per wavelength, so the permittivities and omega = 2*pi/lambda move
    together. That rebuild includes the angle-independent setup the incidence-angle sweep hoists
    out of its loop; that hoist is not valid across wavelength, because the eigen-bases and the
    permittivity tensors are wavelength-dependent even though they are angle-independent. It costs
    little: the setup is a few per cent of a solve, measured, so the wavelength loop pays roughly
    the same per point as the angle loop.

    Geometry arguments default to whatever the rebuilt system already carries; pass them to
    override. ``channel`` selects the reflected or the transmitted second harmonic.
    """

    from .multilayer_shg_boundary import (
        analyze_reflected_2omega_polarimetry,
        analyze_transmitted_2omega_polarimetry,
        solve_multilayer_shg_from_system_polarimetry,
    )

    if channel not in ("reflected", "transmitted"):
        raise ValueError("channel must be 'reflected' or 'transmitted'")
    if mrassumption not in (0, 1, 2):
        raise ValueError("mrassumption must be 0 (FMR), 1 (JK) or 2 (HH)")
    # the same mapping the angle sweep and the GUI use, kept identical on purpose
    single_pass_omega = single_pass_2omega = False
    single_pass_omega_writeback = True
    if mrassumption == 2:
        single_pass_omega = True
    elif mrassumption == 1:
        single_pass_omega = True
        single_pass_omega_writeback = False
        single_pass_2omega = True

    grid = (wavelength_grid(lambda_min_um, lambda_max_um, lambda_step_um)
            if wavelength_um is None else np.asarray(wavelength_um, dtype=float))
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("wavelength_um must be a non-empty one-dimensional grid.")
    support = check_spectral_support(spectrum, grid, allow_constant=allow_constant_dispersion)
    spectrum = _quietly(spectrum)       # its clamps are in the report; say them once

    analyze = (analyze_reflected_2omega_polarimetry if channel == "reflected"
               else analyze_transmitted_2omega_polarimetry)

    n = grid.size
    amplitude = np.empty(n, dtype=complex)
    intensity = np.empty(n, dtype=float)
    fundamental_residual = np.empty(n, dtype=float)
    shg_residual = np.empty(n, dtype=float)
    results = []
    geometry = {}

    for i, lam in enumerate(grid):
        system = spectrum(float(lam))
        overrides = {k: v for k, v in (("theta_deg", theta_deg), ("phi_deg", phi_deg),
                                       ("psi_deg", psi_deg),
                                       ("ellipticity_deg", ellipticity_deg)) if v is not None}
        if overrides:
            system = replace(system, polarimetry=replace(system.polarimetry, **overrides))
        if not geometry:
            pol = system.polarimetry
            geometry = {"theta_deg": float(np.ravel(pol.theta_deg)[0]),
                        "phi_deg": float(np.ravel(pol.phi_deg)[0]),
                        "psi_deg": float(np.ravel(pol.psi_deg)[0]),
                        "ellipticity_deg": float(np.ravel(pol.ellipticity_deg)[0])}
        result = solve_multilayer_shg_from_system_polarimetry(
            system, mu=1.0, eps0=1.0, condition_threshold=condition_threshold,
            inhomogeneous_source_policy=inhomogeneous_source_policy,
            inhomogeneous_solution_policy=inhomogeneous_solution_policy,
            single_pass_omega=single_pass_omega,
            single_pass_omega_writeback=single_pass_omega_writeback,
            single_pass_2omega=single_pass_2omega)
        amp, inten = analyze(result.shg, system)
        amplitude[i] = amp
        intensity[i] = inten
        fundamental_residual[i] = float(np.linalg.norm(result.fundamental.residual))
        shg_residual[i] = float(np.linalg.norm(result.shg.residual))
        results.append(result)

    return MlSpectralSweepResult(
        wavelength_um=grid, omega=2.0 * np.pi / grid, channel=channel,
        analyzer_amplitude=amplitude, intensity=intensity,
        fundamental_residual_norm=fundamental_residual, shg_residual_norm=shg_residual,
        support=support, assumptions=dict(SPECTRAL_ASSUMPTIONS), results=tuple(results),
        **geometry)


@dataclass(frozen=True)
class SpectralAngleMapResult:
    """A wavelength-by-incidence-angle map, stored in LONG format.

    Every array is one-dimensional of length ``n_wavelength * n_theta``, including the two axes,
    which is what lets the ordinary CSV exporter write a two-dimensional result unchanged -- it
    requires equal flattened lengths and cannot hold a 2-D block beside 1-D axes. Use
    :meth:`grid` to get a channel back as a rectangle for plotting.

    A collapsed axis is a legitimate request rather than an error: a single wavelength makes this
    the ordinary incidence-angle scan, and a single angle makes it the ordinary spectrum. ``shape``
    is what a renderer branches on."""

    wavelength_um: np.ndarray
    theta_deg: np.ndarray
    shape: tuple[int, int]
    kind: str
    channels: dict
    support: SpectralSupportReport
    assumptions: dict

    @property
    def wavelength_axis(self) -> np.ndarray:
        return self.wavelength_um.reshape(self.shape)[:, 0]

    @property
    def theta_axis(self) -> np.ndarray:
        return self.theta_deg.reshape(self.shape)[0, :]

    @property
    def is_spectrum_only(self) -> bool:
        return self.shape[1] == 1

    @property
    def is_angle_scan_only(self) -> bool:
        return self.shape[0] == 1

    def grid(self, channel: str) -> np.ndarray:
        """One channel as an (n_wavelength, n_theta) rectangle."""
        return np.asarray(self.channels[channel]).reshape(self.shape)


MAP_CHANNELS = {"maker": ("parallel_intensity", "perpendicular_intensity"),
                "fresnel": ("rp", "rs", "tp", "ts")}


def solve_spectral_angle_map(
    spectrum: SystemSpectrum,
    *,
    wavelength_um: Sequence[float] | None = None,
    lambda_min_um: float = 0.60,
    lambda_max_um: float = 1.60,
    lambda_step_um: float = 0.05,
    theta_deg: Sequence[float] | None = None,
    theta_min_deg: float = 0.0,
    theta_max_deg: float = 45.0,
    theta_step_deg: float = 1.0,
    kind: str = "maker",
    phi_deg: float | None = None,
    psi_deg: float | None = None,
    ellipticity_deg: float | None = None,
    mrassumption: int = 0,
    inhomogeneous_source_policy: str = "forward_only",
    transmitted_wave_policy: str = "shaarp_ml_selected",
    allow_constant_dispersion: bool = False,
) -> SpectralAngleMapResult:
    """Maker fringes or Fresnel coefficients over a wavelength-by-angle grid.

    Hoisting is two-level and the split is the point. The incidence-angle scan's setup hoist stays
    valid INSIDE each wavelength, because the eigen-bases and permittivities are angle-independent;
    it is not valid ACROSS wavelength, because they are wavelength-dependent. So the system is
    rebuilt once per wavelength and the existing angle sweep runs underneath it unchanged -- which
    also means this adds nothing to the validated angle path and cannot perturb its fences.

    Cost is the product of the two grids. At roughly 27 ms a point a 21-by-46 map is about half a
    minute; a 0.1-degree angle step over a fine wavelength grid is minutes, so pick the wavelength
    grid coarse and the angle grid fine rather than both fine.
    """

    from .api import run_fresnel_sweep
    from .multilayer_shg_boundary import solve_multilayer_maker_fringes_sweep

    if kind not in MAP_CHANNELS:
        raise ValueError(f"kind must be one of {sorted(MAP_CHANNELS)}")
    lam_grid = (wavelength_grid(lambda_min_um, lambda_max_um, lambda_step_um)
                if wavelength_um is None else np.asarray(wavelength_um, dtype=float))
    th_grid = (inclusive_grid(theta_min_deg, theta_max_deg, theta_step_deg, what="Angle")
               if theta_deg is None else np.asarray(theta_deg, dtype=float))
    if lam_grid.ndim != 1 or lam_grid.size == 0:
        raise ValueError("wavelength_um must be a non-empty one-dimensional grid.")
    if th_grid.ndim != 1 or th_grid.size == 0:
        raise ValueError("theta_deg must be a non-empty one-dimensional grid.")

    support = check_spectral_support(spectrum, lam_grid, allow_constant=allow_constant_dispersion,
                                     harmonic=(kind != "fresnel"))
    spectrum = _quietly(spectrum)       # its clamps are in the report; say them once
    overrides = {k: v for k, v in (("phi_deg", phi_deg), ("psi_deg", psi_deg),
                                   ("ellipticity_deg", ellipticity_deg)) if v is not None}
    rows: dict[str, list[np.ndarray]] = {name: [] for name in MAP_CHANNELS[kind]}

    for lam in lam_grid:
        system = spectrum(float(lam))
        if overrides:
            system = replace(system, polarimetry=replace(system.polarimetry, **overrides))
        if kind == "maker":
            scan = solve_multilayer_maker_fringes_sweep(
                system, theta_deg=th_grid, mu=1.0, eps0=1.0, mrassumption=mrassumption,
                inhomogeneous_source_policy=inhomogeneous_source_policy,
                transmitted_wave_policy=transmitted_wave_policy)
            rows["parallel_intensity"].append(np.asarray(scan.parallel_intensity, dtype=float))
            rows["perpendicular_intensity"].append(
                np.asarray(scan.perpendicular_intensity, dtype=float))
        else:
            # the GUI multilayer workflow, not the legacy single-interface Fresnel helper: that
            # is the one compared against live SHAARP.ml. With the multiple-reflection model the
            # single-wavelength Fresnel run uses -- this call passed none, so every Fresnel map was
            # FMR whatever the Assumptions panel said (a JK map row was off by up to 0.25 in T_s).
            scan = run_fresnel_sweep(system, th_grid, {"workflow": "gui_multilayer",
                                                       "mrassumption": int(mrassumption)})
            for name in MAP_CHANNELS["fresnel"]:
                rows[name].append(np.asarray(scan.numeric[name], dtype=float))

    shape = (int(lam_grid.size), int(th_grid.size))
    lam_long = np.repeat(lam_grid, th_grid.size)
    th_long = np.tile(th_grid, lam_grid.size)
    channels = {name: np.concatenate(blocks) for name, blocks in rows.items()}
    assumptions = dict(SPECTRAL_ASSUMPTIONS)
    if kind == "fresnel":
        # Fresnel coefficients are linear optics at the fundamental: neither the SHG tensor nor the
        # second-harmonic permittivity enters. Listing how they were treated would state an
        # assumption this result never made -- the plot caption leaves it out for the same reason,
        # and an exported file must not say more than the plot it came from.
        assumptions.pop("d_voigt", None)
        assumptions.pop("epsilon_2omega", None)
        assumptions["optics"] = "linear, at the fundamental only"
    return SpectralAngleMapResult(
        wavelength_um=lam_long, theta_deg=th_long, shape=shape, kind=kind,
        channels=channels, support=support, assumptions=assumptions)
