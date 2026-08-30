"""Step-by-step multilayer SHG workflow scaffold over the validated public facades.

This module is a thin, notebook/REPL-friendly *scaffold* on top of the
already-validated SHAARP.py public API facades in :mod:`shaarp.api`:

* :func:`shaarp.api.run_maker_fringes` -- transmitted-SHG Maker fringes sweep,
* :func:`shaarp.api.run_sample_rotation` -- physical sample-azimuth sweep,
* :func:`shaarp.api.run_ml_numeric` -- staged multilayer numeric workflow.

Those facades are the validated end-to-end multilayer entry points (the
live-Wolfram multilayer breadth -- single-film, 4-layer isotropic interlayer,
5-layer anisotropic interlayer, and two coupled active layers -- agrees with
the Mathematica references for the SHAARP-selected transmitted-wave policy; see
``run_maker_fringes`` validation notes and
``stages['mathematica_validation_artifacts']`` when called with
``include_validation_summary=True``). This module does NOT re-implement any
physics; it only orchestrates those facades through small, documented,
independently-callable "cell" functions that walk a user through one validated
multilayer run.

Design rules (mirrored from :mod:`shaarp.interactive`):

* Only the *public* API is used (imported from :mod:`shaarp.api` and
  :mod:`shaarp.config`/:mod:`shaarp.presets`); no private internals are touched.
* Every function is pure and side-effect-free: no file writes, no global state,
  and no plotting that requires a display. Functions return plain ``dict`` data.
* ``matplotlib`` is NOT imported at module import time. The optional
  :func:`plot_maker_demo` helper imports nothing itself -- it only *draws onto*
  an ``ax`` object that the caller supplies, so the module stays importable in
  headless environments and without matplotlib installed.

How to use from a notebook or REPL
----------------------------------
.. code-block:: python

    from shaarp.interactive_multilayer import (
        build_demo_multilayer_system,
        run_maker_demo,
        run_sample_rotation_demo,
        run_ml_numeric_demo,
        summarize,
    )

    # Step 1 -- build a small, validated-style anisotropic stack.
    system = build_demo_multilayer_system()

    # Step 2 -- transmitted-SHG Maker fringes on a tiny incidence-angle grid.
    maker = run_maker_demo(system, theta_deg=[-10.0, 0.0, 10.0])
    print(maker["validation_status"])
    print(maker["theta_deg"], maker["parallel_intensity"])

    # Step 3 -- physical sample-azimuth sweep.
    rotation = run_sample_rotation_demo(system, azimuths_deg=[0.0, 45.0, 90.0])

    # Step 4 -- staged multilayer numeric workflow at the system polarimetry.
    numeric = run_ml_numeric_demo(system)

    # Optional: pull the full SHAARPResult shape into a plain dict.
    summarize(maker["result"])
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .api import run_maker_fringes, run_ml_numeric, run_sample_rotation
from .config import (
    CrystalOrientation,
    CrystalStructure,
    Layer,
    Material,
    MultilayerSystem,
    Polarimetry,
)
from . import presets
from .interactive import default_interactive_material


# A small default incidence-angle grid (degrees) kept intentionally tiny so the
# Maker fringes solve stays fast in a notebook. The Maker solver can be slow, so
# the saved demo grid is deliberately short.
DEFAULT_THETA_GRID_DEG: tuple[float, ...] = (-10.0, 0.0, 10.0)

# A small default sample-azimuth grid (degrees) for the rotation demo.
DEFAULT_AZIMUTH_GRID_DEG: tuple[float, ...] = (0.0, 45.0, 90.0)


def build_demo_substrate_material() -> Material:
    """Return a nondegenerate anisotropic, linear-only substrate material.

    The dielectric tensors are diagonal but non-degenerate (three distinct
    principal indices) at both omega and 2omega, which keeps the multilayer
    eigenmode basis well-conditioned. (Identical degenerate transverse
    eigenvectors produce a rank-deficient multilayer matrix; the validated
    facades expect non-degenerate stacks for the nonsingular path.) The
    substrate carries no SHG tensor -- it is a passive exit medium.
    """

    return Material(
        name="anisotropic-substrate-demo",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation.from_cubic_miller_surface((0, 0, 1)),
        epsilon_omega=np.diag([2.10**2, 2.25**2, 2.40**2]),
        epsilon_2omega=np.diag([2.30 + 0.02j, 2.50 + 0.02j, 2.70 + 0.02j]),
        d_voigt_pm_v=np.zeros((3, 6), dtype=complex),
    )


def build_demo_multilayer_system(*, theta_deg: float = 20.0) -> MultilayerSystem:
    """Return a small, validated-style multilayer stack for the demo workflow.

    The stack is ``air (incident) / nonlinear anisotropic film / anisotropic
    substrate``:

    * ``air in`` -- passive incident medium (``shg_active=False``),
    * ``nonlinear film`` -- the SHG-active anisotropic film
      (func:`shaarp.interactive.default_interactive_material`, a non-degenerate
      triclinic material with a complex ``d`` tensor), 0.2 um thick,
    * ``substrate`` -- a passive non-degenerate anisotropic exit medium.

    This mirrors the structure of
    :func:`shaarp.interactive.default_interactive_system` but uses an
    anisotropic substrate (instead of air-out) so the stack is representative of
    the validated multilayer cases. The returned system is already
    :meth:`MultilayerSystem.validate`-clean.

    Parameters
    ----------
    theta_deg:
        Incidence/polarizer angle (degrees) stored on the system polarimetry.
    """

    system = MultilayerSystem(
        wavelength_um=1.064,
        polarimetry=Polarimetry(
            theta_deg=float(theta_deg), phi_deg=0.0, psi_deg=0.0, ellipticity_deg=0.0
        ),
        layers=[
            Layer("air in", presets.air(), shg_active=False),
            Layer("nonlinear film", default_interactive_material(), thickness_um=0.2, shg_active=True),
            Layer("substrate", build_demo_substrate_material(), shg_active=False),
        ],
    )
    system.validate()
    return system


def run_maker_demo(
    system: MultilayerSystem,
    theta_deg: Sequence[float] | np.ndarray | None = None,
    *,
    transmitted_wave_policy: str = "shaarp_ml_selected",
) -> dict[str, Any]:
    """Run a transmitted-SHG Maker fringes sweep via :func:`run_maker_fringes`.

    Parameters
    ----------
    system:
        A :class:`MultilayerSystem` (e.g. from :func:`build_demo_multilayer_system`).
    theta_deg:
        One-dimensional incidence-angle grid in degrees. Defaults to the small
        :data:`DEFAULT_THETA_GRID_DEG` to stay fast.
    transmitted_wave_policy:
        Passed straight through to ``run_maker_fringes``. ``"shaarp_ml_selected"``
        (the default) is the Mathematica-validated SHAARP.ml GUI policy;
        ``"physical_sum"`` is the separate physical total-transmitted-field
        option.

    Returns
    -------
    dict
        A compact, JSON-friendly dict with keys:
        ``theta_deg``, ``parallel_intensity``, ``perpendicular_intensity``
        (all 1-D ``numpy`` arrays), ``parallel_amplitude``,
        ``perpendicular_amplitude`` (complex 1-D arrays), ``validation_status``
        (non-empty string), ``kind``, and ``result`` (the full
        :class:`~shaarp.api.SHAARPResult` for further inspection).
    """

    grid = np.asarray(
        DEFAULT_THETA_GRID_DEG if theta_deg is None else theta_deg, dtype=float
    )
    result = run_maker_fringes(
        system,
        grid,
        {
            "transmitted_wave_policy": transmitted_wave_policy,
            "include_validation_summary": False,
        },
    )
    numeric = result.numeric
    return {
        "kind": result.kind,
        "validation_status": result.validation.status,
        "theta_deg": np.asarray(numeric["theta_deg"], dtype=float),
        "parallel_intensity": np.asarray(numeric["parallel_intensity"], dtype=float),
        "perpendicular_intensity": np.asarray(numeric["perpendicular_intensity"], dtype=float),
        "parallel_amplitude": np.asarray(numeric["parallel_amplitude"]),
        "perpendicular_amplitude": np.asarray(numeric["perpendicular_amplitude"]),
        "result": result,
    }


def run_sample_rotation_demo(
    system: MultilayerSystem,
    azimuths_deg: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run a physical sample-azimuth sweep via :func:`run_sample_rotation`.

    The sample azimuth is a physical crystal/sample rotation about the lab
    surface normal, distinct from the polarizer angle ``phi_deg``.

    Parameters
    ----------
    system:
        A :class:`MultilayerSystem`.
    azimuths_deg:
        One-dimensional sample-azimuth grid in degrees. Defaults to the small
        :data:`DEFAULT_AZIMUTH_GRID_DEG`.

    Returns
    -------
    dict
        A compact dict with keys: ``sample_azimuth_deg``, ``intensity`` (1-D
        float arrays), ``analyzer_amplitude`` (complex 1-D array),
        ``validation_status`` (non-empty string), ``kind``, and ``result``.
    """

    grid = np.asarray(
        DEFAULT_AZIMUTH_GRID_DEG if azimuths_deg is None else azimuths_deg, dtype=float
    )
    result = run_sample_rotation(system, grid, {"include_validation_summary": False})
    numeric = result.numeric
    return {
        "kind": result.kind,
        "validation_status": result.validation.status,
        "sample_azimuth_deg": np.asarray(numeric["sample_azimuth_deg"], dtype=float),
        "intensity": np.asarray(numeric["intensity"], dtype=float),
        "analyzer_amplitude": np.asarray(numeric["analyzer_amplitude"]),
        "result": result,
    }


def run_ml_numeric_demo(system: MultilayerSystem) -> dict[str, Any]:
    """Run the staged multilayer numeric workflow via :func:`run_ml_numeric`.

    This evaluates the reflected 2omega analyzer response at the system's
    current single polarimetry setting (no sweep), and reports the boundary
    residual norms so the user can confirm the staged solve is well-conditioned.

    Returns
    -------
    dict
        A compact dict with keys: ``reflected_intensity`` (float),
        ``reflected_analyzer_amplitude`` (complex),
        ``fundamental_residual_norm`` and ``shg_residual_norm`` (floats),
        ``validation_status`` (non-empty string), ``kind``, and ``result``.
    """

    result = run_ml_numeric(system, {"include_validation_summary": False})
    numeric = result.numeric
    return {
        "kind": result.kind,
        "validation_status": result.validation.status,
        "reflected_intensity": float(np.asarray(numeric["reflected_intensity"])),
        "reflected_analyzer_amplitude": complex(np.asarray(numeric["reflected_analyzer_amplitude"]).item()),
        "fundamental_residual_norm": float(numeric["fundamental_residual_norm"]),
        "shg_residual_norm": float(numeric["shg_residual_norm"]),
        "result": result,
    }


def summarize(result: Any) -> dict[str, Any]:
    """Pull the key fields of a :class:`~shaarp.api.SHAARPResult` into a plain dict.

    Accepts either a raw ``SHAARPResult`` or one of the demo-step dicts returned
    by the ``run_*_demo`` functions (in which case its ``"result"`` entry is
    summarized). Returns ``kind``, ``validation_status``, ``validation_notes``,
    the physics ``conventions``, the shapes of every numeric array, and the
    stage keys -- mirroring :func:`shaarp.interactive._result_summary` but using
    only public attributes.
    """

    if isinstance(result, dict) and "result" in result:
        result = result["result"]
    return {
        "kind": result.kind,
        "validation_status": result.validation.status,
        "validation_notes": list(result.validation.notes),
        "conventions": {
            "field": result.conventions.field_convention,
            "boundary": result.conventions.tangential_boundary,
            "intensity": result.conventions.intensity_normalization,
            "phase_matching": result.conventions.phase_matching_policy,
            "maker_transmitted": result.conventions.maker_transmitted_policy,
        },
        "numeric_shapes": {
            key: list(np.asarray(value).shape) for key, value in result.numeric.items()
        },
        "numeric_keys": list(result.numeric),
        "stage_keys": list(result.stages),
    }


def plot_maker_demo(maker_demo: dict[str, Any], ax: Any) -> Any:
    """Plot a Maker demo result onto a caller-supplied matplotlib ``ax``.

    This helper never imports matplotlib itself and is never required at module
    import time: the caller creates the figure/axes (``fig, ax = plt.subplots()``)
    and passes ``ax`` in. It draws the parallel and perpendicular transmitted-SHG
    intensities versus incidence angle and returns the same ``ax``.

    Parameters
    ----------
    maker_demo:
        The dict returned by :func:`run_maker_demo`.
    ax:
        A matplotlib ``Axes`` object supplied by the caller.
    """

    theta = maker_demo["theta_deg"]
    ax.plot(theta, maker_demo["parallel_intensity"], marker="o", label="parallel")
    ax.plot(theta, maker_demo["perpendicular_intensity"], marker="s", label="perpendicular")
    ax.set_xlabel("incidence angle theta (deg)")
    ax.set_ylabel("transmitted SHG intensity (arb. units)")
    ax.set_title("Maker fringes demo")
    ax.legend()
    return ax
