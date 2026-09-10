"""Generate the committed external reference for the isotropic-stack Fresnel benchmark.

Runs the three NON-SHAARP legs -- ``tmm``, ``inkstone`` and the Abeles closed form -- over the
cases in :mod:`benchmarks.isotropic_stack_cases` and writes
``benchmarks/isotropic_stack_reference_v1.json``.

    python benchmarks/generate_isotropic_stack_reference.py

Requires the ``benchmark`` extra::

    pip install -e ".[benchmark]"

The fixture is committed so the gating test runs everywhere -- in CI, in the frozen app, offline --
without either package installed. ``tests/test_isotropic_stack_live_recheck.py`` recomputes the
same numbers when the packages ARE present and fails if the fixture has drifted, so it cannot go
stale silently.

Two refusals are built in, because a reference that is quietly wrong is worse than no reference:

1. The three legs must agree with each other to ``CROSS_LEG_TOLERANCE``. They are two different
   formalisms plus a hand-written textbook expression; if they disagree, the reference is broken
   and nothing should be written.
2. Any case declaring a fringe period must carry at least ``MIN_SAMPLES_PER_FRINGE`` samples across
   the narrowest fringe. Aliased oscillatory sweeps have twice shipped wrong answers in this repo.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.isotropic_stack_cases import (  # noqa: E402
    build_cases,
    closed_form_rt,
    inkstone_rt,
    tmm_rt,
)

OUTPUT_PATH = ROOT / "benchmarks" / "isotropic_stack_reference_v1.json"
CROSS_LEG_TOLERANCE = 1e-11
MIN_SAMPLES_PER_FRINGE = 5.0
CURVE_KEYS = ("R_s", "R_p", "T_s", "T_p")


def _complex_pairs(values):
    return [[float(np.real(v)), float(np.imag(v))] for v in values]


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:  # pragma: no cover - only when the extra is missing
        return "not-installed"


def _check_fringe_sampling(case) -> float | None:
    """Return samples-per-fringe, raising if the swept grid cannot resolve the oscillation."""
    if case.fringe_axis_period is None:
        return None
    grid = np.asarray(case.grid(), dtype=float)
    if grid.size < 2:
        raise ValueError("%s declares a fringe period but has a single grid point." % case.case_id)
    step = float(np.max(np.diff(grid)))
    samples = float(case.fringe_axis_period) / step
    if samples < MIN_SAMPLES_PER_FRINGE:
        raise ValueError(
            "%s samples its narrowest fringe %.2f times (need >= %.1f). Refusing to write an "
            "aliased reference." % (case.case_id, samples, MIN_SAMPLES_PER_FRINGE)
        )
    return samples


def _cross_leg_error(legs: dict) -> float:
    worst = 0.0
    names = sorted(legs)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for key in CURVE_KEYS:
                worst = max(worst, float(np.max(np.abs(legs[first][key] - legs[second][key]))))
    return worst


def build_payload() -> dict:
    cases_out = []
    worst_cross_leg = 0.0

    for case in build_cases():
        samples_per_fringe = _check_fringe_sampling(case)
        legs = {
            "tmm": tmm_rt(case),
            "inkstone": inkstone_rt(case),
            "closed_form": closed_form_rt(case),
        }
        cross = _cross_leg_error(legs)
        worst_cross_leg = max(worst_cross_leg, cross)
        if cross > CROSS_LEG_TOLERANCE:
            raise ValueError(
                "%s: the three reference legs disagree by %.3e (tolerance %.1e). The reference is "
                "broken -- not writing it." % (case.case_id, cross, CROSS_LEG_TOLERANCE)
            )

        cases_out.append(
            {
                "case_id": case.case_id,
                "description": case.description,
                "notes": case.notes,
                "n_list_real_imag": _complex_pairs(case.n_list),
                "thickness_um": [float(d) for d in case.thickness_um],
                "sweep": case.sweep,
                "theta_deg": [float(v) for v in case.theta_deg],
                "wavelength_um": [float(v) for v in case.wavelength_um],
                "grid": [float(v) for v in case.grid()],
                "samples_per_fringe": samples_per_fringe,
                "cross_leg_max_abs_error": cross,
                "metadata": dict(case.metadata),
                "legs": {name: {k: [float(x) for x in curves[k]] for k in CURVE_KEYS}
                         for name, curves in legs.items()},
            }
        )

    return {
        "source": (
            "External transfer-matrix and RCWA reference for the SHAARP.py isotropic multilayer "
            "Fresnel benchmark. Legs: tmm (Byrnes, transfer-matrix), inkstone (Song/Catrysse/Fan, "
            "RCWA truncated to the zeroth order), and an Abeles characteristic-matrix closed form."
        ),
        "status": "external_reference_exported",
        "generated_by": "benchmarks/generate_isotropic_stack_reference.py",
        "packages": {
            "tmm": _package_version("tmm"),
            "inkstone": _package_version("inkstone"),
            "numpy": np.__version__,
        },
        "python_version": platform.python_version(),
        "convention": "n = n' + i k with exp(-i omega t); loss is a positive imaginary part.",
        "units": "thicknesses and wavelengths in micrometres; angles in degrees",
        "quantities": (
            "R and T are POWER ratios. T carries the obliquity factor "
            "Re(n_exit cos theta_exit)/Re(n_inc cos theta_inc), so a lossless stack satisfies R+T=1."
        ),
        "cross_leg_tolerance": CROSS_LEG_TOLERANCE,
        "cross_leg_max_abs_error": worst_cross_leg,
        "min_samples_per_fringe": MIN_SAMPLES_PER_FRINGE,
        "case_count": len(cases_out),
        "cases": cases_out,
    }


def main() -> int:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("wrote %s" % OUTPUT_PATH.relative_to(ROOT).as_posix())
    print("  cases                    : %d" % payload["case_count"])
    print("  tmm / inkstone versions  : %s / %s" % (payload["packages"]["tmm"], payload["packages"]["inkstone"]))
    print("  worst cross-leg deviation: %.3e (tolerance %.1e)"
          % (payload["cross_leg_max_abs_error"], payload["cross_leg_tolerance"]))
    for case in payload["cases"]:
        spf = case["samples_per_fringe"]
        print("  %-28s pts=%3d  cross-leg %.2e%s"
              % (case["case_id"], len(case["grid"]), case["cross_leg_max_abs_error"],
                 "" if spf is None else "  samples/fringe %.1f" % spf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
