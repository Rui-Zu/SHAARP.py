"""Compare SHAARP.py's linear Fresnel curves against the external isotropic-stack reference.

Consumed by ``tests/test_isotropic_stack_reference_comparison.py`` and by the live recheck. The
summary dict follows the same shape as the other reference comparators in this directory
(``status`` / ``max_abs_error`` / ``nonsingular_fail_count`` / per-case detail), so the tests read
the same way.

The transmittance is reported TWICE per case -- once as SHAARP's shipped ``transmittance="power"``
curves, and once for the legacy ``transmittance="amplitude"`` (bare ``|t|**2``) convention that
SHAARP.ml's ``listFresnel`` emits. Reporting both is deliberate: the test then SHOWS which of the
two the external codes agree with instead of asserting it, and the size of the difference stays on
the record permanently rather than living only in a commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .isotropic_stack_cases import build_cases, flux_factor, shaarp_rt

__all__ = [
    "load_reference",
    "reference_curves",
    "build_isotropic_stack_agreement_summary",
    "REFERENCE_PATH",
    "CURVE_KEYS",
    "REFLECTANCE_KEYS",
    "TRANSMITTANCE_KEYS",
]

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "isotropic_stack_reference_v1.json"
CURVE_KEYS = ("R_s", "R_p", "T_s", "T_p")
REFLECTANCE_KEYS = ("R_s", "R_p")
TRANSMITTANCE_KEYS = ("T_s", "T_p")


def load_reference(path: Path | None = None) -> dict:
    target = Path(path) if path is not None else REFERENCE_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def reference_curves(case_payload: dict, leg: str = "tmm") -> dict:
    """Curves for one case from one reference leg, as float arrays."""
    if leg not in case_payload["legs"]:
        raise KeyError("reference case %r has no leg %r" % (case_payload["case_id"], leg))
    return {k: np.asarray(case_payload["legs"][leg][k], dtype=float) for k in CURVE_KEYS}


def case_flux_factors(case) -> np.ndarray:
    """The obliquity factor at every point of a case's swept axis."""
    if case.sweep == "theta":
        angles = case.theta_deg
    else:
        angles = np.full(case.wavelength_um.shape, float(case.theta_deg[0]))
    return np.asarray(
        [flux_factor(case.n_list[0], case.n_list[-1], float(th)) for th in angles], dtype=float
    )


def build_isotropic_stack_agreement_summary(
    reference: dict,
    *,
    atol: float = 1e-12,
    rtol: float = 0.0,
    leg: str = "tmm",
    shaarp_curves: dict | None = None,
    legacy_curves: dict | None = None,
):
    """Compare SHAARP against ``leg`` of the reference.

    ``shaarp_curves`` maps ``case_id`` to the dict returned by
    :func:`benchmarks.isotropic_stack_cases.shaarp_rt`; when it is ``None`` the curves are computed
    here. Passing them in lets a caller run the (slower) SHAARP sweep once and reuse it.

    ``legacy_curves`` is the same mapping computed with ``transmittance="amplitude"``. When given,
    the per-case detail additionally records how far the legacy bare-``|t|**2`` convention sits
    from the external reference, which is what documents why the power weighting is needed.
    """
    cases = {case.case_id: case for case in build_cases()}
    per_case = []
    worst_r = 0.0
    worst_t = 0.0
    worst_t_legacy = 0.0
    fail_count = 0

    for payload in reference["cases"]:
        case_id = payload["case_id"]
        if case_id not in cases:
            raise KeyError(
                "reference carries case %r which build_cases() does not define -- the fixture and "
                "the case list have diverged; regenerate the fixture." % case_id
            )
        case = cases[case_id]
        ref = reference_curves(payload, leg=leg)
        curves = shaarp_curves[case_id] if shaarp_curves is not None else shaarp_rt(case)

        detail = {"case_id": case_id, "point_count": len(payload["grid"])}

        case_worst_r = 0.0
        for key in REFLECTANCE_KEYS:
            err = float(np.max(np.abs(curves[key] - ref[key])))
            detail["max_abs_error_" + key] = err
            case_worst_r = max(case_worst_r, err)

        case_worst_t = 0.0
        for key in TRANSMITTANCE_KEYS:
            err = float(np.max(np.abs(curves[key] - ref[key])))
            detail["max_abs_error_" + key] = err
            case_worst_t = max(case_worst_t, err)

        case_worst_t_legacy = None
        if legacy_curves is not None:
            case_worst_t_legacy = 0.0
            for key in TRANSMITTANCE_KEYS:
                raw_err = float(np.max(np.abs(legacy_curves[case_id][key] - ref[key])))
                detail["max_abs_error_legacy_" + key] = raw_err
                case_worst_t_legacy = max(case_worst_t_legacy, raw_err)
            worst_t_legacy = max(worst_t_legacy, case_worst_t_legacy)

        detail["max_abs_error_reflectance"] = case_worst_r
        detail["max_abs_error_transmittance"] = case_worst_t
        detail["max_abs_error_transmittance_legacy"] = case_worst_t_legacy

        case_worst = max(case_worst_r, case_worst_t)
        detail["max_abs_error"] = case_worst
        tolerance = atol + rtol * float(np.max(np.abs(np.concatenate([ref[k] for k in CURVE_KEYS]))))
        detail["tolerance"] = tolerance
        detail["passed"] = bool(case_worst <= tolerance)
        if not detail["passed"]:
            fail_count += 1

        worst_r = max(worst_r, case_worst_r)
        worst_t = max(worst_t, case_worst_t)
        per_case.append(detail)

    status = (
        "isotropic_stack_outputs_match_all_compared_cases"
        if fail_count == 0
        else "isotropic_stack_outputs_disagree"
    )
    return {
        "status": status,
        "reference_leg": leg,
        "case_count": len(per_case),
        "nonsingular_fail_count": fail_count,
        "max_abs_error": max(worst_r, worst_t),
        "max_abs_error_reflectance": worst_r,
        "max_abs_error_transmittance": worst_t,
        "max_abs_error_transmittance_legacy": worst_t_legacy if legacy_curves is not None else None,
        "atol": atol,
        "rtol": rtol,
        "per_case": per_case,
    }
