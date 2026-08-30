from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
LINE_AUDIT_PATH = REF_DIR / "shaarp_si_candidate_region_line_audit.json"
OUTPUT_PATH = REF_DIR / "shaarp_si_refined_extraction_map_v1.json"


REFINED_REGIONS = [
    (
        "principal_axis_and_index_setup",
        277,
        307,
        "Principal-axis/eigenvector setup for Qw/Q2w and refractive-index tensors.",
    ),
    (
        "shg_tensor_input_matrix",
        309,
        311,
        "Construction of the crystal-frame 3x6 SHG tensor matrix dold from dij entries.",
    ),
    (
        "fresnel_sweep_table_initialization",
        318,
        320,
        "GUI-style incident-angle sweep setup for Index and Fresnel tables.",
    ),
    (
        "fresnel_sweep_angle_root_assignments",
        340,
        348,
        "GUI-style Fresnel sweep ordinary/extraordinary root and angle assignments inside RefractiveIndexModule.",
    ),
    (
        "omega_angle_roots_initial_path",
        368,
        371,
        "Initial ordinary/extraordinary angle branch-factor check in the Fresnel sweep path.",
    ),
    (
        "fresnel_sweep_table_rows",
        449,
        465,
        "GUI-style Fresnel sweep row assignments for reflected and transmitted p/s coefficients.",
    ),
    (
        "active_wave_equation_setup",
        481,
        491,
        "Active single-angle wave-equation setup for L, Rstandard, Cstandard, AwNum, and A2wNum.",
    ),
    (
        "active_backward_wave_equation_setup",
        493,
        504,
        "Active backward-wave equation setup for reflected 2omega branch calculations.",
    ),
    (
        "omega_angle_root_assignments_active_path",
        508,
        519,
        "Active single-angle ordinary/extraordinary root assignments for ThetaOrd and ThetaExt before branch-factor check.",
    ),
    (
        "two_omega_angle_root_assignments_active_path",
        524,
        533,
        "Active single-angle 2omega ordinary/extraordinary root assignments before branch-factor check.",
    ),
    (
        "omega_angle_roots_active_numerical_path",
        536,
        574,
        "Later ordinary/extraordinary omega and 2omega effective-index branch-factor checks used by the active numerical path.",
    ),
    (
        "omega_transmitted_field_directions_active_path",
        576,
        695,
        "Active omega transmitted ordinary/extraordinary field direction, displacement, wavevector, and magnetic field setup.",
    ),
    (
        "omega_incident_controls_active_path",
        697,
        716,
        "Active omega incident polarizer, ellipticity, and analyzer control setup before Fresnel boundary equations.",
    ),
    (
        "omega_fresnel_field_setup_active_path",
        720,
        728,
        "Active omega incident/reflected E/D/k/H field setup before boundary equations.",
    ),
    (
        "omega_fresnel_equations_active_path",
        732,
        737,
        "Active omega Fresnel tangential boundary equations and EwrootsNum solve.",
    ),
    (
        "omega_fresnel_substitution_active_path",
        740,
        753,
        "Active omega coefficient substitution and transmitted field/wavevector setup after EwrootsNum.",
    ),
    (
        "shg_tensor_transformation_active_path",
        759,
        773,
        "Active SHG tensor rotation from crystal frame dold to lab-frame dSHG before complex-field PNL.",
    ),
    (
        "commented_real_field_pnl_block",
        775,
        789,
        "Real-field nonlinear polarization block. It is inside a Mathematica comment and should not be treated as active code.",
    ),
    (
        "active_complex_field_pnl_block",
        790,
        810,
        "Active complex-field nonlinear polarization block for P2eo, P2o, and Peoo.",
    ),
    (
        "inhomogeneous_particular_solution_coefficients",
        815,
        838,
        "Particular-solution coordinate vector, coefficient equations, and Croots solve.",
    ),
    (
        "numerical_coefficients_and_two_omega_field_setup",
        840,
        890,
        "Numerical effective indices, wave constants, Croots substitution, nonlinear displacement, and magnetic-field setup.",
    ),
    (
        "two_omega_boundary_solution",
        899,
        908,
        "2omega displacement/magnetic fields, E2wroots solve, and reflected/transmitted 2omega coefficients.",
    ),
    (
        "reflected_signal_and_basic_intensity",
        920,
        938,
        "RSignal, I2wx/I2wy, RSignalCrystal, I2wParallel/I2wPerpdicular, and basic max-intensity definitions before plotting.",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate refined SHAARP.si extraction map from line audit.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_map()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote SHAARP.si refined extraction map to {args.output}")


def build_map() -> dict:
    audit = json.loads(LINE_AUDIT_PATH.read_text(encoding="utf-8"))
    numerical = next(item for item in audit["records"] if item["region_id"] == "si_numerical_shg_compute_candidate")
    lines = _line_index(numerical)
    regions = [_region_record(numerical, lines, *item) for item in REFINED_REGIONS]
    return {
        "schema_version": 1,
        "source": "SHAARP.si refined extraction map from PlainText numerical candidate line audit",
        "status": "refined_extraction_map_not_validation_evidence",
        "notebook_path": audit["notebook_path"],
        "coordinate_system": audit.get("coordinate_system", "unknown"),
        "line_audit": "benchmarks/mathematica_reference/shaarp_si_candidate_region_line_audit.json",
        "base_region": {
            "region_id": numerical["region_id"],
            "absolute_start": numerical["start"],
            "absolute_end": numerical["end"],
            "line_count": numerical["line_count"],
        },
        "regions": regions,
        "notes": [
            "Line ranges are based on front-end PlainText conversion, not original notebook box offsets.",
            "Offsets use the line audit coordinate system and must be applied to the same normalized text.",
            "The real-field PNL block is explicitly marked commented and should not be used as active numerical code.",
            "The active complex-field PNL block is the current target for SI nonlinear polarization export.",
            "This map is routing evidence for exporter construction, not a Mathematica value export.",
        ],
    }


def _line_index(region: dict) -> dict[int, dict]:
    records = (
        region["first_lines"]
        + region.get("target_window_lines", [])
        + region["interesting_lines"]
        + region["last_lines"]
    )
    indexed: dict[int, dict] = {}
    for record in records:
        indexed[record["line_index"]] = record
    return indexed


def _region_record(
    base_region: dict,
    lines: dict[int, dict],
    region_id: str,
    line_start: int,
    line_end: int,
    rationale: str,
) -> dict:
    start_record = lines.get(line_start)
    end_record = lines.get(line_end)
    absolute_start = None if start_record is None else base_region["start"] + start_record["offset"] - 1
    absolute_end = None if end_record is None else base_region["start"] + end_record["offset"] + end_record["length"] - 2
    return {
        "region_id": region_id,
        "line_start": line_start,
        "line_end": line_end,
        "absolute_start": absolute_start,
        "absolute_end": absolute_end,
        "byte_count": None if absolute_start is None or absolute_end is None else absolute_end - absolute_start + 1,
        "status": "candidate_not_validated",
        "rationale": rationale,
        "start_line_text": None if start_record is None else start_record["text"],
        "end_line_text": None if end_record is None else end_record["text"],
    }


if __name__ == "__main__":
    main()
