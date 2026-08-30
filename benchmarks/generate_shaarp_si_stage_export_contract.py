from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.wolfram_paths import PREAMBLE


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
REQUEST_PATH = REF_DIR / "shaarp_si_stage_reference_request_v1.json"
SYMBOL_SOURCE_PATH = REF_DIR / "shaarp_si_stage_symbol_sources_v1.json"
GAP_AUDIT_PATH = REF_DIR / "shaarp_si_symbol_gap_audit_v1.json"
CONTRACT_OUTPUT = REF_DIR / "shaarp_si_stage_export_contract_v1.json"
WL_OUTPUT = REF_DIR / "export_shaarp_si_stage_reference_template.wl"


ACTIVE_NUMERIC_REGION_ORDER = [
    "principal_axis_and_index_setup",
    "shg_tensor_input_matrix",
    "active_wave_equation_setup",
    "active_backward_wave_equation_setup",
    "omega_angle_root_assignments_active_path",
    "two_omega_angle_root_assignments_active_path",
    "omega_angle_roots_active_numerical_path",
    "omega_fresnel_field_setup_active_path",
    "omega_fresnel_equations_active_path",
    "omega_fresnel_substitution_active_path",
    "shg_tensor_transformation_active_path",
    "active_complex_field_pnl_block",
    "inhomogeneous_particular_solution_coefficients",
    "numerical_coefficients_and_two_omega_field_setup",
    "two_omega_boundary_solution",
    "reflected_signal_and_basic_intensity",
]

FRESNEL_SWEEP_REGION_ORDER = [
    "fresnel_sweep_table_initialization",
    "fresnel_sweep_angle_root_assignments",
    "omega_angle_roots_initial_path",
    "fresnel_sweep_table_rows",
]

COMMENTED_REFERENCE_REGION_ORDER = [
    "commented_real_field_pnl_block",
]

RESOLVED_INTENSITY_OUTPUTS = [
    "I2wx",
    "I2wy",
    "I2wParallel",
    "I2wPerpdicular",
    "MaxI2wx",
    "MaxI2wy",
    "MaxI2w",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAARP.si Mathematica stage export contract.")
    parser.add_argument("--contract-output", type=Path, default=CONTRACT_OUTPUT)
    parser.add_argument("--wolfram-output", type=Path, default=WL_OUTPUT)
    args = parser.parse_args()
    contract = build_contract()
    args.contract_output.parent.mkdir(parents=True, exist_ok=True)
    args.contract_output.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    args.wolfram_output.write_text(_wolfram_template(contract), encoding="utf-8")
    print(f"Wrote SHAARP.si stage export contract to {args.contract_output}")
    print(f"Wrote SHAARP.si Wolfram export template to {args.wolfram_output}")


def build_contract() -> dict:
    request = _load_json(REQUEST_PATH)
    sources = _load_json(SYMBOL_SOURCE_PATH)
    gap_audit = _load_json(GAP_AUDIT_PATH)
    active_symbols = _active_symbol_records(sources)
    return {
        "schema_version": 1,
        "source": "SHAARP.si Mathematica stage export contract generated from refined snippet evidence",
        "status": "export_contract_not_validation_evidence",
        "validation_claim": False,
        "request_file": "benchmarks/mathematica_reference/shaarp_si_stage_reference_request_v1.json",
        "symbol_source_file": "benchmarks/mathematica_reference/shaarp_si_stage_symbol_sources_v1.json",
        "gap_audit_file": "benchmarks/mathematica_reference/shaarp_si_symbol_gap_audit_v1.json",
        "dependency_audit_file": "benchmarks/mathematica_reference/shaarp_si_active_route_dependency_audit_v1.json",
        "case_input_mapping_file": "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_v1.json",
        "case_input_mapping_preflight_file": "benchmarks/mathematica_reference/shaarp_si_case_input_mapping_preflight_v1.json",
        "case_count": request["case_count"],
        "active_numeric_region_order": ACTIVE_NUMERIC_REGION_ORDER,
        "fresnel_sweep_region_order": FRESNEL_SWEEP_REGION_ORDER,
        "commented_reference_region_order": COMMENTED_REFERENCE_REGION_ORDER,
        "exact_active_requested_symbols": active_symbols["exact_active_requested_symbols"],
        "commented_reference_only_symbols": active_symbols["commented_reference_only_symbols"],
        "resolved_intensity_outputs_for_I2_gap": RESOLVED_INTENSITY_OUTPUTS,
        "unresolved_exact_requested_symbols": [
            {
                "symbol": record["symbol"],
                "classification": record["classification"],
                "assigned_symbols_containing_name": record["assigned_symbols_containing_name"],
                "policy": _gap_policy(record["symbol"]),
            }
            for record in gap_audit["records"]
        ],
        "export_status": {
            "mathematica_values_exported": False,
            "comparison_against_python_completed": False,
            "safe_to_claim_shaarp_agreement": False,
        },
        "notes": [
            "This contract defines what an eventual Mathematica export must emit; it does not contain Mathematica reference values.",
            "Do not export substring matches as exact symbols. P2e and Peo remain unresolved exact-name gaps.",
            "For the I2 gap, export the specific intensity functions/normalizers until an exact SHAARP.si I2 symbol is found.",
            "The active numeric route excludes commented real-field PNL except as reference-only evidence.",
        ],
    }


def _active_symbol_records(sources: dict) -> dict:
    exact_active = {}
    commented_only = {}
    for symbol, records in sources["symbol_sources"].items():
        active_records = [record for record in records if record["region_active_status"] == "active_or_setup_code"]
        commented_records = [record for record in records if record["region_active_status"] == "commented_reference_only"]
        if active_records:
            exact_active[symbol] = active_records
        elif commented_records:
            commented_only[symbol] = commented_records
    return {
        "exact_active_requested_symbols": exact_active,
        "commented_reference_only_symbols": commented_only,
    }


def _gap_policy(symbol: str) -> str:
    if symbol in {"P2e", "Peo"}:
        return "do_not_alias_to_P2eo_or_Peoo_without_notebook_evidence"
    if symbol == "I2":
        return "export_specific_intensity_functions_until_exact_I2_assignment_is_found"
    return "requires_manual_resolution_before_reference_export"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _wolfram_template(contract: dict) -> str:
    symbols = sorted(contract["exact_active_requested_symbols"])
    unresolved = [record["symbol"] for record in contract["unresolved_exact_requested_symbols"]]
    return "\n".join(
        [
            PREAMBLE,
            'contractPath = SHAARPPaths`Ref["shaarp_si_stage_export_contract_v1.json"];',
            'outputPath = SHAARPPaths`Ref["shaarp_si_stage_reference_values_v1.json"];',
            'contract = Import[contractPath, "RawJSON"];',
            "",
            "(*",
            "  Template only. This file intentionally does not claim validation.",
            "  Next implementation step: load/evaluate the verified SHAARP.si active numeric route,",
            "  assign each requested case input, then replace the missing-value placeholders below",
            "  with actual Mathematica values obtained from the original SHAARP.si equations.",
            "*)",
            "",
            "exportableSymbols = " + ToWolframString(symbols) + ";",
            "unresolvedSymbols = " + ToWolframString(unresolved) + ";",
            "",
            "Export[",
            "  outputPath,",
            "  <|",
            '    "source" -> "SHAARP.si stage reference export template",',
            '    "status" -> "mathematica_reference_export_template_not_values",',
            '    "validation_claim" -> False,',
            '    "wolfram_version" -> $Version,',
            '    "contract_path" -> contractPath,',
            '    "exportable_symbols" -> exportableSymbols,',
            '    "unresolved_symbols" -> unresolvedSymbols,',
            '    "case_count" -> contract["case_count"],',
            '    "cases" -> {}',
            "  |>,",
            '  "JSON"',
            "];",
            "Quit[]",
            "",
        ]
    )


def ToWolframString(values: list[str]) -> str:
    escaped = [value.replace("\\", "\\\\").replace('"', '\\"') for value in values]
    return "{" + ",".join(f'"{value}"' for value in escaped) + "}"


if __name__ == "__main__":
    main()
