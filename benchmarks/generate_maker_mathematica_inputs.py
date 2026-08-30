from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.generate_maker_fringes_benchmarks import build_cases  # noqa: E402
from benchmarks.generate_multilayer_system_benchmarks import complex_array_to_json  # noqa: E402


BENCHMARK_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Export complete Maker cases for Mathematica MFList generation.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "mathematica_reference" / "maker_mathematica_inputs_v1.json",
    )
    args = parser.parse_args()

    cases = [case_to_record(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "source": "Python explicit Maker input manifest for Mathematica SHAARP.ml MF export",
        "status": "python_input_manifest_not_mathematica_validation",
        "case_count": len(cases),
        "notes": [
            "These are inputs only, not validation results.",
            "Each layer includes tensors, orientation rows, QC = orientation^T, thickness, and a nonlinear flag for mat[pg][[-1]].",
            "The Mathematica exporter must compute MFList/listMFpara/listMFperp from SHAARP.ml; it must not copy Python output values.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} Maker Mathematica input cases to {args.output}")


def case_to_record(case: dict) -> dict:
    system = case["system"]
    pol = system.polarimetry
    return {
        "id": case["id"],
        "theta_deg": [float(theta) for theta in case["theta_deg"]],
        "phi_deg": float(case["phi_deg"]),
        "psi_deg": float(case["psi_deg"]),
        "ellipticity_deg": float(case["ellipticity_deg"]),
        "sample_azimuth_deg": float(case["sample_azimuth_deg"]),
        "wavelength_um": float(system.wavelength_um),
        "omega0": float(2 * np.pi / system.wavelength_um),
        "mrassumption": 0,
        "winhAssumption": 0,
        "polarimetry_check": {
            "theta_deg_system_scalar": float(np.asarray(pol.theta_deg, dtype=float)),
            "phi_deg_system_scalar": float(np.asarray(pol.phi_deg, dtype=float)),
            "psi_deg_system_scalar": float(np.asarray(pol.psi_deg, dtype=float)),
            "ellipticity_deg_system_scalar": float(np.asarray(pol.ellipticity_deg, dtype=float)),
        },
        "layers": [layer_record(layer, index) for index, layer in enumerate(system.layers)],
    }


def layer_record(layer, index: int) -> dict:
    material = layer.material
    orientation = material.orientation.rotation_matrix()
    return {
        "index": index,
        "name": layer.name,
        "material_name": material.name,
        "shg_active": bool(layer.shg_active),
        "nonlinear_flag": 1 if layer.shg_active else 0,
        "thickness_um": None if layer.thickness_um is None else float(layer.thickness_um),
        "point_group": material.structure.point_group,
        "latcon": [
            float(material.structure.a),
            float(material.structure.b),
            float(material.structure.c),
            float(material.structure.alpha_deg),
            float(material.structure.beta_deg),
            float(material.structure.gamma_deg),
        ],
        "orientation_rows": complex_array_to_json(orientation),
        "qc": complex_array_to_json(orientation.T),
        "epsilon_omega_crystal": complex_array_to_json(material.eps_w()),
        "epsilon_2omega_crystal": complex_array_to_json(material.eps_2w()),
        "d_voigt_crystal": complex_array_to_json(material.d_voigt()),
    }


if __name__ == "__main__":
    main()
