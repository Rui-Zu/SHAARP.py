from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shaarp.multilayer_boundary import build_multilayer_boundary_system, solve_multilayer_boundary
from shaarp.waves import Wave, make_isotropic_wave


BENCHMARK_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate solveFresnelN-style boundary-system benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "multilayer_boundary_system_benchmarks_v1.json",
    )
    args = parser.parse_args()

    records = [run_case(case) for case in build_cases()]
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "python_solve_fresnel_n_system_regression_not_mathematica_validation",
        "case_count": len(records),
        "notes": [
            "These cases expose the raw solveFresnelN-style matrix, RHS, known residual, equation labels/metadata, unknown labels/metadata, and the waves used to build them.",
            "Equation labels use Mathematica's flattened order: E_x, E_y, H_x, H_y at each interface.",
            "Unknown labels are Python matrix-column labels, not Mathematica symbols. They name region and propagation role; mode_1/mode_2 are basis-order labels.",
            "In these isotropic fixtures, mode_1 is s and mode_2 is p. In tensor-generated anisotropic bases, mode_1/mode_2 correspond to the basis generator's fast/slow ordering.",
            "These are Python regression fixtures, not Mathematica-vs-Python validation.",
        ],
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} multilayer boundary-system benchmark cases to {args.output}")


def build_cases() -> list[dict]:
    return [
        make_case("boundary_system_01", theta_deg=0.0, layer_indices=[1.42 + 0.02j], substrate_index=1.18 + 0.01j, known_source=False),
        make_case("boundary_system_02", theta_deg=12.0, layer_indices=[1.55 + 0.04j], substrate_index=1.27 + 0.02j, known_source=True),
        make_case("boundary_system_03", theta_deg=28.0, layer_indices=[1.48 + 0.03j, 1.82 + 0.05j], substrate_index=1.33 + 0.025j, known_source=False),
        make_case("boundary_system_04", theta_deg=43.0, layer_indices=[1.62 + 0.06j, 1.95 + 0.04j], substrate_index=1.44 + 0.03j, known_source=True),
        make_case("boundary_system_05", theta_deg=55.0, layer_indices=[1.7 + 0.05j], substrate_index=1.52 + 0.02j, known_source=True),
        make_case("boundary_system_06", theta_deg=18.0, layer_indices=[1.36 + 0.02j, 1.66 + 0.04j, 2.02 + 0.06j], substrate_index=1.24 + 0.015j, known_source=True),
    ]


def make_case(
    case_id: str,
    *,
    theta_deg: float,
    layer_indices: list[complex],
    substrate_index: complex,
    known_source: bool,
) -> dict:
    return {
        "id": case_id,
        "theta_deg": theta_deg,
        "omega": 1.7,
        "top_index": 1.0 + 0.0j,
        "layer_indices": layer_indices,
        "substrate_index": substrate_index,
        "thicknesses": [0.18 + 0.07 * idx for idx in range(len(layer_indices))],
        "known_source": known_source,
    }


def run_case(case: dict) -> dict:
    theta_air = np.deg2rad(case["theta_deg"])
    omega = case["omega"]
    layer_thetas = [_snell_theta(case["top_index"], n, theta_air) for n in case["layer_indices"]]
    substrate_theta = _snell_theta(case["top_index"], case["substrate_index"], theta_air)

    top_known = [make_isotropic_wave(n=case["top_index"], theta_rad=theta_air, polarization="p", omega=omega)]
    top_basis = [
        make_isotropic_wave(n=case["top_index"], theta_rad=theta_air, polarization="s", direction_sign_z=-1, omega=omega),
        make_isotropic_wave(n=case["top_index"], theta_rad=theta_air, polarization="p", direction_sign_z=-1, omega=omega),
    ]
    layer_basis = [
        [
            make_isotropic_wave(n=n, theta_rad=theta, polarization="s", omega=omega),
            make_isotropic_wave(n=n, theta_rad=theta, polarization="p", omega=omega),
            make_isotropic_wave(n=n, theta_rad=theta, polarization="s", direction_sign_z=-1, omega=omega),
            make_isotropic_wave(n=n, theta_rad=theta, polarization="p", direction_sign_z=-1, omega=omega),
        ]
        for n, theta in zip(case["layer_indices"], layer_thetas)
    ]
    substrate_basis = [
        make_isotropic_wave(n=case["substrate_index"], theta_rad=substrate_theta, polarization="s", omega=omega),
        make_isotropic_wave(n=case["substrate_index"], theta_rad=substrate_theta, polarization="p", omega=omega),
    ]
    layer_known = _known_sources(layer_basis, case["known_source"])

    system = build_multilayer_boundary_system(
        top_known=top_known,
        layer_known=layer_known,
        top_unknown_basis=top_basis,
        layer_unknown_basis=layer_basis,
        substrate_unknown_basis=substrate_basis,
        thicknesses=case["thicknesses"],
        mu=1.0,
    )
    solution = solve_multilayer_boundary(
        top_known=top_known,
        layer_known=layer_known,
        top_unknown_basis=top_basis,
        layer_unknown_basis=layer_basis,
        substrate_unknown_basis=substrate_basis,
        thicknesses=case["thicknesses"],
        mu=1.0,
    )
    record = {
        "id": case["id"],
        "theta_deg": case["theta_deg"],
        "layer_count": len(case["layer_indices"]),
        "known_source": case["known_source"],
        "inputs": {
            "omega": complex_scalar_to_json(omega),
            "top_index": complex_scalar_to_json(case["top_index"]),
            "layer_indices": complex_vector_to_json(case["layer_indices"]),
            "substrate_index": complex_scalar_to_json(case["substrate_index"]),
            "thicknesses": case["thicknesses"],
        },
        "outputs": {
            "matrix": complex_array_to_json(system.matrix),
            "rhs": complex_vector_to_json(system.rhs),
            "known_residual": complex_vector_to_json(system.known_residual),
            "coefficients": complex_vector_to_json(solution.coefficients),
            "residual": complex_vector_to_json(solution.residual),
            "equation_labels": system.equation_labels,
            "equation_metadata": system.equation_metadata,
            "unknown_labels": system.unknown_labels,
            "unknown_metadata": system.unknown_metadata,
            "unknown_basis_waves": _unknown_basis_wave_records(
                system.unknown_metadata,
                top_basis,
                layer_basis,
                substrate_basis,
                mu=1.0,
            ),
            "known_waves": _known_wave_records(
                top_known=top_known,
                layer_known=layer_known,
                substrate_known=[],
                mu=1.0,
            ),
        },
        "metrics": {
            "matrix_shape": list(system.matrix.shape),
            "rhs_l2": round(float(np.linalg.norm(system.rhs)), 15),
            "coefficient_l2": round(float(np.linalg.norm(solution.coefficients)), 15),
            "residual_l2": round(float(np.linalg.norm(solution.residual)), 15),
            "condition_number": round(float(np.linalg.cond(system.matrix)), 15),
            "max_abs_matrix_imag": round(float(np.max(np.abs(np.imag(system.matrix)))), 15),
            "max_abs_rhs_imag": round(float(np.max(np.abs(np.imag(system.rhs)))), 15),
        },
    }
    record["fingerprint"] = fingerprint_record(record)
    return record


def _known_sources(layer_basis: list[list[Wave]], enabled: bool) -> list[list[Wave]]:
    if not enabled:
        return [[] for _ in layer_basis]
    sources = []
    for layer_idx, layer in enumerate(layer_basis):
        base = layer[layer_idx % len(layer)]
        electric = np.array(
            [
                0.02 * (layer_idx + 1) + 0.015j,
                -0.01j * (layer_idx + 2),
                0.008 - 0.006j * (layer_idx + 1),
            ],
            dtype=complex,
        )
        sources.append([replace(base, electric=electric)])
    return sources


def _unknown_basis_wave_records(
    unknown_metadata: list[dict[str, object]],
    top_basis: list[Wave],
    layer_basis: list[list[Wave]],
    substrate_basis: list[Wave],
    *,
    mu: float,
) -> list[dict]:
    waves = list(top_basis) + [wave for layer in layer_basis for wave in layer] + list(substrate_basis)
    return [
        {
            "metadata": metadata,
            "wave": wave_to_json(wave, mu=mu),
        }
        for metadata, wave in zip(unknown_metadata, waves)
    ]


def _known_wave_records(
    *,
    top_known: list[Wave],
    layer_known: list[list[Wave]],
    substrate_known: list[Wave],
    mu: float,
) -> list[dict]:
    records = []
    for index, wave in enumerate(top_known, start=1):
        records.append(
            {
                "region": "top",
                "layer_index": None,
                "basis_index": index,
                "role": "known_top_wave",
                "wave": wave_to_json(wave, mu=mu),
            }
        )
    for layer_index, layer in enumerate(layer_known, start=1):
        for basis_index, wave in enumerate(layer, start=1):
            records.append(
                {
                    "region": "layer",
                    "layer_index": layer_index,
                    "basis_index": basis_index,
                    "role": "known_layer_source",
                    "wave": wave_to_json(wave, mu=mu),
                }
            )
    for index, wave in enumerate(substrate_known, start=1):
        records.append(
            {
                "region": "substrate",
                "layer_index": None,
                "basis_index": index,
                "role": "known_substrate_wave",
                "wave": wave_to_json(wave, mu=mu),
            }
        )
    return records


def _snell_theta(n0: complex, n1: complex, theta0: complex) -> complex:
    return np.arcsin(n0 * np.sin(theta0) / n1)


def complex_array_to_json(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=complex)
    return [[complex_scalar_to_json(item) for item in row] for row in arr]


def complex_vector_to_json(values) -> list:
    return [complex_scalar_to_json(item) for item in np.asarray(values, dtype=complex)]


def complex_scalar_to_json(value: complex) -> dict:
    return {
        "real": round(float(np.real(value)), 15),
        "imag": round(float(np.imag(value)), 15),
    }


def wave_to_json(wave: Wave, *, mu: float) -> dict:
    magnetic = wave.magnetic(mu=mu)
    return {
        "omega": complex_scalar_to_json(wave.omega),
        "k": complex_vector_to_json(wave.k),
        "electric": complex_vector_to_json(wave.electric),
        "magnetic": complex_vector_to_json(magnetic),
        "tangential_eh": complex_vector_to_json([wave.electric[0], wave.electric[1], magnetic[0], magnetic[1]]),
    }


def fingerprint_record(record: dict) -> str:
    comparable = dict(record)
    comparable.pop("fingerprint", None)
    data = json.dumps(comparable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    main()
