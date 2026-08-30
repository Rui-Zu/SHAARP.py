from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .waves import MU0, Wave, tangential_eh


@dataclass(frozen=True)
class MultilayerBoundarySolution:
    """Solved amplitudes for a SHAARP `solveFresnelN`-style boundary system."""

    top_unknown: list[Wave]
    layer_unknowns: list[list[Wave]]
    substrate_unknown: list[Wave]
    coefficients: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class MultilayerBoundarySystem:
    """Linear system backing a SHAARP `solveFresnelN`-style boundary solve."""

    matrix: np.ndarray
    rhs: np.ndarray
    known_residual: np.ndarray
    equation_labels: list[str]
    equation_metadata: list[dict[str, object]]
    unknown_labels: list[str]
    unknown_metadata: list[dict[str, object]]


def solve_multilayer_boundary(
    *,
    top_known: list[Wave],
    layer_known: list[list[Wave]] | None = None,
    substrate_known: list[Wave] | None = None,
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
    thicknesses: list[float],
    mu: float = MU0,
) -> MultilayerBoundarySolution:
    """Solve the numeric core of Mathematica `solveFresnelN`.

    The equations enforce tangential E/H continuity at the top surface, every
    internal layer boundary, and the bottom surface. Fields inside each layer
    are propagated to that layer's bottom by exp(I k_z thickness), matching the
    structure extracted from the Mathematica notebook.
    """

    if layer_known is None:
        layer_known = [[] for _ in layer_unknown_basis]
    if substrate_known is None:
        substrate_known = []
    system = build_multilayer_boundary_system(
        top_known=top_known,
        layer_known=layer_known,
        substrate_known=substrate_known,
        top_unknown_basis=top_unknown_basis,
        layer_unknown_basis=layer_unknown_basis,
        substrate_unknown_basis=substrate_unknown_basis,
        thicknesses=thicknesses,
        mu=mu,
    )
    coeffs = np.linalg.solve(system.matrix, system.rhs)
    top_unknown, layer_unknowns, substrate_unknown = _apply_coefficients(
        coeffs,
        top_unknown_basis,
        layer_unknown_basis,
        substrate_unknown_basis,
    )
    residual = _multilayer_residual(
        top_waves=top_known + top_unknown,
        layer_waves=[known + unknown for known, unknown in zip(layer_known, layer_unknowns)],
        substrate_waves=substrate_known + substrate_unknown,
        thicknesses=thicknesses,
        mu=mu,
    )
    return MultilayerBoundarySolution(
        top_unknown=top_unknown,
        layer_unknowns=layer_unknowns,
        substrate_unknown=substrate_unknown,
        coefficients=coeffs,
        residual=residual,
    )


def build_multilayer_boundary_system(
    *,
    top_known: list[Wave],
    layer_known: list[list[Wave]] | None = None,
    substrate_known: list[Wave] | None = None,
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
    thicknesses: list[float],
    mu: float = MU0,
) -> MultilayerBoundarySystem:
    """Build the linear system solved by `solve_multilayer_boundary`.

    `equation_labels` follow Mathematica's `solveFresnelN` flattened equation
    order: for each interface, `E_x`, `E_y`, `H_x`, `H_y`.
    `equation_metadata` stores the same row order in structured form.
    `unknown_labels` follow the Python/SHAARP basis order and name the region
    plus propagation role. They are Python column labels, not Mathematica
    unknown symbols. `unknown_metadata` stores the same order in a structured
    form for future matrix-column comparison.
    """

    _validate_boundary_inputs(
        layer_known=layer_known,
        substrate_known=substrate_known,
        top_unknown_basis=top_unknown_basis,
        layer_unknown_basis=layer_unknown_basis,
        substrate_unknown_basis=substrate_unknown_basis,
        thicknesses=thicknesses,
    )
    if layer_known is None:
        layer_known = [[] for _ in layer_unknown_basis]
    if substrate_known is None:
        substrate_known = []

    known_residual = _multilayer_residual(
        top_waves=top_known,
        layer_waves=layer_known,
        substrate_waves=substrate_known,
        thicknesses=thicknesses,
        mu=mu,
    )
    columns = []
    for region, layer_index, wave in _iter_basis_regions(
        top_unknown_basis,
        layer_unknown_basis,
        substrate_unknown_basis,
    ):
        columns.append(
            _multilayer_residual(
                top_waves=[wave] if region == "top" else [],
                layer_waves=_single_layer_wave_list(region, layer_index, wave, len(layer_unknown_basis)),
                substrate_waves=[wave] if region == "substrate" else [],
                thicknesses=thicknesses,
                mu=mu,
            )
        )
    return MultilayerBoundarySystem(
        matrix=np.column_stack(columns),
        rhs=-known_residual,
        known_residual=known_residual,
        equation_labels=_equation_labels(len(layer_unknown_basis)),
        equation_metadata=_equation_metadata(len(layer_unknown_basis)),
        unknown_labels=_unknown_labels(top_unknown_basis, layer_unknown_basis, substrate_unknown_basis),
        unknown_metadata=_unknown_metadata(top_unknown_basis, layer_unknown_basis, substrate_unknown_basis),
    )


def _validate_boundary_inputs(
    *,
    layer_known: list[list[Wave]] | None,
    substrate_known: list[Wave] | None,
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
    thicknesses: list[float],
) -> None:
    del substrate_known
    if len(layer_unknown_basis) == 0:
        raise ValueError("At least one internal layer is required.")
    if len(thicknesses) != len(layer_unknown_basis):
        raise ValueError("thicknesses must match the number of internal layers.")
    if layer_known is not None and len(layer_known) != len(layer_unknown_basis):
        raise ValueError("layer_known must match the number of internal layers.")

    bases = _flatten_unknowns(top_unknown_basis, layer_unknown_basis, substrate_unknown_basis)
    expected_unknowns = 4 * (len(layer_unknown_basis) + 1)
    if len(bases) != expected_unknowns:
        raise ValueError(
            f"Expected {expected_unknowns} unknown basis waves for {len(layer_unknown_basis)} layers, "
            f"got {len(bases)}."
        )


def _equation_labels(n_layers: int) -> list[str]:
    return [str(item["label"]) for item in _equation_metadata(n_layers)]


def _equation_metadata(n_layers: int) -> list[dict[str, object]]:
    interfaces = ["top"]
    interfaces.extend(f"layer_{idx}_to_layer_{idx + 1}" for idx in range(1, n_layers))
    interfaces.append("bottom")
    metadata = []
    for interface_index, interface in enumerate(interfaces):
        for component in ("E_x", "E_y", "H_x", "H_y"):
            metadata.append(
                {
                    "row_index": len(metadata),
                    "label": f"{interface}:{component}",
                    "interface": interface,
                    "interface_index": interface_index,
                    "component": component,
                    "field": component[0],
                    "axis": component[-1],
                }
            )
    return metadata


def _unknown_labels(
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
) -> list[str]:
    return [
        str(item["label"])
        for item in _unknown_metadata(top_unknown_basis, layer_unknown_basis, substrate_unknown_basis)
    ]


def _unknown_metadata(
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
) -> list[dict[str, object]]:
    metadata = []
    for mode_index, _ in enumerate(top_unknown_basis, start=1):
        metadata.append(
            _unknown_column_metadata(
                column_index=len(metadata),
                label=_top_unknown_label(mode_index, len(top_unknown_basis)),
                region="top",
                layer_index=None,
                basis_index=mode_index,
                propagation="reflected" if len(top_unknown_basis) == 2 else "unknown",
                mode_index=mode_index if len(top_unknown_basis) == 2 else None,
            )
        )

    for layer_index, layer in enumerate(layer_unknown_basis, start=1):
        for basis_index, _ in enumerate(layer, start=1):
            propagation, mode_index = _layer_basis_role(basis_index, len(layer))
            metadata.append(
                _unknown_column_metadata(
                    column_index=len(metadata),
                    label=_layer_unknown_label(layer_index, basis_index, len(layer)),
                    region="layer",
                    layer_index=layer_index,
                    basis_index=basis_index,
                    propagation=propagation,
                    mode_index=mode_index,
                )
            )

    for mode_index, _ in enumerate(substrate_unknown_basis, start=1):
        metadata.append(
            _unknown_column_metadata(
                column_index=len(metadata),
                label=_substrate_unknown_label(mode_index, len(substrate_unknown_basis)),
                region="substrate",
                layer_index=None,
                basis_index=mode_index,
                propagation="forward" if len(substrate_unknown_basis) == 2 else "unknown",
                mode_index=mode_index if len(substrate_unknown_basis) == 2 else None,
            )
        )
    return metadata


def _unknown_column_metadata(
    *,
    column_index: int,
    label: str,
    region: str,
    layer_index: int | None,
    basis_index: int,
    propagation: str,
    mode_index: int | None,
) -> dict[str, object]:
    return {
        "column_index": column_index,
        "label": label,
        "region": region,
        "layer_index": layer_index,
        "basis_index": basis_index,
        "propagation": propagation,
        "mode_index": mode_index,
    }


def _top_unknown_label(mode_index: int, basis_length: int) -> str:
    if basis_length == 2:
        return f"top_reflected_mode_{mode_index}"
    return f"top_unknown_{mode_index}"


def _layer_unknown_label(layer_index: int, basis_index: int, basis_length: int) -> str:
    propagation, mode_index = _layer_basis_role(basis_index, basis_length)
    if mode_index is None:
        return f"layer_{layer_index}_unknown_{basis_index}"
    return f"layer_{layer_index}_{propagation}_mode_{mode_index}"


def _substrate_unknown_label(mode_index: int, basis_length: int) -> str:
    if basis_length == 2:
        return f"substrate_forward_mode_{mode_index}"
    return f"substrate_unknown_{mode_index}"


def _layer_basis_role(basis_index: int, basis_length: int) -> tuple[str, int | None]:
    if basis_length != 4:
        return "unknown", None
    if basis_index <= 2:
        return "forward", basis_index
    return "backward", basis_index - 2


def _flatten_unknowns(
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
) -> list[Wave]:
    return list(top_unknown_basis) + [wave for layer in layer_unknown_basis for wave in layer] + list(substrate_unknown_basis)


def _iter_basis_regions(
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
):
    for wave in top_unknown_basis:
        yield "top", None, wave
    for idx, layer in enumerate(layer_unknown_basis):
        for wave in layer:
            yield "layer", idx, wave
    for wave in substrate_unknown_basis:
        yield "substrate", None, wave


def _single_layer_wave_list(region: str, layer_index: int | None, wave: Wave, n_layers: int) -> list[list[Wave]]:
    layers = [[] for _ in range(n_layers)]
    if region == "layer":
        if layer_index is None:
            raise ValueError("layer_index is required for layer waves.")
        layers[layer_index] = [wave]
    return layers


def _apply_coefficients(
    coeffs: np.ndarray,
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
) -> tuple[list[Wave], list[list[Wave]], list[Wave]]:
    cursor = 0
    top = [_scale_wave(w, coeffs[cursor + i]) for i, w in enumerate(top_unknown_basis)]
    cursor += len(top_unknown_basis)

    layers = []
    for layer in layer_unknown_basis:
        layers.append([_scale_wave(w, coeffs[cursor + i]) for i, w in enumerate(layer)])
        cursor += len(layer)

    substrate = [_scale_wave(w, coeffs[cursor + i]) for i, w in enumerate(substrate_unknown_basis)]
    return top, layers, substrate


def _multilayer_residual(
    *,
    top_waves: list[Wave],
    layer_waves: list[list[Wave]],
    substrate_waves: list[Wave],
    thicknesses: list[float],
    mu: float,
) -> np.ndarray:
    residuals = []
    residuals.append(_tangential(top_waves, mu=mu) - _tangential(layer_waves[0], mu=mu))
    for idx in range(len(layer_waves) - 1):
        residuals.append(
            _tangential(_propagate(layer_waves[idx], thicknesses[idx]), mu=mu)
            - _tangential(layer_waves[idx + 1], mu=mu)
        )
    residuals.append(
        _tangential(_propagate(layer_waves[-1], thicknesses[-1]), mu=mu)
        - _tangential(substrate_waves, mu=mu)
    )
    return np.concatenate(residuals)


def _propagate(waves: list[Wave], thickness: float) -> list[Wave]:
    return [replace(w, electric=w.electric * np.exp(1j * w.k[2] * thickness)) for w in waves]


def _scale_wave(wave: Wave, amplitude: complex) -> Wave:
    return replace(wave, electric=wave.electric * amplitude)


def _tangential(waves: list[Wave], *, mu: float) -> np.ndarray:
    if not waves:
        return np.zeros(4, dtype=complex)
    e = np.sum([w.electric for w in waves], axis=0)
    h = np.sum([w.magnetic(mu=mu) for w in waves], axis=0)
    return np.array(tangential_eh(e, h), dtype=complex)  # canonical SHAARP [Ex,Ey,Hx,Hy]


def solve_multilayer_boundary_single_pass(
    *,
    top_known: list[Wave],
    layer_known: list[list[Wave]] | None = None,
    substrate_known: list[Wave] | None = None,
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
    thicknesses: list[float],
    mu: float = MU0,
    writeback: bool = True,
) -> MultilayerBoundarySolution:
    """SHAARP.ml ``f4L[..., iterative -> True]`` single-pass (no multiple-reflection)
    boundary solve. Drop-in for :func:`solve_multilayer_boundary`; used for both the
    omega fundamental and (JK only) the 2 omega SHG stages.

    Instead of the full simultaneous boundary solve it solves interface-by-interface:

      1. entrance (air/film): tangential E/H continuity for the reflected waves and
         the FORWARD film waves only (no backward waves -> single pass). The driving
         term is the incident wave (omega stage) or the KNOWN inhomogeneous source
         waves inside the film (2 omega SHG stage, ``layer_known``);
      2. propagate the forward waves to the film exit;
      3. exit (film/substrate): continuity for the backward film waves and the
         transmitted substrate waves, given the propagated forward (and, at 2 omega,
         propagated inhomogeneous source) waves.

    ``writeback`` selects whether the PROPAGATED forward waves are stored as the
    layer fundamental (SHAARP's ``If[flagHH, wF1[[1]] = wF1tmp]`` -> HH convention,
    True) or the unpropagated forward waves are kept (JK convention, False).

    Validated vs live SHAARP.ml MF to ~1e-15 for both HH (mrassumption 2: omega
    single-pass writeback + full 2 omega) and JK (mrassumption 1: omega single-pass
    no-writeback + 2 omega single-pass), for BOTH a single internal layer AND a
    multilayer stack (the For-loop over interior interfaces), in
    tests/test_jkhh_maker_agreement.py / test_jkhh_multilayer_agreement.py.
    """

    if len(top_unknown_basis) != 2 or len(substrate_unknown_basis) != 2:
        raise ValueError("expected 2 reflected and 2 substrate basis waves")
    n_layers = len(layer_unknown_basis)
    if n_layers < 1 or any(len(layer) != 4 for layer in layer_unknown_basis):
        raise ValueError("each internal layer basis must be ordered [F1, F2, B1, B2]")
    if len(thicknesses) != n_layers:
        raise ValueError("thicknesses must align with the internal layers")

    reflected_s, reflected_p = top_unknown_basis
    transmitted1, transmitted2 = substrate_unknown_basis

    # 2 omega SHG stage carries known inhomogeneous source waves inside each layer
    # (only SHG-active layers are non-empty); the omega stage has none (driven by the
    # incident wave instead).
    layer_inhomogeneous = [
        list(layer_known[i]) if (layer_known and i < len(layer_known) and layer_known[i]) else []
        for i in range(n_layers)
    ]
    is_two_omega = any(len(inh) > 0 for inh in layer_inhomogeneous)

    def tang(wave: Wave) -> np.ndarray:
        return _tangential([wave], mu=mu)

    def tang_list(waves: list[Wave]) -> np.ndarray:
        return _tangential(waves, mu=mu) if waves else np.zeros(4, dtype=complex)

    layer_forward_out: list[list[Wave]] = [[] for _ in range(n_layers)]
    layer_backward_out: list[list[Wave]] = [[] for _ in range(n_layers)]

    # --- entrance (air / layer 1): reflected + forward only, no backward (single pass) ---
    f1, f2, _b1, _b2 = layer_unknown_basis[0]
    entrance_matrix = np.column_stack([tang(reflected_s), tang(reflected_p), -tang(f1), -tang(f2)])
    if is_two_omega:
        entrance_rhs = tang_list(layer_inhomogeneous[0])  # no incident wave; layer source drives it
    else:
        entrance_rhs = -_tangential(top_known, mu=mu)  # tang(incident) + a.R - b.F = 0
    a_rs, a_rp, b_f1, b_f2 = np.linalg.solve(entrance_matrix, entrance_rhs)
    reflected = [_scale_wave(reflected_s, a_rs), _scale_wave(reflected_p, a_rp)]
    forward_scaled = [_scale_wave(f1, b_f1), _scale_wave(f2, b_f2)]
    forward_propagated = _propagate(forward_scaled, thicknesses[0])
    layer_forward_out[0] = forward_propagated if writeback else forward_scaled
    prev_forward_propagated = forward_propagated
    prev_inhom_propagated = _propagate(layer_inhomogeneous[0], thicknesses[0]) if is_two_omega else []

    # --- interior interfaces (layer i / layer i+1): solve backward[i] + forward[i+1] ---
    for i in range(1, n_layers):
        _fi1, _fi2, bi1, bi2 = layer_unknown_basis[i - 1]
        fn1, fn2, _bn1, _bn2 = layer_unknown_basis[i]
        bi1_exit, bi2_exit = _propagate([bi1, bi2], thicknesses[i - 1])
        interface_matrix = np.column_stack([tang(bi1_exit), tang(bi2_exit), -tang(fn1), -tang(fn2)])
        interface_rhs = -_tangential(prev_forward_propagated, mu=mu)
        if is_two_omega:
            interface_rhs = interface_rhs + tang_list(layer_inhomogeneous[i]) - tang_list(prev_inhom_propagated)
        c_b1, c_b2, d_f1, d_f2 = np.linalg.solve(interface_matrix, interface_rhs)
        layer_backward_out[i - 1] = [_scale_wave(bi1, c_b1), _scale_wave(bi2, c_b2)]
        forward_scaled = [_scale_wave(fn1, d_f1), _scale_wave(fn2, d_f2)]
        forward_propagated = _propagate(forward_scaled, thicknesses[i])
        layer_forward_out[i] = forward_propagated if writeback else forward_scaled
        prev_forward_propagated = forward_propagated
        prev_inhom_propagated = _propagate(layer_inhomogeneous[i], thicknesses[i]) if is_two_omega else []

    # --- exit (last layer / substrate): solve backward[last] + transmitted ---
    _fm1, _fm2, bm1, bm2 = layer_unknown_basis[-1]
    bm1_exit, bm2_exit = _propagate([bm1, bm2], thicknesses[-1])
    exit_matrix = np.column_stack([tang(bm1_exit), tang(bm2_exit), -tang(transmitted1), -tang(transmitted2)])
    exit_rhs = -_tangential(prev_forward_propagated, mu=mu)
    if is_two_omega:
        exit_rhs = exit_rhs - tang_list(prev_inhom_propagated)
    c_bm1, c_bm2, d_t1, d_t2 = np.linalg.solve(exit_matrix, exit_rhs)
    layer_backward_out[-1] = [_scale_wave(bm1, c_bm1), _scale_wave(bm2, c_bm2)]
    transmitted = [_scale_wave(transmitted1, d_t1), _scale_wave(transmitted2, d_t2)]

    layer_unknowns = [
        [layer_forward_out[i][0], layer_forward_out[i][1], layer_backward_out[i][0], layer_backward_out[i][1]]
        for i in range(n_layers)
    ]
    return MultilayerBoundarySolution(
        top_unknown=reflected,
        layer_unknowns=layer_unknowns,
        substrate_unknown=transmitted,
        coefficients=np.zeros(4 * (n_layers + 1), dtype=complex),
        residual=np.zeros(4, dtype=complex),
    )
