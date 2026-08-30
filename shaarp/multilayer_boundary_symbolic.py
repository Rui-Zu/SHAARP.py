"""Symbolic layer-thickness (h) solve for the multilayer linear boundary problem.

The eigenmodes -- the per-layer k_z values and field polarizations -- depend ONLY
on the dielectric permittivity tensor and the in-plane wavevector, NOT on the
layer thickness. The thickness ``h`` enters the multilayer boundary problem ONLY
through the propagation phase ``exp(i k_z h)`` (see ``_propagate`` in
``multilayer_boundary``). So with the (numeric) eigenmode basis waves fixed, the
boundary linear system can be solved with each layer thickness kept as a sympy
symbol, yielding the reflected / transmitted amplitudes as closed-form functions
of ``h``.

Substituting a numeric thickness into the symbolic result reproduces
``solve_multilayer_boundary`` exactly (to machine precision), so this is the
SHAARP.ml "symbolic-h" partial-analytical capability for the linear (omega)
boundary stage -- valid for isotropic AND anisotropic layers (the eigenmode
tangential vectors are numeric either way) and for any number of internal layers
(each with its own thickness symbol).

MIXED SYMBOLIC / NUMERIC THICKNESSES. The original marks layers
symbolic ONE AT A TIME -- ``SHAARP.ml.nb:5135-5146`` stores a distinct symbol
``h<i>`` in the selected layer's material association, and ``setup.nb:11011-11017``
gathers ``thickness = Key[h] /@ mAll[[2;;-2]]`` as a plain LIST consumed
element-wise, so a stack may freely mix numeric and symbolic entries. This module
follows that contract exactly: an entry is either a number or a symbol and nothing
downstream branches on which -- see :func:`_phase_factor`.

ASSEMBLY vs SOLVE. :class:`SymbolicBoundarySystem` holds the assembled matrix
so that MANY right-hand sides -- every nonlinear source of every layer -- share ONE
LU decomposition. The 2-omega boundary matrix does not depend on the SHG tensor at
all, so the d-linearity decomposition belongs inside a single batched solve rather
than around repeated decompositions of the same matrix.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass

import sympy as sp

from .multilayer_boundary import _tangential
from .waves import MU0, Wave


def _csym(value) -> sp.Expr:
    z = complex(value)
    return sp.Float(z.real) + sp.I * sp.Float(z.imag)


def _tangential_symbolic(wave: Wave, mu: float) -> list[sp.Expr]:
    """Numeric tangential [Ex, Ey, Hx, Hy] of a single wave, as sympy constants."""
    return [_csym(x) for x in _tangential([wave], mu=mu)]


def _is_symbolic_thickness(entry) -> bool:
    """True when a thickness entry carries free symbols (the layer is "analytical h").

    Anything else -- float, complex, ``sp.Float`` -- is a plain number, i.e. a layer the
    user left un-flagged, which the original substitutes directly."""
    return isinstance(entry, sp.Basic) and bool(entry.free_symbols)


def _phase_factor(kz, entry) -> sp.Expr:
    """``exp(i k_z h)`` for one layer.

    Symbolic entry -> a symbol-carrying ``exp`` node. Numeric entry -> the collapsed
    complex constant, so the elimination sees a LEAF rather than an unevaluated ``exp``
    tree (smaller expressions, identical value to ~1e-16)."""
    if _is_symbolic_thickness(entry):
        return sp.exp(sp.I * _csym(kz) * entry)
    return _csym(cmath.exp(1j * complex(kz) * complex(entry)))


@dataclass(frozen=True)
class SymbolicThicknessBoundarySolution:
    """Closed-form (in the thickness symbols) boundary-unknown amplitudes."""

    coefficients: list[sp.Expr]
    thickness_symbols: list  # one ENTRY per internal layer: sp.Symbol or a number
    unknown_count: int

    @property
    def symbolic_symbols(self) -> list:
        """Only the entries that are actually symbolic, in layer order."""
        return [e for e in self.thickness_symbols if _is_symbolic_thickness(e)]

    def at(self, *thicknesses: float) -> list[complex]:
        """Evaluate every coefficient at numeric value(s) for the SYMBOLIC thicknesses.

        One value per symbolic entry, in layer order; numeric entries were already
        substituted at assembly time (mixed-list contract)."""
        symbols = self.symbolic_symbols
        if len(thicknesses) != len(symbols):
            raise ValueError(
                f"expected {len(symbols)} thickness value(s) for the symbolic layer(s) "
                f"{[str(s) for s in symbols]}, got {len(thicknesses)}.")
        subs = {sym: float(t) for sym, t in zip(symbols, thicknesses)}
        return [complex(expr.subs(subs).evalf()) for expr in self.coefficients]


@dataclass(frozen=True)
class SymbolicBoundarySystem:
    """An assembled multilayer boundary matrix with symbolic/numeric layer thicknesses.

    Separating assembly from the solve is what makes the N-layer 2-omega SHG tractable:
    every nonlinear source of every layer produces one right-hand-side COLUMN, and all
    of them are solved together through a single LU decomposition (meth:`solve`)."""

    matrix: sp.Matrix
    columns: list
    layer_col_ranges: list
    substrate_col_start: int
    thickness_entries: list
    mu: float

    @property
    def n_layers(self) -> int:
        return len(self.layer_col_ranges)

    @property
    def n_rows(self) -> int:
        return self.matrix.rows

    @property
    def symbolic_indices(self) -> list[int]:
        return [i for i, e in enumerate(self.thickness_entries) if _is_symbolic_thickness(e)]

    @property
    def symbolic_symbols(self) -> list:
        return [self.thickness_entries[i] for i in self.symbolic_indices]

    def layer_columns(self, index: int) -> tuple[int, int]:
        return self.layer_col_ranges[index]

    def rhs(self, *, top_known=(), layer_known=None, substrate_known=None) -> sp.Matrix:
        """One right-hand-side column for the given known (source) waves.

        Same region/propagation roles as the unknown columns: a known internal-layer wave
        enters ``+tang`` at the interface above the layer and ``-tang*exp(i k_z h)`` at the
        interface below it."""
        rhs = sp.zeros(self.n_rows, 1)
        for wave in top_known or ():
            tang = _tangential_symbolic(wave, self.mu)
            for r in range(4):
                rhs[r] -= tang[r]
        if layer_known is not None:
            if len(layer_known) != self.n_layers:
                raise ValueError("layer_known must match the number of internal layers.")
            for layer_index, known_waves in enumerate(layer_known):
                entry = self.thickness_entries[layer_index]
                for wave in known_waves or ():
                    tang = _tangential_symbolic(wave, self.mu)
                    prop = _phase_factor(wave.k[2], entry)
                    for r in range(4):
                        rhs[4 * layer_index + r] += tang[r]
                        rhs[4 * (layer_index + 1) + r] -= tang[r] * prop
        if substrate_known is not None:
            for wave in substrate_known:
                tang = _tangential_symbolic(wave, self.mu)
                for r in range(4):
                    rhs[4 * self.n_layers + r] += tang[r]
        return rhs

    def solve(self, rhs: sp.Matrix) -> sp.Matrix:
        """Solve for one or MANY right-hand-side columns with a single decomposition.

        ``sp.Matrix.LUsolve`` decomposes the (source-independent) matrix once and then
        substitutes each column, so batching N sources costs one decomposition instead of
        N -- the single biggest cost lever at multiple layers."""
        if not self.matrix.free_symbols and not rhs.free_symbols:
            # when EVERY thickness is numeric the boundary problem is a
            # plain complex linear system -- solve it in numpy. sympy has no native complex
            # Float, so its LU over (a + b*I) entries expands each product into four terms
            # that never collapse: GaAs(111) Partial Analytical took 105 s with a numeric
            # 1 um film vs 14.5 s with a symbolic h for the SAME closed form (the "numeric thickness costs 3-6x more" regression; it also blew the gate's module
            # budget in tests.test_si_normal_incidence). The symbolic-h path is untouched.
            import numpy as _np
            a = _np.array(self.matrix.evalf().tolist(), dtype=complex)
            b = _np.array(rhs.evalf().tolist(), dtype=complex)
            x = _np.linalg.solve(a, b)
            return sp.Matrix(x.shape[0], x.shape[1], lambda i, j: _csym(x[i, j]))
        return self.matrix.LUsolve(rhs)


def build_multilayer_boundary_symbolic_system(
    *,
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
    thickness_symbols: list,
    mu: float = MU0,
) -> SymbolicBoundarySystem:
    """Assemble (without solving) the multilayer boundary matrix.

    ``thickness_symbols`` is one entry per internal layer, each either a sympy Symbol
    (that layer's thickness stays symbolic) or a number (substituted at assembly)."""
    n_layers = len(layer_unknown_basis)
    if len(thickness_symbols) != n_layers:
        raise ValueError("thickness_symbols must match the number of internal layers.")

    columns: list[Wave] = list(top_unknown_basis)
    layer_col_ranges: list[tuple[int, int]] = []
    for layer in layer_unknown_basis:
        start = len(columns)
        columns.extend(layer)
        layer_col_ranges.append((start, len(columns)))
    sub_start = len(columns)
    columns.extend(substrate_unknown_basis)

    n_unknowns = len(columns)
    n_interfaces = n_layers + 1
    n_rows = 4 * n_interfaces
    if n_unknowns != n_rows:
        raise ValueError(f"Expected {n_rows} unknown basis waves for {n_layers} layers, got {n_unknowns}.")

    matrix = sp.zeros(n_rows, n_unknowns)

    # Top unknown basis (reflected): region above the top interface (interface 0), +.
    for col in range(len(top_unknown_basis)):
        tang = _tangential_symbolic(columns[col], mu)
        for r in range(4):
            matrix[r, col] += tang[r]

    # Each internal layer appears at the interface ABOVE it (region below that
    # interface -> sign -, un-propagated) and the interface BELOW it (region above
    # that interface -> sign +, propagated by exp(i k_z h)).
    for layer_index, (cstart, cend) in enumerate(layer_col_ranges):
        top_interface = layer_index
        bottom_interface = layer_index + 1
        entry = thickness_symbols[layer_index]
        for col in range(cstart, cend):
            wave = columns[col]
            tang = _tangential_symbolic(wave, mu)
            prop = _phase_factor(wave.k[2], entry)
            for r in range(4):
                matrix[4 * top_interface + r, col] -= tang[r]
                matrix[4 * bottom_interface + r, col] += tang[r] * prop

    # Substrate: region below the bottom interface, sign -.
    bottom_interface = n_layers
    for col in range(sub_start, n_unknowns):
        tang = _tangential_symbolic(columns[col], mu)
        for r in range(4):
            matrix[4 * bottom_interface + r, col] -= tang[r]

    return SymbolicBoundarySystem(
        matrix=matrix,
        columns=columns,
        layer_col_ranges=layer_col_ranges,
        substrate_col_start=sub_start,
        thickness_entries=list(thickness_symbols),
        mu=mu,
    )


def solve_multilayer_boundary_symbolic_thickness(
    *,
    top_known: list[Wave],
    top_unknown_basis: list[Wave],
    layer_unknown_basis: list[list[Wave]],
    substrate_unknown_basis: list[Wave],
    thickness_symbols: list,
    layer_known: list[list[Wave]] | None = None,
    substrate_known: list[Wave] | None = None,
    mu: float = MU0,
) -> SymbolicThicknessBoundarySolution:
    """Solve the multilayer linear boundary system with symbolic layer thicknesses.

    Same interface/region convention as :func:`solve_multilayer_boundary`:
    each interface equation is ``tang(region_above) - tang(region_below) = 0``,
    where an internal layer's waves are propagated by ``exp(i k_z h)`` only at the
    interface *below* the layer.

    ``layer_known`` / ``substrate_known`` are fixed (e.g. inhomogeneous 2-omega SHG
    source) waves -- they move to the RHS with the same region/propagation roles.
    This is the boundary-solve half of a symbolic-h 2-omega SHG solve.

    (a thin wrapper over :func:`build_multilayer_boundary_symbolic_system` +
    :meth:`SymbolicBoundarySystem.rhs` + :meth:`SymbolicBoundarySystem.solve`, kept so
    every existing caller and fence is untouched.)"""
    if layer_known is not None and len(layer_known) != len(layer_unknown_basis):
        raise ValueError("layer_known must match the number of internal layers.")
    system = build_multilayer_boundary_symbolic_system(
        top_unknown_basis=top_unknown_basis,
        layer_unknown_basis=layer_unknown_basis,
        substrate_unknown_basis=substrate_unknown_basis,
        thickness_symbols=thickness_symbols,
        mu=mu,
    )
    rhs = system.rhs(top_known=top_known, layer_known=layer_known, substrate_known=substrate_known)
    coefficients = list(system.solve(rhs))
    return SymbolicThicknessBoundarySolution(
        coefficients=coefficients,
        thickness_symbols=list(thickness_symbols),
        unknown_count=system.matrix.cols,
    )


__all__ = [
    "SymbolicBoundarySystem",
    "SymbolicThicknessBoundarySolution",
    "build_multilayer_boundary_symbolic_system",
    "solve_multilayer_boundary_symbolic_thickness",
]
