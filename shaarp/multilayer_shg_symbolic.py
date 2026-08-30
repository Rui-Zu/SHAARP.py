"""Symbolic layer-thickness (h) SHG: the reflected/transmitted second-harmonic
amplitudes of a nonlinear multilayer as CLOSED-FORM functions of the layer thicknesses
(the Maker fringe as an explicit symbolic function of h). This is the SHAARP.ml
"partial analytical" thickness dependence extended from the linear omega boundary
through the full 2 omega SHG stage.

WHY IT FACTORS (the key that makes it tractable): eigenmodes do not depend on h; h enters
only through the exp(i k_z h) propagation phase. The 2 omega boundary solve is LINEAR in
the nonlinear sources, and each layer source amplitude FACTORS as A_ij(h) = c_i(h) c_j(h),
the product of the (symbolic-in-h) omega-boundary layer-mode amplitudes. So, summing over
the interior layers ell:

    SHG(h1..hN) = sum_ell sum_ij A^ell_ij(h) * [ 2 omega boundary solved with the UNIT
                                                  (numeric) bound field of layer ell's
                                                  source ij ]

Substituting numeric thicknesses reproduces the numeric
``solve_multilayer_shg_from_fundamental`` to machine precision, so this is the
closed-form-in-h form of the validated numeric Maker-fringe SHG.

N LAYERS, MIXED SYMBOLIC/NUMERIC (parity with the original).
``SHAARP.ml.nb:5135-5146`` gives every INTERIOR layer its own "analytical h" button which
stores a distinct symbol h1, h2, ... in that layer's material association;
``setup.nb:11011-11017`` then gathers ``thickness = Key[h] /@ mAll[[2;;-2]]`` as a plain
LIST consumed element-wise, so any SUBSET of layers may be symbolic while the rest stay
numeric. Nothing in the original branches on which -- and nothing here does either. The
half-spaces are structurally excluded from carrying a source in the original too
(``setup.nb:10370-10374`` loops ``i <= nM`` over ``mAll[[i+1]]``), which is why only
interior layers appear below.

COST (why this is one batched solve, not a loop of solves): the 2 omega boundary matrix
does not depend on the SHG tensor at all, so every source of every layer becomes one
right-hand-side COLUMN of a single LU decomposition
(meth:`SymbolicBoundarySystem.solve`). The previous single-film code decomposed the same
matrix once per source; at N layers that would have been 10 * sum_ell nnz(d_ell)
decompositions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from .multilayer_basis import HomogeneousMultilayerBasis
from .multilayer_boundary_symbolic import (
    _is_symbolic_thickness,
    build_multilayer_boundary_symbolic_system,
    solve_multilayer_boundary_symbolic_thickness,
)
from .multilayer_shg_boundary import _field_to_wave, _fields_for_source_policy
from .nonlinear import multilayer_sources, solve_multilayer_inhomogeneous_fields
from .multilayer_shg_boundary import _warn_if_units_break_dispersion
from .waves import EPS0, MU0

# Layer source amplitude A_ij = c_i * c_j over the omega layer modes [F1,F2,B1,B2]
# (indices 0..3), in the SHAARP.ml source order f1f1,f2f2,f1f2,b1b1,b2b2,b1b2,f1b1,f2b2,f1b2,f2b1.
_SOURCE_AMPLITUDE_INDICES = [(0, 0), (1, 1), (0, 1), (2, 2), (3, 3), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]


class SymbolicThicknessBudgetExceeded(RuntimeError):
    """The symbolic solve was refused (or abandoned) because it exceeded its budget.

    Raised with a message that states the reason AND the way forward, so a caller can
    surface it as a result rather than a crash."""


@dataclass(frozen=True)
class SymbolicThicknessBudget:
    """Bounds on a symbolic-thickness solve.

    HONESTY NOTE: the checks below run at STAGE BOUNDARIES (before assembly, after each
    omega solve, after the batched 2 omega solve). A single sympy ``LUdecomposition`` is
    ATOMIC -- it cannot be interrupted in-process -- so the real protection is the
    pre-flight cap, and a stack that passes pre-flight is committed to at least one
    decomposition. (Same class as the in-process hang limit recorded in
    docs/residual_risks.md R13; a true hang is caught by driving the case in its own subprocess.)"""

    max_symbolic_layers: int = 3
    max_total_layers: int = 8  # the original's maxmaterialnumber - 2
    seconds: float | None = None
    max_nodes: int | None = None

    def preflight(self, n_layers: int, n_symbolic: int) -> None:
        if n_layers > self.max_total_layers:
            raise SymbolicThicknessBudgetExceeded(
                f"{n_layers} interior layers exceeds the supported maximum of "
                f"{self.max_total_layers} (the original SHAARP.ml caps at "
                f"maxmaterialnumber - 2 = 8). Reduce the stack, or use the numeric modes.")
        if n_symbolic > self.max_symbolic_layers:
            raise SymbolicThicknessBudgetExceeded(
                f"{n_symbolic} layers are marked 'analytical h' (limit "
                f"{self.max_symbolic_layers}): the closed-form solve grows steeply with each "
                f"added symbolic thickness and would exceed the interactive budget. Un-mark "
                f"layers -- their thickness is then substituted, exactly as the original does "
                f"for un-flagged layers -- or use SHG Simulation / Maker Fringes for the "
                f"numeric answer.")

    def check(self, stage: str, started: float, expr=None) -> None:
        if self.seconds is not None and (time.monotonic() - started) > self.seconds:
            raise SymbolicThicknessBudgetExceeded(
                f"symbolic solve exceeded {self.seconds:g} s at stage '{stage}'. "
                f"Un-mark some layers or use the numeric modes.")
        if self.max_nodes is not None and expr is not None:
            n = sum(sp.count_ops(e) for e in expr) if isinstance(expr, (list, tuple)) else sp.count_ops(expr)
            if n > self.max_nodes:
                raise SymbolicThicknessBudgetExceeded(
                    f"symbolic expression reached {int(n)} operations at stage '{stage}' "
                    f"(limit {self.max_nodes}). Un-mark some layers or use the numeric modes.")


@dataclass(frozen=True)
class SymbolicThicknessSHGSolution:
    """Reflected/transmitted SHG amplitudes as closed forms in the layer thicknesses.

    ``coefficients`` is the boundary-unknown vector
    ``[r_s, r_p, (B1,B2 per interior layer)..., sub_s, sub_p]``."""

    coefficients: list  # SHG amplitudes (functions of the symbolic thicknesses)
    thickness_symbols: list  # one ENTRY per interior layer: sp.Symbol or a number
    _eval_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def thickness_symbol(self):
        """Back-compat single-film accessor."""
        if len(self.thickness_symbols) != 1:
            raise ValueError(
                "this solution has %d layers; use .thickness_symbols" % len(self.thickness_symbols))
        return self.thickness_symbols[0]

    @property
    def n_layers(self) -> int:
        return len(self.thickness_symbols)

    @property
    def symbolic_symbols(self) -> list:
        return [e for e in self.thickness_symbols if _is_symbolic_thickness(e)]

    @property
    def reflected_s(self):
        return self.coefficients[0]

    @property
    def reflected_p(self):
        return self.coefficients[1]

    @property
    def substrate_s(self):
        """Transmitted-into-substrate SHG s-amplitude (the Maker-fringe transmitted signal).

        indexed from the END. The layout is [r_s, r_p] + 4 per interior layer +
        [sub_s, sub_p], so the old fixed [6]/[7] was the substrate only for a SINGLE film --
        at 2+ layers it silently returned layer 2's F1/F2 amplitudes."""
        return self.coefficients[-2]

    @property
    def substrate_p(self):
        """Transmitted-into-substrate SHG p-amplitude (see :attr:`substrate_s`)."""
        return self.coefficients[-1]

    def layer_amplitudes(self, index: int) -> list:
        """The 4 SHG amplitudes [F1,F2,B1,B2] inside interior layer ``index`` (0-based)."""
        start = 2 + 4 * index
        return self.coefficients[start:start + 4]

    def at(self, *thicknesses: float) -> list[complex]:
        """Evaluate every coefficient: one value per SYMBOLIC layer, in layer order.

        COMPILED, not substituted (measurement): at two layers the closed form is a
        ~10^5-node tree, where ``expr.subs(...).evalf()`` costs ~107 s per point while the
        SOLVE itself is ~6 s. ``lambdify`` (with common-subexpression elimination) compiles
        the whole coefficient vector ONCE and evaluates it in microseconds, which is what
        makes a multilayer closed form usable in a sweep at all. The compiled function is
        cached per solution; a fallback to ``subs`` keeps exotic expressions working."""
        symbols = self.symbolic_symbols
        if len(thicknesses) != len(symbols):
            raise ValueError(
                f"expected {len(symbols)} thickness value(s) for the symbolic layer(s) "
                f"{[str(s) for s in symbols]}, got {len(thicknesses)}.")
        fn = self._eval_cache.get("fn")
        if fn is None and "failed" not in self._eval_cache:
            try:
                fn = sp.lambdify(symbols, self.coefficients, modules="numpy", cse=True)
                self._eval_cache["fn"] = fn
            except Exception:  # pragma: no cover - exotic expression fallback
                self._eval_cache["failed"] = True
        if fn is not None:
            return [complex(v) for v in fn(*[complex(t) for t in thicknesses])]
        subs = {sym: complex(t) for sym, t in zip(symbols, thicknesses)}
        return [complex(expr.subs(subs).evalf()) for expr in self.coefficients]


def _layer_bound_waves(d_voigt_lab, layer_basis, eps_2omega_lab, policy, mu, eps0):
    """The numeric bound (inhomogeneous) 2-omega waves of ONE interior layer, in the
    ``_SOURCE_AMPLITUDE_INDICES`` order."""
    f1, f2, b1, b2 = layer_basis
    sources = multilayer_sources(d_voigt_lab, f1, f2, b1, b2)
    fields = solve_multilayer_inhomogeneous_fields(eps_2omega_lab, sources, mu=mu, eps0=eps0)
    return [_field_to_wave(field) for field in _fields_for_source_policy(fields, policy)]


def _omega_layer_amplitudes_all(omega_basis: HomogeneousMultilayerBasis, thickness_entries, mu):
    """The symbolic-h omega layer-mode amplitudes [F1,F2,B1,B2] for EVERY interior layer.

    One omega solve regardless of the layer count -- which is why polarimetry stays at two
    omega solves (s and p) at any N."""
    sol = solve_multilayer_boundary_symbolic_thickness(
        top_known=omega_basis.incident,
        top_unknown_basis=omega_basis.reflected_basis,
        layer_unknown_basis=omega_basis.layer_basis,
        substrate_unknown_basis=omega_basis.substrate_basis,
        thickness_symbols=list(thickness_entries),
        mu=mu,
    )
    n_layers = len(omega_basis.layer_basis)
    return [sol.coefficients[2 + 4 * i:6 + 4 * i] for i in range(n_layers)]


def _normalize_layer_d(layer_d, n_layers, name):
    if len(layer_d) != n_layers:
        raise ValueError(f"{name} must have one entry per interior layer ({n_layers}).")
    out = []
    for d in layer_d:
        mat = sp.Matrix(d)
        if mat.shape != (3, 6):
            raise ValueError(f"each {name} entry must be a 3x6 Voigt matrix.")
        out.append(mat)
    return out


def solve_multilayer_shg_symbolic_thickness(
    *,
    omega_basis: HomogeneousMultilayerBasis,
    twoomega_basis: HomogeneousMultilayerBasis,
    layer_d_voigt_lab,
    layer_epsilon_2omega_lab,
    thickness_symbols,
    layer_amplitudes=None,
    inhomogeneous_source_policy: str = "all",
    mu: float = 1.0,   # natural units -- the locked multilayer convention
    eps0: float = 1.0,
    budget: SymbolicThicknessBudget | None = None,
) -> SymbolicThicknessSHGSolution:
    """Closed-form-in-thickness reflected/transmitted SHG for an N-layer stack.

    ``layer_d_voigt_lab`` / ``layer_epsilon_2omega_lab`` are per-interior-layer lists (a
    layer with an all-zero d is SHG-inactive and contributes nothing -- the original runs
    ``solveInhom`` on it unconditionally and gets a zero source, so skipping is equal by
    construction and is fenced as such). ``thickness_symbols`` is one entry per interior
    layer, each a sympy Symbol (that layer stays symbolic) or a number (substituted).

    ``layer_amplitudes`` (optional): a per-layer list of the 4 omega layer-mode amplitudes
    c = [F1,F2,B1,B2] (e.g. Jones-combined for polarimetry), which SKIPS the omega solve."""
    n_layers = len(omega_basis.layer_basis)
    if len(twoomega_basis.layer_basis) != n_layers:
        raise ValueError("omega and 2 omega bases must describe the same number of layers.")
    if len(thickness_symbols) != n_layers:
        raise ValueError("thickness_symbols must have one entry per interior layer.")
    if len(layer_epsilon_2omega_lab) != n_layers:
        raise ValueError("layer_epsilon_2omega_lab must have one entry per interior layer.")
    d_layers = _normalize_layer_d(layer_d_voigt_lab, n_layers, "layer_d_voigt_lab")

    started = time.monotonic()
    n_symbolic = sum(1 for e in thickness_symbols if _is_symbolic_thickness(e))
    if budget is not None:
        budget.preflight(n_layers, n_symbolic)

    # (1) symbolic-h omega boundary -> per-layer mode amplitudes c^ell_i(h)
    if layer_amplitudes is not None:
        if len(layer_amplitudes) != n_layers or any(len(c) != 4 for c in layer_amplitudes):
            raise ValueError("layer_amplitudes must be one length-4 list per interior layer.")
        amps = [list(c) for c in layer_amplitudes]
    else:
        amps = _omega_layer_amplitudes_all(omega_basis, thickness_symbols, mu)
    if budget is not None:
        budget.check("omega", started)

    # (2) assemble the 2 omega system ONCE; every source of every layer is one RHS column
    system = build_multilayer_boundary_symbolic_system(
        top_unknown_basis=twoomega_basis.reflected_basis,
        layer_unknown_basis=twoomega_basis.layer_basis,
        substrate_unknown_basis=twoomega_basis.substrate_basis,
        thickness_symbols=list(thickness_symbols),
        mu=mu,
    )

    columns: list[sp.Matrix] = []
    weights: list = []
    for ell in range(n_layers):
        d_mat = d_layers[ell]
        # NOTE: sympy's Float(0.0) == 0 is FALSE (structural equality), so a numpy
        # float d must be tested with .is_zero -- otherwise every zero entry becomes a
        # source column (18 instead of 3: six times the work, and an all-zero layer
        # would never be detected). A SYMBOLIC entry has is_zero None -> kept.
        positions = [(m, l) for m in range(3) for l in range(6)
                     if not sp.sympify(d_mat[m, l]).is_zero]
        if not positions:
            continue  # SHG-inactive layer: zero source (still carries phase via the matrix)
        for (m, l) in positions:
            unit = np.zeros((3, 6))
            unit[m, l] = 1.0
            bound_waves = _layer_bound_waves(
                unit, omega_basis.layer_basis[ell], layer_epsilon_2omega_lab[ell],
                inhomogeneous_source_policy, mu, eps0)
            if len(bound_waves) > len(_SOURCE_AMPLITUDE_INDICES):
                raise ValueError("more bound waves than known source-amplitude pairings.")
            c = amps[ell]
            for k, bound_wave in enumerate(bound_waves):
                i, j = _SOURCE_AMPLITUDE_INDICES[k]
                one_hot = [[] for _ in range(n_layers)]
                one_hot[ell] = [bound_wave]
                columns.append(system.rhs(layer_known=one_hot))
                weights.append(d_mat[m, l] * c[i] * c[j])

    if not columns:
        raise ValueError("no SHG-active interior layer: every layer_d_voigt_lab entry is zero.")

    # (3) ONE decomposition for every source column, then recombine
    solved = system.solve(sp.Matrix.hstack(*columns))
    if budget is not None:
        budget.check("twoomega", started)
    total = [sp.Add(*[weights[k] * solved[r, k] for k in range(len(weights))])
             for r in range(solved.rows)]
    if budget is not None:
        budget.check("recombine", started, total)

    return SymbolicThicknessSHGSolution(coefficients=total, thickness_symbols=list(thickness_symbols))


def solve_multilayer_shg_symbolic_thickness_and_d(
    *,
    omega_basis: HomogeneousMultilayerBasis,
    twoomega_basis: HomogeneousMultilayerBasis,
    layer_d_voigt_symbolic,
    layer_epsilon_2omega_lab,
    thickness_symbols,
    mu: float = 1.0,   # natural units -- the locked multilayer convention
    eps0: float = 1.0,
    budget: SymbolicThicknessBudget | None = None,
) -> SymbolicThicknessSHGSolution:
    """SHG of an N-layer stack as a closed form in BOTH the thicknesses AND the per-layer
    SHG tensors d_il -- the SHAARP.ml "partial analytical" expression in (h, d).

    The SHG is LINEAR in every d component, so a symbolic d entry is simply carried as the
    weight of its unit-d source column (see the module docstring): no extra solves."""
    return solve_multilayer_shg_symbolic_thickness(
        omega_basis=omega_basis, twoomega_basis=twoomega_basis,
        layer_d_voigt_lab=layer_d_voigt_symbolic,
        layer_epsilon_2omega_lab=layer_epsilon_2omega_lab,
        thickness_symbols=thickness_symbols, mu=mu, eps0=eps0, budget=budget,
    )


def solve_multilayer_shg_symbolic_polarimetry(
    *,
    omega_basis_s: HomogeneousMultilayerBasis,
    omega_basis_p: HomogeneousMultilayerBasis,
    twoomega_basis: HomogeneousMultilayerBasis,
    layer_d_voigt_symbolic,
    layer_epsilon_2omega_lab,
    thickness_symbols,
    phi_symbol: sp.Symbol,
    ellipticity=0,
    mu: float = 1.0,   # natural units -- the locked multilayer convention
    eps0: float = 1.0,
    budget: SymbolicThicknessBudget | None = None,
) -> SymbolicThicknessSHGSolution:
    """SHAARP.ml PARTIAL-analytical SHG POLARIMETRY for an N-layer stack: a closed form in
    the input polarization ``phi``, the per-layer d, and the symbolic thicknesses, with the
    eigenmodes / refraction angles / indices kept NUMERIC.

    ``mu``/``eps0`` MUST be natural units (mu*eps0 == 1): the bases are built with c = 1, so the
    inhomogeneous operator ``curl_curl(k) - omega^2 mu eps0 eps`` is dispersion-consistent only
    there. At SI constants its two terms differ by ~1e16 and the operator is numerically singular
    (measured condition 7.7e17, every solve flagging ill_conditioned, Partial
    Analytical breaking the crystal's own point-group symmetry by 1.1e-2). A violation now warns.

    The incident Jones is the SHAARP convention ``(J_s,J_p) = (sin(phi) e^{i*ell}, cos(phi))``;
    the omega boundary is linear in the incident, so each layer's combined amplitudes are
    ``c^ell = J_s c^{s,ell} + J_p c^{p,ell}`` -- still exactly two omega solves at any N."""
    _warn_if_units_break_dispersion(mu, eps0)
    sin_phi = sp.sin(phi_symbol)
    j_s = sin_phi * sp.exp(sp.I * ellipticity) if ellipticity != 0 else sin_phi
    j_p = sp.cos(phi_symbol)
    c_s = _omega_layer_amplitudes_all(omega_basis_s, thickness_symbols, mu)
    c_p = _omega_layer_amplitudes_all(omega_basis_p, thickness_symbols, mu)
    combined = [[j_s * c_s[ell][i] + j_p * c_p[ell][i] for i in range(4)]
                for ell in range(len(c_s))]
    return solve_multilayer_shg_symbolic_thickness(
        omega_basis=omega_basis_s, twoomega_basis=twoomega_basis,
        layer_d_voigt_lab=layer_d_voigt_symbolic,
        layer_epsilon_2omega_lab=layer_epsilon_2omega_lab,
        thickness_symbols=thickness_symbols, layer_amplitudes=combined,
        mu=mu, eps0=eps0, budget=budget,
    )


# --------------------------------------------------------------------------------------
# Back-compat single-film entry points (unchanged signatures; every existing caller and
# fence keeps working, and proves the delegation is exact).
# --------------------------------------------------------------------------------------

def solve_single_film_shg_symbolic_thickness(
    *,
    omega_basis: HomogeneousMultilayerBasis,
    twoomega_basis: HomogeneousMultilayerBasis,
    d_voigt_lab,
    eps_2omega_lab,
    thickness_symbol: sp.Symbol,
    layer_amplitudes=None,
    mu: float = 1.0,   # natural units -- the locked multilayer convention
    eps0: float = 1.0,
) -> SymbolicThicknessSHGSolution:
    """Closed-form-in-thickness SHG for ONE nonlinear film (the N=1 case)."""
    if len(omega_basis.layer_basis) != 1 or len(twoomega_basis.layer_basis) != 1:
        raise ValueError("This symbolic-thickness SHG solve is for a single nonlinear film.")
    if layer_amplitudes is not None:
        if len(layer_amplitudes) != 4:
            raise ValueError("layer_amplitudes must be the 4 layer modes [F1,F2,B1,B2].")
        layer_amplitudes = [list(layer_amplitudes)]
    return solve_multilayer_shg_symbolic_thickness(
        omega_basis=omega_basis, twoomega_basis=twoomega_basis,
        layer_d_voigt_lab=[d_voigt_lab], layer_epsilon_2omega_lab=[eps_2omega_lab],
        thickness_symbols=[thickness_symbol], layer_amplitudes=layer_amplitudes,
        mu=mu, eps0=eps0,
    )


def solve_single_film_shg_symbolic_thickness_and_d(
    *,
    omega_basis: HomogeneousMultilayerBasis,
    twoomega_basis: HomogeneousMultilayerBasis,
    d_voigt_symbolic,
    eps_2omega_lab,
    thickness_symbol: sp.Symbol,
    mu: float = 1.0,   # natural units -- the locked multilayer convention
    eps0: float = 1.0,
) -> SymbolicThicknessSHGSolution:
    """Single-film SHG as a closed form in both h and d."""
    d_mat = sp.Matrix(d_voigt_symbolic)
    if d_mat.shape != (3, 6):
        raise ValueError("d_voigt_symbolic must be a 3x6 Voigt matrix.")
    if all(sp.sympify(d_mat[m, l]).is_zero for m in range(3) for l in range(6)):
        raise ValueError("d_voigt_symbolic has no nonzero entries.")
    return solve_multilayer_shg_symbolic_thickness_and_d(
        omega_basis=omega_basis, twoomega_basis=twoomega_basis,
        layer_d_voigt_symbolic=[d_mat], layer_epsilon_2omega_lab=[eps_2omega_lab],
        thickness_symbols=[thickness_symbol], mu=mu, eps0=eps0,
    )


def solve_single_film_shg_symbolic_polarimetry(
    *,
    omega_basis_s: HomogeneousMultilayerBasis,
    omega_basis_p: HomogeneousMultilayerBasis,
    twoomega_basis: HomogeneousMultilayerBasis,
    d_voigt_symbolic,
    eps_2omega_lab,
    thickness_symbol: sp.Symbol,
    phi_symbol: sp.Symbol,
    ellipticity=0,
    mu: float = 1.0,   # natural units -- the locked multilayer convention
    eps0: float = 1.0,
) -> SymbolicThicknessSHGSolution:
    """Single-film partial-analytical SHG polarimetry in (phi, d, h)."""
    d_mat = sp.Matrix(d_voigt_symbolic)
    if d_mat.shape != (3, 6):
        raise ValueError("d_voigt_symbolic must be a 3x6 Voigt matrix.")
    if all(sp.sympify(d_mat[m, l]).is_zero for m in range(3) for l in range(6)):
        raise ValueError("d_voigt_symbolic has no nonzero entries.")
    return solve_multilayer_shg_symbolic_polarimetry(
        omega_basis_s=omega_basis_s, omega_basis_p=omega_basis_p,
        twoomega_basis=twoomega_basis, layer_d_voigt_symbolic=[d_mat],
        layer_epsilon_2omega_lab=[eps_2omega_lab], thickness_symbols=[thickness_symbol],
        phi_symbol=phi_symbol, ellipticity=ellipticity, mu=mu, eps0=eps0,
    )


__all__ = [
    "SymbolicThicknessBudget",
    "SymbolicThicknessBudgetExceeded",
    "SymbolicThicknessSHGSolution",
    "solve_multilayer_shg_symbolic_thickness",
    "solve_multilayer_shg_symbolic_thickness_and_d",
    "solve_multilayer_shg_symbolic_polarimetry",
    "solve_single_film_shg_symbolic_thickness",
    "solve_single_film_shg_symbolic_thickness_and_d",
    "solve_single_film_shg_symbolic_polarimetry",
]
