from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .waves import tangential_eh


def _sympy():
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "Symbolic SHAARP helpers require SymPy. Install with "
            "`python -m pip install sympy`."
        ) from exc
    return sp


def d_voigt_symbolic(point_group: str = "1", *, prefix: str = "d") -> Any:
    """Return the symbolic 3x6 SHG Voigt tensor used by SHAARP partial output.

    Patterns are ported from the `doldExp` branch in SHAARP.si's "Partial
    Analytical Expressions" path. Barred groups use ASCII aliases such as
    `-43m`, `-42m`, `-6`, and `-6m2`; infinite groups use `inf`, `infm`,
    and `inf2`.
    """

    sp = _sympy()
    pg = _canonical_point_group(point_group)

    def d(row: int, col: int) -> Any:
        return sp.Symbol(f"{prefix}{row}{col}")

    zero = sp.Integer(0)
    if pg == "1":
        return sp.Matrix([[d(i, j) for j in range(1, 7)] for i in range(1, 4)])
    if pg == "2":
        return sp.Matrix([[zero, zero, zero, d(1, 4), zero, d(1, 6)], [d(2, 1), d(2, 2), d(2, 3), zero, d(2, 5), zero], [zero, zero, zero, d(3, 4), zero, d(3, 6)]])
    if pg == "m":
        return sp.Matrix([[d(1, 1), d(1, 2), d(1, 3), zero, d(1, 5), zero], [zero, zero, zero, d(2, 4), zero, d(2, 6)], [d(3, 1), d(3, 2), d(3, 3), zero, d(3, 5), zero]])
    if pg == "mm2":
        return sp.Matrix([[zero, zero, zero, zero, d(1, 5), zero], [zero, zero, zero, d(2, 4), zero, zero], [d(3, 1), d(3, 2), d(3, 3), zero, zero, zero]])
    if pg == "222":
        return sp.Matrix([[zero, zero, zero, d(1, 4), zero, zero], [zero, zero, zero, zero, d(2, 5), zero], [zero, zero, zero, zero, zero, d(3, 6)]])
    if pg == "3":
        return sp.Matrix([[d(1, 1), -d(1, 1), zero, d(1, 4), d(1, 5), -d(2, 2)], [-d(2, 2), d(2, 2), zero, d(1, 5), -d(1, 4), -d(1, 1)], [d(3, 1), d(3, 1), d(3, 3), zero, zero, zero]])
    if pg == "32":
        return sp.Matrix([[d(1, 1), -d(1, 1), zero, d(1, 4), zero, zero], [zero, zero, zero, zero, -d(1, 4), -d(1, 1)], [zero, zero, zero, zero, zero, zero]])
    if pg == "3m":
        return sp.Matrix([[zero, zero, zero, zero, d(1, 5), -d(2, 2)], [-d(2, 2), d(2, 2), zero, d(1, 5), zero, zero], [d(3, 1), d(3, 1), d(3, 3), zero, zero, zero]])
    if pg in {"4", "6", "inf"}:
        return sp.Matrix([[zero, zero, zero, d(1, 4), d(1, 5), zero], [zero, zero, zero, d(1, 5), -d(1, 4), zero], [d(3, 1), d(3, 1), d(3, 3), zero, zero, zero]])
    if pg == "-4":
        return sp.Matrix([[zero, zero, zero, d(1, 4), d(1, 5), zero], [zero, zero, zero, -d(1, 5), d(1, 4), zero], [d(3, 1), -d(3, 1), zero, zero, zero, d(3, 6)]])
    if pg in {"4mm", "6mm", "infm"}:
        return sp.Matrix([[zero, zero, zero, zero, d(1, 5), zero], [zero, zero, zero, d(1, 5), zero, zero], [d(3, 1), d(3, 1), d(3, 3), zero, zero, zero]])
    if pg in {"422", "622", "inf2"}:
        return sp.Matrix([[zero, zero, zero, d(1, 4), zero, zero], [zero, zero, zero, zero, -d(1, 4), zero], [zero, zero, zero, zero, zero, zero]])
    if pg == "-42m":
        return sp.Matrix([[zero, zero, zero, d(1, 4), zero, zero], [zero, zero, zero, zero, d(1, 4), zero], [zero, zero, zero, zero, zero, d(3, 6)]])
    if pg == "-6":
        return sp.Matrix([[d(1, 1), -d(1, 1), zero, zero, zero, -d(2, 2)], [-d(2, 2), d(2, 2), zero, zero, zero, -d(1, 1)], [zero, zero, zero, zero, zero, zero]])
    if pg == "-6m2":
        return sp.Matrix([[zero, zero, zero, zero, zero, -d(2, 2)], [-d(2, 2), d(2, 2), zero, zero, zero, zero], [zero, zero, zero, zero, zero, zero]])
    if pg in {"-43m", "23"}:
        return sp.Matrix([[zero, zero, zero, d(1, 4), zero, zero], [zero, zero, zero, zero, d(1, 4), zero], [zero, zero, zero, zero, zero, d(1, 4)]])
    # the original's "Centrosymmetric ->" popup (SHAARP.ml.nb:5630) carries an
    # ALL-ZERO 3x6 pattern for each of its 16 groups (the 11 centrosymmetric classes, 432 whose
    # Kleinman-symmetric d vanishes, and the Curie groups inf/m, inf/mm, infinf, infinfm) and runs
    # the pipeline with P_NL = 0. Returning that pattern -- instead of raising -- is what lets the
    # GUI offer those groups and lets "SHG activity" be decided by the point group alone.
    from .point_groups import canonical_point_group, is_known_point_group, is_shg_active
    if is_known_point_group(point_group) and not is_shg_active(point_group):
        return sp.zeros(3, 6)
    raise ValueError(f"Unsupported point group for symbolic SHAARP tensor: {point_group!r}")


def compute_pnl_voigt_symbolic(
    d_voigt: Any,
    e1: Any,
    e2: Any | None = None,
    *,
    factor: Any = 1,
    eps0: Any = 1,
    simplify: bool = True,
) -> Any:
    """Symbolic version of SHAARP `computePNL`."""

    sp = _sympy()
    d_mat = sp.Matrix(d_voigt)
    a = sp.Matrix(e1)
    b = sp.Matrix(e1 if e2 is None else e2)
    if d_mat.shape != (3, 6):
        raise ValueError("d_voigt must be a 3x6 matrix.")
    if a.shape != (3, 1) or b.shape != (3, 1):
        raise ValueError("e1 and e2 must be length-3 vectors.")

    products = sp.Matrix([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[1] * b[2] + a[2] * b[1], a[0] * b[2] + a[2] * b[0], a[1] * b[0] + a[0] * b[1]])
    out = eps0 * factor * (d_mat * products)
    if simplify:
        out = out.applyfunc(sp.simplify)
    return out


def partial_pnl_expressions(point_group: str, e_fast: Any, e_slow: Any, *, prefix: str = "d", eps0: Any = 1) -> dict[str, Any]:
    """Return symbolic `ee`, `oo`, and `eo` nonlinear source polarizations."""

    d_voigt = d_voigt_symbolic(point_group, prefix=prefix)
    return {
        "ee": compute_pnl_voigt_symbolic(d_voigt, e_fast, eps0=eps0),
        "oo": compute_pnl_voigt_symbolic(d_voigt, e_slow, eps0=eps0),
        "eo": compute_pnl_voigt_symbolic(d_voigt, e_fast, e_slow, factor=2, eps0=eps0),
    }


def rotate_d_voigt_symbolic(d_voigt: Any, rotation_crystal_to_lab: Any, *, simplify: bool = True) -> Any:
    """Rotate a symbolic 3x6 SHG Voigt tensor using SHAARP's `dnewExp` loop."""

    sp = _sympy()
    d_mat = sp.Matrix(d_voigt)
    a = sp.Matrix(rotation_crystal_to_lab)
    if d_mat.shape != (3, 6):
        raise ValueError("d_voigt must be a 3x6 matrix.")
    if a.shape != (3, 3):
        raise ValueError("rotation_crystal_to_lab must be a 3x3 matrix.")

    voigt = [
        [0, 5, 4],
        [5, 1, 3],
        [4, 3, 2],
    ]
    out = sp.zeros(3, 6)
    for i in range(3):
        for j in range(3):
            for k in range(j, 3):
                col = voigt[j][k]
                total = sp.Integer(0)
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            total += a[i, l] * a[j, m] * a[k, n] * d_mat[l, voigt[m][n]]
                out[i, col] = sp.simplify(total) if simplify else total
    return out


@dataclass(frozen=True)
class DTensorStructure:
    """Symmetry structure of a point group's Voigt SHG ``d`` tensor -- the headless
    data behind the GUI's "analytical d_ij" display (gui_shg_tensor_inputs_and_analytical_dij).

    ``independent_constants`` are the free d_ij symbols a crystal class allows;
    ``nonzero_entries`` lists every non-vanishing Voigt position as ``(label, value)``
    where ``value`` is the symbolic entry in terms of the independent constants
    (e.g. for 3m, ``("d16", "-d22")`` and ``("d24", "d15")``)."""

    point_group: str
    independent_constants: list[str]
    independent_count: int
    nonzero_entries: list[tuple[str, str]]


def d_tensor_structure(point_group: str = "1", *, prefix: str = "d") -> DTensorStructure:
    """Summarize the symmetry structure of a point group's Voigt SHG ``d`` tensor.

    Derived from :func:`d_voigt_symbolic`: the independent constants (free symbols),
    and every nonzero Voigt entry with its value in terms of those constants. Pure
    symbolic -- no numerics, no display."""
    d = d_voigt_symbolic(point_group, prefix=prefix)
    constants = sorted((str(s) for s in d.free_symbols), key=str)
    nonzero: list[tuple[str, str]] = []
    for row in range(3):
        for col in range(6):
            entry = d[row, col]
            if entry != 0:
                nonzero.append((f"{prefix}{row + 1}{col + 1}", str(entry)))
    return DTensorStructure(
        point_group=str(point_group),
        independent_constants=constants,
        independent_count=len(constants),
        nonzero_entries=nonzero,
    )


@dataclass(frozen=True)
class SymbolicWave:
    """SymPy plane-wave fields used by symbolic boundary solvers."""

    k: Any
    electric: Any
    omega: Any = 1

    def magnetic(self, mu: Any = 1) -> Any:
        sp = _sympy()
        return sp.Matrix(self.k).cross(sp.Matrix(self.electric)) / (self.omega * mu)


@dataclass(frozen=True)
class SymbolicLinearInterfaceResult:
    """Symbolic linear interface amplitudes for isotropic media."""

    incident: SymbolicWave
    reflected_s: SymbolicWave
    reflected_p: SymbolicWave
    transmitted_s: SymbolicWave
    transmitted_p: SymbolicWave
    coefficients: Any
    theta_transmitted: Any

    @property
    def r_s(self) -> Any:
        return self.coefficients[0]

    @property
    def r_p(self) -> Any:
        return self.coefficients[1]

    @property
    def t_s(self) -> Any:
        return self.coefficients[2]

    @property
    def t_p(self) -> Any:
        return self.coefficients[3]


@dataclass(frozen=True)
class SymbolicNonlinearSource:
    """Symbolic nonlinear polarization source wave at 2 omega."""

    label: str
    k: Any
    polarization: Any
    omega: Any


@dataclass(frozen=True)
class SymbolicInhomogeneousField:
    """Symbolic particular 2 omega electric field."""

    label: str
    k: Any
    electric: Any
    polarization: Any
    omega: Any
    residual: Any


@dataclass(frozen=True)
class SymbolicSingleInterfaceSources:
    """Symbolic `ee`, `oo`, and `eo` nonlinear source polarizations."""

    ee: SymbolicNonlinearSource
    oo: SymbolicNonlinearSource
    eo: SymbolicNonlinearSource


@dataclass(frozen=True)
class SymbolicSingleInterfaceInhomogeneousFields:
    """Symbolic `ee`, `oo`, and `eo` particular fields."""

    ee: SymbolicInhomogeneousField
    oo: SymbolicInhomogeneousField
    eo: SymbolicInhomogeneousField


@dataclass(frozen=True)
class SymbolicSingleInterfaceSHGBoundaryResult:
    """Symbolic final 2 omega single-interface boundary solve."""

    reflected_s_2omega: SymbolicWave
    reflected_p_2omega: SymbolicWave
    transmitted_fast_2omega: SymbolicWave
    transmitted_slow_2omega: SymbolicWave
    coefficients: Any
    boundary_residual: Any


@dataclass(frozen=True)
class SymbolicSingleInterfaceSHGKnownWavesResult:
    """Symbolic single-interface SHG solve from already-known wave bases."""

    sources: SymbolicSingleInterfaceSources
    inhomogeneous: SymbolicSingleInterfaceInhomogeneousFields
    boundary: SymbolicSingleInterfaceSHGBoundaryResult


@dataclass(frozen=True)
class SymbolicIsotropicSingleInterfaceSHGResult:
    """Symbolic isotropic single-interface SHG assembly."""

    linear_omega: SymbolicLinearInterfaceResult
    shg: SymbolicSingleInterfaceSHGKnownWavesResult
    reflected_basis_2omega: list[SymbolicWave]
    transmitted_basis_2omega: list[SymbolicWave]


def make_isotropic_wave_symbolic(
    *,
    n: Any,
    theta_rad: Any,
    polarization: str,
    amplitude: Any = 1,
    omega: Any = 1,
    direction_sign_z: Any = 1,
) -> SymbolicWave:
    """Create a symbolic s or p polarized isotropic plane wave."""

    sp = _sympy()
    if polarization not in {"s", "p"}:
        raise ValueError("polarization must be 's' or 'p'.")
    k_dir = sp.Matrix([sp.sin(theta_rad), 0, direction_sign_z * sp.cos(theta_rad)])
    if polarization == "s":
        electric_dir = sp.Matrix([0, 1, 0])
    else:
        electric_dir = sp.Matrix([direction_sign_z * sp.cos(theta_rad), 0, -sp.sin(theta_rad)])
    return SymbolicWave(k=n * omega * k_dir, electric=amplitude * electric_dir, omega=omega)


def fresnel_quadratic_eigenmodes_symbolic(principal_eps: Any, s: Any, *, simplify: bool = True) -> list:
    """Closed-form (symbolic) transverse eigenmodes for a wave normal ``s`` in a medium
    with PRINCIPAL-axis dielectric tensor ``diag(ex, ey, ez)`` -- the symbolic
    building block the anisotropic full-analytical SI assembly needs (the isotropic
    symbolic path only has s/p; the numeric ``modes_for_direction`` is already
    closed-form-validated for all classes, this is its explicit symbolic form).

    INTENTION / derivation (stated before use, per standing practice): the two
    transverse-mode indices are the roots of the Fresnel equation of wave normals

        sum_i s_i^2 / (u - 1/e_i) = 0, u = 1/n^2.

    Clearing the denominators and USING the unit wave normal s_x^2+s_y^2+s_z^2 = 1 cancels
    the cubic term, leaving a QUADRATIC in u:

        u^2 + B u + C = 0,
        B = -( s_x^2(a_y+a_z) + s_y^2(a_x+a_z) + s_z^2(a_x+a_y) ) , a_i = 1/e_i,
        C = s_x^2 a_y a_z + s_y^2 a_x a_z + s_z^2 a_x a_y.

    Each root u gives ``n = 1/sqrt(u)``; the field direction follows from D being
    transverse to s with ``D_i ~ s_i/(u - a_i)`` and ``E = eps^{-1} D`` (so
    ``E_i = D_i / e_i``). Returns ``[(n_plus, E_dir_plus), (n_minus, E_dir_minus)]``.

    DEGENERACY (intention/domain, learned the hard way): the ``D_i ~ s_i/(u-1/e_i)``
    field formula is SINGULAR when the mode index equals a principal index, i.e. when
    ``u = 1/e_i`` for some axis -- this is EXACTLY the ORDINARY mode of a UNIAXIAL
    crystal (``e_x = e_y`` -> ordinary root ``u = 1/e_x`` makes two denominators 0).
    The clean D-formula is only valid for NON-degenerate (biaxial) media. So the
    field direction is computed degeneracy-aware:

      * 0 zero-denominators (biaxial generic): the D-formula.
      * 2 zero-denominators (uniaxial ordinary mode): the field lies in the
        (isotropic) degenerate plane, perpendicular to k -> ``E ~ s x optic_axis``
        (optic_axis = the single NON-degenerate principal axis).
      * 1 zero-denominator (biaxial optic-axis special direction): ``E`` along that axis.
      * 3 zero-denominators (isotropic): any direction perpendicular to ``s``.

    ``s`` need NOT be pre-normalized; it is normalized here so that s_x^2+s_y^2+s_z^2 = 1 holds
    (the cubic-cancellation that makes the quadratic exact depends on it).
    """

    sp = _sympy()
    eps = sp.Matrix(principal_eps)
    if eps.shape == (3, 3):
        ex, ey, ez = eps[0, 0], eps[1, 1], eps[2, 2]
        offdiag = sum(abs(eps[i, j]) for i in range(3) for j in range(3) if i != j)
        if simplify and sp.simplify(offdiag) != 0:
            raise ValueError("principal_eps must be diagonal (principal-axis frame).")
    else:
        ex, ey, ez = eps[0], eps[1], eps[2]
    svec = sp.Matrix(s)
    svec = svec / sp.sqrt(svec.dot(svec))  # normalize so |s|^2 = 1 (load-bearing)
    sx, sy, sz = svec[0], svec[1], svec[2]
    ai = [1 / ex, 1 / ey, 1 / ez]
    ax, ay, az = ai
    basis_axes = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])]
    B = -(sx**2 * (ay + az) + sy**2 * (ax + az) + sz**2 * (ax + ay))
    C = sx**2 * ay * az + sy**2 * ax * az + sz**2 * ax * ay

    # Degeneracy is detected from the EPS VALUES (simple, reliable), NOT from the
    # radical-valued root u (sp.simplify cannot reduce `u - 1/e_i` through the sqrt,
    # which silently broke the earlier version for uniaxial crystals).
    def _eq(a, b):
        return (sp.simplify(a - b) == 0) if simplify else (a == b)

    def _d_formula(u):
        d_dir = sp.Matrix([svec[i] / (u - ai[i]) for i in range(3)])
        return sp.Matrix([d_dir[i] * ai[i] for i in range(3)])  # E_i = D_i / e_i

    def _perp_to_s(ref_index):
        ref = basis_axes[ref_index]
        cross = svec.cross(ref)
        if _eq(cross.dot(cross), 0):  # s parallel to this axis -> use another
            cross = svec.cross(basis_axes[(ref_index + 1) % 3])
        return cross

    # ---- isotropic: e_x = e_y = e_z. Both modes share n=sqrt(e); field undetermined
    # in the 2D plane perpendicular to s, so return any orthonormal perp pair. ----
    if _eq(ex, ey) and _eq(ey, ez):
        n_iso = sp.sqrt(ex)
        e1 = _perp_to_s(0)
        e2 = svec.cross(e1)
        modes = [(n_iso, e1), (n_iso, e2)]
    else:
        # ---- uniaxial: one degenerate principal PAIR; optic axis = the odd one.
        # Ordinary root is cleanly u_o = 1/e_o (NO radical); extraordinary follows
        # by Vieta (u_o + u_e = -B). Ordinary E ~ s x optic_axis; extraordinary
        # via the (now non-degenerate) D-formula. ----
        optic = None
        if _eq(ex, ey):
            optic, eo_val = 2, ex
        elif _eq(ey, ez):
            optic, eo_val = 0, ey
        elif _eq(ex, ez):
            optic, eo_val = 1, ex
        if optic is not None:
            u_o = 1 / eo_val
            u_e = -B - u_o  # Vieta: sum of roots of u^2 + B u + C
            modes = [
                (1 / sp.sqrt(u_o), svec.cross(basis_axes[optic])),
                (1 / sp.sqrt(u_e), _d_formula(u_e)),
            ]
        else:
            # ---- biaxial generic: two radical roots, clean D-formula for both. ----
            disc = sp.sqrt(B * B - 4 * C)
            modes = [
                (1 / sp.sqrt((-B + disc) / 2), _d_formula((-B + disc) / 2)),
                (1 / sp.sqrt((-B - disc) / 2), _d_formula((-B - disc) / 2)),
            ]

    if simplify:
        modes = [(sp.simplify(n), e_dir.applyfunc(sp.simplify)) for n, e_dir in modes]
    return modes


def _pick_inplane_mode(modes):
    """Return the (n, E) eigenmode lying in the x-z plane of incidence (E_y == 0), i.e.
    the extraordinary-like / in-plane mode (the other being the y-polarized ordinary mode).

    Symbolic-safe: prefers the mode whose y-component is STRUCTURALLY zero -- so it works
    when the refractive indices are symbolic (sympy cannot order symbolic magnitudes, which
    is why a ``max(|E_x|+|E_z|)`` selection raised "cannot determine truth value of
    Relational"). Falls back to the smallest |E_y| for purely numeric entries (where E_y of
    the in-plane mode is a tiny float, not exactly 0)."""
    sp = _sympy()
    structural = [m for m in modes if sp.simplify(m[1][1]) == 0]
    if len(structural) == 1:
        return structural[0]

    def _y_magnitude(mode):
        try:
            return abs(complex(mode[1][1]))
        except (TypeError, ValueError):
            return float("inf")

    return min(modes, key=_y_magnitude)


def solve_biaxial_linear_interface_symbolic(
    *,
    eps_x: Any,
    eps_y: Any,
    eps_z: Any,
    incident_index: Any,
    incident_theta_rad: Any,
    incident_polarization: str = "p",
    incident_jones: Any = None,
    omega: Any = 1,
    mu: Any = 1,
    simplify: bool = True,
):
    """Symbolic omega Fresnel boundary: isotropic incident medium -> anisotropic crystal
    with PRINCIPAL dielectric axes ALIGNED to the lab (eps = diag(ex, ey, ez)), surface
    normal z, plane of incidence x-z. General principal-aligned case; uniaxial and
    isotropic are special cases (``solve_uniaxial_linear_interface_symbolic`` delegates here).

    INTENTION / geometry (stated first): with the plane of incidence x-z (ky = 0), the
    diagonal-eps wave equation FACTORS into two transmitted eigenmodes that share the
    tangential wavevector kx = n_inc sin(theta_i) but have distinct z-components:

    * y-mode (E perpendicular to the plane of incidence, along y): from
      kx^2 + kz^2 = ey omega^2, so kz = sqrt(ey omega^2 - kx^2) and E = (0, 1, 0).
    * xz-mode (E in the x-z plane): from (kz^2 - ex omega^2)(kx^2 - ez omega^2) = kx^2 kz^2,
      so kz = sqrt(ex omega^2 - (ex/ez) kx^2); E from the (degeneracy-safe)
      ``fresnel_quadratic_eigenmodes_symbolic``.

    The 4 tangential continuity equations (2 reflected s/p + the 2 transmitted eigenmodes)
    are solved by the shared ``solve_boundary_continuity_symbolic``. Returns
    ``(coefficients, reflected_waves, transmitted_waves)`` with coefficients ordered
    ``[r_s, r_p, t_y, t_xz]``. Validated to ~1e-15 vs numeric ``solve_linear_interface``.

    INPUT POLARIZATION (the polarimetry handle): pass ``incident_jones=(J_s, J_p)`` to set
    the incident field to ``J_s * E_s + J_p * E_p`` (an arbitrary, possibly SYMBOLIC, input
    polarization -- e.g. ``J_s=sin(phi), J_p=cos(phi)`` for a polarimetry scan). The boundary
    solve is LINEAR in the incident field, so the returned coefficients/transmitted waves are
    ``J_s * (s-incidence) + J_p * (p-incidence)`` -- the building block that makes the
    downstream SHG bilinear in (J_s, J_p), i.e. the closed-form polarimetry I(phi). When
    omitted, falls back to the discrete ``incident_polarization`` ('s'/'p').
    """

    sp = _sympy()
    n_inc = incident_index
    kx = n_inc * omega * sp.sin(incident_theta_rad)
    kz_y = sp.sqrt(eps_y * omega**2 - kx**2)
    kz_xz = sp.sqrt(eps_x * omega**2 - (eps_x / eps_z) * kx**2)

    s_xz = sp.Matrix([kx, 0, kz_xz])
    modes = fresnel_quadratic_eigenmodes_symbolic([eps_x, eps_y, eps_z], s_xz, simplify=simplify)
    e_field = _pick_inplane_mode(modes)[1]  # in-plane (xz / extraordinary) mode -- symbolic-safe

    if incident_jones is not None:
        j_s, j_p = incident_jones
        e_s_wave = make_isotropic_wave_symbolic(n=n_inc, theta_rad=incident_theta_rad, polarization="s", omega=omega)
        e_p_wave = make_isotropic_wave_symbolic(n=n_inc, theta_rad=incident_theta_rad, polarization="p", omega=omega)
        incident = SymbolicWave(
            k=e_s_wave.k,
            electric=j_s * sp.Matrix(e_s_wave.electric) + j_p * sp.Matrix(e_p_wave.electric),
            omega=omega,
        )
    else:
        incident = make_isotropic_wave_symbolic(
            n=n_inc, theta_rad=incident_theta_rad, polarization=incident_polarization, omega=omega
        )
    reflected_basis = [
        make_isotropic_wave_symbolic(n=n_inc, theta_rad=incident_theta_rad, polarization="s", omega=omega, direction_sign_z=-1),
        make_isotropic_wave_symbolic(n=n_inc, theta_rad=incident_theta_rad, polarization="p", omega=omega, direction_sign_z=-1),
    ]
    transmitted_basis = [
        SymbolicWave(k=sp.Matrix([kx, 0, kz_y]), electric=sp.Matrix([0, 1, 0]), omega=omega),    # y-mode
        SymbolicWave(k=sp.Matrix([kx, 0, kz_xz]), electric=e_field, omega=omega),                # xz-mode
    ]
    reflected, transmitted, coeffs = solve_boundary_continuity_symbolic(
        top_known=[incident], bottom_known=[],
        top_unknown_basis=reflected_basis, bottom_unknown_basis=transmitted_basis,
        mu=mu, simplify=simplify,
    )
    return coeffs, reflected, transmitted


def solve_uniaxial_linear_interface_symbolic(
    *,
    eps_ordinary: Any,
    eps_extraordinary: Any,
    incident_index: Any,
    incident_theta_rad: Any,
    incident_polarization: str = "p",
    incident_jones: Any = None,
    omega: Any = 1,
    mu: Any = 1,
    simplify: bool = True,
):
    """Symbolic omega Fresnel boundary for a UNIAXIAL crystal with the OPTIC AXIS along
    the surface normal (z), plane of incidence x-z. This is the eps_x = eps_y special
    case of :func:`solve_biaxial_linear_interface_symbolic` (one unified method): the
    y-mode is the ordinary wave (kz=sqrt(eo-kx^2), E along y) and the xz-mode is the
    extraordinary wave (kz=sqrt(eo-(eo/ee)kx^2)). Delegates so there is no duplicated
    boundary logic. Validated to ~1e-15 vs numeric ``solve_linear_interface``.
    """

    return solve_biaxial_linear_interface_symbolic(
        eps_x=eps_ordinary, eps_y=eps_ordinary, eps_z=eps_extraordinary,
        incident_index=incident_index, incident_theta_rad=incident_theta_rad,
        incident_polarization=incident_polarization, incident_jones=incident_jones,
        omega=omega, mu=mu, simplify=simplify,
    )


def _biaxial_2omega_basis(ex2, ey2, ez2, kx, omega, simplify):
    """The two transmitted principal-aligned eigenmodes at 2 omega sharing tangential
    wavevector 2*kx (the SHG bound-source tangential k): the y-mode (E along y) and the
    xz-mode (E in x-z), with the same dispersion as at omega but at frequency 2*omega."""
    sp = _sympy()
    w2 = 2 * omega
    ktan = 2 * kx
    kz_y = sp.sqrt(ey2 * w2**2 - ktan**2)
    kz_xz = sp.sqrt(ex2 * w2**2 - (ex2 / ez2) * ktan**2)
    s_xz = sp.Matrix([ktan, 0, kz_xz])
    modes = fresnel_quadratic_eigenmodes_symbolic([ex2, ey2, ez2], s_xz, simplify=simplify)
    e_field = _pick_inplane_mode(modes)[1]  # in-plane (xz / extraordinary) mode -- symbolic-safe
    return [
        SymbolicWave(k=sp.Matrix([ktan, 0, kz_y]), electric=sp.Matrix([0, 1, 0]), omega=w2),
        SymbolicWave(k=sp.Matrix([ktan, 0, kz_xz]), electric=e_field, omega=w2),
    ]


def solve_biaxial_single_interface_shg_symbolic(
    *,
    eps_x_omega: Any,
    eps_y_omega: Any,
    eps_z_omega: Any,
    eps_x_2omega: Any,
    eps_y_2omega: Any,
    eps_z_2omega: Any,
    d_voigt_lab: Any,
    incident_theta_rad: Any,
    incident_polarization: str = "p",
    incident_jones: Any = None,
    omega: Any = 1,
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = True,
    capture_stages: dict | None = None,
):
    """End-to-end symbolic single-interface reflected SHG for an anisotropic crystal with
    PRINCIPAL dielectric axes ALIGNED to the lab (surface normal z, plane of incidence
    x-z). General principal-aligned case (biaxial); uniaxial and isotropic are special
    cases (``solve_uniaxial_single_interface_shg_symbolic`` delegates here).

    Stages (each independently validated): (1) the principal-aligned omega Fresnel
    boundary (``solve_biaxial_linear_interface_symbolic``) gives the two transmitted
    fundamental eigenmodes (y-mode + xz-mode); (2) they drive the ee/oo/eo nonlinear
    sources; (3) ``solveInhom`` gives the bound 2 omega fields; (4) the final 2 omega
    boundary is solved against the principal-aligned 2 omega eigenmode bases. DISPERSION
    NOTE: pass distinct omega and 2 omega indices -- equal indices put the bound source on
    a free-wave shell (phase matching) and make ``solveInhom`` singular (a real resonance).

    Returns the ``SymbolicSingleInterfaceSHGKnownWavesResult`` whose
    ``.boundary.coefficients`` are the reflected SHG amplitudes ``[E_s^2w, E_p^2w, ...]``.
    Validated to ~1e-15 vs the numeric ``shaarp.shg.solve_single_interface_shg``.

    SYMBOLIC SHG COEFFICIENT (d_ijk): ``d_voigt_lab`` MAY be a symbolic sympy Matrix (the
    whole chain is symbolic and computePNL is linear in d), so the result is a closed form
    in the d components (and angles). Use ``simplify=False`` for speed. NOTE for validation:
    a symbolic-vs-numeric comparison must pass IDENTICAL ``mu, eps0`` to both this function
    and ``solve_single_interface_shg`` (mixing eps0=1 with the numeric default EPS0 rescales
    the inhomogeneous field per polarization row -- a units artifact, not a discrepancy).
    """

    sp = _sympy()
    _, _, transmitted_omega = solve_biaxial_linear_interface_symbolic(
        eps_x=eps_x_omega, eps_y=eps_y_omega, eps_z=eps_z_omega,
        incident_index=1, incident_theta_rad=incident_theta_rad,
        incident_polarization=incident_polarization, incident_jones=incident_jones,
        omega=omega, mu=mu, simplify=simplify,
    )
    kx = omega * sp.sin(incident_theta_rad)  # incident air (n=1)
    transmitted_basis_2omega = _biaxial_2omega_basis(
        eps_x_2omega, eps_y_2omega, eps_z_2omega, kx, omega, simplify
    )
    theta_2omega = sp.asin(kx / omega)  # reflected 2omega in air, tangential 2kx, |k|=2*omega
    reflected_basis_2omega = [
        make_isotropic_wave_symbolic(n=1, theta_rad=theta_2omega, polarization="s", omega=2 * omega, direction_sign_z=-1),
        make_isotropic_wave_symbolic(n=1, theta_rad=theta_2omega, polarization="p", omega=2 * omega, direction_sign_z=-1),
    ]
    eps_2omega_lab = sp.diag(eps_x_2omega, eps_y_2omega, eps_z_2omega)
    result = solve_single_interface_shg_known_waves_symbolic(
        epsilon_2omega_lab=eps_2omega_lab,
        d_voigt_lab=d_voigt_lab,
        transmitted_fast_omega=transmitted_omega[0],
        transmitted_slow_omega=transmitted_omega[1],
        reflected_basis_2omega=reflected_basis_2omega,
        transmitted_basis_2omega=transmitted_basis_2omega,
        mu=mu, eps0=eps0, simplify=simplify,
    )
    if capture_stages is not None:
        # Expose the FOUR published derivation stages (the docstring's steps 1-4) so the GUI can
        # show the full step-by-step build-up, not just the final reflected amplitude. These are
        # the ACTUAL objects used in THIS solve (no re-derivation) -- eq 29-32 / 20-28 / 11-19 / 9-10.
        capture_stages["omega_eigenmodes"] = transmitted_omega          # (1) transmitted fundamental fields
        capture_stages["sources"] = result.sources                     # (2) nonlinear polarization P_NL
        capture_stages["inhomogeneous"] = result.inhomogeneous         # (3) bound 2omega fields
        capture_stages["boundary"] = result.boundary                   # (4) final 2omega boundary solve
    return result


def solve_uniaxial_single_interface_shg_symbolic(
    *,
    eps_ordinary_omega: Any,
    eps_extraordinary_omega: Any,
    eps_ordinary_2omega: Any,
    eps_extraordinary_2omega: Any,
    d_voigt_lab: Any,
    incident_theta_rad: Any,
    incident_polarization: str = "p",
    incident_jones: Any = None,
    omega: Any = 1,
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = True,
):
    """End-to-end symbolic UNIAXIAL single-interface reflected SHG (optic axis along the
    surface normal z). This is the eps_x = eps_y special case of
    :func:`solve_biaxial_single_interface_shg_symbolic` (one unified method): the y-mode
    is the ordinary wave and the xz-mode the extraordinary. Delegates so there is no
    duplicated assembly logic. Validated to ~1e-15 vs numeric ``solve_single_interface_shg``.
    DISPERSION NOTE: pass distinct omega and 2 omega indices (equal -> phase-matching
    singularity in solveInhom).
    """

    return solve_biaxial_single_interface_shg_symbolic(
        eps_x_omega=eps_ordinary_omega, eps_y_omega=eps_ordinary_omega, eps_z_omega=eps_extraordinary_omega,
        eps_x_2omega=eps_ordinary_2omega, eps_y_2omega=eps_ordinary_2omega, eps_z_2omega=eps_extraordinary_2omega,
        d_voigt_lab=d_voigt_lab, incident_theta_rad=incident_theta_rad,
        incident_polarization=incident_polarization, incident_jones=incident_jones,
        omega=omega, mu=mu, eps0=eps0, simplify=simplify,
    )


@dataclass(frozen=True)
class SymbolicSIFullPolarimetry:
    """The 'full analytical' single-interface reflected SHG POLARIMETRY -- the reflected SHG
    field as a closed form in the input polarization angle ``phi`` (and, optionally, the
    analyzer angle ``psi``, the sample azimuth, the incidence angle, the refractive indices,
    and the d_ijk tensor). This is the expression an experimentalist fits to a measured
    polarimetry scan to EXTRACT the SHG coefficient d -- the project's key analytical output.

    ``reflected_s`` / ``reflected_p`` are ``E_s^2w(phi, ...)`` and ``E_p^2w(phi, ...)``.
    ``intensity`` is the analyzed intensity ``I(phi, psi) = |sin(psi) E_s + cos(psi) E_p|^2``
    (SHAARP analyzer convention). ``intensity_s`` / ``intensity_p`` are the pure-channel
    intensities ``|E_s|^2`` / ``|E_p|^2`` (psi = 90 / 0 deg)."""

    reflected_s: Any
    reflected_p: Any
    phi_symbol: Any
    analyzer_symbol: Any
    intensity: Any
    raw: Any  # underlying SymbolicSingleInterfaceSHGKnownWavesResult

    @property
    def intensity_s(self) -> Any:
        sp = _sympy()
        return sp.Abs(self.reflected_s) ** 2

    @property
    def intensity_p(self) -> Any:
        sp = _sympy()
        return sp.Abs(self.reflected_p) ** 2

    @property
    def transmitted_field(self) -> Any:
        """The TOTAL transmitted SHG electric field at the interface as a closed form in
        ``phi`` (the sum of the two homogeneous transmitted 2 omega eigenmodes) -- the
        transmitted-channel polarimetry (Maker-fringe) observable. Validated vs the numeric
        total transmitted field to ~1e-15; the per-mode split uses a different basis than the
        numeric fast/slow modes, but the TOTAL field is basis-independent and agrees."""
        sp = _sympy()
        b = self.raw.boundary
        return sp.Matrix(b.transmitted_fast_2omega.electric) + sp.Matrix(b.transmitted_slow_2omega.electric)

    @property
    def transmitted_intensity(self) -> Any:
        """Total transmitted SHG intensity (squared magnitude of E_t) as a closed form in ``phi``."""
        sp = _sympy()
        e_t = self.transmitted_field
        return sum(sp.Abs(c) ** 2 for c in e_t)


def solve_si_shg_full_analytical_symbolic(
    *,
    eps_x_omega: Any,
    eps_y_omega: Any,
    eps_z_omega: Any,
    eps_x_2omega: Any,
    eps_y_2omega: Any,
    eps_z_2omega: Any,
    d_voigt_lab: Any,
    incident_theta_rad: Any,
    phi_symbol: Any = None,
    analyzer_symbol: Any = None,
    sample_azimuth_symbol: Any = None,
    ellipticity: Any = 0,
    omega: Any = 1,
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = False,
    layered: bool = False,
    capture_stages: dict | None = None,
) -> SymbolicSIFullPolarimetry:
    """FULL analytical single-interface reflected SHG polarimetry: symbolic in EVERYTHING
    SHAARP.si supports -- input polarization ``phi``, analyzer ``psi``, sample azimuth,
    incidence ``theta``, refractive indices (principal-aligned, possibly symbolic), and the
    d_ijk tensor. Returns the reflected SHG field as a closed form ``E_s/E_p^2w(phi, ...)``
    plus the analyzed intensity ``I(phi, psi)`` -- the d-extraction expression.

    Composition (each stage already validated): the input polarization sets the incident
    Jones ``(J_s, J_p) = (sin(phi) e^{i*ellipticity}, cos(phi))`` (SHAARP convention); the
    sample azimuth rotates the lab d-tensor about the surface normal z by
    ``sample_azimuth_symbol`` (``rotate_d_voigt_symbolic``). NOTE: the azimuth rotates ONLY d,
    not eps, so it is physical only for azimuth-INVARIANT eps (isotropic, or uniaxial with the
    optic axis along z); for a biaxial crystal under a non-trivial azimuth the result is
    unphysical (it emits a RuntimeWarning) -- use the numeric path (``extract_si_d_voigt``
    ``basis="numeric"``) for rotated-biaxial azimuth scans. The reflected SHG is then the
    end-to-end principal-aligned chain ``solve_biaxial_single_interface_shg_symbolic``
    (linear in the incident field, bilinear in the fundamental -> the polarimetry curve).

    Tractable/validated for the isotropic (published GaAs(111)) and uniaxial classes; the
    fully-symbolic biaxial form is available but large (use ``simplify=False``). DISPERSION
    NOTE: pass distinct omega/2omega indices (equal -> phase-matching singularity).
    """

    sp = _sympy()
    phi = phi_symbol if phi_symbol is not None else sp.Symbol("phi", real=True)
    psi = analyzer_symbol if analyzer_symbol is not None else sp.Symbol("psi", real=True)

    d_used = d_voigt_lab
    if sample_azimuth_symbol is not None:
        # A sample azimuth physically rotates BOTH d and eps. This closed form rotates ONLY d:
        # rotating eps would turn a biaxial crystal into a general ROTATED biaxial, whose kz solve
        # is the full Booker QUARTIC (intractable as a closed form). So the azimuth result is
        # physical only when eps is azimuth-INVARIANT about z -- isotropic, or uniaxial with the
        # optic axis along the surface normal. For a BIAXIAL crystal (eps_x != eps_y) under a
        # non-trivial azimuth, use the NUMERIC path instead (e.g. extract_si_d_voigt
        # basis="numeric", which rotates eps too). Warn for the clearly-detectable numeric case.
        try:
            import math as _math
            import warnings as _warnings
            _az = float(sample_azimuth_symbol)
            _ex, _ey = complex(eps_x_omega), complex(eps_y_omega)
            _r = _az % _math.pi  # Rz(az) leaves a diagonal eps invariant only when az = 0 (mod pi)
            if min(_r, _math.pi - _r) > 1e-9 and abs(_ex - _ey) > 1e-9 * max(abs(_ex), abs(_ey), 1.0):
                _warnings.warn(
                    "solve_si_shg_full_analytical_symbolic: a sample azimuth rotates only the "
                    "d-tensor here, not eps; for a BIAXIAL crystal (eps_x != eps_y) the result is "
                    "unphysical under a non-trivial azimuth (the eigenmodes would need the rotated "
                    "eps -- the intractable rotated-biaxial quartic). Use the numeric path "
                    "(extract_si_d_voigt basis='numeric') for rotated-biaxial azimuth scans.",
                    RuntimeWarning, stacklevel=2,
                )
        except (TypeError, ValueError):
            pass  # symbolic eps/azimuth -> cannot check numerically; see the docstring note
        c, s = sp.cos(sample_azimuth_symbol), sp.sin(sample_azimuth_symbol)
        # rotate d about the surface normal z. NOTE the transpose: rotate_d_voigt_symbolic
        # uses out_ijk = sum a[i,l] d[l..] (a = R), whereas the validated numeric physical
        # rotation rotate_d_voigt_crystal_to_lab uses einsum("ai,..") (a = R^T). Passing
        # R^T here makes a positive sample_azimuth the SAME physical sample rotation as the
        # numeric path (matches to ~1e-16) -- so the C3v 3-fold azimuth signature is correct.
        rot_z_T = sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        d_used = rotate_d_voigt_symbolic(d_voigt_lab, rot_z_T, simplify=simplify)

    j_s = sp.sin(phi) * sp.exp(sp.I * ellipticity) if ellipticity != 0 else sp.sin(phi)
    j_p = sp.cos(phi)
    common = dict(
        eps_x_omega=eps_x_omega, eps_y_omega=eps_y_omega, eps_z_omega=eps_z_omega,
        eps_x_2omega=eps_x_2omega, eps_y_2omega=eps_y_2omega, eps_z_2omega=eps_z_2omega,
        d_voigt_lab=d_used, incident_theta_rad=incident_theta_rad,
        incident_jones=(j_s, j_p), omega=omega, mu=mu, eps0=eps0, simplify=simplify,
        capture_stages=capture_stages,
    )
    if layered:
        # His printed form: keep each eigenmode a NAMED symbol with its own defining equation
        # instead of substituting the radical inline. Same physics, same result -- but the algebra
        # stays small, which is what makes a BIAXIAL eps tractable (measured 0.4 s vs >420 s).
        res = solve_si_shg_layered_symbolic(**common).raw
    else:
        res = solve_biaxial_single_interface_shg_symbolic(**common)
    e_s = res.boundary.coefficients[0]
    e_p = res.boundary.coefficients[1]
    analyzed = sp.sin(psi) * e_s + sp.cos(psi) * e_p  # I(phi,psi)=|sin psi E_s + cos psi E_p|^2
    intensity = sp.Abs(analyzed) ** 2
    return SymbolicSIFullPolarimetry(
        reflected_s=e_s, reflected_p=e_p, phi_symbol=phi, analyzer_symbol=psi,
        intensity=intensity, raw=res,
    )


def solve_isotropic_linear_interface_symbolic(
    *,
    incident_index: Any,
    transmitted_index: Any,
    incident_theta_rad: Any,
    incident_polarization: str,
    omega: Any = 1,
    mu: Any = 1,
    simplify: bool = True,
) -> SymbolicLinearInterfaceResult:
    """Solve symbolic linear Fresnel boundary conditions for isotropic media."""

    sp = _sympy()
    if incident_polarization not in {"s", "p"}:
        raise ValueError("incident_polarization must be 's' or 'p'.")
    theta_t = sp.asin(incident_index * sp.sin(incident_theta_rad) / transmitted_index)
    incident = make_isotropic_wave_symbolic(
        n=incident_index,
        theta_rad=incident_theta_rad,
        polarization=incident_polarization,
        omega=omega,
    )
    reflected_basis = [
        make_isotropic_wave_symbolic(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="s",
            omega=omega,
            direction_sign_z=-1,
        ),
        make_isotropic_wave_symbolic(
            n=incident_index,
            theta_rad=incident_theta_rad,
            polarization="p",
            omega=omega,
            direction_sign_z=-1,
        ),
    ]
    transmitted_basis = [
        make_isotropic_wave_symbolic(n=transmitted_index, theta_rad=theta_t, polarization="s", omega=omega),
        make_isotropic_wave_symbolic(n=transmitted_index, theta_rad=theta_t, polarization="p", omega=omega),
    ]
    reflected, transmitted, coeffs = solve_fresnel_boundary_symbolic(
        [incident],
        reflected_basis,
        transmitted_basis,
        mu=mu,
        simplify=simplify,
    )
    return SymbolicLinearInterfaceResult(
        incident=incident,
        reflected_s=reflected[0],
        reflected_p=reflected[1],
        transmitted_s=transmitted[0],
        transmitted_p=transmitted[1],
        coefficients=coeffs,
        theta_transmitted=theta_t,
    )


def _source_from_pair_symbolic(label: str, d_voigt_lab: Any, wave_i: SymbolicWave, wave_j: SymbolicWave, *, factor: Any) -> SymbolicNonlinearSource:
    """Symbolic counterpart of the numeric ``_source_from_pair`` primitive."""

    sp = _sympy()
    return SymbolicNonlinearSource(
        label=label,
        k=sp.Matrix(wave_i.k) + sp.Matrix(wave_j.k),
        polarization=compute_pnl_voigt_symbolic(d_voigt_lab, wave_i.electric, wave_j.electric, factor=factor),
        omega=wave_i.omega + wave_j.omega,
    )


def pairwise_nonlinear_sources_symbolic(d_voigt_lab: Any, waves: list, pairs: list) -> dict:
    """Symbolic counterpart of the unified numeric :func:`pairwise_nonlinear_sources`.

    Same SHAARP pairwise rule (self pair -> factor 1, k=2k_i, omega=2omega_i; cross
    pair -> factor 2, k=k_i+k_j) so the symbolic single-interface path uses the SAME
    convention as the numeric SI/ML source generators -- one rule, one place.
    """

    sources: dict = {}
    for i, j, label in pairs:
        factor = 1 if i == j else 2
        sources[label] = _source_from_pair_symbolic(label, d_voigt_lab, waves[i], waves[j], factor=factor)
    return sources


def single_interface_sources_symbolic(d_voigt_lab: Any, fast_wave: SymbolicWave, slow_wave: SymbolicWave) -> SymbolicSingleInterfaceSources:
    """Generate symbolic SHAARP single-interface source waves: ee, oo, eo.

    Thin wrapper over :func:`pairwise_nonlinear_sources_symbolic` -- the symbolic
    analogue of the unified numeric primitive, with the SI pair set ee/oo/eo.
    """

    src = pairwise_nonlinear_sources_symbolic(
        d_voigt_lab, [fast_wave, slow_wave], [(0, 0, "ee"), (1, 1, "oo"), (0, 1, "eo")]
    )
    return SymbolicSingleInterfaceSources(ee=src["ee"], oo=src["oo"], eo=src["eo"])


def cross_matrix_symbolic(vector: Any) -> Any:
    """Return the symbolic matrix K such that K @ a == vector x a."""

    sp = _sympy()
    v = sp.Matrix(vector)
    if v.shape != (3, 1):
        raise ValueError("vector must be a length-3 vector.")
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def curl_curl_matrix_symbolic(k: Any) -> Any:
    """Symbolic plane-wave curl-curl matrix matching the numerical solver."""

    kx = cross_matrix_symbolic(k)
    return -(kx * kx)


def solve_inhomogeneous_field_symbolic(
    epsilon_2omega_lab: Any,
    source: SymbolicNonlinearSource,
    *,
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = True,
) -> SymbolicInhomogeneousField:
    """Symbolic counterpart of the Mathematica `solveInhom` stage."""

    sp = _sympy()
    epsilon = sp.Matrix(epsilon_2omega_lab)
    k = sp.Matrix(source.k)
    polarization = sp.Matrix(source.polarization)
    if epsilon.shape != (3, 3):
        raise ValueError("epsilon_2omega_lab must be a 3x3 matrix.")
    if k.shape != (3, 1) or polarization.shape != (3, 1):
        raise ValueError("source k and polarization must be length-3 vectors.")

    alpha = source.omega**2 * mu * eps0
    operator = curl_curl_matrix_symbolic(k)
    system = operator - alpha * epsilon
    electric = system.LUsolve(alpha * polarization)
    residual = operator * electric - alpha * (epsilon * electric + polarization)
    if simplify:
        electric = electric.applyfunc(sp.simplify)
        residual = residual.applyfunc(sp.simplify)
    return SymbolicInhomogeneousField(
        label=source.label,
        k=k,
        electric=electric,
        polarization=polarization,
        omega=source.omega,
        residual=residual,
    )


def solve_single_interface_inhomogeneous_fields_symbolic(
    epsilon_2omega_lab: Any,
    sources: SymbolicSingleInterfaceSources,
    *,
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = True,
) -> SymbolicSingleInterfaceInhomogeneousFields:
    """Solve symbolic `solveInhom` fields for `ee`, `oo`, and `eo` sources."""

    return SymbolicSingleInterfaceInhomogeneousFields(
        ee=solve_inhomogeneous_field_symbolic(epsilon_2omega_lab, sources.ee, mu=mu, eps0=eps0, simplify=simplify),
        oo=solve_inhomogeneous_field_symbolic(epsilon_2omega_lab, sources.oo, mu=mu, eps0=eps0, simplify=simplify),
        eo=solve_inhomogeneous_field_symbolic(epsilon_2omega_lab, sources.eo, mu=mu, eps0=eps0, simplify=simplify),
    )


def solve_single_interface_shg_boundary_symbolic(
    *,
    reflected_basis: list[SymbolicWave],
    transmitted_basis: list[SymbolicWave],
    inhomogeneous: SymbolicSingleInterfaceInhomogeneousFields | list[SymbolicWave],
    mu: Any = 1,
    simplify: bool = True,
) -> SymbolicSingleInterfaceSHGBoundaryResult:
    """Solve the symbolic final 2 omega boundary system in SHAARP `f1NL`.

    This matches the numerical stage with no incident SH wave: two reflected
    homogeneous waves on the incident side, two transmitted homogeneous waves
    plus three known inhomogeneous waves on the transmitted side.
    """

    if len(reflected_basis) != 2 or len(transmitted_basis) != 2:
        raise ValueError("Expected two reflected and two transmitted homogeneous basis waves.")
    inhomogeneous_waves = _inhomogeneous_waves_symbolic(inhomogeneous)
    reflected, transmitted, coeffs = solve_boundary_continuity_symbolic(
        top_known=[],
        bottom_known=inhomogeneous_waves,
        top_unknown_basis=reflected_basis,
        bottom_unknown_basis=transmitted_basis,
        mu=mu,
        simplify=simplify,
    )
    residual = boundary_residual_symbolic(reflected, transmitted + inhomogeneous_waves, mu=mu)
    if simplify:
        residual = residual.applyfunc(_sympy().simplify)
    return SymbolicSingleInterfaceSHGBoundaryResult(
        reflected_s_2omega=reflected[0],
        reflected_p_2omega=reflected[1],
        transmitted_fast_2omega=transmitted[0],
        transmitted_slow_2omega=transmitted[1],
        coefficients=coeffs,
        boundary_residual=residual,
    )


def solve_single_interface_shg_known_waves_symbolic(
    *,
    epsilon_2omega_lab: Any,
    d_voigt_lab: Any,
    transmitted_fast_omega: SymbolicWave,
    transmitted_slow_omega: SymbolicWave,
    reflected_basis_2omega: list[SymbolicWave],
    transmitted_basis_2omega: list[SymbolicWave],
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = True,
) -> SymbolicSingleInterfaceSHGKnownWavesResult:
    """Assemble symbolic single-interface SHG once homogeneous waves are known.

    This is the symbolic analogue of the verified numerical `f1NL` core after
    the omega linear solve and 2omega homogeneous Snell solve have supplied
    their wave bases.
    """

    sources = single_interface_sources_symbolic(d_voigt_lab, transmitted_fast_omega, transmitted_slow_omega)
    inhomogeneous = solve_single_interface_inhomogeneous_fields_symbolic(
        epsilon_2omega_lab,
        sources,
        mu=mu,
        eps0=eps0,
        simplify=simplify,
    )
    boundary = solve_single_interface_shg_boundary_symbolic(
        reflected_basis=reflected_basis_2omega,
        transmitted_basis=transmitted_basis_2omega,
        inhomogeneous=inhomogeneous,
        mu=mu,
        simplify=simplify,
    )
    return SymbolicSingleInterfaceSHGKnownWavesResult(
        sources=sources,
        inhomogeneous=inhomogeneous,
        boundary=boundary,
    )


def solve_isotropic_single_interface_shg_symbolic(
    *,
    transmitted_epsilon_2omega: Any,
    d_voigt_lab: Any,
    incident_index_omega: Any,
    transmitted_index_omega: Any,
    incident_index_2omega: Any,
    transmitted_index_2omega: Any,
    incident_theta_rad: Any,
    incident_polarization: str,
    omega: Any = 1,
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = True,
) -> SymbolicIsotropicSingleInterfaceSHGResult:
    """Assemble symbolic isotropic single-interface SHG.

    This is a scoped symbolic path for isotropic linear media. It does not
    solve anisotropic symbolic Snell roots; instead, it uses s/p transmitted
    omega waves as the two homogeneous fundamental waves and builds
    phase-matched isotropic s/p bases at 2 omega.
    """

    sp = _sympy()
    linear = solve_isotropic_linear_interface_symbolic(
        incident_index=incident_index_omega,
        transmitted_index=transmitted_index_omega,
        incident_theta_rad=incident_theta_rad,
        incident_polarization=incident_polarization,
        omega=omega,
        mu=mu,
        simplify=simplify,
    )
    omega_2 = 2 * omega
    tangential_target = incident_index_omega * sp.sin(incident_theta_rad)
    theta_reflected_2omega = sp.asin(tangential_target / incident_index_2omega)
    theta_transmitted_2omega = sp.asin(tangential_target / transmitted_index_2omega)
    reflected_basis = [
        make_isotropic_wave_symbolic(
            n=incident_index_2omega,
            theta_rad=theta_reflected_2omega,
            polarization="s",
            omega=omega_2,
            direction_sign_z=-1,
        ),
        make_isotropic_wave_symbolic(
            n=incident_index_2omega,
            theta_rad=theta_reflected_2omega,
            polarization="p",
            omega=omega_2,
            direction_sign_z=-1,
        ),
    ]
    transmitted_basis = [
        make_isotropic_wave_symbolic(n=transmitted_index_2omega, theta_rad=theta_transmitted_2omega, polarization="s", omega=omega_2),
        make_isotropic_wave_symbolic(n=transmitted_index_2omega, theta_rad=theta_transmitted_2omega, polarization="p", omega=omega_2),
    ]
    shg = solve_single_interface_shg_known_waves_symbolic(
        epsilon_2omega_lab=transmitted_epsilon_2omega,
        d_voigt_lab=d_voigt_lab,
        transmitted_fast_omega=linear.transmitted_s,
        transmitted_slow_omega=linear.transmitted_p,
        reflected_basis_2omega=reflected_basis,
        transmitted_basis_2omega=transmitted_basis,
        mu=mu,
        eps0=eps0,
        simplify=simplify,
    )
    return SymbolicIsotropicSingleInterfaceSHGResult(
        linear_omega=linear,
        shg=shg,
        reflected_basis_2omega=reflected_basis,
        transmitted_basis_2omega=transmitted_basis,
    )


def solve_fresnel_boundary_symbolic(
    incident: list[SymbolicWave],
    reflected_basis: list[SymbolicWave],
    transmitted_basis: list[SymbolicWave],
    *,
    mu: Any = 1,
    simplify: bool = True,
) -> tuple[list[SymbolicWave], list[SymbolicWave], Any]:
    """Symbolic counterpart of `solve_fresnel_boundary`."""

    return solve_boundary_continuity_symbolic(
        top_known=incident,
        bottom_known=[],
        top_unknown_basis=reflected_basis,
        bottom_unknown_basis=transmitted_basis,
        mu=mu,
        simplify=simplify,
    )


def solve_boundary_continuity_symbolic(
    *,
    top_known: list[SymbolicWave],
    bottom_known: list[SymbolicWave],
    top_unknown_basis: list[SymbolicWave],
    bottom_unknown_basis: list[SymbolicWave],
    mu: Any = 1,
    simplify: bool = True,
) -> tuple[list[SymbolicWave], list[SymbolicWave], Any]:
    """Solve symbolic tangential E/H continuity for four unknown amplitudes."""

    sp = _sympy()
    bases = list(top_unknown_basis) + list(bottom_unknown_basis)
    if len(bases) != 4:
        raise ValueError("This boundary solve expects exactly 4 unknown basis waves.")

    known_mismatch = _tangential_sum_symbolic(top_known, mu=mu) - _tangential_sum_symbolic(bottom_known, mu=mu)
    rhs = -known_mismatch
    matrix = sp.Matrix.hstack(
        *[_signed_tangential_column_symbolic(w, idx, len(top_unknown_basis), mu) for idx, w in enumerate(bases)]
    )
    coeffs = matrix.LUsolve(rhs)
    if simplify:
        coeffs = coeffs.applyfunc(sp.simplify)

    top_unknown = [scale_wave_symbolic(w, coeffs[i], simplify=simplify) for i, w in enumerate(top_unknown_basis)]
    bottom_unknown = [
        scale_wave_symbolic(w, coeffs[len(top_unknown_basis) + i], simplify=simplify)
        for i, w in enumerate(bottom_unknown_basis)
    ]
    return top_unknown, bottom_unknown, coeffs


def boundary_residual_symbolic(top_waves: list[SymbolicWave], bottom_waves: list[SymbolicWave], *, mu: Any = 1) -> Any:
    """Return symbolic tangential top-minus-bottom E/H mismatch."""

    return _tangential_sum_symbolic(top_waves, mu=mu) - _tangential_sum_symbolic(bottom_waves, mu=mu)


def scale_wave_symbolic(wave: SymbolicWave, amplitude: Any, *, simplify: bool = True) -> SymbolicWave:
    sp = _sympy()
    electric = sp.Matrix(wave.electric) * amplitude
    if simplify:
        electric = electric.applyfunc(sp.simplify)
    return SymbolicWave(k=wave.k, electric=electric, omega=wave.omega)


def _inhomogeneous_waves_symbolic(inhomogeneous: SymbolicSingleInterfaceInhomogeneousFields | list[SymbolicWave]) -> list[SymbolicWave]:
    if isinstance(inhomogeneous, list):
        return inhomogeneous
    return [
        SymbolicWave(k=inhomogeneous.ee.k, electric=inhomogeneous.ee.electric, omega=inhomogeneous.ee.omega),
        SymbolicWave(k=inhomogeneous.oo.k, electric=inhomogeneous.oo.electric, omega=inhomogeneous.oo.omega),
        SymbolicWave(k=inhomogeneous.eo.k, electric=inhomogeneous.eo.electric, omega=inhomogeneous.eo.omega),
    ]


def _tangential_sum_symbolic(waves: list[SymbolicWave], *, mu: Any) -> Any:
    sp = _sympy()
    if not waves:
        return sp.zeros(4, 1)
    e = sp.zeros(3, 1)
    h = sp.zeros(3, 1)
    for wave in waves:
        e += sp.Matrix(wave.electric)
        h += wave.magnetic(mu=mu)
    return sp.Matrix(tangential_eh(e, h))  # canonical SHAARP [Ex,Ey,Hx,Hy] (unified with numeric)


def _signed_tangential_column_symbolic(wave: SymbolicWave, index: int, n_reflected: int, mu: Any) -> Any:
    sp = _sympy()
    sign = sp.Integer(1) if index < n_reflected else sp.Integer(-1)
    e = sp.Matrix(wave.electric)
    h = wave.magnetic(mu=mu)
    return sign * sp.Matrix(tangential_eh(e, h))  # canonical SHAARP [Ex,Ey,Hx,Hy] (unified with numeric)


def _canonical_point_group(point_group: str) -> str:
    # the package-wide alias table (Mathematica OverscriptBox spellings, the .si popup's
    # "-62m", Curie groups, Schoenflies) lives in point_groups; its symbolic_key() yields the
    # ASCII keys this function's callers branch on (inf", "infm", "in", "-43m", ...).
    from .point_groups import is_known_point_group, symbolic_key
    if is_known_point_group(point_group):
        return symbolic_key(point_group).lower()
    pg = point_group.strip().lower().replace(" ", "")
    aliases = {
        "infinity": "inf",
        "infinite": "inf",
        "\\[infinity]": "inf",
        "infinitym": "infm",
        "\\[infinity]m": "infm",
        "infinity2": "inf2",
        "\\[infinity]2": "inf2",
        "c3v": "3m",
        "c4v": "4mm",
        "c6v": "6mm",
        "td": "-43m",
        "43m": "-43m",
        "4bar": "-4",
        "overscript[4,_]": "-4",
        "\\!\\(\\*overscriptbox[\\(4\\),\\(_\\)]\\)": "-4",
        "6bar": "-6",
        "overscript[6,_]": "-6",
        "\\!\\(\\*overscriptbox[\\(6\\),\\(_\\)]\\)": "-6",
        "4bar2m": "-42m",
        "overscript[4,_]2m": "-42m",
        "\\!\\(\\*overscriptbox[\\(4\\),\\(_\\)]\\)2m": "-42m",
        "6barm2": "-6m2",
        "overscript[6,_]m2": "-6m2",
        "\\!\\(\\*overscriptbox[\\(6\\),\\(_\\)]\\)m2": "-6m2",
        "overscript[4,_]3m": "-43m",
        "\\!\\(\\*overscriptbox[\\(4\\),\\(_\\)]\\)3m": "-43m",
    }
    return aliases.get(pg, pg)


@dataclass(frozen=True)
class SymbolicSILayeredDerivation:
    """The single-interface reflected SHG as ♯SHAARP.si prints it: a CHAIN of named symbolic
    intermediates, each defined by its own equation, with the final 2 omega amplitudes written in
    terms of those names rather than one expanded blob.

    ``definitions`` is an ORDERED list of ``(name, expression)`` following the derivation:
    the transmitted omega eigenmodes (k and E per mode), then the 2 omega homogeneous bases.
    ``reflected_s`` / ``reflected_p`` are the final amplitudes IN TERMS OF the names.

    WHY (FA-1, -- "in my full analytical Mathematica, everything are symbolic"):
    his output is written with named quantities (k^(T,e,w), theta^(T,o,w), E_(Li)^(T,o,w),
    C_(Li)^(T,ee,2w)). Beyond matching that FORM, naming is what keeps the algebra tractable:
    substituting each eigenmode's explicit radical into the chain makes a BIAXIAL eps explode
    (measured >420 s, vs 0.40 s with named waves -- a >1000x difference), which is why the
    expanded formulation had to fall back to substituted numbers for KTP/LBO."""

    definitions: list
    reflected_s: Any
    reflected_p: Any
    raw: Any = None


def _name_wave_symbolic(wave, tag: str, definitions: list):
    """Replace a wave's k and E components by NAMED symbols, recording each definition.

    Exact zeros stay zero (naming them would add noise); every other component becomes
    ``kT_o_w_x`` / ``ET_o_w_L1`` etc. and its expression is appended to ``definitions``.

    NAMING: these are SYMPIFY-SAFE Python identifiers, not the typeset labels. ``kT_o_w_z`` is
    displayed as k^(T,o,ω)_z by the GUI's math typesetter (the same mechanism that shows
    ``n_2omega`` as n_2ω). They must round-trip through ``sympy.sympify`` because the copyable
    expressions are re-parsed by the Mathematica export -- decorated names like ``k^(T,o,w)_z``
    raise SympifyError (verified), which would break Copy-as-Mathematica for every case."""
    sp = _sympy()
    k = sp.Matrix(wave.k)
    e = sp.Matrix(wave.electric)
    named_k, named_e = [], []
    for comp, axis in zip(k, ("x", "y", "z")):
        if comp == 0:
            named_k.append(sp.Integer(0))
            continue
        s = sp.Symbol(f"k{tag}_{axis}")
        definitions.append((str(s), comp))
        named_k.append(s)
    for comp, axis in zip(e, ("L1", "L2", "L3")):
        if comp == 0:
            named_e.append(sp.Integer(0))
            continue
        s = sp.Symbol(f"E{tag}_{axis}")
        definitions.append((str(s), comp))
        named_e.append(s)
    if k[0] != 0 and k[2] != 0:  # his chain also prints the refraction angle of each mode
        definitions.append((f"theta{tag}", sp.atan2(k[0], k[2])))
    return SymbolicWave(k=sp.Matrix(named_k), electric=sp.Matrix(named_e), omega=wave.omega)


def _name_inhomogeneous_symbolic(inhom, definitions: list):
    """Name the bound (inhomogeneous) 2 omega field components ``C_(Li)^(T,..,2w)``.

    This is the SECOND naming layer, and it is what shortens the final amplitude to his printed
    length: the 2 omega boundary solve then sees the driven fields as symbols instead of the
    fully-expanded product of eigenmode radicals and d_ijk."""
    sp = _sympy()
    fields = {}
    for tag in ("ee", "oo", "eo"):
        f = getattr(inhom, tag)
        comps = []
        for comp, axis in zip(sp.Matrix(f.electric), ("L1", "L2", "L3")):
            if comp == 0:
                comps.append(sp.Integer(0))
                continue
            s = sp.Symbol(f"CT_{tag}_2w_{axis}")
            definitions.append((str(s), comp))
            comps.append(s)
        fields[tag] = SymbolicInhomogeneousField(
            label=f.label, k=f.k, electric=sp.Matrix(comps), polarization=f.polarization,
            omega=f.omega, residual=f.residual)
    return SymbolicSingleInterfaceInhomogeneousFields(**fields)


def solve_si_shg_layered_symbolic(
    *,
    eps_x_omega: Any, eps_y_omega: Any, eps_z_omega: Any,
    eps_x_2omega: Any, eps_y_2omega: Any, eps_z_2omega: Any,
    d_voigt_lab: Any,
    incident_theta_rad: Any,
    incident_polarization: str = "p",
    incident_jones: Any = None,
    omega: Any = 1,
    mu: Any = 1,
    eps0: Any = 1,
    simplify: bool = False,
    capture_stages: dict | None = None,
) -> SymbolicSILayeredDerivation:
    """Single-interface reflected SHG as a LAYERED chain of named symbolic intermediates.

    Same physics and same conventions as :func:`solve_biaxial_single_interface_shg_symbolic`
    (this reuses its exact basis construction), but the homogeneous waves are handed to the
    validated known-waves assembly as NAMED symbols, so the final amplitudes read as functions
    of ``k^(T,e,w)``, ``E^(T,o,w)_L2``, ... instead of expanded radicals -- his printed form,
    and the reason a biaxial eps stays tractable."""
    sp = _sympy()
    definitions: list = []

    # (1) omega: the two transmitted eigenmodes, from the SAME solver the expanded path uses
    _, _, transmitted_omega = solve_biaxial_linear_interface_symbolic(
        eps_x=eps_x_omega, eps_y=eps_y_omega, eps_z=eps_z_omega,
        incident_index=1, incident_theta_rad=incident_theta_rad,
        incident_polarization=incident_polarization, incident_jones=incident_jones,
        omega=omega, mu=mu, simplify=simplify,
    )
    kx = omega * sp.sin(incident_theta_rad)  # incident air (n=1)

    # (2) 2omega homogeneous bases (transmitted eigenmodes + the two reflected air waves)
    transmitted_basis_2omega = _biaxial_2omega_basis(
        eps_x_2omega, eps_y_2omega, eps_z_2omega, kx, omega, simplify)
    theta_2omega = sp.asin(kx / omega)
    reflected_basis_2omega = [
        make_isotropic_wave_symbolic(n=1, theta_rad=theta_2omega, polarization="s",
                                     omega=2 * omega, direction_sign_z=-1),
        make_isotropic_wave_symbolic(n=1, theta_rad=theta_2omega, polarization="p",
                                     omega=2 * omega, direction_sign_z=-1),
    ]

    # (3) NAME every homogeneous wave (the labels: T/R, mode, frequency)
    named_T_omega = [_name_wave_symbolic(w, f"T_{tag}_w", definitions)
                     for w, tag in zip(transmitted_omega, ("o", "e"))]
    named_T_2omega = [_name_wave_symbolic(w, f"T_{tag}_2w", definitions)
                      for w, tag in zip(transmitted_basis_2omega, ("o", "e"))]
    named_R_2omega = [_name_wave_symbolic(w, f"R_{tag}_2w", definitions)
                      for w, tag in zip(reflected_basis_2omega, ("s", "p"))]

    # (4) the validated known-waves assembly, now on NAMED waves (fast even for biaxial).
    # Inlined as its three published steps so the bound fields can be NAMED between (3) and (4)
    # -- byte-identical to solve_single_interface_shg_known_waves_symbolic apart from that naming.
    eps2_lab = sp.diag(eps_x_2omega, eps_y_2omega, eps_z_2omega)
    sources = single_interface_sources_symbolic(
        d_voigt_lab, named_T_omega[0], named_T_omega[1])
    inhomogeneous = solve_single_interface_inhomogeneous_fields_symbolic(
        eps2_lab, sources, mu=mu, eps0=eps0, simplify=simplify)
    named_inhom = _name_inhomogeneous_symbolic(inhomogeneous, definitions)
    boundary = solve_single_interface_shg_boundary_symbolic(
        reflected_basis=named_R_2omega, transmitted_basis=named_T_2omega,
        inhomogeneous=named_inhom, mu=mu, simplify=simplify)
    res = SymbolicSingleInterfaceSHGKnownWavesResult(
        sources=sources, inhomogeneous=inhomogeneous, boundary=boundary)
    if capture_stages is not None:
        # Same four published stages the expanded path exposes (eq 29-32 / 20-28 / 11-19 / 9-10) --
        # but written in the NAMED intermediates, so P_NL and the bound fields read as functions of
        # E^(T,o,w)_Li rather than expanded radicals. Plus the definition chain itself.
        capture_stages["omega_eigenmodes"] = named_T_omega
        capture_stages["sources"] = res.sources
        capture_stages["inhomogeneous"] = res.inhomogeneous
        capture_stages["boundary"] = res.boundary
        capture_stages["layered_definitions"] = definitions
    return SymbolicSILayeredDerivation(
        definitions=definitions,
        reflected_s=res.boundary.coefficients[0],  # the AMPLITUDES (the waves carry fields)
        reflected_p=res.boundary.coefficients[1],
        raw=res,
    )
