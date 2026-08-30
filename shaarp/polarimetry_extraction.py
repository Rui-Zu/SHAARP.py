"""Extract the SHG coefficient tensor d_ijk from a (reflected) SHG polarimetry scan, using
the closed-form full-analytical polarimetry as the fitting model -- the d-extraction the
analytical capability exists for.

Because the reflected SHG field is LINEAR in the d-components, the closed form is
``E_channel(phi) = sum_k d_k * B_k(phi)`` with basis ``B_k = d E / d d_k`` (read with
``sympy.diff``, exact for the degree-1 form). Stacking measurement geometries (incidence
angle x sample azimuth) and input-polarization angles phi gives an over-determined linear
system that recovers d:

  * ``method="field"`` -- if the complex field amplitude is measured, solve ``M d = y``
    directly (linear least squares). Unique up to identifiability.
  * ``method="intensity"`` -- a real polarimeter records ``I(phi) = |E(phi)|^2``, which is
    linear in the Gram matrix ``D_kl = d_k d_l``; solve for D then rank-1 factor ``D = d d^T``
    to recover d UP TO AN OVERALL SIGN (intensity is invariant under ``d -> -d``). The
    recovered D being rank-1 is a built-in data-consistency check.

Identifiability is reported honestly: a single geometry is rank-deficient (one scan cannot
determine the whole tensor); a multi-angle/azimuth scan restores full rank. ``identifiable``
is True iff the design rank equals the number of requested components.

NOTE on geometry/consistency: a sample azimuth about the surface normal rotates the d-tensor
AND, physically, eps too. The ``basis="symbolic"`` path rotates only d, so it is exact for an
azimuth only when eps is invariant under it (isotropic, or uniaxial with the optic axis along
the normal); for a general biaxial azimuth use ``basis="numeric"`` (which rotates BOTH eps and
d, via the same Rz convention) or supply theta-varying geometries. Either way the user's
``measure`` must rotate the sample consistently (eps and d by the same Rz(azimuth)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .symbolic import _sympy, solve_si_shg_full_analytical_symbolic


@dataclass(frozen=True)
class DExtractionResult:
    """Result of a d-tensor extraction from an SHG polarimetry scan."""

    component_positions: list  # list of Voigt (row, col) positions that were solved for
    values: np.ndarray         # recovered d values aligned with component_positions
    d_voigt: np.ndarray        # 3x6 Voigt tensor with the recovered values placed
    rank: int                  # numerical rank of the design (field) / Gram (intensity) system
    n_components: int          # number of requested components
    identifiable: bool         # rank == n_components (the requested set is fully determined)
    condition_number: float    # conditioning of the solved linear system
    residual: float            # max |model - data| at the recovered solution
    method: str                # "field" or "intensity"
    sign_ambiguous: bool       # True for "intensity" (d determined only up to overall sign)
    relative_residual: float = float("nan")  # residual / max|data| -- scale-free fit quality.
    # For noise-free synthetic data this is ~1e-12 when the model is consistent with the data;
    # a LARGE value on a full-rank (identifiable) fit signals a model/units mismatch -- most
    # commonly that `measure` used a different (mu, eps0) than the extractor (see the contract
    # in extract_si_d_voigt/extract_ml_film_d_voigt). Always check this before trusting `values`.


def _basis_functions(d_positions, geometry, phi_symbol, eps_omega_principal, eps_2omega_principal, d_symbols, mu, eps0):
    """Closed-form reflected-SHG basis functions B_k(phi) = dE/dd_k for one geometry."""
    sp = _sympy()
    d_mat = sp.zeros(3, 6)
    for (m, ell), s in zip(d_positions, d_symbols):
        d_mat[m, ell] = s
    theta, azimuth = geometry
    sol = solve_si_shg_full_analytical_symbolic(
        eps_x_omega=eps_omega_principal[0], eps_y_omega=eps_omega_principal[1], eps_z_omega=eps_omega_principal[2],
        eps_x_2omega=eps_2omega_principal[0], eps_y_2omega=eps_2omega_principal[1], eps_z_2omega=eps_2omega_principal[2],
        d_voigt_lab=d_mat, incident_theta_rad=theta, phi_symbol=phi_symbol,
        sample_azimuth_symbol=(sp.Float(azimuth) if azimuth else None),
        omega=1, mu=mu, eps0=eps0, simplify=False,
    )
    e_t = sol.transmitted_field  # total transmitted SHG field (3-vector), linear in d
    basis = {
        "s": [sp.diff(sol.reflected_s, s) for s in d_symbols],
        "p": [sp.diff(sol.reflected_p, s) for s in d_symbols],
        # transmitted (Maker) channel: one basis row per field component (x, y, z)
        "t": [[sp.diff(e_t[i], s) for s in d_symbols] for i in range(3)],
    }
    return basis


def _extract_numeric_basis(eps_w, eps_2w, d_positions, geometries, phi_values, measure,
                           method, observable, channels, mu, eps0):
    """Numeric-basis d-extraction: SHG is linear in d, so the design column for Voigt position
    p is the NUMERIC single-interface SHG response to a UNIT d at p (rotated by the sample
    azimuth). Exact for any crystal (incl. near-degenerate biaxial), no symbolic engine."""
    from .shg import solve_single_interface_shg
    from .tensors import rotate_d_voigt_crystal_to_lab, rotate_rank2_crystal_to_lab

    ew = np.diag([complex(x) for x in eps_w])
    e2 = np.diag([complex(x) for x in eps_2w])

    def _rz(az):
        c, s = math.cos(az), math.sin(az)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def unit_response(pos, theta, az, phi):
        du = np.zeros((3, 6))
        du[pos] = 1.0
        if not az:
            d_rot, ew_use, e2_use = du, ew, e2
        else:
            # A sample azimuth about the surface normal rotates BOTH the d-tensor and the
            # dielectric tensors (eps is invariant only for isotropic / uniaxial-z; for a
            # biaxial crystal eps_x != eps_y, so eps MUST rotate too). Use the SAME rotation
            # convention as the d rotation (rotate_rank2_crystal_to_lab == Rz^T. eps. Rz) so a
            # user whose `measure` rotates the sample by Rz(az) is reproduced exactly.
            rz = _rz(az)
            d_rot = np.real(np.asarray(rotate_d_voigt_crystal_to_lab(du, rz), dtype=complex))
            ew_use = rotate_rank2_crystal_to_lab(ew, rz)
            e2_use = rotate_rank2_crystal_to_lab(e2, rz)
        r = solve_single_interface_shg(
            ew_use, e2_use, d_rot, incident_index_omega=1.0, incident_index_2omega=1.0,
            incident_theta_rad=theta, incident_jones=(math.sin(phi), math.cos(phi)),
            omega=1.0, mu=mu, eps0=eps0,
        )
        if observable == "reflected":
            c = np.asarray(r.coefficients)
            return {"s": complex(c[0]), "p": complex(c[1])}
        return np.asarray(r.transmitted_fast_2omega.electric) + np.asarray(r.transmitted_slow_2omega.electric)

    rows, e_meas = [], []
    for theta, az in geometries:
        for v in phi_values:
            responses = [unit_response(pos, theta, az, v) for pos in d_positions]
            measured = measure(theta, az, v)
            if observable == "reflected":
                fields = {"s": complex(measured[0]), "p": complex(measured[1])}
                for ch in channels:
                    rows.append([resp[ch] for resp in responses])
                    e_meas.append(fields[ch])
            else:
                for i in range(3):
                    rows.append([complex(resp[i]) for resp in responses])
                    e_meas.append(complex(measured[i]))
    return _recover_d_from_design(np.asarray(rows, dtype=complex), np.asarray(e_meas, dtype=complex), d_positions, method)


def extract_si_d_voigt(
    *,
    eps_omega_principal,
    eps_2omega_principal,
    d_positions,
    geometries,
    phi_values,
    measure,
    method: str = "field",
    observable: str = "reflected",
    basis: str = "symbolic",
    channels=("s", "p"),
    mu=1,
    eps0=1,
):
    """Recover the d-tensor components at ``d_positions`` (Voigt (row, col) tuples) from an
    SHG polarimetry scan.

    .. important::
       **(mu, eps0) MUST match the values your ``measure`` callback used.** d-extraction solves a
       linear system ``M(mu,eps0) . d = y`` where the design ``M`` is built with *these* ``mu,
       eps0`` and ``y`` comes from your ``measure``; if the two disagree the recovered ``d`` is
       silently rescaled/corrupted while the *absolute* residual stays modest. Both clean units
       (``mu=eps0=1``, the default and the convention used throughout this package's examples)
       and physical SI constants recover to machine precision -- as long as BOTH sides use the
       same pair. Check ``result.relative_residual`` (~1e-12 for a consistent noise-free fit; a
       large value on a full-rank/identifiable fit means a units/model mismatch).

    ``geometries`` is a list of ``(incidence_theta_rad, sample_azimuth_rad)``; ``phi_values``
    the input-polarization angles sampled at each geometry.

    ``basis`` selects how the linear-in-d model (design matrix) is built:
      * ``"symbolic"`` (default) -- the closed-form ``solve_si_shg_full_analytical_symbolic``
        differentiated w.r.t. each d component. Exact for isotropic, uniaxial, and
        well-separated biaxial; ILL-CONDITIONED for NEAR-degenerate biaxial (eps_x ~= eps_y,
        e.g. KTP) in the reflected-p channel.
      * ``"numeric"`` -- the basis is the NUMERIC ``solve_single_interface_shg`` response to a
        unit d at each position (SHG is linear in d). Exact for ANY crystal, including
        near-degenerate biaxial (KTP), and -- unlike the symbolic basis -- it rotates BOTH eps
        and d under a sample azimuth, so it also handles a GENERAL ROTATED BIAXIAL azimuth scan
        (the case the symbolic closed form cannot do). Requires ``measure`` outputs consistent
        with the numeric solver (the default in this package), with the sample rotated by the
        same Rz(azimuth) for both eps and d.

    ``observable="reflected"`` (default): ``measure(theta, azimuth, phi)`` returns the complex
    reflected SHG ``(E_s, E_p)`` and the requested ``channels`` are used.
    ``observable="transmitted"`` (the Maker-fringe geometry): ``measure(...)`` returns the
    complex total transmitted SHG field as a length-3 vector ``(E_x, E_y, E_z)`` (e.g.
    ``transmitted_fast_2omega.electric + transmitted_slow_2omega.electric`` from the numeric
    solver); all three components are used (so each scan point gives 3 observables, often
    making the tensor identifiable with fewer geometries).

    (For an intensity-only experiment return any complex numbers with the measured magnitudes
    -- ``method="intensity"`` uses only ``|.|^2``.) Returns a :class:`DExtractionResult`. For
    ``method="field"`` d is unique up to identifiability; for ``method="intensity"`` up to an
    overall sign.
    """
    if method not in {"field", "intensity"}:
        raise ValueError("method must be 'field' or 'intensity'.")
    if observable not in {"reflected", "transmitted"}:
        raise ValueError("observable must be 'reflected' or 'transmitted'.")
    if basis not in {"symbolic", "numeric"}:
        raise ValueError("basis must be 'symbolic' or 'numeric'.")
    if basis == "numeric":
        return _extract_numeric_basis(
            eps_omega_principal, eps_2omega_principal, d_positions, geometries,
            phi_values, measure, method, observable, channels, mu, eps0,
        )
    sp = _sympy()
    n = len(d_positions)
    d_symbols = sp.symbols(f"_d0:{n}")
    phi = sp.Symbol("_phi", real=True)

    # complex field design matrix M (rows: geometry x phi x channel/component, cols: d comps)
    rows, e_meas = [], []
    for geometry in geometries:
        basis = _basis_functions(
            d_positions, geometry, phi, eps_omega_principal, eps_2omega_principal, d_symbols, mu, eps0
        )
        for v in phi_values:
            measured = measure(geometry[0], geometry[1], v)
            if observable == "reflected":
                fields = {"s": complex(measured[0]), "p": complex(measured[1])}
                for ch in channels:
                    rows.append([complex(b.subs(phi, v).evalf()) for b in basis[ch]])
                    e_meas.append(fields[ch])
            else:  # transmitted: measured is the 3-vector total transmitted field
                for i in range(3):
                    rows.append([complex(b.subs(phi, v).evalf()) for b in basis["t"][i]])
                    e_meas.append(complex(measured[i]))
    M = np.asarray(rows, dtype=complex)
    e_meas = np.asarray(e_meas, dtype=complex)
    return _recover_d_from_design(M, e_meas, d_positions, method)


def _recover_d_from_design(M, e_meas, d_positions, method):
    """Solve the linear d-extraction system from a complex field design matrix M (rows:
    measurements, cols: d-components) and complex field measurements e_meas. Shared by the
    SI and ML extractors. ``method="field"`` solves M d = e_meas directly; ``method="intensity"``
    solves the Gram system (I = |E|^2 is linear in D_kl = d_k d_l) then rank-1 factors."""
    n = len(d_positions)
    d_voigt = np.zeros((3, 6))
    if method == "field":
        values, _, rank, sv = np.linalg.lstsq(M, e_meas, rcond=None)
        cond = float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 0 else float("inf")
        residual = float(np.max(np.abs(M @ values - e_meas)))
        data_scale = float(np.max(np.abs(e_meas)))
        values = np.asarray(values)
        sign_ambiguous = False
        identifiable = int(rank) == n
    else:  # intensity: I = |E|^2 is linear in the Gram matrix D_kl = d_k d_l
        i_meas = np.abs(e_meas) ** 2
        pairs = [(k, l) for k in range(n) for l in range(k, n)]
        W = np.empty((M.shape[0], len(pairs)))
        for i in range(M.shape[0]):
            b = M[i]
            for j, (k, l) in enumerate(pairs):
                W[i, j] = (abs(b[k]) ** 2) if k == l else 2.0 * np.real(np.conj(b[k]) * b[l])
        gram_vec, _, rank, sv = np.linalg.lstsq(W, i_meas, rcond=None)
        cond = float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 0 else float("inf")
        gram = np.zeros((n, n))
        for (k, l), val in zip(pairs, gram_vec):
            gram[k, l] = gram[l, k] = val
        eig, vecs = np.linalg.eigh(gram)
        values = float(np.sqrt(max(eig[-1], 0.0))) * vecs[:, -1]
        residual = float(np.max(np.abs(W @ gram_vec - i_meas)))
        data_scale = float(np.max(np.abs(i_meas)))
        rank = int(np.linalg.matrix_rank(W, tol=1e-9 * (sv[0] if len(sv) else 1.0)))
        sign_ambiguous = True
        identifiable = int(rank) == len(pairs)

    relative_residual = residual / data_scale if data_scale > 0 else float("inf")
    real_values = np.real_if_close(values, tol=1e6)
    for (m, ell), val in zip(d_positions, real_values):
        d_voigt[m, ell] = np.real(val)
    return DExtractionResult(
        component_positions=list(d_positions),
        values=np.asarray(real_values),
        d_voigt=d_voigt,
        rank=int(rank),
        n_components=n,
        identifiable=bool(identifiable),
        condition_number=cond,
        residual=residual,
        method=method,
        sign_ambiguous=sign_ambiguous,
        relative_residual=relative_residual,
    )


def _rotated_d_symbolic(d_mat, azimuth):
    """Rotate a symbolic Voigt d-tensor about the surface normal z by `azimuth`, in the
    convention that matches the numeric ``rotate_d_voigt_crystal_to_lab(d, Rz(azimuth))``
    (so a user whose `measure` rotates the sample by Rz(azimuth) is consistent)."""
    from .symbolic import rotate_d_voigt_symbolic

    sp = _sympy()
    if not azimuth:
        return d_mat
    c, s = sp.cos(sp.Float(azimuth)), sp.sin(sp.Float(azimuth))
    rot_zt = sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # R_z^T (matches crystal_to_lab(R_z))
    return rotate_d_voigt_symbolic(d_mat, rot_zt, simplify=False)


def extract_ml_film_d_voigt(
    *,
    film_index_omega,
    substrate_index_omega,
    film_index_2omega,
    substrate_index_2omega,
    thickness,
    d_positions,
    geometries,
    phi_values,
    measure,
    method: str = "field",
    observable: str = "reflected",
    channels=("s", "p"),
    mu=1.0,
    eps0=1.0,
):
    """Recover a single nonlinear FILM's d-tensor from an SHG polarimetry scan, using the ML
    partial-analytical closed form (``solve_single_film_shg_symbolic_polarimetry``) as the
    model. Analogue of :func:`extract_si_d_voigt` for SHAARP.ml thin films.

    .. important::
       **(mu, eps0) MUST match the values your ``measure`` callback used** (same contract as
       :func:`extract_si_d_voigt`): a mismatch silently rescales/corrupts the recovered ``d``.
       Default is clean units ``mu=eps0=1`` (now consistent with :func:`extract_si_d_voigt`);
       physical SI constants also work if used on BOTH sides. Check ``result.relative_residual``.

    The film and substrate are given by scalar refractive indices at omega / 2omega (the film
    eps is taken ISOTROPIC, so a sample azimuth about the normal rotates only the d-tensor --
    consistent). ``thickness`` is the (known) film thickness. ``geometries`` is a list of
    ``(incidence_theta_rad, sample_azimuth_rad)``.

    ``observable="reflected"`` (default): ``measure(theta, azimuth, phi)`` returns the complex
    reflected SHG ``(E_s, E_p)``. ``observable="transmitted"`` (the standard thin-film
    **Maker-fringe** geometry): ``measure(...)`` returns the complex SHG transmitted into the
    substrate ``(T_s, T_p)`` (``solve_multilayer_shg_from_tensors_jones`` ``.shg.coefficients``
    indices 6 and 7). Returns a :class:`DExtractionResult` (field: exact up to identifiability;
    intensity: up to overall sign).
    """
    import numpy as _np

    from .multilayer_basis import build_multilayer_2omega_basis, build_multilayer_omega_basis
    from .multilayer_shg_symbolic import solve_single_film_shg_symbolic_polarimetry

    if method not in {"field", "intensity"}:
        raise ValueError("method must be 'field' or 'intensity'.")
    if observable not in {"reflected", "transmitted"}:
        raise ValueError("observable must be 'reflected' or 'transmitted'.")
    # Default is clean units (1,1) -- the package-wide convention, now consistent with
    # extract_si_d_voigt; (mu, eps0) must MATCH the values `measure` used (see the contract note).
    # (Previously defaulted to MU0/EPS0, inconsistently with the SI extractor -- a mismatch footgun.)
    sp = _sympy()
    n = len(d_positions)
    d_symbols = sp.symbols(f"_d0:{n}")
    d_mat = sp.zeros(3, 6)
    for (m, ell), s in zip(d_positions, d_symbols):
        d_mat[m, ell] = s
    phi = sp.Symbol("_phi", real=True)
    h = sp.Symbol("_h", positive=True)

    eps_w = [_np.diag([film_index_omega**2] * 3).astype(complex)]
    eps_w_sub = _np.diag([substrate_index_omega**2] * 3).astype(complex)
    eps_2w = [_np.diag([film_index_2omega**2] * 3).astype(complex)]
    eps_2w_sub = _np.diag([substrate_index_2omega**2] * 3).astype(complex)

    rows, e_meas = [], []
    for theta, azimuth in geometries:
        d_rot = _rotated_d_symbolic(d_mat, azimuth)
        common = dict(incident_index=1.0, incident_theta_rad=theta, layer_epsilon_lab=eps_w,
                      substrate_epsilon_lab=eps_w_sub, omega=1.0)
        bs = build_multilayer_omega_basis(incident_polarization="s", **common)
        bp = build_multilayer_omega_basis(incident_polarization="p", **common)
        b2 = build_multilayer_2omega_basis(
            top_index_2omega=1.0, tangential_index_omega=1.0, incident_theta_rad=theta,
            layer_epsilon_2omega_lab=eps_2w, substrate_epsilon_2omega_lab=eps_2w_sub, omega_2=2.0,
        )
        sol = solve_single_film_shg_symbolic_polarimetry(
            omega_basis_s=bs, omega_basis_p=bp, twoomega_basis=b2, d_voigt_symbolic=d_rot,
            eps_2omega_lab=eps_2w[0], thickness_symbol=h, phi_symbol=phi, mu=mu, eps0=eps0,
        )
        src_s, src_p = (sol.substrate_s, sol.substrate_p) if observable == "transmitted" else (sol.reflected_s, sol.reflected_p)
        basis = {
            "s": [sp.diff(src_s, s).subs(h, thickness) for s in d_symbols],
            "p": [sp.diff(src_p, s).subs(h, thickness) for s in d_symbols],
        }
        for v in phi_values:
            es, ep = measure(theta, azimuth, v)
            fields = {"s": complex(es), "p": complex(ep)}
            for ch in channels:
                rows.append([complex(b.subs(phi, v).evalf()) for b in basis[ch]])
                e_meas.append(fields[ch])
    M = np.asarray(rows, dtype=complex)
    e_meas = np.asarray(e_meas, dtype=complex)
    return _recover_d_from_design(M, e_meas, d_positions, method)


__all__ = ["DExtractionResult", "extract_si_d_voigt", "extract_ml_film_d_voigt"]
