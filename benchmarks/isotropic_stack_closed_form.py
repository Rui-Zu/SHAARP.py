"""Textbook closed form for reflectance and transmittance of an isotropic multilayer.

PROVENANCE
----------
This is the Abeles characteristic-matrix formulation as given in

  * M. Born and E. Wolf, *Principles of Optics*, 7th ed., section 1.6.5 ("Reflection and
    transmission at a stratified medium"), Cambridge University Press (1999).
  * H. A. Macleod, *Thin-Film Optical Filters*, 4th ed., chapter 2 ("Basic theory"),
    CRC Press (2010) -- the optical-admittance form used verbatim below.

It exists so the benchmark carries an ANALYTIC third leg alongside the two external packages
(``tmm`` and ``inkstone``). Two independent codes that happened to share a convention error would
still agree with each other; a hand-written textbook expression would not. No SHAARP.py code is
imported here on purpose -- this module must stay an outside opinion.

Conventions
-----------
* Complex index ``n = n' + i k`` with implicit ``exp(-i omega t)``, so ``Im(n) > 0`` is loss.
  This matches SHAARP.py, ``tmm`` and ``inkstone``.
* ``n_list[0]`` is the semi-infinite incident medium, ``n_list[-1]`` the semi-infinite exit medium,
  and ``thickness`` carries ONLY the interior layers (so ``len(thickness) == len(n_list) - 2``).
* Thicknesses and the wavelength must share units (micrometres throughout this repo).
* Only ``R`` and ``T`` -- power quantities -- are returned. Amplitude ``r``/``t`` are deliberately
  NOT exposed: their p-polarization sign convention differs between textbooks and between packages,
  and nothing in this benchmark needs them. Power ratios are convention-free.

The transmittance returned here carries the obliquity factor by construction, because the tilted
optical admittance ``eta`` already contains ``cos(theta)``:

    eta_TE = n cos(theta)        eta_TM = n / cos(theta)

which is the whole point of the comparison this module supports.
"""

from __future__ import annotations

import numpy as np

__all__ = ["stack_rt", "single_interface_rt", "tilted_admittance", "layer_cos_theta"]

_POLARIZATIONS = ("s", "p")


def layer_cos_theta(n_layer: complex, n_incident: complex, theta_incident_rad: float) -> complex:
    """``cos(theta)`` inside a layer, from Snell's law, on the physical branch.

    The branch is fixed by requiring a forward-decaying wave: ``Im(n cos(theta)) >= 0`` for the
    ``exp(-i omega t)`` convention. ``numpy``'s principal square root already delivers this for
    every case exercised here, but the sign is asserted rather than assumed.
    """
    sin_t = n_incident * np.sin(theta_incident_rad) / n_layer
    cos_t = np.sqrt(1.0 - sin_t * sin_t + 0j)
    if np.imag(n_layer * cos_t) < 0.0:
        cos_t = -cos_t
    return cos_t


def tilted_admittance(n_layer: complex, cos_theta: complex, polarization: str) -> complex:
    """Macleod's tilted optical admittance: ``n cos(theta)`` for TE/s, ``n / cos(theta)`` for TM/p."""
    if polarization not in _POLARIZATIONS:
        raise ValueError("polarization must be 's' or 'p', got %r" % (polarization,))
    return n_layer * cos_theta if polarization == "s" else n_layer / cos_theta


def stack_rt(n_list, thickness, theta_incident_rad: float, wavelength: float, polarization: str):
    """Return ``(R, T)`` for an isotropic stack via the characteristic-matrix method.

    Parameters mirror the module docstring. Returns real power ratios; for a lossless stack
    ``R + T == 1`` to machine precision, which the benchmark asserts directly.
    """
    n_arr = np.asarray(n_list, dtype=complex)
    d_arr = np.asarray(thickness, dtype=float)
    if n_arr.ndim != 1 or n_arr.size < 2:
        raise ValueError("n_list must be a 1-D sequence with at least two entries.")
    if d_arr.size != n_arr.size - 2:
        raise ValueError(
            "thickness must carry only interior layers: expected %d entries, got %d."
            % (n_arr.size - 2, d_arr.size)
        )
    if polarization not in _POLARIZATIONS:
        raise ValueError("polarization must be 's' or 'p', got %r" % (polarization,))

    n_inc = n_arr[0]
    eta_inc = tilted_admittance(n_inc, np.cos(theta_incident_rad) + 0j, polarization)
    cos_exit = layer_cos_theta(n_arr[-1], n_inc, theta_incident_rad)
    eta_exit = tilted_admittance(n_arr[-1], cos_exit, polarization)

    # Characteristic matrix of the interior assembly, ordered top layer first.
    matrix = np.eye(2, dtype=complex)
    for n_layer, d_layer in zip(n_arr[1:-1], d_arr):
        cos_l = layer_cos_theta(n_layer, n_inc, theta_incident_rad)
        eta_l = tilted_admittance(n_layer, cos_l, polarization)
        delta = 2.0 * np.pi * n_layer * cos_l * d_layer / wavelength
        # NOTE the MINUS sign on the off-diagonal terms. Macleod and Born & Wolf write this matrix
        # with +i, because they use the exp(+i omega t) / n = n' - i k convention. This module (and
        # SHAARP.py, tmm and inkstone) use exp(-i omega t) / n = n' + i k, whose characteristic
        # matrix is the complex conjugate off-diagonal. Getting this wrong is invisible for
        # lossless stacks -- every real-index case agrees to 1e-16 either way -- and shows up only
        # once a layer absorbs: with +i, a 13.9 nm gold film gave R + T = 1.053, i.e. more light out
        # than in. Verified against tmm for the Au case at 0/30/60/80 deg, both polarizations.
        layer_matrix = np.array(
            [
                [np.cos(delta), -1j * np.sin(delta) / eta_l],
                [-1j * eta_l * np.sin(delta), np.cos(delta)],
            ],
            dtype=complex,
        )
        matrix = matrix @ layer_matrix

    b_field, c_field = matrix @ np.array([1.0 + 0j, eta_exit], dtype=complex)

    denom = eta_inc * b_field + c_field
    reflectance = abs((eta_inc * b_field - c_field) / denom) ** 2
    transmittance = 4.0 * np.real(eta_inc) * np.real(eta_exit) / abs(denom) ** 2
    return float(reflectance), float(transmittance)


def single_interface_rt(n_incident: complex, n_exit: complex, theta_incident_rad: float, polarization: str):
    """``(R, T)`` for one bare interface, straight from the Fresnel equations.

    Independent of :func:`stack_rt` -- written out longhand rather than calling it with an empty
    layer list -- so that the zero-layer limit of the matrix method is itself checked rather than
    assumed.
    """
    if polarization not in _POLARIZATIONS:
        raise ValueError("polarization must be 's' or 'p', got %r" % (polarization,))
    n_i, n_t = complex(n_incident), complex(n_exit)
    cos_i = np.cos(theta_incident_rad) + 0j
    cos_t = layer_cos_theta(n_t, n_i, theta_incident_rad)

    if polarization == "s":
        r_amp = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
        t_amp = 2.0 * n_i * cos_i / (n_i * cos_i + n_t * cos_t)
    else:
        r_amp = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)
        t_amp = 2.0 * n_i * cos_i / (n_t * cos_i + n_i * cos_t)

    reflectance = abs(r_amp) ** 2
    # The obliquity factor -- the quantity the SHAARP.py multilayer Fresnel path was missing.
    # s and p take the conjugation on opposite factors; for real indices the two coincide, and the
    # difference only matters when a medium absorbs.
    if polarization == "s":
        flux_factor = np.real(n_t * cos_t) / np.real(n_i * cos_i)
    else:
        flux_factor = np.real(np.conjugate(n_t) * cos_t) / np.real(np.conjugate(n_i) * cos_i)
    transmittance = float(flux_factor) * abs(t_amp) ** 2
    return float(reflectance), float(transmittance)
