from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np


ArrayLike = Any


def _canonical_shaarp_point_group_label(point_group: str) -> str:
    pg = point_group.strip()
    compact = pg.lower().replace(" ", "")
    aliases = {
        "overscript[4,_]": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)",
        r"\!\(\*overscriptbox[\(4\),\(_\)]\)": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)",
        "overscript[4,_]2m": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)2m",
        r"\!\(\*overscriptbox[\(4\),\(_\)]\)2m": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)2m",
        "overscript[4,_]3m": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)3m",
        r"\!\(\*overscriptbox[\(4\),\(_\)]\)3m": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)3m",
        "overscript[6,_]": r"\!\(\*OverscriptBox[\(6\), \(_\)]\)",
        r"\!\(\*overscriptbox[\(6\),\(_\)]\)": r"\!\(\*OverscriptBox[\(6\), \(_\)]\)",
        "overscript[6,_]m2": r"\!\(\*OverscriptBox[\(6\), \(_\)]\)m2",
        r"\!\(\*overscriptbox[\(6\),\(_\)]\)m2": r"\!\(\*OverscriptBox[\(6\), \(_\)]\)m2",
        r"\[infinity]": "3",
        r"\[infinity]m": "3m",
        r"\[infinity]2": "32",
        # ASCII barred-group spellings (the package-wide convention, e.g. "-43m")
        "-4": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)",
        "-42m": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)2m",
        "-43m": r"\!\(\*OverscriptBox[\(4\), \(_\)]\)3m",
        "-6": r"\!\(\*OverscriptBox[\(6\), \(_\)]\)",
        "-6m2": r"\!\(\*OverscriptBox[\(6\), \(_\)]\)m2",
        "-3": r"\!\(\*OverscriptBox[\(3\), \(_\)]\)",
        "-3m": r"\!\(\*OverscriptBox[\(3\), \(_\)]\)m",
    }
    return aliases.get(compact, pg)


class PolarizerMode(str, Enum):
    ROTATE = "rotate"
    FIXED = "fixed"


class AnalyzerMode(str, Enum):
    ROTATE = "rotate"
    FIXED = "fixed"


@dataclass(frozen=True)
class Polarimetry:
    """Polarimetry settings (measurement geometry) from the SHAARP input panel.

    Public angles are in **degrees**, mirroring the Mathematica notebooks; ``phi_deg`` and ``psi_deg``
    may be scalars or NumPy arrays (e.g. ``np.linspace(0, 360, 361)`` to sweep a polar plot).

    Attributes:
        theta_deg: incidence angle from the surface normal (0 = normal incidence).
        phi_deg: input polarization angle (0 = p, 90 = s); the swept variable in *Rotate Polarizer*.
        psi_deg: analyzer angle; used when the analyzer is fixed.
        ellipticity_deg: incident-field phase between components (the ``Delta-delta`` ellipticity).
        polarizer_mode: :class:`PolarizerMode` -- ``ROTATE`` (sweep phi) or ``FIX`` (hold phi, sweep psi).
        analyzer_mode: :class:`AnalyzerMode` -- ``FIXED`` analyzer at psi, or rotating (parallel +
            perpendicular channels).
    """

    theta_deg: ArrayLike = 45.0
    phi_deg: ArrayLike = 0.0
    psi_deg: ArrayLike = 0.0
    ellipticity_deg: ArrayLike = 0.0
    polarizer_mode: PolarizerMode = PolarizerMode.ROTATE
    analyzer_mode: AnalyzerMode = AnalyzerMode.FIXED

    def radians(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.deg2rad(np.asarray(x, dtype=float))
            for x in (self.theta_deg, self.phi_deg, self.psi_deg, self.ellipticity_deg)
        )


@dataclass(frozen=True)
class CrystalStructure:
    point_group: str = "1"
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha_deg: float = 90.0
    beta_deg: float = 90.0
    gamma_deg: float = 90.0

    def direct_lattice_vectors(self) -> np.ndarray:
        """Return direct lattice vectors as columns in an orthonormal frame."""

        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            raise ValueError("Lattice constants a, b, and c must be positive.")
        alpha, beta, gamma = np.deg2rad([self.alpha_deg, self.beta_deg, self.gamma_deg])
        sin_gamma = np.sin(gamma)
        if abs(sin_gamma) < 1e-12:
            raise ValueError("gamma angle gives a degenerate lattice.")
        avec = np.array([self.a, 0.0, 0.0])
        bvec = np.array([self.b * np.cos(gamma), self.b * sin_gamma, 0.0])
        cx = self.c * np.cos(beta)
        cy = self.c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
        cz_sq = self.c**2 - cx**2 - cy**2
        if cz_sq <= 1e-12:
            raise ValueError("Lattice constants/angles give a degenerate lattice.")
        cvec = np.array([cx, cy, np.sqrt(cz_sq)])
        return np.column_stack([avec, bvec, cvec])

    def reciprocal_lattice_vectors(self) -> np.ndarray:
        """Return reciprocal lattice vectors as columns, without the 2*pi factor."""

        direct = self.direct_lattice_vectors()
        volume = float(np.dot(direct[:, 0], np.cross(direct[:, 1], direct[:, 2])))
        if abs(volume) < 1e-12:
            raise ValueError("Lattice volume is zero.")
        b1 = np.cross(direct[:, 1], direct[:, 2]) / volume
        b2 = np.cross(direct[:, 2], direct[:, 0]) / volume
        b3 = np.cross(direct[:, 0], direct[:, 1]) / volume
        return np.column_stack([b1, b2, b3])

    def miller_plane_normal(self, hkl: ArrayLike) -> np.ndarray:
        """Return real-space normal direction for a reciprocal-space Miller plane."""

        hkl_vector = _real_vector3(hkl, "hkl")
        if np.linalg.norm(hkl_vector) == 0:
            raise ValueError("hkl must be non-zero.")
        normal = self.reciprocal_lattice_vectors() @ hkl_vector
        return normal / np.linalg.norm(normal)

    def direct_lattice_direction(self, uvw: ArrayLike) -> np.ndarray:
        """Return real-space direction for direct-lattice indices `[u v w]`."""

        uvw_vector = _real_vector3(uvw, "uvw")
        if np.linalg.norm(uvw_vector) == 0:
            raise ValueError("uvw must be non-zero.")
        direction = self.direct_lattice_vectors() @ uvw_vector
        return direction / np.linalg.norm(direction)

    def shaarp_physics_axes(self, point_group: str | None = None) -> np.ndarray:
        """Return SHAARP's point-group-dependent crystal physics axes.

        The Mathematica `hklConvert` routine first builds direct lattice
        vectors, then chooses Z1/Z2/Z3 from the lattice according to the point
        group before solving the Miller-orientation transform.
        """

        pg = _canonical_shaarp_point_group_label(point_group or self.point_group)
        direct = self.direct_lattice_vectors()
        avec = direct[:, 0]
        bvec = direct[:, 1]
        cvec = direct[:, 2]

        def unit(vector: np.ndarray) -> np.ndarray:
            norm = np.linalg.norm(vector)
            if norm < 1e-12:
                raise ValueError("Point-group physics axis construction is degenerate.")
            return vector / norm

        if pg in {"1", r"\!\(\*OverscriptBox[\(1\), \(_\)]\)"}:
            z3 = unit(cvec)
            z2 = unit(np.cross(avec, cvec))
            z1 = unit(np.cross(z2, z3))
        elif pg in {"2", "m", "2/m"}:
            z2 = unit(bvec)
            z3 = unit(cvec)
            z1 = unit(np.cross(z2, z3))
        elif pg in {"mm2", "222", "mmm", "4", r"\!\(\*OverscriptBox[\(4\), \(_\)]\)", "4mm", "422", r"\!\(\*OverscriptBox[\(4\), \(_\)]\)2m", "4/m", "4/mmm"}:
            z1 = unit(avec)
            z2 = unit(bvec)
            z3 = unit(np.cross(z1, z2))
        elif pg in {"3", "32", "3m", r"\!\(\*OverscriptBox[\(3\), \(_\)]\)", r"\!\(\*OverscriptBox[\(3\), \(_\)]\)m", "∞2", "∞m", "∞", "6", r"\!\(\*OverscriptBox[\(6\), \(_\)]\)", "6mm", "622", r"\!\(\*OverscriptBox[\(6\), \(_\)]\)m2", "6/m", "6/mmm", "∞/mm", "∞/m", r"\!\(\*OverscriptBox[\(4\), \(_\)]\)3m", "23", "∞∞m", "∞∞", "m3", "432", "m3m"}:
            z3 = unit(cvec)
            z1 = unit(avec)
            z2 = unit(np.cross(z3, z1))
        else:
            raise ValueError(f"Unsupported SHAARP point group for hkl conversion: {pg!r}")
        return np.vstack([z1, z2, z3])


@dataclass(frozen=True)
class CrystalOrientation:
    """Crystal physics axes expressed in lab coordinates.

    Rows correspond to Z1, Z2, Z3. The lab frame follows SHAARP:
    L1 is in the plane of incidence, L2 is perpendicular to it, and L3 is
    surface normal.
    """

    z_axes_in_lab: ArrayLike = field(
        default_factory=lambda: np.eye(3, dtype=float)
    )

    @classmethod
    def from_cubic_miller_surface(
        cls,
        hkl: ArrayLike,
        *,
        in_plane_hint: ArrayLike = (1.0, 0.0, 0.0),
    ) -> "CrystalOrientation":
        """Build a real orientation with cubic `[h k l]` aligned to lab L3.

        This is intentionally scoped to orthonormal/cubic crystal axes. General
        Miller-index handling for non-cubic metrics needs the lattice geometry.
        """

        normal = _real_vector3(hkl, "hkl")
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            raise ValueError("hkl must be non-zero.")
        lab_l3_in_crystal = normal / normal_norm

        hint = _real_vector3(in_plane_hint, "in_plane_hint")
        lab_l1_in_crystal = hint - np.dot(hint, lab_l3_in_crystal) * lab_l3_in_crystal
        if np.linalg.norm(lab_l1_in_crystal) < 1e-12:
            fallback = np.array([0.0, 1.0, 0.0])
            lab_l1_in_crystal = fallback - np.dot(fallback, lab_l3_in_crystal) * lab_l3_in_crystal
        if np.linalg.norm(lab_l1_in_crystal) < 1e-12:
            fallback = np.array([0.0, 0.0, 1.0])
            lab_l1_in_crystal = fallback - np.dot(fallback, lab_l3_in_crystal) * lab_l3_in_crystal
        lab_l1_in_crystal = lab_l1_in_crystal / np.linalg.norm(lab_l1_in_crystal)
        lab_l2_in_crystal = np.cross(lab_l3_in_crystal, lab_l1_in_crystal)
        axes = np.column_stack([lab_l1_in_crystal, lab_l2_in_crystal, lab_l3_in_crystal])
        return cls(axes)

    @classmethod
    def from_miller_surface(
        cls,
        structure: CrystalStructure,
        hkl: ArrayLike,
        *,
        in_plane_direction_uvw: ArrayLike = (1.0, 0.0, 0.0),
    ) -> "CrystalOrientation":
        """Build a real orientation from a general Miller plane and lattice.

        The surface normal comes from reciprocal space:
        `G = h b1 + k b2 + l b3`. The in-plane hint is a direct-lattice
        direction `[u v w]`, projected into the plane if needed.
        """

        lab_l3_in_crystal = structure.miller_plane_normal(hkl)
        hint = structure.direct_lattice_direction(in_plane_direction_uvw)
        lab_l1_in_crystal = hint - np.dot(hint, lab_l3_in_crystal) * lab_l3_in_crystal
        if np.linalg.norm(lab_l1_in_crystal) < 1e-12:
            direct = structure.direct_lattice_vectors()
            for idx in range(3):
                fallback = direct[:, idx] / np.linalg.norm(direct[:, idx])
                lab_l1_in_crystal = fallback - np.dot(fallback, lab_l3_in_crystal) * lab_l3_in_crystal
                if np.linalg.norm(lab_l1_in_crystal) >= 1e-12:
                    break
        if np.linalg.norm(lab_l1_in_crystal) < 1e-12:
            raise ValueError("Could not construct an in-plane direction.")
        lab_l1_in_crystal = lab_l1_in_crystal / np.linalg.norm(lab_l1_in_crystal)
        lab_l2_in_crystal = np.cross(lab_l3_in_crystal, lab_l1_in_crystal)
        axes = np.column_stack([lab_l1_in_crystal, lab_l2_in_crystal, lab_l3_in_crystal])
        return cls(axes)

    @classmethod
    def from_shaarp_miller_surface(
        cls,
        structure: CrystalStructure,
        plane_hkl: ArrayLike,
        vertical_uvw: ArrayLike,
        *,
        in_plane_tol: float = 1e-8,
    ) -> "CrystalOrientation":
        """Build the orientation stored by SHAARP.ml `hklConvert`.

        `plane_hkl` is interpreted in reciprocal space. `vertical_uvw` is a
        direct-lattice direction that should lie in the selected surface plane.
        The returned rows correspond to Mathematica `orientation[[3]]`.
        """

        z_axes = structure.shaarp_physics_axes()
        v3 = structure.miller_plane_normal(plane_hkl)
        v2 = structure.direct_lattice_direction(vertical_uvw)
        if abs(float(np.dot(v2, v3))) > in_plane_tol:
            raise ValueError("vertical_uvw must lie in the Miller plane for SHAARP hkl orientation.")
        v1 = np.cross(v2, v3)
        v1_norm = np.linalg.norm(v1)
        if v1_norm < 1e-12:
            raise ValueError("Miller plane and vertical direction are degenerate.")
        target_axes = np.vstack([v1 / v1_norm, v2, v3])
        transform = target_axes @ np.linalg.inv(z_axes)
        stored_axes = transform.T
        stored_axes = stored_axes / np.linalg.norm(stored_axes, axis=1)[:, None]
        return cls(stored_axes)

    def rotation_matrix(self) -> np.ndarray:
        axes = _real_matrix(self.z_axes_in_lab, "z_axes_in_lab")
        if axes.shape != (3, 3):
            raise ValueError("z_axes_in_lab must be a 3x3 matrix.")
        norms = np.linalg.norm(axes, axis=1)
        if np.any(norms == 0):
            raise ValueError("Crystal axis vectors must be non-zero.")
        q = axes / norms[:, None]
        if not np.allclose(q @ q.T, np.eye(3), atol=1e-7):
            raise ValueError("Crystal axes must be mutually orthogonal.")
        return q

    def with_lab_azimuth_deg(self, azimuth_deg: float) -> "CrystalOrientation":
        """Return this orientation after rotating the sample about lab L3.

        This is a physical crystal/sample azimuth, distinct from the polarizer
        angle `phi_deg`. Tensor rotation follows:
        `epsilon_lab(azimuth) = Rz epsilon_lab(0) Rz.T`.
        """

        angle = np.deg2rad(float(azimuth_deg))
        c = np.cos(angle)
        s = np.sin(angle)
        lab_rotation = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return CrystalOrientation(self.rotation_matrix() @ lab_rotation.T)

    @classmethod
    def lab_azimuth_deg(cls, azimuth_deg: float) -> "CrystalOrientation":
        """Return an orientation that is only a sample azimuth about lab L3."""

        return cls().with_lab_azimuth_deg(azimuth_deg)


@dataclass(frozen=True)
class Material:
    """A crystal at an optical interface: its symmetry, orientation, and optical/nonlinear tensors.

    Attributes:
        name: human-readable label.
        structure: the :class:`CrystalStructure` (point group + lattice constants).
        orientation: the :class:`CrystalOrientation` rotating crystal-physics axes into the lab frame.
        epsilon_omega: 3x3 complex relative permittivity at the fundamental omega (crystal frame).
        epsilon_2omega: 3x3 complex relative permittivity at 2*omega (crystal frame).
        d_voigt_pm_v: the 3x6 Voigt SHG tensor :math:`d_{i\\mu}` in pm/V (crystal frame).
    """

    name: str
    structure: CrystalStructure
    orientation: CrystalOrientation
    epsilon_omega: ArrayLike
    epsilon_2omega: ArrayLike
    d_voigt_pm_v: ArrayLike

    def eps_w(self) -> np.ndarray:
        return _matrix3(self.epsilon_omega, "epsilon_omega")

    def eps_2w(self) -> np.ndarray:
        return _matrix3(self.epsilon_2omega, "epsilon_2omega")

    def d_voigt(self) -> np.ndarray:
        d = np.asarray(self.d_voigt_pm_v, dtype=complex)
        if d.shape != (3, 6):
            raise ValueError("d_voigt_pm_v must be a 3x6 matrix.")
        return d


@dataclass(frozen=True)
class Layer:
    """One layer in a :class:`MultilayerSystem`.

    Attributes:
        name: label (e.g. "air in", "film", "substrate").
        material: the layer :class:`Material`.
        thickness_um: thickness in microns; ``None`` for a semi-infinite medium (ambient or substrate).
        shg_active: whether this layer generates second-harmonic polarization.
        analytic_h: keep THIS layer's thickness a SYMBOL in the partial-analytical closed
            form (the original's per-layer "analytical h" button, ``SHAARP.ml.nb:5135-5146``,
            which stores a distinct symbol h1, h2, ... in the selected layer). Consumed ONLY
            by the symbolic path; every numeric solver ignores it. Half-spaces cannot carry
            it -- their thickness is infinite, not a variable.
        analytic_d: keep THIS layer's SHG tensor symbolic in the closed form (the original's
            sibling per-layer "analytical dij" button, ``SHAARP.ml.nb:7256-7292``, whose
            symbols are suffixed m1, m2, ... per layer). Also symbolic-path only.
    """

    name: str
    material: Material
    thickness_um: float | None = None
    shg_active: bool = True
    analytic_h: bool = False
    analytic_d: bool = False


@dataclass(frozen=True)
class MultilayerSystem:
    """An ordered air / film(s) / substrate stack for a multilayer SHG calculation.

    Layer 0 is the incidence (ambient) medium and the last layer is the substrate half-space; interior
    layers are finite films (each needs a ``thickness_um``). Call :meth:`validate` to check it.

    Attributes:
        wavelength_um: fundamental wavelength in microns.
        layers: ordered list of :class:`Layer` (incidence medium first, substrate last).
        polarimetry: the :class:`Polarimetry` measurement geometry.
        include_multiple_reflections: keep multiply-reflected waves (cf. the Maker-fringe assumptions).
        include_backward_shg: include backward-propagating SHG sources.
    """

    wavelength_um: float
    layers: list[Layer]
    polarimetry: Polarimetry = field(default_factory=Polarimetry)
    include_multiple_reflections: bool = False
    include_backward_shg: bool = False

    def validate(self) -> None:
        if len(self.layers) < 2:
            raise ValueError("A multilayer system needs at least incident and exit media.")
        if self.wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive.")
        for idx, layer in enumerate(self.layers[1:-1], start=1):
            if layer.thickness_um is None or layer.thickness_um < 0:
                raise ValueError(f"Internal layer {idx} needs a non-negative thickness.")


def with_sample_azimuth_deg(
    system: MultilayerSystem,
    azimuth_deg: float,
    *,
    rotate_top: bool = False,
    rotate_substrate: bool = True,
) -> MultilayerSystem:
    """Return a copy of `system` with material orientations rotated about lab L3.

    This represents a crystal/sample azimuth rotation. It does not change
    `Polarimetry.phi_deg`, which remains the incident polarizer angle.
    """

    layers = []
    for idx, layer in enumerate(system.layers):
        is_top = idx == 0
        is_substrate = idx == len(system.layers) - 1
        should_rotate = (not is_top or rotate_top) and (not is_substrate or rotate_substrate)
        if should_rotate:
            material = replace(
                layer.material,
                orientation=layer.material.orientation.with_lab_azimuth_deg(azimuth_deg),
            )
            layers.append(replace(layer, material=material))
        else:
            layers.append(layer)
    return replace(system, layers=layers)


def _matrix3(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.shape == ():
        arr = np.eye(3, dtype=complex) * arr
    if arr.shape == (3,):
        arr = np.diag(arr)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be scalar, length-3 diagonal, or 3x3 matrix.")
    return arr


def _real_matrix(value: ArrayLike, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) and np.max(np.abs(np.imag(raw))) > 0:
        raise ValueError(f"{name} must be real; complex spatial rotations are not supported.")
    return np.asarray(value, dtype=float)


def _real_vector3(value: ArrayLike, name: str) -> np.ndarray:
    vector = _real_matrix(value, name)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a length-3 vector.")
    return vector
