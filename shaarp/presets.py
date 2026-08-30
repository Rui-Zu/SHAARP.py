from __future__ import annotations

import numpy as np

from .config import CrystalOrientation, CrystalStructure, Material
from .tensors import impose_point_group


def air() -> Material:
    return Material(
        name="Air",
        structure=CrystalStructure(point_group="1"),
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3),
        epsilon_2omega=np.eye(3),
        d_voigt_pm_v=np.zeros((3, 6)),
    )


def linbo3_zcut_1064() -> Material:
    d = np.zeros((3, 6), dtype=complex)
    d[0, 4] = 4.6
    d[1, 1] = 2.2
    d[2, 0] = 4.6
    d[2, 2] = 27.0
    no_w, ne_w = 2.232, 2.156
    no_2w, ne_2w = 2.322, 2.234
    return Material(
        name="LiNbO3 z-cut 1064 nm",
        structure=CrystalStructure("3m", a=5.148, b=5.148, c=13.863, gamma_deg=120),
        orientation=CrystalOrientation(np.eye(3)),
        epsilon_omega=np.diag([no_w**2, no_w**2, ne_w**2]),
        epsilon_2omega=np.diag([no_2w**2, no_2w**2, ne_2w**2]),
        d_voigt_pm_v=impose_point_group(d, "3m"),
    )


def linbo3_1120_xcut() -> Material:
    base = linbo3_zcut_1064()
    # Approximate x-cut convention used by the SHAARP.si demo: optic axis in L1.
    orientation = CrystalOrientation(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    return Material(
        name="LiNbO3 (11-20) x-cut",
        structure=base.structure,
        orientation=orientation,
        epsilon_omega=base.epsilon_omega,
        epsilon_2omega=base.epsilon_2omega,
        d_voigt_pm_v=base.d_voigt_pm_v,
    )


def gaas_111_800() -> Material:
    d = np.zeros((3, 6), dtype=complex)
    d[0, 3] = 200.0
    n_w, n_2w = 3.67 + 0.08j, 4.35 + 0.18j
    return Material(
        name="GaAs (111) 800 nm",
        structure=CrystalStructure("-43m", a=5.653, b=5.653, c=5.653),
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3) * n_w**2,
        epsilon_2omega=np.eye(3) * n_2w**2,
        d_voigt_pm_v=impose_point_group(d, "-43m"),
    )


def blank_linear(name: str = "Blank linear") -> Material:
    return Material(
        name=name,
        structure=CrystalStructure("1"),
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3),
        epsilon_2omega=np.eye(3),
        d_voigt_pm_v=np.zeros((3, 6)),
    )


def blank_nonlinear(name: str = "Blank nonlinear") -> Material:
    d = np.zeros((3, 6), dtype=complex)
    d[2, 2] = 1.0
    return Material(
        name=name,
        structure=CrystalStructure("1"),
        orientation=CrystalOrientation(),
        epsilon_omega=np.eye(3),
        epsilon_2omega=np.eye(3),
        d_voigt_pm_v=d,
    )


# --- material registry: the headless data behind the GUI material picker ---------
# (gui_material_selection_blank_linear_nonlinear). Maps a stable name to a builder
# that returns a FRESH Material on every call, so a caller may mutate freely.
MATERIAL_PRESETS: dict[str, "callable[[], Material]"] = {
    "air": air,
    "linbo3_zcut_1064": linbo3_zcut_1064,
    "linbo3_1120_xcut": linbo3_1120_xcut,
    "gaas_111_800": gaas_111_800,
    "blank_linear": blank_linear,
    "blank_nonlinear": blank_nonlinear,
}


def list_materials() -> list[str]:
    """Sorted names of the built-in material presets (the GUI material picker)."""
    return sorted(MATERIAL_PRESETS)


def build_material(name: str) -> Material:
    """Build a fresh :class:`Material` from a named preset.

    ``blank_linear``/``blank_nonlinear`` use their default display name; pass the
    builder directly if a custom name is needed.
    """
    if name not in MATERIAL_PRESETS:
        raise KeyError(f"unknown material preset {name!r}; available: {', '.join(list_materials())}")
    return MATERIAL_PRESETS[name]()
