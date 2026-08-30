"""Named layer- and stack-property presets (the headless data/logic behind the
SHAARP.ml GUI "layer property presets" capability, ``gui_layer_property_presets``).

The GUI offers ready-made layer configurations (a material + thickness + SHG-active
flag) and complete demo stacks the user can load and then edit. This module is the
pure-data registry behind that: no display, no ipywidgets -- just builders that
return validated :class:`~shaarp.config.Layer` / :class:`~shaarp.config.MultilayerSystem`
objects, fully unit-testable headlessly.

It composes the single-material presets in :mod:`shaarp.presets` into layer- and
stack-level presets, and plugs directly into the headless
:class:`~shaarp.layer_editor_model.LayerEditorModel` (load a preset stack, then
add/remove/reorder/edit).
"""

from __future__ import annotations

from typing import Callable

from .config import Layer, MultilayerSystem, Polarimetry
from . import presets


# --- layer presets: a material + thickness + active flag -------------------------
_LAYER_PRESETS: dict[str, Callable[[], Layer]] = {
    "air_cover": lambda: Layer("air cover", presets.air(), shg_active=False),
    "linbo3_zcut_film": lambda: Layer(
        "LiNbO3 z-cut film", presets.linbo3_zcut_1064(), thickness_um=1.0, shg_active=True
    ),
    "linbo3_xcut_film": lambda: Layer(
        "LiNbO3 x-cut film", presets.linbo3_1120_xcut(), thickness_um=1.0, shg_active=True
    ),
    "gaas_film": lambda: Layer(
        "GaAs (111) film", presets.gaas_111_800(), thickness_um=0.5, shg_active=True
    ),
    "passive_interlayer": lambda: Layer(
        "passive interlayer", presets.blank_linear("interlayer"), thickness_um=0.2, shg_active=False
    ),
    "linbo3_substrate": lambda: Layer(
        "LiNbO3 substrate", presets.linbo3_zcut_1064(), shg_active=False
    ),
    "gaas_substrate": lambda: Layer("GaAs substrate", presets.gaas_111_800(), shg_active=False),
    "blank_substrate": lambda: Layer("substrate", presets.blank_linear("substrate"), shg_active=False),
}


def list_layer_presets() -> list[str]:
    """Sorted names of the available single-layer presets."""
    return sorted(_LAYER_PRESETS)


def build_layer_preset(name: str) -> Layer:
    """Build a fresh :class:`Layer` from a named preset.

    A new object is returned on every call so the caller may mutate it freely
    (e.g. set a different thickness) without affecting the registry.
    """
    if name not in _LAYER_PRESETS:
        raise KeyError(
            f"unknown layer preset {name!r}; available: {', '.join(list_layer_presets())}"
        )
    return _LAYER_PRESETS[name]()


# --- stack presets: complete, validated multilayer systems -----------------------
def _stack(layers: list[Layer], *, wavelength_um: float = 1.064, theta_deg: float = 20.0) -> MultilayerSystem:
    system = MultilayerSystem(
        wavelength_um=wavelength_um,
        polarimetry=Polarimetry(theta_deg=theta_deg, phi_deg=0.0, psi_deg=0.0, ellipticity_deg=0.0),
        layers=layers,
    )
    system.validate()
    return system


_STACK_PRESETS: dict[str, Callable[[], MultilayerSystem]] = {
    # air / LiNbO3 z-cut film / blank substrate -- the canonical single-film demo
    "single_film_linbo3": lambda: _stack(
        [build_layer_preset("air_cover"), build_layer_preset("linbo3_zcut_film"), build_layer_preset("blank_substrate")]
    ),
    # air / LiNbO3 film / GaAs substrate -- two distinct crystals
    "linbo3_on_gaas": lambda: _stack(
        [build_layer_preset("air_cover"), build_layer_preset("linbo3_zcut_film"), build_layer_preset("gaas_substrate")]
    ),
    # air / LiNbO3 film / passive interlayer / blank substrate -- a 4-layer multilayer
    "film_interlayer_substrate": lambda: _stack(
        [
            build_layer_preset("air_cover"),
            build_layer_preset("linbo3_zcut_film"),
            build_layer_preset("passive_interlayer"),
            build_layer_preset("blank_substrate"),
        ]
    ),
}


def list_stack_presets() -> list[str]:
    """Sorted names of the available full-stack presets."""
    return sorted(_STACK_PRESETS)


def build_stack_preset(name: str) -> MultilayerSystem:
    """Build a fresh, validated :class:`MultilayerSystem` from a named stack preset."""
    if name not in _STACK_PRESETS:
        raise KeyError(
            f"unknown stack preset {name!r}; available: {', '.join(list_stack_presets())}"
        )
    return _STACK_PRESETS[name]()
