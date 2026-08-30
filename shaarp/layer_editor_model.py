"""Headless layer-editor STATE MODEL for SHAARP.py multilayer stacks.

This is the non-widget data model BEHIND an interactive material/layer editor: a
GUI/notebook widget would bind its controls to this model's operations and render
its `.validate()` report. It is fully headless and testable -- it never imports a
widget toolkit -- so the editor logic (add/remove/reorder/edit/set-active layers
+ physical-constraint validation + build into a MultilayerSystem) is verifiable
without a display. The remaining piece (the live ipywidgets UI) requires a
notebook display and is out of scope for headless validation.

Constraints enforced (SHAARP multilayer conventions):
- at least 3 layers (top medium / >=1 internal / substrate);
- exactly one semi-infinite top medium (index 0, thickness None) and one
  semi-infinite substrate (last index, thickness None);
- every INTERIOR layer has a finite positive thickness;
- at least one SHG-active layer;
- top medium and substrate are passive (not SHG-active) by SHAARP convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .config import Layer, MultilayerSystem, Polarimetry


@dataclass
class LayerSpec:
    """Editable spec for one layer (mirrors config.Layer but mutable for editing)."""

    name: str
    material: Any
    thickness_um: float | None = None
    shg_active: bool = False


@dataclass
class ValidationReport:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass
class LayerEditorModel:
    """Ordered, editable layer stack with validation and a build-to-system step."""

    layers: list[LayerSpec] = field(default_factory=list)
    wavelength_um: float = 1.064
    polarimetry: Polarimetry = field(default_factory=Polarimetry)

    # ---- editing operations (return self for chaining; pure on the model) ----
    def add_layer(self, spec: LayerSpec, index: int | None = None) -> "LayerEditorModel":
        if index is None:
            self.layers.append(spec)
        else:
            self.layers.insert(index, spec)
        return self

    def remove_layer(self, index: int) -> "LayerEditorModel":
        if not 0 <= index < len(self.layers):
            raise IndexError(f"remove_layer: index {index} out of range (n={len(self.layers)})")
        del self.layers[index]
        return self

    def move_layer(self, src: int, dst: int) -> "LayerEditorModel":
        n = len(self.layers)
        if not (0 <= src < n and 0 <= dst < n):
            raise IndexError(f"move_layer: indices ({src},{dst}) out of range (n={n})")
        spec = self.layers.pop(src)
        self.layers.insert(dst, spec)
        return self

    def set_thickness(self, index: int, thickness_um: float | None) -> "LayerEditorModel":
        self.layers[index] = replace(self.layers[index], thickness_um=thickness_um)
        return self

    def set_active(self, index: int, active: bool) -> "LayerEditorModel":
        self.layers[index] = replace(self.layers[index], shg_active=bool(active))
        return self

    # ---- validation ----
    def validate(self) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        n = len(self.layers)
        if n < 3:
            errors.append(f"stack needs >=3 layers (top/internal/substrate); have {n}")
            return ValidationReport(ok=False, errors=tuple(errors))
        if self.layers[0].thickness_um is not None:
            errors.append("top medium (layer 0) must be semi-infinite (thickness_um=None)")
        if self.layers[-1].thickness_um is not None:
            errors.append("substrate (last layer) must be semi-infinite (thickness_um=None)")
        for i in range(1, n - 1):
            t = self.layers[i].thickness_um
            if t is None:
                errors.append(f"interior layer {i} ({self.layers[i].name}) needs a finite thickness")
            elif not (float(t) > 0):
                errors.append(f"interior layer {i} ({self.layers[i].name}) thickness must be > 0 (got {t})")
        if not any(l.shg_active for l in self.layers):
            errors.append("stack needs at least one SHG-active layer")
        if self.layers[0].shg_active:
            warnings.append("top medium marked SHG-active (unusual; SHAARP convention is a passive top)")
        if self.layers[-1].shg_active:
            warnings.append("substrate marked SHG-active (unusual; SHAARP convention is a passive substrate)")
        if float(self.wavelength_um) <= 0:
            errors.append(f"wavelength_um must be > 0 (got {self.wavelength_um})")
        return ValidationReport(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))

    # ---- build into the validated solver input ----
    def build_system(self) -> MultilayerSystem:
        report = self.validate()
        if not report.ok:
            raise ValueError("cannot build MultilayerSystem from invalid editor state: " + "; ".join(report.errors))
        layers = [
            Layer(name=s.name, material=s.material, thickness_um=s.thickness_um, shg_active=s.shg_active)
            for s in self.layers
        ]
        return MultilayerSystem(wavelength_um=self.wavelength_um, layers=layers, polarimetry=self.polarimetry)

    # ---- convenience: seed from an existing system ----
    @classmethod
    def from_system(cls, system: MultilayerSystem) -> "LayerEditorModel":
        specs = [
            LayerSpec(name=l.name, material=l.material, thickness_um=l.thickness_um, shg_active=l.shg_active)
            for l in system.layers
        ]
        return cls(layers=specs, wavelength_um=system.wavelength_um, polarimetry=system.polarimetry)


__all__ = ["LayerSpec", "ValidationReport", "LayerEditorModel"]
