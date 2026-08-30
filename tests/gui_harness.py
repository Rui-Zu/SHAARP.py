"""Shared GUI-audit harness.

One implementation of the widget-tree walk, result-surface snapshotting, and causality
assertions, consumed by:
  - tests/test_gui_causality_matrix.py (drive controls as a user, assert results change)
  - tests/test_gui_causality_coverage.py (every interactive widget is covered-or-exempt)
  - tests/test_session_state.py (session save/load round-trips every walked widget)
  - the Phase-A inventory generator (build/gui_audit_inventory.md)

Identity scheme: every interactive widget gets a stable string ident
    "{tab}:{ClassName}:{key}#{ordinal}"
where key is, in priority order: the TOOLTIPS dict key its tooltip matches; a snippet of a
custom tooltip; the widget's text()/prefix()/placeholder; its objectName; else "anon". The
ordinal disambiguates same-key groups (e.g. z_axes#0..8). Idents are what the coverage gate's
COVERED/EXEMPT registries name — human-readable, resilient to layout moves, and they change
only when a widget's semantic handle (tooltip/text) changes.
"""

from __future__ import annotations

import os
from collections import defaultdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

try:
    from PySide6 import QtWidgets
    HAVE_QT = True
except Exception:  # pragma: no cover
    QtWidgets = None
    HAVE_QT = False


# The walker lives IN THE PACKAGE (shaarp/gui_introspect.py) Phase D: the app's
# session save/load serializes exactly what this walk enumerates, so audit coverage and
# session coverage cannot drift apart. Tests re-export it here.
if HAVE_QT:
    from shaarp.gui_introspect import (  # noqa: F401 (re-exports for the test modules)
        _interactive_class_map,
        _inside_composite,
        _semantic_key,
        _tooltip_key,
        iter_interactive_widgets,
    )


def snapshot_curves(page, canvas_cls=None):
    """All line data on every matplotlib canvas of the page (input-panel triad excluded)."""
    if canvas_cls is None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as canvas_cls
    out = []
    for cv in page.findChildren(canvas_cls):
        if cv.objectName().startswith("orient_canvas"):
            continue
        for ax in cv.figure.axes:
            for ln in ax.lines:
                out.append(np.asarray(ln.get_ydata(), dtype=float))
    return out


def curves_changed(before, after, rel_tol=1e-6):
    """True if the snapshots differ structurally or by data beyond rel_tol."""
    if len(before) != len(after) or any(a.shape != b.shape for a, b in zip(before, after)):
        return True
    scale = max((float(np.nanmax(np.abs(a))) for a in before if a.size), default=0.0) + 1e-300
    diff = max((float(np.nanmax(np.abs(a - b))) for a, b in zip(before, after) if a.size),
               default=0.0)
    return diff / scale > rel_tol


def click_update(page, app):
    btn = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == "Update / Run")
    btn.click()
    app.processEvents()


def spins_with_tip(page, tip_key):
    from shaarp.desktop_app import TOOLTIPS
    return [s for s in page.findChildren(QtWidgets.QDoubleSpinBox)
            if s.toolTip() == TOOLTIPS[tip_key]]


def combo_with_item(page, item_text, exclude=()):
    return next(c for c in page.findChildren(QtWidgets.QComboBox)
                if c.findText(item_text) >= 0 and c not in exclude)


def ml_case_combo(page):
    """The ML tab's Case Study / System combo (presets + palette films + editor + custom)."""
    return combo_with_item(page, "Custom film (use fields)")


def ml_film_labels(page):
    """The selectable single-film rows of the ML case combo, as STRIPPED display labels
    (case-study fidelity audit: films are palette labels, some indented under a
    disabled material master-title row -- header rows are excluded here)."""
    from shaarp.shaarp_gui import ML_SYSTEM_PRESETS

    from shaarp.user_materials import USER_SECTION_HEADER

    combo = ml_case_combo(page)
    out = []
    for i in range(combo.count()):
        text = combo.itemText(i)
        if text == USER_SECTION_HEADER:
            break  # the user's own materials are not part of the curated palette
        item = combo.model().item(i)
        if item is not None and not item.isEnabled():
            continue  # header rows (both "— ... —" and bare master titles) are disabled
        if text in ML_SYSTEM_PRESETS or text in ("N-layer stack (editor)", "Custom film (use fields)"):
            continue
        out.append(text.strip())
    return out


def select_ml_film(page, app, film_label):
    """Select a single-film row by its stripped display label; returns the combo."""
    combo = ml_case_combo(page)
    for i in range(combo.count()):
        if combo.itemText(i).strip() == film_label:
            combo.setCurrentIndex(i)
            app.processEvents()
            return combo
    raise AssertionError(f"film row {film_label!r} not found in the ML case combo")


# ======================================================================================
# COVERAGE REGISTRY -- the single source of truth mapping every interactive
# widget ident to its audit status. Consumed by tests/test_gui_causality_coverage.py (the
# gate: every walked ident MUST match exactly one rule, and no rule may be "todo") and by
# the Phase-A inventory generator (build/gui_audit_inventory.md).
#
# Rule = (regex on ident, status, note). status:
# "covered" -- a causality/correctness test drives this control; note names the test.
# "exempt" -- deliberately not causality-driven; note carries the REASON (layout-only,
# mirrors another covered control, modal dialog trigger, ...).
# During Phase C, gaps are entered as "todo" + note = the planned step; the gate FAILS
# while any todo remains, so the registry is also the enumerated work list.
# ======================================================================================

import re as _re

COVERAGE_RULES = [
    # ---- case / preset / functionality selectors -------------------------------------
    (r"^(si|ml):QComboBox:tip=case_study#0$", "covered",
     "matrix baselines select cases/presets (SiCausalityMatrix, MlCausalityMatrix)"),
    (r"^si:QComboBox:tip=functionality_si#0$", "covered",
     "matrix drives SHG Simulation + Partial Analytical (theta->expr seam)"),
    (r"^ml:QComboBox:tip=functionality_ml#0$", "covered",
     "matrix drives Maker Fringes + SHG Simulation (MlCausalityMatrix*)"),
    # ---- orientation group ------------------------------------------------------------
    (r"^(si|ml):QComboBox:tip=orientation_mode#0$", "covered",
     "F40 flip fence + SiCausalityMatrix identity step + ML orient matrix step"),
    (r"^si:QDoubleSpinBox:tip=z_axes#[0-8]$", "covered",
     "SiCausalityMatrix z-axes y-tilt step (Crystal Physics Directions mode)"),
    (r"^ml:QDoubleSpinBox:tip=z_axes#[0-8]$", "covered",
     "ML z-cell tilt step on the custom film (mirror of the SI step)"),
    (r"^si:QSpinBox:tip=hkl#[0-2]$", "covered",
     "SiCausalityMatrixExtended Miller (001)->(011) step"),
    (r"^si:QSpinBox:tip=uvw#[0-2]$", "covered",
     "SI Miller uvw step (in-plane rotation changes the pattern)"),
    (r"^ml:QSpinBox:tip=(hkl|uvw)#[0-2]$", "covered",
     "ML Miller hkl+uvw step on the custom film"),
    (r"^(si|ml):QDoubleSpinBox:tip=lattice#[0-5]$", "covered",
     "lattice cell step in Miller mode (c stretch on a non-cubic case; F63 locks cubic cells)"),
    # ---- material tensors -------------------------------------------------------------
    (r"^si:QComboBox:tip=point_group#0$", "covered",
     "point-group combo step (custom mode d-pattern repopulation)"),
    (r"^ml:QComboBox:tip=point_group#0$", "covered",
     "ML point-group step on the custom film"),
    (r"^si:QLineEdit:tip=full_tensor#([0-9]|1[0-7])$", "covered",
     "SI eps(w)/eps(2w) cell step (cells #0-17 in creation order)"),
    (r"^si:QLineEdit:tip=full_tensor#(1[89]|2\d|3[0-5])$", "covered",
     "SI d cell step (SiCausalityMatrix d33 746.8->500)"),
    (r"^ml:QLineEdit:tip=full_tensor#\d+$", "covered",
     "ML custom-film tensor cell step (one eps + one d cell)"),
    # ---- polarimetry ------------------------------------------------------------------
    (r"^(si|ml):QDoubleSpinBox:tip=theta#0$", "covered",
     "theta steps (both matrices)"),
    (r"^(si|ml):QSlider:tip=theta#0$", "covered",
     "slider step: slider.setValue must move the theta spin (mirror assert, no Update)"),
    (r"^(si|ml):QDoubleSpinBox:tip=ellipticity#0$", "covered",
     "ellipticity steps (both matrices)"),
    (r"^si:QComboBox:tip=(polarizer|analyzer)#0$", "covered",
     "SI polarizer/analyzer mode steps (SiCausalityMatrixExtended)"),
    (r"^ml:QComboBox:tip=(polarizer|analyzer)#0$", "covered",
     "ML polarizer/analyzer mode steps (SHG Simulation polar)"),
    (r"^si:QDoubleSpinBox:tip=polarizer#0$", "covered",
     "SI fix-polarizer phi VALUE step"),
    (r"^ml:QDoubleSpinBox:tip=polarizer#0$", "covered",
     "ML fix-polarizer phi VALUE step"),
    (r"^si:QDoubleSpinBox:tip=analyzer#0$", "covered",
     "SI fixed-analyzer psi VALUE step"),
    (r"^ml:QDoubleSpinBox:tip=analyzer#0$", "covered",
     "ML fixed-analyzer psi VALUE step"),
    (r"^si:QDoubleSpinBox:tipx=Analyzer–polarizer offset angle \(deg\) fo#0$", "covered",
     "SI co-rotating offset VALUE step"),
    (r"^ml:QDoubleSpinBox:tipx=Analyzer–polarizer offset angle \(deg\) fo#0$", "covered",
     "ML co-rotating offset VALUE step"),
    (r"^(si|ml):QPushButton:text=-?\d+(\.\d+)?#\d+$", "covered",
     "quick-angle buttons: representative click must set the bound spin "
     "(one assert covers the shared _angle_buttons wiring)"),
    # ---- wavelength / thickness / scan ------------------------------------------------
    (r"^(si|ml):QDoubleSpinBox:tip=wavelength#0$", "covered",
     "wavelength steps (dispersive quartz, data-dispersion guarded; ML wavelength step)"),
    (r"^ml:QDoubleSpinBox:tip=thickness#0$", "covered",
     "unified per-layer thickness step — the SINGLE thickness source, F55 "
     "(MlCausalityMatrix + causality gaps + StackModeOwnershipContract)"),
    (r"^ml:QDoubleSpinBox:tip=theta_range#\d$", "covered",
     "scan min/max/step steps (theta_max covered implicitly; make all three explicit)"),
    (r"^ml:QDoubleSpinBox:tip=maker_ellipticity#\d+$", "covered",
     "F74 Maker-Fringes-specific ellipticity, range -90..90 as the original .ml gives it in "
     "Maker Fringes Collection Settings (the general polarimetry delta-delta drives every "
     "other mode): tests/test_polarimetry_combinations.py::MakerFringesPolarizationFences"
     "::test_maker_specific_ellipticity_drives_the_maker_sweep"),
    (r"^(si|ml):QComboBox:tip=orientation_view#\d+$", "covered",
     "F74 crystal-axes view 2D/3D toggle (default 3D): tests/test_desktop_app.py"
     "::test_orientation_view_toggle_switches_the_figure -- flipping it swaps the canvas between "
     "the 3D triad and the 2D top view, and the default is pinned to 3D"),
    (r"^ml:QDoubleSpinBox:tip=fresnel_range#\d+$", "covered",
     "F70 Fresnel Coefficients scan range (own min/max/step, default step 0.1): "
     "tests/test_polarimetry_combinations.py FresnelScanRangeFences (custom grid honored, "
     "defaults reproduce the original 0-89.9 convention, max clamped below 90)"),
    (r"^ml:(QComboBox|QDoubleSpinBox):tip=sample_rotation#\d+$", "covered",
     "F69 sample-rotation toggle + step size + direction: pin the polarizer/analyzer, drive the "
     "azimuth sweep; fenced by tests/test_ra_scan.py (step/direction causality, fixed phi/psi/"
     "ellipticity reach the sweep, equal-results vs run_sample_rotation)"),
    # F57 moved half-space media into the Dielectric Tensors section; F58 made an ISOTROPIC
    # medium there take ONE number per frequency (the grids hide) — these are those rows.
    (r"^ml:QDoubleSpinBox:tip=medium_n#[0-1]$", "covered",
     "F58 isotropic-medium index pair (the selected layer's n at w/2w): display + write-back + "
     "stay-in-mode fenced by LayerMediumPanelContract; drives Maker via the substrate step in "
     "MlCausalityMatrixExtended; the non-air AMBIENT is certified against live Mathematica by "
     "tests/test_maker_fringes_multilayer_mlamb_reference_comparison.py"),
    (r"^si:QDoubleSpinBox:tip=incident_n#[0-1]$", "covered",
     "F47-3b SI incident-medium index pair: causality + equal-results-at-air + index-matched "
     "physics (tests/test_incident_medium.py)"),
    # ---- ML stack editor / assumptions ------------------------------------------------
    (r"^ml:QComboBox:tip=assumptions#0$", "covered",
     "assumption switch step (MlCausalityMatrix)"),
    (r"^ml:QComboBox:tip=fmr_submode#0$", "covered",
     "FMR submode step (source-policy changes the Maker curve)"),
    (r"^ml:QComboBox:tip=(layer_select|layer_material)#0$", "covered",
     "N-layer editor steps (layer select + material swap)"),
    (r"^ml:QLineEdit:tipx=This layer's thickness is SYMBOLIC in th#0$", "exempt",
     "F61 display-only mirror: shows the layer's thickness SYMBOL (h2) while 'analytical h' is "
     "set; read-only, carries no state of its own (the flag drives it, and the flag is fenced by "
     "test_gui_causality_gaps::test_analytical_checkboxes_drive_the_partial_expression)"),
    (r"^ml:QSpinBox:tip=n_layers#0$", "covered",
     "n_layers step (stack depth change drives the result)"),
    (r"^ml:QLineEdit:tipx=Optional layer name shown in the layer l#0$", "covered",
     "FB3 layer-name step (name reaches the layer list label)"),
    (r"^ml:QCheckBox:tipx=Original \.ml 'analytical h': ON keeps TH#0$", "covered",
     "F60 per-layer symbolic thickness; driven by "
     "test_gui_causality_gaps::test_analytical_checkboxes_drive_the_partial_expression"),
    (r"^(si|ml):QComboBox:tip=eps_mode#0$", "covered",
     "F60 n-vs-epsilon entry mode (the original's LinearInput); driven by "
     "test_gui_causality_gaps::test_eps_entry_mode_converts_the_display_without_changing_the_physics"),
    (r"^ml:QCheckBox:tipx=Original \.ml 'analytical dij' \(SHAARP\.ml#0$", "covered",
     "analytical-dij step (Partial expr known/unknown mixing)"),
    # ---- run / export / copy / presets ------------------------------------------------
    (r"^(si|ml):QPushButton:tip=run#[0-1]$", "covered",
     "Update-button equivalence step (input-panel + output-panel + window global drive "
     "the same run; matrix steps already click one of them)"),
    (r"^(si|ml):QPushButton:tip=export#0$", "exempt",
     "opens a modal save dialog; the payload path is fenced by SiExportCausality via "
     "page._build_export_payload"),
    (r"^(si|ml):QPushButton:tipx=Save the currently shown plot as PNG / S#0$", "exempt",
     "F49-B1 Export figure: opens a modal save dialog; the save path is fenced by "
     "test_session_state.DebugTooling.test_export_figure_writes_a_file via page._on_export_figure"),
    (r"^(si|ml):QPushButton:tipx=Copy the currently shown plot to the cli#0$", "exempt",
     "F50-#6 Copy figure: writes the plot pixmap to the clipboard; fenced by "
     "test_session_state.DebugTooling.test_copy_figure_and_provenance via page._on_copy_figure"),
    (r"^(si|ml):QPushButton:tip=closed_form#0$", "covered",
     "Copy closed form: clipboard content non-empty + tracks the run"),
    (r"^(si|ml):QPushButton:tipx=Copy the closed-form expressions in Wolf#0$", "covered",
     "Copy Mathematica form: clipboard content step"),
    (r"^(si|ml):QPushButton:tip=presets#\d+$", "covered",
     "preset save->load round-trip step (slots + label + Clear)"),
    (r"^(si|ml):QLineEdit:tip=presets#0$", "covered",
     "preset label feeds the round-trip step"),
    (r"^(si|ml):QPushButton:tip=my_materials#[0-3]$", "covered",
     "F64 My Materials save/update/rename/delete round-trip step (tests/test_update_contract.py UserMaterials)"),
    (r"^(si|ml):QLineEdit:tip=my_materials#0$", "covered",
     "F64 My Materials name field feeds the round-trip step"),
    (r"^(si|ml):QGroupBox:tip=my_materials#0$", "exempt",
     "collapsible group disclosure (F64 My Materials)"),
    # ---- structural / chrome ----------------------------------------------------------
    (r"^(si|ml):QToolButton:", "exempt",
     "collapsible-group header: layout-only (no compute effect); rendering integrity is "
     "covered by the layout harness + screenshot walk"),
    (r"^(si|ml):QGroupBox:", "exempt",
     "collapsible disclosure group (F48): toggling collapses/expands the panel, no compute "
     "effect; the F48 RELEVANCE gating that drives these on functionality change is fenced "
     "separately by test_gui_causality_gaps.MlGapSteps.test_scan_polarimetry_relevance_gating"),
    (r"^win:QPushButton:tip=run#0$", "covered",
     "global Update button: part of the Update-equivalence step"),
    (r"^win:QPushButton:tipx=(Save EVERY input on both tabs to a JSON |Restore all inputs from a saved session )#0$",
     "exempt",
     "session Save/Load buttons open modal FILE dialogs; the serialize/apply machinery they "
     "call is directly fenced by tests/test_session_state.py (round-trip + result reproduction)"),
    (r"^win:QPushButton:tipx=Save the ENTIRE current setup \(every[^#]*#0$", "exempt",
     "F68 Save-Full-Setup button (ex Save-to-Library) opens a modal name dialog; the named "
     "save/apply/delete machinery is fenced by tests/test_update_contract.py MaterialLibrary"),
    (r"^win:QPushButton:tipx=Load \(or delete\) a full setup prev[^#]*#0$", "exempt",
     "F68 Saved-Setups button (ex Library) opens a modal picker; the machinery is fenced by "
     "tests/test_update_contract.py MaterialLibrary"),
    (r"^win:QToolButton:name=(ScrollLeftButton|ScrollRightButton|qt_menubar_ext_button)#0$",
     "exempt", "Qt-internal chrome (tab-bar scrollers, menubar overflow) -- not app controls"),
]


if HAVE_QT:
    from shaarp.gui_introspect import walk_all  # noqa: F401 (package implementation)
def coverage_status(ident):
    """(status, note) for an ident, or ("UNMATCHED", "") if no rule claims it."""
    for pattern, status, note in COVERAGE_RULES:
        if _re.match(pattern, ident):
            return status, note
    return "UNMATCHED", ""
