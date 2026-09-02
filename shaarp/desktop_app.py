"""Standalone DESKTOP GUI for SHAARP.py (PySide6) -- the .exe-packaged merged SHAARP.si + SHAARP.ml.

Layout mirrors the ORIGINAL Mathematica GUIs' input-panel / output-panel split, and every control
carries a TOOLTIP whose text is sourced from the original package's own documentation
(Github/SHAARP.ml/docs/input.md), so the in-app help is faithful to the original, not invented.
All computation routes through the same validated cores as the notebook GUI
(compute_si_gui_result / compute_ml_gui_result and the validated figure builders).

Run directly: python -m shaarp.desktop_app
Headless smoke (CI): set QT_QPA_PLATFORM=offscreen and call build_main_window().
"""

from __future__ import annotations

import sys
import time

from .shaarp_gui import (
    FUNCTIONALITY_CANON,
    ML_ASSUMPTIONS,
    ML_FUNCTIONALITIES,
    ML_FUNCTIONALITY_DISPLAY,
    ML_SYSTEM_PRESETS,
    ORIENTATION_MODES,
    SHG_POINT_GROUPS,
    SI_FUNCTIONALITIES,
    SI_FUNCTIONALITY_DISPLAY,
    PresetStore,
    analytical_expression_text,
    build_custom_ml_system,
    build_custom_si_material,
    build_maker_figure,
    build_fresnel_figure,
    build_ml_polarimetry_figure,
    build_schematic_figure,
    build_si_polarimetry_figure,
    compute_ml_gui_result,
    compute_si_gui_result,
    ml_polarimetry_curve,
    point_group_free_components,
    resolve_ml_system_preset,
)

# ---------------------------------------------------------------------------
# Tooltip texts -- adapted from the ORIGINAL SHAARP.ml documentation (docs/input.md), EXCEPT
# where a control has no counterpart there. input.md documents the layer editor ("Number of
# Layers", per-layer name/material) but says NOTHING about a substrate or exit medium — the word
# never appears — because the released .ml GUI fixes BOTH half-spaces to air. So the
# medium/substrate/incident-index wording below is OURS, not transcribed; the older
# SHAARP.SLAB build did expose both half-spaces and its still-shipped docs/getstarted.md
# describes that UI. Attribution corrected after checking the primary sources.
# so the in-app guidance matches what the Mathematica GUI's manual tells its users.
# ---------------------------------------------------------------------------
TOOLTIPS = {
    "functionality_si": (
        "Choose what to calculate for the single interface.\n"
        "SHG Simulation: polarimetry settings to simulate the reflected SHG intensities.\n"
        "Partial/Full Analytical: derives the reflected SHG as a closed-form expression with the\n"
        "input polarization phi and the SHG tensor components d_ij as symbolic variables."
    ),
    "functionality_ml": (
        "Choose what to calculate for the multilayer.\n"
        "SHG Simulation simulates the SHG intensities for the given multilayer (polar plots;\n"
        "sample rotation available in Polarimetry Settings), applying various assumptions as\n"
        "required. Maker Fringes sweeps the transmitted SHG vs incidence angle; Fresnel\n"
        "Coefficients sweeps the linear reflection/transmission -- each with its own scan-range\n"
        "section. Partial Analytical Expressions derives the SHG reflectance/transmittance with\n"
        "thicknesses and SHG tensor elements as symbolic variables."
    ),
    "point_group": (
        "Select the point group of the layer from the drop-down menu.\n"
        "Constraints due to the point group symmetry are automatically imposed on the SHG tensor."
    ),
    "lattice": (
        "Enter the crystallographic constants describing the unit cell.\n"
        "The lengths a, b, c are to be entered in Angstroms and the bond angles\n"
        "alpha, beta, gamma in degrees."
    ),
    "orientation_mode": (
        "To define the crystal orientation of the layer, use either z-cut (crystal physics axes\n"
        "aligned with the lab axes) or Miller Indices (hkl): the first set of h-k-l values\n"
        "corresponds to the surface plane (hkl) while u-v-w defines the direction perpendicular\n"
        "to the plane of incidence [uvw]. An error is shown if the surface-plane Miller indices\n"
        "are inconsistent with the in-plane direction."
    ),
    "z_axes": (
        "Crystal Physics Directions: enter the components of the crystal-physics axes Z1, Z2, Z3\n"
        "in the lab coordinate system (L1, L2, L3) -- one row per Z axis. They must form an\n"
        "orthonormal triad (the entry is validated). Identity = z-cut."
    ),
    "hkl": "Surface plane Miller indices (hkl): the plane normal becomes the lab surface normal L3.",
    "uvw": (
        "Direction [uvw] perpendicular to the plane of incidence (must lie IN the (hkl) surface\n"
        "plane); it is mapped to the lab L2 axis."
    ),
    "dielectric": (
        "The dimensionless dielectric tensors eps_ij at the fundamental (w) and second-harmonic\n"
        "(2w) frequencies, in the crystal physics coordinate system. On the multilayer tab this\n"
        "section displays and edits the SELECTED layer's medium: an ISOTROPIC medium (air, a\n"
        "plain substrate, an immersion fluid) takes ONE number per frequency and shows the two\n"
        "refractive-index rows instead of the 3x3 grids (eps = n^2 * I; air = 1). Anisotropic\n"
        "layers show the full grids. The TOP (incident) medium must stay isotropic. On the SI\n"
        "tab the incident-medium rows set the entrance medium (air = 1)."
    ),
    "medium_n": (
        "Refractive index of the selected ISOTROPIC layer at the fundamental (w) and\n"
        "second-harmonic (2w) frequencies — the whole medium in one number per frequency\n"
        "(eps = n^2 * I). Air = 1; fused silica ~1.45 / 1.46; water ~1.33.\n"
        "Row 1 is the incident (entrance) medium and the last row the exit medium; the solver\n"
        "requires the incident medium to be isotropic. NOTE: the released Mathematica GUI fixed\n"
        "BOTH half-spaces to air — settable media here are an extension of that tool (the\n"
        "underlying SHAARP.ml engine always supported an arbitrary substrate)."
    ),
    "shg_tensor": (
        "Enter the components of the SHG tensor d_ij (expressed in Voigt notation) in units of\n"
        "pm/V. Note that constraints due to the point group symmetry are automatically imposed\n"
        "on the SHG tensor -- only the symmetry-independent components are shown."
    ),
    "full_tensor": (
        "Advanced: enter the FULL complex dielectric tensors eps(w), eps(2w) (3x3) and the full SHG\n"
        "tensor d (3x6, Voigt) directly, e.g. an absorbing material like '6.26+18.81 I'. When the box\n"
        "is checked these override the refractive-index and symmetry-constrained-d entry above. The\n"
        "multilayer solver uses the full complex tensors; the single-interface polar plot carries\n"
        "the full COMPLEX principal values (absorption included) and approximates only a lab-frame\n"
        "eps with off-diagonal terms."
    ),
    "ellipticity": (
        "The incident electric field is characterized by the polarization angle phi and the\n"
        "ellipticity Delta-delta (both in degrees). Enter the ellipticity between 0 and 360 deg."
    ),
    "analyzer": (
        "Choose a rotating analyzer (the plot shows the parallel and perpendicular SHG channels)\n"
        "or a fixed analyzer set to the angle psi (in degrees). phi=0 is p-polarized, phi=90 is s."
    ),
    "polarizer": (
        "Incident polarization angle phi. 'Rotate Polarizer' sweeps phi for the polar plot (the\n"
        "standard polarimetry). 'Fix Polarizer' holds phi at the entered value; the polar plot then\n"
        "traces the rotating-analyzer SHG intensity I(psi). phi=0 deg is p-polarized, phi=90 deg is s."
    ),
    "n_layers": (
        "Number of Layers in the multilayer: EVERY medium counts, numbered 1..N — the\n"
        "air / quartz / Au / air example is 4 layers. Layer 1 is the incident medium and layer\n"
        "N the exit medium: both are semi-infinite, so they carry no thickness and generate no\n"
        "SHG, but their optical constants are yours to set. The interior layers are the films,\n"
        "each with its own material, thickness, SHG-active flag and symbolic-thickness flag."
    ),
    "layer_select": (
        "Select the layer to view/modify. Layers are numbered 1..N in stacking order; layer 1\n"
        "(incident) and layer N (exit) are the semi-infinite media — no thickness, no SHG."
    ),
    "eps_mode": (
        "Enter the linear optical constants as the complex REFRACTIVE INDEX ñ or as the complex\n"
        "DIELECTRIC PERMITTIVITY ε̃ — the original's LinearInput switch, over the same two 3x3\n"
        "grids. Switching CONVERTS the values on screen (ε = ñ·ñ as a matrix square, ñ = ε^(1/2)),\n"
        "so the panel keeps describing the same medium: it changes the units you type in, never\n"
        "the physics. Complex entries are accepted in both modes (absorbing media)."
    ),
    "layer_material": (
        "Assign a material to the selected layer: air, a plain isotropic medium (n entered via\n"
        "the Dielectric Tensors section below), any original Case Study material (tensors taken\n"
        "at the fundamental wavelength), or Custom (fields)."
    ),
    "wavelength": (
        "Enter the wavelength (in vacuum) of the fundamental light incident on the multilayer, in um.\n"
        "Drives the Custom-film / Film / N-layer stack modes. A named paper preset carries its OWN\n"
        "wavelength; selecting one syncs this field to it (the row label then says 'set by the preset')."
    ),
    "thickness": (
        "Per-layer thickness in um — the ONLY thickness input: the stack editor holds the\n"
        "geometric truth in every mode. In the simple 'Custom film' / 'Film:' modes this row IS\n"
        "the 3-layer template's film thickness (editing it stays in the mode); under a named\n"
        "preset the editor shows the preset's real layers, and editing one switches to the\n"
        "N-layer editor owning a faithful copy."
    ),
    "incident_n": (
        "Refractive index of the (isotropic) INCIDENT medium above the crystal surface — air\n"
        "(n = 1, the default) unless the sample is immersed or index-matched. Scalar n at the\n"
        "fundamental and second-harmonic frequencies. Drives the numeric SHG simulation and the\n"
        "effective-index tile; the analytical derivations assume n = 1 (air)."
    ),
    "theta": (
        "Incident angle theta_i: the angle (in degrees) between the incident ray and the normal\n"
        "to the incident surface. Enter a value between 0 and 90 or use the slider."
    ),
    "sample_rotation": (
        "Rotational anisotropy (the original .ml 'Sample Rotation'): turn the SAMPLE about its\n"
        "surface normal and record SHG vs that angle, 0-360 deg in the chosen direction with the\n"
        "chosen step size. Direction is stated as seen looking at the sample from the beam side.\n"
        "The polarizer and analyzer keep their OWN rotate/fix choices -- any combination is\n"
        "legal: an element set to rotate follows the same scan angle, a fixed element holds its\n"
        "fixed angle at every azimuth point. Runs under SHG Simulation and (numerically) under\n"
        "Partial Analytical Expressions. (The original spells the azimuth phi; this port spells\n"
        "it psi_s, since phi is the polarizer and psi the analyzer here.)"
    ),
    "fresnel_range": (
        "Fresnel Coefficients scan range: incident angle theta_i is swept from theta min to\n"
        "theta max in steps of theta step (default 0.1 deg). Controlled separately from the\n"
        "Maker Fringes scan range. (The original SHAARP.ml fixed this range at 0-90 deg and\n"
        "exposed only the step size.)"
    ),
    "maker_ellipticity": (
        "Incident ellipticity Delta-delta used by the MAKER FRINGES sweep, restricted to\n"
        "-90..90 deg with quick-set steps -- the range the original SHAARP.ml gives this\n"
        "control in its Maker Fringes Collection Settings. The general polarimetry Delta-delta\n"
        "above (+-360) drives every other mode."
    ),
    "orientation_view": (
        "How to draw the crystal-axes view: 3D shows the Zi and Li triads in perspective;\n"
        "2D looks straight DOWN the surface normal L3 (a top view), which reads an in-plane\n"
        "azimuth far more clearly. In 2D an axis tilted out of the plane is drawn SHORT and its\n"
        "out-of-plane component along L3 is printed in brackets beside its label."
    ),
    "theta_range": (
        "Maker fringes are calculated for incident angles between theta_min and theta_max.\n"
        "The step size (in degrees) sets the resolution: a smaller step is finer but slower.\n"
        "(Fresnel Coefficients has its own separate scan-range section.)"
    ),
    "assumptions": (
        "Apply full multiple reflections (FMR), Jerphagnon-Kurtz (JK) or Herman-Hayden (HH)\n"
        "assumptions for the calculation of the polar plots, Fresnel coefficients and/or Maker\n"
        "fringes. All three modes are validated against live SHAARP.ml."
    ),
    "fmr_submode": (
        "For Full Multiple Reflections, specify whether to consider backward waves and, if so,\n"
        "whether to consider standing waves (the original's winhAssumption 0/1/2). All three are\n"
        "validated against live SHAARP.ml (quartz+Au docs system, agreement ~1e-9)."
    ),
    "case_study": (
        "Crystal and optical properties of validated example systems are pre-defined for\n"
        "convenience (e.g. the LiNbO3 film and the docs' quartz + Au Maker Fringes case). Select\n"
        "'Custom film (use fields)' to define the multilayer from the entry fields instead."
    ),
    "presets": (
        "Layer Properties Preset Values: save the entered crystal and optical properties as a\n"
        "preset and re-apply them later. When a preset is saved, its button turns blue.\n"
        "Clear Presets removes all the saved preset information. Presets do not persist across sessions.\n"
        "(Superseded by the persistent 'My Materials' store below.)"
    ),
    "my_materials": (
        "My Materials: save the material currently entered in the crystal panels (point group,\n"
        "lattice, orientation, eps(w), eps(2w), d, at the current wavelength) under a name of your\n"
        "own. It is stored in ~/.shaarp/user_materials.json and appears as a selectable material on\n"
        "the SI case list and the ML layer-material list. Update overwrites the selected material with\n"
        "the current panels; Rename and Delete act on the selected material only. The built-in case\n"
        "studies are never modified, and their names cannot be reused. A saved material is\n"
        "single-wavelength: selecting it sets the wavelength it was saved at."
    ),
    "run": (
        "Click Update/Run after specifying the inputs for the changes to take place on the\n"
        "output panel. This includes any changes on any of the input sub-panels."
    ),
    "export": "Write the last result's numeric data as JSON (and, for analytical runs, the closed form as .txt).",
    "closed_form": (
        "The closed-form SHG polarimetry expression from the analytical run -- the expression an\n"
        "experimentalist fits to a polarimetry scan to extract the d_ij coefficients. Use Copy to\n"
        "place it on the clipboard."
    ),
}


# The original GUIs' default 'User Guide' / welcome page, condensed from the SHAARP.ml docs.
USER_GUIDE_HTML = """
<h2>&#9839;SHAARP.py &mdash; User Guide</h2>
<p>A validated Python port of the Mathematica <b>SHAARP.si</b> (single interface) and
<b>SHAARP.ml</b> (multilayer) second-harmonic-generation packages, merged into one panel.
Use the two tabs at the top to switch between the single-interface and multilayer interfaces.</p>
<p>The original Mathematica packages this port reproduces:
<b>&#9839;SHAARP.si</b> &mdash;
<a href="https://github.com/Rui-Zu/SHAARP">github.com/Rui-Zu/SHAARP</a> &middot;
<b>&#9839;SHAARP.ml</b> &mdash;
<a href="https://github.com/bzw133/SHAARP.ml">github.com/bzw133/SHAARP.ml</a></p>
<p><b>Workflow</b> (per tab):</p>
<ol>
<li>Pick a <b>Functionality</b> (SHG Simulation, Maker Fringes, Fresnel Coefficients, or the
analytical modes).</li>
<li>Describe the material: choose a <b>Case Study</b> material or set the point group, lattice,
<b>crystal orientation</b> (z-cut, Miller hkl + in-plane uvw, or Crystal Physics Directions),
refractive indices, and the symmetry-constrained SHG <b>d</b> tensor.</li>
<li>Set the <b>wavelength</b>; for the multilayer tab also set the layer stack, thicknesses, the
<b>Assumptions</b> (Full multiple reflections / JK / HH, plus backward / standing-wave options),
and the scan range.</li>
<li>Set the <b>Polarimetry</b> (incident angle, ellipticity, polarizer/analyzer).</li>
<li>Click <b>Update / Run</b>. Hover any control for help (text taken from the original SHAARP
documentation). Use <b>Export data</b> to save the numeric results (and analytical closed forms).</li>
</ol>
<p><i>&#966; = 0&deg; is p-polarized; &#966; = 90&deg; is s-polarized. Every computation routes
through the same backend validated value-by-value against live Mathematica SHAARP.</i></p>
<hr>
<h3>References &mdash; how to cite</h3>
<p>If you use SHAARP (or this validated Python port) in any published work, please cite the
original SHAARP papers:</p>
<ol>
<li>Zu, R., Wang, B., He, J. <i>et al.</i> &ldquo;Analytical and numerical modeling of optical second
harmonic generation in anisotropic crystals using &#9839;SHAARP package.&rdquo;
<i>npj Computational Materials</i> <b>8</b>, 246 (2022). <a href="https://doi.org/10.1038/s41524-022-00930-4">doi:10.1038/s41524-022-00930-4</a></li>
<li>Zu, R., Wang, B., He, J. <i>et al.</i> &ldquo;Optical second harmonic generation in anisotropic
multilayers with complete multireflection of linear and nonlinear waves using &#9839;SHAARP.ml
package.&rdquo; <i>npj Computational Materials</i> <b>10</b>, 64 (2024). <a href="https://doi.org/10.1038/s41524-024-01229-2">doi:10.1038/s41524-024-01229-2</a></li>
</ol>
<p><b>Authors:</b> R. Zu, B. Wang, L. Weber, A. Saha, L.-Q. Chen &amp; V. Gopalan
(The Pennsylvania State University).</p>
<p><b>Acknowledgment:</b> Development of the SHAARP software was supported as part of the Computational
Materials Sciences Program funded by the U.S. Department of Energy, Office of Science, Basic Energy
Sciences, under Award No. DE-SC0020145.</p>
<p style="color:#555"><i>SHAARP.py is an independent, validated Python port; please acknowledge the
original SHAARP software and cite the references above.</i></p>
"""


# Branding text faithful to the original GUIs' header (authors/version/acknowledgment), updated
# to note this is the validated Python port.
BRANDING_HTML = (
    "<b>SHAARP.py</b> &mdash; Second Harmonic Analysis of Anisotropic Rotational Polarimetry"
    "<br><span style='color:#555'>Validated Python port of &#9839;SHAARP.si + &#9839;SHAARP.ml "
    "(Zu, Wang, Weber, Saha, Chen &amp; Gopalan). Original Version 1.00 &middot; Port v1.0.0. "
    "Please properly acknowledge the SHAARP software.</span>"
)


# ---------------------------------------------------------------------------
# Modern, Apple-style appearance (light panels, rounded controls, SF-Pro/Segoe
# font, blue accent, slim scrollbars). A single QSS keeps the widget tree
# unchanged (so behaviour/tests are unaffected) -- it only restyles.
# ---------------------------------------------------------------------------
MODERN_QSS = """
* { font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 13px; }
QMainWindow, QScrollArea, QSplitter { background: #f5f5f7; }
QScrollArea { border: none; }
QMenuBar { background: #f5f5f7; border: none; }
QMenuBar::item:selected { background: #e6e6ea; border-radius: 5px; }
QStatusBar { background: #f5f5f7; color: #6e6e73; }
QToolTip { background: #1d1d1f; color: #ffffff; border: none; padding: 5px 7px; border-radius: 6px; }

QGroupBox {
    background: #ffffff; border: 1px solid #e4e4e7; border-radius: 10px;
    margin-top: 14px; padding: 10px 10px 8px 10px; font-weight: 600; color: #3a3a3c;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 4px; }

QLabel { color: #1d1d1f; background: transparent; }

QPushButton {
    background: #ffffff; border: 1px solid #d0d0d4; border-radius: 7px;
    padding: 5px 12px; color: #1d1d1f;
}
QPushButton:hover { background: #f0f0f3; }
QPushButton:pressed { background: #e2e2e6; }
QPushButton:disabled { color: #b0b0b5; border-color: #e8e8ea; }

QComboBox, QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit {
    background: #ffffff; border: 1px solid #d0d0d4; border-radius: 6px; padding: 3px 6px;
    selection-background-color: #0a84ff; selection-color: #ffffff;
}
QComboBox:focus, QLineEdit:focus, QAbstractSpinBox:focus, QPlainTextEdit:focus { border: 1px solid #0a84ff; }
QComboBox::drop-down { border: none; width: 18px; }

QTabWidget::pane { border: 1px solid #e4e4e7; border-radius: 8px; top: -1px; background: #ffffff; }
QTabBar::tab {
    background: transparent; padding: 6px 14px; margin-right: 3px; color: #6e6e73;
    border-top-left-radius: 7px; border-top-right-radius: 7px;
}
QTabBar::tab:selected { background: #ffffff; color: #0a84ff; font-weight: 600;
    border: 1px solid #e4e4e7; border-bottom: 2px solid #0a84ff; }
QTabBar::tab:hover:!selected { color: #1d1d1f; }

QCheckBox { spacing: 6px; color: #1d1d1f; }
QSlider::groove:horizontal { height: 4px; background: #d8d8dc; border-radius: 2px; }
QSlider::handle:horizontal { background: #ffffff; border: 1px solid #c0c0c5; width: 16px;
    margin: -7px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #0a84ff; border-radius: 2px; }

QProgressBar { border: 1px solid #d0d0d4; border-radius: 7px; background: #ececef;
    text-align: center; color: #1d1d1f; }
QProgressBar::chunk { background: #34c759; border-radius: 6px; }

QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: #c7c7cc; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #a8a8ad; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal { background: #c7c7cc; border-radius: 5px; min-width: 30px; }
QScrollBar::handle:horizontal:hover { background: #a8a8ad; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QSplitter::handle { background: #e4e4e7; }
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:vertical { height: 6px; }
QSplitter::handle:hover { background: #0a84ff; }
"""

# Accent style for the primary "Update" buttons (Apple blue).
PRIMARY_BTN_QSS = (
    "QPushButton { background:#0a84ff; color:#ffffff; font-weight:600; border:none; "
    "border-radius:7px; padding:6px 16px; } "
    "QPushButton:hover { background:#369bff; } QPushButton:pressed { background:#0060df; }"
)


def _load_asset_pixmap(QtGui, filename):
    """Load a bundled shaarp/assets image as a QPixmap (works from source and inside the exe)."""
    try:
        from importlib.resources import files

        data = (files("shaarp") / "assets" / filename).read_bytes()
    except (ModuleNotFoundError, FileNotFoundError, AttributeError, OSError):
        from pathlib import Path

        p = Path(__file__).with_name("assets") / filename
        if not p.exists():
            return None
        data = p.read_bytes()
    pix = QtGui.QPixmap()
    pix.loadFromData(data)
    return pix if not pix.isNull() else None


def _load_logo_pixmap(QtGui):
    """Load the bundled #SHAARP.ml wordmark logo (header banner)."""
    return _load_asset_pixmap(QtGui, "shaarp_ml_logo.png")


def _layer_specs_equal(old: dict, new: dict) -> bool:
    """True only when two layer specs are CERTAINLY identical.

    Used to decide whether writing the edit fields back into the stack was a real user edit.
    Clicking Update settles the selected row first, and that write-back used to mark a named
    preset "edited (re-select to reset)" even when nothing had been touched -- which also routed
    the compute down the modified-working-copy branch instead of the pristine factory one.

    Deliberately conservative: any value we cannot compare confidently (a numpy tensor inside a
    custom row, a nested dict, an exotic type) reports "different", which preserves the previous
    always-dirty behaviour. A false "changed" costs a cosmetic marker; a false "unchanged" would
    silently drop a real edit.
    """
    if old is new:
        return True
    if set(old) != set(new):
        return False
    for key in old:
        a, b = old[key], new[key]
        if key == "material":
            # The stack holds the REGISTRY KEY ("Quartz z-cut (800 nm)") while the picker displays
            # the palette label ("Quartz z-cut · 800 nm"). Comparing those raw strings made every
            # settle look like a material change -- the actual reason untouched presets went dirty.
            from .casestudy_materials import resolve_case_label
            if resolve_case_label(a) != resolve_case_label(b):
                return False
            continue
        if isinstance(a, (str, bool, int, float)) and isinstance(b, (str, bool, int, float)):
            if type(a) is bool or type(b) is bool:
                if bool(a) != bool(b):
                    return False
            elif a != b:
                return False
            continue
        return False  # anything richer: do not guess
    return True


def _hline(QtWidgets):
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


def _collapsible_group(QtWidgets, title, layout_cls=None, collapsed=False):
    """A QGroupBox that expands/collapses like the original GUI's sub-panels (the check
    box in the title acts as the expand/collapse toggle). Returns ``(box, body_layout)``;
    add controls to ``body_layout`` exactly as to a normal group-box layout.

    Collapsing hides the body but keeps each inner control's OWN enabled state
    authoritative -- the checkable group box does not grey the contents on collapse/expand
    (so per-field enable logic, e.g. the orientation/analyzer hints, survives a toggle)."""
    box = QtWidgets.QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(not collapsed)
    shell = QtWidgets.QVBoxLayout(box)
    shell.setContentsMargins(6, 2, 6, 6)
    body = QtWidgets.QWidget()
    lay = (layout_cls or QtWidgets.QVBoxLayout)(body)
    lay.setContentsMargins(4, 4, 4, 4)
    shell.addWidget(body)

    def _toggle(on):
        body.setVisible(on)
        body.setEnabled(True)  # inner per-field enabled flags stay authoritative

    box.toggled.connect(_toggle)
    body.setVisible(not collapsed)
    return box, lay


def _symbols_summary(result) -> str:
    """Say WHAT the closed form contains, at a glance.

    A multilayer closed form is a single ~10^5-character line; the panel shows whatever slice
    the scrollbar happens to sit on, which is how a run carrying d11m2 and d14m2 could look as
    though it had no d symbols at all. Naming the free symbols -- and the size
    -- makes the panel honest without making the reader scroll."""
    try:
        import sympy as _sp
        names, chars = set(), 0
        for key in ("reflected_p_2omega", "reflected_s_2omega",
                    "transmitted_p_2omega", "transmitted_s_2omega"):
            expr = result.stages.get(key)
            if expr is None:
                continue
            chars = max(chars, len(str(expr)))
            if hasattr(expr, "free_symbols"):
                names |= {str(sym) for sym in expr.free_symbols}
            else:
                parsed = _sp.sympify(str(expr), evaluate=False)
                names |= {str(sym) for sym in parsed.free_symbols}
        if not names:
            return ""
        return (f"   |   symbols: {', '.join(sorted(names))}"
                f"   ({chars:,} characters — use Copy/Export for the full form)")
    except Exception:
        return ""


def _friendly_validation_status(raw: str) -> str:
    """Human wording for the validation tag shown under the output panel.

    The raw tags are internal enums (e.g. ``staged_python_not_fully_mathematica_validated``) --
    useful in logs, but alarming/opaque on screen. Map them to a
    short sentence; the raw tag stays available in the label's tooltip.
    """

    raw = str(raw)
    if "not_fully_mathematica_validated" in raw or raw.startswith("staged_python"):
        return ("Validation: validated solver core; this specific configuration is not one of the "
                "mirrored Mathematica reference cases.")
    if "mathematica" in raw and "validated" in raw:
        return "Validation: matches a live-Mathematica-validated reference workflow."
    if "unavailable" in raw:
        return "Validation: reference metadata not bundled in this build."
    return f"Validation: {raw.replace('_', ' ')}"


def _friendly_error_message(exc: BaseException) -> str:
    """User-facing wording for compute errors: lead with what to
    check, keep the technical detail underneath instead of a bare exception name."""

    detail = f"{type(exc).__name__}: {exc}"
    hints = []
    text = str(exc).lower()
    if text.startswith("scan range:"):
        return str(exc)  # already a complete, user-facing instruction
    if "singular" in text:
        hints.append("The dielectric tensor looks unphysical (singular). Check the ε entries -- "
                      "the diagonal should be n² (of order 1-20), never all zeros.")
    if "point group" in text:
        hints.append("The selected point group is not supported for this mode (centrosymmetric/"
                      "isotropic classes have no SHG closed form).")
    if not hints:
        hints.append("The computation failed for the current inputs. Check the material tensors, "
                      "angles, and layer stack for unphysical values.")
    return "\n\n".join(hints) + f"\n\nDetail: {detail}"


def _angle_buttons(QtWidgets, values, target_spin):
    """A row of small quick-select angle buttons (the original GUI's 0/15/30/45/... presets) that set
    ``target_spin`` when clicked. Returns a QWidget so the whole row can be enabled/disabled.


    the row now SHOWS the active preset -- the button matching the spin's CURRENT value is
    highlighted, and the highlight tracks every change of the spin (typed, slider, preset click,
    session restore). A typed value matching no preset lights nothing: the spin itself carries the
    custom value, so free input remains the escape hatch when the presets don't satisfy the need."""
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(3)
    buttons = []
    for v in values:
        b = QtWidgets.QPushButton(str(v))
        # width must FIT the label -- a fixed 42 px cap clipped "180" into "18C" with the styled
        # button padding
        b.setMaximumWidth(max(42, 14 + 9 * len(str(v))))
        b.setCheckable(True)
        b.setStyleSheet("QPushButton:checked { background-color: #58a6ff; color: white; "
                        "border: 1px solid #2f6fb3; border-radius: 3px; }")
        buttons.append((float(v), b))
        lay.addWidget(b)

    def _sync(*_a):
        cur = float(target_spin.value())
        for val, b in buttons:
            b.setChecked(abs(cur - val) < 1e-9)

    for v, b in buttons:
        # setValue on an unchanged value emits no signal, so re-sync explicitly -- otherwise
        # clicking the ALREADY-active preset would toggle its highlight off while still active
        b.clicked.connect(lambda _checked=False, val=v: (target_spin.setValue(float(val)),
                                                         _sync()))
    target_spin.valueChanged.connect(_sync)
    _sync()  # initial state: the default value's preset lights up immediately
    lay.addStretch(1)
    return w


def _spin_row(QtWidgets, spin, buttons=None):
    """One compact row: [spin][quick-preset buttons] on a single line.

    The presets sit beside their spin rather than on a separate labelled row, which halves the
    row count and makes the spin/presets association visual.
    """
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lay.addWidget(spin, 0)
    if buttons is not None:
        lay.addWidget(buttons, 1)
    return w


def _parse_complex(text, default=0j):
    """Parse a GUI complex entry: accepts 'a', 'a+bj', or Mathematica-style 'a+b I'. Empty -> default."""
    s = str(text).strip().replace(" ", "").replace("I", "j").replace("i", "j")
    if not s:
        return default
    try:
        return complex(s)
    except ValueError:
        return default


def _scalar_eps(grid, rtol=1e-6):
    """The scalar e when a 3x3 complex grid equals e * I within tolerance, else None.
    (an isotropic half-space's eps grid acts as a single scalar entry.)"""
    try:
        d0, d1, d2 = complex(grid[0][0]), complex(grid[1][1]), complex(grid[2][2])
    except Exception:
        return None
    scale = max(abs(d0), abs(d1), abs(d2), 1e-30)
    for r in range(3):
        for c in range(3):
            v = complex(grid[r][c])
            want = d0 if r == c else 0j
            if abs(v - want) > rtol * scale:
                return None
    return d0


def _complex_grid(QtWidgets, defaults):
    """A grid of complex-number QLineEdits (the original's full ε / d matrix entry). ``defaults`` is a
    2-D list; returns (container_widget, cells[r][c] QLineEdit)."""
    w = QtWidgets.QWidget()
    g = QtWidgets.QGridLayout(w)
    g.setContentsMargins(0, 0, 0, 0)
    g.setSpacing(2)
    cells = []
    for r, drow in enumerate(defaults):
        row = []
        for c, dval in enumerate(drow):
            e = QtWidgets.QLineEdit(str(dval))
            # Compact cells so the 3x6 d-matrix (6 columns) doesn't force the input panel -- and thus
            # the whole window's minimum width -- so wide that the output/geometry panel gets clipped on
            # narrower laptops. Real values fit; long complex entries scroll within the field.
            e.setMaximumWidth(62)
            e.setMinimumWidth(44)
            g.addWidget(e, r, c)
            row.append(e)
        cells.append(row)
    return w, cells


def _matrix_block(QtWidgets, label, defaults):
    """A Mathematica-style labeled, bracketed matrix of complex QLineEdits: label ( [grid] ).
    Returns (container_widget, cells[r][c]). Used for the full ε (3×3) and d (3×6) tensor entry."""
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(0, 2, 0, 2)
    h.setSpacing(3)
    lab = QtWidgets.QLabel(label)
    lab.setStyleSheet("font-style: italic; color:#1d1d1f; min-width: 42px;")
    h.addWidget(lab)
    lbr = QtWidgets.QLabel("⎡\n⎢\n⎣")
    lbr.setStyleSheet("color:#8a8a8e;")
    h.addWidget(lbr)
    grid_w, cells = _complex_grid(QtWidgets, defaults)
    h.addWidget(grid_w)
    rbr = QtWidgets.QLabel("⎤\n⎥\n⎦")
    rbr.setStyleSheet("color:#8a8a8e;")
    h.addWidget(rbr)
    h.addStretch(1)
    return w, cells


def _wire_symmetric_grid(cells):
    """Constrain a 3×3 grid of QLineEdits so ε_ij = ε_ji: editing either cell of an off-diagonal pair
    mirrors the value to its partner (guarded against the obvious signal loop)."""
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = cells[i][j], cells[j][i]

        def _make(src, dst):
            def _sync(_=None):
                if dst.text() != src.text():
                    dst.blockSignals(True)
                    dst.setText(src.text())
                    dst.blockSignals(False)
            return _sync

        a.textChanged.connect(_make(a, b))
        b.textChanged.connect(_make(b, a))
    return cells


def _require_qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The SHAARP desktop app requires PySide6 (pip install PySide6).") from exc
    return QtCore, QtGui, QtWidgets


def _progress_format_for(canon) -> str:
    """Progress-bar text for a canonical functionality (R9, closed):
    the analytical modes run a CAS solve whose FIRST run for a configuration can take minutes
    (identity ~10 s, Rz-rotated ~1 min, some multilayer cases ~6 min) while repeats hit the
    cache instantly -- without a message users on slower machines read the wait as a hang."""
    if "Analytical" in str(canon or ""):
        return "Symbolic solve — first run for a configuration can take minutes (cached after)... %p%"
    return "Computing... %p%"


_AUTOSESSION = False  # enabled only by main() so tests / build_main_window keep pure defaults


def _last_session_path():
    import os
    import tempfile
    return os.path.join(tempfile.gettempdir(), "shaarp_last_session.json")


def _ml_stack_hooks(win):
    """(payload_fn, apply_fn) exported by the ML page for session stack serialization (R15),
    or (None, None) on a window without them (defensive: tests building partial UIs)."""
    from PySide6 import QtWidgets
    tabs = win.findChild(QtWidgets.QTabWidget)
    page = tabs.widget(1) if tabs is not None and tabs.count() > 1 else None
    return (getattr(page, "_ml_stack_payload", None),
            getattr(page, "_ml_stack_apply", None))


def _force_default_functionality(win) -> None:
    """Put BOTH tabs on "SHG Simulation". Called after a session restore."""
    from PySide6 import QtWidgets as _qw
    tabs = win.findChild(_qw.QTabWidget)
    if tabs is None:
        return
    for idx in range(tabs.count()):
        page = tabs.widget(idx)
        for combo in page.findChildren(_qw.QComboBox):
            if combo.toolTip().startswith("Choose what to calculate"):
                if combo.findText("SHG Simulation") >= 0:
                    combo.setCurrentText("SHG Simulation")
                break


def _autosave_session(win):
    """Persist the current input state to a fixed path so the next launch can restore it."""
    import json
    import time as _t
    payload = {"kind": "shaarp_gui_session", "version": 1,
               "saved": _t.strftime("%Y-%m-%d %H:%M:%S"),
               "widgets": win._collect_session_state()}
    get_stack, _ = _ml_stack_hooks(win)
    if get_stack is not None:  # R15: the ML layer stack rides along with the widget state
        payload["ml_stack"] = get_stack()
    with open(_last_session_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _restore_last_session(win):
    """Restore a prior autosave if one exists; silently ignore a corrupt/incompatible file
    (a bad autosave must NEVER block launch). Returns the number of inputs restored (0 if none)."""
    import json
    import os
    p = _last_session_path()
    if not os.path.exists(p):
        return 0
    try:
        with open(p, encoding="utf-8") as fh:
            payload = json.load(fh)
        widgets = payload.get("widgets", {})
        if payload.get("kind") != "shaarp_gui_session" or not isinstance(widgets, dict):
            return 0
        win._apply_session_state(widgets)
        # R15: the stack applies AFTER the widgets (the mode combo restore above already
        # rebuilt a base stack; pre-R15 files simply have no "ml_stack" key -> no-op).
        _, apply_stack = _ml_stack_hooks(win)
        if apply_stack is not None and isinstance(payload.get("ml_stack"), dict):
            apply_stack(payload["ml_stack"])
        # the app always
        # OPENS on the computable mode. Everything else -- materials, layers, angles, flags --
        # still restores; only the functionality is pinned, so a session saved in a slow
        # analytical mode never greets you with a multi-second solve.
        _force_default_functionality(win)
        return len(widgets)
    except Exception:
        return 0


# --- user material library. A personal, named, on-disk collection of setups (material +
# inputs) so a custom material survives across sessions instead of dying with it. Reuses the SAME
# validated collect/apply-session machinery the coverage gate enforces -- pure serialization, no
# physics path touched. ---
def _material_library_path():
    import os
    d = os.path.join(os.path.expanduser("~"), ".shaarp")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        import tempfile
        d = tempfile.gettempdir()
    return os.path.join(d, "materials.json")


def _load_material_library():
    import json
    import os
    p = _material_library_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        mats = data.get("materials", {})
        return mats if isinstance(mats, dict) else {}
    except Exception:
        return {}


def _write_material_library(mats):
    import json
    with open(_material_library_path(), "w", encoding="utf-8") as fh:
        json.dump({"kind": "shaarp_material_library", "version": 1, "materials": mats}, fh, indent=1)


def _save_material_to_library(win, name):
    """Store the current setup (material + inputs) under ``name`` in the personal library."""
    import time as _t
    name = (name or "").strip()
    if not name:
        raise ValueError("material name must not be empty")
    mats = _load_material_library()
    mats[name] = {"saved": _t.strftime("%Y-%m-%d %H:%M:%S"),
                  "widgets": win._collect_session_state()}
    _write_material_library(mats)
    return name


def _apply_library_material(win, name):
    entry = _load_material_library().get(name)
    if not entry or not isinstance(entry.get("widgets"), dict):
        return 0
    win._apply_session_state(entry["widgets"])
    return len(entry["widgets"])


def _delete_library_material(name):
    mats = _load_material_library()
    if name in mats:
        del mats[name]
        _write_material_library(mats)
        return True
    return False


def build_main_window():
    """Construct (but do not exec) the main window. Headless-testable with QT_QPA_PLATFORM=offscreen."""

    QtCore, QtGui, QtWidgets = _require_qt()
    import matplotlib

    matplotlib.use("QtAgg", force=False)
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from . import __version__ as _pkg_version

    win = QtWidgets.QMainWindow()
    win.setWindowTitle(f"♯SHAARP.py v{_pkg_version} — SHAARP.si + SHAARP.ml (validated Python port)")
    win.resize(1480, 940)
    win.setStyleSheet(MODERN_QSS)  # modern, Apple-style appearance (restyle only; widget tree unchanged)

    # --- branding header (faithful to the original GUIs' #SHAARP banner) ---
    header = QtWidgets.QWidget()
    h_lay = QtWidgets.QHBoxLayout(header)
    h_lay.setContentsMargins(10, 6, 10, 4)
    logo_lbl = QtWidgets.QLabel()
    logo_pix = _load_logo_pixmap(QtGui)
    if logo_pix is not None:
        logo_lbl.setPixmap(logo_pix.scaledToHeight(54, QtCore.Qt.SmoothTransformation))
    else:
        logo_lbl.setText("♯SHAARP")
        logo_lbl.setStyleSheet("font-size: 28px; font-weight: bold;")
    h_lay.addWidget(logo_lbl, 0)
    text_lbl = QtWidgets.QLabel(BRANDING_HTML)
    text_lbl.setTextFormat(QtCore.Qt.RichText)
    text_lbl.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
    text_lbl.setOpenExternalLinks(True)
    # Wrap the long banner text so it does NOT force a huge window minimum width (the non-wrapping
    # label was pinning the min ~1388 px, which clipped the geometry/output panel on smaller laptops).
    text_lbl.setWordWrap(True)
    text_lbl.setMinimumWidth(1)
    h_lay.addWidget(text_lbl, 1)
    # Top-of-panel progress bar + global Update button (mirrors the original GUI's top-right
    # "Update" next to the "...% Completed" progress bar). The Update button recomputes the
    # currently-active tab.
    progress = QtWidgets.QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setFormat("Ready")  # idle state; becomes "Computing…" then "100% Completed" per Update
    progress.setMaximumWidth(220)
    # ...and never narrower than its own longest label. With only a maximum set, the bar could be
    # squeezed until "100% Completed" rendered as "00% Completed" -- which is what both shipped
    # documentation screenshots caught it doing. Measured from the widget's own font so it holds
    # under another theme or DPI, and capped at the 220 maximum so it can never widen the window.
    progress.setMinimumWidth(
        min(220, progress.fontMetrics().horizontalAdvance("100% Completed") + 26))
    progress.setToolTip("Computation progress. Shows '100% Completed' after each Update (top, "
                        "input-panel, and output-panel Update buttons all recompute).")
    h_lay.addWidget(progress, 0)
    update_all_btn = QtWidgets.QPushButton("Update")
    update_all_btn.setStyleSheet(PRIMARY_BTN_QSS)
    update_all_btn.setToolTip(TOOLTIPS["run"])
    update_all_btn.setStatusTip(TOOLTIPS["run"].replace("\n", " "))
    h_lay.addWidget(update_all_btn, 0)
    # Phase D: one-click bug reproduction. "Save Session" serializes EVERY interactive
    # input widget on both tabs (the same walk the coverage gate enforces -- shaarp/
    # gui_introspect.py), "Load Session" restores it. A bug report = the session file + one
    # sentence; a test loads the same file for an exact repro.
    save_sess_btn = QtWidgets.QPushButton("Save Session…")
    load_sess_btn = QtWidgets.QPushButton("Load Session…")
    # personal material library (named on-disk collection, distinct from the file-per-session
    # Save/Load above) so a custom material survives across sessions.
    # renamed so the two stores cannot be confused -- a single MATERIAL is saved with the
    # per-tab "My Materials" group (the primary concept); these buttons save/load a FULL SETUP
    # (every input on both tabs, the whole-state store).
    lib_save_btn = QtWidgets.QPushButton("Save Full Setup…")
    lib_load_btn = QtWidgets.QPushButton("Saved Setups…")
    for _b, _tip in ((save_sess_btn, "Save EVERY input on both tabs to a JSON session file — "
                                     "attach it to a bug report for an exact reproduction."),
                     (load_sess_btn, "Restore all inputs from a saved session file."),
                     (lib_save_btn, "Save the ENTIRE current setup (every input on both tabs) to "
                                    "your personal collection under a name. For a single "
                                    "material, use the 'My Materials' group instead."),
                     (lib_load_btn, "Load (or delete) a full setup previously saved to your "
                                    "personal collection.")):
        _b.setToolTip(_tip)
        h_lay.addWidget(_b, 0)

    def _on_save_session():
        import json as _json
        import time as _t

        from .gui_introspect import collect_session_state
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            win, "Save session", "shaarp_session.json", "JSON (*.json)")
        if not path:
            return
        payload = {"kind": "shaarp_gui_session", "version": 1,
                   "saved": _t.strftime("%Y-%m-%d %H:%M:%S"),
                   "widgets": collect_session_state(win)}
        get_stack, _ = _ml_stack_hooks(win)
        if get_stack is not None:  # R15: the ML layer stack rides along
            payload["ml_stack"] = get_stack()
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, indent=1)
        win.statusBar().showMessage(f"Session saved -> {path}")

    def _on_load_session():
        import json as _json

        from .gui_introspect import apply_session_state
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            win, "Load session", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                payload = _json.load(fh)
            widgets = payload.get("widgets", {})
            if payload.get("kind") != "shaarp_gui_session" or not isinstance(widgets, dict):
                raise ValueError("not a SHAARP session file")
            missing = apply_session_state(win, widgets)
            # R15: apply the saved layer stack after the widgets (absent in pre-R15 files)
            _, apply_stack = _ml_stack_hooks(win)
            if apply_stack is not None and isinstance(payload.get("ml_stack"), dict):
                apply_stack(payload["ml_stack"])
        except Exception as exc:
            win.statusBar().showMessage(f"Session load failed: {type(exc).__name__}: {exc}")
            return
        msg = f"Session loaded ({len(widgets)} inputs) — press Update to recompute."
        if missing:
            msg += f"  [{len(missing)} entries had no matching control in this build]"
        win.statusBar().showMessage(msg)

    def _on_lib_save():
        name, ok = QtWidgets.QInputDialog.getText(
            win, "Save Full Setup", "Name for this setup:")
        if not ok or not name.strip():
            return
        try:
            saved = _save_material_to_library(win, name)
        except Exception as exc:
            win.statusBar().showMessage(f"Save to library failed: {type(exc).__name__}: {exc}")
            return
        win.statusBar().showMessage(f"Saved '{saved}' to your material library.")

    def _on_lib_load():
        mats = _load_material_library()
        if not mats:
            win.statusBar().showMessage("No saved setups yet — use 'Save Full Setup…' first.")
            return
        names = sorted(mats)
        # a trailing " (delete)" pseudo-entry per name keeps the MVP to one dialog without a custom UI
        choices = names + [f"🗑 delete: {n}" for n in names]
        pick, ok = QtWidgets.QInputDialog.getItem(
            win, "Material Library", "Load a saved material (or pick a 🗑 delete entry):",
            choices, 0, False)
        if not ok or not pick:
            return
        if pick.startswith("🗑 delete: "):
            target = pick[len("🗑 delete: "):]
            if _delete_library_material(target):
                win.statusBar().showMessage(f"Deleted '{target}' from your material library.")
            return
        n = _apply_library_material(win, pick)
        win.statusBar().showMessage(
            f"Loaded material '{pick}' ({n} inputs) — press Update to recompute." if n
            else f"Could not load '{pick}'.")

    save_sess_btn.clicked.connect(_on_save_session)
    load_sess_btn.clicked.connect(_on_load_session)
    lib_save_btn.clicked.connect(_on_lib_save)
    lib_load_btn.clicked.connect(_on_lib_load)
    win._collect_session_state = lambda: __import__(
        "shaarp.gui_introspect", fromlist=["collect_session_state"]).collect_session_state(win)
    win._apply_session_state = lambda st: __import__(
        "shaarp.gui_introspect", fromlist=["apply_session_state"]).apply_session_state(win, st)
    icon_pix = _load_asset_pixmap(QtGui, "shaarp_icon.png")  # square # mark for the taskbar
    win.setWindowIcon(QtGui.QIcon(icon_pix) if icon_pix is not None
                      else (QtGui.QIcon(logo_pix) if logo_pix is not None else QtGui.QIcon()))

    tabs = QtWidgets.QTabWidget()
    central = QtWidgets.QWidget()
    c_lay = QtWidgets.QVBoxLayout(central)
    c_lay.setContentsMargins(0, 0, 0, 0)
    c_lay.addWidget(header, 0)
    c_lay.addWidget(_hline(QtWidgets), 0)
    c_lay.addWidget(tabs, 1)
    win.setCentralWidget(central)

    # The header wordmark follows the ACTIVE tab (SHAARP.si logo on the .si tab, SHAARP.ml on the
    # .ml tab) -- the merged app should not brand itself ".ml" while the user works on the .si tab
    # Falls back silently if an asset is missing.
    _si_logo = _load_asset_pixmap(QtGui, "shaarp_si_logo.png")
    _ml_logo = _load_asset_pixmap(QtGui, "shaarp_ml_logo.png")

    def _sync_header_logo(index: int) -> None:
        pix = _si_logo if index == 0 else _ml_logo
        if pix is not None:
            logo_lbl.setPixmap(pix.scaledToHeight(54, QtCore.Qt.SmoothTransformation))

    tabs.currentChanged.connect(_sync_header_logo)
    _sync_header_logo(0)
    win.statusBar().showMessage("Hover any control for help (tooltips from the original SHAARP documentation).")

    # Help menu -> User Guide (the original GUIs' default 'User Guide' Functionality / welcome page).
    def _show_user_guide():
        dlg = QtWidgets.QMessageBox(win)
        dlg.setWindowTitle("SHAARP.py -- User Guide")
        dlg.setTextFormat(QtCore.Qt.RichText)
        dlg.setText(USER_GUIDE_HTML)
        # the guide links to the two ORIGINAL Mathematica repos; a QMessageBox
        # label does not open external links by default, so enable it on the standard text label
        # (guarded: if Qt ever renames it the guide still shows, the URLs just aren't clickable).
        _lbl = dlg.findChild(QtWidgets.QLabel, "qt_msgbox_label")
        if _lbl is not None:
            _lbl.setOpenExternalLinks(True)
        dlg.exec()

    help_menu = win.menuBar().addMenu("Help")
    act = help_menu.addAction("User Guide")
    act.triggered.connect(_show_user_guide)

    def _show_debug_info():
        # Everything a bug report needs, copyable in one place -- the last full
        # traceback (previously lost: the run handler kept only a one-line message), the log
        # file path, the version, and the size of the current session state.
        from . import __version__ as _ver
        from .debuglog import log_path
        try:
            n_inputs = len(win._collect_session_state())
        except Exception:
            n_inputs = -1
        tb = getattr(win, "_last_traceback", None) or "(no exception recorded this session)"
        dlg = QtWidgets.QMessageBox(win)
        dlg.setWindowTitle("Debug Info — SHAARP.py")
        dlg.setTextFormat(QtCore.Qt.PlainText)
        dlg.setText(
            f"SHAARP.py {_ver}\n"
            f"Debug log: {log_path()}\n"
            f"Session inputs tracked: {n_inputs}\n\n"
            "For a reproducible bug report: Save Session… (top bar), then attach the session\n"
            "file + the debug log.\n\n"
            f"Last exception traceback:\n{tb}")
        dlg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        dlg.addButton("Copy to clipboard", QtWidgets.QMessageBox.ActionRole)
        dlg.buttonClicked.connect(
            lambda b: QtWidgets.QApplication.clipboard().setText(dlg.text())
            if b.text().startswith("Copy") else None)
        dlg.exec()

    dbg_act = help_menu.addAction("Debug Info")
    dbg_act.triggered.connect(_show_debug_info)
    win._show_debug_info = _show_debug_info  # test hook
    def _show_about():
        dlg = QtWidgets.QMessageBox(win)
        dlg.setWindowTitle("About SHAARP.py")
        dlg.setTextFormat(QtCore.Qt.RichText)
        dlg.setText(
            f"<b>SHAARP.py v{_pkg_version}</b> &mdash; a validated Python port of the Mathematica "
            "&#9839;SHAARP.si + &#9839;SHAARP.ml second-harmonic-generation packages, merged into one "
            "desktop GUI. The computational core is validated value-by-value against live Mathematica "
            "SHAARP (agreement typically 1e-9 to 1e-15, case-dependent).<br><br>"
            "<b>Authors:</b> R. Zu, B. Wang, L. Weber, A. Saha, L.-Q. Chen &amp; V. Gopalan "
            "(The Pennsylvania State University).<br><br>"
            "<b>Please cite:</b><br>"
            "&bull; Zu, R., Wang, B., He, J. <i>et al.</i> &ldquo;Analytical and numerical modeling of "
            "optical second harmonic generation in anisotropic crystals using &#9839;SHAARP package.&rdquo; "
            "<i>npj Comput. Mater.</i> <b>8</b>, 246 (2022). <a href=\"https://doi.org/10.1038/s41524-022-00930-4\">doi:10.1038/s41524-022-00930-4</a><br>"
            "&bull; Zu, R., Wang, B., He, J. <i>et al.</i> &ldquo;Optical second harmonic generation in "
            "anisotropic multilayers with complete multireflection of linear and nonlinear waves using "
            "&#9839;SHAARP.ml package.&rdquo; <i>npj Comput. Mater.</i> <b>10</b>, 64 (2024). "
            "<a href=\"https://doi.org/10.1038/s41524-024-01229-2\">doi:10.1038/s41524-024-01229-2</a><br><br>"
            "<span style='color:#555'>Acknowledgment: U.S. DOE, Office of Science, Basic Energy "
            "Sciences, Computational Materials Sciences Program, Award No. DE-SC0020145.</span>")
        dlg.exec()

    about = help_menu.addAction("About / References")
    about.triggered.connect(_show_about)

    def _tip(widget, key):
        widget.setToolTip(TOOLTIPS[key])
        widget.setStatusTip(TOOLTIPS[key].replace("\n", " "))
        return widget

    class _WheelGuard(QtCore.QObject):
        """A mouse wheel over any input control must NEVER change its
        value — "this can easily go wrong when user scroll from sections to sections" (worst
        case: the wheel lands on the case selector and switches the whole simulated
        configuration). Flat disable: values change by typing / arrows / preset buttons only.
        The swallowed wheel is re-sent to the parent chain so the input PANEL still scrolls."""

        def eventFilter(self, obj, ev):
            if ev.type() == QtCore.QEvent.Type.Wheel:
                p = obj.parentWidget() if hasattr(obj, "parentWidget") else None
                if p is not None:
                    QtWidgets.QApplication.sendEvent(p, ev)
                return True  # the input control never sees the wheel
            return False

    class _NumBox(QtWidgets.QDoubleSpinBox):
        """Original-GUI-style numeric entry: shows exactly the typed value (no trailing-zero
        padding), no spinner arrows, high internal precision so nothing silently rounds."""

        def __init__(self, value, lo=-1e9, hi=1e9, decimals=8, width=None, arrows=False):
            super().__init__()
            self.setRange(lo, hi)
            self.setDecimals(decimals)
            self.setValue(value)
            if not arrows:
                self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            if width:
                self.setMaximumWidth(width)
            self.setKeyboardTracking(False)

        def textFromValue(self, v):
            s = f"{v:.{self.decimals()}f}".rstrip("0").rstrip(".")
            return s if s else "0"

        # (The focus-gated wheelEvent override lived here. supersedes it: flat rule is "disable mouse scrolling to change values" — the focus gate still bit him
        # because focus persists after a click, and it never covered combos or plain QSpinBoxes.
        # One mechanism now: the _WheelGuard event filter installed on every input control.)

    def make_interface_tab(which: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(page)

        # ----- INPUT PANEL (left, scrollable, grouped like the original's sub-panels) -----
        controls_host = QtWidgets.QWidget()
        form_col = QtWidgets.QVBoxLayout(controls_host)

        # "wavelength ... should be properties defined very upfront, before
        # layer definition" — the wavelength is the experiment's global property (the original .ml
        # enters λ0 up front too), so it is the FIRST input group on both tabs, above even the
        # case/preset selector.
        wavelength = _NumBox(1.064, 0.05, 20.0, decimals=6)
        _tip(wavelength, "wavelength")
        g_wave, w_lay = _collapsible_group(QtWidgets, "Wavelength Setting", QtWidgets.QFormLayout)
        w_lay.addRow("wavelength (µm)", wavelength)
        # R2 (closed): outside a case-study material's exported
        # dispersion grid the tensors CLAMP to the grid end -- say so instead of clamping silently.
        wl_note = QtWidgets.QLabel("")
        wl_note.setObjectName(f"wl_note_{which}")
        wl_note.setWordWrap(True)
        wl_note.setStyleSheet("color: #b26a00; font-size: 8pt;")
        wl_note.setVisible(False)
        w_lay.addRow("", wl_note)
        form_col.addWidget(g_wave)

        # Functionality (collapsible, like the original's expand/collapse sub-panels). The
        # dropdown lists COMPUTE MODES ONLY; on_run resolves each to its canonical compute mode.
        # (The Guide is on the Help menu + startup tab; the schematic is the persistent banner; the
        # crystal-axes view is in the orientation input group.)
        g_func, f_lay = _collapsible_group(QtWidgets, "Functionality")
        functionality = QtWidgets.QComboBox()
        functionality.addItems(list(SI_FUNCTIONALITY_DISPLAY if which == "si" else ML_FUNCTIONALITY_DISPLAY))
        functionality.setCurrentText("SHG Simulation")  # default to a computable mode
        _tip(functionality, "functionality_si" if which == "si" else "functionality_ml")
        f_lay.addWidget(functionality)
        # NO Generate-Fresnel/Maker checkboxes -- they duplicated the
        # "Maker Fringes" / "Fresnel Coefficients" functionalities ("this maker fringes plot
        # contradicts the maker fringes from functionality"). The dropdown is the ONE control:
        # selecting a mode computes that plot, fills its output tab and switches to it (and the
        # result exports, which the old checkbox path never did). Deliberate deviation from the
        # original GUI's opt-in checkboxes -- the author's own redesign.
        form_col.addWidget(g_func)

        # Case study: SI applies a predefined material; ML selects a predefined system. Both lists
        # are STRICTLY the examples the ORIGINAL packages present (case-study fidelity audit): the SI combo mirrors the original ♯SHAARP.si palette groups (all DOI cases at
        # 800 nm); the ML combo mirrors the original ♯SHAARP.ml palette (16 buttons) with
        # multi-variant materials grouped under a disabled master-material header and the paper's
        # full-system stacks as presets. Display labels carry the provenance wavelength; selections
        # resolve to registry keys via resolve_case_label.
        from .casestudy_materials import (CASE_LABEL_BY_KEY, GUI_ML_GROUPS, GUI_SI_GROUPS,
                                          resolve_case_label)

        def _disable_header_rows(combo):
            _model = combo.model()
            for _i in range(combo.count()):
                _t = combo.itemText(_i)
                if _t.startswith("—") or (_t and not _t.startswith(" ")
                                          and _t in _ML_MASTER_HEADERS):
                    _item = _model.item(_i)
                    if _item is not None:
                        _item.setEnabled(False)

        def _guard_header_selection(combo):
            """Keyboard/mouse skip disabled rows, but a PROGRAMMATIC
            setCurrentIndex/setCurrentText (session replay, scripts) can land on a header and the
            stack would then store the header string. Step to the next enabled row instead."""
            def _fix(idx, *_a):
                item = combo.model().item(idx)
                if item is None or item.isEnabled():
                    return
                for j in list(range(idx + 1, combo.count())) + list(range(idx - 1, -1, -1)):
                    it = combo.model().item(j)
                    if it is None or it.isEnabled():
                        combo.setCurrentIndex(j)
                        return
            combo.currentIndexChanged.connect(_fix)

        _ML_MASTER_HEADERS = {hdr for hdr, _e in GUI_ML_GROUPS if hdr}

        def _append_user_rows(combo):
            """The user's saved materials as a disabled-header section (only when non-empty)."""
            from .user_materials import USER_SECTION_HEADER, list_names
            names = list_names()
            if names:
                combo.addItem(USER_SECTION_HEADER)
                for _n in names:
                    combo.addItem(_n)

        si_case = QtWidgets.QComboBox()
        if which == "si":
            g_case_si, cs_lay = _collapsible_group(QtWidgets, "Case Study and Examples")
            si_case.addItem("Custom (use fields)")
            for _hdr, _entries in GUI_SI_GROUPS:
                si_case.addItem(_hdr)
                for _label, _key in _entries:
                    si_case.addItem(_label)
            _append_user_rows(si_case)
            _disable_header_rows(si_case)
            _guard_header_selection(si_case)
            _tip(si_case, "case_study")
            cs_lay.addWidget(si_case)
            form_col.addWidget(g_case_si)
        system_preset = QtWidgets.QComboBox()
        if which == "ml":
            g_case, c_lay = _collapsible_group(QtWidgets, "Case Study and Examples")
            # paper full-system presets, then each original palette material as the active film
            # (master-material headers group the multi-variant entries), then editor/custom.
            system_preset.addItems(list(ML_SYSTEM_PRESETS))
            system_preset.addItem("—  Single film in air (original palette)  —")
            for _hdr, _entries in GUI_ML_GROUPS:
                if _hdr:
                    system_preset.addItem(_hdr)
                for _label, _key in _entries:
                    system_preset.addItem(_label)
            system_preset.addItem("N-layer stack (editor)")
            system_preset.addItem("Custom film (use fields)")
            _disable_header_rows(system_preset)
            _tip(system_preset, "case_study")
            c_lay.addWidget(system_preset)
            form_col.addWidget(g_case)

        # ----- Layer Selection: the original's N-layer stack editor (.ml only) -----
        stack_state: dict = {"stack": None}
        if which == "ml":
            from .layer_stack import (CUSTOM_LAYER_CHOICE, ISOTROPIC_LAYER_CHOICE,
                                      LAYER_MATERIAL_CHOICES, default_stack,
                                      layer_role_label, set_layer_count,
                                      simple_film_stack, stack_from_system,
                                      stack_film_thickness_um, stack_halfspace_n)

            stack_state["stack"] = default_stack()
            g_layers, l_lay = _collapsible_group(QtWidgets, "Layer Selection (N-layer stack)", QtWidgets.QFormLayout)
            n_layers = QtWidgets.QSpinBox()
            # counts EVERY medium -- air/quartz/Au/air reads 4. The released .ml
            # counts only the interior films because it hardcodes air at both ends; here both
            # half-spaces are user-settable media, so they are counted and numbered like any
            # other layer. Minimum 2 = the two half-spaces alone.
            n_layers.setRange(2, 20)
            n_layers.setValue(len(stack_state["stack"]))
            _tip(n_layers, "n_layers")
            edit_layer = QtWidgets.QComboBox()
            _tip(edit_layer, "layer_select")
            layer_mat = QtWidgets.QComboBox()
            from .layer_stack import layer_material_choices as _layer_material_choices
            layer_mat.addItems(_layer_material_choices())  # palette + the user's materials
            _disable_header_rows(layer_mat)
            _guard_header_selection(layer_mat)
            _tip(layer_mat, "layer_material")
            layer_name = QtWidgets.QLineEdit()  # original .ml: "each layer can be assigned a name"
            layer_name.setPlaceholderText("(auto: role: material)")
            layer_name.setToolTip("Optional layer name shown in the layer list and the stack "
                                  "schematics; leave empty for the automatic 'role: material' label.")
            layer_thick = _NumBox(1.0, 0.0, 100000.0, decimals=6)
            _tip(layer_thick, "thickness")
            # while a layer's
            # thickness is symbolic the field displays its SYMBOL (h2), not a number the closed
            # form does not contain. The entered value stays in the spec and returns when the
            # flag clears. A stacked widget keeps the spin itself untouched.
            layer_thick_sym = QtWidgets.QLineEdit()
            layer_thick_sym.setReadOnly(True)
            layer_thick_sym.setToolTip(
                "This layer's thickness is SYMBOLIC in the Partial Analytical closed form; the "
                "number you entered is kept and returns when 'analytical h' is cleared.")
            layer_thick_stack = QtWidgets.QStackedWidget()
            layer_thick_stack.addWidget(layer_thick)      # page 0: numeric entry
            layer_thick_stack.addWidget(layer_thick_sym)  # page 1: the symbol
            layer_thick_stack.setFixedHeight(layer_thick.sizeHint().height())
            # the Dielectric Tensors panel below now displays and edits the SELECTED layer's
            # medium, including the isotropic half-spaces (eps = n^2 * I; air = the identity).
            # NO "SHG active" box -- the original has none. A layer is a
            # nonlinear source iff its POINT GROUP is in the original's "Noncentrosymmetric ->"
            # popup (SHAARP.ml.nb:5191); the "Centrosymmetric ->" popup (5630) carries an
            # all-zero d. Activity is derived in layer_stack.spec_shg_active at build time.
            l_lay.addRow("Number of Layers", n_layers)
            l_lay.addRow("Edit layer", edit_layer)
            l_lay.addRow("Layer name", layer_name)
            l_lay.addRow("Layer material", layer_mat)
            l_lay.addRow("Layer thickness (µm)", layer_thick_stack)
            # 'analytical h' is a PER-LAYER control, exactly as in the
            # original -- `SHAARP.ml.nb:5135-5146` gives every interior layer its own button
            # storing a distinct symbol h1, h2, ..., and `setup.nb:11011-11017` consumes the
            # thicknesses as a LIST, so any subset may stay symbolic while the rest are numbers.
            analytic_h_chk = QtWidgets.QCheckBox("analytical h (this layer's thickness stays symbolic)")
            analytic_h_chk.setToolTip(
                "Original .ml 'analytical h': ON keeps THIS layer's thickness a SYMBOL (h1, h2, ...) "
                "in the Partial Analytical closed form; OFF substitutes the thickness entered above. "
                "Any subset of layers may be symbolic. Not available for the two semi-infinite media "
                "(their thickness is infinite, not a variable).")
            l_lay.addRow("", analytic_h_chk)
            # the per-layer analytical-dij checkbox that used to sit here is GONE.
            # The SHG-Tensor panel's own checkbox is now the per-layer control -- it lives with
            # the tensor it symbolises, which is the same rule that moved thickness into the
            # layer editor and the media into the dielectric panel.
            _thick_label = l_lay.labelForField(layer_thick_stack)
            form_col.addWidget(g_layers)

            _loading = {"f": False}
            # the analytical-d control now lives in the SHG-Tensor panel, which is built
            # LATER in this column, so the layer editor reaches it through this holder.
            _analytic_d_hook: dict = {}

            def _refresh_thickness_display():
                """Show the SYMBOL (h2) while this layer's thickness is symbolic, the numeric
                spin otherwise. Called both when a row loads and when the flag is clicked --
                clicking only STORES the flag, so without this the panel kept showing a number
                the closed form no longer contains."""
                i = edit_layer.currentIndex()
                stack = stack_state["stack"] or []
                if not (0 <= i < len(stack)):
                    return
                from .layer_stack import layer_number
                is_half = i == 0 or i == len(stack) - 1
                sym = bool(stack[i].get("analytic_h", False)) and not is_half
                layer_thick_sym.setText(f"h{layer_number(i)}" if sym else "")
                layer_thick_stack.setCurrentIndex(1 if sym else 0)

            def _refresh_d_display():
                """While the selected layer is flagged 'analytical dij', the
                SHG-Tensor grid shows the components the closed form will contain -- the point
                group's symbolic pattern with the layer's suffix (d11m2, -d11m2, ...) and hard 0
                where symmetry forbids a component. Clearing the flag re-mirrors the numbers."""
                if _loading["f"]:
                    return
                # the symbolic grid is painted by _sync_layer_crystal_view itself (its tail
                # step reads the STACK flag), so every path that mirrors the crystal panels --
                # row selection, material change, stack reload, count change -- shows the
                # symbols a flagged layer will compute with. Before F62 only this click path
                # painted them: re-selecting a flagged row repainted NUMBERS under a ticked
                # box, and Update then took every number as KNOWN (no d symbols left).
                _sync_layer_crystal_view()

            def _apply_symbolic_d_if_flagged(i: int, stack) -> None:
                """While layer i is flagged 'analytical dij', the
                SHG-Tensor grid shows the point group's symbolic pattern with the layer's row
                suffix (d11m2, -d11m2, ...) and hard 0 where symmetry forbids a component.
                Called as the LAST step of every crystal-panel mirror so the grid can never be
                left numeric for a flagged layer. Reads the stack flag, not the checkbox (the
                checkbox may be mid-load)."""
                if not _row_symbolic_on_screen(i, stack):
                    return
                from .layer_stack import layer_number
                try:
                    sym = _symbolic_d_for_layer(point_group.currentText(), layer_number(i))
                    # a KNOWN value typed over a symbol is
                    # stored on the layer and re-shown on every display; the dependent
                    # entries (-d11m2 ...) follow by substitution.
                    known = _known_d_subs(i, stack)
                    if known:
                        sym = sym.subs(known)
                    _fill_grid(d_full_cells, sym)
                except Exception:
                    pass  # unknown point group: leave the numbers alone

            def _row_flagged(i: int, stack=None) -> bool:
                stack = stack_state["stack"] if stack is None else stack
                if not (0 <= i < len(stack)):
                    return False
                is_half = i == 0 or i == len(stack) - 1
                return bool(stack[i].get("analytic_d")) and not is_half

            def _row_symbolic_on_screen(i: int, stack=None) -> bool:
                """The grid is symbolic only while the panel computes with the symbols --
                a flagged row in Partial Analytical. In the numeric functionalities (SHG
                Simulation, Maker Fringes, ...) the flags are inert and the compute uses the
                NUMBERS, so the grid shows numbers ("the panel shows what it computes"). Caught
                by tests.test_gui_causality_gaps: the Custom-film template flags its film row
                by default, and F62 had turned its d grid symbolic under Maker Fringes."""
                return _row_flagged(i, stack) and functionality.currentText() == _PA_DISPLAY

            def _known_d_subs(i: int, stack=None) -> dict:
                """{Symbol('d11m2'): 0.3+0j, ...} for layer i's stored known components."""
                import sympy as _sp

                from .layer_stack import layer_number
                stack = stack_state["stack"] if stack is None else stack
                known = (stack[i].get("analytic_d_known") or {}) if 0 <= i < len(stack) else {}
                return {_sp.Symbol(f"{name}m{layer_number(i)}"): complex(val)
                        for name, val in known.items()}

            def _stack_known_d() -> dict | None:
                """The Update's analytical_d_known: every flagged interior layer's stored
                known components, keyed with the layer's ROW suffix (d11m2)."""
                from .layer_stack import layer_number
                out = {}
                stack = stack_state["stack"] or []
                for k, spec in enumerate(stack):
                    if _row_flagged(k, stack):
                        for name, val in (spec.get("analytic_d_known") or {}).items():
                            out[f"{name}m{layer_number(k)}"] = complex(val)
                return out or None

            def _numeric_d_for_row(i: int, spec: dict):
                """The layer's REAL numeric d tensor (3x6 complex) regardless of what the grid
                shows -- for a flagged row the grid is symbolic and must not be snapshotted."""
                if spec.get("material") == CUSTOM_LAYER_CHOICE and spec.get("custom"):
                    d = spec["custom"].get("d_full")
                    if d is not None:
                        return [[complex(v) for v in row] for row in d]
                try:
                    from .layer_stack import material_for_label as build_casestudy_material
                    mat = build_casestudy_material(resolve_case_label(spec.get("material")),
                                                   wavelength_um=wavelength.value())
                    return [[complex(v) for v in row] for row in mat.d_voigt()]
                except Exception:
                    return None

            def _snapshot_grids_for_row(i: int, spec: dict) -> dict:
                """Snapshot_with_grids() with the d grid taken from the layer's numeric
                tensor while the row is flagged (the grid then shows d11m2 symbols, which
                read_full_tensors parsed as zeros -- the GUI walkthrough SUSPECT 3)."""
                snap = snapshot_with_grids()  # late-bound (Presets section)
                if _row_symbolic_on_screen(i):
                    d = _numeric_d_for_row(i, spec)
                    if d is not None:
                        snap["d_full"] = d
                return snap

            def _commit_known_d(i: int) -> None:
                """A committed edit in a FLAGGED row's d grid declares known/unknown
                components for that layer: a number = KNOWN (incl. a typed 0), a symbol name
                or an empty cell = unknown. Stored on the spec, re-shown by the paint."""
                from .shaarp_gui import point_group_free_components
                stack = stack_state["stack"]
                if not _row_symbolic_on_screen(i, stack):
                    return
                known = {}
                for (r, c, name) in point_group_free_components(point_group.currentText()):
                    txt = d_full_cells[r][c].text().strip()
                    if not txt:
                        continue
                    try:
                        known[name] = complex(txt.replace(" ", "").replace("I", "j").replace("i", "j"))
                    except ValueError:
                        continue  # a symbol name -> unknown
                spec = dict(stack[i])
                if known:
                    spec["analytic_d_known"] = known
                else:
                    spec.pop("analytic_d_known", None)
                stack[i] = spec
                _apply_symbolic_d_if_flagged(i, stack)

            _analytic_d_hook["refresh_d"] = _refresh_d_display  # registered once
            _analytic_d_hook["row_symbolic"] = _row_symbolic_on_screen
            _analytic_d_hook["commit_known"] = _commit_known_d
            # leaving/entering Partial Analytical re-mirrors the grid (numbers <-> symbols)
            functionality.currentTextChanged.connect(lambda *_: _refresh_d_display())
            _analytic_d_hook["snapshot_for_row"] = _snapshot_grids_for_row
            _analytic_d_hook["stack_known_d"] = _stack_known_d

            def _analytic_d_checked() -> bool:
                box = _analytic_d_hook.get("box")
                return bool(box.isChecked()) if box is not None else False

            def _set_analytic_d_checked(on: bool) -> None:
                box = _analytic_d_hook.get("box")
                if box is not None:
                    box.setChecked(bool(on))

            def _refresh_layer_selector():
                _loading["f"] = True
                cur = edit_layer.currentIndex()
                edit_layer.clear()
                n = len(stack_state["stack"])
                for i in range(n):
                    # the ORIGINAL's convention — interior media are the numbered LAYERS
                    # (1..N); the half-spaces are unnumbered ambient/exit media.
                    custom_nm = stack_state["stack"][i].get("name") if stack_state["stack"] else None
                    edit_layer.addItem(layer_role_label(i, n, custom_nm))
                edit_layer.setCurrentIndex(min(max(cur, 0), n - 1) if cur >= 0 else min(1, n - 1))
                _loading["f"] = False

            def _load_layer_into_fields(*_a):
                if _loading["f"]:
                    return
                i = edit_layer.currentIndex()
                if not (0 <= i < len(stack_state["stack"])):
                    return
                spec = stack_state["stack"][i]
                _loading["f"] = True
                layer_name.setText(str(spec.get("name") or ""))
                mat = spec["material"]
                if layer_mat.findText(mat) < 0:
                    # pre-audit specs/sessions stored registry KEYS; the combo now lists labels
                    mat = CASE_LABEL_BY_KEY.get(resolve_case_label(mat), mat)
                if layer_mat.findText(mat) < 0:
                    # a user material that no longer exists in the store -> Custom (the row
                    # keeps any snapshot it carries; _my_delete rewrites rows with the spec)
                    mat = CUSTOM_LAYER_CHOICE
                layer_mat.setCurrentText(mat)
                layer_thick.setValue(spec["thickness_um"])
                # per-layer CUSTOM crystal: load this layer's saved snapshot into the entry panels
                if spec.get("material") == CUSTOM_LAYER_CHOICE and spec.get("custom"):
                    restore(spec["custom"])  # late-bound (defined in the Presets section below)
                # the two SEMI-INFINITE media carry no thickness, no SHG source and
                # no symbolic thickness -- so those controls are not merely disabled but ABSENT
                # for them ("you should not have the options of using analytical h and SHG
                # active option for both top and bottom"). The original states the same rule in
                # its own words: "(No input: thickness of first and last material is infinite)"
                # (SHAARP.ml.nb:5097-5104), and excludes the half-spaces from the source loop
                # structurally (setup.nb:10370-10374).
                is_half = i == 0 or i == len(stack_state["stack"]) - 1
                layer_thick.setEnabled(not is_half)
                analytic_h_chk.setChecked(bool(spec.get("analytic_h", False)) and not is_half)
                _refresh_thickness_display()
                _set_analytic_d_checked(bool(spec.get("analytic_d", False)) and not is_half)
                # the analytical-dij control (living in the SHG-Tensor panel) is ABSENT for
                # the two half-spaces too -- same rule as the two controls below; it
                # stayed on screen for "air in"/"air out" and a click there was silently dropped.
                _d_box = _analytic_d_hook.get("box")
                for _w in (analytic_h_chk,) + ((_d_box,) if _d_box is not None else ()):
                    _w.setVisible(not is_half)
                if _thick_label is not None:  # the original's own wording for a half-space
                    _thick_label.setText(
                        "(no input: this medium is semi-infinite)" if is_half
                        else "Layer thickness (µm)")
                _loading["f"] = False

            # the simple modes' remembered template values — film thickness plus the
            # ambient/substrate isotropic indices (the old Substrate group's numbers now live in
            # the editor's half-space rows). Written only by USER edits made in a simple mode;
            # read whenever a Custom/Film 3-layer template is rebuilt — so a preset's values
            # (e.g. Fig-4's 121.2 um) can never leak into the simple modes (the sweep once
            # caught exactly that leak as a MoS2 RA OverflowError).
            _simple_template = {"um": 1.0, "top": (1.0, 1.0), "bottom": (1.45, 1.46)}

            # dirty flag: True = the user MODIFIED the working copy of a mode whose
            # pristine compute is a factory path (named preset / Film: case / Custom film).
            # Update then computes the stack instead; (re)selecting the mode clears it
            # (stay under the example case ... reupdate the input back if you select
            # quartz+Au again"). The mode combo NEVER auto-switches (supersedes the flip-to-Custom and the flip-to-editor).
            _ml_dirty = {"on": False}

            def _set_ml_dirty(on):
                _ml_dirty["on"] = bool(on)
                # full relevance re-run: the dirty state changes the case-group hint AND the
                # lambda-ownership hints (late-bound; the pristine-only guard inside keeps a
                # dirty example's lambda untouched)
                _apply_stack_mode_relevance()

            def _store_layer_from_fields(*_a, source="structural"):
                if _loading["f"]:
                    return
                i = edit_layer.currentIndex()
                if not (0 <= i < len(stack_state["stack"])):
                    return
                old = stack_state["stack"][i]
                # a SEMI-INFINITE medium can never be an SHG source, nor carry a symbolic
                # thickness -- enforce it in the MODEL, not just by hiding the checkboxes, so a
                # stale session or preset cannot smuggle a flag back in. (The original excludes
                # the half-spaces structurally: setup.nb:10370-10374 loops i <= nM over
                # mAll[[i+1]], so index 1 and -1 never reach solveInhom.)
                is_half = i == 0 or i == len(stack_state["stack"]) - 1
                spec = {
                    "material": layer_mat.currentText(),
                    "thickness_um": float(layer_thick.value()),
                    "analytic_h": bool(analytic_h_chk.isChecked()) and not is_half,
                    "analytic_d": bool(_analytic_d_checked()) and not is_half,
                }
                if layer_name.text().strip():
                    spec["name"] = layer_name.text().strip()
                if spec["analytic_d"] and old.get("analytic_d_known"):
                    spec["analytic_d_known"] = dict(old["analytic_d_known"])
                # per-layer CUSTOM crystal: capture the entry panels INCLUDING the tensor grids
                # (latent-bug fix: the grid-less snapshot() made custom rows compute from
                # the dead scalar spins)
                if layer_mat.currentText() == CUSTOM_LAYER_CHOICE:
                    spec["custom"] = _snapshot_grids_for_row(i, old)  # numeric d if flagged
                if layer_mat.currentText() == ISOTROPIC_LAYER_CHOICE:
                    # the grids (Dielectric Tensors panel) own the values now — carry them over
                    spec["iso_n"] = list(old.get("iso_n") or [1.45, 1.46])
                # settled_unchanged: the fields still hold exactly what is already in the stack,
                # so this write-back is not a user edit and must not dirty the preset.
                settled_unchanged = _layer_specs_equal(old, spec)
                stack_state["stack"][i] = spec
                sel = system_preset.currentText()
                if sel == "N-layer stack (editor)":
                    return  # the editor mode always computes the stack
                simple = sel == "Custom film (use fields)"
                film = _ml_film_key(sel) is not None
                if (simple or film) and source == "thickness":
                    # the film row IS these modes' thickness input, and their pristine compute
                    # already reads stack_film_thickness_um — live, no divergence
                    _simple_template["um"] = float(layer_thick.value())
                    return
                # any other REAL edit diverges the working copy from the factory path
                if not settled_unchanged:
                    _set_ml_dirty(True)

            def _on_count_change(n):
                # no mode switch — a count change modifies the working copy in place
                stack_state["stack"] = set_layer_count(stack_state["stack"], int(n))
                _refresh_layer_selector()
                _load_layer_into_fields()
                _sync_layer_crystal_view()  # the guarded mirror no longer self-fires mid-refresh
                if system_preset.currentText() != "N-layer stack (editor)":
                    _set_ml_dirty(True)

            n_layers.valueChanged.connect(_on_count_change)
            edit_layer.currentIndexChanged.connect(_load_layer_into_fields)
            layer_name.editingFinished.connect(lambda: (_store_layer_from_fields(), _refresh_layer_selector()))
            layer_mat.currentTextChanged.connect(_store_layer_from_fields)
            layer_thick.valueChanged.connect(lambda *_: _store_layer_from_fields(source="thickness"))

            # ---- per-layer symbolic flags + the auto-pivot -------------------------
            # Rui: "you auto detect and pivot to partial analytical generation".
            # This is a DELIBERATE, NARROW exception to the rule that the mode combo never
            # auto-switches -- that rule exists because the flips DESTROYED user
            # state (they replaced the stack). This switch changes only WHICH OUTPUT is
            # computed: the stack, the selected layer, the case selection and every input value
            # are untouched, and clearing the last flag restores the mode you came from. Do not
            # "fix" it back to a no-switch rule without asking him.
            _auto_pa = {"prev": None}
            _PA_DISPLAY = "Partial Analytical Expressions"

            def _stack_analytic_flags():
                """(analytic_h, analytic_d) per INTERIOR layer, in stacking order."""
                stack = stack_state["stack"] or []
                return [(bool(s.get("analytic_h")), bool(s.get("analytic_d")))
                        for s in stack[1:-1]]

            page._stack_analytic_flags = _stack_analytic_flags  # test hook

            def _on_analytic_flag():
                if _loading["f"]:
                    return
                _store_layer_from_fields()
                _refresh_thickness_display()
                flagged = any(h or d for h, d in _stack_analytic_flags())
                current = functionality.currentText()
                if flagged and current != _PA_DISPLAY:
                    _auto_pa["prev"] = current
                    functionality.setCurrentText(_PA_DISPLAY)
                elif not flagged and current == _PA_DISPLAY and _auto_pa["prev"]:
                    previous, _auto_pa["prev"] = _auto_pa["prev"], None
                    functionality.setCurrentText(previous)

            def _reconcile_activity_flags_ml():
                """After a point-group pick is committed to the row, re-store the layer from
                the fields (the disabled, unchecked analytical-dij box drops the flag and its known
                values for an SHG-inactive group) and UN-pivot Partial Analytical if no flagged
                layer remains. It never pivots INTO Partial Analytical: a point-group pick is not a
                flag click (no-auto-switch rule; the exception is the click only)."""
                if _loading["f"]:
                    return
                _store_layer_from_fields()
                _refresh_thickness_display()
                flagged = any(h or d for h, d in _stack_analytic_flags())
                if not flagged and functionality.currentText() == _PA_DISPLAY and _auto_pa["prev"]:
                    previous, _auto_pa["prev"] = _auto_pa["prev"], None
                    functionality.setCurrentText(previous)

            _analytic_d_hook["reconcile"] = _reconcile_activity_flags_ml

            # clicked, NOT stateChanged: session restore writes checkbox state programmatically
            # (gui_introspect.apply_session_state) OUTSIDE the _loading guard, and a stateChanged
            # connection let that restore auto-pivot the mode into Partial Analytical.
            analytic_h_chk.clicked.connect(lambda *_: _on_analytic_flag())
            _analytic_d_hook["on_flag"] = _on_analytic_flag  # the d box (built later) calls this
            _refresh_layer_selector()
            _load_layer_into_fields()

        # Crystal structure
        g_struct, s_lay = _collapsible_group(QtWidgets, "Crystal Structure", QtWidgets.QFormLayout)
        point_group = QtWidgets.QComboBox()
        # the original's TWO popups per layer (SHAARP.ml.nb:5191 "Noncentrosymmetric ->":5630 "Centrosymmetric ->"), as two header-separated sections of one combo. Child rows
        # are NOT indented: item text == canonical label, so every currentText()/setCurrentText/
        # findText site, session replay and the test locators keep working unchanged.
        from .point_groups import POINT_GROUP_SECTIONS as _PG_SECTIONS
        for _title, _groups in _PG_SECTIONS:
            point_group.addItem(f"—  {_title}  —")
            point_group.addItems(list(_groups))
        _disable_header_rows(point_group)
        point_group.setCurrentText("-43m")
        _tip(point_group, "point_group")
        s_lay.addRow("Point group", point_group)

        def _skip_point_group_header(text):
            # a header row can only be reached programmatically / by keyboard stepping
            if text.startswith("—"):
                i = point_group.currentIndex()
                model = point_group.model()
                for j in list(range(i + 1, point_group.count())) + list(range(0, i)):
                    item = model.item(j)
                    if item is None or item.isEnabled():
                        point_group.setCurrentIndex(j)
                        return

        point_group.currentTextChanged.connect(_skip_point_group_header)
        lattice_edits = []
        lat_abc, lat_ang = QtWidgets.QHBoxLayout(), QtWidgets.QHBoxLayout()
        for name, default, row in (("a", 1.0, lat_abc), ("b", 1.0, lat_abc), ("c", 1.0, lat_abc),
                                   ("alpha", 90.0, lat_ang), ("beta", 90.0, lat_ang), ("gamma", 90.0, lat_ang)):
            e = _NumBox(default, 0.01, 10000.0, decimals=6, width=110)
            e.setPrefix(f"{name} ")
            _tip(e, "lattice")
            lattice_edits.append(e)
            row.addWidget(e)
        s_lay.addRow("Lattice (A)", lat_abc)
        s_lay.addRow("Angles (deg)", lat_ang)
        form_col.addWidget(g_struct)

        # Crystal orientation
        g_orient, o_lay = _collapsible_group(QtWidgets, "Crystal Orientation", QtWidgets.QFormLayout)
        orient_mode = QtWidgets.QComboBox()
        orient_mode.addItems(list(ORIENTATION_MODES))
        _tip(orient_mode, "orientation_mode")
        o_lay.addRow("Mode", orient_mode)
        # "make the crystal/lab rendering in crystal orientation section to be
        # 2d/3d, default to 3d". The view was already 3D-only; 2D is the new companion.
        orient_view = QtWidgets.QComboBox()
        orient_view.addItems(["3D", "2D (top view)"])
        _tip(orient_view, "orientation_view")
        o_lay.addRow("View", orient_view)
        hkl_edits, uvw_edits = [], []
        hkl_row, uvw_row = QtWidgets.QHBoxLayout(), QtWidgets.QHBoxLayout()
        for name, default, store, row, tipkey in (
            ("h", 0, hkl_edits, hkl_row, "hkl"), ("k", 0, hkl_edits, hkl_row, "hkl"), ("l", 1, hkl_edits, hkl_row, "hkl"),
            ("u", 1, uvw_edits, uvw_row, "uvw"), ("v", 0, uvw_edits, uvw_row, "uvw"), ("w", 0, uvw_edits, uvw_row, "uvw"),
        ):
            e = QtWidgets.QSpinBox()
            e.setRange(-99, 99)
            e.setValue(default)
            e.setPrefix(f"{name} ")
            e.setMaximumWidth(80)
            _tip(e, tipkey)
            store.append(e)
            row.addWidget(e)
        hkl_row_w = QtWidgets.QWidget()
        hkl_row_w.setLayout(hkl_row)
        uvw_row_w = QtWidgets.QWidget()
        uvw_row_w.setLayout(uvw_row)
        o_lay.addRow("Surface (hkl)", hkl_row_w)
        o_lay.addRow("In-plane [uvw]", uvw_row_w)
        # Crystal Physics Directions: direct Z1/Z2/Z3 (rows) in the lab frame (default = identity).
        z_edits = []  # 9 fields, row-major Z1=(z[0..2]), Z2=(z[3..5]), Z3=(z[6..8])
        z_rows_layouts = []
        z_row_ws = []
        for ri in range(3):
            zr = QtWidgets.QHBoxLayout()
            for ci in range(3):
                e = _NumBox(1.0 if ri == ci else 0.0, -1.0, 1.0, decimals=6, width=90)
                _tip(e, "z_axes")
                z_edits.append(e)
                zr.addWidget(e)
            z_rows_layouts.append(zr)
            zw = QtWidgets.QWidget()
            zw.setLayout(zr)
            z_row_ws.append(zw)
            o_lay.addRow(f"Z{ri + 1} (lab)", zw)

        # the crystal-axes (Zi) vs lab-axes (Li) view lives HERE, inside the
        # orientation group where it is actually useful WHILE entering the orientation -- not as a
        # permanent output-panel pane. It refreshes live as the orientation inputs change (wired at
        # `_refresh_orientation_view`, connected below once all deps exist).
        import numpy as _np_orient

        from .shaarp_gui import build_orientation_axes_figure as _boa_init
        from .config import CrystalOrientation as _CO_init

        orient_canvas = FigureCanvasQTAgg(
            _boa_init(_CO_init(_np_orient.eye(3)), title=r"crystal axes $Z_i$ vs lab $L_i$"))
        orient_canvas.setObjectName(f"orient_canvas_{which}")
        # near-square + width-capped: this is a 3D mplot3d figure, and a wide-short 3D canvas clips
        # its box + Li/Zi labels (same constraint the old 3D banner had; enforced by the layout
        # harness's THREED_ASPECT_MAX rule).
        orient_canvas.setMinimumSize(240, 250)
        orient_canvas.setMaximumWidth(340)
        # the unconstrained height let the 540x500 figure
        # sizeHint stretch the canvas to ~500 px with the triad centred in blank background.
        # 280 px keeps the 3D-aspect rule (340 <= 1.7 * 280) and the [240, 320] min-height fence.
        orient_canvas.setMaximumHeight(280)
        _tip(orient_canvas, "z_axes")
        o_lay.addRow(orient_canvas)
        form_col.addWidget(g_orient)

        def _orient_visibility():
            mode = orient_mode.currentText()
            miller = mode == "Miller (hkl + in-plane uvw)"
            zmode = mode.startswith("Crystal Physics")
            for e in hkl_edits + uvw_edits:
                e.setEnabled(miller)
            for e in z_edits:
                e.setEnabled(zmode)
            # the rows the
            # selected mode does not consume are HIDDEN, not just greyed -- z-cut shows neither,
            # Miller shows hkl/uvw only, Crystal-Physics shows the Z rows only. QFormLayout rows
            # hide via the field widget + its buddy label.
            def _row_visible(field, on):
                field.setVisible(on)
                lbl = o_lay.labelForField(field)
                if lbl is not None:
                    lbl.setVisible(on)
            for w in (hkl_row_w, uvw_row_w):
                _row_visible(w, miller)
            for w in z_row_ws:
                _row_visible(w, zmode)

        orient_mode.currentTextChanged.connect(lambda *_: _orient_visibility())
        _orient_visibility()

        def _z_axes():
            import numpy as _np
            v = [float(e.value()) for e in z_edits]
            m = _np.array([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]], dtype=float)
            # the z-axis cells are 6-decimal, but a case study's rotation
            # matrix needs ~1e-7 orthogonality (config.py). Selecting a rotated case (TaAs, GaAs(111),
            # x-cut LNO) then switching to Custom left the 6-decimal-TRUNCATED matrix in the cells,
            # whose orthogonality residual (~7e-7) exceeded the tolerance -> a cryptic "Crystal axes
            # must be mutually orthogonal" error. SVD-snap a NEAR-orthonormal entry to the nearest
            # orthonormal matrix (drift ~1.6e-8, physically negligible); a GROSSLY non-orthonormal
            # entry (user typo) is left as-is so the validator still flags it clearly.
            try:
                rows = m / (_np.linalg.norm(m, axis=1, keepdims=True) + 1e-300)
                resid = float(_np.max(_np.abs(rows @ rows.T - _np.eye(3))))
                if 1e-9 < resid < 1e-2 and _np.all(_np.isfinite(m)):
                    U, _s, Vt = _np.linalg.svd(m)
                    m = U @ Vt  # nearest orthonormal (Frobenius) to the entered matrix
            except Exception:
                pass  # never let the snap itself break the reader; the validator handles bad input
            return m.tolist()

        # ------------------------------------------------------------------
        # Tensor entry -- Mathematica-style FULL MATRICES (the primary entry):
        # * Dielectric ε_ij(ω), ε_ij(2ω): full 3×3, SYMMETRIC (ε_ij = ε_ji).
        # * SHG d_ij: the FULL 3×6 Voigt tensor, every component editable.
        # The condensed refractive-index / free-d widgets are still CREATED (the
        # case-study + ML snapshot code and the build_custom_* signatures reference
        # them) but NOT shown -- the matrices are what the user sees/edits and what
        # drives the custom-mode compute (the validated full-tensor path).
        # ------------------------------------------------------------------
        def _dspin(v):
            b = _NumBox(v, 0.1, 30.0, decimals=8)
            _tip(b, "dielectric")
            return b

        n_w, n_2w, ne_w, ne_2w = _dspin(2.0), _dspin(2.2), _dspin(2.0), _dspin(2.2)
        # the SI tab's incident (entrance) medium — an isotropic index pair, default air.
        # (The ML tab's incident medium is the stack's FIRST LAYER in the editor — .)
        inc_n_w, inc_n_2w = _dspin(1.0), _dspin(1.0)

        # hidden free-d map (kept only for the build_custom_* d_free arg; the full d matrix overrides it)
        d_field_map: dict[tuple[int, int], QtWidgets.QDoubleSpinBox] = {}

        def rebuild_d_fields():
            d_field_map.clear()
            for (r, c, _name) in point_group_free_components(point_group.currentText()):
                d_field_map[(r, c)] = _NumBox(1.0, -1000.0, 1000.0, decimals=6)

        rebuild_d_fields()

        g_epsm, epsm_lay = _collapsible_group(
            QtWidgets, "Dielectric Tensors (εᵢⱼ at ω and 2ω, symmetric)", QtWidgets.QVBoxLayout)
        _tip(g_epsm, "dielectric")
        if which == "si":
            # the SI incident medium —
            # the entrance medium above the crystal, isotropic, air = 1 — lives here rather
            # than in its own group. (On ML the incident medium is the stack's first layer.)
            inc_form = QtWidgets.QWidget()
            inc_lay = QtWidgets.QFormLayout(inc_form)
            inc_lay.setContentsMargins(0, 0, 0, 4)
            _tip(inc_n_w, "incident_n")
            _tip(inc_n_2w, "incident_n")
            inc_lay.addRow("incident medium n (ω)", inc_n_w)
            inc_lay.addRow("incident medium n (2ω)", inc_n_2w)
            epsm_lay.addWidget(inc_form)
        # an ISOTROPIC medium is one number per frequency. When the
        # selected ML layer is isotropic (air / isotropic-n rows) these two spins are shown and
        # the 3x3 grids are HIDDEN; anisotropic layers show the grids and hide the spins.
        eps_mode = QtWidgets.QComboBox()
        eps_mode.addItems(["Complex Dielectric Permittivity ε̃", "Complex Refractive Index ñ"])
        _tip(eps_mode, "eps_mode")
        _mode_form = QtWidgets.QWidget()
        _mode_lay = QtWidgets.QFormLayout(_mode_form)
        _mode_lay.setContentsMargins(0, 0, 0, 4)
        _mode_lay.addRow("enter as", eps_mode)
        epsm_lay.addWidget(_mode_form)
        medium_form = QtWidgets.QWidget()
        medium_lay = QtWidgets.QFormLayout(medium_form)
        medium_lay.setContentsMargins(0, 0, 0, 4)
        medium_n_w = _NumBox(1.0, 0.1, 30.0, decimals=8)
        medium_n_2w = _NumBox(1.0, 0.1, 30.0, decimals=8)
        _tip(medium_n_w, "medium_n")
        _tip(medium_n_2w, "medium_n")
        medium_lay.addRow("refractive index n (ω)", medium_n_w)
        medium_lay.addRow("refractive index n (2ω)", medium_n_2w)
        if which == "ml":
            epsm_lay.addWidget(medium_form)
        medium_form.setVisible(False)  # shown only while an isotropic layer is selected
        # - this is the ORIGINAL's `LinearInput` SetterBar,
        # restored: `SHAARP_V1.04.nb:35215-35235` offers "Complex Refractive Index ñ" /
        # "Complex Dielectric Permittivity ε̃" over the SAME two 3x3 grids, and `:5452-5520`
        # converts BOTH ways with MatrixPower -- eps = n.n (a matrix square, NOT elementwise)
        # and n = eps^(1/2). Switching therefore CONVERTS what is on screen, so the panel always
        # describes the same medium; epsilon stays the internal truth everywhere else.
        eps_w_w, eps_w_cells = _matrix_block(QtWidgets, "ε(ω) =", [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]])
        eps_2w_w, eps_2w_cells = _matrix_block(QtWidgets, "ε(2ω) =", [[4.84, 0, 0], [0, 4.84, 0], [0, 0, 4.84]])
        epsm_lay.addWidget(eps_w_w)
        epsm_lay.addWidget(eps_2w_w)
        _wire_symmetric_grid(eps_w_cells)
        _wire_symmetric_grid(eps_2w_cells)
        form_col.addWidget(g_epsm)

        g_dm, dm_lay = _collapsible_group(
            QtWidgets, "SHG Tensor dᵢⱼ (full 3×6 Voigt, pm/V)", QtWidgets.QVBoxLayout)
        _tip(g_dm, "shg_tensor")
        d_full_w, d_full_cells = _matrix_block(
            QtWidgets, "dᵢⱼ =", [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        dm_lay.addWidget(d_full_w)
        analytic_d_chk = QtWidgets.QCheckBox("analytical dᵢⱼ (this layer's SHG tensor stays symbolic)")
        analytic_d_chk.setToolTip(
            "Original .ml 'analytical dij' (SHAARP.ml.nb:7256-7292), per layer: ON keeps THIS "
            "layer's SHG tensor symbolic in the Partial Analytical closed form and shows the "
            "symmetry-allowed components by name (d11m2, d14m2, ... suffixed with the layer "
            "number), with 0 where the point group forbids a component. Type a NUMBER over a "
            "component to declare it KNOWN: it is substituted and only the remaining unknowns "
            "stay symbolic. OFF substitutes this layer's entered tensor.")
        if which == "ml":
            dm_lay.addWidget(analytic_d_chk)
            # this IS the per-layer flag now -- register it with the layer editor and let a
            # USER click (clicked, never stateChanged: a programmatic setChecked during session
            # restore must not move anything) store it and drive the auto-pivot.
            _analytic_d_hook["box"] = analytic_d_chk

            def _on_analytic_d_clicked(*_a):
                hook = _analytic_d_hook.get("on_flag")
                if hook is not None:
                    hook()  # store the flag on the layer + auto-pivot
                refresh = _analytic_d_hook.get("refresh_d")
                if refresh is not None:
                    refresh()  # and show what the expression will contain

            analytic_d_chk.clicked.connect(_on_analytic_d_clicked)
        else:
            analytic_d_chk.setVisible(False)
        form_col.addWidget(g_dm)

        # ---- the point group decides SHG activity AND the crystal system ----------------
        _g_dm_base_title = g_dm.title()
        _shg_hint_state = {"inactive": None}

        def _apply_shg_activity_hint(*_a):
            """SHG-inactive point group (the original's "Centrosymmetric ->" list, d = 0 by
            symmetry): the d grid is zero, the SHG-Tensor group collapses with a title hint
            (house rule: hint + collapse) and the analytical-dij control is unchecked AND
            disabled (it sits inside the group, so its own enabled flag is authoritative).
            An SHG-active group restores the pattern and the control. An isotropic ML row is
            owned by _apply_layer_medium_hints and is left alone."""
            from .point_groups import is_shg_active
            pg = point_group.currentText()
            if pg.startswith("—"):
                return
            if which == "ml":
                _i = edit_layer.currentIndex()
                _stack = stack_state["stack"] or []
                _spec = _stack[_i] if 0 <= _i < len(_stack) else {}
                if _spec.get("material") in ("air", ISOTROPIC_LAYER_CHOICE):
                    _shg_hint_state["inactive"] = None
                    return
            inactive = not is_shg_active(pg)
            if inactive:
                import numpy as _np
                _fill_grid(d_full_cells, _np.zeros((3, 6)))
                g_dm.setTitle(_g_dm_base_title + "   — not used: SHG-inactive point group (d ≡ 0)")
                g_dm.setChecked(False)  # idempotent: the ML medium hint re-expands before calling us
                # The stack flag is reconciled AFTER the pick has been committed to the row
                # (_on_panel_user_edit -> _to_custom / the Custom branch): un-pivoting here,
                # mid-change, would re-mirror the palette material's group into the combo.
                analytic_d_chk.setChecked(False)
                analytic_d_chk.setEnabled(False)
            else:
                g_dm.setTitle(_g_dm_base_title)
                analytic_d_chk.setEnabled(True)
                if _shg_hint_state["inactive"] is True:
                    g_dm.setChecked(True)
                    populate_matrices(d_only=True)  # the pattern is back
            _shg_hint_state["inactive"] = inactive

        _lattice_lock_state = {"busy": False}

        def _enforce_crystal_system(*_a):
            """Rui: "ensure crystal symmetry consistency and lattice consistency" -- the
            point group's crystal system LOCKS the dependent cell parameters (cubic a=b=c and
            90 deg angles, hexagonal/trigonal b=a and gamma=120, ...). Locked spins are coerced
            (signal-blocked) and greyed out; an edit of `a` under a locked system propagates.
            Idempotent, never marks the case dirty, runs during loads too (it cannot flip a
            mode). Every palette lattice already satisfies its rule, so presets are untouched."""
            from .point_groups import (apply_lattice_constraints, is_known_point_group,
                                       lattice_constraints)
            if _lattice_lock_state["busy"]:
                return
            pg = point_group.currentText()
            if pg.startswith("—") or not is_known_point_group(pg):
                for e in lattice_edits:
                    e.setEnabled(True)
                return
            _lattice_lock_state["busy"] = True
            try:
                lock = lattice_constraints(pg)
                vals = [float(e.value()) for e in lattice_edits]
                new = apply_lattice_constraints(pg, vals)
                changed = False
                for k, (e, v) in enumerate(zip(lattice_edits, new)):
                    if abs(float(e.value()) - v) > 1e-9:
                        e.blockSignals(True)
                        e.setValue(float(v))
                        e.blockSignals(False)
                        changed = True
                    e.setEnabled(k not in lock.locked_indices)
                if changed:
                    _refresh_orientation_view()  # late-bound (Orientation section)
            finally:
                _lattice_lock_state["busy"] = False

# (the SI incident-medium rows moved INTO the Dielectric Tensors group — : media entry
# shares the dielectric section. See the g_epsm construction.)

        for _row in (eps_w_cells + eps_2w_cells + d_full_cells):
            for _e in _row:
                _tip(_e, "full_tensor")

        # ---- the n / epsilon entry mode (the original's LinearInput) --------------------
        # Internal truth is ALWAYS epsilon; the mode is a display/entry transform applied at two
        # seams only -- reading the grids and filling them.
        def _index_mode() -> bool:
            return eps_mode.currentIndex() == 1

        def _matrix_square(mat):
            import numpy as _np

            m = _np.asarray(mat, dtype=complex)
            return m @ m  # the original's MatrixPower[n, 2] -- NOT elementwise

        def _matrix_sqrt(mat):
            """Principal matrix square root (the original's MatrixPower[eps, 0.5])."""
            import numpy as _np

            m = _np.asarray(mat, dtype=complex)
            w, v = _np.linalg.eig(m)
            return v @ _np.diag(_np.sqrt(w.astype(complex))) @ _np.linalg.inv(v)

        def _eps_from_display(grid):
            """Grid values as EPSILON (squares them when the panel is in index mode)."""
            return _matrix_square(grid).tolist() if _index_mode() else grid

        def _display_from_eps(mat):
            """EPSILON as the panel should show it (its matrix square root in index mode)."""
            return _matrix_sqrt(mat).tolist() if _index_mode() else mat

        def _fill_eps(cells, mat):
            _fill_grid(cells, _display_from_eps(mat))

        def _on_eps_mode_changed(*_a):
            """Switching the entry mode CONVERTS what is on screen (choice), so the panel
            keeps describing the SAME medium -- a change of units, never of physics. Read the
            grids in the mode we are LEAVING, then re-render in the mode we are entering."""
            leaving_index_mode = not _index_mode()  # the combo has already moved
            raw_w = [[_parse_complex(e.text()) for e in row] for row in eps_w_cells]
            raw_2 = [[_parse_complex(e.text()) for e in row] for row in eps_2w_cells]
            eps_w_val = _matrix_square(raw_w).tolist() if leaving_index_mode else raw_w
            eps_2_val = _matrix_square(raw_2).tolist() if leaving_index_mode else raw_2
            _fill_eps(eps_w_cells, eps_w_val)
            _fill_eps(eps_2w_cells, eps_2_val)
            label_w, label_2 = ("ñ(ω) =", "ñ(2ω) =") if _index_mode() else ("ε(ω) =", "ε(2ω) =")
            for _blk, _txt in ((eps_w_w, label_w), (eps_2w_w, label_2)):
                _lbls = [c for c in _blk.findChildren(QtWidgets.QLabel)
                         if c.text().startswith(("ε(", "ñ("))]
                if _lbls:
                    _lbls[0].setText(_txt)

        eps_mode.currentIndexChanged.connect(_on_eps_mode_changed)

        # The full matrices ALWAYS drive the custom-mode compute (the original's full-tensor entry).
        def read_full_tensors():
            """(eps_w 3×3, eps_2w 3×3, d 3×6) complex lists read from the matrix grids.

            Always EPSILON, whichever entry mode the panel is in."""
            ew = [[_parse_complex(e.text()) for e in row] for row in eps_w_cells]
            e2 = [[_parse_complex(e.text()) for e in row] for row in eps_2w_cells]
            dd = [[_parse_complex(e.text()) for e in row] for row in d_full_cells]
            return _eps_from_display(ew), _eps_from_display(e2), dd

        # the standalone "film thickness" spin and its "Wavelength Setting / Layer Thickness"
        # group are GONE (you already have layer thickness definition in layer definition").
        # Wavelength moved to the top-of-column group (g_wave, built first); thickness lives ONLY
        # in the Layer Selection editor, whose stack is now the single geometric truth for every
        # ML mode (presets load their real stack; the simple film modes are 3-layer templates).

        # ---- keep the full-matrix grids populated with the ACTIVE material's tensors ----
        # Selecting a case study (SI) / a film row (ML) loads that material's ε(ω), ε(2ω), d into
        # the matrices so the user SEES the real numbers (Mathematica behaviour); editing them then
        # drives the custom-mode compute. Changing the point group in custom mode resets d to that
        # symmetry pattern. Wavelength change re-loads ε for a dispersive case material.
        def _has_symbols(matrix) -> bool:
            try:
                return any(getattr(v, "free_symbols", None) for row in matrix for v in row)
            except TypeError:
                return False

        def _symbolic_d_for_layer(pg: str, row_number: int):
            """The point group's d pattern with every symbol suffixed by the layer's row number."""
            import sympy as _sp

            from .symbolic import d_voigt_symbolic
            base = _sp.Matrix(d_voigt_symbolic(pg))
            return base.subs({sym: _sp.Symbol(f"{sym.name}m{row_number}")
                              for sym in sorted(base.free_symbols, key=str)})

        def _fmt_cell(z):
            # a SYMBOLIC entry (a sympy expression carrying free symbols, e.g. d11m2 or
            # -d11m2) renders as its name -- that is how a flagged layer's d tensor shows the
            # components the closed form will actually contain, with the point group's zeros
            # preserved. Numbers keep the 6-significant-digit form.
            if hasattr(z, "free_symbols") and getattr(z, "free_symbols", None):
                return str(z)
            z = complex(z)
            if z.real == 0:
                z = complex(0.0, z.imag)  # never render "-0" (a -d14 with d14 = 0)
            if abs(z.imag) < 1e-9:
                return f"{z.real:.6g}"
            return f"{z.real:.6g}{'+' if z.imag >= 0 else '-'}{abs(z.imag):.6g}j"

        def _fill_grid(cells, matrix):
            import numpy as _np

            M = _np.asarray(matrix, dtype=object) if _has_symbols(matrix) else _np.asarray(matrix)
            for r in range(len(cells)):
                for c in range(len(cells[r])):
                    cells[r][c].blockSignals(True)
                    cells[r][c].setText(_fmt_cell(M[r][c]))
                    cells[r][c].blockSignals(False)

        def _ml_film_key(sel: str) -> str | None:
            """Registry key when the ML combo selection is a single-film case row, else None
            (presets / N-layer / Custom / header rows resolve to None)."""
            from .casestudy_materials import CASE_STUDY_ORDER
            if sel in ML_SYSTEM_PRESETS or sel in ("N-layer stack (editor)", "Custom film (use fields)"):
                return None
            key = resolve_case_label(sel)
            return key if key in CASE_STUDY_ORDER else None

        def _active_case_material():
            """The Material whose tensors should fill the matrices, or None for custom/default."""
            try:
                from .layer_stack import material_for_label as build_casestudy_material
                if which == "si" and si_case.currentText() != "Custom (use fields)":
                    return build_casestudy_material(resolve_case_label(si_case.currentText()),
                                                    wavelength_um=wavelength.value())
                if which == "ml":
                    key = _ml_film_key(system_preset.currentText())
                    if key:
                        return build_casestudy_material(key, wavelength_um=wavelength.value())
            except Exception:
                return None
            return None

        def populate_matrices(d_only=False):
            try:
                mat = _active_case_material()
                if mat is not None:
                    if not d_only:
                        _fill_eps(eps_w_cells, mat.eps_w())
                        _fill_eps(eps_2w_cells, mat.eps_2w())
                    _fill_grid(d_full_cells, mat.d_voigt())
                    return
                # custom / default: d from the point-group symmetry pattern; ε diagonal from n fields
                from .shaarp_gui import build_constrained_d_voigt as _bcd
                free = {(r, c): 1.0 for (r, c, _n) in point_group_free_components(point_group.currentText())}
                _fill_grid(d_full_cells, _bcd(point_group.currentText(), free))
                if not d_only:
                    import numpy as _np
                    _fill_eps(eps_w_cells, _np.diag([n_w.value() ** 2, n_w.value() ** 2, ne_w.value() ** 2]))
                    _fill_eps(eps_2w_cells, _np.diag([n_2w.value() ** 2, n_2w.value() ** 2, ne_2w.value() ** 2]))
            except Exception:
                pass

        populate_matrices()  # initial fill so the matrices are never confusingly blank/zero

        def _sync_case_panel(mat):
            """Mirror the WHOLE input panel to the selected case-study material -- point group,
            lattice, and orientation, not just the tensor grids. Before this sync the panel kept
            showing stale defaults (point group '-43m', identity orientation) while the compute used
            the material, so the schematic/title and the visible inputs contradicted each other
            ."""
            import numpy as _np

            st = mat.structure
            from .point_groups import canonical_point_group as _cpg
            pg = _cpg(str(st.point_group))  # every palette group (m3m, 6/mmm, inf inf m) is listed now
            if point_group.findText(pg) >= 0:
                point_group.blockSignals(True)
                point_group.setCurrentText(pg)
                point_group.blockSignals(False)
            for widget, value in zip(lattice_edits,
                                     (st.a, st.b, st.c, st.alpha_deg, st.beta_deg, st.gamma_deg)):
                # lattice edits now flip-to-Custom like the orientation spins, so this
                # programmatic mirror must be signal-blocked or selecting a case would flip itself.
                widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(False)
            _enforce_crystal_system()      # a fixed point for every palette lattice
            _apply_shg_activity_hint()     # F63 (signals were blocked above)
            rot = _np.asarray(mat.orientation.rotation_matrix(), dtype=float)
            # Programmatic mirror of the case's orientation: block signals so this sync is never
            # mistaken for a USER edit (user edits to these controls flip the case to Custom so the
            # edit is what gets computed).
            orient_mode.blockSignals(True)
            if _np.max(_np.abs(rot - _np.eye(3))) < 1e-9:
                orient_mode.setCurrentText(ORIENTATION_MODES[0])  # z-cut (identity)
                rot = _np.eye(3)
            else:
                orient_mode.setCurrentText(ORIENTATION_MODES[2])  # Crystal Physics Directions
            orient_mode.blockSignals(False)
            for idx, widget in enumerate(z_edits):
                widget.blockSignals(True)
                widget.setValue(float(rot[idx // 3, idx % 3]))
                widget.blockSignals(False)
            # the blocked setCurrentText above skips the currentTextChanged hook, so refresh the
            # per-mode enabled states explicitly — otherwise the freshly filled Z1/Z2/Z3 cells stay
            # greyed from the previous mode and the case's orientation reads as "not updated"
            # (report).
            _orient_visibility()

        def _refresh_orientation_view(*_a):
            """Repaint the crystal-axes (Zi) vs lab-axes (Li) figure in the orientation group
            whenever the orientation inputs change. Self-contained figure swap (does NOT depend on
            the later-defined _replace_canvas_figure), so it is safe to call at construction time."""
            try:
                from .shaarp_gui import (build_orientation, build_orientation_axes_figure,
                                          build_orientation_axes_figure_2d)
                from .config import CrystalStructure

                mat = _active_case_material()
                if mat is not None and getattr(mat, "orientation", None) is not None:
                    orient = mat.orientation  # a case study drives the panel; use its orientation
                else:
                    latt = [w.value() for w in lattice_edits]
                    st = CrystalStructure(point_group=point_group.currentText(),
                                          a=latt[0], b=latt[1], c=latt[2],
                                          alpha_deg=latt[3], beta_deg=latt[4], gamma_deg=latt[5])
                    orient = build_orientation(
                        st, mode=orient_mode.currentText(),
                        surface_hkl=tuple(e.value() for e in hkl_edits),
                        in_plane_uvw=tuple(e.value() for e in uvw_edits),
                        z_axes=_z_axes())
                _mk = (build_orientation_axes_figure_2d
                       if orient_view.currentText().startswith("2D")
                       else build_orientation_axes_figure)
                fig = _mk(orient, title=r"crystal axes $Z_i$ vs lab $L_i$")
                orient_canvas.figure.clear()
                orient_canvas.figure = fig
                fig.set_canvas(orient_canvas)
                try:
                    fig.set_size_inches(max(orient_canvas.width(), 10) / fig.dpi,
                                        max(orient_canvas.height(), 10) / fig.dpi)
                except Exception:
                    pass
                orient_canvas.draw_idle()
            except Exception:
                pass  # invalid transient orientation (e.g. uvw not in hkl plane) -> keep last good

        def _on_case_changed():
            # Loading a REAL case study fills the matrices from its tensors; switching to "Custom"
            # leaves the current values untouched so the user can edit them.
            mat = _active_case_material()
            if mat is not None:
                _sync_case_panel(mat)
                populate_matrices()
            _refresh_orientation_view()

        # orientation-view live wiring: repaint on every input that changes the orientation
        orient_mode.currentTextChanged.connect(_refresh_orientation_view)
        point_group.currentTextChanged.connect(_refresh_orientation_view)
        for _e in (z_edits + hkl_edits + uvw_edits + lattice_edits):
            _e.valueChanged.connect(_refresh_orientation_view)
        _refresh_orientation_view()  # initial paint (self-contained swap; safe here)

        # a user edit NEVER switches the mode. "Stay under the example case,
        # and you may reupdate the input back if you select [it] again." A panel edit writes
        # into the working copy (on ML: the SELECTED layer of the stack; on SI: the panels
        # themselves) and, when the active mode's pristine compute is a factory path, marks it
        # DIRTY so Update computes the modified copy; re-selecting the example resets.
        def _on_panel_user_edit(surface="struct", commit=True, cell=None):
            if which == "si":
                t = si_case.currentText()
                if t != "Custom (use fields)" and not t.startswith("—"):
                    _set_si_dirty(True)  # late-bound (defined with the SI relevance hooks)
                return
            if _loading["f"]:
                return  # programmatic panel writes (mirror / restore) are not user edits
            i = edit_layer.currentIndex()
            stack = stack_state["stack"]
            if not (0 <= i < len(stack)):
                return
            spec = stack[i]
            mat = spec.get("material")
            sel = system_preset.currentText()
            named = sel in ML_SYSTEM_PRESETS or _ml_film_key(sel) is not None
            simple = sel == "Custom film (use fields)"
            if surface == "d" and _analytic_d_hook["row_symbolic"](i, stack):
                # typing a number over a SYMBOL in a flagged row declares that component
                # KNOWN for this layer (partial-analytical tensor) -- it is not an edit of the
                # material's tensor, so the row is NOT converted to Custom.
                if not commit:
                    return
                _analytic_d_hook["commit_known"](i)
                if named or simple:
                    _set_ml_dirty(True)
                return

            def _reconcile_activity_flags():
                """After a point-group pick is committed to the row, re-store the layer
                from the fields -- the (now disabled, unchecked) analytical-dij box drops the
                flag and its known values for an SHG-inactive group, and the un-pivot runs
                only if no flagged layer remains."""
                hook = _analytic_d_hook.get("reconcile")
                if hook is not None:
                    hook()

            def _to_custom(note):
                # carry the per-layer flags across the conversion. Before this, editing a
                # palette layer's tensors silently cleared its analytical h/d flags (and any
                # isotropic-n entry), so a converted layer quietly stopped being symbolic.
                new = {"material": CUSTOM_LAYER_CHOICE,
                       "thickness_um": float(spec.get("thickness_um") or 0.0),
                       "analytic_h": bool(spec.get("analytic_h", False)),
                       "analytic_d": bool(spec.get("analytic_d", False)),
                       "custom": _analytic_d_hook["snapshot_for_row"](i, spec)}
                if spec.get("name"):
                    new["name"] = spec["name"]
                if spec.get("analytic_d_known"):
                    new["analytic_d_known"] = dict(spec["analytic_d_known"])
                stack[i] = new
                _loading["f"] = True
                try:
                    layer_mat.setCurrentText(CUSTOM_LAYER_CHOICE)
                finally:
                    _loading["f"] = False
                _refresh_layer_selector()
                win.statusBar().showMessage(note)
                if named or simple:
                    _set_ml_dirty(True)
                if surface == "struct":
                    _reconcile_activity_flags()

            if mat in ("air", ISOTROPIC_LAYER_CHOICE):
                if surface in ("medium_n", "eps_w", "eps_2w"):
                    # an isotropic medium is ONE number per frequency — the scalar spins
                    # are the entry surface (the grids are hidden for these rows). The eps
                    # branch stays for programmatic/legacy callers and reads the same way.
                    if surface == "medium_n":
                        n_w_val, n_2w_val = medium_n_w.value(), medium_n_2w.value()
                        real_ok = n_w_val > 0 and n_2w_val > 0
                        s_w, s_2 = n_w_val ** 2, n_2w_val ** 2
                    else:
                        ew, e2, _dd = read_full_tensors()
                        s_w, s_2 = _scalar_eps(ew), _scalar_eps(e2)
                        real_ok = (s_w is not None and s_2 is not None
                                   and abs(complex(s_w).imag) <= 1e-9 * max(abs(complex(s_w)), 1.0)
                                   and abs(complex(s_2).imag) <= 1e-9 * max(abs(complex(s_2)), 1.0)
                                   and complex(s_w).real > 0 and complex(s_2).real > 0)
                    if real_ok:
                        import math
                        n_pair = [math.sqrt(complex(s_w).real), math.sqrt(complex(s_2).real)]
                        keep = spec.get("name")
                        stack[i] = {"material": ISOTROPIC_LAYER_CHOICE, "thickness_um": 0.0,
                                    "iso_n": n_pair}
                        if keep:
                            stack[i]["name"] = keep
                        if mat == "air":  # the row's combo follows its new kind
                            _loading["f"] = True
                            try:
                                layer_mat.setCurrentText(ISOTROPIC_LAYER_CHOICE)
                            finally:
                                _loading["f"] = False
                        if simple or _ml_film_key(sel) is not None:
                            if i == 0:
                                _simple_template["top"] = (n_pair[0], n_pair[1])
                            elif i == len(stack) - 1:
                                _simple_template["bottom"] = (n_pair[0], n_pair[1])
                        if named:
                            _set_ml_dirty(True)
                        _apply_layer_medium_hints()
                        return
                    if not commit:
                        return  # transient typing — judge at commit
                if not commit and surface in ("eps_w", "eps_2w"):
                    return
                if i == 0:
                    win.statusBar().showMessage(
                        "Row 1 is the incident medium — the solver requires it isotropic; "
                        "entry reverted")
                    _sync_layer_crystal_view()
                    return
                _to_custom(f"Layer {i + 1} is now a Custom layer (anisotropic entry).")
                return
            if mat == CUSTOM_LAYER_CHOICE:
                spec["custom"] = _analytic_d_hook["snapshot_for_row"](i, spec)
                if named:
                    _set_ml_dirty(True)
                if surface == "struct":
                    _reconcile_activity_flags()
                return
            # palette-material row: the edit takes it over as a Custom layer (the panels
            # currently mirror that material, so the copy is faithful + the edit)
            _to_custom(f"Layer {i + 1} is now a Custom layer carrying your edit.")

        def _propagate_iso_diagonal(cells, cell=None):
            """While an isotropic row is selected, the eps grid acts as ONE scalar entry: a
            diagonal edit propagates to the other two diagonal cells (signal-blocked)."""
            src = None
            probe = cell if cell is not None else QtWidgets.QApplication.focusWidget()
            for k in range(3):
                if cells[k][k] is probe:
                    src = k
                    break
            if src is None:
                return
            txt = cells[src][src].text()
            for k in range(3):
                if k != src:
                    cells[k][k].blockSignals(True)
                    cells[k][k].setText(txt)
                    cells[k][k].blockSignals(False)

        # F57 SI dirty machinery: an SI panel edit under a selected case study stays under it
        # (the compute uses the edited panels; re-selecting the case resets the panels).
        _si_dirty = {"on": False}
        _si_case_base = g_case_si.title() if which == "si" else ""

        def _set_si_dirty(on):
            _si_dirty["on"] = bool(on)
            if which == "si":
                g_case_si.setTitle(_si_case_base + (
                    "   — edited (re-select to reset)" if _si_dirty["on"] else ""))

        if which == "si":
            def _si_clear_dirty(*_a):
                _set_si_dirty(False)

            def _si_example_reset(*_a):
                populate_matrices()  # reload the case's panels (same-item re-pick)
                _sync_case_wavelength()
                _set_si_dirty(False)

            si_case.currentTextChanged.connect(_si_clear_dirty)
            si_case.activated.connect(_si_example_reset)

        def _orient_user_edit(*_a):
            _on_panel_user_edit("struct")
            _refresh_orientation_view()

        orient_mode.activated.connect(_orient_user_edit)
        orient_view.currentTextChanged.connect(_refresh_orientation_view)
        for _e in (z_edits + hkl_edits + uvw_edits):
            _e.valueChanged.connect(lambda *_: _on_panel_user_edit("struct"))
        point_group.activated.connect(lambda *_: _on_panel_user_edit("struct"))
        for _e in lattice_edits:
            _e.valueChanged.connect(lambda *_: _on_panel_user_edit("struct"))

        def _update_wl_note(*_a):
            """Show which selected case-study material(s) the entered wavelength falls OUTSIDE of
            (their tensors clamp to the nearest tabulated value) -- residual risk R2 made visible."""
            from .casestudy_materials import casestudy_lambda_range
            names: list[str] = []
            if which == "si":
                t = si_case.currentText()
                if t != "Custom (use fields)" and not t.startswith("—"):
                    names.append(resolve_case_label(t))
            else:
                sel = system_preset.currentText()
                film_key = _ml_film_key(sel)
                if film_key:
                    names.append(film_key)
                elif sel == "N-layer stack (editor)":
                    names += [resolve_case_label(s.get("material")) for s in stack_state["stack"]
                              if s.get("material") and s.get("material") != CUSTOM_LAYER_CHOICE
                              and s.get("material") != "air"]
            lam = float(wavelength.value())
            out = []
            for nm in names:
                rng = casestudy_lambda_range(nm)
                if rng and not (rng[0] - 1e-12 <= lam <= rng[1] + 1e-12):
                    out.append(f"{nm}: tabulated {rng[0]:g}–{rng[1]:g} µm")
            if out:
                wl_note.setText("⚠ λ outside the exported dispersion range — tensors clamped to "
                                "the nearest tabulated value (" + "; ".join(out) + ")")
                wl_note.setVisible(True)
            else:
                wl_note.setVisible(False)

        point_group.currentTextChanged.connect(lambda *_: populate_matrices(d_only=True))
        point_group.currentTextChanged.connect(_apply_shg_activity_hint)
        point_group.currentTextChanged.connect(_enforce_crystal_system)
        for _e in lattice_edits:
            _e.valueChanged.connect(_enforce_crystal_system)               # a -> b, c
        _enforce_crystal_system()      # the construction-time group ("-43m") locks too
        _apply_shg_activity_hint()
        for _surface, _rows in (("eps_w", eps_w_cells), ("eps_2w", eps_2w_cells),
                                ("d", d_full_cells)):
            for _row in _rows:
                for _e in _row:
                    _e.textEdited.connect(
                        lambda *_a, e=_e, s=_surface: _on_panel_user_edit(
                            s, commit=False, cell=e))
                    # editingFinished also fires on mere focus-out: only a cell the user
                    # actually MODIFIED commits (isModified is set by typing, cleared by
                    # the programmatic setText of the mirror/populate fills)
                    _e.editingFinished.connect(
                        lambda e=_e, s=_surface: (_on_panel_user_edit(s, commit=True, cell=e)
                                                  if e.isModified() else None))
        if which == "ml":
            # the scalar index spins are the isotropic-medium entry surface
            medium_n_w.valueChanged.connect(lambda *_: _on_panel_user_edit("medium_n"))
            medium_n_2w.valueChanged.connect(lambda *_: _on_panel_user_edit("medium_n"))
        # the wavelength spin is CASE-OWNED whenever the active case
        # material carries SINGLE-LAMBDA data (every SI palette case; the fixed-lambda ML films):
        # the dataset defines lambda, any other spin value is clamp-ignored by the dispersion
        # lookup, so the spin syncs to the case's lambda, the row label says so, and any later
        # write (user edit or session restore) snaps back. Multi-lambda datasets keep the spin
        # user-owned — it genuinely drives the dispersion lookup (the R2 clamp note covers the
        # edges). Named ML presets are owned by the preset sync instead.
        _wl_row_lbl = w_lay.labelForField(wavelength)
        _wl_case_guard = {"on": False}

        def _active_case_lambda():
            """(case lambda, is_single_lambda) for the active case material, else (None, False)."""
            from .casestudy_materials import casestudy_lambda_range
            name = None
            if which == "si":
                t = si_case.currentText()
                if t != "Custom (use fields)" and not t.startswith("—"):
                    from .user_materials import get as _user_entry
                    _ue = _user_entry(t)
                    if _ue is not None:
                        return float(_ue["wavelength_um"]), True  # saved at one wavelength
                    name = resolve_case_label(t)
            else:
                sel = system_preset.currentText()
                if sel in ML_SYSTEM_PRESETS:
                    return None, False  # the preset sync owns lambda there
                fk = _ml_film_key(sel)
                if fk:
                    name = fk
            if not name:
                return None, False
            try:
                lo, hi = casestudy_lambda_range(name)
            except Exception:
                return None, False
            return float(lo), abs(float(hi) - float(lo)) < 1e-12

        def _sync_case_wavelength(*_a):
            lam, single = _active_case_lambda()
            if single and not _wl_case_guard["on"]:
                _wl_case_guard["on"] = True
                try:
                    wavelength.setValue(lam)
                finally:
                    _wl_case_guard["on"] = False
            if _wl_row_lbl is not None and which == "si":
                # (the ML row label is owned by the stack-mode handler, which folds this in)
                _wl_row_lbl.setText("wavelength (µm)" + (
                    "   — set by the case (single-λ data)" if single else ""))

        def _reassert_case_wavelength(*_a):
            if _wl_case_guard["on"]:
                return
            lam, single = _active_case_lambda()
            if single and abs(wavelength.value() - lam) > 1e-12:
                _sync_case_wavelength()

        wavelength.valueChanged.connect(_reassert_case_wavelength)

        stack_mode_hook = None  # set on the ml branch below; stays None on si
        if which == "si":
            si_case.currentTextChanged.connect(lambda *_: _on_case_changed())
            si_case.currentTextChanged.connect(_update_wl_note)
            si_case.currentTextChanged.connect(_sync_case_wavelength)  # F54 case-owned lambda
            wavelength.valueChanged.connect(
                lambda *_: populate_matrices() if si_case.currentText() != "Custom (use fields)" else None)
            wavelength.valueChanged.connect(_update_wl_note)
        else:
            # the layer editor is the single stack truth. Selecting a stack mode LOADS the
            # matching stack into the editor — a named preset's REAL layers (screenshot
            # showed the editor stuck on default LiNbO3 while Fig-4 computed quartz+Au), or the
            # simple modes' 3-layer air/film/substrate template. Connected FIRST on
            # currentTextChanged so every later handler sees the loaded stack. All programmatic
            # editor writes run under the _loading guard (+ blockSignals on n_layers), so the user-edit flip cannot self-fire on a selection.
            def _load_stack_into_editor(new_stack, select_index=1):
                stack_state["stack"] = new_stack
                n_layers.blockSignals(True)
                n_layers.setValue(len(new_stack))  # every medium counts
                n_layers.blockSignals(False)
                _refresh_layer_selector()
                # the index change below fires BEFORE the fields reload — keep the crystal-panel
                # mirror suppressed until state is consistent, then fire it once (the mirror
                # early-returns for Custom rows, so user-owned panels are never clobbered)
                _loading["f"] = True
                edit_layer.setCurrentIndex(min(select_index, len(new_stack) - 1))
                _loading["f"] = False
                _load_layer_into_fields()
                _sync_layer_crystal_view()
                _refresh_orientation_view()

            def _enter_stack_mode(*_a):
                sel = system_preset.currentText()
                if sel == "N-layer stack (editor)":
                    # Ownership handoff: the editor keeps whatever stack is current (the
                    # simple templates' half-spaces are real isotropic layers, directly
                    # buildable — no conversion step exists anymore).
                    return
                if sel == "Custom film (use fields)":
                    _load_stack_into_editor(
                        simple_film_stack(CUSTOM_LAYER_CHOICE, _simple_template["um"],
                                          top_n=_simple_template["top"],
                                          bottom_n=_simple_template["bottom"]))
                    try:
                        # F57 seed: the film row carries the panels' tensors from the start, so
                        # Update computes the FILM even while a half-space row is mirrored
                        stack_state["stack"][1]["custom"] = snapshot_with_grids()
                    except Exception:
                        pass  # startup ordering: the snapshot closure binds later
                elif _ml_film_key(sel) is not None:
                    _load_stack_into_editor(
                        simple_film_stack(sel.strip(), _simple_template["um"],
                                          top_n=_simple_template["top"],
                                          bottom_n=_simple_template["bottom"]))
                elif sel in ML_SYSTEM_PRESETS:
                    try:
                        _load_stack_into_editor(
                            stack_from_system(resolve_ml_system_preset(sel)))
                    except Exception:
                        pass  # header rows / resolve failures: leave the stack unchanged

            system_preset.currentTextChanged.connect(_enter_stack_mode)
            system_preset.currentTextChanged.connect(lambda *_: _on_case_changed())
            system_preset.currentTextChanged.connect(_update_wl_note)

            # ---- R15: the layer STACK joins the session payload. The widget walk
            # cannot see stack_state, so an N-layer-editor session used to restore onto the
            # rebuilt base stack. The page exports a payload/apply pair the session save/load
            # functions call alongside the widget state. ----
            def _ml_stack_payload():
                from .layer_stack import encode_stack
                return {
                    "mode": system_preset.currentText(),
                    "stack": encode_stack(stack_state["stack"]),
                    "simple_template": {"um": float(_simple_template["um"]),
                                        "top": list(_simple_template["top"]),
                                        "bottom": list(_simple_template["bottom"])},
                    "edit_row": int(edit_layer.currentIndex()),
                    # whether the named-preset / simple-mode working copy was EDITED --
                    # an edited copy is what the user actually computed and must survive a
                    # restore; a pristine one keeps rebuilding deterministically (R15 rationale).
                    "dirty": bool(_ml_dirty["on"]),
                }

            def _ml_stack_apply(payload):
                """Apply a saved stack payload AFTER the widget state (the mode combo is already
                restored). The saved STACK is applied when the mode is the N-layer editor, and —
                when a preset/simple mode's working copy was EDITED (`dirty`): what the user
                computed must survive the restore. A PRISTINE preset/simple mode
                still rebuilds deterministically, so a stale payload can never diverge a preset
                definition (the original R15 rationale). The simple-template memory (film
                thickness + halfspace n) applies always."""
                from .layer_stack import decode_stack
                try:
                    st = payload.get("simple_template") or {}
                    if "um" in st:
                        _simple_template["um"] = float(st["um"])
                    if "top" in st:
                        _simple_template["top"] = tuple(float(x) for x in st["top"])
                    if "bottom" in st:
                        _simple_template["bottom"] = tuple(float(x) for x in st["bottom"])
                    # The SAVED mode is authoritative. The widget replay restores the combo
                    # early, but replaying the crystal-panel cells afterwards fires the ownership hooks, which can flip the mode (an editor session used to land
                    # in "Custom film") — re-assert it before deciding what the stack needs.
                    want_mode = payload.get("mode")
                    if (isinstance(want_mode, str)
                            and system_preset.findText(want_mode) >= 0
                            and system_preset.currentText() != want_mode):
                        system_preset.setCurrentText(want_mode)
                    sel = system_preset.currentText()
                    simple = sel == "Custom film (use fields)" or _ml_film_key(sel) is not None
                    named = sel in ML_SYSTEM_PRESETS
                    edited = bool(payload.get("dirty")) and payload.get("stack")
                    if (simple or named) and edited:
                        # the EDITED working copy of a preset/simple mode
                        # is what the user computed -- restore it exactly (e.g. a user material
                        # or Custom row inside Fig-4) and re-mark the case "-- edited". A
                        # PRISTINE preset still rebuilds from the factory below, so a stale
                        # payload can never diverge a preset definition (the R15 rationale).
                        stack = decode_stack(payload["stack"])
                        _load_stack_into_editor(
                            stack, select_index=min(int(payload.get("edit_row", 1)),
                                                    len(stack) - 1))
                        _set_ml_dirty(True)
                    elif simple:
                        _enter_stack_mode()  # rebuild the template from the restored memory
                    elif sel == "N-layer stack (editor)" and payload.get("stack"):
                        stack = decode_stack(payload["stack"])
                        _load_stack_into_editor(
                            stack, select_index=min(int(payload.get("edit_row", 1)),
                                                    len(stack) - 1))
                except Exception:
                    pass  # a malformed stack payload must never block a session restore

            page._ml_stack_payload = _ml_stack_payload
            page._ml_stack_apply = _ml_stack_apply
            wavelength.valueChanged.connect(
                lambda *_: populate_matrices() if _ml_film_key(system_preset.currentText()) is not None else None)
            wavelength.valueChanged.connect(_update_wl_note)

            def _sync_layer_crystal_view(*_a):
                """Mirror the crystal panels (point group, lattice, orientation, tensors) to the
                LAYER currently selected in the N-layer editor -- previously they kept showing
                stale defaults while the compute used the layer's real material (audit
                F10c). Custom layers keep whatever the user is editing."""
                if _loading["f"]:
                    # during a programmatic stack load the selector/material signals fire
                    # mid-transition with the OLD row's material still in layer_mat — mirroring
                    # then wiped USER panel state (a Miller orientation was reset to the stale
                    # case's Crystal Physics Directions). _load_stack_into_editor fires this
                    # mirror once explicitly after the load, with consistent state.
                    return
                i = edit_layer.currentIndex()
                stack = stack_state["stack"] or []
                spec = stack[i] if 0 <= i < len(stack) else {}
                name = layer_mat.currentText()
                if name == CUSTOM_LAYER_CHOICE:
                    # panels were restored by _load_layer_into_fields; additionally show the
                    # snapshot's GRIDS when present (previously the grids kept the
                    # previous row's tensors)
                    c = spec.get("custom") or {}
                    if c.get("eps_omega_full") is not None:
                        _fill_eps(eps_w_cells, c["eps_omega_full"])
                    if c.get("eps_2omega_full") is not None:
                        _fill_eps(eps_2w_cells, c["eps_2omega_full"])
                    if c.get("d_full") is not None:
                        _fill_grid(d_full_cells, c["d_full"])
                    _apply_symbolic_d_if_flagged(i, stack)
                    _apply_layer_medium_hints()
                    return
                if name in ("air", ISOTROPIC_LAYER_CHOICE):
                    # an isotropic half-space DISPLAYS its true medium —
                    # eps = n^2 * I (air = the identity), point group 1, no d, no orientation.
                    # and it displays it as ONE NUMBER per frequency (the grids hide).
                    from .layer_stack import _material_from_iso
                    n = ((1.0, 1.0) if name == "air"
                         else tuple(spec.get("iso_n") or (1.45, 1.46)))
                    mat = _material_from_iso(float(n[0]), float(n[1]))
                    _sync_case_panel(mat)
                    _fill_eps(eps_w_cells, mat.eps_w())  # kept in sync (hidden, but exported)
                    _fill_eps(eps_2w_cells, mat.eps_2w())
                    _fill_grid(d_full_cells, mat.d_voigt())
                    medium_n_w.blockSignals(True)
                    medium_n_2w.blockSignals(True)
                    medium_n_w.setValue(float(n[0]))
                    medium_n_2w.setValue(float(n[1]))
                    medium_n_w.blockSignals(False)
                    medium_n_2w.blockSignals(False)
                    _apply_layer_medium_hints()
                    return
                try:
                    from .layer_stack import material_for_label as build_casestudy_material
                    mat = build_casestudy_material(resolve_case_label(name),
                                                   wavelength_um=wavelength.value())
                except Exception:
                    _apply_layer_medium_hints()
                    return
                _sync_case_panel(mat)
                _fill_eps(eps_w_cells, mat.eps_w())
                _fill_eps(eps_2w_cells, mat.eps_2w())
                _fill_grid(d_full_cells, mat.d_voigt())
                _apply_symbolic_d_if_flagged(i, stack)  # flagged layer stays symbolic
                _apply_layer_medium_hints()

            _medium_hint_bases = {}

            def _apply_layer_medium_hints(*_a):
                """F57 title hints: the tensor group names WHOSE medium it shows; for isotropic
                media the d/structure/orientation groups collapse with a 'not used' note (
                pattern — hint + collapse, never disable)."""
                for grp in (g_epsm, g_dm, g_struct, g_orient):
                    if grp not in _medium_hint_bases:
                        _medium_hint_bases[grp] = grp.title()
                i = edit_layer.currentIndex()
                stack = stack_state["stack"] or []
                spec = stack[i] if 0 <= i < len(stack) else {}
                mat = spec.get("material", "")
                # ONE number per layer, the original's — interior layers are "layer k"
                # (k = 1..N), the half-spaces are named media with no layer number.
                role = layer_role_label(i, len(stack), spec.get("name"))
                iso = mat in ("air", ISOTROPIC_LAYER_CHOICE)
                if mat == "air":
                    desc = f"{role} (air, ε = I)"
                elif mat == ISOTROPIC_LAYER_CHOICE:
                    n = spec.get("iso_n") or [1.45, 1.46]
                    desc = (f"{role} (isotropic n(ω)={float(n[0]):g}, "
                            f"n(2ω)={float(n[1]):g})")
                elif mat == CUSTOM_LAYER_CHOICE:
                    desc = f"{role} (Custom fields)"
                else:
                    desc = f"{role} — {mat}" if spec.get("name") else f"{role}: {mat}"
                g_epsm.setTitle(_medium_hint_bases[g_epsm] + "   — " + desc)
                for grp in (g_dm, g_struct, g_orient):
                    grp.setChecked(not iso)
                    grp.setTitle(_medium_hint_bases[grp] + (
                        "   — not used: isotropic medium" if iso else ""))
                # ONE number per frequency for an isotropic medium — the scalar rows show
                # and the 3x3 grids hide (and vice versa). The grids stay populated underneath
                # (eps = n^2*I) so exports/snapshots keep carrying the full tensor form.
                medium_form.setVisible(bool(iso))
                eps_w_w.setVisible(not iso)
                eps_2w_w.setVisible(not iso)
                # the n-vs-eps switch governs the TENSOR grids ("either refractive index
                # in a tensor form or dielectric permittivities"). An isotropic medium shows a
                # single index instead, so the switch has nothing to act on and would only
                # contradict the visible "refractive index n" rows -- hide it there.
                _mode_form.setVisible(not iso)
                if not iso:
                    _apply_shg_activity_hint()

            layer_mat.currentTextChanged.connect(_sync_layer_crystal_view)
            layer_mat.currentTextChanged.connect(_update_wl_note)
            layer_mat.currentTextChanged.connect(_refresh_orientation_view)
            edit_layer.currentIndexChanged.connect(_sync_layer_crystal_view)
            edit_layer.currentIndexChanged.connect(_refresh_orientation_view)

            def _ml_sync_from_preset(*_a):
                """Named preset or the layer editor selected: mirror the crystal panels to the
                preset's FILM layer (or the editor's selected layer). Covers the startup state and
                the named presets that the case-material hook misses (audit follow-up)."""
                sel = system_preset.currentText()
                if sel == "N-layer stack (editor)":
                    _sync_layer_crystal_view()
                    return
                if _active_case_material() is not None or sel == "Custom film (use fields)":
                    return  # film rows handled by _on_case_changed; custom stays user-owned
                try:
                    sys_ = resolve_ml_system_preset(sel)
                except Exception:
                    return
                if len(sys_.layers) > 1:
                    mat = sys_.layers[1].material
                    _sync_case_panel(mat)
                    _fill_eps(eps_w_cells, mat.eps_w())
                    _fill_eps(eps_2w_cells, mat.eps_2w())
                    _fill_grid(d_full_cells, mat.d_voigt())
                    # a factory preset material can carry a PLACEHOLDER cell
                    # (the Fig-4 docs quartz is built with a = b = c = 1), which the crystal-
                    # system lock then coerced into a visible [1,1,1,90,90,120]. The editor's
                    # palette-mapped row carries the real lattice -- re-mirror from it.
                    _sync_layer_crystal_view()

            system_preset.currentTextChanged.connect(_ml_sync_from_preset)
            _ml_sync_from_preset()  # startup: panel mirrors the default preset's film layer

            # the wavelength SYNCS to a named preset when the preset is
            # (re)selected, and is thereafter just an input — an edit under the example marks
            # the working copy modified (dirty) and is KEPT; re-selecting the example resets it.
            # No value is ever snapped back or restored from a stash.
            _g_wave_base = g_wave.title()
            _g_case_base = g_case.title()
            _wl_lbl = w_lay.labelForField(wavelength)

            _stack_sync_guard = {"on": False}

            def _apply_example_dirty_hint(*_a):
                g_case.setTitle(_g_case_base + (
                    "   — edited (re-select to reset)" if _ml_dirty["on"] else ""))

            def _apply_stack_mode_relevance(*_a):
                sel = system_preset.currentText()
                is_custom = sel == "Custom film (use fields)"
                is_film = _ml_film_key(sel) is not None
                is_stack = sel == "N-layer stack (editor)"
                is_preset = not (is_custom or is_film or is_stack)
                if is_preset and not _ml_dirty["on"] and not _stack_sync_guard["on"]:
                    _stack_sync_guard["on"] = True
                    try:  # a PRISTINE example selection loads the preset's TRUE lambda
                        # (a dirty example keeps the user's lambda)
                        _sys = resolve_ml_system_preset(sel)
                        wavelength.setValue(float(_sys.wavelength_um))
                    except Exception:
                        pass
                    finally:
                        _stack_sync_guard["on"] = False
                g_wave.setTitle(_g_wave_base + (
                    "   — set by the selected preset" if is_preset and not _ml_dirty["on"]
                    else ""))
                _case_lam, _case_single = _active_case_lambda()
                if _wl_lbl is not None:
                    _wl_lbl.setText("wavelength (µm)" + (
                        "   — set by the preset" if (is_preset and not _ml_dirty["on"]) else
                        "   — set by the case (single-λ data)"
                        if (is_film and _case_single and not _ml_dirty["on"])
                        else ""))
                _apply_example_dirty_hint()

            def _clear_ml_dirty(*_a):
                # every mode (re)selection starts pristine — the loaders just rebuilt the copy
                _ml_dirty["on"] = False
                _apply_example_dirty_hint()
                _apply_stack_mode_relevance()

            system_preset.currentTextChanged.connect(_clear_ml_dirty)
            system_preset.currentTextChanged.connect(_apply_stack_mode_relevance)
            system_preset.currentTextChanged.connect(_sync_case_wavelength)  # F54 film-mode lambda

            def _ml_example_reset(*_a):
                """Same-item re-pick of the example (activated fires; currentTextChanged does
                not): re-run the loaders so the working copy returns to the example's true
                values — the author's reset gesture ("reupdate the input back if you select quartz+Au
                again")."""
                _enter_stack_mode()
                _ml_sync_from_preset()
                _sync_case_wavelength()
                _clear_ml_dirty()

            system_preset.activated.connect(_ml_example_reset)

            def _wl_user_edit(*_a):
                if _stack_sync_guard["on"] or _loading["f"]:
                    return
                sel = system_preset.currentText()
                # only a NAMED PRESET carries its own lambda; in Film: modes lambda drives the
                # dispersive lookup live and in Custom/editor it is a plain input
                if (sel != "Custom film (use fields)" and sel != "N-layer stack (editor)"
                        and _ml_film_key(sel) is None):
                    _set_ml_dirty(True)  # a lambda edit under the example modifies the copy

            wavelength.valueChanged.connect(_wl_user_edit)
            _enter_stack_mode()  # startup: the editor mirrors the initial selection's stack
            _apply_stack_mode_relevance()  # startup state
            stack_mode_hook = _apply_stack_mode_relevance  # test hook (exposed on the page below)
        _sync_case_wavelength()  # F54 startup: lambda + label reflect the initial selection
        _update_wl_note()  # startup: correct clamp-note visibility for the initial selection

        # (the "Calculation Controls" group is gone -- its two Generate-* checkboxes moved
        # into the Functionality group at the top of the column, shown only for SHG Simulation.)

        # Assumptions (.ml): FMR/JK/HH + the FMR backward/standing-wave sub-options.
        assumption_combo = QtWidgets.QComboBox()
        assumption_combo.addItems(list(ML_ASSUMPTIONS))
        fmr_submode = QtWidgets.QComboBox()
        from .shaarp_gui import FMR_SUBMODES

        fmr_submode.addItems(list(FMR_SUBMODES))
        _tip(fmr_submode, "fmr_submode")
        if which == "ml":
            g_assum, a_lay = _collapsible_group(QtWidgets, "Assumptions", QtWidgets.QFormLayout)
            _tip(assumption_combo, "assumptions")
            a_lay.addRow("multiple-reflection", assumption_combo)
            a_lay.addRow("FMR: backward waves", fmr_submode)
            # the sub-options apply only to Full (FMR)
            assumption_combo.currentTextChanged.connect(
                lambda t: fmr_submode.setEnabled(ML_ASSUMPTIONS.get(t, 0) == 0))
            form_col.addWidget(g_assum)

        # Scan-range spins (Maker / Fresnel angle sweep). Constructed here so the dedicated
        # "Scan Range" group can be placed BEFORE Polarimetry Settings (reorg); their default
        # 0.5 deg step keeps the swept curves smooth out of the box (angle-independent setup is
        # hoisted out of the sweep loops, so a fine scan stays responsive).
        th_min = _NumBox(0.0, 0.0, 89.0, decimals=3)
        th_max = _NumBox(45.0, 0.0, 89.9, decimals=3)
        th_step = _NumBox(0.5, 0.01, 30.0, decimals=3)
        # Fresnel scan range controlled SEPARATELY from Maker Fringes,
        # each in its own separately-toggled section; Fresnel default step = 0.1 deg.
        fr_min = _NumBox(0.0, 0.0, 89.0, decimals=3)
        fr_max = _NumBox(89.9, 0.0, 89.9, decimals=3)
        fr_step = _NumBox(0.1, 0.01, 30.0, decimals=3)
        g_scan = g_fres = None
        if which == "ml":
            for b in (th_min, th_max, th_step):
                _tip(b, "theta_range")
            # scan range is NOT
            # polarimetry -- give it its own group beside Calculation Controls, with each spin's
            # quick presets inline on the same row. (RA azimuth points moved to Polarimetry Settings, -- RA is a polarimetry-class measurement, not a Maker/Fresnel angle sweep.)
            g_scan, sc_lay = _collapsible_group(QtWidgets, "Maker Fringes Scan Range", QtWidgets.QFormLayout)
            sc_lay.addRow("θ min (deg)", _spin_row(QtWidgets, th_min,
                                                   _angle_buttons(QtWidgets, [0, 5, 10, 20, 30], th_min)))
            sc_lay.addRow("θ max (deg)", _spin_row(QtWidgets, th_max,
                                                   _angle_buttons(QtWidgets, [30, 45, 60, 80, 89], th_max)))
            sc_lay.addRow("θ step (deg)", _spin_row(QtWidgets, th_step,
                                                    _angle_buttons(QtWidgets, [0.1, 0.5, 1, 2, 5], th_step)))
            form_col.addWidget(g_scan)
            for b in (fr_min, fr_max, fr_step):
                _tip(b, "fresnel_range")
            g_fres, fs_lay = _collapsible_group(QtWidgets, "Fresnel Coefficients Scan Range",
                                                QtWidgets.QFormLayout)
            fs_lay.addRow("θ min (deg)", _spin_row(QtWidgets, fr_min,
                                                   _angle_buttons(QtWidgets, [0, 5, 10, 20, 30], fr_min)))
            fs_lay.addRow("θ max (deg)", _spin_row(QtWidgets, fr_max,
                                                   _angle_buttons(QtWidgets, [30, 45, 60, 80, 89], fr_max)))
            fs_lay.addRow("θ step (deg)", _spin_row(QtWidgets, fr_step,
                                                    _angle_buttons(QtWidgets, [0.1, 0.5, 1, 2, 5], fr_step)))
            form_col.addWidget(g_fres)

        # Polarimetry settings (now polarimetry ONLY -- scan range moved out)
        g_pol, p_lay = _collapsible_group(QtWidgets, "Polarimetry Settings", QtWidgets.QFormLayout)
        # NORMAL INCIDENCE is the default on both tabs. theta = 0 is the
        # singular edge (`_desingularize_theta_deg` maps it to 1e-3 deg for the solvers), and the
        # frozen-exe gui-smoke drives it for every tab x functionality with 0 errors.
        # Max: the .si original allows 0-90, the .ml original 0-89 -- the port had clamped BOTH to
        # 89, so only SI was wrong. 89.9 rather than 90 because the solver's interval is open at 90.
        theta_spin = _NumBox(0.0, 0.0, 89.9 if which == "si" else 89.0, decimals=3, arrows=True)
        theta_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        theta_slider.setRange(0, 89)
        theta_slider.setValue(int(theta_spin.value()))
        _tip(theta_spin, "theta")
        _tip(theta_slider, "theta")
        theta_slider.valueChanged.connect(lambda v: theta_spin.setValue(float(v)))
        theta_spin.valueChanged.connect(lambda v: theta_slider.setValue(int(v)))
        p_lay.addRow("incident angle θⁱ (deg)",
                     _spin_row(QtWidgets, theta_spin, _angle_buttons(QtWidgets, [0, 15, 30, 45, 60, 75], theta_spin)))
        p_lay.addRow("", theta_slider)
        # rich text so the phase renders as a real superscript (not raw "e^{iΔδ}" braces)
        formula_lbl = QtWidgets.QLabel("E = E₀ ( cos φ ,  sin φ · e<sup>iΔδ</sup> ,  0 )")
        formula_lbl.setStyleSheet("color:#555; font-style: italic;")
        p_lay.addRow("incident field", formula_lbl)
        # Incident Polarization phi: Rotate / Fix Polarizer (+ fixed-phi field + quick buttons)
        polarizer_mode = QtWidgets.QComboBox()
        polarizer_mode.addItems(["Rotate Polarizer", "Fix Polarizer"])
        _tip(polarizer_mode, "polarizer")
        fixed_phi = _NumBox(0.0, -360.0, 360.0, decimals=3)
        _tip(fixed_phi, "polarizer")
        fixed_phi.setEnabled(False)
        fixed_phi_btns = _angle_buttons(QtWidgets, [0, 45, 90, 135, 180], fixed_phi)
        fixed_phi_btns.setEnabled(False)
        p_lay.addRow("incident polarization φ", polarizer_mode)
        p_lay.addRow("fixed φ (deg)", _spin_row(QtWidgets, fixed_phi, fixed_phi_btns))
        # the original .ml prints this convention as a VISIBLE row in the polarimetry
        # column -- it decides whether a reported number is p- or s-polarized, and the port had it
        # in tooltips only. Rendered as a label row so it cannot be missed.
        _ps_note = QtWidgets.QLabel("0° = p-polarized · 90° = s-polarized   (φ and ψ)")
        _ps_note.setStyleSheet("color:#555; font-style: italic;")
        p_lay.addRow("", _ps_note)
        # incident ellipticity Delta-delta
        ellipticity = _NumBox(0.0, -360.0, 360.0, decimals=3)
        _tip(ellipticity, "ellipticity")
        p_lay.addRow("ellipticity Δδ (deg)", ellipticity)
        # the original .ml carries a SECOND, Maker-Fringes-specific Delta-delta restricted to
        # -90..90 with a quick-set bar. Keeping the general +-360 box AND this one mirrors that.
        maker_ell = None
        if which == "ml":
            maker_ell = _NumBox(0.0, -90.0, 90.0, decimals=3)
            _tip(maker_ell, "maker_ellipticity")
            maker_ell_btns = _angle_buttons(QtWidgets, [-90, -60, -30, 0, 30, 60, 90], maker_ell)
            p_lay.addRow("Maker Fringes Δδ (deg)", _spin_row(QtWidgets, maker_ell, maker_ell_btns))
        # output polarization (analyzer): match ♯SHAARP.si -- just Rotating or Fixed.
        # The two former "Rotate Analyzer" entries collapse into ONE, controlled by the offset field below:
        # offset = 0 gives the parallel + perpendicular channels (the default polarimetry), and a
        # nonzero offset co-rotates the analyzer at ψ = φ + offset.
        analyzer_mode = QtWidgets.QComboBox()
        analyzer_mode.addItems(["Rotate Analyzer", "Fix Analyzer"])
        analyzer_mode.setCurrentText("Fix Analyzer")  # default = fixed
        _tip(analyzer_mode, "analyzer")
        analyzer_psi = _NumBox(0.0, -360.0, 360.0, decimals=3)
        _tip(analyzer_psi, "analyzer")
        analyzer_psi.setEnabled(True)   # default mode is Fixed analyzer ψ
        psi_btns = _angle_buttons(QtWidgets, [0, 45, 90, 135, 180], analyzer_psi)
        psi_btns.setEnabled(True)
        # the ORIGINAL's rotating-polarizer/rotating-analyzer offset: "For a rotating polarizer,
        # rotating analyzer setup, the analyzer-polarizer offset angle may be entered." (fidelity FB1)
        analyzer_offset = _NumBox(0.0, -360.0, 360.0, decimals=3)
        analyzer_offset.setToolTip("Analyzer–polarizer offset angle (deg) for the Rotating analyzer: "
                                   "0 = the parallel + perpendicular channels; nonzero co-rotates the "
                                   "analyzer at ψ = φ + offset (perpendicular channel at +90°).")
        analyzer_offset.setEnabled(False)  # default mode is Fixed analyzer ψ -> offset (Rotating-only) greyed
        p_lay.addRow("analyzer", analyzer_mode)
        p_lay.addRow("analyzer ψ (deg)", _spin_row(QtWidgets, analyzer_psi, psi_btns))
        p_lay.addRow("analyzer–polarizer offset (deg)", analyzer_offset)
        # - SAMPLE ROTATION, the original's `samplerotationcontrol` block:
        # `{{samplerotationcontrol, False, "Sample Rotation phi: "}, {True->"Rotate Sample",
        # False->"Fix Sample"}}` + `Dynamic[If[samplerotationcontrol, RotatePolarizer = False;
        # RotateAnalyzer = False; {{samplerotatestep, 10, "Step Size: "}, 0, 360} + SetterBar
        # 10/20/30]]`. Turning the sample IS SHG Simulation with the polarizer and analyzer held
        # still, so it belongs here and not in the functionality list (which cut back to 4).
        sample_mode = sample_step = sample_dir = None
        if which == "ml":
            sample_mode = QtWidgets.QComboBox()
            sample_mode.addItems(["Rotate Sample", "Fix Sample"])
            sample_mode.setCurrentText("Fix Sample")  # the original's default (False)
            sample_step = _NumBox(10.0, 0.5, 360.0, decimals=3)  # 0.5 deg == 721 points
            sample_step_btns = _angle_buttons(QtWidgets, [10, 20, 30], sample_step)  # original SetterBar
            sample_dir = QtWidgets.QComboBox()
            sample_dir.addItems(["CCW (counter-clockwise)", "CW (clockwise)"])
            for _w in (sample_mode, sample_step, sample_dir):
                _tip(_w, "sample_rotation")
            p_lay.addRow("sample rotation ψₛ", sample_mode)
            p_lay.addRow("step size Δψₛ (deg)", _spin_row(QtWidgets, sample_step, sample_step_btns))
            p_lay.addRow("rotation direction", sample_dir)

            def _sample_rotating():
                return bool(sample_mode.isEnabled()
                            and sample_mode.currentText().startswith("Rotate"))

            page._sample_rotation_state = lambda: {
                "on": _sample_rotating(), "step_deg": float(sample_step.value()),
                "ccw": sample_dir.currentText().startswith("CCW"),
                "rotate_polarizer": polarizer_mode.currentText().startswith("Rotate"),
                "rotate_analyzer": analyzer_mode.currentText().startswith("Rotate Analyzer"),
                "analyzer_offset_deg": float(analyzer_offset.value()),
            }

        # ONE function owns every polarimetry widget's enabled state, from
        # (functionality, polarizer, analyzer, sample) together. This replaces the pinning --
        # polarizer, analyzer, and sample each keep their own rotate/fix and ANY of the 8
        # combinations is legal (each rotating element follows the common scan angle t). It also
        # fixes the analyzer-polarizer offset gate: the original enables it only when BOTH the
        # polarizer AND the analyzer rotate (`Dynamic[If[RotateAnalyzer, If[RotatePolarizer,
        # offset, ""]]]`); the port previously consulted the analyzer alone.
        def _sync_pol_enabled(*_a):
            func_txt = functionality.currentText()
            is_maker = which == "ml" and "Maker" in func_txt
            is_fresnel = which == "ml" and "Fresnel" in func_txt
            rot_ok = not (is_maker or is_fresnel)  # rotate/fix triple: SHG Sim + analytical only
            pol_rot = polarizer_mode.currentText().startswith("Rotate")
            ana_rot = analyzer_mode.currentText().startswith("Rotate Analyzer")
            # theta: the Maker/Fresnel sweeps take theta from their own scan grid
            theta_spin.setEnabled(rot_ok)
            theta_slider.setEnabled(rot_ok)
            polarizer_mode.setEnabled(rot_ok)
            analyzer_mode.setEnabled(rot_ok)
            # fixed phi / psi: live whenever their element is fixed -- and ALWAYS in Maker mode,
            # where they set the sweep's input and detection polarization
            _phi_on = is_maker or (rot_ok and not pol_rot)
            fixed_phi.setEnabled(_phi_on)
            fixed_phi_btns.setEnabled(_phi_on)
            _psi_on = is_maker or (rot_ok and not ana_rot)
            analyzer_psi.setEnabled(_psi_on)
            psi_btns.setEnabled(_psi_on)
            # D11: in Maker Fringes BOTH ellipticity boxes were live and editable,
            # but the compute path reads only the Maker one -- a second editable input for the same
            # physical quantity, silently ignored. One control per concept: the Maker box owns
            # Delta-delta in Maker Fringes, the general box owns it everywhere else.
            ellipticity.setEnabled(not is_fresnel and not is_maker)
            if maker_ell is not None:          # the Maker-specific Delta-delta
                maker_ell.setEnabled(is_maker)
                maker_ell_btns.setEnabled(is_maker)
            analyzer_offset.setEnabled(rot_ok and pol_rot and ana_rot)
            if sample_mode is not None:
                _pa = "Partial" in func_txt
                _sr_ok = rot_ok and (func_txt.startswith("SHG") or _pa)
                sample_mode.setEnabled(_sr_ok)
                _on = _sr_ok and sample_mode.currentText().startswith("Rotate")
                for _w in (sample_step, sample_step_btns, sample_dir):
                    _w.setEnabled(_on)

        # currentTextChanged (NOT `activated`): must also run on programmatic/session restores,
        # so a restored combination never leaves a control live that the compute path ignores.
        polarizer_mode.currentTextChanged.connect(_sync_pol_enabled)
        analyzer_mode.currentTextChanged.connect(_sync_pol_enabled)
        if sample_mode is not None:
            sample_mode.currentTextChanged.connect(_sync_pol_enabled)
        functionality.currentTextChanged.connect(_sync_pol_enabled)
        _sync_pol_enabled()
        page._sync_pol_enabled = _sync_pol_enabled  # test hook
        form_col.addWidget(g_pol)

        # Presets (original's Layer Properties Preset Values: blue when saved)
        g_pre, pr_lay = _collapsible_group(QtWidgets, "Layer Properties Preset Values", QtWidgets.QGridLayout)
        _tip(g_pre, "presets")
        store = PresetStore(8 if which == "ml" else 4)  # original: 8 preset slots (.ml), 4 (.si)
        preset_buttons = []
        preset_label_edit = QtWidgets.QLineEdit()
        preset_label_edit.setPlaceholderText("optional label for the next saved preset")
        _tip(preset_label_edit, "presets")

        def snapshot() -> dict:
            # the substrate spins are gone (half-spaces are stack rows, outside the
            # crystal-panel preset scope); the SI incident-medium pair is panel state and joins.
            return {
                "point_group": point_group.currentText(),
                "n_w": n_w.value(), "n_2w": n_2w.value(), "ne_w": ne_w.value(), "ne_2w": ne_2w.value(),
                "inc_n_w": inc_n_w.value(), "inc_n_2w": inc_n_2w.value(),
                "thickness": (stack_film_thickness_um(stack_state["stack"])
                              if which == "ml" and stack_state.get("stack") else 1.0),
                "wavelength": wavelength.value(),
                "orientation_mode": orient_mode.currentText(),
                "lattice": [e.value() for e in lattice_edits],
                "surface_hkl": [e.value() for e in hkl_edits],
                "in_plane_uvw": [e.value() for e in uvw_edits],
                "d_free": {f"{r},{c}": d_field_map[(r, c)].value() for (r, c) in d_field_map},
            }

        def snapshot_with_grids() -> dict:
            """Snapshot() PLUS the tensor grids and (when set) the numeric z-axes — the
            keys `_material_from_custom_spec` consumes, so a Custom layer computes exactly what
            the panels show (the grid-less snapshot() made custom rows fall back to the dead
            parentless index spins, n = 2.0/2.2)."""
            ew, e2, dd = read_full_tensors()
            snap = snapshot()
            snap.pop("wavelength", None)  # a LAYER spec must not own the global wavelength
            snap["eps_omega_full"] = ew
            snap["eps_2omega_full"] = e2
            snap["d_full"] = dd
            if orient_mode.currentText().startswith("Crystal Physics"):
                snap["z_axes"] = _z_axes()
            return snap

        def restore(snap: dict) -> None:
            point_group.setCurrentText(snap.get("point_group", "-43m"))  # rebuilds d fields
            # grid-only snapshots (conversion-created custom layers / preset stack loads) lack
            # the scalar-index keys — default to the current construction values
            n_w.setValue(float(snap.get("n_w", 2.0))); n_2w.setValue(float(snap.get("n_2w", 2.2)))
            ne_w.setValue(float(snap.get("ne_w", 2.0))); ne_2w.setValue(float(snap.get("ne_2w", 2.2)))
            # legacy snapshots carry "sub_n_w"/"sub_n_2w" from the deleted Substrate group —
            # ignored (half-spaces are stack rows now); the SI incident pair restores when present.
            inc_n_w.setValue(float(snap.get("inc_n_w", 1.0)))
            inc_n_2w.setValue(float(snap.get("inc_n_2w", 1.0)))
            if "wavelength" in snap:
                wavelength.setValue(float(snap["wavelength"]))
            orient_mode.setCurrentText(snap.get("orientation_mode", ORIENTATION_MODES[0]))
            for e, v in zip(lattice_edits, snap.get("lattice", [1, 1, 1, 90, 90, 90])):
                e.setValue(float(v))
            _enforce_crystal_system()  # an old inconsistent snapshot is coerced
            for e, v in zip(hkl_edits, snap.get("surface_hkl", [0, 0, 1])):
                e.setValue(int(v))
            for e, v in zip(uvw_edits, snap.get("in_plane_uvw", [1, 0, 0])):
                e.setValue(int(v))
            for key, v in snap.get("d_free", {}).items():
                r, c = (int(x) for x in key.split(","))
                if (r, c) in d_field_map:
                    d_field_map[(r, c)].setValue(float(v))
            # thickness lives in the layer editor now — apply it LAST so it lands in the
            # simple-mode template the lattice/orientation writes above just flipped into
            # (Custom-film ownership), instead of being clobbered by the template rebuild.
            if which == "ml":
                layer_thick.setValue(float(snap.get("thickness", 1.0)))

        def on_preset(i):
            def handler():
                if store.filled(i):
                    restore(store.recall(i))
                    win.statusBar().showMessage(f"Preset {i + 1} applied.")
                else:
                    lbl = preset_label_edit.text().strip()
                    store.save(i, snapshot(), label=lbl)
                    preset_buttons[i].setText(f"Preset {i + 1}: {lbl}" if lbl else f"Preset {i + 1} *")
                    preset_buttons[i].setStyleSheet("background-color: #58a6ff; color: white;")
                    preset_label_edit.clear()
                    win.statusBar().showMessage(f"Preset {i + 1} saved (click again to re-apply).")
            return handler

        for i in range(store.n_slots):
            b = QtWidgets.QPushButton(f"Preset {i + 1}")
            _tip(b, "presets")
            b.clicked.connect(on_preset(i))
            preset_buttons.append(b)
            pr_lay.addWidget(b, i // 2, i % 2)
        nrow = (store.n_slots + 1) // 2
        pr_lay.addWidget(QtWidgets.QLabel("Preset label:"), nrow, 0)
        pr_lay.addWidget(preset_label_edit, nrow, 1)
        clear_b = QtWidgets.QPushButton("Clear Presets")
        _tip(clear_b, "presets")

        def on_clear():
            store.clear()
            for j, b in enumerate(preset_buttons):
                b.setStyleSheet("")
                b.setText(f"Preset {j + 1}")
            win.statusBar().showMessage("All presets cleared.")

        def on_show_info():
            lines = [f"Preset {j + 1}: {store.label(j) or '(no label)'}"
                     for j in range(store.n_slots) if store.filled(j)]
            QtWidgets.QMessageBox.information(win, "Saved Presets",
                                             "\n".join(lines) if lines else "No presets saved yet.")

        clear_b.clicked.connect(on_clear)
        show_b = QtWidgets.QPushButton("Show Preset Info")
        _tip(show_b, "presets")
        show_b.clicked.connect(on_show_info)
        pr_lay.addWidget(clear_b, nrow + 1, 0)
        pr_lay.addWidget(show_b, nrow + 1, 1)
        form_col.addWidget(g_pre)
        g_pre.setVisible(False)  # superseded by the persistent My Materials store

        # ---- My Materials -- the user's own materials in ~/.shaarp/user_materials.json ----
        g_my, my_lay = _collapsible_group(QtWidgets, "My Materials", QtWidgets.QGridLayout)
        _tip(g_my, "my_materials")
        my_name = QtWidgets.QLineEdit()
        my_name.setPlaceholderText("name for the current material")
        _tip(my_name, "my_materials")
        my_save = QtWidgets.QPushButton("Save current as new")
        my_update = QtWidgets.QPushButton("Update selected")
        my_rename = QtWidgets.QPushButton("Rename…")
        my_delete = QtWidgets.QPushButton("Delete selected")
        for _b in (my_save, my_update, my_rename, my_delete):
            _tip(_b, "my_materials")
        my_info = QtWidgets.QLabel("selected: —")
        my_info.setWordWrap(True)
        my_lay.addWidget(my_name, 0, 0, 1, 2)
        my_lay.addWidget(my_save, 1, 0)
        my_lay.addWidget(my_update, 1, 1)
        my_lay.addWidget(my_rename, 2, 0)
        my_lay.addWidget(my_delete, 2, 1)
        my_lay.addWidget(my_info, 3, 0, 1, 2)
        form_col.addWidget(g_my)

        def _my_source_combo():
            return si_case if which == "si" else layer_mat

        def _my_selected():
            from .user_materials import is_user_material
            t = _my_source_combo().currentText().strip()
            return t if is_user_material(t) else None

        def _my_refresh_state(*_a):
            from .user_materials import get as _get
            sel = _my_selected()
            entry = _get(sel) if sel else None
            if entry is not None:
                my_info.setText(f"selected: {sel}   (saved {entry['saved']}, "
                                f"λ = {entry['wavelength_um']:g} µm)")
            else:
                my_info.setText("selected: — (pick one of your materials to update / rename / delete)")
            for _b in (my_update, my_rename, my_delete):
                _b.setEnabled(entry is not None)

        def _my_rebuild_rows(deleted=None, deleted_spec=None, renamed=None):
            """Rebuild ONLY the user section of this page's combo, keeping the selection by text
            (a renamed material keeps its selection under the new name); a deleted selection falls
            back to Custom with a status note."""
            from .user_materials import USER_SECTION_HEADER, list_names
            combo = _my_source_combo()
            prev = combo.currentText()
            if renamed and prev == renamed[0]:
                prev = renamed[1]
            end_marker = None if which == "si" else CUSTOM_LAYER_CHOICE
            combo.blockSignals(True)
            try:
                start = combo.findText(USER_SECTION_HEADER)
                if start >= 0:
                    stop = combo.count() if end_marker is None else combo.findText(end_marker)
                    for _ in range(stop - start):
                        combo.removeItem(start)
                names = list_names()
                if names:
                    at = combo.count() if end_marker is None else combo.findText(end_marker)
                    combo.insertItem(at, USER_SECTION_HEADER)
                    for k, n in enumerate(names, start=1):
                        combo.insertItem(at + k, n)
                _disable_header_rows(combo)
                if combo.findText(prev) >= 0:
                    combo.setCurrentText(prev)
            finally:
                combo.blockSignals(False)
            if combo.findText(prev) < 0:
                combo.setCurrentText("Custom (use fields)" if which == "si" else CUSTOM_LAYER_CHOICE)
                win.statusBar().showMessage(f"Material '{prev}' was deleted — switched to Custom.")
            if which == "ml" and renamed is not None:
                for _spec in stack_state["stack"] or []:
                    if _spec.get("material") == renamed[0]:
                        _spec["material"] = renamed[1]
                _refresh_layer_selector()
            if which == "ml" and deleted is not None:
                # rows of the stack that carried the deleted material keep computing as Custom
                for _spec in stack_state["stack"] or []:
                    if _spec.get("material") == deleted:
                        _spec["material"] = CUSTOM_LAYER_CHOICE
                        if deleted_spec is not None:
                            _spec["custom"] = dict(deleted_spec)
                _load_layer_into_fields()
                _sync_layer_crystal_view()
            _my_refresh_state()

        def _my_refresh_all(deleted=None, deleted_spec=None, renamed=None):
            for fn in getattr(win, "_user_material_refreshers", []):
                fn(deleted, deleted_spec, renamed)

        win._user_material_refreshers = list(getattr(win, "_user_material_refreshers", [])) + [_my_rebuild_rows]

        def _my_current_spec():
            spec = snapshot_with_grids()
            return spec, float(wavelength.value())

        def _my_save(name=None, confirm=True):
            from .user_materials import save
            text = my_name.text() if name is None else name
            try:
                spec, lam = _my_current_spec()
                canonical = save(text, spec, lam)
            except ValueError as exc:
                win.statusBar().showMessage(f"My Materials: {exc}")
                if confirm:
                    QtWidgets.QMessageBox.warning(win, "My Materials", str(exc))
                    return None
                raise
            _my_refresh_all()
            _my_source_combo().setCurrentText(canonical)
            win.statusBar().showMessage(f"Saved material '{canonical}' (λ = {lam:g} µm).")
            return canonical

        def _my_update(confirm=True):
            from .user_materials import save
            sel = _my_selected()
            if sel is None:
                return None
            if confirm and QtWidgets.QMessageBox.question(
                    win, "My Materials", f"Overwrite '{sel}' with the current panels?") != QtWidgets.QMessageBox.Yes:
                return None
            spec, lam = _my_current_spec()
            if which == "ml":
                from .user_materials import get as _get_entry
                _e = _get_entry(sel)
                if _e is not None:
                    lam = float(_e["wavelength_um"])  # the ML wavelength belongs to the STACK
            save(sel, spec, lam)
            _my_refresh_all()
            if which == "si":
                _set_si_dirty(False)
                _on_case_changed()
            else:
                _sync_layer_crystal_view()
            win.statusBar().showMessage(f"Updated material '{sel}' (λ = {lam:g} µm).")
            return sel

        def _my_rename(new=None, confirm=True):
            from .user_materials import rename
            sel = _my_selected()
            if sel is None:
                return None
            if new is None:
                new, ok = QtWidgets.QInputDialog.getText(win, "My Materials", "New name:", text=sel)
                if not ok:
                    return None
            try:
                canonical = rename(sel, new)
            except (ValueError, KeyError) as exc:
                win.statusBar().showMessage(f"My Materials: {exc}")
                if confirm:
                    QtWidgets.QMessageBox.warning(win, "My Materials", str(exc))
                    return None
                raise
            _my_refresh_all(renamed=(sel, canonical))
            _my_source_combo().setCurrentText(canonical)
            win.statusBar().showMessage(f"Renamed '{sel}' → '{canonical}'.")
            return canonical

        def _my_delete(confirm=True):
            from .user_materials import delete, get as _get
            sel = _my_selected()
            if sel is None:
                return False
            if confirm and QtWidgets.QMessageBox.question(
                    win, "My Materials", f"Delete '{sel}' from your materials?") != QtWidgets.QMessageBox.Yes:
                return False
            entry = _get(sel)
            ok = delete(sel)
            _my_refresh_all(deleted=sel, deleted_spec=(entry or {}).get("spec"))
            win.statusBar().showMessage(f"Deleted material '{sel}'.")
            return ok

        my_save.clicked.connect(lambda *_: _my_save())
        my_update.clicked.connect(lambda *_: _my_update())
        my_rename.clicked.connect(lambda *_: _my_rename())
        my_delete.clicked.connect(lambda *_: _my_delete())
        _my_source_combo().currentTextChanged.connect(_my_refresh_state)
        if which == "ml":
            edit_layer.currentIndexChanged.connect(_my_refresh_state)
        _my_refresh_state()
        page._my_materials = {"save": _my_save, "update": _my_update, "rename": _my_rename,
                              "delete": _my_delete, "selected": _my_selected,
                              "refresh": _my_rebuild_rows, "name_edit": my_name, "group": g_my}

        # F48 relevance gating: collapse + hint the input groups the SELECTED functionality does
        # not consume, so the user sees at a glance which inputs matter now (irrelevant
        # controls always visible'). Realized as collapse + a muted title suffix, NOT a disable --
        # a disabled QGroupBox greys its children and blocks pre-setting values for the next mode,
        # which the grill decision explicitly wanted preserved. Expanding a hinted group still works.
        if which == "ml" and g_scan is not None:
            _base_titles = {g_scan: g_scan.title(), g_fres: g_fres.title(),
                            g_pol: g_pol.title(), g_assum: g_assum.title()}

            def _apply_relevance(*_a):
                func_txt = functionality.currentText()
                is_maker = "Maker" in func_txt
                is_fresnel = "Fresnel" in func_txt
                # each sweep mode gets its OWN separately-toggled scan section --
                # Maker Fringes Scan Range for Maker Fringes, Fresnel Coefficients Scan Range
                # for Fresnel Coefficients.
                # Polarimetry stays open in Maker Fringes too (the sweep's input phi
                # and detection psi live there); only Fresnel (linear, s/p per angle) drops it.
                pol_rel = not is_fresnel
                for grp, rel in ((g_scan, is_maker), (g_fres, is_fresnel),
                                 (g_pol, pol_rel), (g_assum, True)):
                    grp.setChecked(bool(rel))  # expand relevant, collapse irrelevant
                    grp.setTitle(_base_titles[grp] + ("" if rel else "   — not used by this mode"))
                _sync_pol_enabled()

            functionality.currentTextChanged.connect(_apply_relevance)
            _apply_relevance()  # startup state
            page_relevance_hook = _apply_relevance  # test hook (exposed on the page below)
        else:
            page_relevance_hook = None

        # Update/Run + Export
        run_btn = QtWidgets.QPushButton("Update / Run")
        run_btn.setStyleSheet(PRIMARY_BTN_QSS)
        _tip(run_btn, "run")
        export_btn = QtWidgets.QPushButton("Export data")
        _tip(export_btn, "export")
        export_fig_btn = QtWidgets.QPushButton("Export figure")  # B1
        export_fig_btn.setToolTip("Save the currently shown plot as PNG / SVG / PDF "
                                  "(publication-ready; vector for SVG/PDF).")
        copy_fig_btn = QtWidgets.QPushButton("Copy figure")  # #6
        copy_fig_btn.setToolTip("Copy the currently shown plot to the clipboard "
                                "(paste straight into notes, slides, or a paper).")
        rr = QtWidgets.QHBoxLayout()
        rr.addWidget(run_btn)
        rr.addWidget(export_btn)
        rr.addWidget(export_fig_btn)
        rr.addWidget(copy_fig_btn)
        form_col.addLayout(rr)
        form_col.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls_host)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)

        # ----- OUTPUT PANEL (right): three DRAGGABLE sections (schematics | plot tabs | analytical).
        # A QSplitter lets the user enlarge/shrink each section; the matplotlib canvases re-render to
        # whatever size their pane is dragged to. The whole output is in a scroll area so it still
        # works if the panes (at their minimums) exceed the window. -----
        _Fig = __import__("matplotlib.figure", fromlist=["Figure"]).Figure

        # (1) schematics section
        schem_widget = QtWidgets.QWidget()
        canv_row = QtWidgets.QHBoxLayout(schem_widget)
        canv_row.setContentsMargins(0, 0, 10, 0)
        # the output banner now carries ONLY the 2D optical-setup schematic --
        # it conveys the reflection/transmission geometry AND the multiple-reflection assumption, which
        # you need to interpret Maker/Fresnel. The former always-on 3D "sample stack" pane was a
        # decorative box for single-interface SHG; its one real use, the crystal-axes (Zi vs Li) view,
        # moved next to the orientation inputs (orient_canvas). Dropping it here reclaims the full
        # banner width for the 2D scene and gives the plots more room.
        # D14: the launch banner used the function's default theta_deg=20 and printed
        # "theta_i = 20" while the panel next to it read 0 -- a picture asserting a number no input
        # holds. Draw it at normal incidence, matching the default.
        schematic_canvas = FigureCanvasQTAgg(
            build_schematic_figure([("air", None), ("crystal", None)], theta_deg=0.0))
        schematic_canvas.setMinimumSize(260, 250)
        canv_row.setSpacing(8)
        canv_row.addWidget(schematic_canvas, 1)    # 2D fills the full banner width now
        schem_widget.setMinimumHeight(270)

        # (2) plot-tabs section: Polar Plots (+ ML Fresnel/Maker) + Analytical Expression. The
        # analytical closed form lives in its OWN tab so it gets the full output area (the original
        # shows the expressions in the output region); an analytical run auto-switches to it.
        plot_canvas = FigureCanvasQTAgg(_Fig(figsize=(7, 7)))
        # Empty-state guidance: until the first Update the output is a blank canvas, which
        # leaves a first-time user unsure what to do. Draw a short "how to use" hint that the
        # first Update replaces with the real plot (_replace_canvas_figure swaps the figure).
        _hint_ax = plot_canvas.figure.add_subplot(111)
        _hint_ax.axis("off")
        _hint_text = (
            "Getting started\n\n"
            "1.  Pick a Case Study example (or set your material).\n"
            "2.  Press the blue  Update  button.\n\n"
            "Your reflected-SHG polar plots appear here."
            if which == "si" else
            "Getting started\n\n"
            "1.  Pick a Case Study example (or build a layer stack).\n"
            "2.  Choose a Functionality (SHG Simulation, Maker Fringes, Fresnel…).\n"
            "3.  Press the blue  Update  button.\n\n"
            "Results appear in the tabs above (Polar Plots / Fresnel Coefficients / Maker Fringes)."
        )
        _hint_ax.text(0.5, 0.5, _hint_text, ha="center", va="center", fontsize=12.5,
                      color="#4a4a4f", linespacing=1.6, transform=_hint_ax.transAxes)
        plot_canvas.figure.patch.set_alpha(0.0)
        plot_canvas.draw_idle()
        fresnel_canvas = FigureCanvasQTAgg(_Fig(figsize=(7, 4))) if which == "ml" else None
        maker_canvas = FigureCanvasQTAgg(_Fig(figsize=(7, 4))) if which == "ml" else None
        # QTextEdit (rich text) so the closed forms render with REAL super/subscripts like the
        # original package's typeset output: the display shows n_ω², θᵢ, d₁₄; the Copy
        # button and the .txt export keep the machine-readable sympy text (stored as a property).
        expr_box = QtWidgets.QTextEdit()
        expr_box.setObjectName("expr_box")
        expr_box.setReadOnly(True)
        expr_box.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)  # long closed forms scroll horizontally
        expr_box.setPlaceholderText("The closed-form analytical expression appears here after an "
                                    "analytical run (Partial / Full Analytical). Use 'Copy' below.")
        _tip(expr_box, "closed_form")
        output_tabs = QtWidgets.QTabWidget()

        def _scrollable(canvas):
            # Each plot tab wraps its canvas in a QScrollArea with widgetResizable=True: the canvas
            # FILLS the viewport (tracks both width and height), so a figure auto-fits the window.
            # A SINGLE plot (Maker/Fresnel) has a small min height -> it always shrinks to fit (no
            # clipping). A MULTI-TILE figure (SI 6-tile / ML 4-polar) keeps a larger min height so its
            # tiles stay square/readable; only then does a vertical scrollbar appear to navigate it.
            sa = QtWidgets.QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QtWidgets.QFrame.NoFrame)
            sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            sa.setWidget(canvas)
            return sa

        # Initial instruction page (shown on startup): how-to-use steps + the SHAARP references,
        # authors and acknowledgment -- so a new user lands on guidance, not a blank plot.
        guide_view = QtWidgets.QTextBrowser()
        guide_view.setOpenExternalLinks(True)
        guide_view.setHtml(USER_GUIDE_HTML)
        output_tabs.addTab(guide_view, "Guide")

        # Multi-tile polar grid (SI = 2 polar + effective-index + ellipticity; ML = 4 polar) keeps a
        # tall min so its tiles stay square/readable (scrolls when the window is short). The SINGLE
        # Maker/Fresnel plots get a SMALL min so they always shrink to FIT the viewport (the user's
        # ask: a single plot should auto-fit, never be clipped). Width flexes via the scroll area.
        plot_canvas.setMinimumSize(440, 500)  # low enough that the 2x2 polar grid FILLS a maximized viewport (no scroll); still scrolls on short laptops
        if which == "ml":
            fresnel_canvas.setMinimumSize(440, 240)
            maker_canvas.setMinimumSize(440, 240)
            # name the result canvases so tests can address them by ROLE. The Maker-gating
            # fence previously selected canvases by a figure-height heuristic, which swept in the
            # polarimetry tile whenever an earlier test changed the render history -- a fragile
            # locator that fails for reasons unrelated to what it is testing.
            fresnel_canvas.setObjectName("fresnel_canvas")
            maker_canvas.setObjectName("maker_canvas")
        plot_tab = _scrollable(plot_canvas)
        output_tabs.addTab(plot_tab, "Polar Plots")
        fresnel_tab = maker_tab = None
        if which == "ml":
            fresnel_tab = _scrollable(fresnel_canvas)
            output_tabs.addTab(fresnel_tab, "Fresnel Coefficients")
            maker_tab = _scrollable(maker_canvas)
            output_tabs.addTab(maker_tab, "Maker Fringes")
        # the Analytical Expression tab stacks the full published
        # derivation as COLLAPSIBLE steps (transmitted omega fields -> P_NL -> inhomogeneous 2omega
        # fields) ABOVE the final reflected E_p/E_s/I. Steps are rebuilt on each Full-Analytical run
        # (collapsed by default so the panel stays compact); the final-expression box keeps driving
        # Copy/Mathematica/export unchanged.
        analytical_page = QtWidgets.QWidget()
        analytical_v = QtWidgets.QVBoxLayout(analytical_page)
        analytical_v.setContentsMargins(2, 2, 2, 2)
        deriv_container = QtWidgets.QWidget()
        deriv_layout = QtWidgets.QVBoxLayout(deriv_container)
        deriv_layout.setContentsMargins(0, 0, 0, 0)
        deriv_container.setVisible(False)  # shown only when a Full-Analytical run yields steps
        deriv_heading = QtWidgets.QLabel("<b>Full analytical derivation — step by step</b>")
        deriv_heading.setVisible(False)
        analytical_v.addWidget(deriv_heading)
        analytical_v.addWidget(deriv_container)
        analytical_v.addWidget(expr_box, 1)
        analytical_scroll = QtWidgets.QScrollArea()
        analytical_scroll.setWidgetResizable(True)
        analytical_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        analytical_scroll.setWidget(analytical_page)

        def _rebuild_derivation_steps(result):
            """Clear + repopulate the collapsible derivation-step boxes from a Full-Analytical result."""
            while deriv_layout.count():
                w = deriv_layout.takeAt(0).widget()
                if w is not None:
                    w.setParent(None)  # remove from the child tree NOW (deleteLater is deferred)
                    w.deleteLater()
            steps = []
            try:
                from .shaarp_gui import analytical_derivation_items
                steps = analytical_derivation_items(result)
            except Exception:
                steps = []
            for heading, subtitle, body_html in steps:
                box, blay = _collapsible_group(QtWidgets, heading, collapsed=True)
                if subtitle:
                    sub = QtWidgets.QLabel(subtitle)
                    sub.setWordWrap(True)
                    sub.setStyleSheet("color:#666; font-size:8pt;")
                    blay.addWidget(sub)
                view = QtWidgets.QTextEdit()
                view.setReadOnly(True)
                view.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
                view.setHtml("<div style=\"font-family:'Cambria Math','Times New Roman',serif;"
                             " font-size:11pt; white-space:pre;\">" + body_html + "</div>")
                view.setMinimumHeight(70)
                blay.addWidget(view)
                deriv_layout.addWidget(box)
            has = bool(steps)
            deriv_container.setVisible(has)
            deriv_heading.setVisible(has)

        output_tabs.addTab(analytical_scroll, "Analytical Expression")
        output_tabs.setMinimumHeight(280)
        output_tabs.setCurrentWidget(guide_view)  # land on the instruction page at startup

        # (3) status / actions section (compact: Copy + Update + validation + Time Used)
        expr_widget = QtWidgets.QWidget()
        expr_lay = QtWidgets.QVBoxLayout(expr_widget)
        expr_lay.setContentsMargins(0, 0, 0, 0)
        copy_btn = QtWidgets.QPushButton("Copy closed form (Python/SymPy)")
        _tip(copy_btn, "closed_form")
        copy_btn.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(
            expr_box.property("raw_text") or expr_box.toPlainText()))  # machine-readable, not the typeset view

        def _copy_mathematica():
            """Copy the closed forms in Wolfram-Language format (request).
            Conversion (sympify + mathematica_code) can take a few seconds for the big ML
            expressions, so it runs lazily on first click and is cached per run."""
            cached = expr_box.property("mathematica_text")
            if not cached:
                result = state.get("last_result")
                if result is None or not expr_box.property("raw_text"):
                    win.statusBar().showMessage("Run an analytical mode first (Partial/Full Analytical).")
                    return
                from .shaarp_gui import analytical_expression_mathematica
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    cached = analytical_expression_mathematica(result)
                except Exception as exc:
                    win.statusBar().showMessage(f"Mathematica conversion failed: {exc}")
                    return
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
                expr_box.setProperty("mathematica_text", cached)
            QtWidgets.QApplication.clipboard().setText(cached)
            win.statusBar().showMessage("Closed form copied in Mathematica format.")

        mcopy_btn = QtWidgets.QPushButton("Copy closed form (Mathematica)")
        mcopy_btn.setToolTip("Copy the closed-form expressions in Wolfram Language syntax "
                             "(Sqrt[..], Exp[..], ^ powers; symbols as \\[Theta]i, n\\[Omega], "
                             "\\[CurlyPhi], ...) — paste directly into a Mathematica notebook. "
                             "The conversion is numerically verified against a live Wolfram kernel.")
        mcopy_btn.clicked.connect(_copy_mathematica)
        status_lbl = QtWidgets.QLabel("Validation: (run to populate)")
        # The validation status can be a long single token (e.g. "maker_outputs_nonsingular_...
        # _caveat"); wrap it so it does NOT pin the window's minimum width wide after Update (same
        # class of bug as the banner header -> would clip the geometry panel on a laptop).
        status_lbl.setWordWrap(True)
        status_lbl.setMinimumWidth(1)
        time_lbl = QtWidgets.QLabel("Time Used = -- s")
        time_lbl.setToolTip("Wall-clock time used by the last Update (the original GUI's "
                            "'Time Used' readout).")
        time_lbl.setStatusTip("Wall-clock time used by the last Update.")
        run_btn_out = QtWidgets.QPushButton("Update")
        run_btn_out.setStyleSheet(PRIMARY_BTN_QSS)
        _tip(run_btn_out, "run")
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(mcopy_btn)
        btn_row.addWidget(run_btn_out)
        expr_lay.addLayout(btn_row)
        expr_lay.addWidget(status_lbl)
        expr_lay.addWidget(time_lbl)
        expr_widget.setMinimumHeight(76)

        # vertical splitter: drag the handles to resize the three sections
        out_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        out_splitter.setChildrenCollapsible(False)
        out_splitter.addWidget(schem_widget)
        out_splitter.addWidget(output_tabs)
        out_splitter.addWidget(expr_widget)
        out_splitter.setStretchFactor(0, 1)
        out_splitter.setStretchFactor(1, 5)
        out_splitter.setStretchFactor(2, 0)
        out_splitter.setSizes([270, 650, 80])  # compact schematic banner; the plot tab fits/scrolls internally

        # page-level horizontal splitter: drag the divider between the input and output panels.
        # The output column is the splitter directly (no outer scroll) -- each plot tab has its OWN
        # scrollbar, so the schematic stays put while the full plot region scrolls inside its tab.
        page_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        page_split.setChildrenCollapsible(False)
        page_split.addWidget(scroll)
        page_split.addWidget(out_splitter)
        page_split.setStretchFactor(0, 0)
        page_split.setStretchFactor(1, 1)
        page_split.setSizes([500, 980])
        # stale-plot banner spanning the top. The orientation triad refreshes live while the
        # schematic + plots only refresh on Update, so mid-edit the screen read as half-live / "links
        # not updating" (grill-2 Q1). Make the boundary explicit: any compute-relevant input change
        # after the last Update raises this banner; a completed Update clears it. The post-Update
        # contract (every display == f(current inputs)) is fenced in tests/test_update_contract.py.
        stale_banner = QtWidgets.QLabel("")
        stale_banner.setVisible(False)
        stale_banner.setWordWrap(True)
        stale_banner.setMinimumWidth(1)  # never pin the window minimum-width wide (laptop-geometry safe)
        # Height must stay a thin strip (your notification page is too large" — it
        # rendered ~480 px tall). CAUSE: wordWrap + minimumWidth(1) makes Qt compute sizeHint as if
        # the label were 1 px wide, i.e. ~15 wrapped lines (hint 282 px), and the default Preferred
        # policy then let the layout grow it further. Cap it at two lines and let the splitter below
        # take every remaining pixel.
        stale_banner.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                   QtWidgets.QSizePolicy.Policy.Maximum)
        stale_banner.setMaximumHeight(2 * stale_banner.fontMetrics().height() + 18)
        stale_banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        stale_banner.setStyleSheet(
            "background:#fff3cd; color:#7a5b00; border:1px solid #ffe08a;"
            " border-radius:4px; padding:4px 8px; font-weight:600;")
        _outcol = QtWidgets.QVBoxLayout()
        _outcol.setContentsMargins(0, 0, 0, 0)
        _outcol.setSpacing(4)
        _outcol.addWidget(stale_banner, 0)
        _outcol.addWidget(page_split, 1)   # the splitter absorbs all free height, not the banner
        outer.addLayout(_outcol)

        state: dict = {"last_result": None, "running": False, "stale": False}

        def _mark_stale(*_a):
            # A compute-relevant input changed. Ignore changes emitted BY our own Update (guarded by
            # state['running']); everything else means the shown schematic/plots no longer match the
            # inputs, so raise the banner until the next Update clears it.
            if state.get("running") or state.get("stale"):
                return
            state["stale"] = True
            stale_banner.setText("● Inputs changed since the last Update — press Update to refresh "
                                 "the schematic and plots.")
            stale_banner.setVisible(True)

        def _clear_stale():
            state["stale"] = False
            stale_banner.setVisible(False)

        def _replace_canvas_figure(canvas, fig):
            import matplotlib.pyplot as plt

            canvas.figure.clear()
            # adopt the new figure's axes by swapping the manager-level figure
            canvas.figure = fig
            _stg = state.get("stage_timer")
            _rt0 = time.perf_counter()
            fig.set_canvas(canvas)
            # SYNC the swapped figure's size to the canvas widget. Swapping canvas.figure does NOT
            # resize the new figure to the widget (a resize event only fires when the widget size
            # actually changes), so a freshly built figure keeps its nominal figsize (e.g. 6.4x5.2 ->
            # 640x520 px) and renders OVERSIZED into a smaller fixed canvas (e.g. the 360x270 3D
            # schematic), showing only its clipped top-left corner. This was the real cause of the
            # "geometry plot doesn't rescale / is clipped after Update" report -- it depended on
            # whether a stray resize happened to fire. Force the size here so every Update renders the
            # full figure. (constrained_layout then re-flows the decorations to that size.)
            w = max(int(canvas.width()), 1)
            h = max(int(canvas.height()), 1)
            dpi = fig.get_dpi() or 100.0
            fig.set_size_inches(w / dpi, h / dpi)
            canvas.draw_idle()
            plt.close("all")
            if _stg is not None:  # Phase 0: canvas-swap/render share of the Update
                _stg.stages["render"] = (_stg.stages.get("render", 0.0)
                                         + time.perf_counter() - _rt0)

        def d_free():
            return {pos: complex(b.value()) for pos, b in d_field_map.items()}

        def on_run():
            if state.get("running"):
                return  # ignore a re-entrant Update (double Enter / double-click mid-compute)
            state["running"] = True
            _clear_stale()  # this Update re-syncs every display to the current inputs
            run_btn.setEnabled(False)
            run_btn_out.setEnabled(False)
            t0 = time.perf_counter()
            from .debuglog import StageTimer
            stage = state["stage_timer"] = StageTimer()  # Phase 0: compute/figure/render split
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
            progress.setRange(0, 0)  # indeterminate "busy" bar so a multi-second solve
            progress.setFormat("Computing…")   # never looks frozen (no cancel exists yet)
            QtWidgets.QApplication.processEvents()
            try:
                state["last_si_curve"] = None  # refreshed by the SI SHG-Simulation branch only
                lattice = tuple(e.value() for e in lattice_edits)
                okw = dict(orientation_mode=orient_mode.currentText(),
                           surface_hkl=tuple(e.value() for e in hkl_edits),
                           in_plane_uvw=tuple(e.value() for e in uvw_edits),
                           z_axes=_z_axes())
                # resolve the faithful display label to its canonical compute mode (None = view-only)
                disp = functionality.currentText()
                canon = FUNCTIONALITY_CANON.get(disp, disp)
                progress.setFormat(_progress_format_for(canon))
                QtWidgets.QApplication.processEvents()

                def _show_view_only():
                    if disp == "User Guide":
                        expr_box.setPlainText(
                            "SHAARP.py — see Help ▸ User Guide for the full guide.\n\n"
                            "Workflow: pick a Functionality; set the material (point group, lattice, "
                            "orientation, refractive indices, SHG d tensor); set the wavelength and, "
                            "for the multilayer tab, the layer stack + assumptions + scan range; set "
                            "the polarimetry (incident angle, ellipticity, polarizer/analyzer); click "
                            "Update. φ=0° is p-polarized, φ=90° is s-polarized.")
                    else:
                        expr_box.setPlainText(
                            f"{disp}: the optical-setup schematic (2D + 3D) is shown above. Choose "
                            "'SHG Simulation' or an analytical mode to compute output.")
                    status_lbl.setText(f"view: {disp}")
                    state["last_result"] = None
                    output_tabs.setCurrentWidget(expr_box)  # the guide/notice text shows in this tab
                    win.statusBar().showMessage(f"View: {disp}")

                if which == "si":
                    # a DIRTY case (panels edited under it) computes from the panels —
                    # the case label stays selected; re-selecting the case resets.
                    if si_case.currentText() != "Custom (use fields)" and not _si_dirty["on"]:
                        from .layer_stack import material_for_label as build_casestudy_material

                        # display label -> registry key (case-study fidelity audit)
                        material = build_casestudy_material(
                            resolve_case_label(si_case.currentText()),
                            wavelength_um=wavelength.value())
                    else:
                        material = None
                    if material is None:
                        _full = read_full_tensors()
                        _fkw = (dict(eps_omega_full=_full[0], eps_2omega_full=_full[1], d_full=_full[2])
                                if _full else {})
                        material = build_custom_si_material(
                            point_group.currentText(), n_omega=n_w.value(), n_2omega=n_2w.value(),
                            n_omega_e=ne_w.value(), n_2omega_e=ne_2w.value(), d_free=d_free(),
                            lattice=lattice, **_fkw, **okw)
                    # schematic label: the MATERIAL's point group when a case study is selected
                    # (the combo cannot represent centrosymmetric case classes like m3m/∞∞m, and a
                    # stale combo made the schematic contradict the result title -- audit)
                    pg_label = (material.structure.point_group
                                if material is not None and getattr(material, "structure", None) is not None
                                else point_group.currentText())
                    # the schematic names the true incident medium (post-Update contract)
                    _amb_lbl = ("air" if inc_n_w.value() == 1.0 and inc_n_2w.value() == 1.0
                                else f"ambient (n={inc_n_w.value():g})")
                    _replace_canvas_figure(schematic_canvas, build_schematic_figure(
                        [(_amb_lbl, None), (f"crystal ({pg_label})", None)],
                        theta_deg=theta_spin.value()))
                    _refresh_orientation_view()  # keep the input-panel Zi-vs-Li view current
                    if canon is None:  # defensive: no compute mode selected
                        _show_view_only()
                        return
                    with stage("compute"):
                        result = compute_si_gui_result(canon,
                                                       point_group=point_group.currentText(),
                                                       theta_deg=theta_spin.value(), material=material)
                    analyzer_deg = (analyzer_psi.value()
                                    if analyzer_mode.currentText().startswith("Fix Analyzer") else None)
                    # for a selected case-study material, drive the figure from its OWN tensors;
                    # otherwise use the scalar entry fields.
                    if si_case.currentText() != "Custom (use fields)" or read_full_tensors() is not None:
                        from .shaarp_gui import si_figure_kwargs_from_material

                        fig_kw = si_figure_kwargs_from_material(material)
                    else:
                        fig_kw = dict(point_group=point_group.currentText(),
                                      n_omega=n_w.value(), n_2omega=n_2w.value(),
                                      n_omega_e=ne_w.value(), n_2omega_e=ne_2w.value(), d_free=d_free())
                    pg_fig = fig_kw.pop("point_group")
                    # the incident-medium indices reach the numeric curve (a non-air pair
                    # routes the closed form to the validated numeric branch inside the curve fn).
                    fig_kw["incident_index_omega"] = inc_n_w.value()
                    fig_kw["incident_index_2omega"] = inc_n_2w.value()
                    if polarizer_mode.currentText().startswith("Fix"):  # Fix Polarizer -> I(psi) polar
                        fig_kw["fixed_phi_deg"] = fixed_phi.value()
                        analyzer_deg = None
                    elif analyzer_mode.currentText().startswith("Rotate Analyzer"):
                        # Rotating analyzer ALWAYS co-rotates (psi = phi + offset), including offset 0
                        # (the old "!= 0" gate made offset-0 Rotating fall through to the
                        # fixed p/s channels -> it computed a FIXED analyzer, not a rotating one).
                        fig_kw["corotating_offset_deg"] = analyzer_offset.value()  # 0 -> parallel + perp
                        analyzer_deg = None
                    # compute the plotted curve ONCE, reuse it for the figure,
                    # and keep it for export -- the exported payload previously carried only the
                    # compat stage numerics (computed at the workflow's canonical polarization),
                    # silently ignoring the panel's polarimetry settings; the original GUI's Copy
                    # exported the PLOTTED lists.
                    from .shaarp_gui import si_polarimetry_curve as _si_curve

                    with stage("compute"):
                        _curve = _si_curve(pg_fig, theta_deg=theta_spin.value(),
                                           ellipticity_deg=ellipticity.value(),
                                           analyzer_deg=analyzer_deg, **fig_kw)
                    state["last_si_curve"] = _curve
                    with stage("figure"):
                        _si_fig = build_si_polarimetry_figure(
                            pg_fig, theta_deg=theta_spin.value(), precomputed_curve=_curve,
                            incident_index=inc_n_w.value(),  # effective-index tile
                            ellipticity_deg=ellipticity.value(), analyzer_deg=analyzer_deg, **fig_kw)
                    _replace_canvas_figure(plot_canvas, _si_fig)
                    output_tabs.setCurrentWidget(plot_tab)  # show the polar plots (analytical run re-switches below)
                else:
                    sel = system_preset.currentText()
                    # the simple modes' ambient/substrate media are the stack's half-space
                    # rows (isotropic-n entries in the template; the deleted Substrate group's
                    # defaults are the fallbacks so a malformed stack cannot change physics).
                    _top = stack_halfspace_n(stack_state["stack"], "top") or (1.0, 1.0)
                    _bot = stack_halfspace_n(stack_state["stack"], "bottom") or (1.45, 1.46)
                    _store_layer_from_fields()  # settle the selected row before dispatching
                    if _ml_dirty["on"] and sel != "N-layer stack (editor)":
                        # a MODIFIED example ("stay under the example case") computes its
                        # working copy — the stack the loaders filled plus the user's edits.
                        from .layer_stack import build_system_from_stack

                        sys_arg, preset_arg = build_system_from_stack(
                            stack_state["stack"], wavelength_um=wavelength.value(),
                            theta_deg=theta_spin.value()), None
                    elif sel == "Custom film (use fields)":
                        # the film row's captured grids are the film's tensors (the mirror may
                        # currently show a HALF-SPACE row — reading the live grids would alias)
                        _film_c = (stack_state["stack"][1].get("custom") or {}
                                   if len(stack_state["stack"]) > 1 else {})
                        if _film_c.get("eps_omega_full") is not None:
                            _fkw = dict(film_eps_omega_full=_film_c["eps_omega_full"],
                                        film_eps_2omega_full=_film_c["eps_2omega_full"],
                                        film_d_full=_film_c["d_full"])
                        else:
                            _full = read_full_tensors()
                            _fkw = (dict(film_eps_omega_full=_full[0],
                                         film_eps_2omega_full=_full[1],
                                         film_d_full=_full[2]) if _full else {})
                        sys_arg, preset_arg = build_custom_ml_system(
                            point_group.currentText(), film_n_omega=n_w.value(), film_n_2omega=n_2w.value(),
                            film_n_omega_e=ne_w.value(), film_n_2omega_e=ne_2w.value(),
                            ambient_n_omega=_top[0], ambient_n_2omega=_top[1],
                            substrate_n_omega=_bot[0], substrate_n_2omega=_bot[1],
                            thickness_um=stack_film_thickness_um(stack_state["stack"]),
                            wavelength_um=wavelength.value(),
                            d_free=d_free(), lattice=lattice, **_fkw, **okw), None
                    elif (_film_key := _ml_film_key(sel)) is not None:
                        from .casestudy_materials import build_casestudy_ml_system

                        sys_arg, preset_arg = build_casestudy_ml_system(
                            _film_key, thickness_um=stack_film_thickness_um(stack_state["stack"]),
                            wavelength_um=wavelength.value(),
                            ambient_n_omega=_top[0], ambient_n_2omega=_top[1],
                            substrate_n_omega=_bot[0], substrate_n_2omega=_bot[1]), None
                    elif sel == "N-layer stack (editor)":
                        from .layer_stack import build_system_from_stack

                        sys_arg, preset_arg = build_system_from_stack(
                            stack_state["stack"], wavelength_um=wavelength.value(),
                            theta_deg=theta_spin.value()), None
                    else:
                        sys_arg, preset_arg = None, sel
                    sketch_sys = sys_arg if sys_arg is not None else resolve_ml_system_preset(preset_arg)

                    def _layer_label(L):
                        # original .ml Set-Material figure: "layer name, Miller indices ..., point
                        # group and thickness ... for each layer" (fidelity FB8a); hkl only when the
                        # case-study registry knows it.
                        parts = [L.name]
                        try:
                            from .casestudy_materials import casestudy_miller_label
                            pg = L.material.structure.point_group
                            hkl = casestudy_miller_label(getattr(L.material, "name", ""))
                            extra = " ".join(x for x in (pg, hkl) if x)
                            if extra:
                                parts.append(f"[{extra}]")
                        except Exception:
                            pass
                        return " ".join(parts)

                    layers = [(_layer_label(L), L.thickness_um) for L in sketch_sys.layers]
                    # pass the stack's REAL indices so the drawn rays refract
                    # by Snell at each interface. Without this the figure silently used its
                    # illustrative n = 1.5 fallback and drew a 13.9 nm gold film as glass.
                    from .shaarp_gui import schematic_indices_for
                    _sch_idx = schematic_indices_for(sketch_sys)
                    _replace_canvas_figure(schematic_canvas, build_schematic_figure(
                        layers, theta_deg=theta_spin.value(), wavelength_um=float(sketch_sys.wavelength_um),
                        indices=_sch_idx,
                        assumption=assumption_combo.currentText(),
                        fmr_submode=(fmr_submode.currentText()
                                     if ML_ASSUMPTIONS.get(assumption_combo.currentText(), 0) == 0 else None)))
                    _refresh_orientation_view()  # keep the input-panel Zi-vs-Li view current
                    if canon is None:  # defensive: no compute mode selected
                        _show_view_only()
                        return
                    # 'analytical dij' known/unknown mixing (FB2): with the checkbox on, grid cells
                    # holding plain NUMBERS become KNOWN (substituted) components of the Partial
                    # Analytical closed form; everything else stays symbolic.
                    d_known = None
                    # known components live on the STACK (per flagged layer, keyed by
                    # row suffix) -- not in the displayed grid's text. A flagged layer stays
                    # symbolic in the closed form whichever row is on screen, and its known
                    # values survive row switches (remember 0.3").
                    if canon == "Partial Analytical" and which == "ml":
                        d_known = _analytic_d_hook["stack_known_d"]()
                    # the per-layer flags travel on the SYSTEM. h_val now only carries
                    # the explicit "nothing is symbolic" intent, so clearing every box really
                    # substitutes every thickness (see compute_ml_gui_result's back-compat rule).
                    h_val = (None if any(h or d for h, d in _stack_analytic_flags())
                             else float(stack_film_thickness_um(stack_state["stack"])))
                    _sr = (page._sample_rotation_state() if sample_mode is not None
                           else {"on": False})
                    if _sr["on"] and canon == "SHG Simulation":
                        # the original computes the azimuth sweep INSIDE SHG Simulation
                        # (`If[Functionality == "SHG Simulation", If[samplerotationcontrol, ...]]`),
                        # passing its fixed phi/psi into every point. the polarizer and
                        # analyzer may each ALSO rotate with the common scan angle.
                        from .shaarp_gui import ml_sample_rotation_result
                        with stage("compute"):
                            result = ml_sample_rotation_result(
                                sketch_sys, theta_deg=theta_spin.value(),
                                fixed_phi_deg=fixed_phi.value(),
                                analyzer_psi_deg=analyzer_psi.value(),
                                ellipticity_deg=ellipticity.value(),
                                step_deg=_sr["step_deg"], ccw=_sr["ccw"],
                                rotate_polarizer=_sr.get("rotate_polarizer", False),
                                rotate_analyzer=_sr.get("rotate_analyzer", False),
                                analyzer_offset_deg=_sr.get("analyzer_offset_deg", 0.0))
                    else:
                        # F70 the GUI walkthrough: the Fresnel spins need the same friendly min<max guard
                        # as the Maker ones (the grid builder's raw error names no field).
                        if canon == "Fresnel Coefficients" and fr_min.value() >= fr_max.value():
                            raise ValueError(
                                f"Fresnel scan range: θ min ({fr_min.value():g}°) must be smaller "
                                f"than θ max ({fr_max.value():g}°). Fix the Fresnel Coefficients "
                                f"Scan Range fields.")
                        with stage("compute"):
                            result = compute_ml_gui_result(canon,
                                                           point_group=point_group.currentText(),
                                                           theta_deg=theta_spin.value(),
                                                           sample_rotation=bool(_sr["on"]),
                                                           sample_rotation_step_deg=_sr.get("step_deg", 10.0),
                                                           sample_rotation_ccw=_sr.get("ccw", True),
                                                           sample_rotate_polarizer=_sr.get("rotate_polarizer", False),
                                                           sample_rotate_analyzer=_sr.get("rotate_analyzer", False),
                                                           sample_analyzer_offset_deg=_sr.get("analyzer_offset_deg", 0.0),
                                                           fixed_phi_deg=fixed_phi.value(),
                                                           analyzer_psi_deg=analyzer_psi.value(),
                                                           ellipticity_deg=(
                                                               maker_ell.value()
                                                               if (maker_ell is not None
                                                                   and canon == "Maker Fringes")
                                                               else ellipticity.value()),
                                                           fresnel_min_deg=fr_min.value(),
                                                           fresnel_max_deg=fr_max.value(),
                                                           fresnel_step_deg=fr_step.value(),
                                                           theta_min_deg=th_min.value(), theta_max_deg=th_max.value(),
                                                           theta_step_deg=th_step.value(),
                                                           assumption=assumption_combo.currentText(),
                                                           fmr_submode=fmr_submode.currentText(),
                                                           system_preset=preset_arg, system=sys_arg,
                                                           analytical_d_known=d_known,
                                                           analytical_d_symbolic=(
                                                               None if any(d for _h, d in _stack_analytic_flags())
                                                               else False),
                                                           analytical_h_value=h_val)
                    # scan-range sanity (audit): a min >= max typo previously ran "successfully"
                    # with an empty/garbage sweep instead of telling the user what to fix
                    # theta_min/theta_max here are the MAKER FRINGES range; Fresnel has
                    # its own fr_min/fr_max/fr_step (guarded above, before compute).
                    if canon == "Maker Fringes" and th_min.value() >= th_max.value():
                        raise ValueError(
                            f"scan range: θ min ({th_min.value():g}°) must be smaller than θ max "
                            f"({th_max.value():g}°). Fix the 'scan: min / max / step (deg)' fields.")
                    assum = assumption_combo.currentText()
                    is_fmr = ML_ASSUMPTIONS.get(assum, 0) == 0
                    a_label = f"{assum} — {fmr_submode.currentText()}" if is_fmr else assum
                    if getattr(result, "kind", "") == "sample_rotation":
                        # polar RA figure (2ω SHG vs SAMPLE azimuth), angular axis in the
                        # user's CW/CCW sense. The curve rides along for the equal-results fence.
                        from .shaarp_gui import build_ra_scan_figure
                        _sr_stage = result.stages.get("sample_rotation", {})
                        with stage("figure"):
                            _rafig = build_ra_scan_figure(
                                result, azimuth_deg=_sr_stage.get("azimuth_deg_user"),
                                title_suffix=(
                                    f"ψₛ 0→360° {_sr_stage.get('direction', 'CCW')}, "
                                    + ("φ co-rotating, " if _sr_stage.get("rotate_polarizer")
                                       else f"φ={_sr_stage.get('fixed_phi_deg', 0.0):g}°, ")
                                    + ("ψ co-rotating, " if _sr_stage.get("rotate_analyzer")
                                       else f"ψ={_sr_stage.get('analyzer_psi_deg', 0.0):g}°, ")
                                    + f"θⁱ={_sr_stage.get('theta_deg', 0.0):g}°"))
                        _replace_canvas_figure(plot_canvas, _rafig)
                        output_tabs.setCurrentWidget(plot_tab)
                        state["last_ra_result"] = result
                        state["last_ra_system"] = _sr_stage.get("system", sketch_sys)
                        state["last_ra_grid"] = _sr_stage.get("azimuth_deg_solver")
                    elif canon == "Maker Fringes":
                        with stage("figure"):
                            _mkfig = build_maker_figure(result, assumption_label=a_label)
                        _replace_canvas_figure(maker_canvas, _mkfig)
                        output_tabs.setCurrentWidget(maker_tab)
                    elif canon == "Fresnel Coefficients":
                        with stage("figure"):
                            _frfig = build_fresnel_figure(result)
                        _replace_canvas_figure(fresnel_canvas, _frfig)
                        output_tabs.setCurrentWidget(fresnel_tab)
                    elif canon == "SHG Simulation":
                        # the original .ml SHG-Simulation output = FOUR polar plots (reflected +
                        # transmitted I_p/I_s vs incident polarization phi).
                        policy = FMR_SUBMODES[fmr_submode.currentText()] if is_fmr else "all"
                        fphi = fixed_phi.value() if polarizer_mode.currentText().startswith("Fix") else None
                        coro = (analyzer_offset.value()
                                if (analyzer_mode.currentText().startswith("Rotate Analyzer")
                                    and fphi is None) else None)  # Rotating ALWAYS co-rotates (offset 0
                        # -> parallel+perp; Rui, mirrors the SI fix -- the old "!= 0" gate made
                        # offset-0 Rotating fall through to the fixed p/s channels = a fixed analyzer).
                        # the ML tab's "Fix Analyzer" mode was a
                        # DEAD CONTROL -- analyzer_psi never reached the ML polar compute (the SI
                        # branch honored it). Same precedence as SI: Fix-Polarizer sweeps psi
                        # itself and co-rotating tracks phi, so the fixed analyzer applies only
                        # in the plain rotating-polarizer mode.
                        ml_anal = (analyzer_psi.value()
                                   if (analyzer_mode.currentText().startswith("Fix Analyzer")
                                       and fphi is None and coro is None) else None)
                        with stage("compute"):
                            ml_curve = ml_polarimetry_curve(
                                sketch_sys, theta_deg=theta_spin.value(),
                                ellipticity_deg=ellipticity.value(),
                                inhomogeneous_source_policy=policy,
                                mrassumption=ML_ASSUMPTIONS.get(assum, 0),  # 4-polar honors FMR/JK/HH
                                fixed_phi_deg=fphi, corotating_offset_deg=coro,
                                analyzer_deg=ml_anal)
                        # the original's incident/reflected/transmitted beam-ellipticity tile (FB6),
                        # at the CURRENT polarizer setting (fixed phi, else the p-polarized default)
                        from .shaarp_gui import ml_beam_ellipses
                        try:
                            ell3 = ml_beam_ellipses(
                                sketch_sys, theta_deg=theta_spin.value(),
                                phi_deg=(fphi if fphi is not None else 0.0),
                                ellipticity_deg=ellipticity.value())
                        except Exception:
                            ell3 = None  # ellipse tile is auxiliary; never block the polar plots
                        with stage("figure"):
                            _mlfig = build_ml_polarimetry_figure(
                            ml_curve, point_group=point_group.currentText(), assumption_label=a_label,
                            ellipses=ell3)
                        _replace_canvas_figure(plot_canvas, _mlfig)
                        output_tabs.setCurrentWidget(plot_tab)
                state["last_result"] = result
                had_expr = False
                try:
                    raw = analytical_expression_text(result)
                    from .shaarp_gui import analytical_expression_html
                    expr_box.setProperty("raw_text", raw)  # Copy + export stay machine-readable
                    expr_box.setProperty("mathematica_text", "")  # lazy cache invalid on new run
                    expr_box.setHtml(analytical_expression_html(result))  # typeset display
                    expr_box.moveCursor(QtGui.QTextCursor.MoveOperation.Start)
                    expr_box.ensureCursorVisible()  # always start at the BEGINNING
                    had_expr = bool(raw.strip())
                except ValueError:
                    expr_box.setProperty("raw_text", "")
                    expr_box.setProperty("mathematica_text", "")
                    expr_box.setPlainText("")
                _rebuild_derivation_steps(result if had_expr else None)  # F37 step-by-step
                status_lbl.setText(_friendly_validation_status(result.validation.status)
                                   + _symbols_summary(result))
                status_lbl.setToolTip(f"raw validation tag: {result.validation.status}")
                # F70 the GUI walkthrough: a Partial-Analytical run that fell back to the numeric
                # sample-rotation sweep must SAY so (the reason was write-only before) -- and it
                # must survive to the FINAL statusBar write, not be clobbered by "Run complete.".
                _fb_reason = (result.stages.get("analytic_azimuth_fallback_reason")
                              if getattr(result, "stages", None) else None)
                win.statusBar().showMessage(
                    ("Run complete — Partial Analytical fell back to the numeric "
                     "sample-rotation sweep: " + str(_fb_reason)) if _fb_reason
                    else "Run complete.")
                # ANALYTICAL modes: the closed form IS the output -> switch to its tab (full area).
                if had_expr:
                    output_tabs.setCurrentWidget(analytical_scroll)
            except Exception as exc:  # surface errors in-panel, like the original GUI
                # Phase D: the full traceback goes to the rotating debug log + the Debug
                # Info dialog -- a one-line statusBar message alone made field reports
                # unreproducible (no traceback existed anywhere).
                from .debuglog import log_exception
                win._last_traceback = log_exception(which, disp, exc)
                win.statusBar().showMessage(f"{type(exc).__name__}: {exc}")
                # Record into the smoke-test sink if one is armed (--gui-smoke) so an Update that
                # errors is DETECTED rather than silently swallowed into a dialog -- the exact way the
                # frozen-exe FileNotFoundError hid from "I clicked Update and it looked fine".
                _sink = getattr(win, "_gui_smoke_errors", None)
                if _sink is not None:
                    _sink.append(f"{disp}: {type(exc).__name__}: {exc}")
                else:
                    # NON-MODAL error surface. A blocking
                    # QMessageBox.warning(...).exec() froze the app until dismissed -- and hung a
                    # headless run forever (the 10-h profiling stall). Show the friendly message
                    # persistently in the status bar + a red result-panel note instead; the full
                    # traceback is already in the rotating debug log + Help > Debug Info.
                    _msg = _friendly_error_message(exc).replace("\n", " ")
                    win.statusBar().showMessage(_msg, 15000)
                    try:
                        status_lbl.setText("⚠ " + _msg)
                        status_lbl.setStyleSheet("color:#b00020;")
                        status_lbl.setToolTip("See Help ▸ Debug Info for the full traceback.")
                    except Exception:
                        pass
            finally:
                state["running"] = False  # re-arm; a new Update is allowed again
                run_btn.setEnabled(True)
                run_btn_out.setEnabled(True)
                QtWidgets.QApplication.restoreOverrideCursor()
                dt = time.perf_counter() - t0
                time_lbl.setText(f"Time Used = {dt:.3f} s")
                progress.setRange(0, 100)  # leave the indeterminate/busy state
                progress.setValue(100)
                progress.setFormat("100% Completed")
                try:  # Phase D: per-Update telemetry (never allowed to break a run)
                    from .debuglog import log_run
                    _case = (si_case.currentText() if which == "si"
                             else system_preset.currentText())
                    _stages = dict(getattr(state.get("stage_timer"), "stages", {}) or {})
                    _known = sum(_stages.values())
                    if _stages and dt > _known:
                        _stages["rest"] = dt - _known  # Qt/layout/misc not inside a stage block
                    log_run(which, functionality.currentText(), _case, dt, stages=_stages)
                except Exception:
                    pass
                if _AUTOSESSION:  # remember this state for next-launch restore (never breaks a run)
                    try:
                        _autosave_session(win)
                    except Exception:
                        pass

        def _build_export_payload():
            """The exported JSON payload: the SI SHG-Simulation numeric block is
            the compat workflow's stage result at its CANONICAL incident polarization -- it does
            not track the panel's polarimetry controls. So the PLOTTED polarimetry curves ride
            along in 'polarimetry_curves' (the original GUI's Copy exported the plotted lists),
            with a note declaring what each block is. Exposed on the page for the causality gate."""
            from .shaarp_gui import export_result_payload

            payload = export_result_payload(state["last_result"])
            _curve = state.get("last_si_curve")
            if which == "si" and _curve is not None:
                import numpy as _np

                def _ser_curve(v):
                    a = _np.asarray(v)
                    if a.ndim == 0:
                        return v if isinstance(v, (str, int, float, bool)) else str(v)
                    if _np.iscomplexobj(a):
                        return {"re": _np.real(a).tolist(), "im": _np.imag(a).tolist()}
                    return a.tolist()

                payload["polarimetry_curves"] = {k: _ser_curve(v) for k, v in _curve.items()}
                payload["numeric_note"] = (
                    "The 'numeric' block is the validated shaarp_si_compat stage result at the "
                    "workflow's canonical incident polarization; 'polarimetry_curves' is the "
                    "plotted result reflecting the panel's polarimetry settings.")
            # #5 provenance: every export carries the FULL input state + version + timestamp,
            # so any exported result is exactly reproducible (Load Session with 'inputs' -> Update).
            try:
                import time as _t

                from . import __version__ as _ver
                from .gui_introspect import collect_session_state
                payload["provenance"] = {
                    "shaarp_py_version": str(_ver),
                    "tab": which,
                    "exported": _t.strftime("%Y-%m-%d %H:%M:%S"),
                    "inputs": collect_session_state(win),
                    "note": "Reproduce: File > Load Session with 'inputs', then press Update.",
                }
            except Exception:
                pass  # provenance is best-effort; never block a data export
            return payload

        def on_export():
            import json

            if state["last_result"] is None:
                win.statusBar().showMessage("Nothing to export yet -- press Update/Run first.")
                return

            payload = _build_export_payload()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                win, "Export result data", f"shaarp_export_{which}_{payload['kind']}.json", "JSON (*.json)")
            if not path:
                return
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            raw_expr = expr_box.property("raw_text") or expr_box.toPlainText()
            if raw_expr:
                with open(path.rsplit(".", 1)[0] + "_closed_form.txt", "w", encoding="utf-8") as fh:
                    fh.write(raw_expr)  # machine-readable sympy text, not the typeset view
                try:  # Mathematica-format twin (request); failure never blocks export
                    from .shaarp_gui import analytical_expression_mathematica
                    mtext = expr_box.property("mathematica_text") or analytical_expression_mathematica(
                        state["last_result"])
                    with open(path.rsplit(".", 1)[0] + "_closed_form_mathematica.txt", "w",
                              encoding="utf-8") as fh:
                        fh.write(mtext)
                except Exception:
                    pass
            win.statusBar().showMessage(f"Exported -> {path}")

        def _current_result_figure():
            """The figure on the OUTPUT tab the user is looking at. The visible output
            sub-tab decides which canvas: Maker/Fresnel on ML, else the main polar/analytical plot."""
            cur = output_tabs.currentWidget() if output_tabs is not None else None
            for cv, tab in ((maker_canvas, maker_tab if which == "ml" else None),
                            (fresnel_canvas, fresnel_tab if which == "ml" else None),
                            (plot_canvas, plot_tab)):
                if cv is not None and tab is not None and cur is tab:
                    return cv.figure
            return plot_canvas.figure  # default: the primary plot

        def on_export_figure():
            fig = _current_result_figure()
            if fig is None or not fig.axes:
                win.statusBar().showMessage("Nothing to export yet -- press Update/Run first.")
                return
            path, _flt = QtWidgets.QFileDialog.getSaveFileName(
                win, "Export figure", f"shaarp_{which}_figure.png",
                "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
            if not path:
                return
            try:
                fig.savefig(path, dpi=200, bbox_inches="tight")
                win.statusBar().showMessage(f"Figure exported -> {path}")
            except Exception as exc:
                win.statusBar().showMessage(f"Figure export failed: {type(exc).__name__}: {exc}")

        def _current_result_canvas():
            cur = output_tabs.currentWidget() if output_tabs is not None else None
            for cv, tab in ((maker_canvas, maker_tab if which == "ml" else None),
                            (fresnel_canvas, fresnel_tab if which == "ml" else None),
                            (plot_canvas, plot_tab)):
                if cv is not None and tab is not None and cur is tab:
                    return cv
            return plot_canvas

        def on_copy_figure():
            """#6: copy the shown plot to the clipboard as an image."""
            cv = _current_result_canvas()
            if cv is None or not cv.figure.axes:
                win.statusBar().showMessage("Nothing to copy yet -- press Update/Run first.")
                return
            try:
                cv.draw()  # ensure the buffer is current
                QtWidgets.QApplication.clipboard().setPixmap(cv.grab())
                win.statusBar().showMessage("Figure copied to clipboard.")
            except Exception as exc:
                win.statusBar().showMessage(f"Copy failed: {type(exc).__name__}: {exc}")

        def _popout_current_figure(*_a):
            """#8: double-click a plot to open it enlarged in a resizable window."""
            fig = _current_result_figure()
            if fig is None or not fig.axes:
                return
            dlg = QtWidgets.QDialog(win)
            dlg.setWindowTitle("SHAARP.py — enlarged plot")
            dlg.resize(900, 760)
            lay = QtWidgets.QVBoxLayout(dlg)
            big = FigureCanvasQTAgg(fig)  # same Figure object, its own big canvas
            lay.addWidget(big)
            big.draw_idle()
            dlg.finished.connect(lambda *_: fig.set_canvas(plot_canvas))  # rebind on close
            dlg.show()

        for _cv in (plot_canvas, maker_canvas, fresnel_canvas):
            if _cv is not None:
                _cv.mouseDoubleClickEvent = (lambda ev, f=_popout_current_figure: f())

        run_btn.clicked.connect(on_run)
        run_btn_out.clicked.connect(on_run)  # the output-panel Update drives the same recompute
        export_btn.clicked.connect(on_export)
        export_fig_btn.clicked.connect(on_export_figure)  # B1
        copy_fig_btn.clicked.connect(on_copy_figure)  # #6
        # Enter/Return anywhere in the input column triggers Update (an input-tab shortcut,
        # not global, so it never fires while the user is in an output/dialog context).
        _run_sc = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), page)
        _run_sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        _run_sc.activated.connect(on_run)
        _run_sc2 = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), page)  # numpad Enter
        _run_sc2.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        _run_sc2.activated.connect(on_run)
        # raise the stale banner on ANY compute-relevant input change. Wire via the same
        # widget walk the session/coverage gate uses, so new inputs are covered automatically. Skip
        # buttons (no held state; the Update button must never self-mark) and checkable-group toggles
        # (collapse/expand + relevance-gating is not a compute change); programmatic populate uses
        # setText/setValue -- textEdited never fires on it, and setValue during our own Update is
        # guarded by state['running'].
        from .gui_introspect import iter_interactive_widgets
        for _ident, _w in iter_interactive_widgets(page, which):
            if isinstance(_w, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                _w.valueChanged.connect(_mark_stale)
            elif isinstance(_w, QtWidgets.QComboBox):
                _w.currentIndexChanged.connect(_mark_stale)
            elif isinstance(_w, QtWidgets.QLineEdit):
                _w.textEdited.connect(_mark_stale)
            elif isinstance(_w, QtWidgets.QCheckBox):
                _w.toggled.connect(_mark_stale)
        page._build_export_payload = _build_export_payload  # F41 export-causality gate hook
        page._relevance_hook = page_relevance_hook  # F48 relevance-gating test hook (None on SI)
        page._stack_mode_hook = stack_mode_hook  # F53 stack-mode ownership test hook (None on SI)
        # flat wheel-disable on every input control in the column (spins, dropdowns, the θ
        # slider) — the wheel only ever scrolls the panel. Installed last so every widget exists.
        _wheel_guard = _WheelGuard(page)
        page._wheel_guard = _wheel_guard  # keep a reference + test hook
        for _cls in (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QSlider):
            for _w in controls_host.findChildren(_cls):
                _w.installEventFilter(_wheel_guard)
        page._on_export_figure = on_export_figure  # B1 test hook
        page._on_copy_figure = on_copy_figure  # #6 test hook
        page._popout_current_figure = _popout_current_figure  # #8 test hook
        page._stale_banner = stale_banner  # A1 test hook
        page._is_stale = lambda: bool(state.get("stale"))  # A1 test hook
        page._mark_stale = _mark_stale  # A1 test hook
        page._schematic_canvas = schematic_canvas  # A1 Update-contract fence hook
        page._state = state  # C3 re-entry-guard fence hook
        page._last_ra_result = lambda: state.get("last_ra_result")  # sample-rotation fence hook
        page._last_ra_system = lambda: state.get("last_ra_system")  # equal-results fence hook
        page._last_ra_grid = lambda: state.get("last_ra_grid")  # the SOLVER-sense grid
        return page, on_run

    si_page, si_run = make_interface_tab("si")
    tabs.addTab(si_page, "SHAARP.si (single interface)")
    ml_page, ml_run = make_interface_tab("ml")
    tabs.addTab(ml_page, "SHAARP.ml (multilayer)")
    # the global top Update button recomputes whichever tab is active
    update_all_btn.clicked.connect(lambda: (si_run if tabs.currentIndex() == 0 else ml_run)())
    return win


def main() -> int:  # pragma: no cover - interactive entry
    global _AUTOSESSION
    _AUTOSESSION = True  # the real app remembers the last session; tests keep pure defaults
    QtCore, QtGui, QtWidgets = _require_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = build_main_window()
    if "--smoke-test" in sys.argv:  # frozen-exe self check: build the window, never exec
        print("SHAARP desktop smoke OK")
        return 0
    if "--self-check" in sys.argv:  # frozen-exe deeper check: figures + the REAL compute path
        from shaarp.layer_stack import build_system_from_stack, default_stack
        from .shaarp_gui import (build_ml_polarimetry_figure, build_si_polarimetry_figure,
                                 compute_ml_gui_result, compute_si_gui_result, ml_polarimetry_curve)
        si_fig = build_si_polarimetry_figure("3m", theta_deg=45.0, analyzer_deg=None)
        si_axes = si_fig.axes
        si_polar = sum(1 for a in si_axes if a.name == "polar")
        c = ml_polarimetry_curve(build_system_from_stack(default_stack(), wavelength_um=1.064),
                                 n_phi=13, theta_deg=20.0)
        ml_axes = len(build_ml_polarimetry_figure(c).axes)
        # Exercise the ACTUAL GUI compute path (the same calls on_run makes, include_validation_summary=
        # True) for every compute mode -- in the FROZEN exe this is the only place the bundled
        # benchmarks/ data is loaded. The runtime is graceful if that data is missing (it must never
        # crash a user's Update), so the self-check explicitly ASSERTS the benchmark metadata WAS
        # bundled (not the "unavailable" fallback) -- catching a packaging gap at build time, which is
        # exactly the class that shipped the FileNotFoundError on the Quartz+Au case.
        compute_si_gui_result("SHG Simulation", point_group="3m")
        rml = compute_ml_gui_result("SHG Simulation", system_preset="Quartz + Au (Fig 4, 800 nm)")
        compute_ml_gui_result("Maker Fringes", system_preset="Quartz + Au (Fig 4, 800 nm)",
                              theta_min_deg=0.0, theta_max_deg=20.0, theta_step_deg=5.0)
        compute_ml_gui_result("Fresnel Coefficients", theta_min_deg=0.0, theta_max_deg=20.0, theta_step_deg=5.0)
        art = rml.stages.get("mathematica_validation_artifacts", {})
        benchmarks_bundled = art.get("status") != "validation_metadata_unavailable_in_this_build"
        print(f"SHAARP desktop self-check: SI tiles = {len(si_axes)} ({si_polar} polar + index + ellipse), "
              f"ML polar panels = {ml_axes}; compute paths SI/ML/Maker/Fresnel ran; "
              f"benchmarks bundled = {benchmarks_bundled}")
        return 0 if (len(si_axes) == 4 and si_polar == 2 and ml_axes == 4 and benchmarks_bundled) else 2
    # the oblique leg of the smoke sweep, pinned independently of any panel default.
    SMOKE_OBLIQUE_DEG = 45.0
    if "--gui-smoke" in sys.argv:  # drive the REAL Update path for every tab x functionality IN THE EXE
        import time as _time
        win._gui_smoke_errors = []  # arm the error sink (on_run appends here instead of a dialog)
        top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
                   if t.count() >= 2 and "SHAARP" in t.tabText(0))

        def _incidence_spin(page):
            # incidence spin = the one tooltip-tagged "theta" (the scan θ_min spin also has
            # max==89.0 and is now constructed first, so the old max-based match grabbed θ_min).
            theta_tip = TOOLTIPS["theta"]
            spins = page.findChildren(QtWidgets.QDoubleSpinBox)
            for s in spins:
                if s.toolTip() == theta_tip:
                    return s
            for s in spins:
                # match the incidence spin by its TOOLTIP (identity), not by its maximum
                # (a value that moved to 89.9 on SI). A value-based match here would silently drop
                # the theta=0 leg of the gui-smoke sweep while still reporting success.
                if s.toolTip() == TOOLTIPS["theta"]:
                    return s
            return None

        checked = 0
        for ti in range(top.count()):
            page = top.widget(ti)
            combos = [c for c in page.findChildren(QtWidgets.QComboBox) if c.findText("SHG Simulation") >= 0]
            if not combos:
                continue
            func_combo = combos[0]
            update_btns = [b for b in page.findChildren(QtWidgets.QPushButton) if "Update" in b.text()]
            inc_spin = _incidence_spin(page)
            top.setCurrentIndex(ti)
            for fi in range(func_combo.count()):
                func_combo.setCurrentIndex(fi)
                app.processEvents()
                # Drive theta_i = 0 (normal incidence -- the singular edge that crashed in the
                # field with NonInvertibleMatrixError, det==0) AND an oblique angle, plus whatever
                # the panel default happens to be.
                # the angle set is EXPLICIT and no longer derived from the default. It used to
                # be `(default_ang, 0.0)` deduped, so when the default became 0 the pair collapsed
                # to one value and the smoke silently stopped exercising oblique incidence at all
                # (count 14 -> 7, still reporting `errors = 0`). Coverage must not depend on a
                # default that anyone is free to change.
                default_ang = inc_spin.value() if inc_spin is not None else None
                angles = []
                if inc_spin is not None:
                    for a in (0.0, SMOKE_OBLIQUE_DEG, default_ang):
                        if all(abs(a - b) > 1e-9 for b in angles):
                            angles.append(a)
                else:
                    angles = [None]
                for ang in angles:
                    if ang is not None:
                        inc_spin.setValue(ang)
                        app.processEvents()
                    if update_btns:
                        update_btns[0].click()  # synchronous on_run; errors land in win._gui_smoke_errors
                    t0 = _time.time()
                    while _time.time() - t0 < 3.0:  # drain any queued events
                        app.processEvents()
                        _time.sleep(0.03)
                    checked += 1
                if inc_spin is not None and default_ang is not None:
                    inc_spin.setValue(default_ang)  # restore for the next functionality
        errs = win._gui_smoke_errors
        for e in errs:
            print("GUI-SMOKE ERROR:", e)
        print(f"SHAARP desktop gui-smoke: drove {checked} (tab x functionality x angle incl. theta=0) "
              f"Updates; errors = {len(errs)}")
        return 0 if not errs else 3
    n = _restore_last_session(win)  # reopen where the user left off (interactive run only)
    if n:
        win.statusBar().showMessage(
            f"Restored your last session ({n} inputs) — press Update to recompute.")

    def _save_on_quit():
        try:
            _autosave_session(win)
        except Exception:
            pass

    app.aboutToQuit.connect(_save_on_quit)  # also capture the final state if the user never re-Updated
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
