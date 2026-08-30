# Configuration file for the Sphinx documentation builder -- SHAARP.py.
# Builds an HTML site (ReadTheDocs theme) covering the desktop GUI guide AND the Python module/API
# reference (autodoc). See docs/index.md for the table of contents.

from __future__ import annotations

import os
import sys

# Make the `shaarp` package importable for autodoc (docs/ is one level under the repo root).
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------------------------
project = "SHAARP.py"
author = "R. Zu, B. Wang, L. Weber, A. Saha, L.-Q. Chen & V. Gopalan"
copyright = "2026, The Pennsylvania State University"
release = "1.0.0"   # keep in lockstep with pyproject.toml
version = "1.0.0"

# -- General configuration ----------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google / NumPy style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",              # grids / cards / dropdowns (diffractio-like polish)
    "myst_nb",                    # Markdown source + render Jupyter notebooks (loads myst_parser)
]

# Markdown + notebooks: myst-nb owns both .md and .ipynb (it loads myst_parser internally, so we do
# NOT also list "myst_parser" in extensions -- that would double-register its config values).
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 3

# Do NOT execute notebooks at build time -- they can require heavy compute / external tools; render
# their stored outputs as-is. (Set to "auto" later if you want live execution on the build host.)
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc / autosummary ----------------------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# Note: ``undoc-members`` is intentionally NOT a default -- it duplicates the dataclass fields that the
# napoleon ``Attributes:`` docstrings already document (the per-module API pages opt in explicitly).
# The GUI modules pull in Qt; the API reference documents the computational surface, so mock the
# desktop layers if they are unavailable on a headless build host (they import fine locally).
autodoc_mock_imports = []
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Don't fail the build on the (expected) cross-reference nits from dense scientific docstrings.
nitpicky = False
suppress_warnings = ["myst.header", "autosummary"]

# -- HTML output --------------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
html_static_path = ["_static"]
html_title = "SHAARP.py documentation"
_logo = os.path.join(os.path.dirname(__file__), "_static", "shaarp_ml_logo.png")
if os.path.exists(_logo):
    html_logo = "_static/shaarp_ml_logo.png"
