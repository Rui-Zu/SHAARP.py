"""Build the three showcase thumbnails the README strip displays, in light and dark.

    python scripts/make_readme_figures.py [--out DIR]

Writes six files to docs/_static/readme/:

    polarimetry.png / polarimetry_dark.png    reflected SHG lobes, LiNbO3 (3m) at 45 deg
    maker.png       / maker_dark.png          Maker fringes, the paper's quartz + Au stack
    fresnel.png     / fresnel_dark.png        linear R and T through the same stack

Two things are deliberate.

FIRST, the figures come from the APP'S OWN builders (build_si_polarimetry_figure,
build_maker_figure, build_fresnel_figure), not from plotting code written for the README. What a
visitor sees on the landing page is then literally what the app draws, and it cannot drift away
from the product.

SECOND, the physics is computed ONCE per case and only the DRAWING is repeated for the dark
variant, so the two images can never disagree about the numbers.

Sampling follows the rule the rest of this repo now uses: choose the step for the finest feature in
the figure, not for its envelope. The quartz + Au slab puts its Maker fringes 0.56 deg apart and
modulates the Fresnel coefficients with a 0.59 deg period, so 0.05 deg gives about eleven points on
each oscillation and the curves draw smoothly.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import shaarp  # noqa: E402
from shaarp.shaarp_gui import (  # noqa: E402
    build_fresnel_figure,
    build_maker_figure,
    build_si_polarimetry_figure,
)

PRESET = "Quartz + Au (Fig 4, 800 nm)"
STEP_DEG = 0.05          # ~11 points per oscillation on this stack; see the module docstring
FMR = "Full Multiple Reflections (FMR)"

# Rendered by GitHub at roughly 290 px across in a three-cell row, so keep them small enough to load
# quickly and large enough that the axis labels survive the downscale.
FIGSIZE_IN = (5.2, 3.9)
DPI = 150


def _save(fig, out: Path, name: str, scheme: str) -> Path:
    suffix = "" if scheme == "light" else "_dark"
    path = out / f"{name}{suffix}.png"
    # TRANSPARENT, not a baked background. GitHub ships several dark themes (dark, dimmed,
    # high-contrast); a hardcoded #0d1117 rectangle shows a visible seam in the ones it does not
    # match. Transparent canvas plus themed ink sits correctly on all of them, and on white too.
    #
    # NO bbox_inches="tight". It trims each figure to its own content, which gave the three tiles
    # different aspect ratios -- the polarimetry one came out 677x378 against 761x564 -- and since
    # the README scales them all to the same cell width, their heights and captions came out
    # ragged. A fixed canvas keeps the strip aligned.
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"  wrote {path.name}  ({path.stat().st_size // 1024} KB)")
    return path


def _lighten_curves_for_dark(fig) -> None:
    """Lift any curve too dark to read on a dark page.

    matplotlib's dark_background style themes the ink it controls -- text, axes, ticks -- but the
    app sets its own curve colours, and the navy it uses for the p-channel is nearly invisible on
    GitHub's #0d1117. Blend such colours toward white until they clear a luminance floor, leaving
    the hue recognisably the same so the light and dark variants still read as one figure.
    """
    import colorsys

    import matplotlib.colors as mcolors

    floor = 0.42

    def lift(color):
        """Raise brightness in HSV, keeping the hue.

        Blending toward white was the first attempt and it washed the navy out to near-white: the
        curve became legible but stopped being blue, so the light and dark variants no longer read
        as the same figure. Lifting value (and easing saturation just enough to keep it from going
        neon) preserves the identity of each channel.
        """
        try:
            r, g, b = mcolors.to_rgb(color)
        except ValueError:
            return color
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum >= floor:
            return color
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return colorsys.hsv_to_rgb(h, max(0.45, s * 0.8), min(1.0, max(v, 0.55) * 1.45))

    for ax in fig.axes:
        for line in ax.get_lines():
            line.set_color(lift(line.get_color()))
        legend = ax.get_legend()
        if legend is not None:
            for line in legend.get_lines():
                line.set_color(lift(line.get_color()))


def _thin_polar_ticks(fig) -> None:
    """Drop the radial numbers on a thumbnail-sized rosette.

    They overlap each other and the lobes at this size, and they carry nothing a visitor needs from
    a landing page: the shape is the message, and the axis values are in the app and the guide.
    """
    for ax in fig.axes:
        if ax.name == "polar":
            ax.set_yticklabels([])
            ax.tick_params(labelsize=8)


def _both_schemes(name: str, draw, out: Path, *, polar_only: bool = False) -> list[Path]:
    """Draw the same already-computed result twice, once per colour scheme.

    dark_background supplies the light ink; transparent=True on save drops its black canvas, so the
    result is white-on-nothing and sits on whatever the page's own background happens to be.
    """
    paths = []
    for scheme in ("light", "dark"):
        style = "default" if scheme == "light" else "dark_background"
        with plt.style.context(style):
            fig = draw()
            if polar_only:
                # The app's single-interface panel is four tiles; at ~300 px in a README cell the
                # index and ellipticity plots are unreadable specks. Keep the two rosettes, which
                # are the recognisable output, and let the guide show the full panel.
                for ax in list(fig.axes):
                    if ax.name != "polar":
                        ax.remove()
                _thin_polar_ticks(fig)
                for ax in fig.axes:
                    ax.set_title(ax.get_title(), fontsize=9)
                if fig._suptitle is not None:
                    fig._suptitle.set_fontsize(9)
            if scheme == "dark":
                _lighten_curves_for_dark(fig)
            # One box for all three tiles: a shorter polarimetry tile leaves the strip ragged.
            fig.set_size_inches(*FIGSIZE_IN)
            fig.tight_layout()
            paths.append(_save(fig, out, name, scheme))
    light, dark = (Image.open(p) for p in paths)
    if light.size != dark.size:
        raise SystemExit(f"{name}: light {light.size} != dark {dark.size}; the pair must match")
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write here instead of docs/_static/readme")
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "docs" / "_static" / "readme"
    out.mkdir(parents=True, exist_ok=True)

    print("computing the three showcase cases (each solved once, drawn twice) ...")

    # 1. Reflected SHG polarimetry -- the single-interface signature output.
    print("  polarimetry: LiNbO3 (3m), 45 deg")
    _both_schemes("polarimetry",
                  lambda: build_si_polarimetry_figure("3m", theta_deg=45.0), out,
                  polar_only=True)

    # 2. Maker fringes through the paper's quartz + Au heterostructure.
    print(f"  maker: {PRESET} at {STEP_DEG} deg ...")
    maker = shaarp.compute_ml_gui_result(
        "Maker Fringes", system_preset=PRESET,
        theta_min_deg=0.0, theta_max_deg=45.0, theta_step_deg=STEP_DEG)
    _both_schemes("maker", lambda: build_maker_figure(maker, assumption_label=FMR), out)

    # 3. The linear Fresnel coefficients of the same stack.
    print(f"  fresnel: {PRESET} at {STEP_DEG} deg ...")
    fresnel = shaarp.compute_ml_gui_result(
        "Fresnel Coefficients", system_preset=PRESET,
        theta_min_deg=0.0, theta_max_deg=89.9, theta_step_deg=STEP_DEG)
    _both_schemes("fresnel", lambda: build_fresnel_figure(fresnel), out)

    print(f"\ndone -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
