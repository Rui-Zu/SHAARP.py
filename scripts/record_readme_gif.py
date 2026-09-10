"""Record the short animation the README shows: one calculation, start to finish.

    python scripts/record_readme_gif.py [--out DIR]

Writes docs/_static/readme/first_run.gif -- the SHAARP.si tab computing GaAs (111): the app as it
opens, the case chosen, the solve running, the polar plots drawn, then the same case at a second
incidence angle so the lobes visibly change.

WHY DISCRETE STATES AND NOT A SCREEN RECORDING. on_run is synchronous, so the window does not
repaint while a solve is in flight; frames grabbed on a timer would be a run of identical frozen
images and then a jump. These are the states a user actually distinguishes.

The "computing" frame is not staged. on_run disables the Update buttons, sets the progress control
to its running text, and only then calls processEvents() before starting the solve; this script
hooks that one call, so the frame is the window as the app itself painted it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Without a font dir, offscreen Qt has no fonts, substitutes a very wide fallback, and every label
# renders at the wrong width.
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ.setdefault("MPLBACKEND", "Agg")

from PIL import Image  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from shaarp.desktop_app import MODERN_QSS, build_main_window  # noqa: E402

# The app's own default size, and one of the sizes tests/test_input_column_fits.py blesses. Do not
# invent a fourth size: below about 1280 px the input column legitimately grows a scrollbar.
WIN_W, WIN_H = 1480, 940
TARGET_WIDTH = 1120
SIZE_BUDGET_MB = 2.5
# 256, the GIF maximum. At 64 the palette was spent on the app's greys and the plot
# curves quantised to grey and tan -- the navy and orange the app actually draws were
# gone. The file is small enough that there is no reason to economise here.
COLORS = 256


def pump(app, n=10):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.03)


def qt_to_pil(pixmap) -> Image.Image:
    img = pixmap.toImage().convertToFormat(pixmap.toImage().Format.Format_RGBA8888)
    return Image.frombytes("RGBA", (img.width(), img.height()),
                           bytes(img.constBits())).convert("RGB")


def combo_with(page, text):
    for c in page.findChildren(QtWidgets.QComboBox):
        if c.findText(text) >= 0:
            return c
    raise SystemExit(f"no combo offers {text!r}")


def set_incidence(app, page, deg):
    for s in page.findChildren(QtWidgets.QDoubleSpinBox):
        if 89.0 <= s.maximum() <= 90.0:
            s.setValue(float(deg))
            pump(app)
            return
    raise SystemExit("no incidence spin found")


def update_button(page):
    return next(b for b in page.findChildren(QtWidgets.QPushButton)
                if b.text().strip().lower().startswith("update") and b.isEnabled())


def run_update(app, win, page, frames, holds, busy_ms, done_ms, label):
    """Click Update, keep the app's own busy repaint, then the finished state."""
    button = update_button(page)
    busy: list = []
    real = QtWidgets.QApplication.processEvents

    def hooked(*a, **k):
        real(*a, **k)
        if not busy and not button.isEnabled():
            busy.append(win.grab())

    QtWidgets.QApplication.processEvents = staticmethod(hooked)
    try:
        button.click()
    finally:
        QtWidgets.QApplication.processEvents = staticmethod(real)

    if busy:
        frames.append(qt_to_pil(busy[0]))
        holds.append(busy_ms)
        print(f"  frame {len(frames)}: computing ({label}) -- from the app's own repaint")
    else:
        print(f"  note: no busy repaint captured for {label}; the solve returned first")

    for _ in range(400):
        app.processEvents()
        time.sleep(0.05)
        if not getattr(page, "_state", {}).get("running"):
            break
    pump(app, 14)
    frames.append(qt_to_pil(win.grab()))
    holds.append(done_ms)
    print(f"  frame {len(frames)}: result ({label})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "docs" / "_static" / "readme"
    out.mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = build_main_window()
    win.setStyleSheet(MODERN_QSS)
    win.resize(WIN_W, WIN_H)
    win.show()
    pump(app, 14)

    top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
               if t.count() >= 2 and "SHAARP" in t.tabText(0))
    top.setCurrentIndex(0)
    pump(app)
    si = top.widget(0)

    frames: list[Image.Image] = []
    holds: list[int] = []

    combo = combo_with(si, "GaAs (111)")
    combo.setCurrentIndex(combo.findText("GaAs (111)"))
    set_incidence(app, si, 0.0)
    pump(app, 12)
    frames.append(qt_to_pil(win.grab()))
    holds.append(1500)
    print(f"  frame {len(frames)}: case chosen, nothing computed yet")

    run_update(app, win, si, frames, holds, busy_ms=900, done_ms=2000, label="0 deg")

    set_incidence(app, si, 45.0)
    pump(app, 10)
    frames.append(qt_to_pil(win.grab()))
    holds.append(800)
    print(f"  frame {len(frames)}: incidence changed to 45 deg")

    run_update(app, win, si, frames, holds, busy_ms=900, done_ms=2800, label="45 deg")

    scale = TARGET_WIDTH / frames[0].width
    size = (TARGET_WIDTH, int(round(frames[0].height * scale)))
    frames = [f.resize(size, Image.LANCZOS) for f in frames]

    # ONE palette for every frame. Quantising each frame on its own makes the plot colours shimmer
    # as the quantisation shifts, and it destroys the LZW runs that keep the file small.
    montage = Image.new("RGB", (size[0], size[1] * len(frames)))
    for i, f in enumerate(frames):
        montage.paste(f, (0, i * size[1]))
    palette = montage.quantize(colors=COLORS, method=Image.MEDIANCUT)
    paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    gif = out / "first_run.gif"
    paletted[0].save(gif, save_all=True, append_images=paletted[1:], duration=holds,
                     loop=0, optimize=True, disposal=1)

    mb = gif.stat().st_size / 1e6
    print(f"\n  wrote {gif.name}: {len(paletted)} frames, {size[0]}x{size[1]}, {mb:.2f} MB, "
          f"{sum(holds) / 1000:.1f} s loop")
    if mb > SIZE_BUDGET_MB:
        raise SystemExit(
            f"over the {SIZE_BUDGET_MB} MB budget. Reduce in this order: drop the 45-deg pair, "
            f"then COLORS to 48, then TARGET_WIDTH to 960. Do not dither -- it bands the curves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
