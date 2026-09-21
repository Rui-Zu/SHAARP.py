"""Retake the three screenshots the README and the docs landing page show.

    QT_QPA_PLATFORM=offscreen python scripts/capture_doc_screenshots.py

Writes, at the sizes the existing files use:

    docs/screenshot_si.png                 1600x1000  hero: SI tab, LiNbO3 x-cut, after Update
    docs/_static/screens/si_tab.png        1480x920   SI tab card
    docs/_static/screens/ml_tab.png        1480x920   ML tab card, Maker fringes on quartz + Au

These carry the app's own header text, so they go stale whenever that text changes -- which is why
they are captured by a script rather than by hand. Pass --out DIR to write elsewhere and compare
before overwriting the shipped files.

QT_QPA_FONTDIR matters: without it, offscreen Qt has no fonts, substitutes a very wide fallback,
and every label renders at the wrong width (F82).
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
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ.setdefault("MPLBACKEND", "Agg")

from PySide6 import QtWidgets  # noqa: E402

from shaarp.desktop_app import MODERN_QSS, build_main_window  # noqa: E402


def pump(app, n=8):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.03)


def combo_with(page, text):
    for c in page.findChildren(QtWidgets.QComboBox):
        if c.findText(text) >= 0:
            return c
    return None


def set_combo(app, page, text):
    c = combo_with(page, text)
    if c is None:
        raise SystemExit(f"no combo offers {text!r}")
    c.setCurrentIndex(c.findText(text))
    pump(app)
    return c


def set_incidence(app, page, deg):
    """The incidence spin is the one whose maximum sits just under grazing."""
    for s in page.findChildren(QtWidgets.QDoubleSpinBox):
        if 89.0 <= s.maximum() <= 90.0:
            s.setValue(float(deg))
            pump(app)
            return s
    raise SystemExit("no incidence spin found")


def press_update(app, page):
    for b in page.findChildren(QtWidgets.QPushButton):
        if b.text().strip().lower().startswith("update") and b.isEnabled():
            b.click()
            break
    else:
        raise SystemExit("no enabled Update button on this page")
    for _ in range(400):          # the solve runs on the GUI thread; let it finish
        app.processEvents()
        time.sleep(0.05)
        if not getattr(page, "_state", {}).get("running"):
            break
    pump(app, 12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write here instead of over the shipped files")
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "docs"
    (out / "_static" / "screens").mkdir(parents=True, exist_ok=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = build_main_window()
    win.setStyleSheet(MODERN_QSS)
    win.show()
    pump(app, 10)

    top = next(t for t in win.findChildren(QtWidgets.QTabWidget)
               if t.count() >= 2 and "SHAARP" in t.tabText(0))
    si, ml = top.widget(0), top.widget(1)

    # --- SI tab: the case the captions name, computed ---
    top.setCurrentIndex(0)
    pump(app)
    set_combo(app, si, "LiNbO3 (11-20) MTI X-cut")
    set_combo(app, si, "SHG Simulation")
    # 45 deg, not the 0 deg default: at normal incidence the schematic rays are all vertical and
    # the reflected pattern is the degenerate one, which is not what the caption describes.
    set_incidence(app, si, 45.0)
    press_update(app, si)

    win.resize(1600, 1000)
    pump(app, 10)
    hero = out / "screenshot_si.png"
    win.grab().save(str(hero))
    print(f"  wrote {hero.relative_to(out.parent) if args.out is None else hero}")

    win.resize(1480, 920)
    pump(app, 10)
    si_card = out / "_static" / "screens" / "si_tab.png"
    win.grab().save(str(si_card))
    print(f"  wrote {si_card.name}")

    # --- ML tab: Maker fringes on the quartz + Au preset ---
    top.setCurrentIndex(1)
    pump(app)
    set_combo(app, ml, "Quartz + Au (Fig 4, 800 nm)")
    set_combo(app, ml, "Maker Fringes")
    press_update(app, ml)
    pump(app, 10)
    ml_card = out / "_static" / "screens" / "ml_tab.png"
    win.grab().save(str(ml_card))
    print(f"  wrote {ml_card.name}")

    # --- SI tab: a wavelength sweep, on the Spectrum tab ---
    # No shipped image showed the sweep in use, so a reader met the feature only as prose. KTP
    # (dispersive) is the crystal the guide's own walkthrough reaches for, and picking it fits the
    # scan range to the span its index data can answer, so this is the state a reader arrives at.
    top.setCurrentIndex(0)
    pump(app)
    case = set_combo(app, si, "KTP (dispersive) 0.43-3.54 um")
    case.textActivated.emit("KTP (dispersive) 0.43-3.54 um")   # a user's own pick fits the range
    pump(app)
    set_combo(app, si, "SHG Simulation")
    sweep = next(c for c in si.findChildren(QtWidgets.QCheckBox)
                 if "sweep the wavelength" in c.text())
    sweep.setChecked(True)
    pump(app)
    # phi = 30 deg, not the 0 deg default: at 0 the p channel of this crystal is zero by symmetry,
    # so the figure showed one curve rising and one flat line along the axis -- true, but it reads
    # as a half-broken plot rather than as a spectrum.
    from shaarp.desktop_app import TOOLTIPS

    phi = next(s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
               if s.toolTip() == TOOLTIPS["polarizer"] and s.isEnabled())
    phi.setValue(30.0)
    pump(app)
    press_update(app, si)
    pump(app, 10)
    spectrum = out / "_static" / "screens" / "spectrum.png"
    win.grab().save(str(spectrum))
    print(f"  wrote {spectrum.name}")

    # --- SI tab: the first-run page's OWN result ---
    # guide/first_run.md walks a reader through GaAs (111) at normal incidence and then showed
    # them the si_tab card, which is LiNbO3 at 45 deg: a different case, a different pattern. The
    # page's figure should be the result its steps produce.
    sweep.setChecked(False)
    pump(app)
    set_combo(app, si, "GaAs (111)")
    set_combo(app, si, "SHG Simulation")
    phi.setValue(0.0)
    set_incidence(app, si, 0.0)
    # ...and back to the wavelength a genuine first launch opens at. This case does not own a
    # wavelength, so the field keeps whatever the previous capture left in it (0.8 um, from the
    # LiNbO3 card above) -- and the first-run page's own step 1 tells the reader it reads 1.064.
    wavelength = next(s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
                      if s.toolTip() == TOOLTIPS["wavelength"])
    wavelength.setValue(1.064)
    # The KTP pick above fitted the scan range to its table (0.86-3.54 um); a first launch shows
    # the fields' own defaults, so put them back before the frame is taken.
    for spin, default in zip((s for s in si.findChildren(QtWidgets.QDoubleSpinBox)
                              if s.toolTip() == TOOLTIPS["lambda_range"]), (0.55, 0.8, 0.01)):
        spin.setValue(default)
    pump(app)
    press_update(app, si)
    pump(app, 10)
    first = out / "_static" / "screens" / "first_run_result.png"
    win.grab().save(str(first))
    print(f"  wrote {first.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
