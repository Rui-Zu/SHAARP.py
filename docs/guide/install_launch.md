# Installation & launch

## Option 1 — the packaged app (Windows, macOS — no Python needed)

Download the bundle for your system from this repository's **Releases** page (the app and the
library live in one repo; each bundle is built by CI directly from the tagged source), extract it,
and run. The bundle contains the interpreter, all dependencies, and the validation benchmark data.

| System | Download | Run |
|---|---|---|
| Windows (64-bit) | `SHAARP_py_v…_win64.zip` | double-click `SHAARP_py\SHAARP_py.exe` |
| macOS (Apple Silicon) | `SHAARP_py_v…_macos.zip` | open `SHAARP_py.app` |

**macOS first launch** — the app is not Apple-signed, so macOS blocks the first open. One-time fix:

- macOS 15 (Sequoia) and newer: try to open the app once, then go to **System Settings → Privacy &
  Security** and click **"Open Anyway"**.
- macOS 14 and older: **right-click (Control-click) the app → Open → Open**.
- Terminal alternative: `xattr -dr com.apple.quarantine SHAARP_py.app`

## Option 2 — from source (for scripting; Python ≥ 3.10)

Only needed if you want to *script* SHAARP.py — the app in Option 1 needs none of this. The steps
assume no prior Python experience.

**1. Install Python** (skip if you already have it) — from
[python.org/downloads](https://www.python.org/downloads/), version 3.10 or newer. On Windows, tick
**"Add python.exe to PATH"** on the installer's first screen; that one checkbox prevents most
beginner problems. Check it worked by typing `python --version` in a terminal.

**2. Get the code** — SHAARP.py is not on PyPI, so you install it from a copy of the repository:
on the GitHub page, *Code ▸ Download ZIP* and unzip it, or `git clone <repository URL>`.

**3. Open a terminal inside that folder** — Windows: open the folder in File Explorer, click the
address bar, type `cmd`, press Enter. macOS: right-click the folder → **New Terminal at Folder**.

**4. Install it** — run ONE of these. Each installs the library plus the optional extras named in
brackets (the base requirements — NumPy, SciPy, matplotlib — always come along). If your system
says `pip` is not found, write `python -m pip install …` instead:

```bash
pip install. # library only
pip install ".[symbolic]" # + closed-form symbolic expressions & d-extraction (SymPy)
pip install ".[interactive]" # + the Jupyter-widget session (ipywidgets)
pip install ".[desktop,symbolic]" # everything, incl. the desktop GUI — recommended
```

You now have `import shaarp` from anywhere, plus two commands: **`shaarp-gui`** (launches the app)
and `shaarp` (a small CLI). Add `-e` (`pip install -e ".[desktop,symbolic]"`) only if you plan to
edit the source — it makes your edits take effect without reinstalling.

**Running without installing** — from the repository root, with the dependencies present:

```bash
python run_shaarp_desktop.py
# or, equivalently:
python -m shaarp.desktop_app
```

## Verifying the build (developers)

The frozen executable exposes self-check flags used by the release gate:

```bash
SHAARP_py.exe --self-check # runs the real SI/ML/Maker/Fresnel compute paths, asserts data bundled
SHAARP_py.exe --gui-smoke # drives every tab x functionality x angle (incl. theta=0) headlessly
```

A headless GUI smoke run from source is also possible by setting `QT_QPA_PLATFORM=offscreen` before
launching.

## Built-in help

Inside the app, use **Help → User Guide** for a condensed workflow summary and **Help → About /
References** for citation information. Hover any control for a tooltip describing it.

Next: {doc}`interface`.
