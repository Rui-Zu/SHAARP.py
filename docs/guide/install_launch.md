# Installation & launch

## Option 1 — the packaged app (Windows, macOS — no Python needed)

**→ [Download the latest release](https://github.com/Rui-Zu/SHAARP.py/releases/latest)**

Pick the file for your system, extract it, and run it. There is nothing else to install: the
bundle already contains the Python interpreter, every dependency, and the validation benchmark
data. (The app and the library live in one repository; each bundle is built by CI directly from
the tagged source.)

| System | File to download | How to run it |
|---|---|---|
| Windows (64-bit) | `SHAARP_py_v1.0.0_win64.zip` (120 MB) | extract, then double-click `SHAARP_py\SHAARP_py.exe` |
| macOS, Apple Silicon (M-series) | `SHAARP_py_v1.0.0_macos.zip` (83 MB) | extract, then open `SHAARP_py/SHAARP_py.app` |

On an **Intel Mac**, Linux, or any other system there is no packaged build — use Option 2 below.
The macOS bundle is Apple Silicon only and an Intel Mac refuses it with *"not supported on this
Mac"*.

**Windows first launch** — extract the zip before running anything: the app needs the `_internal`
folder next to the `.exe`, so double-clicking straight out of the zip fails, usually with no
message at all. Then:

- Windows may show a blue **"Windows protected your PC"** box, because the app is not code-signed
  → **More info → Run anyway**.
- The first launch takes 10–30 seconds while the system scans the bundle, with no window on
  screen. Later launches are quick.

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

**2. Install SHAARP.py** — one command, from any directory. Nothing to download or unzip:

```bash
pip install "shaarp-py[desktop,interactive] @ git+https://github.com/Rui-Zu/SHAARP.py"
```

If your system says `pip` is not found, write `python -m pip install …` instead.

You now have `import shaarp` from anywhere, plus two commands: **`shaarp-gui`** (launches the app)
and `shaarp` (a small CLI).

**Choosing less than everything.** The base requirements — NumPy, SciPy, matplotlib and SymPy —
always come along. Replace the bracketed extras in the command above with:

| Extras | Adds |
|---|---|
| *(omit the brackets entirely)* | nothing — the full solver library, including the closed-form symbolic tools |
| `[interactive]` | the Jupyter-widget session (ipywidgets) |
| `[desktop]` | the desktop GUI and the `shaarp-gui` command (Qt) |
| `[desktop,interactive]` | everything — recommended |

**Working on the source** — clone it and install in place, so edits take effect without
reinstalling:

```bash
git clone https://github.com/Rui-Zu/SHAARP.py
cd SHAARP.py
pip install -e ".[desktop,interactive]"
```

**Running without installing** — from the repository root, with the dependencies present:

```bash
python run_shaarp_desktop.py
```

or, equivalently, `python -m shaarp.desktop_app`.

## Verifying the build (developers)

The frozen executable exposes self-check flags used by the release gate:

```bash
SHAARP_py.exe --self-check
SHAARP_py.exe --gui-smoke
```

`--self-check` runs the real SI / ML / Maker / Fresnel compute paths and asserts the benchmark data
is bundled; `--gui-smoke` drives every tab × functionality × angle (including θ = 0) headlessly.

A headless GUI smoke run from source is also possible by setting `QT_QPA_PLATFORM=offscreen` before
launching.

## Built-in help

Inside the app, use **Help → User Guide** for a condensed workflow summary and **Help → About /
References** for citation information. Hover any control for a tooltip describing it.

Next: {doc}`first_run` — a three-click calculation to confirm the install works.
