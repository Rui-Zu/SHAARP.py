# Installation & launch

## Option 1 — the packaged app (Windows, macOS — no Python needed)

Download the bundle for your system from this repository's **Releases** page (the app and the
library live in one repo; each bundle is built by CI directly from the tagged source), extract it,
and run. The bundle contains the interpreter, all dependencies, and the validation benchmark data.
If that page is empty, no packaged build has been published yet — use Option 2 below in the
meantime.

| System | Download | Run |
|---|---|---|
| Windows (64-bit) | `SHAARP_py_v…_win64.zip` | double-click `SHAARP_py\SHAARP_py.exe` |
| macOS (Apple Silicon) | `SHAARP_py_v…_macos.zip` | open `SHAARP_py/SHAARP_py.app` |

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
pip install "shaarp-py[desktop,symbolic] @ git+https://github.com/Rui-Zu/SHAARP.py"
```

If your system says `pip` is not found, write `python -m pip install …` instead.

You now have `import shaarp` from anywhere, plus two commands: **`shaarp-gui`** (launches the app)
and `shaarp` (a small CLI).

**Choosing less than everything.** The base requirements — NumPy, SciPy, matplotlib — always come
along. Replace the bracketed extras in the command above with:

| Extras | Adds |
|---|---|
| *(omit the brackets entirely)* | nothing — solvers only |
| `[symbolic]` | closed-form symbolic expressions and d-extraction (SymPy) |
| `[interactive]` | the Jupyter-widget session (ipywidgets) |
| `[desktop,symbolic]` | everything, including the desktop GUI — recommended |

**Working on the source** — clone it and install in place, so edits take effect without
reinstalling:

```bash
git clone https://github.com/Rui-Zu/SHAARP.py
cd SHAARP.py
pip install -e ".[desktop,symbolic]"
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

Next: {doc}`interface`.
