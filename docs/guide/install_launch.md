# Installation & launch

## Option 1 — the packaged app (Windows and macOS, no Python needed)

**[Download the latest release](https://github.com/Rui-Zu/SHAARP.py/releases/latest)**

Pick the file for your system, extract it, and run it. There is nothing else to install and no
license to buy: the bundle already contains the Python interpreter, every dependency and the
reference data.

| System | File to download | How to run it |
|---|---|---|
| Windows (64-bit) | `SHAARP_py_v…_win64.zip` (about 120 MB) | extract, then double-click `SHAARP_py\SHAARP_py.exe` |
| macOS, Apple Silicon (M-series) | `SHAARP_py_v…_macos_arm64.zip` (about 83 MB) | extract, then open `SHAARP_py/SHAARP_py.app` |

Extract the zip before running: the app needs the `_internal` folder next to it. The app is not
code-signed, so Windows and macOS ask you to allow it the first time, and the first launch takes a
moment with nothing on screen. If it does not open, {doc}`faq` has the fix for each system.

On an Intel Mac, on Linux, or on any other system there is no packaged build, so use Option 2 below.

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

This installs straight from GitHub, so pip needs **git** on your PATH: if it says git is not
found, install it from [git-scm.com/downloads](https://git-scm.com/downloads) and reopen the
terminal. If your system says `pip` is not found, write `python -m pip install …` instead.

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

## Built-in help

Inside the app, the **Help** menu has a **User Guide** with a condensed workflow summary and an
**About / References** entry with citation information. Hover any control for a tooltip describing it.

Next: {doc}`first_run` — a three-click calculation to confirm the install works.
