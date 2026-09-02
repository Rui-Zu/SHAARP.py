"""Build the installation-free SHAARP.py GUI bundle — the ONE cross-platform build definition.

Used by both build paths so they can never drift:
  * GitHub Actions release workflow (all platforms): python scripts/build_gui_bundle.py
  * build_desktop_exe.bat convenience build: ... --no-package

Pipeline: stage the git-shippable benchmarks tree (scripts/stage_bundle_data.py — .gitignore is the
single source of truth) -> per-OS icon (.ico on Windows; .icns generated from the 1085 px
shaarp_icon.png via sips+iconutil on macOS) -> the exact PyInstaller invocation the
Windows gate validated (only the --add-data separator and icon differ per OS) -> locate the built
app -> copy RELEASE_README.txt/LICENSE next to it -> bundle-hygiene check (scripts/bundle_hygiene.py)
-> run the FROZEN app --self-check + --gui-smoke headless -> package into dist/release/ as
SHAARP_py_v{version}_{platform}.zip.

Packaging notes: macOS MUST use `ditto -c -k --keepParent` (shutil's zip breaks .app symlinks
and executable bits); the archive root is always a single `SHAARP_py` folder. When GITHUB_REF_NAME is a v-tag the version
in pyproject.toml must match it (catches tagging v1.0.1 while pyproject still says 1.0.0).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bundle_hygiene import check_bundle  # noqa: E402
from stage_bundle_data import stage_shippable_benchmarks  # noqa: E402

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

SMOKE_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
             "QT_QPA_PLATFORM": "offscreen"}
if IS_WIN:
    # Windows-offscreen font-database workaround only; macOS (CoreText)
    # resolve system fonts natively under the offscreen platform.
    SMOKE_ENV["QT_QPA_FONTDIR"] = "C:/Windows/Fonts"

# Ceiling for ONE frozen smoke invocation. The smoke takes a couple of minutes on a cold runner, so
# this is not a performance budget -- it is the guard that turns a hang into a diagnosis. Raise via
# SHAARP_SMOKE_TIMEOUT for a genuinely slow machine.
SMOKE_TIMEOUT = int(os.environ.get("SHAARP_SMOKE_TIMEOUT", "600"))


def _rmtree_retry(path: Path, tries: int = 5) -> None:
    # Cloud file-sync clients transiently lock dist/_internal. Retry with backoff.
    for i in range(tries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(2.0 * (i + 1))
    if path.exists():
        raise SystemExit(f"could not remove {path} (locked by Drive sync or a running app?)")


def project_version() -> str:
    import tomllib
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def _assert_tag_coherence(version: str) -> None:
    tag = os.environ.get("GITHUB_REF_NAME", "")
    if tag.startswith("v") and tag != f"v{version}":
        raise SystemExit(f"tag {tag!r} does not match pyproject version {version!r} — "
                         f"bump pyproject.toml (or retag) before releasing")


def _make_icns() -> Path:
    """Generate build/shaarp_icon.icns from the 1085 px master PNG (macOS only; sips + iconutil
    ship with macOS, including the GitHub runners)."""
    src = ROOT / "shaarp" / "assets" / "shaarp_icon.png"
    iconset = ROOT / "build" / "shaarp_icon.iconset"
    icns = ROOT / "build" / "shaarp_icon.icns"
    _rmtree_retry(iconset)
    iconset.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 64, 128, 256, 512):
        for scale, suffix in ((1, ""), (2, "@2x")):
            px = size * scale
            out = iconset / f"icon_{size}x{size}{suffix}.png"
            subprocess.run(["sips", "-z", str(px), str(px), str(src), "--out", str(out)],
                           check=True, capture_output=True)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    return icns


def scipy_compat_collect_args() -> list[str]:
    """Tell PyInstaller where THIS scipy vendors ``array_api_compat`` -- the location moved.

    scipy <= 1.17 keeps it at ``scipy._lib.array_api_compat``; scipy >= 1.18 at
    ``scipy._external.array_api_compat``. PyInstaller 6.x's bundled scipy hook still names the old
    path, so against a newer scipy its hidden import resolves to nothing (the build only prints
    "Hidden import ... not found"), the real location is never collected, and the FROZEN app raises
    ModuleNotFoundError for ``...array_api_compat.numpy.fft`` the first time it touches scipy.

    Nothing run from source can see this, and the build itself exits 0 -- it surfaced only in the
    release job's frozen --self-check, on runners whose unpinned scipy had moved ahead of the
    development environment. Detecting the path that actually exists lets the bundle follow scipy
    across the rename instead of being pinned to one version's internal layout.
    """
    args: list[str] = []
    for pkg in ("scipy._external.array_api_compat", "scipy._lib.array_api_compat"):
        try:
            found = importlib.util.find_spec(pkg) is not None
        except (ImportError, AttributeError, ValueError):
            found = False  # parent package absent, or not a package
        if found:
            args += ["--collect-submodules", pkg]
    return args


def build_app(*, skip_hygiene: bool = False, skip_smoke: bool = False) -> Path:
    """Stage + PyInstaller + identity files (+ hygiene + frozen smoke unless skipped).
    Returns the built app path (dist/SHAARP_py or dist/SHAARP_py.app)."""
    stage_shippable_benchmarks()
    _rmtree_retry(ROOT / "dist" / "SHAARP_py")
    _rmtree_retry(ROOT / "dist" / "SHAARP_py.app")
    _rmtree_retry(ROOT / "build" / "SHAARP_py")

    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--windowed", "--name", "SHAARP_py",
           "--paths", ".",
           "--hidden-import", "shaarp.quartz_au_docs_case",
           "--hidden-import", "shaarp.desktop_app", "--hidden-import", "shaarp.casestudy_materials",
           "--collect-submodules", "shaarp", "--collect-data", "shaarp",
           *scipy_compat_collect_args(),
           "--add-data", f"build/bundle_benchmarks{os.pathsep}benchmarks",
           "--exclude-module", "torch", "--exclude-module", "torchvision",
           "--exclude-module", "notebook", "--exclude-module", "jupyterlab",
           "--exclude-module", "jupyter_server", "--exclude-module", "IPython",
           "--exclude-module", "ipywidgets", "run_shaarp_desktop.py"]
    # insert BEFORE "--paths" (index 7) — between "--name SHAARP_py" and "--paths ."; inserting any
    # later would split an option from its value
    if IS_WIN:
        cmd[7:7] = ["--icon", "shaarp/assets/shaarp_icon.ico"]
    elif IS_MAC:
        cmd[7:7] = ["--icon", str(_make_icns())]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

    app = ROOT / "dist" / ("SHAARP_py.app" if IS_MAC else "SHAARP_py")
    if not app.exists():
        raise SystemExit(f"build produced no app at {app}")
    exe = frozen_executable(app)
    if not exe.exists():
        raise SystemExit(f"frozen executable missing: {exe}")

    # release identity: README + LICENSE travel next to the app (inside the onedir folder on
    # Windows; alongside the .app in the archive staging folder on macOS — handled there)
    if not IS_MAC:
        shutil.copyfile(ROOT / "RELEASE_README.txt", app / "README.txt")
        shutil.copyfile(ROOT / "LICENSE", app / "LICENSE.txt")

    if not skip_hygiene:
        ok, detail = check_bundle(data_root(app), ROOT)
        print(f"bundle hygiene: {detail}", flush=True)
        if not ok:
            raise SystemExit("bundle hygiene FAILED")
    if not skip_smoke:
        # each flag carries ALL its needles and runs ONCE -- the smoke previously reported
        # success while driving half the matrix (14 -> 7) because only "errors = 0" was checked.
        for flag, needles in (("--self-check", ("benchmarks bundled = True",)),
                              ("--gui-smoke", ("errors = 0", "drove 14 "))):
            try:
                r = subprocess.run([str(exe), flag], env=SMOKE_ENV, text=True,
                                   capture_output=True, timeout=SMOKE_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                # A --windowed build has no console, so on Windows an unhandled exception opens a
                # modal "Fatal error" MESSAGE BOX and waits for OK. Headless, nobody clicks it: the
                # same import failure that macOS reported in seconds hung a release job for 3h43m,
                # to GitHub's 6-hour ceiling. Bound it, and print whatever the app managed to say.
                partial = ((exc.stdout or "") + (exc.stderr or "")).strip()
                print(f"{flag}: TIMED OUT after {SMOKE_TIMEOUT}s", flush=True)
                for line in partial.splitlines()[-15:]:
                    print(f"  | {line}", flush=True)
                raise SystemExit(
                    f"frozen {flag} HUNG (>{SMOKE_TIMEOUT}s). A windowed build blocks on a modal "
                    f"error dialog, so this usually means the app raised at startup -- most often a "
                    f"module PyInstaller did not collect. Run the app locally to see the dialog, or "
                    f"raise SHAARP_SMOKE_TIMEOUT if the machine is merely slow.")
            out = r.stdout + r.stderr
            tail = out.strip().splitlines()[-1:] or [""]
            print(f"{flag}: rc={r.returncode} | {tail[0]}", flush=True)
            missing = [n for n in needles if n not in out]
            if r.returncode != 0 or missing:
                raise SystemExit(f"frozen {flag} FAILED (needles {missing!r} not found)")
    return app


def frozen_executable(app: Path) -> Path:
    if IS_MAC:
        return app / "Contents" / "MacOS" / "SHAARP_py"
    return app / ("SHAARP_py.exe" if IS_WIN else "SHAARP_py")


def data_root(app: Path) -> Path:
    """Directory holding the bundled ``benchmarks/`` + ``shaarp/`` data. Windows onedir:
    ``_internal``. macOS .app: PyInstaller >= 6 places the payload under Contents/Frameworks with
    symlinks from Contents/Resources — pick whichever actually holds the sentinel, resolved."""
    if not IS_MAC:
        return app / "_internal"
    sentinel = Path("benchmarks") / "multilayer_system_benchmarks_v1.json"
    for cand in (app / "Contents" / "Frameworks", app / "Contents" / "Resources"):
        if (cand / sentinel).exists():
            return cand.resolve()
    raise SystemExit(f"could not locate bundled benchmarks inside {app}")


def platform_token() -> str:
    """Suffix identifying a release archive.

    macOS carries its CPU architecture because the bundle is NOT universal2. An arm64 build on an
    Intel Mac fails with "not supported on this Mac" -- which looks nothing like the Gatekeeper
    prompt the documentation prepares people for, so the reader has no idea what went wrong. The
    asset was previously called just "macos", which told an Intel user nothing until they had
    downloaded 80 MB. Deriving the architecture rather than hardcoding it means the name cannot
    lie about a bundle built on some other machine.
    """
    if IS_WIN:
        return "win64"
    if IS_MAC:
        return f"macos_{platform.machine()}"  # arm64 on Apple Silicon, x86_64 on Intel
    return sys.platform


def archive_path(out_dir: Path, version: str, token: str | None = None) -> Path:
    """Where the release archive for ``version`` goes.

    NOT ``Path.with_suffix``: a stem like ``SHAARP_py_v1.0.0_macos`` looks suffixed to pathlib,
    which reads ``.0_macos`` as the extension and replaces it -- yielding ``SHAARP_py_v1.0.zip``,
    with the patch digit and the platform token gone, and 1.0.1 colliding with 1.0.0.
    """
    stem = f"SHAARP_py_v{version}_{token or platform_token()}"
    return out_dir / (stem + ".zip")


def package(app: Path, version: str) -> Path:
    out_dir = ROOT / "dist" / "release"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_name = archive_path(out_dir, version)
    base = archive_name.with_name(archive_name.stem)
    if IS_MAC:
        # stage a single SHAARP_py folder holding the .app + identity files, then ditto it
        # (shutil zip breaks .app symlinks/exec bits; ditto preserves both)
        staging = ROOT / "dist" / "release_stage" / "SHAARP_py"
        _rmtree_retry(staging.parent)
        staging.mkdir(parents=True)
        subprocess.run(["ditto", str(app), str(staging / app.name)], check=True)
        shutil.copyfile(ROOT / "RELEASE_README.txt", staging / "README.txt")
        shutil.copyfile(ROOT / "LICENSE", staging / "LICENSE.txt")
        archive = archive_name
        archive.unlink(missing_ok=True)
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(staging), str(archive)], check=True)
    else:
        archive = Path(shutil.make_archive(str(base), "zip", root_dir=app.parent, base_dir=app.name))
        # make_archive APPENDS ".zip" to the stem, so it must land back on archive_path()'s answer.
        # The fence covers archive_path itself, not this round-trip: a future change there could
        # rename the Windows asset out from under the README download table with nothing going red.
        if archive != archive_name:
            raise SystemExit(f"archive name drift: got {archive.name}, expected {archive_name.name}")
    print(f"packaged {archive} ({archive.stat().st_size / 1e6:.0f} MB)", flush=True)
    return archive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-package", action="store_true", help="build + verify, skip archiving")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="skip the frozen --self-check/--gui-smoke (the release gate runs them "
                         "as its own steps)")
    ap.add_argument("--skip-hygiene", action="store_true",
                    help="skip the bundle-hygiene scan (the release gate runs it as step 2b)")
    args = ap.parse_args()
    version = project_version()
    _assert_tag_coherence(version)
    app = build_app(skip_hygiene=args.skip_hygiene, skip_smoke=args.skip_smoke)
    if not args.no_package:
        package(app, version)
    print("build_gui_bundle: OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
