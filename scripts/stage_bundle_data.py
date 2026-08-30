"""Stage the PUBLIC (git-shippable) benchmarks tree for the frozen-app build.

WHY: PyInstaller's ``--add-data benchmarks;benchmarks`` copies the
ON-DISK folder verbatim — .gitignore does not apply — so the Release-asset zip used to ship every
dev-scratch file the public repo filters out (one-off Wolfram diagnostic scripts, ``.wolfram_tmp``
transients, ``__pycache__`` bytecode with absolute source paths, internal notes — several carrying
the developer's local paths). This helper stages ONLY the files git would ship into
``build/bundle_benchmarks/``; both build entry points (``verify_release.py`` and
``build_desktop_exe.bat``) point ``--add-data`` at the staged tree, and the release gate's
"bundle hygiene" step verifies the built app against the same git listing.

Single source of truth = .gitignore, via::

    git ls-files -z --cached --others --exclude-standard -- benchmarks

(``--cached --others`` covers both tracked and untracked-but-not-ignored files, so the listing is
correct before AND after the first commit.) If git is unavailable (e.g. building from a source zip
rather than a checkout), the helper falls back to a FULL copy with a loud warning — the gate's
bundle-hygiene step will then fail on any scratch content, which is the intended backstop.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "build" / "bundle_benchmarks"
# Staging must never silently produce a gutted tree: the reference layer is ~600+ files, and the
# frozen --self-check depends on this file being present.
MIN_FILES = 500
SENTINEL = "multilayer_system_benchmarks_v1.json"


def _rmtree_retry(path: Path, tries: int = 5) -> None:
    # Cloud file-sync clients transiently lock folders -> WinError 145. Retry with backoff.
    for i in range(tries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except OSError:
            time.sleep(2.0 * (i + 1))
    if path.exists():
        raise OSError(f"could not remove {path} (locked by Drive sync?)")


def git_shippable_benchmarks() -> list[Path] | None:
    """Repo-relative paths (under benchmarks/) that git would ship, or None if git is unusable."""
    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard", "--", "benchmarks"],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    rels = sorted({p for p in r.stdout.split("\0") if p})
    return [Path(p) for p in rels]


def stage_shippable_benchmarks(dst: Path = DEFAULT_DST) -> int:
    """Stage the git-shippable benchmarks tree into ``dst``; return the staged file count."""
    rels = git_shippable_benchmarks()
    _rmtree_retry(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if rels is None:
        print("WARNING: git unavailable — staging a FULL benchmarks copy (the release gate's "
              "bundle-hygiene step will reject scratch content).", flush=True)
        shutil.copytree(ROOT / "benchmarks", dst, dirs_exist_ok=True)
        count = sum(1 for p in dst.rglob("*") if p.is_file())
    else:
        count = 0
        for rel in rels:
            src = ROOT / rel
            if not src.is_file():
                continue
            out = dst / rel.relative_to("benchmarks")
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            count += 1
    if count < MIN_FILES or not (dst / SENTINEL).exists():
        raise SystemExit(
            f"stage_bundle_data: staged tree looks wrong ({count} files; sentinel "
            f"{SENTINEL} {'present' if (dst / SENTINEL).exists() else 'MISSING'}) — refusing to build")
    print(f"staged {count} git-shippable benchmarks files -> {dst}", flush=True)
    return count


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DST
    stage_shippable_benchmarks(target)
