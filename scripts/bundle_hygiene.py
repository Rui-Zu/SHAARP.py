"""Bundle-hygiene check for the frozen SHAARP.py app — shared by verify_release.py (Windows gate
step 2b) and scripts/build_gui_bundle.py (all CI platforms).

Permanent ratchet (extracted cross-platform): the built app must carry
EXACTLY the git-shippable benchmarks set and ZERO dev-scratch / local-path / username content.
This is what keeps the Release-asset zip as clean as the public repo — .gitignore alone cannot do
it (PyInstaller copies from disk). The username content scan bans only the ``Users/<name>`` class;
the developer's PROJECT path inside frozen Wolfram provenance is the documented accepted residual.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# the banned username, assembled from split literals so THIS shipped file never contains it
_USERNAME = "51" "093"
_PII = re.compile(rb"Users[\\/]+" + _USERNAME.encode())
_BAD_DIRS = {"__pycache__", ".wolfram_tmp", "discrepancy_drafts"}
_SCRATCH_WL = re.compile(r"^(analyze_|audit_|convert_|debug_|diagnose_|dump_|extract_|inspect_"
                         r"|parse_|probe_|wolfram_current_smoke)|^load_.*_smoke\.wl$")


def check_bundle(data_root: Path, repo_root: Path) -> tuple[bool, str]:
    """Scan the frozen app's data root (the dir holding ``benchmarks/`` + ``shaarp/`` data —
    ``_internal`` on Windows/Linux onedir, ``Contents/Frameworks`` inside a macOS .app).
    Returns (ok, detail). Roots are resolved first (macOS .app uses directory symlinks)."""
    sys.path.insert(0, str(repo_root / "scripts"))
    from stage_bundle_data import git_shippable_benchmarks  # noqa: E402
    bench = (data_root / "benchmarks").resolve()
    shp = (data_root / "shaarp").resolve()
    if not bench.is_dir():
        return False, f"bundled benchmarks dir missing under {data_root}"
    problems: list[str] = []
    # (a) file-set equality vs the git listing (skipped with a note if git is unavailable)
    rels = git_shippable_benchmarks()
    if rels is None:
        problems.append("git unavailable for set-equality check")
        set_note = "set-eq: SKIPPED (no git)"
    else:
        want = {p.relative_to("benchmarks").as_posix() for p in rels}
        have = {p.relative_to(bench).as_posix() for p in bench.rglob("*") if p.is_file()}
        extra, missing = sorted(have - want), sorted(want - have)
        if extra:
            problems.append(f"{len(extra)} non-shippable file(s) bundled, e.g. {extra[:3]}")
        if missing:
            problems.append(f"{len(missing)} shippable file(s) MISSING from bundle, e.g. {missing[:3]}")
        set_note = f"set-eq: {len(have)} bundled == {len(want)} shippable" if not (extra or missing) \
            else "set-eq: MISMATCH"
    # (b) forbidden names anywhere in the bundled benchmarks + shaarp data
    for base in (bench, shp):
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            rel = p.relative_to(base).as_posix()
            if p.is_dir() and p.name in _BAD_DIRS:
                problems.append(f"forbidden dir {base.name}/{rel}")
            elif p.is_file():
                if (p.suffix in (".pyc", ".log", ".md") or ".bak" in p.name
                        or p.name == "orientation_reference_diagnostics.txt"
                        or (p.suffix == ".wl" and _SCRATCH_WL.match(p.name))):
                    problems.append(f"forbidden file {base.name}/{rel}")
    # (c) username must never appear in bundled data content
    for base in (bench, shp):
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and _PII.search(p.read_bytes()):
                problems.append(f"username PII in {base.name}/{p.relative_to(base).as_posix()}")
    if problems:
        return False, f"{len(problems)} problem(s): " + "; ".join(problems[:8])
    return True, f"{set_note}; 0 forbidden names; 0 username hits"
