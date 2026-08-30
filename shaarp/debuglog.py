"""Always-on GUI debug log.

Before this module, a failed Update left only a one-line statusBar message (type + str of the
exception) -- no traceback anywhere, so a field report ("I clicked Update and it looked fine /
it failed") was unreproducible without a live session. Now every Update appends one line to a
rotating log file, and every swallowed exception lands WITH its full traceback:

    <system temp dir>/shaarp_gui.log (1 MB x 3 rotations; plain text, safe to attach)

The GUI's Help > Debug Info dialog shows the same last-traceback + the log path.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import tempfile
import time
import traceback

_LOGGER: logging.Logger | None = None


def log_path() -> str:
    # tempfile.gettempdir() resolves TEMP/TMP on Windows and TMPDIR//tmp on macOS/Linux — the
    # previous explicit TEMP/TMP env lookup silently dropped the log into the CWD on POSIX
    # (unwritable "/" inside a macOS .app bundle).
    return os.path.join(tempfile.gettempdir(), "shaarp_gui.log")


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        lg = logging.getLogger("shaarp.gui")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        try:
            h = logging.handlers.RotatingFileHandler(
                log_path(), maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            lg.addHandler(h)
        except Exception:  # logging must NEVER break the app (read-only media, etc.)
            lg.addHandler(logging.NullHandler())
        _LOGGER = lg
    return _LOGGER


def log_run(tab: str, functionality: str, case: str, dt_s: float,
            stages: dict[str, float] | None = None) -> None:
    """One line per Update. ``stages`` (Phase 0) carries the per-stage split the GUI
    measures -- compute (physics), figure (matplotlib build), render (canvas swap/draw) --
    so future 'it feels slow' reports localize to a stage without a profiling session."""
    parts = [f"run tab={tab} func={functionality!r} case={case!r} total={dt_s:.3f}s"]
    if stages:
        parts += [f"{k}={v:.3f}s" for k, v in stages.items()]
    get_logger().info(" ".join(parts))


class StageTimer:
    """Cheap cumulative stage stopwatch for on_run (Phase 0): ``with timer("compute"):``
    around each block; overlapping/nested use is summed per label. Never raises."""

    def __init__(self):
        self.stages: dict[str, float] = {}
        self._label = None
        self._t0 = 0.0

    def __call__(self, label: str):
        self._label = label
        return self

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        try:
            self.stages[self._label] = (self.stages.get(self._label, 0.0)
                                        + time.perf_counter() - self._t0)
        except Exception:
            pass
        return False


def log_exception(tab: str, functionality: str, exc: BaseException) -> str:
    """Log the FULL traceback; return it so the GUI can keep it for the Debug Info dialog."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    get_logger().error("EXCEPTION tab=%s func=%r at %s\n%s",
                       tab, functionality, time.strftime("%H:%M:%S"), tb)
    return tb
