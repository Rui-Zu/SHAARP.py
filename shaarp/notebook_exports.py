"""Notebook diagnostics + export controls for the validated SHAARP.py workflow.

This is the "diagnostics panels and export controls" piece of the interactive
Jupyter workstream, built over the already-validated figure-data helpers
(shaarp.figuredata) and public facades. Pure functions: every exporter takes an
explicit destination path and returns the path; nothing writes implicitly. No
plotting library is imported.

  - figure_data_to_csv / figure_data_to_json: serialize a FigureData (the
    plot-ready Maker/sample-rotation/Fresnel data) to disk for a notebook's
    "Export" button or downstream analysis.
  - diagnostics(result_or_figuredata): a compact dict panel -- validation status,
    series names, per-series min/max/mean, x-range, point count -- the
    "diagnostics panel" a notebook would render next to a plot.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .figuredata import FigureData


def figure_data_to_csv(fd: FigureData, path: str | Path) -> Path:
    """Write a FigureData to CSV: first column x, one column per y-series."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series_names = list(fd.series)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([fd.x_label, *series_names])
        for i, xv in enumerate(fd.x):
            writer.writerow([xv, *[fd.series[name][i] for name in series_names]])
    return path


def figure_data_to_json(fd: FigureData, path: str | Path) -> Path:
    """Write a FigureData to JSON (round-trippable: title/labels/x/series/status)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": fd.title,
        "x_label": fd.x_label,
        "y_label": fd.y_label,
        "x": fd.x,
        "series": fd.series,
        "validation_status": fd.validation_status,
        "note": fd.note,
        "meta": fd.meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def diagnostics(obj: Any) -> dict[str, Any]:
    """Compact diagnostics panel for a FigureData (or a SHAARPResult-like object).

    For a FigureData: validation status + per-series min/max/mean + x-range +
    point count. For a SHAARPResult-like object (has .validation and .numeric),
    summarize validation status + numeric array shapes."""
    if isinstance(obj, FigureData):
        x = np.asarray(obj.x, dtype=float)
        per_series = {}
        for name, ys in obj.series.items():
            arr = np.asarray(ys, dtype=float)
            per_series[name] = {
                "min": float(np.min(arr)) if arr.size else None,
                "max": float(np.max(arr)) if arr.size else None,
                "mean": float(np.mean(arr)) if arr.size else None,
            }
        return {
            "kind": "figure_data",
            "title": obj.title,
            "validation_status": obj.validation_status,
            "point_count": int(x.size),
            "x_range": [float(np.min(x)), float(np.max(x))] if x.size else None,
            "series": per_series,
            "note": obj.note,
        }
    # SHAARPResult-like duck-typing
    validation = getattr(obj, "validation", None)
    numeric = getattr(obj, "numeric", None)
    if validation is not None and isinstance(numeric, dict):
        shapes = {}
        for key, val in numeric.items():
            try:
                shapes[key] = list(np.asarray(val, dtype=complex).shape)
            except (TypeError, ValueError):
                shapes[key] = "scalar"
        return {
            "kind": getattr(obj, "kind", "shaarp_result"),
            "validation_status": getattr(validation, "status", str(validation)),
            "numeric_keys": list(numeric),
            "numeric_shapes": shapes,
        }
    raise TypeError("diagnostics() expects a FigureData or a SHAARPResult-like object.")


__all__ = ["figure_data_to_csv", "figure_data_to_json", "diagnostics"]
