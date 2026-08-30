"""GUI widget-tree introspection.

The ONE implementation of the interactive-widget walk, shared by:
  - the app itself: session save/load serializes EVERY walked input widget (so "full input
    coverage" is DEFINED by the same walk the coverage gate enforces -- a widget cannot be
    audited but unsaved, or saved but unaudited);
  - tests/gui_harness.py (the causality matrices + the coverage gate re-export from here).

Identity scheme: "{tab}:{ClassName}:{key}#{ordinal}" where key is, in priority order, the
TOOLTIPS dict key the widget's tooltip matches; a snippet of a custom tooltip; text()/prefix()/
placeholder; objectName; else "anon". Idents are human-readable, resilient to layout moves, and
change only when a widget's semantic handle (tooltip/text) changes.
"""

from __future__ import annotations

from collections import defaultdict

from PySide6 import QtWidgets


def _interactive_class_map():
    """Ordered (canonical-name, Qt class) pairs. isinstance-based: the app subclasses
    QDoubleSpinBox (_NumBox), and a NAME match would silently skip every numeric spin --
    exactly the enumeration gap this module exists to prevent."""
    return (
        ("QComboBox", QtWidgets.QComboBox),
        ("QDoubleSpinBox", QtWidgets.QDoubleSpinBox),
        ("QSpinBox", QtWidgets.QSpinBox),
        ("QCheckBox", QtWidgets.QCheckBox),
        ("QPushButton", QtWidgets.QPushButton),
        ("QLineEdit", QtWidgets.QLineEdit),
        ("QSlider", QtWidgets.QSlider),
        ("QToolButton", QtWidgets.QToolButton),  # collapsible-group headers
        # checkable collapsible groups (the _collapsible_group disclosure toggles). Added
        # to close the walk gap the grill named -- their state is now walked (so the
        # coverage gate sees them) but NOT serialized (widget_value returns None): collapse is
        # transient disclosure, and F48 relevance is DERIVED from functionality, not saved.
        ("QGroupBox", QtWidgets.QGroupBox),
    )


def _tooltip_key(widget):
    from .desktop_app import TOOLTIPS

    tip = widget.toolTip()
    if not tip:
        return None
    for key, text in TOOLTIPS.items():
        if tip == text:
            return key
    return None


def _semantic_key(widget):
    key = _tooltip_key(widget)
    if key:
        return f"tip={key}"
    tip = widget.toolTip()
    if tip:  # custom tooltip (e.g. the analyzer-offset spin)
        snippet = " ".join(tip.split())[:40]
        return f"tipx={snippet}"
    for attr in ("text", "prefix", "placeholderText"):
        fn = getattr(widget, attr, None)
        if fn:
            try:
                val = fn()
            except Exception:
                val = ""
            if val:
                return f"{attr}={' '.join(str(val).split())[:40]}"
    if widget.objectName():
        return f"name={widget.objectName()}"
    return "anon"


def _inside_composite(widget):
    """True for the internal QLineEdits living inside QComboBox/QAbstractSpinBox."""
    p = widget.parent()
    while p is not None:
        if isinstance(p, (QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
            return True
        p = p.parent()
    return False


def iter_interactive_widgets(page, tab_name):
    """Yield (ident, widget) for every interactive control on a tab page, stable order."""
    counters = defaultdict(int)
    out = []
    for w in page.findChildren(QtWidgets.QWidget):
        canon = None
        for name, qcls in _interactive_class_map():
            if isinstance(w, qcls):
                canon = name
                break
        if canon is None:
            continue
        if isinstance(w, QtWidgets.QLineEdit) and _inside_composite(w):
            continue  # composite internals, not user-facing controls
        out.append((canon, w))
    for canon, w in out:
        key = f"{canon}:{_semantic_key(w)}"
        ident = f"{tab_name}:{key}#{counters[key]}"
        counters[key] += 1
        yield ident, w


def walk_all(win, tabs=None):
    """Walk both tab pages plus window-level chrome; yield (ident, widget)."""
    if tabs is None:
        tabs = win.findChild(QtWidgets.QTabWidget)
    pages = [(tabs.widget(0), "si"), (tabs.widget(1), "ml")]
    seen = set()
    for page, name in pages:
        for ident, w in iter_interactive_widgets(page, name):
            seen.add(id(w))
            yield ident, w
    counters = defaultdict(int)
    for w in win.findChildren(QtWidgets.QWidget):
        canon = None
        for cname, qcls in _interactive_class_map():
            if isinstance(w, qcls):
                canon = cname
                break
        if canon is None or id(w) in seen:
            continue
        if isinstance(w, QtWidgets.QLineEdit) and _inside_composite(w):
            continue
        key = f"{canon}:{_semantic_key(w)}"
        ident = f"win:{key}#{counters[key]}"
        counters[key] += 1
        yield ident, w


# ---------------------------------------------------------------------------------------
# Session serialization: value <-> widget, driven by the SAME walk.
# Buttons / tool-buttons / sliders are ACTIONS or mirrors, not state -> not serialized.
# ---------------------------------------------------------------------------------------

def widget_value(w):
    """The serializable state of an input widget, or None if it carries no state."""
    if isinstance(w, QtWidgets.QComboBox):
        return {"t": "combo", "v": w.currentText()}
    if isinstance(w, QtWidgets.QDoubleSpinBox):
        return {"t": "dspin", "v": float(w.value())}
    if isinstance(w, QtWidgets.QSpinBox):
        return {"t": "spin", "v": int(w.value())}
    if isinstance(w, QtWidgets.QCheckBox):
        return {"t": "check", "v": bool(w.isChecked())}
    if isinstance(w, QtWidgets.QLineEdit):
        return {"t": "line", "v": w.text()}
    return None


def apply_widget_value(w, spec):
    """Apply a serialized value; EQUALITY-GUARDED so restoring an unchanged value emits no
    signal (a case-study session must not be flipped to Custom by its own restore)."""
    t, v = spec.get("t"), spec.get("v")
    if t == "combo" and isinstance(w, QtWidgets.QComboBox):
        if w.currentText() != v and w.findText(str(v)) >= 0:
            w.setCurrentText(str(v))
    elif t == "dspin" and isinstance(w, QtWidgets.QDoubleSpinBox):
        if abs(w.value() - float(v)) > 0.0:
            w.setValue(float(v))
    elif t == "spin" and isinstance(w, QtWidgets.QSpinBox):
        if w.value() != int(v):
            w.setValue(int(v))
    elif t == "check" and isinstance(w, QtWidgets.QCheckBox):
        if w.isChecked() != bool(v):
            w.setChecked(bool(v))
    elif t == "line" and isinstance(w, QtWidgets.QLineEdit):
        if w.text() != str(v):
            w.setText(str(v))


def collect_session_state(win) -> dict:
    """{"tabs": {ident: value-spec}} for every state-carrying walked widget."""
    state = {}
    for ident, w in walk_all(win):
        spec = widget_value(w)
        if spec is not None:
            state[ident] = spec
    return state


def apply_session_state(win, state: dict) -> list[str]:
    """Apply a collected state in WALK ORDER (construction order approximates dependency
    order: case/preset combos configure panels before the cell values land). Returns the
    idents present in the file but absent from the live tree (version drift), for the UI
    to report instead of failing silently."""
    live = {ident: w for ident, w in walk_all(win)}
    missing = []
    for ident, spec in state.items():
        w = live.get(ident)
        if w is None:
            missing.append(ident)
            continue
        apply_widget_value(w, spec)
    QtWidgets.QApplication.instance().processEvents()
    return missing
