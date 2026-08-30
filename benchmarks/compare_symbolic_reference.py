from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SymbolicComparisonResult:
    expression_id: str
    check: str
    passed: bool
    detail: str
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Mathematica symbolic SHAARP exports against Python/SymPy symbolic outputs."
    )
    parser.add_argument("--mathematica", type=Path, required=True, help="Mathematica symbolic reference JSON.")
    parser.add_argument("--python", type=Path, required=True, help="Python symbolic output JSON.")
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument(
        "--skip-symbolic-residual",
        action="store_true",
        help="Skip simplify(Mathematica - Python) checks and rely on numeric substitution checks.",
    )
    args = parser.parse_args()

    results = compare_symbolic_reference_files(
        args.mathematica,
        args.python,
        atol=args.atol,
        rtol=args.rtol,
        check_symbolic_residual=not args.skip_symbolic_residual,
    )
    failed = [result for result in results if not result.passed]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {result.expression_id} {result.check}: {result.detail} "
            f"max_abs={result.max_abs_error:.3e} max_rel={result.max_rel_error:.3e}"
        )
    if failed:
        raise SystemExit(f"{len(failed)} symbolic reference comparisons failed.")
    print(f"All {len(results)} symbolic reference comparisons passed.")


def compare_symbolic_reference_files(
    mathematica_path: Path,
    python_path: Path,
    *,
    atol: float,
    rtol: float,
    check_symbolic_residual: bool = True,
) -> list[SymbolicComparisonResult]:
    mathematica = json.loads(mathematica_path.read_text(encoding="utf-8"))
    python = json.loads(python_path.read_text(encoding="utf-8"))
    return compare_symbolic_payloads(
        mathematica,
        python,
        atol=atol,
        rtol=rtol,
        check_symbolic_residual=check_symbolic_residual,
    )


def compare_symbolic_payloads(
    mathematica: dict[str, Any],
    python: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    check_symbolic_residual: bool = True,
) -> list[SymbolicComparisonResult]:
    validate_symbolic_reference_payload(mathematica)
    mma_records = _records_by_id(mathematica)
    py_records = _records_by_id(python)
    missing = sorted(set(mma_records) - set(py_records))
    if missing:
        raise ValueError(f"Python symbolic outputs are missing expression ids: {missing}")

    results = []
    for expression_id, mma_record in sorted(mma_records.items()):
        py_record = py_records[expression_id]
        results.extend(_compare_record(mma_record, py_record, atol=atol, rtol=rtol, check_symbolic_residual=check_symbolic_residual))
    return results


def validate_symbolic_reference_payload(payload: dict[str, Any]) -> None:
    source = str(payload.get("source", "")).lower()
    if "mathematica" not in source and "wolfram" not in source:
        raise ValueError("Strict symbolic comparison requires source metadata containing Mathematica or Wolfram.")
    status = str(payload.get("status", "")).lower()
    blocked = ("placeholder", "pending", "not_yet", "requested", "python_regression", "no_values")
    if any(marker in status for marker in blocked):
        raise ValueError(f"Strict symbolic comparison refuses unvalidated reference status: {payload.get('status')!r}.")
    records = payload.get("expressions")
    if not isinstance(records, list) or not records:
        raise ValueError("Symbolic reference payload must contain a non-empty expressions list.")
    for record in records:
        expression_id = record.get("expression_id", "<missing>")
        for key in ("expression_id", "source_stage", "output_component", "symbol_roles"):
            if key not in record:
                raise ValueError(f"Expression {expression_id} is missing required key {key!r}.")
        mode = str(record.get("comparison_mode", "expression"))
        if mode == "singular_diagnostics":
            if "singular_diagnostics" not in record:
                raise ValueError(f"Expression {expression_id} is missing required key 'singular_diagnostics'.")
        elif "expression" not in record:
            raise ValueError(f"Expression {expression_id} is missing required key 'expression'.")


def _records_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("expressions")
    if not isinstance(records, list):
        raise ValueError("Payload must contain an expressions list.")
    out = {}
    for record in records:
        expression_id = record.get("expression_id")
        if not expression_id:
            raise ValueError("Every symbolic expression record must contain expression_id.")
        out[str(expression_id)] = record
    return out


def _compare_record(
    mma_record: dict[str, Any],
    py_record: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    check_symbolic_residual: bool,
) -> list[SymbolicComparisonResult]:
    sp = _sympy()
    expression_id = str(mma_record["expression_id"])
    results = []

    for key in ("source_stage", "output_component"):
        passed = mma_record.get(key) == py_record.get(key)
        results.append(SymbolicComparisonResult(expression_id, key, passed, "metadata match" if passed else "metadata mismatch"))

    mma_roles = dict(mma_record.get("symbol_roles", {}))
    py_roles = dict(py_record.get("symbol_roles", {}))
    roles_match = mma_roles == py_roles
    results.append(SymbolicComparisonResult(expression_id, "symbol_roles", roles_match, "symbol roles match" if roles_match else "symbol roles mismatch"))

    local_symbols = {name: sp.Symbol(name) for name in sorted(set(mma_roles) | set(py_roles))}
    mma_mode = str(mma_record.get("comparison_mode", "expression"))
    py_mode = str(py_record.get("comparison_mode", "expression"))
    mode_match = mma_mode == py_mode
    results.append(SymbolicComparisonResult(expression_id, "comparison_mode", mode_match, "comparison mode match" if mode_match else f"mode mismatch: {mma_mode} vs {py_mode}"))
    if not mode_match:
        return results
    if mma_mode == "singular_diagnostics":
        results.extend(_compare_singular_diagnostics(expression_id, mma_record, py_record, local_symbols, atol=atol, rtol=rtol))
        return results

    mma_expr = _parse_expression(
        mma_record["expression"],
        local_symbols,
        expression_format=str(mma_record.get("expression_format", "sympy")),
    )
    py_expr = _parse_expression(
        py_record["expression"],
        local_symbols,
        expression_format=str(py_record.get("expression_format", "sympy")),
    )

    mma_shape = _shape(mma_expr)
    py_shape = _shape(py_expr)
    shape_match = mma_shape == py_shape
    results.append(SymbolicComparisonResult(expression_id, "shape", shape_match, f"Mathematica shape={mma_shape}, Python shape={py_shape}"))
    if not shape_match:
        return results

    mma_symbols = _free_symbol_names(mma_expr)
    py_symbols = _free_symbol_names(py_expr)
    symbol_sets_match = mma_symbols == py_symbols
    results.append(SymbolicComparisonResult(expression_id, "symbol_set", symbol_sets_match, f"Mathematica symbols={sorted(mma_symbols)}, Python symbols={sorted(py_symbols)}"))

    if check_symbolic_residual:
        symbolic_passed, detail = _symbolic_residual_zero(mma_expr, py_expr)
        results.append(SymbolicComparisonResult(expression_id, "symbolic_residual", symbolic_passed, detail))

    substitutions = mma_record.get("numeric_substitutions") or py_record.get("numeric_substitutions") or []
    if not substitutions:
        substitutions = [_default_substitution(sorted(mma_symbols | py_symbols))]
    for idx, substitution in enumerate(substitutions):
        results.append(
            _compare_numeric_substitution(
                expression_id,
                f"numeric_substitution[{idx}]",
                mma_expr,
                py_expr,
                substitution,
                local_symbols,
                atol=atol,
                rtol=rtol,
            )
        )
    return results


def _compare_singular_diagnostics(
    expression_id: str,
    mma_record: dict[str, Any],
    py_record: dict[str, Any],
    local_symbols: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[SymbolicComparisonResult]:
    mma_diag = dict(mma_record.get("singular_diagnostics", {}))
    py_diag = dict(py_record.get("singular_diagnostics", {}))
    results = []
    key_match = set(mma_diag) == set(py_diag)
    results.append(
        SymbolicComparisonResult(
            expression_id,
            "singular_diagnostics.keys",
            key_match,
            "diagnostic keys match" if key_match else f"Mathematica keys={sorted(mma_diag)}, Python keys={sorted(py_diag)}",
        )
    )
    if not key_match:
        return results

    expression_format = str(mma_diag.get("diagnostic_expression_format", mma_record.get("expression_format", "sympy")))
    for key in sorted(mma_diag):
        if key in {"diagnostic_expression_format"}:
            continue
        mma_value = mma_diag[key]
        py_value = py_diag[key]
        if key in {"denominator_factor", "limit_expression", "residual_expression"}:
            mma_expr = _parse_expression(mma_value, local_symbols, expression_format=expression_format)
            py_expr = _parse_expression(py_value, local_symbols, expression_format=str(py_diag.get("diagnostic_expression_format", py_record.get("expression_format", "sympy"))))
            symbolic_passed, detail = _symbolic_residual_zero(mma_expr, py_expr)
            results.append(SymbolicComparisonResult(expression_id, f"singular_diagnostics.{key}", symbolic_passed, detail))
        elif isinstance(mma_value, (int, float)) or isinstance(py_value, (int, float)):
            results.append(_compare_scalar_diagnostic(expression_id, key, mma_value, py_value, atol=atol, rtol=rtol))
        else:
            passed = mma_value == py_value
            results.append(
                SymbolicComparisonResult(
                    expression_id,
                    f"singular_diagnostics.{key}",
                    passed,
                    "diagnostic value match" if passed else f"{mma_value!r} != {py_value!r}",
                )
            )
    return results


def _compare_scalar_diagnostic(
    expression_id: str,
    key: str,
    mathematica_value: Any,
    python_value: Any,
    *,
    atol: float,
    rtol: float,
) -> SymbolicComparisonResult:
    mma = complex(float(mathematica_value))
    py = complex(float(python_value))
    delta = abs(py - mma)
    scale = max(abs(mma), np.finfo(float).tiny)
    rel = delta / scale
    passed = bool(np.allclose(py, mma, atol=atol, rtol=rtol))
    return SymbolicComparisonResult(
        expression_id,
        f"singular_diagnostics.{key}",
        passed,
        "numeric diagnostic match" if passed else "numeric diagnostic mismatch",
        float(delta),
        float(rel),
    )


def _parse_expression(value: Any, local_symbols: dict[str, Any], *, expression_format: str = "sympy") -> Any:
    sp = _sympy()
    if isinstance(value, str):
        if expression_format in {"sympy", "sympy_string"}:
            return sp.sympify(value, locals=local_symbols)
        if expression_format in {"mathematica_input_form", "mathematica"}:
            return _parse_mathematica_expression(value, local_symbols)
        raise ValueError(f"Unsupported expression_format: {expression_format!r}")
    if isinstance(value, list):
        return sp.Matrix([_parse_expression(item, local_symbols, expression_format=expression_format) for item in value])
    raise ValueError(f"Unsupported symbolic expression value: {value!r}")


def _parse_mathematica_expression(value: str, local_symbols: dict[str, Any]) -> Any:
    sp = _sympy()
    try:
        from sympy.parsing.mathematica import parse_mathematica
    except ImportError as exc:
        raise ImportError("Mathematica InputForm parsing requires sympy.parsing.mathematica.") from exc

    parsed = parse_mathematica(value)
    replacements = {symbol: local_symbols[str(symbol)] for symbol in parsed.free_symbols if str(symbol) in local_symbols}
    return sp.sympify(parsed.xreplace(replacements), locals=local_symbols)


def _shape(expr: Any) -> tuple[int, ...]:
    if hasattr(expr, "shape"):
        return tuple(int(item) for item in expr.shape)
    return ()


def _free_symbol_names(expr: Any) -> set[str]:
    if hasattr(expr, "free_symbols"):
        return {str(symbol) for symbol in expr.free_symbols}
    return set()


def _symbolic_residual_zero(mma_expr: Any, py_expr: Any) -> tuple[bool, str]:
    sp = _sympy()
    residual = mma_expr - py_expr
    if hasattr(residual, "applyfunc"):
        simplified = residual.applyfunc(sp.simplify)
        passed = all(item == 0 for item in simplified)
    else:
        simplified = sp.simplify(residual)
        passed = simplified == 0
    return bool(passed), "simplified residual is zero" if passed else f"nonzero residual: {simplified}"


def _compare_numeric_substitution(
    expression_id: str,
    check: str,
    mma_expr: Any,
    py_expr: Any,
    substitution: dict[str, Any],
    local_symbols: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> SymbolicComparisonResult:
    subs = {local_symbols[name]: _complex_value(value) for name, value in substitution.items() if name in local_symbols}
    mma = np.asarray(_eval_expr(mma_expr, subs), dtype=complex)
    py = np.asarray(_eval_expr(py_expr, subs), dtype=complex)
    if mma.shape != py.shape:
        return SymbolicComparisonResult(expression_id, check, False, f"shape mismatch after substitution: {mma.shape} vs {py.shape}", float("inf"), float("inf"))
    delta = np.abs(mma - py)
    scale = np.maximum(np.abs(mma), np.finfo(float).tiny)
    max_abs = float(np.max(delta)) if delta.size else float(abs(complex(mma - py)))
    max_rel = float(np.max(delta / scale)) if delta.size else max_abs / max(float(abs(complex(mma))), np.finfo(float).tiny)
    passed = bool(np.allclose(mma, py, atol=atol, rtol=rtol))
    return SymbolicComparisonResult(expression_id, check, passed, "numeric values match" if passed else "numeric values mismatch", max_abs, max_rel)


def _eval_expr(expr: Any, substitution: dict[Any, complex]) -> Any:
    if hasattr(expr, "shape"):
        return [[complex(item.subs(substitution).evalf()) for item in expr]]
    return complex(expr.subs(substitution).evalf())


def _default_substitution(symbol_names: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for idx, name in enumerate(symbol_names, start=1):
        out[name] = {"real": 0.37 * idx + 0.11, "imag": -0.19 * idx}
    return out


def _complex_value(value: Any) -> complex:
    if isinstance(value, dict):
        if {"real", "imag"} <= set(value):
            return complex(float(value["real"]), float(value["imag"]))
        if {"re", "im"} <= set(value):
            return complex(float(value["re"]), float(value["im"]))
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, complex):
        return value
    raise ValueError(f"Unsupported substitution value: {value!r}")


def _sympy():
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError("Symbolic reference comparison requires SymPy.") from exc
    return sp


if __name__ == "__main__":
    main()
