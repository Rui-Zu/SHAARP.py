from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a copied Mathematica/Wolfram numeric list into a reference JSON case output."
    )
    parser.add_argument("--input", type=Path, required=True, help="Text file containing a copied Wolfram list.")
    parser.add_argument("--output", type=Path, required=True, help="Reference JSON file to create or update.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--key", required=True, help="Output key, e.g. single_interface_intensity.")
    parser.add_argument(
        "--column",
        type=int,
        default=None,
        help="For list-of-pairs data, extract this zero-based column before saving.",
    )
    args = parser.parse_args()

    values = parse_wolfram_real_list(args.input.read_text(encoding="utf-8"))
    if args.column is not None:
        values = extract_column(values, args.column)

    payload = load_or_create_payload(args.output)
    upsert_case_output(payload, args.case_id, args.key, values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Updated {args.output} with {args.case_id} outputs.{args.key}")


def parse_wolfram_real_list(text: str) -> Any:
    """Parse real-valued Wolfram list syntax into Python lists/numbers.

    Supported examples:

    - `{1, 2, 3}`
    - `{{0., 1.2*^-3}, {10., 4.5}}`
    - numbers with Mathematica precision marks such as `1.23`20.`
    """

    normalized = text.strip()
    normalized = re.sub(r"\(\*.*?\*\)", "", normalized, flags=re.DOTALL)
    normalized = re.sub(r"`(?:\d+(?:\.\d*)?)?", "", normalized)
    normalized = normalized.replace("*^", "e")
    normalized = normalized.replace("{", "[").replace("}", "]")
    if re.search(r"(?<![A-Za-z])I(?![A-Za-z])", normalized):
        raise ValueError("Complex Wolfram lists are not supported by this real-list converter.")
    try:
        parsed = ast.literal_eval(normalized)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse Wolfram list: {exc}") from exc
    return coerce_real_tree(parsed)


def coerce_real_tree(value: Any) -> Any:
    if isinstance(value, list):
        return [coerce_real_tree(item) for item in value]
    if isinstance(value, tuple):
        return [coerce_real_tree(item) for item in value]
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Unsupported non-real value in Wolfram list: {value!r}")


def extract_column(values: Any, column: int) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("Column extraction requires a list.")
    out = []
    for row in values:
        if not isinstance(row, list) or column >= len(row):
            raise ValueError(f"Cannot extract column {column} from row {row!r}.")
        out.append(float(row[column]))
    return out


def load_or_create_payload(path: Path) -> dict:
    if not path.exists():
        return {
            "source": "Mathematica SHAARP copied output",
            "status": "manual_wolfram_copy_reference",
            "cases": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_case_output(payload: dict, case_id: str, key: str, values: Any) -> None:
    cases = payload.setdefault("cases", [])
    for case in cases:
        if case.get("id") == case_id:
            case.setdefault("outputs", {})[key] = values
            return
    cases.append({"id": case_id, "outputs": {key: values}})


if __name__ == "__main__":
    main()
