from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.compare_mathematica_reference import numeric_array  # noqa: E402

MMA_PATH = ROOT / "benchmarks" / "mathematica_reference" / "maker_fringes_reference_v1.json"
PY_PATH = ROOT / "benchmarks" / "maker_fringes_benchmarks_v1.json"


def main() -> None:
    mathematica = json.loads(MMA_PATH.read_text(encoding="utf-8"))
    python = json.loads(PY_PATH.read_text(encoding="utf-8"))
    mma_cases = {case["id"]: case for case in mathematica["suites"][0]["items"]}
    py_cases = {case["id"]: case for case in python["cases"]}

    for case_id in mma_cases:
        mma_outputs = mma_cases[case_id]["mathematica_outputs"]
        py_outputs = py_cases[case_id]["outputs"]
        mma_para = numeric_array(mma_outputs["listMFpara"])[:, 1]
        py_para = numeric_array(py_outputs["list_mf_para"])[:, 1]
        mma_perp = numeric_array(mma_outputs["listMFperp"])[:, 1]
        py_perp = numeric_array(py_outputs["list_mf_perp"])[:, 1]
        mma_parallel_amp = numeric_array(mma_outputs["parallel_analyzer_amplitude"])
        py_parallel_amp = numeric_array(py_outputs["parallel_amplitude"])
        mma_perp_amp = numeric_array(mma_outputs["perpendicular_analyzer_amplitude"])
        py_perp_amp = numeric_array(py_outputs["perpendicular_amplitude"])
        mma_jones = numeric_array(mma_outputs["transmitted_two_omega_jones_sp"])
        py_jones = numeric_array(py_outputs["transmitted_2omega_jones_sp"])
        print(f"CASE {case_id}")
        _print_series("para", mma_para, py_para)
        _print_series("perp", mma_perp, py_perp)
        _print_series("parallel amplitude", mma_parallel_amp, py_parallel_amp)
        _print_series("perpendicular amplitude", mma_perp_amp, py_perp_amp)
        print(f"  jones max abs delta: {np.max(np.abs(py_jones - mma_jones)):.15g}")


def _print_series(label: str, mathematica: np.ndarray, python: np.ndarray) -> None:
    ratio = np.divide(python, mathematica, out=np.full_like(python, np.nan), where=np.abs(mathematica) > 0)
    print(f"  {label} Mathematica: {np.array2string(mathematica, precision=15)}")
    print(f"  {label} Python:      {np.array2string(python, precision=15)}")
    print(f"  {label} py/mma:      {np.array2string(ratio, precision=15)}")
    print(f"  {label} abs delta:   {np.array2string(np.abs(python - mathematica), precision=15)}")


if __name__ == "__main__":
    main()
