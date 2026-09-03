# Examples

Six self-contained scripts in the repository's `examples/` folder. Run any of them with:

```bash
python examples/<name>.py
```

## Which one do I want?

```{list-table}
:header-rows: 1
:widths: 26 40 18 16

* - Script
  - What it shows
  - You get
  - Runtime
* - `d_extraction_demo.py`
  - **The package's purpose:** simulate a polarimetry scan from a known $d$ tensor, then recover
    that tensor from it — agreement to ~1e-9.
  - two-panel figure: fit vs data, recovered vs true $d$
  - ~10 s
* - `gaas111_shg_polarimetry.py`
  - The validated **closed form** for GaAs (111), plus its three-fold $C_{3v}$ signature (120°
    azimuth leaves the pattern unchanged, 60° does not).
  - polar plots from the analytical expression
  - ~6 s
* - `maker_mathematica_overlay.py`
  - **Visual proof of the Mathematica agreement:** SHAARP.py Maker intensities overlaid on live
    ♯SHAARP.ml points at 0.1° sampling, with a residual panel (~1e-15).
  - overlay + residual figure
  - ~8 s
* - `maker_fringes_dense.py`
  - Maker fringes across all three multiple-reflection assumptions (FMR / HH / JK), at the
    angular sampling rigor of the 2024 paper.
  - two-panel fringe comparison
  - ~80 s
* - `single_interface_demo.py`
  - The shortest path from a preset to a plot — deliberately uses the **reduced** convenience
    model, and prints the warning that says so.
  - one reflected-SHG polar plot
  - ~4 s
* - `multilayer_demo.py`
  - The same, for a layer stack: shortest path to a multilayer result, also the **reduced** model.
  - one multilayer polar plot
  - ~3 s
```

```{tip}
Starting your own script? Copy `d_extraction_demo.py` or `gaas111_shg_polarimetry.py` — both use
validated solvers. The two `*_demo.py` files use the reduced convenience model, which is fine for a
sketch but not for published numbers ({doc}`../usage`).
```

## Single-interface reflected SHG

```{literalinclude} ../../examples/single_interface_demo.py
:language: python
:caption: examples/single_interface_demo.py
```

## Multilayer thin film

```{literalinclude} ../../examples/multilayer_demo.py
:language: python
:caption: examples/multilayer_demo.py
```

## GaAs(111) polarimetry

```{literalinclude} ../../examples/gaas111_shg_polarimetry.py
:language: python
:caption: examples/gaas111_shg_polarimetry.py
```

## Dense Maker fringes

```{literalinclude} ../../examples/maker_fringes_dense.py
:language: python
:caption: examples/maker_fringes_dense.py
```

## d-tensor extraction

```{literalinclude} ../../examples/d_extraction_demo.py
:language: python
:caption: examples/d_extraction_demo.py
```

## Maker fringes overlaid on Mathematica reference

```{literalinclude} ../../examples/maker_mathematica_overlay.py
:language: python
:caption: examples/maker_mathematica_overlay.py
```
