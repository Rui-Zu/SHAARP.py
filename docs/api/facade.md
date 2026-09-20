# Facade & high-level functions

The entry points the desktop GUI calls, plus the high-level convenience functions and the
$d$-tensor extraction. Each returns either a plottable result object or a
{py:class}`~shaarp.SHAARPResult`.

## What the app's Update button runs

The two functions behind the desktop app's **Update** button. They take the same arguments the
GUI controls take (functionality name, preset or case-study name, angles) and return the same
{py:class}`~shaarp.SHAARPResult`, so anything you can click you can script.

```{eval-rst}
.. currentmodule:: shaarp

.. autofunction:: compute_si_gui_result
.. autofunction:: compute_ml_gui_result
```

## GUI-mirroring facades

```{eval-rst}
.. currentmodule:: shaarp

.. autofunction:: run_si_numeric
.. autofunction:: run_si_full_analytical
.. autofunction:: run_ml_numeric
.. autofunction:: run_maker_fringes
.. autofunction:: run_fresnel_sweep
.. autofunction:: run_ml_partial_analytical
.. autofunction:: run_sample_rotation
```

## Wavelength sweeps

Each takes a factory, a function that returns the material or the system at a given fundamental
wavelength, because the permittivities have to be rebuilt at every point. Worked examples, and the
materials that carry published index data, are in [Sweeping the wavelength](../usage.md#sweeping-the-wavelength).

```{eval-rst}
.. currentmodule:: shaarp

.. autofunction:: run_si_spectrum
.. autofunction:: run_ml_spectrum
.. autofunction:: run_spectral_map
```

## High-level convenience

```{eval-rst}
.. currentmodule:: shaarp

.. autofunction:: single_interface_intensity
.. autofunction:: multilayer_shg
.. autofunction:: export_result
.. autofunction:: analytical_expression_text
```

## d-tensor extraction

```{eval-rst}
.. currentmodule:: shaarp

.. autofunction:: extract_si_d_voigt
.. autofunction:: extract_ml_film_d_voigt
```
