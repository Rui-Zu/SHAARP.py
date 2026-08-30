# Facade & high-level functions

The entry points the desktop GUI calls, plus the high-level convenience functions and the
$d$-tensor extraction. Each returns either a plottable result object or a
{py:class}`~shaarp.SHAARPResult`.

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
