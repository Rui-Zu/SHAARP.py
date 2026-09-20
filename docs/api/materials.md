# Materials & presets

Built-in crystal presets, the wavelength-interpolated case-study materials taken from the original
♯SHAARP `setup.nb`, and the index tables and factories a wavelength sweep uses.

## Case-study materials

```{eval-rst}
.. automodule:: shaarp.casestudy_materials
   :members:
   :undoc-members:
   :show-inheritance:
```

## Presets

```{eval-rst}
.. automodule:: shaarp.presets
   :members:
   :undoc-members:
   :show-inheritance:
```

## Index tables and wavelength factories

The five crystals with published index data, index tables of your own, and the factories that the
wavelength sweeps take. See [Sweeping the wavelength](../usage.md#sweeping-the-wavelength).

```{eval-rst}
.. autofunction:: shaarp.dispersion.dispersive_material_names
.. autofunction:: shaarp.dispersion.dispersive_spectrum
.. autofunction:: shaarp.dispersion.load_index_table
.. autofunction:: shaarp.dispersion.table_spectrum
.. autofunction:: shaarp.spectral.casestudy_spectrum
.. autofunction:: shaarp.spectral.casestudy_ml_spectrum
.. autofunction:: shaarp.spectral.constant_spectrum
```
