# Result objects

What the solvers return. {py:class}`~shaarp.SHAARPResult` is the general payload from the `run_*`
facades (numeric data + stages + validation metadata); the dedicated result classes carry richer,
typed fields for specific solvers.

```{eval-rst}
.. currentmodule:: shaarp

.. autoclass:: SHAARPResult
   :members:

.. autoclass:: SingleInterfaceSHGResult
   :members:

.. autoclass:: MultilayerSHGBoundaryResult
   :members:

.. autoclass:: MultilayerMakerFringesSweepResult
   :members:

.. autoclass:: FresnelCoefficientSweepResult
   :members:

.. autoclass:: DExtractionResult
   :members:

.. autoclass:: ValidationStatus
   :members:

.. autoclass:: PhysicsConventions
   :members:
```
