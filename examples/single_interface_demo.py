"""Smallest possible reflected-SHG call, using the REDUCED convenience model.

``single_interface_intensity`` is the one-line convenience API: it collapses the anisotropic
dielectric tensors to one effective refractive index, so it prints a RuntimeWarning saying it is
not the full anisotropic eigenmode and boundary-condition solver. That warning is expected here --
this file exists to show the shortest path from a preset to a plot.

For numbers you intend to publish, use the validated solver instead:
``run_si_numeric(sample, {"workflow": "shaarp_si_compat", ...})``, as in
``gaas111_shg_polarimetry.py``. See docs/usage.md for the difference.
"""
import numpy as np
import matplotlib.pyplot as plt

from shaarp import Polarimetry, presets, single_interface_intensity


sample = presets.linbo3_1120_xcut()
polarimetry = Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 360, 361), psi_deg=0)
result = single_interface_intensity(sample, polarimetry)

result.plot()
plt.show()
