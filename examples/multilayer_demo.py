"""Smallest possible multilayer-SHG call, using the REDUCED convenience model.

``multilayer_shg`` is the one-line convenience API: it is a coherent source-sum model, not the
full SHAARP.ml forward/backward anisotropic multilayer solver, and says so in a RuntimeWarning at
runtime. That warning is expected here -- this file shows the shortest path to a layer stack.

For validated Maker fringes and transmitted SHG use ``run_maker_fringes`` /
``compute_ml_gui_result``, as in ``maker_fringes_dense.py``. See docs/usage.md.
"""
import numpy as np
import matplotlib.pyplot as plt

from shaarp import Layer, MultilayerSystem, Polarimetry, multilayer_shg, presets


system = MultilayerSystem(
    wavelength_um=1.064,
    polarimetry=Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 360, 181), psi_deg=0),
    layers=[
        Layer("air in", presets.air(), thickness_um=None, shg_active=False),
        Layer("LNO", presets.linbo3_zcut_1064(), thickness_um=1.0, shg_active=True),
        Layer("air out", presets.air(), thickness_um=None, shg_active=False),
    ],
)

result = multilayer_shg(system)
result.plot()
plt.show()
