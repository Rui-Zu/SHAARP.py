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
