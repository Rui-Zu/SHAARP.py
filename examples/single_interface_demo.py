import numpy as np
import matplotlib.pyplot as plt

from shaarp import Polarimetry, presets, single_interface_intensity


sample = presets.linbo3_1120_xcut()
polarimetry = Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 360, 361), psi_deg=0)
result = single_interface_intensity(sample, polarimetry)

result.plot()
plt.show()
