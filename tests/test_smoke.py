import numpy as np
import tempfile
import unittest

from shaarp import Layer, MultilayerSystem, Polarimetry, multilayer_shg, presets
from shaarp.single_interface import single_interface_intensity


class SmokeTests(unittest.TestCase):
    def test_single_interface_shape(self):
        pol = Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 360, 19), psi_deg=0)
        result = single_interface_intensity(presets.linbo3_1120_xcut(), pol)
        self.assertEqual(result.intensity.shape, (19,))
        self.assertTrue(np.all(np.isfinite(result.intensity)))
        self.assertEqual(result.model, "reduced_effective_index")


    def test_multilayer_shape(self):
        system = MultilayerSystem(
            wavelength_um=1.064,
            polarimetry=Polarimetry(theta_deg=30, phi_deg=np.linspace(0, 360, 13), psi_deg=0),
            layers=[
                Layer("air", presets.air(), shg_active=False),
                Layer("LNO", presets.linbo3_zcut_1064(), thickness_um=1.0),
                Layer("air", presets.air(), shg_active=False),
            ],
        )
        result = multilayer_shg(system)
        self.assertEqual(result.reflected_intensity.shape, (13,))
        self.assertTrue(np.all(np.isfinite(result.reflected_intensity)))
        self.assertEqual(result.model, "reduced_effective_index_layer_sum")


    def test_result_csv_exports(self):
        pol = Polarimetry(theta_deg=45, phi_deg=np.linspace(0, 180, 3), psi_deg=0)
        with tempfile.TemporaryDirectory() as tmp:
            from pathlib import Path

            tmp_path = Path(tmp)
            si = single_interface_intensity(presets.blank_nonlinear(), pol)
            si_path = tmp_path / "si.csv"
            si.save_csv(si_path)
            self.assertTrue(si_path.exists())

            system = MultilayerSystem(
                wavelength_um=1.064,
                polarimetry=pol,
                layers=[
                    Layer("air", presets.air(), shg_active=False),
                    Layer("test", presets.blank_nonlinear(), thickness_um=1.0),
                    Layer("air", presets.air(), shg_active=False),
                ],
            )
            ml = multilayer_shg(system)
            ml_path = tmp_path / "ml.csv"
            ml.save_csv(ml_path)
            self.assertTrue(ml_path.exists())

    def test_multilayer_refuses_unimplemented_multiple_reflection(self):
        system = MultilayerSystem(
            wavelength_um=1.064,
            include_multiple_reflections=True,
            layers=[
                Layer("air", presets.air(), shg_active=False),
                Layer("LNO", presets.linbo3_zcut_1064(), thickness_um=1.0),
                Layer("air", presets.air(), shg_active=False),
            ],
        )
        with self.assertRaises(NotImplementedError):
            multilayer_shg(system)


if __name__ == "__main__":
    unittest.main()
