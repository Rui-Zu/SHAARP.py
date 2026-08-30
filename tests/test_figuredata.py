import math
import unittest

from shaarp.figuredata import (
    FigureData,
    fresnel_figure_data,
    maker_figure_data,
    sample_rotation_figure_data,
)
from shaarp.interactive_multilayer import build_demo_multilayer_system


class FigureDataTests(unittest.TestCase):
    """The headless figure-data helpers return plot-ready arrays from the
    validated facades, with no plotting library required at import."""

    @classmethod
    def setUpClass(cls):
        cls.system = build_demo_multilayer_system()

    def _check(self, fd: FigureData, n_x: int):
        self.assertIsInstance(fd, FigureData)
        self.assertTrue(fd.title and fd.x_label and fd.y_label)
        self.assertEqual(len(fd.x), n_x)
        self.assertTrue(fd.series, "expected at least one y-series")
        for name, ys in fd.series.items():
            self.assertEqual(len(ys), n_x, f"series {name} length mismatch")
            self.assertTrue(all(math.isfinite(v) for v in ys), f"series {name} has non-finite values")
        self.assertIsInstance(fd.validation_status, str)
        self.assertTrue(fd.validation_status)

    def test_maker_figure_data(self):
        fd = maker_figure_data(self.system, [0.0, 20.0, 40.0])
        self._check(fd, 3)
        self.assertIn("parallel analyzer", fd.series)
        self.assertIn("perpendicular analyzer", fd.series)

    def test_sample_rotation_figure_data(self):
        fd = sample_rotation_figure_data(self.system, [0.0, 30.0, 60.0])
        self._check(fd, 3)
        self.assertIn("reflected parallel", fd.series)
        self.assertIn("transmitted perpendicular", fd.series)

    def test_fresnel_figure_data(self):
        fd = fresnel_figure_data(self.system, [0.0, 20.0, 40.0])
        self._check(fd, 3)
        self.assertEqual(set(fd.series), {"|rp|", "|rs|", "|tp|", "|ts|"})
        # Fresnel magnitudes are non-negative.
        for ys in fd.series.values():
            self.assertTrue(all(v >= 0 for v in ys))

    def test_stack_schematic_data(self):
        from shaarp.figuredata import StackSchematic, stack_schematic_data

        sch = stack_schematic_data(self.system)
        self.assertIsInstance(sch, StackSchematic)
        n = len(self.system.layers)
        for arr in (sch.layer_index, sch.names, sch.materials, sch.thickness_um,
                    sch.shg_active, sch.is_semi_infinite, sch.cumulative_depth_um):
            self.assertEqual(len(arr), n)
        # all display thicknesses positive (semi-infinite media get nominal thickness)
        self.assertTrue(all(t > 0 for t in sch.thickness_um))
        # cumulative depth is monotincreasing
        self.assertTrue(all(b >= a for a, b in zip(sch.cumulative_depth_um, sch.cumulative_depth_um[1:])))
        # at least one active layer (the film)
        self.assertTrue(any(sch.shg_active))
        # top + substrate are semi-infinite in the demo stack
        self.assertTrue(sch.is_semi_infinite[0] and sch.is_semi_infinite[-1])

    def test_module_imports_without_matplotlib(self):
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "shaarp" / "figuredata.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        top_imports = []
        for node in tree.body:  # only module-level imports
            if isinstance(node, ast.Import):
                top_imports += [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                top_imports.append(node.module or "")
        self.assertFalse(any("matplotlib" in m for m in top_imports),
                         "matplotlib must not be imported at module load")


class StackOrientationSchematicTests(unittest.TestCase):
    """Headless 3D crystal-orientation schematic data (gui_2d_3d_schematics)."""

    def test_orientation_data_geometry_and_optics(self):
        import numpy as np
        from shaarp import layer_presets as lp
        from shaarp.figuredata import stack_orientation_schematic_data

        system = lp.build_stack_preset("linbo3_on_gaas")
        schematic = stack_orientation_schematic_data(system)
        self.assertEqual(len(schematic.layers), len(system.layers))
        for layer in schematic.layers:
            axes = np.array(layer.crystal_axes_lab)
            self.assertEqual(axes.shape, (3, 3))
            self.assertTrue(np.allclose(axes @ axes.T, np.eye(3), atol=1e-6))
            self.assertAlmostEqual(abs(float(np.linalg.det(axes))), 1.0, places=6)
            self.assertIn(layer.optical_class, ("isotropic", "uniaxial", "biaxial"))
            if layer.optical_class == "uniaxial":
                self.assertIsNotNone(layer.optic_axis_lab)
                self.assertAlmostEqual(float(np.linalg.norm(layer.optic_axis_lab)), 1.0, places=6)
            else:
                self.assertIsNone(layer.optic_axis_lab)

    def test_linbo3_zcut_is_uniaxial_with_c_axis_optic_axis(self):
        import numpy as np
        from shaarp import layer_presets as lp
        from shaarp.figuredata import stack_orientation_schematic_data

        system = lp.build_stack_preset("single_film_linbo3")
        film = next(l for l in stack_orientation_schematic_data(system).layers if l.shg_active)
        self.assertEqual(film.optical_class, "uniaxial")
        self.assertTrue(np.allclose(np.abs(film.optic_axis_lab), [0.0, 0.0, 1.0], atol=1e-6))


class FresnelPlotMathematicaValidationTests(unittest.TestCase):
    """The Fresnel-coefficient PLOT (fresnel_figure_data) draws, for a
    MultilayerSystem, the SHAARP.ml multilayer listFresnel coefficients -- and
    those plotted magnitudes match Mathematica's listFresnel values to ~1e-10
    (gui_fresnel_coefficient_plot). This validates the plot itself, not just the
    lower-level facade."""

    def test_plot_series_match_mathematica_listfresnel(self):
        import json
        from pathlib import Path

        import numpy as np

        from benchmarks.compare_mathematica_reference import numeric_array
        from benchmarks.generate_maker_fringes_benchmarks import build_cases

        ref_path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "mathematica_reference"
            / "fresnel_sweep_reference_v1.json"
        )
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        reference_case = ref["suites"][0]["items"][0]
        system = {case["id"]: case["system"] for case in build_cases()}[reference_case["id"]]

        fd = fresnel_figure_data(system, reference_case["inputs"]["theta_deg"])

        curves = reference_case["mathematica_outputs"]["listFresnel"]
        for series_key, curve in zip(("|rp|", "|rs|", "|tp|", "|ts|"), curves):
            expected = np.abs(numeric_array(curve)[:, 1])
            got = np.asarray(fd.series[series_key], dtype=float)
            np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)

        self.assertEqual(
            fd.validation_status,
            "gui_fresnel_list_selected_policy_mathematica_validated_with_transmitted_sum_caveat",
        )

    def test_bare_material_uses_single_interface_fallback(self):
        from shaarp import presets

        fd = fresnel_figure_data(presets.linbo3_1120_xcut(), [0.0, 30.0, 60.0])
        # bare Material -> single-interface workflow, not the multilayer listFresnel path
        self.assertEqual(set(fd.series), {"|rp|", "|rs|", "|tp|", "|ts|"})
        self.assertNotIn("listFresnel", fd.validation_status)


class MakerPlotMathematicaValidationTests(unittest.TestCase):
    """The Maker-fringe PLOT (maker_figure_data) draws listMFpara/listMFperp under
    the SHAARP.ml unit normalization -- the plotted intensities match Mathematica's
    MFList to ~1e-10 on non-singular cases (gui_maker_fringes_plot). This validates
    the plot itself; the phase-matching-like case stays a documented diagnostic."""

    def test_plot_series_match_mathematica_mflist_nonsingular(self):
        import json
        from pathlib import Path

        import numpy as np

        from benchmarks.compare_mathematica_reference import numeric_array
        from benchmarks.compare_maker_fringes_reference import DIAGNOSTIC_CASE_IDS
        from benchmarks.generate_maker_fringes_benchmarks import build_cases

        ref_path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "mathematica_reference"
            / "maker_fringes_reference_v1.json"
        )
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        systems = {case["id"]: case["system"] for case in build_cases()}

        checked = 0
        for item in ref["suites"][0]["items"]:
            cid = item["id"]
            if cid in DIAGNOSTIC_CASE_IDS or cid not in systems:
                continue
            mfl = numeric_array(item["mathematica_outputs"]["MFList"])
            theta = list(mfl[:, 0].real)
            fd = maker_figure_data(systems[cid], theta)
            np.testing.assert_allclose(
                np.asarray(fd.series["parallel analyzer"], dtype=float), mfl[:, 1].real, atol=1e-10, rtol=1e-10
            )
            np.testing.assert_allclose(
                np.asarray(fd.series["perpendicular analyzer"], dtype=float), mfl[:, 2].real, atol=1e-10, rtol=1e-10
            )
            checked += 1
        self.assertGreaterEqual(checked, 2)

    def test_plot_status_is_mathematica_validated(self):
        from benchmarks.generate_maker_fringes_benchmarks import build_cases

        system = build_cases()[0]["system"]
        fd = maker_figure_data(system, [0.0, 20.0, 40.0])
        self.assertIn("mathematica_validated", fd.validation_status)


if __name__ == "__main__":
    unittest.main()
