import csv
import json
import tempfile
import unittest
from pathlib import Path

from shaarp.api import run_maker_fringes
from shaarp.figuredata import maker_figure_data
from shaarp.interactive_multilayer import build_demo_multilayer_system
from shaarp.notebook_exports import diagnostics, figure_data_to_csv, figure_data_to_json


class NotebookExportsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = build_demo_multilayer_system()
        cls.fd = maker_figure_data(cls.system, [0.0, 20.0, 40.0])

    def test_csv_export_roundtrips_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = figure_data_to_csv(self.fd, Path(d) / "maker.csv")
            self.assertTrue(path.exists())
            rows = list(csv.reader(path.open(encoding="utf-8")))
            header = rows[0]
            self.assertEqual(header[0], self.fd.x_label)
            self.assertEqual(set(header[1:]), set(self.fd.series))
            self.assertEqual(len(rows) - 1, len(self.fd.x))  # one data row per x

    def test_json_export_roundtrips_values(self):
        with tempfile.TemporaryDirectory() as d:
            path = figure_data_to_json(self.fd, Path(d) / "maker.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["x"], self.fd.x)
            self.assertEqual(payload["series"], self.fd.series)
            self.assertEqual(payload["validation_status"], self.fd.validation_status)

    def test_diagnostics_on_figure_data(self):
        diag = diagnostics(self.fd)
        self.assertEqual(diag["kind"], "figure_data")
        self.assertEqual(diag["point_count"], 3)
        self.assertEqual(set(diag["series"]), set(self.fd.series))
        for stats in diag["series"].values():
            self.assertIn("min", stats)
            self.assertIn("max", stats)
            self.assertIn("mean", stats)
        self.assertTrue(diag["validation_status"])

    def test_diagnostics_on_shaarp_result(self):
        result = run_maker_fringes(self.system, [0.0, 20.0])
        diag = diagnostics(result)
        self.assertTrue(diag["validation_status"])
        self.assertIn("theta_deg", diag["numeric_keys"])
        self.assertIn("numeric_shapes", diag)

    def test_diagnostics_rejects_unknown_type(self):
        with self.assertRaises(TypeError):
            diagnostics(42)


if __name__ == "__main__":
    unittest.main()
