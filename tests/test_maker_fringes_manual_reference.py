import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.build_maker_fringes_manual_reference import _load_or_create, _make_mf_list, _upsert_case, _validate_pair_list
from benchmarks.compare_maker_fringes_reference import validate_maker_reference_payload


class MakerFringesManualReferenceTests(unittest.TestCase):
    def test_manual_reference_builder_combines_para_perp_into_mflist(self):
        para = [[-8.0, 0.1], [0.0, 0.2]]
        perp = [[-8.0, 0.3], [0.0, 0.4]]

        mf_list = _make_mf_list(para, perp)

        self.assertEqual(mf_list, [[-8.0, 0.1, 0.3], [0.0, 0.2, 0.4]])

    def test_manual_reference_builder_rejects_angle_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Angle mismatch"):
            _make_mf_list([[0.0, 0.1]], [[0.25, 0.2]])

    def test_manual_reference_payload_passes_strict_maker_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manual_maker.json"
            payload = _load_or_create(path, "Mathematica SHAARP.ml GUI copied Maker fringes output")
            _upsert_case(
                payload,
                "maker_fringes_negative_normal_positive",
                {
                    "MFList": [[-8.0, 0.1, 0.2], [0.0, 0.3, 0.4]],
                    "listMFpara": [[-8.0, 0.1], [0.0, 0.3]],
                    "listMFperp": [[-8.0, 0.2], [0.0, 0.4]],
                },
            )

            validate_maker_reference_payload(payload)

    def test_manual_reference_load_rejects_existing_non_mathematica_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text(json.dumps({"source": "Python scratch", "cases": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source"):
                _load_or_create(path, "Mathematica SHAARP.ml GUI copied Maker fringes output")


if __name__ == "__main__":
    unittest.main()
