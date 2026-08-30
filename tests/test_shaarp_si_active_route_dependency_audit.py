import json
import unittest
from pathlib import Path

from benchmarks.generate_shaarp_si_active_route_dependency_audit import build_payload


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "benchmarks" / "mathematica_reference"
AUDIT_PATH = REF_DIR / "shaarp_si_active_route_dependency_audit_v1.json"


class SHAARPSIActiveRouteDependencyAuditTests(unittest.TestCase):
    def test_saved_dependency_audit_matches_generator(self):
        expected = build_payload()
        actual = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(actual, expected)

    def test_dependency_audit_is_not_validation_evidence(self):
        payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "dependency_audit_not_validation_evidence")
        self.assertIn("not a Mathematica evaluation", " ".join(payload["notes"]))

    def test_active_route_tracks_required_user_and_upstream_symbols(self):
        payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        route = payload["active_numeric_route"]
        required = set(route["required_external_symbols"])
        assigned = set(route["assigned_symbols"])

        for symbol in ("a", "ThetaIn", "EIw", "RHTilt", "d11", "d36", "\\[CurlyEpsilon]\\[Omega]Cry"):
            self.assertIn(symbol, required)
        for symbol in ("dold", "dSHG", "P2eo", "P2o", "Peoo", "Croots", "E2wroots", "RSignalCrystal"):
            self.assertIn(symbol, assigned)

    def test_fresnel_sweep_route_keeps_gui_sweep_dependencies_separate(self):
        payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        route = payload["fresnel_sweep_route"]

        self.assertEqual(
            route["region_order"],
            [
                "fresnel_sweep_table_initialization",
                "fresnel_sweep_angle_root_assignments",
                "omega_angle_roots_initial_path",
                "fresnel_sweep_table_rows",
            ],
        )
        self.assertIn("Fresnel", route["assigned_symbols"])
        self.assertIn("Index", route["assigned_symbols"])
        self.assertIn("ThetaTout", route["required_external_symbols"])


if __name__ == "__main__":
    unittest.main()
