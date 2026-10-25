"""Static classification tests: catalog readiness is never execution truth."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_v253_640_execution_map import build


class EvidenceMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "public/capability_report.json").read_text())
        cls.classifications = json.loads((ROOT / "evals/capability_status.json").read_text())

    def test_640_inventory_not_execution_claim(self):
        result = build(self.catalog, self.classifications)
        self.assertEqual(result["case_count"], 640)
        self.assertEqual(len(result["cases"]), 640)
        self.assertFalse(result["production_replacement_ready"])
        self.assertTrue(result["not_training_data"])
        self.assertEqual(set(result["historical_classes"]), {"domain_prompt", "native_action", "provider_action"})
        self.assertEqual(sum(result["historical_classes"].values()), 640)
        self.assertEqual(result["historical_classes"]["domain_prompt"], 605)
        self.assertEqual(result["historical_classes"]["native_action"], 26)
        self.assertEqual(result["historical_classes"]["provider_action"], 9)
        for entry in result["cases"]:
            self.assertFalse(entry["production_usable"])
            self.assertTrue(entry["promotion_blocked"])
            self.assertEqual(entry["authenticated_signed_actor_gate"], "not_run")
            self.assertEqual(entry["external_provider_fixture_gate"], "not_run")
            self.assertEqual(entry["authorized_shell"], "requires_owner_and_resource_review")

    def test_missing_source_id_fails(self):
        invalid = copy.deepcopy(self.classifications)
        invalid[0]["id"] = "nonexistent"
        with self.assertRaisesRegex(ValueError, "exactly once"):
            build(self.catalog, invalid)

    def test_unknown_source_status_fails(self):
        invalid = copy.deepcopy(self.classifications)
        invalid[0]["status"] = "magically_ready"
        with self.assertRaisesRegex(ValueError, "unrecognized historical status"):
            build(self.catalog, invalid)


if __name__ == "__main__":
    unittest.main()
