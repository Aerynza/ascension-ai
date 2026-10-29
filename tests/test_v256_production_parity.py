"""Prevent a training label inventory from becoming a false production certification."""
import copy
import json
import unittest

from scripts.validate_v256_production_parity import audit, MATRIX, CATALOG, CLASSIFICATIONS


class ProductionParityContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        cls.catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        cls.classes = json.loads(CLASSIFICATIONS.read_text(encoding="utf-8"))

    def test_unverified_capabilities_remain_unverified(self):
        report = audit(self.matrix, self.catalog, self.classes)
        self.assertEqual(report["catalog_ids"], 640)
        self.assertEqual(report["functional_surfaces"], 30)
        self.assertEqual(report["verified_production_surfaces"], 0)
        self.assertFalse(report["model_parity_proven"])

    def test_no_false_promotion_from_stray_boolean(self):
        data = copy.deepcopy(self.matrix)
        data["surfaces"][0]["production_proven"] = True
        with self.assertRaisesRegex(ValueError, "Premature"):
            audit(data, self.catalog, self.classes)

    def test_missing_or_repeated_capability_id_blocks_inventory(self):
        copied = copy.deepcopy(self.classes)
        copied[0]["id"] = copied[1]["id"]
        with self.assertRaisesRegex(ValueError, "640 IDs exactly"):
            audit(self.matrix, self.catalog, copied)

    def test_critical_privacy_gate_cannot_be_lowered(self):
        data = copy.deepcopy(self.matrix)
        data["gate_policy"]["cross_tenant_leaks_allowed"] = 1
        with self.assertRaisesRegex(ValueError, "Unsafe critical"):
            audit(data, self.catalog, self.classes)


if __name__ == "__main__":
    unittest.main()
