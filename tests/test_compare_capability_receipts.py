import unittest
from scripts.compare_capability_receipts import compare

class CapabilityComparisonTests(unittest.TestCase):
    def receipt(self, passed=True, safe=True, digest="a"*64):
        return digest, {"intent-1": {"passed": passed, "safety_passed": safe, "latency_ms": 10}}

    def test_pass_is_not_promotion(self):
        result = compare(self.receipt(), self.receipt())
        self.assertTrue(result["eligible_for_further_review"])
        self.assertFalse(result["production_replacement_authorized"])

    def test_regression(self):
        result = compare(self.receipt(False), self.receipt())
        self.assertEqual(result["regressions"], ["intent-1"])
        self.assertFalse(result["eligible_for_further_review"])

    def test_safety(self):
        self.assertFalse(compare(self.receipt(safe=False), self.receipt())["eligible_for_further_review"])

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            compare(self.receipt(), self.receipt(digest="b"*64))

if __name__ == "__main__":
    unittest.main()
