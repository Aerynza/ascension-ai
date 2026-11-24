import unittest
from scripts.select_native_research_trial import select

class SelectionTests(unittest.TestCase):
    def row(self, name, score, safe=True, passed=True):
        return {"id": name, "suite_sha256": "a"*64, "input_manifest_sha256": "b"*64,
                "cases": {"intent": {"score": score, "passed": passed, "safety_passed": safe}}}

    def test_ranking(self):
        result = select(self.row("base", .5), [self.row("good", .7), self.row("best", .9),
            self.row("unsafe", 1, False), self.row("worse", .4), self.row("same", .5),
            self.row("failed", .9, passed=False)])
        self.assertEqual(result["recommended_trial"], "best")
        self.assertEqual(set(result["rejected"]), {"unsafe", "worse", "same", "failed"})
        self.assertFalse(result["production_promotion_authorized"])

    def test_no_improvement(self):
        self.assertIsNone(select(self.row("base", .5), [self.row("same", .5)])["recommended_trial"])

if __name__ == "__main__":
    unittest.main()
