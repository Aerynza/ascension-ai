import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.plan_native_research import plan

class ResearchPlannerTests(unittest.TestCase):
    def test_budget_and_identity(self):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for name in ("checkpoint", "dataset", "config"):
                p = Path(root) / name
                p.write_bytes(name.encode())
                paths.append(p)
            result = plan(*paths, 30, 3)
            self.assertEqual(result["identities"]["checkpoint"], hashlib.sha256(b"checkpoint").hexdigest())
            self.assertEqual(len(result["trials"]), 3)
            self.assertLessEqual(sum(t["max_minutes"] for t in result["trials"]), 30)
            self.assertEqual(result["trials"][0]["kind"], "baseline")
            self.assertFalse(result["paid_compute_authorized"])
            self.assertFalse(result["production_promotion_authorized"])
            for minutes, trials in ((0, 1), (121, 1), (5, 6), (30, 13), (30, 0)):
                with self.assertRaises(ValueError):
                    plan(*paths, minutes, trials)

if __name__ == "__main__":
    unittest.main()
