import json
import sys
import tempfile
import unittest
from pathlib import Path
from scripts.run_native_research_trial import run

class TrialRunnerTests(unittest.TestCase):
    def test_success_and_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "trial"
            result = run([sys.executable, "-c", "import os; print(os.environ['HF_HUB_OFFLINE'])"], output, 10)
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["production_promotion_authorized"])
            self.assertEqual((output / "trial.log").read_text().strip(), "1")
            self.assertEqual(json.loads((output / "receipt.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                run([sys.executable, "-c", "pass"], output, 10)

    def test_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(run([sys.executable, "-c", "raise SystemExit(3)"], Path(root)/"fail", 10)["returncode"], 3)
            self.assertEqual(run([sys.executable, "-c", "import time; time.sleep(5)"], Path(root)/"timeout", 1)["status"], "timed_out")

    def test_invalid_budget(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                run([sys.executable], Path(root)/"invalid", 0)

if __name__ == "__main__":
    unittest.main()
