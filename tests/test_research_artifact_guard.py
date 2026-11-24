import tempfile
import unittest
from pathlib import Path
from scripts.research_artifact_guard import snapshot, verify

class ArtifactGuardTests(unittest.TestCase):
    def test_manifest_and_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            artifacts = {}
            for name in ("checkpoint", "metadata", "tokenizer", "dataset", "config", "gate"):
                path = Path(root) / name
                path.write_text(name)
                artifacts[name] = path
            expected = snapshot(artifacts)
            self.assertEqual(verify(artifacts, expected), expected)
            artifacts["checkpoint"].write_text("changed")
            with self.assertRaises(ValueError):
                verify(artifacts, expected)
            del artifacts["gate"]
            with self.assertRaises(ValueError):
                snapshot(artifacts)

if __name__ == "__main__":
    unittest.main()
