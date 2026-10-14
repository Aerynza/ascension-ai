"""Shared JSON canonicalization fixtures; no model, database or network required."""
import json
from pathlib import Path
import unittest

from src.serving.request_integrity import canonical_payload

FIXTURES = json.loads((Path(__file__).parent / "fixtures/native-canonicalization.json").read_text(encoding="utf-8"))


class CanonicalizationContract(unittest.TestCase):
    def test_shared_vectors(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture["id"]):
                self.assertEqual(canonical_payload(fixture["payload"]).decode("utf-8"), fixture["canonical"])

    def test_unsupported_values_fail_closed(self):
        for value in (float("nan"), float("inf"), 9007199254740992, "\ud800"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    canonical_payload({"value": value})


if __name__ == "__main__":
    unittest.main()

