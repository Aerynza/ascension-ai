"""Dependency-free regression tests for the candidate v251 training package."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.validate_v251_packages import check_catalog, check_records, load, norm

ROOT = Path(__file__).resolve().parents[1]

class V251PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load(ROOT / "evals/training/aerynza_v251_foundation_train.jsonl")
        cls.heldout = load(ROOT / "evals/training/aerynza_v251_foundation_heldout.jsonl")

    def test_preflight_passes_expected_manifest(self) -> None:
        summary = check_records(self.train, self.heldout)
        self.assertEqual(summary["train"], 60)
        self.assertEqual(summary["heldout"], 18)
        self.assertEqual(len(summary["packages"]), 10)

    def test_normalization_removes_unequal_whitespace(self) -> None:
        self.assertEqual(norm("  A  B\nC  "), "a b c")

    def test_train_heldout_prompt_leakage_fails(self) -> None:
        heldout = copy.deepcopy(self.heldout)
        heldout[0]["user"] = self.train[0]["user"]
        with self.assertRaisesRegex(ValueError, "heldout prompt leaked"):
            check_records(self.train, heldout)

    def test_legacy_core_shell_fails(self) -> None:
        train = copy.deepcopy(self.train)
        train[0]["shell"] = "core"
        with self.assertRaisesRegex(ValueError, "legacy/public shell"):
            check_records(train, self.heldout)

    def test_duplicate_record_id_fails(self) -> None:
        train = copy.deepcopy(self.train)
        train[1]["id"] = train[0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate or empty ID"):
            check_records(train, self.heldout)

    def test_no_real_user_conversations(self) -> None:
        train = copy.deepcopy(self.train)
        train[1]["source_kind"] = "real_user_conversation"
        with self.assertRaisesRegex(ValueError, "user data"):
            check_records(train, self.heldout)

    def test_unverified_catalog_is_inventory_only(self) -> None:
        report = json.loads((ROOT / "public/capability_report.json").read_text("utf-8"))
        checked = check_catalog(report)
        self.assertEqual(checked["catalog_records"], 640)
        self.assertIn("not_execution_verified", checked["catalog_status"])

if __name__ == "__main__":
    unittest.main()
