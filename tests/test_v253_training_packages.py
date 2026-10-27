"""Negative controls for v253 SFT provenance, holdouts and readiness claims."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_v253_packages import load_jsonl, preflight, validate


def fixtures():
    manifest = json.loads((ROOT / "training/v253/manifest.json").read_text())
    train = [r for f in manifest["train_files"] for r in load_jsonl(ROOT / f)]
    heldout = load_jsonl(ROOT / manifest["heldout_file"])
    benchmark = json.loads((ROOT / manifest["benchmark_file"]).read_text())
    prior = [r for f in (
        "aerynza_v251_foundation_train.jsonl",
        "aerynza_v251_foundation_heldout.jsonl",
        "aerynza_v252_execution_train.jsonl",
    ) for r in load_jsonl(ROOT / "evals/training" / f)]
    catalog = json.loads((ROOT / manifest["catalog_source"]).read_text())
    return train, heldout, benchmark, prior, manifest, catalog


class V253TrainingTests(unittest.TestCase):
    def setUp(self):
        self.input = fixtures()

    def check(self, *args):
        return validate(*args)

    def test_valid_manifest_is_inventory_only(self):
        report = preflight()
        self.assertEqual(report["train_records"], 72)
        self.assertEqual(report["heldout_records"], 24)
        self.assertEqual(report["independent_benchmark_specs"], 12)
        self.assertEqual(report["catalog_entries"], 640)
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["production_promotion_allowed"])
        self.assertFalse(report["gpu_training_completed"])
        self.assertFalse(report["frontier_parity_verified"])

    def test_duplicate_prompt_across_splits_blocks(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        heldout[0]["user"] = train[0]["user"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.check(train, heldout, bench, prior, manifest, catalog)

    def test_duplicate_record_id_blocks(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        heldout[0]["id"] = train[0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate record ID"):
            self.check(train, heldout, bench, prior, manifest, catalog)

    def test_legacy_core_shell_blocks(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        train[0]["shell"] = "core"
        with self.assertRaisesRegex(ValueError, "unexpected AI shell"):
            self.check(train, heldout, bench, prior, manifest, catalog)

    def test_benchmark_leak_blocks(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        bench["cases"][0]["task"] = train[0]["user"]
        with self.assertRaisesRegex(ValueError, "Benchmark leaked"):
            self.check(train, heldout, bench, prior, manifest, catalog)

    def test_fabricated_training_or_review_status_blocks(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        train[0]["review_status"] = "reviewed_approved"
        with self.assertRaisesRegex(ValueError, "do not fabricate"):
            self.check(train, heldout, bench, prior, manifest, catalog)

    def test_unverified_catalog_cannot_be_shortened(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        catalog["capabilities"].pop()
        with self.assertRaisesRegex(ValueError, "640 unique"):
            self.check(train, heldout, bench, prior, manifest, catalog)

    def test_benchmark_cannot_claim_completed_result(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        bench["cases"][0]["result"] = "passed"
        with self.assertRaisesRegex(ValueError, "must not claim"):
            self.check(train, heldout, bench, prior, manifest, catalog)

    def test_secrets_must_not_enter_synthetic_examples(self):
        train, heldout, bench, prior, manifest, catalog = copy.deepcopy(self.input)
        train[0]["assistant"] += " sk-proj-0123456789abcdefghijkl"
        with self.assertRaisesRegex(ValueError, "possible secret"):
            self.check(train, heldout, bench, prior, manifest, catalog)


if __name__ == "__main__":
    unittest.main()
