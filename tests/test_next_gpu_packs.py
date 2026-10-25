"""Dependency-free integrity checks for the Aerynza next-GPU curriculum.

These tests do not execute training or assert real tool/provider capabilities.
"""
from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_next_gpu_packs import SOURCES, build, heldout_set, normalize
from scripts.train_qwen_ascension_lora import SUPPORTED_SHELLS, SYSTEM_PROMPTS


class NextGPUCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.manifest = build()

    def test_package_is_large_and_manifest_declares_no_product_readiness(self):
        self.assertGreaterEqual(len(self.rows), 1700)
        self.assertEqual(self.manifest["training_examples"], len(self.rows))
        self.assertFalse(self.manifest["production_replacement_ready"])
        self.assertTrue(self.manifest["human_review_required"])
        self.assertTrue(self.manifest["registered_only_not_verified_execution"])

    def test_all_640_catalog_ids_are_covered_once_as_lessons_not_receipts(self):
        catalog = json.loads(SOURCES["catalog"].read_text(encoding="utf-8"))
        self.assertEqual(len(catalog), 640)
        ids = [item["capability_id"] for item in self.rows if "capability_id" in item]
        self.assertEqual(len(ids), 640)
        self.assertEqual(len(set(ids)), 640)
        self.assertEqual(set(ids), {entry["id"] for entry in catalog})

    def test_training_prompts_do_not_overlap_heldout_prompts(self):
        excluded = heldout_set()
        self.assertTrue(excluded)
        prompts = [normalize(row["user"]) for row in self.rows]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertFalse(set(prompts) & excluded)

    def test_all_training_shells_are_trainer_compatible(self):
        shells = Counter(row["shell"] for row in self.rows)
        self.assertTrue(set(shells).issubset(SUPPORTED_SHELLS))
        self.assertTrue(set(shells).issubset(SYSTEM_PROMPTS))
        self.assertIn("executive", SYSTEM_PROMPTS)
        self.assertNotIn("core", shells)  # Core is an entitlement, not a signed shell.

    def test_cross_domain_v254_has_review_pending_origins_and_32_distinct_tasks(self):
        rows = [row for row in self.rows if row.get("source") == "cross_domain_v254"]
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({row["user"] for row in rows}), 32)
        self.assertTrue({"body", "grocery", "home", "family", "creation", "recovery", "local", "systems"}
                        .issubset({row["package"].removeprefix("ngpu_cross_") for row in rows}))
        self.assertTrue({"ap", "lifeos", "executive", "sprout", "nexus_home", "nexus_family"}
                        .issubset({row["shell"] for row in rows}))
        for row in rows:
            self.assertEqual(row["source_kind"], "ai_authored_synthetic")
            self.assertEqual(row["review_status"], "pending_human_review")
            self.assertIn("cross_domain_v254", row["tags"])
        self.assertEqual(self.manifest["package_counts"]["ngpu_cross_body"], 4)

    def test_cross_domain_independent_holdout_never_enters_training(self):
        heldout_file = SOURCES["cross_domain_v254_heldout"]
        evaluations = [json.loads(line) for line in heldout_file.read_text().splitlines() if line.strip()]
        self.assertEqual(len(evaluations), 12)
        train_prompts = {normalize(item["user"]) for item in self.rows}
        for item in evaluations:
            self.assertTrue(item["must_not_train"])
            self.assertEqual(item["review_status"], "independent_eval_only")
            self.assertNotIn(normalize(item["user"]), train_prompts)
            self.assertNotIn("assistant", item)
        self.assertIn("cross_domain_v254", self.manifest["source_sha256"])
        self.assertIn("cross_domain_v254_heldout", self.manifest["source_sha256"])

    def test_training_output_is_deterministic(self):
        rows, manifest = build()
        self.assertEqual(rows, self.rows)
        self.assertEqual(manifest, self.manifest)

    def test_provenance_is_explicit_for_every_record(self):
        for row in self.rows:
            self.assertIn("source", row)
            self.assertIn("package", row)
            self.assertNotIn("reviewed_approved", str(row.get("review_status", "")))


if __name__ == "__main__":
    unittest.main()
