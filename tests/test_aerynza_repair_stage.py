"""Preflight-only tests: no GPU, no model load, no training jobs."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aerynza_training_optimizer import sha256_file
from scripts.evaluate_qwen_ascension_lora import directory_sha256
from scripts.run_aerynza_repair_stage import ROOT, verify_stage


class ReviewBoundRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = self.root / "reviewed_adapter"
        self.parent.mkdir()
        (self.parent / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (self.parent / "adapter_model.safetensors").write_bytes(b"fixture")
        self.plan = self.root / "plan.json"
        self.train = ROOT / "evals/training/aerynza_v251_foundation_train.jsonl"
        self.heldout = ROOT / "evals/training/aerynza_v251_foundation_heldout.jsonl"
        self.out = self.root / "repair_candidate"
        self.approval = self.root / "approval.json"
        self.write_plan()
        self.write_approval()

    def write_plan(self, **kwargs):
        plan = {
            "schema_version": 1,
            "optimization_not_executed": True,
            "optimizer_action": "investigate_and_prepare_reviewed_repair",
            "attempt": 0,
            "max_attempts": 2,
            "candidate_base": "Qwen/Qwen3-1.7B",
            "candidate_model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "candidate_adapter": str(self.parent),
        }
        plan.update(kwargs)
        self.plan.write_text(json.dumps(plan))

    def write_approval(self, **kwargs):
        approval = {
            "approved": True,
            "reviewer": "Independent test fixture only",
            "plan_sha256": sha256_file(self.plan),
            "training_sha256": sha256_file(self.train),
            "heldout_sha256": sha256_file(self.heldout),
            "parent_adapter_sha256": directory_sha256(self.parent),
            "independently_authored_examples": True,
            "no_heldout_or_model_output_leakage": True,
            "scoped_data_permissions_reviewed": True,
        }
        approval.update(kwargs)
        self.approval.write_text(json.dumps(approval))

    def check(self):
        return verify_stage(self.plan, self.approval, self.parent,
                            self.train, self.heldout, self.out)

    def test_approved_stage_has_new_output_and_parent_adapter(self):
        plan, cmd = self.check()
        self.assertEqual(plan["attempt"], 0)
        self.assertIn("--resume-adapter", cmd)
        self.assertIn(str(self.parent), cmd)
        self.assertIn("--save-steps", cmd)
        self.assertNotIn("--resume-checkpoint", cmd)

    def test_unapproved_blocks_training(self):
        self.write_approval(approved=False)
        with self.assertRaisesRegex(PermissionError, "approval"):
            self.check()

    def test_plan_changed_after_approval_blocks(self):
        self.write_plan(max_attempts=3)
        with self.assertRaisesRegex(PermissionError, "plan has changed"):
            self.check()

    def test_parent_changed_after_approval_blocks(self):
        (self.parent / "adapter_model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(PermissionError, "approved bytes"):
            self.check()

    def test_heldout_leakage_attestation_required(self):
        self.write_approval(no_heldout_or_model_output_leakage=False)
        with self.assertRaisesRegex(PermissionError, "held-out"):
            self.check()

    def test_critical_failure_cannot_run(self):
        self.write_plan(optimizer_action="stop_for_independent_review")
        self.write_approval()
        with self.assertRaises(PermissionError):
            self.check()

    def test_exhausted_budget_cannot_run(self):
        self.write_plan(attempt=2)
        self.write_approval()
        with self.assertRaises(PermissionError):
            self.check()

    def test_no_candidate_overwrite(self):
        self.out.mkdir()
        (self.out / "older").write_text("preserve me")
        with self.assertRaises(FileExistsError):
            self.check()


if __name__ == "__main__":
    unittest.main()
