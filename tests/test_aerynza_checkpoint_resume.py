"""Dependency-free tests for interrupted, resumable Aerynza LoRA training."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.aerynza_checkpoint_resume import (
    CHECKPOINT_MANIFEST, build_contract, checkpoint_errors, prepare_run,
    select_checkpoint, write_checkpoint_manifest,
)


class ResumeContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "candidate"
        args = SimpleNamespace(
            model="Qwen/Qwen3-1.7B", model_revision="pinned", max_length=1024,
            epochs=1.0, learning_rate=2e-6, seed=3407,
        )
        self.contract = build_contract(args, {"sha256": "training-digest"}, {"sha256": "heldout-digest"})

    def good_checkpoint(self, step=2):
        path = self.out / f"checkpoint-{step}"
        path.mkdir(parents=True)
        for name in ("optimizer.pt", "scheduler.pt", "adapter_config.json",
                     "adapter_model.safetensors", "rng_state.pth"):
            (path / name).write_bytes(b"placeholder")
        (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
        write_checkpoint_manifest(path, self.contract)
        return path

    def test_restarts_are_blocked_if_output_contains_earlier_work(self):
        prepare_run(self.out, self.contract, False)
        with self.assertRaises(FileExistsError):
            prepare_run(self.out, self.contract, False)

    def test_exact_resume_selects_last_complete_checkpoint(self):
        prepare_run(self.out, self.contract, False)
        older = self.good_checkpoint(2)
        self.good_checkpoint(4)
        (self.out / "checkpoint-4" / "optimizer.pt").unlink()
        self.assertEqual(select_checkpoint(self.out, "auto", self.contract), older)

    def test_explicit_incomplete_checkpoint_is_rejected(self):
        prepare_run(self.out, self.contract, False)
        broken = self.good_checkpoint(4)
        (broken / "rng_state.pth").unlink()
        with self.assertRaisesRegex(ValueError, "missing RNG"):
            select_checkpoint(self.out, str(broken), self.contract)

    def test_autoresume_never_silently_restarts_at_zero(self):
        prepare_run(self.out, self.contract, False)
        with self.assertRaises(FileNotFoundError):
            select_checkpoint(self.out, "auto", self.contract)

    def test_training_data_change_fails_closed(self):
        prepare_run(self.out, self.contract, False)
        changed = {**self.contract, "curriculum_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "configuration or dataset"):
            prepare_run(self.out, changed, True)

    def test_incomplete_checkpoint_marker_is_rejected(self):
        prepare_run(self.out, self.contract, False)
        checkpoint = self.good_checkpoint(8)
        (checkpoint / "checkpoint-is-incomplete.txt").write_text("busy")
        self.assertIn("incomplete checkpoint sentinel", checkpoint_errors(checkpoint, self.contract))

    def test_corrupt_checkpoint_manifest_is_rejected(self):
        prepare_run(self.out, self.contract, False)
        checkpoint = self.good_checkpoint(6)
        (checkpoint / CHECKPOINT_MANIFEST).write_text("{bad")
        self.assertIn("corrupt checkpoint metadata", checkpoint_errors(checkpoint, self.contract))

    def test_contract_mismatch_checkpoint_rejected(self):
        prepare_run(self.out, self.contract, False)
        checkpoint = self.good_checkpoint(8)
        changed = {**self.contract, "seed": 1}
        self.assertIn("checkpoint contract mismatch", checkpoint_errors(checkpoint, changed))


if __name__ == "__main__":
    unittest.main()
