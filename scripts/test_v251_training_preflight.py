"""CPU-only tests for the next Aerynza training window."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_v251_640_truth_curriculum import build, read_jsonl
from scripts.qwen_resume_contract import (
    CheckpointContractError, contract_payload, prepare_run,
)


class V251TrainingContractTests(unittest.TestCase):
    def contract(self):
        return contract_payload(
            base_model="Qwen/Qwen3-1.7B", revision="pinned-revision",
            parent_adapter="checkpoints/reviewed-parent",
            curriculum={"sha256": "a" * 64, "files": ["approved.jsonl"], "records": 640},
            eval_curriculum={"sha256": "b" * 64, "files": ["heldout.jsonl"]},
            max_length=768, epochs=1.0, learning_rate=2e-6,
            seed=3407, save_steps=10,
        )

    def test_builder_covers_exactly_640_and_separates_heldout(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "train.jsonl"
            first = build(output)
            rows = read_jsonl(output)
            heldout = read_jsonl(Path(__file__).resolve().parents[1] /
                                 "evals/training/aerynza_v251_independent_heldout.jsonl")
            self.assertEqual(first["catalog_capabilities"], 640)
            self.assertEqual(len(rows), first["total_train_records"])
            self.assertGreaterEqual(first["curated_training_records"], 20)
            self.assertGreaterEqual(len(heldout), 8)
            self.assertFalse(first["heldout_used_in_training"])
            self.assertEqual(len(set(row["id"] for row in rows)), len(rows))
            self.assertFalse({r["user"].casefold() for r in rows} &
                             {r["user"].casefold() for r in heldout})
            self.assertFalse(any(row["shell"] == "core" for row in rows))
            self.assertTrue(any(row["shell"] == "executive" for row in rows))
            before = output.read_bytes()
            self.assertEqual(build(output), first)
            self.assertEqual(output.read_bytes(), before)

    def test_new_run_rejects_reusing_existing_output_even_from_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage"
            result = prepare_run(path, self.contract())
            self.assertEqual(result["mode"], "new_curriculum_stage")
            with self.assertRaisesRegex(CheckpointContractError, "nonempty"):
                prepare_run(path, self.contract())

    def test_exact_resume_requires_full_state_and_unchanged_curriculum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage"
            original = self.contract()
            prepare_run(path, original)
            checkpoint = path / "checkpoint-10"
            checkpoint.mkdir()
            for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt",
                         "rng_state.pth", "adapter_model.safetensors"):
                (checkpoint / name).write_text('{"global_step": 10}' if name ==
                                               "trainer_state.json" else "fixture")
            result = prepare_run(path, original, checkpoint)
            self.assertEqual(result["mode"], "exact_resume")
            self.assertEqual(result["global_step"], 10)
            mutated = {**original, "curriculum_sha256": "c" * 64}
            with self.assertRaisesRegex(CheckpointContractError, "mismatch"):
                prepare_run(path, mutated, checkpoint)
            (checkpoint / "scheduler.pt").unlink()
            with self.assertRaisesRegex(CheckpointContractError, "scheduler"):
                prepare_run(path, original, checkpoint)

    def test_resume_requires_trainer_state_and_parent_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "stage"
            prepare_run(path, self.contract())
            with self.assertRaisesRegex(CheckpointContractError, "exact output"):
                prepare_run(path, self.contract(), root / "another" / "checkpoint-10")
            checkpoint = path / "checkpoint-10"
            checkpoint.mkdir()
            with self.assertRaisesRegex(CheckpointContractError, "Incomplete"):
                prepare_run(path, self.contract(), checkpoint)


if __name__ == "__main__":
    unittest.main()
