"""CPU-only contracts for v256 training, independent benchmarks and durable staging."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_next_gpu_packs import ROOT, SOURCES, build, normalize, read_jsonl
from scripts.build_v255_independent_eval import compile_heldout
from scripts.prepare_v256_gpu_bundle import compile_bundle, ensure_bundle
from scripts.select_v256_exact_resume import select
from scripts.qwen_resume_contract import contract_payload
from scripts.train_qwen_ascension_lora import load_curriculum
from scripts.validate_v256_production_packs import FAMILIES, validate


class V256TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = validate()

    def test_full_catalog_and_quality_split_are_verified_not_promoted(self):
        m = self.summary
        self.assertEqual((m["train_records"], m["heldout_records"],
                          m["runtime_benchmark_specs"], m["families"]), (60, 20, 20, 20))
        self.assertEqual(m["catalog_ids"], 640)
        self.assertGreaterEqual(m["full_gpu_candidate_records"], 1760)
        self.assertGreaterEqual(m["independent_eval_records"], 80)
        self.assertEqual(m["train_heldout_prompt_overlap"], 0)
        self.assertFalse(m["training_performed"])
        self.assertFalse(m["model_replacement_ready"])
        self.assertTrue(m["human_review_required"])

    def test_each_family_is_original_and_heldout_has_distinct_prompt(self):
        rows = read_jsonl(SOURCES["v256_train"])
        heldout = read_jsonl(SOURCES["v256_heldout"])
        ids = [r["id"] for r in rows + heldout]
        self.assertEqual(len(set(ids)), 80)
        self.assertEqual({r["package"].removeprefix("ngpu_v256_") for r in rows}, set(FAMILIES))
        self.assertFalse({normalize(r["user"]) for r in rows} &
                         {normalize(r["user"]) for r in heldout})
        self.assertTrue(all(r["review_status"] == "pending_human_review" for r in rows + heldout))
        self.assertTrue(all(r["reserved_for"] == "independent_evaluation_never_train" for r in heldout))

    def test_benchmarks_are_never_trained_or_claimed_passed(self):
        cases = json.loads(SOURCES["v256_benchmarks"].read_text())["cases"]
        self.assertEqual(len(cases), 20)
        train, _ = build()
        train_prompts = {normalize(row["user"]) for row in train}
        for case in cases:
            self.assertFalse(case["trainable"])
            self.assertEqual(case["result"], "not_run")
            self.assertEqual(case["fixture_status"], "requires_independent_fixture")
            self.assertNotIn(normalize(case["task"]), train_prompts)
            self.assertTrue(case["runtime_evidence_required"])

    def test_compiler_cannot_train_on_v256_independent_supervised_responses(self):
        train, manifest = build()
        eval_rows, metadata = compile_heldout()
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in eval_rows})
        self.assertEqual(manifest["catalog_training_coverage"], 640)
        self.assertFalse(manifest["production_replacement_ready"])
        self.assertEqual(metadata["purpose"], "evaluation_only_never_train")

    def test_immutable_bundle_matches_sources_and_refuses_drift(self):
        content, receipt = compile_bundle()
        self.assertGreater(len(content), 3)
        self.assertEqual(receipt["registered_capability_ids"], 640)
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["production_replacement_ready"])
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp) / "bundle"
            result = ensure_bundle(folder, write=True)
            self.assertEqual(result["mode"], "content_verified_and_written")
            self.assertEqual(ensure_bundle(folder, write=True)["files"], receipt["files"])
            file = folder / "eval" / "independent_supervised.jsonl"
            file.write_text(file.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bundle drift"):
                ensure_bundle(folder, write=True)

    def test_exact_resume_requires_full_optimizer_scheduler_rng_and_contract(self):
        bundle = ROOT / "training_packs/next_gpu/bundles/v256"
        self.assertTrue((bundle / "manifest.json").is_file())
        _, train = load_curriculum(bundle.relative_to(ROOT).as_posix() + "/train/*.jsonl")
        _, evaluation = load_curriculum(bundle.relative_to(ROOT).as_posix() +
                                        "/eval/independent_supervised.jsonl", minimum_records=8)
        contract = contract_payload(
            base_model="Qwen/Qwen3-1.7B",
            revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            parent_adapter=None,
            curriculum=train, eval_curriculum=evaluation, max_length=1024,
            epochs=1.0, learning_rate=2e-6, seed=3407, save_steps=2,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output = Path(temp) / "candidate"
            checkpoint = output / "checkpoint-12"
            checkpoint.mkdir(parents=True)
            (output / "aerynza_exact_resume_contract.json").write_text(
                json.dumps(contract), encoding="utf-8")
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 12}), encoding="utf-8")
            for name in ("adapter_model.safetensors", "adapter_config.json",
                         "optimizer.pt", "scheduler.pt", "rng_state_0.pth"):
                (checkpoint / name).write_bytes(b"test-fixture-not-model-weights")
            self.assertEqual(select(output, bundle)["global_step"], 12)
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "No compatible full-state checkpoint"):
                select(output, bundle)
            (checkpoint / "optimizer.pt").write_bytes(b"recreated-test-fixture")
            (output / "aerynza_exact_resume_contract.json").write_text(
                json.dumps({**contract, "seed": 99}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No compatible full-state checkpoint"):
                select(output, bundle)


if __name__ == "__main__":
    unittest.main()
