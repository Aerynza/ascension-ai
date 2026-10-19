"""Dependency-free tests for fail-closed Aerynza release gate failure optimizer."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aerynza_training_optimizer import analyze, main


def fixtures():
    receipt = {"status": "trained_not_promoted", "adapter_dir": "checkpoints/candidate",
               "base_model": "Qwen/Qwen3-1.7B"}
    canonical = {
        "adapter": "checkpoints/candidate",
        "canonical_passed": True,
        "receipt_truth_passed": True,
        "automatic_gate_passed": True,
        "no_false_completion_claims_passed": True,
        "blocking_capability_readiness_passed": True,
        "canonical_results": [{"structural_pass": True, "semantic_pass": True,
                               "prompt": "HIDDEN HOLDOUT PROMPT", "text": "MODEL ANSWER"}],
        "receipt_truth_results": [{"id": "receipt-7", "truth_passed": True, "readiness_passed": True,
                                   "promotion_blocking": True,
                                   "text": "HIDDEN RECEIPT ANSWER"}],
    }
    capability = {"adapter": "checkpoints/candidate", "total_capabilities": 1, "passed": 1,
                  "results": [{"case_id": "cap-1", "passed": True, "response": "SECRET ANSWER"}]}
    execution = {"adapter": "checkpoints/candidate", "total_capabilities": 1, "passed": 1,
                 "results": [{"case_id": "cap-1", "passed": True,
                              "turn2_response": "UNTRUSTED OUTPUT"}]}
    review = {"cases": [{"id": "review-1", "human_review": {"approved": True},
                         "reference_response": "PRIVATE BENCHMARK"}]}
    return {"training_receipt": receipt, "canonical": canonical, "capability": capability,
            "execution": execution, "independent_review": review}


class TrainingOptimizerTests(unittest.TestCase):
    def plan(self, evidence=None, attempt=0, max_attempts=2):
        return analyze(evidence or fixtures(), {"canonical": "trusted-digest"}, attempt, max_attempts)

    def test_all_automated_gates_does_not_imply_release(self):
        out = self.plan()
        self.assertFalse(out["production_replacement_ready"])
        self.assertFalse(out["automatic_promotion"])
        self.assertNotEqual(out["release_status"], "pass")
        self.assertEqual(out["gate_status"]["execution"], "pass")

    def test_failed_receipt_truth_requires_security_review(self):
        inputs = fixtures()
        inputs["canonical"]["receipt_truth_results"][0].update({
            "truth_passed": False, "failures": ["false_completion_claim"]})
        inputs["canonical"]["receipt_truth_passed"] = False
        inputs["canonical"]["no_false_completion_claims_passed"] = False
        inputs["canonical"]["automatic_gate_passed"] = False
        out = self.plan(inputs)
        self.assertEqual(out["optimizer_action"], "stop_for_independent_review")
        self.assertIn("privacy_receipt_truth", out["diagnosis_counts"])

    def test_execution_failure_does_not_claim_tool_repair(self):
        inputs = fixtures()
        inputs["execution"]["results"][0]["passed"] = False
        inputs["execution"]["passed"] = 0
        out = self.plan(inputs)
        self.assertEqual(out["gate_status"]["execution"], "fail")
        self.assertIn("execution_plan_probe", out["diagnosis_counts"])
        self.assertTrue(out["optimization_not_executed"])

    def test_missing_gates_cannot_be_assumed_passed(self):
        inputs = fixtures()
        inputs["capability"] = None
        inputs["execution"] = None
        inputs["independent_review"] = None
        out = self.plan(inputs)
        self.assertEqual(out["gate_status"]["capability"], "not_run")
        self.assertEqual(out["release_status"], "blocked")

    def test_budget_exhaustion_never_loops(self):
        inputs = fixtures()
        inputs["capability"]["passed"] = 0
        inputs["capability"]["results"][0]["passed"] = False
        out = self.plan(inputs, attempt=2, max_attempts=2)
        self.assertEqual(out["optimizer_action"], "stop_budget_exhausted")

    def test_report_mismatch_rejected(self):
        inputs = fixtures()
        inputs["execution"]["adapter"] = "checkpoints/someone_else"
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.plan(inputs)

    def test_duplicate_capability_ids_rejected(self):
        inputs = fixtures()
        inputs["capability"]["results"] *= 2
        inputs["capability"]["total_capabilities"] = 2
        inputs["capability"]["passed"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
            self.plan(inputs)

    def test_counts_mismatch_rejected(self):
        inputs = fixtures()
        inputs["execution"]["passed"] = 0
        with self.assertRaisesRegex(ValueError, "aggregate counts"):
            self.plan(inputs)

    def test_heldout_text_never_appears_in_planner(self):
        data = json.dumps(self.plan())
        for phrase in ("HIDDEN HOLDOUT PROMPT", "MODEL ANSWER",
                       "HIDDEN RECEIPT ANSWER", "PRIVATE BENCHMARK", "UNTRUSTED OUTPUT"):
            self.assertNotIn(phrase, data)

    def test_unreviewed_heldout_cannot_pass(self):
        inputs = fixtures()
        inputs["independent_review"]["cases"][0]["human_review"]["approved"] = None
        out = self.plan(inputs)
        self.assertEqual(out["gate_status"]["independent_review"], "fail")

    def test_cli_writes_readonly_plan_and_preserves_existing(self):
        inputs = fixtures()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            argv = []
            for name, obj in inputs.items():
                path = root / (name + ".json")
                path.write_text(json.dumps(obj), encoding="utf-8")
                argv += ["--" + name.replace("_", "-"), str(path)]
            output = root / "plan.json"
            rc = main(argv + ["--output", str(output)])
            self.assertEqual(rc, 2)  # separate runtime release review still needed
            self.assertTrue(output.exists())
            self.assertFalse(json.loads(output.read_text())["automatic_promotion"])
            with self.assertRaises(FileExistsError):
                main(argv + ["--output", str(output)])


if __name__ == "__main__":
    unittest.main()
