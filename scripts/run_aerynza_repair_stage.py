"""One authorized, resource-bounded Aerynza remediation stage, not an endless self-training loop.

Preflight is CPU-only. Only --run executes training/evaluation; approval files
bind exact reviewed dataset and parent adapter bytes to one optimizer plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aerynza_training_optimizer import read_object, sha256_file
from scripts.evaluate_qwen_ascension_lora import directory_sha256
from scripts.train_qwen_ascension_lora import load_curriculum

MODEL_REVISION_DEFAULT = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def verify_stage(plan_path: Path, approval_path: Path, parent: Path,
                 training: Path, heldout: Path, output: Path) -> tuple[dict, list[str]]:
    plan = read_object(plan_path)
    approval = read_object(approval_path)
    if plan.get("schema_version") != 1 or plan.get("optimization_not_executed") is not True:
        raise ValueError("unrecognized optimizer plan")
    if plan.get("optimizer_action") != "investigate_and_prepare_reviewed_repair":
        raise PermissionError("critical failure, retry limit, or other condition blocks automatic retraining")
    if plan.get("attempt", -1) >= plan.get("max_attempts", 0):
        raise PermissionError("training retry budget exhausted")
    if approval.get("approved") is not True or not str(approval.get("reviewer") or "").strip():
        raise PermissionError("explicit independent human approval required")
    if approval.get("plan_sha256") != sha256_file(plan_path):
        raise PermissionError("approved plan has changed")
    if approval.get("training_sha256") != sha256_file(training):
        raise PermissionError("reviewed training data changed")
    if approval.get("heldout_sha256") != sha256_file(heldout):
        raise PermissionError("independent held-out data changed")
    if approval.get("independently_authored_examples") is not True:
        raise PermissionError("new correction examples must be authored independently of release fixtures")
    if approval.get("no_heldout_or_model_output_leakage") is not True:
        raise PermissionError("unverified held-out separation")
    if approval.get("scoped_data_permissions_reviewed") is not True:
        raise PermissionError("privacy/consent review is required")
    if not parent.is_dir() or not (parent / "adapter_config.json").is_file():
        raise FileNotFoundError("the actual reviewed parent LoRA adapter is missing")
    if approval.get("parent_adapter_sha256") != directory_sha256(parent):
        raise PermissionError("parent adapter differs from approved bytes")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("repair candidate already exists; refuse overwrite or silent restart")
    if parent.resolve() == output.resolve() or output.resolve().is_relative_to(parent.resolve()):
        raise ValueError("new candidate must not overwrite parent")
    # Reuse trainer's existing validated schema, then reject direct overlap.
    # This cannot prove absence of semantic leakage; independent reviewer attests that.
    rows, _ = load_curriculum(str(training.resolve()), minimum_records=20)
    eval_rows, _ = load_curriculum(str(heldout.resolve()), minimum_records=8)
    if {r["id"] for r in rows} & {r["id"] for r in eval_rows}:
        raise ValueError("training and held-out IDs overlap")
    if {str(r["user"]).strip().casefold() for r in rows} & {
            str(r["user"]).strip().casefold() for r in eval_rows}:
        raise ValueError("training and held-out prompts overlap")
    if training.resolve() == heldout.resolve():
        raise ValueError("heldout set cannot be used as training curriculum")
    if plan.get("candidate_base") != "Qwen/Qwen3-1.7B":
        raise ValueError("repair stage only supports the pinned 1.7B candidate")
    if plan.get("candidate_model_revision") != MODEL_REVISION_DEFAULT:
        raise ValueError("model revision not approved by this runner")
    if plan["candidate_adapter"] != (parent.relative_to(ROOT).as_posix()
                                     if parent.is_relative_to(ROOT) else str(parent)):
        raise ValueError("parent path differs from candidate in optimizer plan")
    cmd = [
        sys.executable, "scripts/train_qwen_ascension_lora.py",
        "--model", "Qwen/Qwen3-1.7B",
        "--model-revision", MODEL_REVISION_DEFAULT,
        "--resume-adapter", str(parent),
        "--curriculum", str(training),
        "--eval-curriculum", str(heldout),
        "--output-dir", str(output),
        "--epochs", "1", "--learning-rate", "2e-6",
        "--max-length", "1024", "--save-steps", "2",
    ]
    return plan, cmd


def execute(plan: dict, command: list[str], output: Path, training_budget_minutes: int,
            gate_budget_minutes: int) -> dict:
    if not 1 <= training_budget_minutes <= 720 or not 1 <= gate_budget_minutes <= 720:
        raise ValueError("resource budgets must be 1 to 720 minutes")
    # No shell; timeout bounds resource usage per stage; no retries or paid GPU provisioning.
    receipts = []
    reports_dir = ROOT / "evals" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)
    label = "aerynza_repair_attempt_" + str(plan["attempt"] + 1)
    def launch(label_part: str, args: list[str], minutes: int):
        try:
            code = subprocess.run(args, cwd=ROOT, timeout=minutes * 60, check=False).returncode
            return {"stage": label_part, "exit_code": code}
        except subprocess.TimeoutExpired:
            return {"stage": label_part, "exit_code": None, "status": "resource_budget_expired"}
    trained = launch("train", command, training_budget_minutes)
    receipts.append(trained)
    if trained["exit_code"] != 0:
        return {"status": "training_incomplete_or_failed", "stages": receipts,
                "not_promoted": True, "exact_checkpoint_resume_required": True}
    args_common = ["--model", "Qwen/Qwen3-1.7B",
                   "--model-revision", MODEL_REVISION_DEFAULT, "--adapter", str(output)]
    evaluations = [
        ("canonical", "scripts/evaluate_qwen_ascension_lora.py",
         str(reports_dir / (label + "_canonical.json"))),
        ("capability", "scripts/evaluate_per_capability.py",
         str(reports_dir / (label + "_capability.json"))),
        ("execution", "scripts/evaluate_capability_execution.py",
         str(reports_dir / (label + "_execution.json"))),
    ]
    gate_paths = {}
    for name, script, report_path in evaluations:
        gate_result = launch(name, [sys.executable, script, *args_common,
                                    "--output", report_path], gate_budget_minutes)
        receipts.append(gate_result)
        if gate_result["exit_code"] not in (0, 2) or not Path(report_path).is_file():
            return {"status": "gate_error_not_promotable", "stages": receipts,
                    "not_promoted": True}
        gate_paths[name] = report_path
        # Never lower gate thresholds, never skip failed gates, but avoid burning
        # free compute on 640 probes when canonical already failed.
        if name == "canonical" and gate_result["exit_code"] != 0:
            break
        if name == "capability" and gate_result["exit_code"] != 0:
            break
    receipt_path = output / "ascension_training_receipt.json"
    if not receipt_path.is_file():
        return {"status": "missing_trainer_receipt", "stages": receipts, "not_promoted": True}
    followup = reports_dir / (label + "_optimizer.json")
    optimizer_args = [sys.executable, "scripts/aerynza_training_optimizer.py",
                      "--training-receipt", str(receipt_path),
                      "--canonical", gate_paths["canonical"],
                      "--attempt", str(plan["attempt"] + 1),
                      "--max-attempts", str(plan["max_attempts"]),
                      "--output", str(followup)]
    for name in ("capability", "execution"):
        if name in gate_paths:
            optimizer_args += ["--" + name, gate_paths[name]]
    diagnosis = launch("diagnose", optimizer_args, 2)
    receipts.append(diagnosis)
    return {"status": "blocked_pending_independent_review",
            "followup_optimizer_plan": str(followup) if followup.is_file() else None,
            "stages": receipts, "not_promoted": True}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reviewed and bounded Aerynza repair stage")
    p.add_argument("--plan", required=True)
    p.add_argument("--approval", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--training-budget-minutes", type=int, default=120)
    p.add_argument("--gate-budget-minutes", type=int, default=120)
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)
    plan, command = verify_stage(
        Path(args.plan).resolve(), Path(args.approval).resolve(),
        Path(args.parent_adapter).resolve(), Path(args.training).resolve(),
        Path(args.heldout).resolve(), Path(args.output_dir).resolve())
    print(json.dumps({"state": "ready_for_one_reviewed_candidate" if not args.run else "executing",
                      "attempt": plan["attempt"] + 1, "max_attempts": plan["max_attempts"],
                      "command": command, "automatic_promotion": False}, indent=2))
    if not args.run:
        return 0
    result = execute(plan, command, Path(args.output_dir).resolve(),
                     args.training_budget_minutes, args.gate_budget_minutes)
    print(json.dumps(result, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
