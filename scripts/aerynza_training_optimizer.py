"""CPU-only failure diagnosis and bounded, fail-closed repair planning for Aerynza.

This script NEVER trains, copies held-out examples, changes release thresholds, or
promotes a model. It consumes evidence from existing v20/v251 gate report JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRIORITY = {"critical": 0, "high": 1, "medium": 2}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required evidence: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return data


def require_bool(item: dict[str, Any], key: str) -> bool:
    val = item.get(key)
    if not isinstance(val, bool):
        raise ValueError(f"required boolean {key!r} missing or invalid")
    return val


def require_results(report: dict[str, Any], key: str, *, allow_empty: bool = False) -> list[dict]:
    values = report.get(key)
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"missing or empty result list {key}")
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"non-object in {key}")
    return values


def add_failure(failures: list[dict], gate: str, case_id: str, code: str, severity: str,
                recommendation: str) -> None:
    failures.append({
        "gate": gate,
        "case_id": case_id[:100],
        "diagnosis": code,
        "severity": severity,
        "next_action": recommendation,
    })


def analyze_canonical(report: dict[str, Any], failures: list[dict]) -> bool:
    required = (
        "canonical_passed", "receipt_truth_passed", "automatic_gate_passed",
        "no_false_completion_claims_passed", "blocking_capability_readiness_passed",
    )
    for name in required:
        require_bool(report, name)
    canonical = require_results(report, "canonical_results")
    receipt = require_results(report, "receipt_truth_results")
    for index, case in enumerate(canonical):
        case_id = f"canonical-{index + 1:03d}"  # do not expose held-out text in planner output
        if case.get("generation_error"):
            add_failure(failures, "canonical", case_id, "generation_or_runtime",
                        "high", "Diagnose inference/model load and decode settings; do not retrain to repair a runtime defect")
        if not require_bool(case, "structural_pass"):
            add_failure(failures, "canonical", case_id, "format_or_decoder",
                        "high", "Inspect tokenizer, template, stopping behavior and independently authored formatting examples")
        if not require_bool(case, "semantic_pass"):
            add_failure(failures, "canonical", case_id, "task_reasoning",
                        "medium", "Review independently authored analogous tasks and mix approved native-domain replay")
    for index, case in enumerate(receipt):
        case_id = str(case.get("id") or f"receipt-{index+1:03d}")
        truth = require_bool(case, "truth_passed")
        readiness = require_bool(case, "readiness_passed")
        blocking = require_bool(case, "promotion_blocking")
        if case.get("generation_error"):
            add_failure(failures, "receipt_truth", case_id, "generation_or_runtime",
                        "high", "Repair model loading or generation failure before planning retraining")
        if not truth:
            add_failure(failures, "receipt_truth", case_id, "privacy_receipt_truth",
                        "critical", "Stop release; independent security review, receipt/authorization runtime audit, then separately reviewed synthetic training examples")
        if blocking and not readiness:
            add_failure(failures, "receipt_truth", case_id, "missing_capability_reasoning",
                        "high", "Review tool availability and improve independently authored honest-capability reasoning examples")
    passed = (report["canonical_passed"] and report["receipt_truth_passed"]
              and report["automatic_gate_passed"]
              and report["no_false_completion_claims_passed"]
              and report["blocking_capability_readiness_passed"])
    if not passed and not failures:
        add_failure(failures, "canonical", "aggregate", "unknown_gate_mismatch",
                    "critical", "Reconcile inconsistent aggregate results before optimizing")
    return passed


def analyze_capability(report: dict[str, Any], gate: str, failures: list[dict]) -> bool:
    count_key = "total_capabilities"
    total = report.get(count_key)
    passed = report.get("passed")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ValueError(f"{gate}: missing positive total_capabilities")
    if not isinstance(passed, int) or isinstance(passed, bool) or not 0 <= passed <= total:
        raise ValueError(f"{gate}: invalid passed count")
    results = require_results(report, "results")
    if len(results) != total:
        raise ValueError(f"{gate}: results length differs from total")
    ids = []
    for item in results:
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{gate}: missing case_id")
        ids.append(case_id)
        if not require_bool(item, "passed"):
            if gate == "capability":
                add_failure(failures, gate, case_id, "capability_text_probe",
                            "medium", "Audit actual capability/tool implementation first; write independent task variations and train only if a model behavior gap is confirmed")
            else:
                add_failure(failures, gate, case_id, "execution_plan_probe",
                            "medium", "Check authorized executor, provider receipts, timeouts and task planning; current two-turn probe does not verify execution")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{gate}: duplicate case IDs")
    if sum(bool(r["passed"]) for r in results) != passed:
        raise ValueError(f"{gate}: aggregate counts conflict with per-case results")
    return passed == total


def analyze_review(report: dict[str, Any], failures: list[dict]) -> bool:
    cases = require_results(report, "cases")
    for row in cases:
        review = row.get("human_review")
        if not isinstance(review, dict):
            raise ValueError("held-out human-review fields missing")
        if review.get("approved") is not True:
            add_failure(failures, "independent_review", str(row.get("id") or "unreviewed"),
                        "review_not_approved", "high",
                        "Obtain independent human assessment; never learn directly from held-out answers")
    return not any(item["gate"] == "independent_review" for item in failures)


def analyze(evidence: dict[str, dict[str, Any] | None], source_hashes: dict[str, str],
            attempt: int, max_attempts: int) -> dict[str, Any]:
    if not 0 <= attempt <= max_attempts or not 1 <= max_attempts <= 5:
        raise ValueError("attempt must be 0..max_attempts; max_attempts must be 1..5")
    receipt = evidence.get("training_receipt")
    canonical = evidence.get("canonical")
    if receipt is None or canonical is None:
        raise ValueError("candidate training receipt and canonical gate evidence are required")
    if receipt.get("status") != "trained_not_promoted" or not receipt.get("adapter_dir"):
        raise ValueError("a verified trained-but-not-promoted candidate receipt is required")
    adapter = receipt["adapter_dir"]
    if not isinstance(adapter, str) or not adapter.strip():
        raise ValueError("invalid candidate adapter")
    for name in ("canonical", "capability", "execution"):
        report = evidence.get(name)
        if report is not None and report.get("adapter") != adapter:
            raise ValueError(f"{name}: report adapter does not match training receipt")
    failures: list[dict] = []
    statuses = {"canonical": "not_run", "capability": "not_run", "execution": "not_run",
                "independent_review": "not_run"}
    statuses["canonical"] = "pass" if analyze_canonical(canonical, failures) else "fail"
    for name in ("capability", "execution"):
        if evidence.get(name) is not None:
            statuses[name] = ("pass" if analyze_capability(evidence[name], name, failures) else "fail")
        else:
            add_failure(failures, name, "missing", "gate_not_run", "high",
                        "Complete this original unchanged release gate before any readiness decision")
    if evidence.get("independent_review") is not None:
        statuses["independent_review"] = ("pass" if analyze_review(evidence["independent_review"], failures) else "fail")
    else:
        add_failure(failures, "independent_review", "missing", "review_not_run", "high",
                    "Independent human review still required")
    if not all(statuses[name] == "pass" for name in statuses):
        release_status = "blocked"
    else:
        release_status = "requires_separate_security_runtime_and_release_approval"
    if release_status != "blocked":
        # A planning tool NEVER certifies a release.
        failures.append({"gate": "release", "case_id": "full_system",
                         "diagnosis": "runtime_coverage_not_proven", "severity": "high",
                         "next_action": "Complete signed tool/runtime, privacy, live execution and human release reviews"})
    counts = Counter(item["diagnosis"] for item in failures)
    failures.sort(key=lambda f: (PRIORITY[f["severity"]], f["gate"], f["case_id"]))
    critical = any(row["severity"] == "critical" for row in failures)
    exhausted = attempt >= max_attempts
    action = ("stop_for_independent_review" if critical else
              "stop_budget_exhausted" if exhausted else
              "investigate_and_prepare_reviewed_repair")
    advice = [
        {"priority": 1, "kind": "engineering_first",
         "detail": "Repair evaluator, tokenizer, executor or provider integration failures in code; model fine-tuning alone cannot supply missing tools."},
        {"priority": 2, "kind": "reviewed_retraining",
         "detail": "If a real model behavior deficit is verified, independently author NEW examples, mix approved replay and train a new adapter stage from selected existing weights, not optimizer state."},
        {"priority": 3, "kind": "unchanged_release_gates",
         "detail": "Rerun original gates plus fresh unseen tests; do not train from held-out prompts, responses or rubrics, lower thresholds, or auto-promote."},
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_adapter": adapter,
        "candidate_base": receipt.get("base_model"),
        "candidate_model_revision": receipt.get("base_model_revision"),
        "attempt": attempt,
        "max_attempts": max_attempts,
        "remaining_attempts": max_attempts - attempt,
        "source_sha256": source_hashes,
        "gate_status": statuses,
        "release_status": release_status,
        "optimizer_action": action,
        "diagnosis_counts": dict(sorted(counts.items())),
        "failure_count": len(failures),
        "failures": failures,
        "recommended_work": advice,
        "optimization_not_executed": True,
        "heldout_examples_reused_for_training": False,
        "requires_human_review": True,
        "automatic_promotion": False,
        "production_replacement_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aerynza failure optimizer: CPU-only, no autonomous retraining")
    p.add_argument("--training-receipt", required=True)
    p.add_argument("--canonical", required=True)
    p.add_argument("--capability")
    p.add_argument("--execution")
    p.add_argument("--independent-review")
    p.add_argument("--attempt", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    paths = {k: Path(getattr(args, k.replace("-", "_"))).resolve()
             for k in ("training_receipt", "canonical", "capability", "execution", "independent_review")
             if getattr(args, k) is not None}
    outputs = {name: read_object(path) for name, path in paths.items()}
    plan = analyze(outputs, {name: sha256_file(path) for name, path in paths.items()},
                   args.attempt, args.max_attempts)
    path = Path(args.output).resolve()
    if path.exists():
        raise FileExistsError("existing optimizer plan is immutable; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "action": plan["optimizer_action"],
                      "failed_cases": plan["failure_count"],
                      "release_status": plan["release_status"]}, indent=2))
    return 0 if plan["gate_status"]["canonical"] == "pass" and not plan["failure_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
