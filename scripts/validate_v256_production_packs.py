#!/usr/bin/env python3
"""Offline, fail-closed preflight for v256 production-readiness curricula.

Strictly separates synthetic training examples, independent heldout answers,
and runtime benchmarks. Does not install a model, call an API, or score quality.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.build_next_gpu_packs import ROOT, SOURCES, build, heldout_set, normalize, read_jsonl

FAMILIES = (
    "context_dialogue", "quantitative_reasoning", "source_grounding",
    "code_debugging", "artifact_quality", "data_integrity",
    "tool_selection", "receipts_and_state", "failure_recovery",
    "memory_provenance", "cross_domain_orchestration", "ambient_agent",
    "family_and_child_safety", "multimodal_truth", "timezone_and_scheduling",
    "offline_and_self_reliance", "security_and_tools", "production_evaluation",
    "creative_originality", "careful_health_finance",
)
ACCEPTED_SHELLS = {"ap", "lifeos", "executive", "sprout", "nexus_home", "nexus_family"}
SPLITS = ("v256_train", "v256_heldout")
EXPECTED = (60, 20, 20)


def validate() -> dict:
    train, heldout = (read_jsonl(SOURCES[name]) for name in SPLITS)
    benchmarks = json.loads(SOURCES["v256_benchmarks"].read_text(encoding="utf-8"))
    cases = benchmarks.get("cases")
    if (len(train), len(heldout), len(cases or [])) != EXPECTED:
        raise ValueError("v256 requires 60 train, 20 heldout and 20 independent benchmarks")
    if benchmarks.get("run_status") != "not_run":
        raise ValueError("Benchmark results must not be fabricated before real execution")
    if benchmarks.get("purpose") != "independent_runtime_evaluation_only_never_train":
        raise ValueError("Runtime benchmark may never be training data")
    ids, prompts = set(), set()
    family_counts = Counter()
    heldout_families = Counter()
    for split, rows in (("train", train), ("heldout", heldout)):
        for item in rows:
            key = item.get("id")
            prompt = item.get("user")
            answer = item.get("assistant")
            shell = item.get("shell")
            if not all(isinstance(v, str) and len(v.strip()) > 18 for v in (prompt, answer)):
                raise ValueError(f"Unusable {split} prompt or target in {key}")
            if not isinstance(key, str) or not key.startswith("ngpu_v256_") or key in ids:
                raise ValueError(f"Duplicate or invalid v256 ID: {key}")
            if not isinstance(shell, str) or shell not in ACCEPTED_SHELLS:
                raise ValueError(f"Invalid shell for {key}; Core is a tier, not a shell")
            if item.get("source_kind") != "ai_authored_synthetic" or item.get("provenance") != "ai_authored_synthetic":
                raise ValueError(f"Unknown provenance for {key}")
            if item.get("review_status") != "pending_human_review":
                raise ValueError(f"Training approval was incorrectly asserted for {key}")
            if split == "heldout" and item.get("reserved_for") != "independent_evaluation_never_train":
                raise ValueError(f"Independent prompt not protected: {key}")
            if split == "train" and item.get("reserved_for"):
                raise ValueError(f"Training item incorrectly reserved: {key}")
            norm = normalize(prompt)
            if norm in prompts:
                raise ValueError(f"Repeated prompt across v256 splits: {key}")
            if any(marker in answer.casefold() for marker in (
                "i've already sent", "i've already paid", "i've already booked",
                "i have transferred", "we have deployed",
            )):
                raise ValueError(f"Potential unverified success assertion in {key}")
            ids.add(key)
            prompts.add(norm)
            if split == "train":
                family = item.get("package", "").removeprefix("ngpu_v256_")
                family_counts[family] += 1
            else:
                if item.get("package") != "v256_independent_heldout":
                    raise ValueError(f"Unexpected heldout package in {key}")
                tags = item.get("tags")
                family = tags[-1] if isinstance(tags, list) and tags else ""
                heldout_families[family] += 1
    if set(family_counts) != set(FAMILIES) or any(family_counts[k] != 3 for k in FAMILIES):
        raise ValueError("Every production-readiness family must contain three original training examples")
    if set(heldout_families) != set(FAMILIES) or any(heldout_families[k] != 1 for k in FAMILIES):
        raise ValueError("Every family must have one independent supervised evaluation")
    benchmark_families = Counter()
    for case in cases:
        key = case.get("id")
        task = case.get("task")
        if key in ids or not isinstance(key, str) or not key.startswith("ngpu_v256_benchmark_"):
            raise ValueError("Repeated or invalid independent benchmark ID")
        if not isinstance(task, str) or len(task.strip()) < 60:
            raise ValueError(f"Benchmark {key} requires a real, detailed task fixture")
        if normalize(task) in prompts:
            raise ValueError(f"Benchmark prompt leaked into supervised splits: {key}")
        if case.get("trainable") is not False or case.get("result") != "not_run":
            raise ValueError("Do not train on benchmarks or fabricate runtime results")
        if case.get("fixture_status") != "requires_independent_fixture":
            raise ValueError("Benchmark must require separately verified fixture")
        if not case.get("runtime_evidence_required"):
            raise ValueError("Benchmark cannot pass from a text-only training record")
        ids.add(key)
        prompts.add(normalize(task))
        benchmark_families[case.get("family")] += 1
    if set(benchmark_families) != set(FAMILIES) or any(v != 1 for v in benchmark_families.values()):
        raise ValueError("Every family needs one unseen executable benchmark")
    training, manifest = build()
    actual = [r for r in training if r.get("source") == "v256_production_readiness"]
    if len(actual) != len(train) or {r["id"] for r in actual} != {r["id"] for r in train}:
        raise ValueError("Some v256 examples were excluded by master GPU compiler")
    train_prompt_set = {normalize(r["user"]) for r in training}
    heldout_set_local = {normalize(r["user"]) for r in heldout}
    if train_prompt_set & heldout_set_local:
        raise ValueError("Heldout overlap detected in full GPU candidate")
    if not heldout_set_local <= heldout_set():
        raise ValueError("Independent v256 prompts are not excluded by the compiler")
    if manifest.get("catalog_training_coverage") != 640 or manifest.get("capability_catalog_ids") != 640:
        raise ValueError("640-ID training coverage registry is incomplete")
    from scripts.build_v255_independent_eval import compile_heldout
    independent, eval_meta = compile_heldout()
    if not heldout_set_local <= {normalize(row["user"]) for row in independent}:
        raise ValueError("Independent supervised union excludes v256 cases")
    return {
        "schema_version": "v256_training_integrity_1",
        "validation": "passed_structural_cpu_only",
        "train_records": len(train),
        "heldout_records": len(heldout),
        "runtime_benchmark_specs": len(cases),
        "families": len(FAMILIES),
        "full_gpu_candidate_records": len(training),
        "catalog_ids": manifest["catalog_training_coverage"],
        "independent_eval_records": len(independent),
        "train_heldout_prompt_overlap": 0,
        "training_performed": False,
        "model_replacement_ready": False,
        "human_review_required": True,
        "source_sha256": {name: hashlib.sha256(SOURCES[name].read_bytes()).hexdigest()
                          for name in (*SPLITS, "v256_benchmarks")},
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
