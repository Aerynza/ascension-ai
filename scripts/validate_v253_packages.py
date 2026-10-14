"""Strict, dependency-free Aerynza v253 data/catalog preflight.

This is not a model evaluation, human approval, a GPU reservation, or proof
that any of the 640 catalog entries can run a provider action.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "training/v253/manifest.json"
PRIOR = [
    ROOT / "evals/training/aerynza_v251_foundation_train.jsonl",
    ROOT / "evals/training/aerynza_v251_foundation_heldout.jsonl",
    ROOT / "evals/training/aerynza_v252_execution_train.jsonl",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing package: {path}")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: invalid JSON") from error
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{number}: expected JSON object")
        records.append(item)
    return records


def normalize(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.casefold()).strip()


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(train: list[dict], heldout: list[dict], benchmarks: dict,
             prior: list[dict], manifest: dict, catalog: dict) -> dict:
    groups = set(manifest["packages"])
    valid_shells = set(manifest["permitted_shells"])
    if len(groups) != 12 or "core" in valid_shells:
        raise ValueError("Expected twelve feature groups, no Core AI shell")
    if len(train) != manifest["train_records"] or len(heldout) != manifest["heldout_records"]:
        raise ValueError("Train/heldout count does not match manifest")
    if not isinstance(benchmarks, dict) or len(benchmarks.get("cases", [])) != manifest["benchmark_cases"]:
        raise ValueError("Benchmark count does not match manifest")
    if benchmarks.get("trainable", True) or not benchmarks.get("benchmark_prompts_must_not_enter_training"):
        raise ValueError("Benchmarks must be kept out of training")

    seen_ids: set[str] = set()
    prompts: set[str] = set()
    for row in prior:
        if isinstance(row.get("id"), str): seen_ids.add(row["id"])
        if isinstance(row.get("user"), str): prompts.add(normalize(row["user"]))

    train_counts, heldout_counts = Counter(), Counter()
    for split, rows, counts, required_count in (
        ("train", train, train_counts, 6),
        ("heldout", heldout, heldout_counts, 2),
    ):
        for row in rows:
            rid = row.get("id")
            if not isinstance(rid, str) or not rid or rid in seen_ids:
                raise ValueError(f"{split}: missing or duplicate record ID: {rid}")
            seen_ids.add(rid)
            if row.get("shell") not in valid_shells:
                raise ValueError(f"{rid}: unexpected AI shell: {row.get('shell')}")
            group = str(row.get("package", "")).removeprefix("v253_")
            if group not in groups:
                raise ValueError(f"{rid}: unknown package group")
            counts[group] += 1
            for key in ("user", "assistant"):
                value = row.get(key)
                if not isinstance(value, str) or len(value.strip()) < 28:
                    raise ValueError(f"{rid}: insufficient conversational example ({key})")
                if re.search(r"sk-proj-[a-z0-9]{12,}|-----BEGIN (?:RSA )?PRIVATE KEY-----|ghp_[a-zA-Z0-9]{25,}", value, re.I):
                    raise ValueError(f"{rid}: possible secret in dataset")
            key = normalize(row["user"])
            if key in prompts:
                raise ValueError(f"{rid}: prompt overlap in prior/train/heldout")
            prompts.add(key)
            if row.get("provenance") != "ai_authored_synthetic" or row.get("source_kind") != "synthetic":
                raise ValueError(f"{rid}: synthetic provenance required")
            if row.get("review_status") != "pending_human_review":
                raise ValueError(f"{rid}: do not fabricate review approval")
            tags = row.get("tags")
            if not isinstance(tags, list) or "v253" not in tags or group not in tags:
                raise ValueError(f"{rid}: version and package tags required")
        if set(counts) != groups or any(counts[x] != required_count for x in groups):
            raise ValueError(f"{split}: expected {required_count} records for every package")

    benchmark_counts = Counter()
    for case in benchmarks["cases"]:
        rid = case.get("id")
        if not isinstance(rid, str) or not rid or rid in seen_ids:
            raise ValueError("Duplicate benchmark or training identifier")
        seen_ids.add(rid)
        group = case.get("family")
        if group not in groups:
            raise ValueError("Unexpected benchmark category")
        benchmark_counts[group] += 1
        task = case.get("task")
        if not isinstance(task, str) or len(task) < 35:
            raise ValueError("Benchmark requires a concrete scenario")
        prompt = normalize(task)
        if prompt in prompts:
            raise ValueError("Benchmark leaked into training or heldout")
        prompts.add(prompt)
        if case.get("result") != "not_run" or case.get("fixture_status") != "requires_independent_fixture":
            raise ValueError("Unrun benchmark must not claim measured success")
        if case.get("review_status") != "pending_independent_review":
            raise ValueError("Benchmark review status misrepresented")
        if len(case.get("acceptance", [])) < 2 or not case.get("critical_failures"):
            raise ValueError("Benchmark needs outcome and safety acceptance criteria")
    if set(benchmark_counts) != groups or any(x != 1 for x in benchmark_counts.values()):
        raise ValueError("Exactly one independent benchmark per group required")

    catalog_items = catalog.get("capabilities")
    ids = [c.get("id") for c in catalog_items] if isinstance(catalog_items, list) else []
    if len(ids) != 640 or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != 640:
        raise ValueError("640 unique registry IDs required; inventory is not execution proof")
    return {
        "schema_version": manifest["schema_version"],
        "train_records": len(train),
        "heldout_records": len(heldout),
        "independent_benchmark_specs": len(benchmarks["cases"]),
        "training_groups": dict(sorted(train_counts.items())),
        "catalog_entries": 640,
        "catalog_status": "inventory_only_not_verified_execution",
        "synthetic_provenance_valid": True,
        "train_heldout_benchmark_overlap": False,
        "model_evaluated": False,
        "gpu_training_completed": False,
        "frontier_parity_verified": False,
        "human_review_required": True,
        "training_authorized": False,
        "production_promotion_allowed": False,
    }


def preflight() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = [ROOT / p for p in manifest["train_files"]]
    train = [x for path in paths for x in load_jsonl(path)]
    heldout_path = ROOT / manifest["heldout_file"]
    heldout = load_jsonl(heldout_path)
    benchmark_path = ROOT / manifest["benchmark_file"]
    benchmarks = json.loads(benchmark_path.read_text(encoding="utf-8"))
    prior = [x for path in PRIOR for x in load_jsonl(path)]
    catalog_path = ROOT / manifest["catalog_source"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    result = validate(train, heldout, benchmarks, prior, manifest, catalog)
    result["sha256"] = {str(p.relative_to(ROOT)): checksum(p) for p in paths + [heldout_path, benchmark_path, MANIFEST, catalog_path]}
    return result


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))
