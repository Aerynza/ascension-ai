"""Fail-closed, dependency-free data and canon preflight for Aerynza v251.

This validates dataset integrity, not model quality or production execution.
Run before a GPU allocation; human review and all runtime gates remain necessary.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "evals/training/aerynza_v251_foundation_train.jsonl"
HELDOUT = ROOT / "evals/training/aerynza_v251_foundation_heldout.jsonl"
REPORT = ROOT / "public/capability_report.json"
SHELLS = {"ap", "lifeos", "sprout", "nexus_home", "nexus_family", "creation"}
PACKAGES = {
    "conversation", "tools_receipts", "privacy_auth", "planning_reasoning",
    "research_tools", "safety_specialist", "shell_coordination",
    "ambient_memory", "creation", "technical_autonomy",
}

def load(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected object")
        rows.append(row)
    return rows

def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()

def check_records(train: list[dict], heldout: list[dict]) -> dict:
    if len(train) < 40 or len(heldout) < 12:
        raise ValueError("curriculum too small for v251 preflight")
    seen_ids: set[str] = set()
    train_prompts: set[str] = set()
    all_pairs: set[tuple[str, str]] = set()
    packages = Counter()

    for split, rows in (("train", train), ("heldout", heldout)):
        for row in rows:
            if not {"id", "shell", "user", "assistant", "tags", "package", "provenance"}.issubset(row):
                raise ValueError(f"{split}: incomplete record: {row.get('id')}")
            rid = row["id"]
            if not isinstance(rid, str) or not rid or rid in seen_ids:
                raise ValueError(f"{split}: duplicate or empty ID: {rid}")
            seen_ids.add(rid)
            if row["shell"] not in SHELLS:
                raise ValueError(f"{rid}: illegal legacy/public shell {row['shell']!r}")
            if any(not isinstance(row[k], str) or len(row[k].strip()) < 8 for k in ("user", "assistant")):
                raise ValueError(f"{rid}: empty/short dialogue")
            if not isinstance(row["tags"], list) or "v251" not in row["tags"]:
                raise ValueError(f"{rid}: version tags required")
            if row.get("source_kind") == "real_user_conversation":
                raise ValueError(f"{rid}: user data cannot enter this original synthetic pack")
            if row.get("review_status") not in {"pending_human_review", "reviewed_approved"}:
                raise ValueError(f"{rid}: explicit human-review state required")
            user, assistant = norm(row["user"]), norm(row["assistant"])
            pair = (user, assistant)
            if pair in all_pairs:
                raise ValueError(f"{rid}: duplicate conversation pair")
            all_pairs.add(pair)
            if split == "train":
                if user in train_prompts:
                    raise ValueError(f"{rid}: duplicate training prompt")
                train_prompts.add(user)
                packages[row["package"]] += 1
            elif user in train_prompts:
                raise ValueError(f"{rid}: heldout prompt leaked into train")

    if set(packages) != PACKAGES:
        raise ValueError(f"missing/unexpected train packages: {sorted(PACKAGES ^ set(packages))}")
    if min(packages.values()) < 4:
        raise ValueError("every curriculum group must have at least four distinct scenarios")
    # A heldout prompt can appear earlier in the heldout set; disallow these too.
    h_prompts = [norm(r["user"]) for r in heldout]
    if len(h_prompts) != len(set(h_prompts)):
        raise ValueError("duplicate heldout prompts")
    return {"train": len(train), "heldout": len(heldout), "packages": dict(packages)}

def check_catalog(report: dict) -> dict:
    items = report.get("capabilities")
    if not isinstance(items, list) or len(items) != 640:
        raise ValueError("expected 640 current catalog entries; re-audit if registry changed")
    ids = [r.get("id") for r in items]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("catalog IDs are missing or duplicated")
    # Stored ready/execution booleans are historical claims, not receipts or evaluator results.
    return {"catalog_records": len(items), "catalog_status": "inventory_only_not_execution_verified"}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--heldout", type=Path, default=HELDOUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    summary = check_records(load(args.train), load(args.heldout))
    summary.update(check_catalog(json.loads(args.report.read_text(encoding="utf-8"))))
    summary["ready_for_gpu_training"] = False
    summary["remaining_gate"] = "human dataset review and existing independent GPU/model gates"
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
