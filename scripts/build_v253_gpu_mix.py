"""Deterministic, reviewed-source GPU mix; never includes benchmark prompts.

Creates a candidate curriculum from v251 foundation, v252 execution and v253
targeted domain examples. Not a dataset-approval or promotion decision.
"""
from __future__ import annotations
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from validate_v253_packages import ROOT, PRIOR, load_jsonl, preflight, normalize

OLD_TRAIN = [PRIOR[0], PRIOR[2]]
OUT = ROOT / "evals/training/aerynza_v253_gpu_mix_train.jsonl"
RECEIPT = ROOT / "evals/results/v253_gpu_mix_preflight.json"


def interleave(*sets: list[dict]) -> list[dict]:
    groups = [list(s) for s in sets]
    for i, group in enumerate(groups):
        random.Random(253 + i).shuffle(group)
    result = []
    while any(groups):
        for group in groups:
            if group:
                result.append(group.pop())
    return result


def build() -> dict:
    pre = preflight()
    manifest = json.loads((ROOT / "training/v253/manifest.json").read_text())
    v251 = load_jsonl(OLD_TRAIN[0])
    v252 = load_jsonl(OLD_TRAIN[1])
    v253 = [r for name in manifest["train_files"] for r in load_jsonl(ROOT / name)]
    rows = interleave(v251, v252, v253)
    if len(rows) != 156 or len({normalize(r["user"]) for r in rows}) != len(rows):
        raise ValueError("GPU mix must contain 156 independent, unique prompts")
    holdout = load_jsonl(ROOT / manifest["heldout_file"])
    heldout_norms = {normalize(r["user"]) for r in holdout}
    if heldout_norms.intersection({normalize(r["user"]) for r in rows}):
        raise ValueError("A training row overlaps heldout")
    benchmark = json.loads((ROOT / manifest["benchmark_file"]).read_text())
    if {normalize(r["task"]) for r in benchmark["cases"]}.intersection({normalize(r["user"]) for r in rows}):
        raise ValueError("Independent benchmark leaked into training")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    OUT.write_bytes(raw)
    receipt = {
        "status": "candidate_mix_not_reviewed",
        "output": str(OUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "records": len(rows),
        "by_source": {"v251_foundation": len(v251), "v252_execution": len(v252), "v253_targeted": len(v253)},
        "source_hashes": pre["sha256"],
        "training_authorized": False,
        "benchmark_used_for_training": False,
        "production_replacement_ready": False,
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
