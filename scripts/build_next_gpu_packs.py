#!/usr/bin/env python3
"""Deterministic, CPU-only compiler for the next Aerynza GPU training window.

Inputs are reviewed, committed curriculum sources. This does NOT train a model,
create tools, or prove the 640 capability catalog is executable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "balanced": ROOT / "evals/training/aerynza_v20_balanced_master.jsonl",
    "depth640": ROOT / "evals/training/aerynza_product_v236_natural_action_correction.jsonl",
    "catalog": ROOT / "evals/capability_status.json",
    "seed": ROOT / "training_packs/next_gpu/seed_scenarios.jsonl",
}
SUPPORTED = {"ap", "lifeos", "sprout", "nexus_home", "nexus_family", "executive"}
SHELL_ALIASES = {"core": "ap", "creation": "ap"}
PRIVATE = (
    re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"),  # US SSN-like
    re.compile(r"\b(?:sk-proj|ghp_|github_pat_|AKIA)[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
)
UNVERIFIED_SUCCESS = re.compile(
    r"\b(?:I(?:'ve| have| already)|we(?:'ve| have| already))\s+"
    r"(?:sent|paid|booked|transferred|deployed|published|saved|scheduled)\b",
    re.I,
)
RECORD_FIELDS = ("id", "shell", "user", "assistant", "package")
MIN_ROWS = 1700
MIN_640 = 640


def normalize(s: str) -> str:
    return " ".join(str(s).casefold().split())


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def read_jsonl(file: Path) -> list[dict]:
    if not file.is_file():
        raise FileNotFoundError(f"missing reviewed source: {file}")
    out = []
    for i, text in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        record = json.loads(text)
        if not isinstance(record, dict):
            raise ValueError(f"{file}:{i} expected object")
        out.append(record)
    return out


def heldout_set() -> set[str]:
    prompts: set[str] = set()
    # Evaluation fixtures are NEVER used to create assistant targets.
    for file in [ROOT / "evals/receipt_truth_prompts.json",
                 ROOT / "evals/replacement_readiness_prompts.json",
                 ROOT / "evals/aerynza_v20_ambient_heldout.json"]:
        if not file.exists():
            continue
        obj = json.loads(file.read_text(encoding="utf-8"))
        cases = obj if isinstance(obj, list) else obj.get("cases", [])
        for record in cases:
            if isinstance(record, dict):
                prompts.add(normalize(record.get("user") or record.get("prompt") or ""))
    for file in sorted((ROOT / "evals/training").glob("*heldout*.jsonl")):
        for record in read_jsonl(file):
            prompts.add(normalize(record.get("user") or record.get("prompt") or ""))
    return {p for p in prompts if p}


def private_or_false_receipt(text: str) -> bool:
    return any(pattern.search(text) for pattern in PRIVATE) or bool(UNVERIFIED_SUCCESS.search(text))


def cap_domain(row: dict) -> str:
    tags = row.get("tags") or []
    if len(tags) >= 5 and isinstance(tags[4], str) and re.fullmatch(r"[a-z_]{2,40}", tags[4]):
        return tags[4]
    return "unspecified"


def build() -> tuple[list[dict], dict]:
    for path in SOURCES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    catalog = json.loads(SOURCES["catalog"].read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or len(catalog) != MIN_640:
        raise ValueError(f"Expected exactly {MIN_640} catalog entries, got {len(catalog)}")
    known = {r["id"]: r for r in catalog}
    if len(known) != MIN_640:
        raise ValueError("Duplicate capability IDs in catalog")
    excluded = heldout_set()
    seen_ids: set[str] = set()
    responses_by_prompt: dict[tuple[str, str], str] = {}
    output: list[dict] = []
    failures = Counter()
    capability_coverage: set[str] = set()

    def add(row: dict, source: str) -> bool:
        row = dict(row)
        row["shell"] = SHELL_ALIASES.get(row.get("shell"), row.get("shell"))
        if row.get("shell") not in SUPPORTED:
            failures["unsupported_shell"] += 1
            return False
        for key in RECORD_FIELDS:
            if not isinstance(row.get(key), str) or not row[key].strip():
                failures["missing_or_invalid_field"] += 1
                return False
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate curriculum ID {row['id']}")
        prompt = normalize(row["user"])
        if prompt in excluded:
            failures["evaluation_prompt_excluded"] += 1
            return False
        if private_or_false_receipt(row["user"]) or private_or_false_receipt(row["assistant"]):
            failures["privacy_or_false_receipt"] += 1
            return False
        k = (row["shell"], prompt)
        answer = normalize(row["assistant"])
        if k in responses_by_prompt:
            if responses_by_prompt[k] != answer:
                failures["contradiction_quarantined"] += 1
            else:
                failures["duplicate_prompt"] += 1
            return False
        seen_ids.add(row["id"])
        responses_by_prompt[k] = answer
        row["source"] = source
        row["tags"] = list(row.get("tags") or [])
        output.append(row)
        return True

    # Proven foundation. Preserve useful natural dialogue and receipt lessons;
    # weighting is done in the trainer rather than repeating/copying records.
    for row in read_jsonl(SOURCES["balanced"]):
        item = dict(row)
        item["id"] = "ngpu_replay_" + str(row["id"])
        item["package"] = "ngpu_replay_" + str(row.get("package") or "general")
        add(item, "aerynza_v20_balanced_master")

    # Use one *reviewed* v236 response for each registered capability. Do not
    # use the original "Use X" evaluation prompt: hold-outs must remain isolated.
    rows = read_jsonl(SOURCES["depth640"])
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not str(row.get("id", "")).startswith("v236_retain_v235_cap_"):
            continue
        tags = row.get("tags") or []
        for tag in tags:
            if isinstance(tag, str) and tag in known:
                by_id[tag].append(row)
    if any(len(by_id[k]) != 1 for k in known):
        missing = sorted(k for k in known if len(by_id[k]) != 1)
        raise ValueError(f"640 source coverage incomplete or ambiguous: {missing[:20]}")
    for index, cap_id in enumerate(sorted(known)):
        row = by_id[cap_id][0]
        human_name = re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", cap_id)).strip().title()
        assistant = str(row["assistant"]).strip()
        # Older curricula overpromised rendering or provider integrations.
        assistant = assistant.replace("I will create the work using", "I can prepare the workflow using")
        assistant = assistant.replace("I will create ", "I can prepare ")
        catalog_status = known[cap_id].get("status")
        if catalog_status == "provider_action":
            assistant += " An external provider result requires a connected, authorized service and a real receipt."
        elif catalog_status == "native_action":
            assistant += " Actual artifact generation requires the relevant installed engine and a verified output."
        else:
            assistant += " This describes available guidance, not a claim that an external action occurred."
        item = {
            "id": f"ngpu_cap_{cap_id}",
            "shell": row.get("shell", "ap"),
            "user": f"Help me make a useful first result with {human_name}. What can we plan and what needs a working tool?",
            "assistant": assistant,
            "package": f"ngpu_capabilities_{index % 8:02d}",
            "tags": ["next_gpu", "640_catalog", "capability_scope", cap_domain(row), cap_id],
            "capability_id": cap_id,
            "catalog_status": catalog_status,
        }
        if not add(item, f"v236:{cap_id}"):
            raise ValueError(f"Required capability lesson excluded: {cap_id}")
        capability_coverage.add(cap_id)

    # Founder canonical examples: native self-reliance must not misrepresent
    # source access, memory writes, tool outcomes or guardian/tenant authority.
    for row in read_jsonl(SOURCES["seed"]):
        if not add(row, "next_gpu_seed_scenarios"):
            raise ValueError(f"Canonical scenario invalid: {row.get('id')}")
    output.sort(key=lambda r: (r["package"], r["id"]))
    if len(output) < MIN_ROWS or capability_coverage != set(known):
        raise ValueError("Training package too small or missing catalog coverage")

    counts = Counter(r["package"] for r in output)
    shells = Counter(r["shell"] for r in output)
    manifest = {
        "version": "next_gpu_2026_10_v1",
        "purpose": "GPU-ready curriculum; no training or provider execution performed",
        "base_model": "Qwen/Qwen3-1.7B",
        "base_model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "recommended_method": "assistant_only_lora_balanced_package_shell_sampler",
        "training_examples": len(output),
        "capability_catalog_ids": len(known),
        "catalog_training_coverage": len(capability_coverage),
        "registered_only_not_verified_execution": True,
        "human_review_required": True,
        "production_replacement_ready": False,
        "source_sha256": {name: digest(file) for name, file in SOURCES.items()},
        "source_exclusions": dict(sorted(failures.items())),
        "package_counts": dict(sorted(counts.items())),
        "shell_counts": dict(sorted(shells.items())),
        "heldout_prompt_count": len(excluded),
    }
    return output, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training_packs/next_gpu/generated")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows, manifest = build()
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    target = ROOT / args.output
    if target.exists() and any(target.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite nonempty training bundle {target}")
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for package in sorted({r["package"] for r in rows}):
        file = target / f"{package}.jsonl"
        with file.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if row["package"] == package:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        paths[package] = {
            "path": file.relative_to(ROOT).as_posix(),
            "sha256": digest(file),
            "rows": manifest["package_counts"][package],
        }
    manifest["generated"] = paths
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": (target / "manifest.json").as_posix(),
                      "records": len(rows), "capabilities": len(paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
