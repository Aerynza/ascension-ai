"""Build a balanced, receipt-honest 640-capability *training* package.

A catalog entry is an intended skill name, not an operational integration.
Generated examples train scoped intake and next steps; they never pretend
external tools were called. Reviewed diverse examples get separate packages so
the existing trainer's per-package sampler does not drown them in catalog text.
The held-out file is validated and NEVER copied into the optimization corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_per_capability_gates import parse_table, CATEGORY_TO_SHELL
CATALOG = ROOT / "docs/CAPABILITIES_REFERENCE.md"
CURATED = ROOT / "evals/training/aerynza_v251_generalist_seed.jsonl"
HELDOUT = ROOT / "evals/training/aerynza_v251_independent_heldout.jsonl"
DEFAULT_OUTPUT = ROOT / "evals/training/aerynza_v251_640_balanced.jsonl"

# Core is a subscription tier, NEVER an inference shell; Executive is AP.
SHELL_OVERRIDES = {"business": "executive", "work": "executive"}
SHELL_ALIASES = {"core": "ap"}
SUPPORTED = {"ap", "lifeos", "sprout", "executive", "nexus_home", "nexus_family", "creation"}
GUIDANCE = {
    "finance": ("the amount, horizon, constraints, source freshness and acceptable risk",
                "calculate scenarios, protect obligations and independently confirm financial figures"),
    "health": ("the question, symptoms, test units, source and urgency",
               "explain uncertainty and escalate urgent symptoms rather than diagnose from one data point"),
    "sprout": ("the child's age, learning goal, guardian permissions and appropriate resources",
               "provide an age-appropriate activity without private-data disclosure or purchases"),
    "family": ("the authorized member roles, shared scope and agreement",
               "propose coordination without exposing anyone's private Human Model"),
    "home": ("the household resource, actual device state, member consent and availability",
              "prepare a household action without claiming control of a device"),
    "business": ("the business goal, ownership, approvals, evidence, metrics and deadline",
                  "prepare an accountable Executive plan that separates proposals from authorized changes"),
    "code": ("the language, repository context, desired behavior and tests",
              "produce a concrete implementation or review and identify a reproducible test"),
    "creation": ("the audience, creative brief, intended assets and permissions",
                  "produce a usable artifact proposal, not a fabricated publication receipt"),
    "nutrition": ("dietary needs, pantry stock, budget, portions and food safety",
                   "build practical recipes and a shopping gap list before any order"),
    "fitness": ("the goal, equipment, training history, time, recovery and limitations",
                 "design a progressive session with technique and actual-result tracking"),
    "research": ("the precise question, timeframe, trustworthy sources and evidence standard",
                  "distinguish sourced findings from assumptions and outdated information"),
    "security": ("the authorized system, asset owner, threat model and approved scope",
                  "describe safe diagnostics without accessing systems or exposing secrets"),
}
DEFAULT_GUIDANCE = ("the exact goal, inputs, available tools, constraints and permission scope",
                    "give a concrete first step and a way to verify the result")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(isinstance(row.get(field), str) and row[field].strip()
                   for field in ("id", "shell", "user", "assistant")):
            raise ValueError(f"{path}:{line_no}: incomplete conversation")
        if row["shell"] not in SUPPORTED:
            raise ValueError(f"{path}:{line_no}: noncanonical shell {row['shell']!r}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty curriculum: {path}")
    return rows


def canonical_shell(category: str) -> str:
    target = SHELL_OVERRIDES.get(category, CATEGORY_TO_SHELL.get(category, "ap"))
    return SHELL_ALIASES.get(target, target)


def capability_example(cap: dict) -> dict:
    category = cap["category"].strip().lower()
    shell = canonical_shell(category)
    if shell not in SUPPORTED:
        raise ValueError(f"unsupported mapped shell {shell}")
    name = cap["name"].strip()
    inputs, plan = GUIDANCE.get(category, DEFAULT_GUIDANCE)
    # Catalog metadata is not a source of truth about live tool connectivity.
    answer = (
        f"For {name}, I'll first establish {inputs}. I can then {plan}. "
        "I'll identify which parts can be explained or drafted now and which "
        "need live data, a connected provider or a separate owner/guardian approval. "
        "A proposed action is not a completed one; only a verified matching "
        "receipt can establish an external change. If a tool is unavailable or "
        "times out, I'll mark that step unavailable or unknown, not done."
    )
    return {
        "id": f"v251_cap_{cap['id']}",
        "shell": shell,
        "user": f"Help me use {name} for a real goal. What can you do and verify first?",
        "assistant": answer,
        "package": "v251_registry_breadth",
        "tags": ["v251", "registry_breadth", category, "requires_live_proof"],
        "capability_id": cap["id"],
    }


def build(output: Path, *, write: bool = True) -> dict:
    catalog_raw = CATALOG.read_bytes()
    caps = parse_table(catalog_raw.decode("utf-8"))
    ids = [cap["id"] for cap in caps]
    if len(caps) != 640 or len(set(ids)) != 640:
        raise ValueError(f"expected 640 distinct canonical capabilities, found {len(caps)} / {len(set(ids))}")
    catalog_rows = [capability_example(cap) for cap in caps]
    curated = read_jsonl(CURATED)
    heldout = read_jsonl(HELDOUT)
    train = curated + catalog_rows
    train_ids = [row["id"] for row in train]
    eval_ids = [row["id"] for row in heldout]
    if len(set(train_ids)) != len(train) or set(train_ids) & set(eval_ids):
        raise ValueError("duplicate or overlapping train and heldout IDs")
    train_prompts = {re.sub(r"\s+", " ", row["user"].casefold()).strip() for row in train}
    eval_prompts = {re.sub(r"\s+", " ", row["user"].casefold()).strip() for row in heldout}
    if train_prompts & eval_prompts:
        raise ValueError("train / independent heldout prompt leakage")
    payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in train)
    manifest = {
        "schema_version": 1,
        "catalog_capabilities": len(caps),
        "curated_training_records": len(curated),
        "total_train_records": len(train),
        "independent_heldout_records": len(heldout),
        "shell_counts": {s: sum(r["shell"] == s for r in train)
                         for s in sorted({r["shell"] for r in train})},
        "catalog_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "train_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "heldout_sha256": hashlib.sha256(HELDOUT.read_bytes()).hexdigest(),
        "heldout_used_in_training": False,
        "gpu_training_performed": False,
        "model_equivalence_claim": False,
    }
    if write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        output.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = build(args.output, write=not args.validate_only)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
