"""Accountable rollout matrix for all 640 Aerynza catalog capabilities.

IMPORTANT: Historical labels ('native_action', 'provider_action', 'domain_prompt')
are source classifications, not proof of production availability or successful
training. Output is an evidence queue and must NOT enter the SFT curriculum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "public/capability_report.json"
CLASSIFICATION = ROOT / "evals/capability_status.json"
DEFAULT_OUTPUT = ROOT / "evals/results/v253_640_execution_map.json"

TRACKS = {
    "domain_prompt": "reasoning_and_domain_expert_eval",
    "native_action": "native_local_tool_and_artifact_eval",
    "provider_action": "signed_provider_connector_and_receipt_eval",
}


def build(catalog: dict, declared: list[dict]) -> dict:
    entries = catalog.get("capabilities")
    if not isinstance(entries, list) or len(entries) != 640 or len({e.get("id") for e in entries}) != 640:
        raise ValueError("Current catalog must have 640 unique identifiers")
    by_id = {s.get("id"): s for s in declared}
    if len(declared) != 640 or len(by_id) != 640 or set(by_id) != {x["id"] for x in entries}:
        raise ValueError("Declared classifications must map exactly once onto every catalog ID")
    rows = []
    for item in sorted(entries, key=lambda r: r["id"]):
        cid = item["id"]
        klass = by_id[cid]
        original_status = klass.get("status")
        if original_status not in TRACKS:
            raise ValueError(f"{cid}: unrecognized historical status {original_status}")
        if item.get("name") is None:
            raise ValueError(f"{cid}: no display name")
        rows.append({
            "capability_id": cid,
            "category": item.get("category"),
            "name": item["name"],
            "historical_status": original_status,
            "historical_kind": klass.get("kind"),
            "legacy_shell_label": item.get("shell"),
            "authorized_shell": "requires_owner_and_resource_review",
            "curriculum_track": TRACKS[original_status],
            "model_response_gate": "not_run",
            "executable_tool_schema_gate": "not_run",
            "authenticated_signed_actor_gate": "not_run",
            "tier_and_guardian_scope_gate": "not_run",
            "external_provider_fixture_gate": "not_run",
            "device_or_artifact_receipt_gate": "not_run",
            "outcome_recovery_gate": "not_run",
            "heldout_evaluation_gate": "not_run",
            "actual_native_local_inference_gate": "not_run",
            "human_review": "pending",
            "production_usable": False,
            "promotion_blocked": True,
        })
    return {
        "schema_version": "v253_640_evidence_map_v1",
        "catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
        "source_status": "historical_descriptions_not_runtime_proof",
        "cases": rows,
        "case_count": len(rows),
        "historical_classes": dict(sorted(Counter(r["historical_status"] for r in rows).items())),
        "categories": dict(sorted(Counter(str(r["category"]) for r in rows).items())),
        "no_model_parity_claim": True,
        "not_training_data": True,
        "production_replacement_ready": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--classification", type=Path, default=CLASSIFICATION)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    opts = ap.parse_args()
    report = build(json.loads(opts.catalog.read_text()), json.loads(opts.classification.read_text()))
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(opts.output),
        "case_count": report["case_count"],
        "historical_classes": report["historical_classes"],
        "production_replacement_ready": False,
        "not_training_data": True,
    }, indent=2))


if __name__ == "__main__":
    main()
