#!/usr/bin/env python3
"""Evidence-first inventory: a 640-item capability catalog is not an execution engine."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "training/v256/production_parity_gates.json"
CATALOG = ROOT / "public/capability_report.json"
CLASSIFICATIONS = ROOT / "evals/capability_status.json"
REQUIRED = {"id", "expected_function", "required_runtime", "required_proof",
            "implementation_readiness", "evaluation_state", "evidence_uri",
            "measured_success_rate", "owner_review", "production_proven"}


def audit(matrix: dict, catalog: dict, classes: list[dict]) -> dict:
    catalog_rows = catalog.get("capabilities")
    if not isinstance(catalog_rows, list) or len(catalog_rows) != 640:
        raise ValueError("Expected exactly 640 canonical capability entries")
    ids = [x.get("id") for x in catalog_rows]
    declared = [x.get("id") for x in classes]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != 640:
        raise ValueError("Catalog has missing or duplicate capability ID")
    if len(declared) != 640 or set(declared) != set(ids) or len(set(declared)) != 640:
        raise ValueError("Capability classification must match all 640 IDs exactly")
    surfaces = matrix.get("surfaces", [])
    if len(surfaces) != 30 or len({r.get("id") for r in surfaces}) != 30:
        raise ValueError("Must define 30 distinct production comparison surfaces")
    if matrix.get("do_not_claim_model_parity") is not True or matrix.get("do_not_train_on_this_file") is not True:
        raise ValueError("The gap matrix cannot be promoted or trained")
    p = matrix.get("gate_policy", {})
    for field in ("critical_privacy_failures_allowed", "fabricated_completion_allowed",
                  "cross_tenant_leaks_allowed"):
        if type(p.get(field)) is not int or p[field] != 0:
            raise ValueError(f"Unsafe critical release threshold: {field}")
    for field in ("missing_evidence_blocks_release", "human_approval_required",
                  "training_pass_not_sufficient", "model_and_runtime_scored_separately"):
        if p.get(field) is not True:
            raise ValueError(f"Mandatory release safeguard disabled: {field}")
    verified = 0
    for case in surfaces:
        if not REQUIRED <= case.keys() or not all(
            isinstance(case.get(field), str) and case[field].strip()
            for field in ("id", "expected_function", "required_runtime", "required_proof")
        ):
            raise ValueError("Incomplete functional target in parity matrix")
        if case["production_proven"]:
            if not (case["evaluation_state"] == "passed"
                    and case["owner_review"] == "approved"
                    and case["evidence_uri"]
                    and isinstance(case["measured_success_rate"], (int, float))
                    and 0 <= case["measured_success_rate"] <= 1):
                raise ValueError(f"Premature production-proven claim: {case['id']}")
            verified += 1
        elif case["implementation_readiness"] == "verified_production" or case["evaluation_state"] == "passed":
            raise ValueError(f"Contradictory unproven status: {case['id']}")
    return {
        "status": "inventory_only_not_production_certification",
        "catalog_ids": len(ids),
        "functional_surfaces": len(surfaces),
        "verified_production_surfaces": verified,
        "historical_classes": dict(sorted(Counter(x.get("status") for x in classes).items())),
        "unverified_capabilities_not_autopromoted": True,
        "model_parity_proven": False,
        "provider_and_artifact_receipts_checked": False,
        "training_run": False,
    }


def main() -> int:
    report = audit(json.loads(MATRIX.read_text()),
                   json.loads(CATALOG.read_text()),
                   json.loads(CLASSIFICATIONS.read_text()))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
