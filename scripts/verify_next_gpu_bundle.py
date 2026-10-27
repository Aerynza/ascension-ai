"""Verify a compiled next-GPU curriculum and optional human review receipt.

This is a CPU-only integrity/review gate, not a quality evaluation or approval
to replace production. No model execution, external API or mutation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .build_next_gpu_packs import ROOT, build
except ImportError:
    from build_next_gpu_packs import ROOT, build


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle: Path, review_receipt: Path | None = None) -> dict:
    expected_rows, expected = build()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"compiled training manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "version", "base_model", "base_model_revision", "recommended_method",
        "training_examples", "capability_catalog_ids", "catalog_training_coverage",
        "human_review_required", "production_replacement_ready", "source_sha256",
        "source_exclusions", "package_counts", "shell_counts", "heldout_prompt_count"
    ):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"manifest changed or stale at {key}")
    if len(expected_rows) != manifest["training_examples"] or manifest["catalog_training_coverage"] != 640:
        raise ValueError("training and capability catalog mismatch")
    files = manifest.get("generated", {})
    if set(files) != set(expected["package_counts"]):
        raise ValueError("manifest missing or extra package files")
    for package, item in files.items():
        path = ROOT / item["path"]
        if path.resolve().parent != bundle.resolve() or path.suffix != ".jsonl":
            raise ValueError("manifest package path points outside bundle")
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"package checksum failed: {package}")
        if item["rows"] != expected["package_counts"][package]:
            raise ValueError(f"package count changed: {package}")
    out = {
        "valid": True,
        "bundle_manifest_sha256": sha256(manifest_path),
        "training_records": len(expected_rows),
        "catalog_coverage": 640,
        "gpu_trained": False,
        "promotion_approved": False,
        "human_review_pending": review_receipt is None,
    }
    if review_receipt:
        review = json.loads(review_receipt.read_text(encoding="utf-8"))
        if (review.get("approved") is not True
            or review.get("bundle_manifest_sha256") != out["bundle_manifest_sha256"]
            or not str(review.get("reviewed_by") or "").strip()
            or not str(review.get("reviewed_at") or "").strip()):
            raise ValueError("human review approval is missing, invalid or for different training bytes")
        out["human_review_pending"] = False
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="training_packs/next_gpu/generated")
    parser.add_argument("--review-receipt", default=None)
    args = parser.parse_args()
    directory = (ROOT / args.bundle).resolve()
    receipt = Path(args.review_receipt).resolve() if args.review_receipt else None
    print(json.dumps(verify(directory, receipt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
