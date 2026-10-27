"""Generate an unapproved, immutable-by-default human review form for one Aerynza repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.aerynza_training_optimizer import sha256_file
from scripts.evaluate_qwen_ascension_lora import directory_sha256


def main(argv=None):
    p = argparse.ArgumentParser(description="Create a local unapproved repair form")
    p.add_argument("--plan", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--training", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    path = Path(args.output).resolve()
    if path.exists():
        raise FileExistsError("approval forms may not be overwritten")
    parent = Path(args.parent_adapter).resolve()
    if not (parent / "adapter_config.json").is_file():
        raise FileNotFoundError("parent adapter is missing")
    record = {
        "approved": False,
        "reviewer": "",
        "plan_sha256": sha256_file(Path(args.plan).resolve()),
        "parent_adapter_sha256": directory_sha256(parent),
        "training_sha256": sha256_file(Path(args.training).resolve()),
        "heldout_sha256": sha256_file(Path(args.heldout).resolve()),
        "independently_authored_examples": False,
        "no_heldout_or_model_output_leakage": False,
        "scoped_data_permissions_reviewed": False,
        "notes": "Review source, permissions, independent fixtures, model capability and actual failure cause. Turn booleans true ONLY after independent approval.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Local approval template created, NOT approved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
