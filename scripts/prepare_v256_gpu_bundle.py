#!/usr/bin/env python3
"""Compile an immutable, restart-verifiable v256 GPU bundle with independent eval.

No model download, CUDA requirement, tool execution, or promotion. All bundle
writes are content-addressed and refuse changed or unexpected existing files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.build_next_gpu_packs import ROOT, build
from scripts.build_v255_independent_eval import compile_heldout

DEFAULT_BUNDLE = Path("training_packs/next_gpu/bundles/v256")
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def lines(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n" for row in rows).encode("utf-8")


def compile_bundle() -> tuple[dict[str, bytes], dict]:
    training, source_manifest = build()
    evaluation, heldout_meta = compile_heldout()
    by_package: dict[str, list[dict]] = defaultdict(list)
    for row in training:
        by_package[row["package"]].append(row)
    content = {f"train/{group}.jsonl": lines(rows)
               for group, rows in sorted(by_package.items())}
    content["eval/independent_supervised.jsonl"] = lines(evaluation)
    if len(training) < 1760 or len(evaluation) < 80:
        raise ValueError("Expanded v256 dataset is unexpectedly small")
    manifest = {
        "schema_version": "aerynza_immutable_gpu_bundle_v256.1",
        "status": "preflight_only_not_approved_or_trained",
        "model": MODEL,
        "model_revision": REVISION,
        "training_examples": len(training),
        "independent_eval_examples": len(evaluation),
        "registered_capability_ids": source_manifest["catalog_training_coverage"],
        "capability_execution_proven": False,
        "training_authorized": False,
        "production_replacement_ready": False,
        "human_review_required": True,
        "dataset_split": "independent_no_benchmark_training",
        "sampler": "assistant_only_loss_balanced_package_and_shell",
        "source_sha256": source_manifest["source_sha256"],
        "heldout_source_sha256": heldout_meta["source_sha256"],
        "files": {name: {"sha256": sha(raw), "bytes": len(raw),
                         "rows": raw.count(b"\n")}
                  for name, raw in sorted(content.items())},
    }
    content["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return content, manifest


def ensure_bundle(folder: Path, *, write: bool) -> dict:
    folder = folder.resolve()
    if not folder.is_relative_to(ROOT.resolve()):
        raise ValueError("Bundle must reside inside repository for deterministic Trainer glob")
    expected, manifest = compile_bundle()
    # Check existing contents first. Never overwrite a partially changed run
    # or mix old/new curriculum bytes silently.
    if folder.exists():
        unexpected = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()} - set(expected)
        if unexpected:
            raise ValueError(f"Unexpected or stale files in bundle: {sorted(unexpected)[:5]}")
    for name, raw in expected.items():
        target = folder / name
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f"Bundle drift for {name}; use a new bundle path, not overwrite")
        if not write and not target.is_file():
            continue
    if write:
        for name, raw in expected.items():
            target = folder / name
            if target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".partial")
            temp.write_bytes(raw)
            temp.replace(target)
    return {
        **manifest,
        "bundle": folder.relative_to(ROOT.resolve()).as_posix(),
        "mode": "content_verified_and_written" if write else "dry_run_no_files_written",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--build", action="store_true",
                        help="write only content-verified, immutable bundle files")
    args = parser.parse_args()
    folder = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    result = ensure_bundle(folder, write=args.build)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
