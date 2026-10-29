#!/usr/bin/env python3
"""Choose only a complete, compatible v256 Trainer checkpoint.

A parent PEFT adapter is a new training stage, not an exact resume. Selection
checks dataset/config identity and required optimizer, scheduler, RNG, and
step state, and never falls back to step zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_next_gpu_packs import ROOT
from scripts.qwen_resume_contract import contract_payload, prepare_run, CheckpointContractError
from scripts.train_qwen_ascension_lora import load_curriculum
from scripts.prepare_v256_gpu_bundle import MODEL, REVISION, DEFAULT_BUNDLE


def select(output: Path, bundle: Path, *, requested: str = "auto",\n           parent_adapter: str | None = None) -> dict:
    root = ROOT.resolve()
    out = (output if output.is_absolute() else root / output).resolve()
    data = (bundle if bundle.is_absolute() else root / bundle).resolve()
    if not out.is_dir():
        raise FileNotFoundError("Output directory absent; exact resume cannot start at step zero")
    train_path = data.relative_to(root).as_posix() + "/train/*.jsonl"
    eval_path = data.relative_to(root).as_posix() + "/eval/independent_supervised.jsonl"
    _, train = load_curriculum(train_path)
    _, heldout = load_curriculum(eval_path, minimum_records=8)
    contract = contract_payload(
        base_model=MODEL, revision=REVISION, parent_adapter=parent_adapter,
        curriculum=train, eval_curriculum=heldout, max_length=1024,
        epochs=1.0, learning_rate=2e-6, seed=3407, save_steps=2,
    )
    candidates = [out / requested] if requested != "auto" else sorted(
        (p for p in out.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()),
        key=lambda p: int(p.name[11:]), reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No checkpoints found; refusing silent restart")
    errors = []
    for checkpoint in candidates:
        if checkpoint.parent != out:
            errors.append("Checkpoint outside requested output directory")
            continue
        if not (checkpoint / "adapter_config.json").is_file():
            errors.append(f"{checkpoint.name}: missing adapter_config.json")
            continue
        try:
            checked = prepare_run(out, contract, checkpoint)
            return {"checkpoint": checkpoint.relative_to(root).as_posix() if
                    checkpoint.is_relative_to(root) else str(checkpoint),
                    "global_step": checked["global_step"],
                    "mode": "exact_resume", "contract_sha256": checked["contract_sha256"]}
        except (ValueError, OSError, CheckpointContractError, json.JSONDecodeError) as error:
            errors.append(f"{checkpoint.name}: {error}")
    raise ValueError("No compatible full-state checkpoint; " + "; ".join(errors[:4]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", default="auto")\n    parser.add_argument("--parent-adapter", default=None)
    parser.add_argument("--path-only", action="store_true")
    args = parser.parse_args()
    selection = select(args.output_dir, args.bundle_dir, requested=args.checkpoint,\n                       parent_adapter=args.parent_adapter)
    print(selection["checkpoint"] if args.path_only else json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
