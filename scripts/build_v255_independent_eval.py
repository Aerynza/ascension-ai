"""Compile a truly independent, multi-domain Qwen evaluation split.

Never imported into the training compiler. The generated heldout union is
provided only to Trainer as eval_dataset and to later independent gate runs.
Reject prompt/ID overlap and refuse to silently replace changed receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_next_gpu_packs import ROOT, SHELL_ALIASES, SUPPORTED, build, normalize, read_jsonl

SOURCES = (
    ROOT / "evals/training/aerynza_v251_foundation_heldout.jsonl",
    ROOT / "evals/training/aerynza_v253_independent_heldout.jsonl",
    ROOT / "training_packs/next_gpu/cross_domain_v254_heldout.jsonl",
    ROOT / "training_packs/next_gpu/v255_production_parity_heldout.jsonl",
)
DEFAULT_OUTPUT = ROOT / "training_packs/next_gpu/evaluation/v255_independent_union.jsonl"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def compile_heldout() -> tuple[list[dict], dict]:
    training, _ = build()
    training_prompts = {normalize(row["user"]) for row in training}
    training_ids = {row["id"] for row in training}
    seen_prompts: set[str] = set()
    seen_ids: set[str] = set()
    cases: list[dict] = []
    rubric_only_ids: list[str] = []
    sources = {}
    for path in SOURCES:
        raw = path.read_bytes()
        sources[path.relative_to(ROOT).as_posix()] = sha256(raw)
        for record in read_jsonl(path):
            prompt = normalize(record.get("user", ""))
            key = str(record.get("id", "")).strip()
            shell = SHELL_ALIASES.get(record.get("shell"), record.get("shell"))
            if not prompt or not key or shell not in SUPPORTED:
                raise ValueError(f"invalid heldout case {path.name}:{key}")
            if prompt in training_prompts or key in training_ids:
                raise ValueError(f"heldout leakage into training: {key}")
            if prompt in seen_prompts or key in seen_ids:
                raise ValueError(f"duplicate independent heldout example: {key}")
            seen_prompts.add(prompt)
            seen_ids.add(key)
            # v254 contains rubric-only behavioral cases without an assistant
            # answer. They are reserved for external execution scoring, not
            # fed as supervised Trainer eval targets or synthesized from rubrics.
            if not isinstance(record.get("assistant"), str) or not record["assistant"].strip():
                if not record.get("expected_behaviors") or not record.get("must_not_train"):
                    raise ValueError(f"heldout lacks answer and explicit external rubric: {key}")
                rubric_only_ids.append(key)
                continue
            cases.append({**record, "shell": shell, "package": "independent_holdout"})
    if len(cases) < 60:
        raise ValueError(f"expected multi-domain heldout coverage, found {len(cases)}")
    manifest = {
        "schema_version": 1,
        "purpose": "evaluation_only_never_train",
        "case_count": len(cases),
        "rubric_only_benchmark_count": len(rubric_only_ids),
        "rubric_only_case_ids": rubric_only_ids,
        "train_prompt_overlap": 0,
        "source_sha256": sources,
        "model_trained": False,
    }
    return cases, manifest


def write_evaluation(output: Path) -> dict:
    cases, manifest = compile_heldout()
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in cases)
    manifest["output_sha256"] = sha256(raw.encode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and sha256(output.read_bytes()) != manifest["output_sha256"]:
        raise ValueError("existing evaluation file differs from rebuilt dataset; refusing to overwrite")
    output.write_text(raw, encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest = compile_heldout()[1] if args.validate_only else write_evaluation(args.output)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
