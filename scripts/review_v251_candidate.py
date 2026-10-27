"""Generate evidence for human comparison on the independent v251 heldout set.

Output deliberately has no automatic production-pass flag: a plausible string
is not a verified action, correct answer, or deployment approval.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_qwen_ascension_lora import directory_sha256, generate
from scripts.train_qwen_ascension_lora import load_curriculum
PINNED_MODEL = "Qwen/Qwen3-1.7B"
PINNED_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

def main() -> int:
    parser = argparse.ArgumentParser(description="v251 independent review; no auto-promotion")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="evals/results/v251_heldout_model_review.json")
    parser.add_argument("--tokens", type=int, default=240)
    args = parser.parse_args()
    adapter = ROOT / args.adapter
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"missing reviewed adapter: {adapter}")
    rows, dataset_receipt = load_curriculum(
        "evals/training/aerynza_v251_foundation_heldout.jsonl", minimum_records=12
    )
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        PINNED_MODEL, revision=PINNED_REVISION,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    reviews = []
    for row in rows:
        try:
            output = generate(model, tokenizer, row["shell"], row["user"], args.tokens)
            error = None
        except Exception as exc:
            output, error = "", f"{type(exc).__name__}: {exc}"
        reviews.append({
            "id": row["id"],
            "shell": row["shell"],
            "prompt": row["user"],
            "reference_response": row["assistant"],
            "model_response": output,
            "generation_error": error,
            "human_review": {
                "correctness": None, "grounding": None, "receipt_truth": None,
                "privacy_and_scope": None, "safety": None, "helpfulness": None,
                "approved": None, "notes": ""
            }
        })
    result = {
        "version": "v251",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": PINNED_MODEL,
        "model_revision": PINNED_REVISION,
        "adapter": str(adapter),
        "adapter_sha256": directory_sha256(adapter),
        "dataset_receipt": dataset_receipt,
        "cases": reviews,
        "generation_errors": sum(bool(r["generation_error"]) for r in reviews),
        "human_review_required": True,
        "automatic_promotion": False,
        "production_replacement_ready": False,
        "status": "blocked_pending_independent_review"
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "cases": len(reviews),
                      "generation_errors": result["generation_errors"],
                      "production_replacement_ready": False}, indent=2))
    return 2 if result["generation_errors"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
