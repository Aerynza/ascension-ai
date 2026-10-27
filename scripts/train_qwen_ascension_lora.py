"""Train a Qwen adapter on the reviewed Aerynza product curriculum.

The trainer masks every system/user token so loss is calculated only on the
approved assistant answer.  It saves a PEFT adapter, never a replacement model,
and emits a receipt that binds the result to the exact curriculum bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .aerynza_checkpoint_resume import (
        build_contract, prepare_run, select_checkpoint, write_checkpoint_manifest,
    )
else:
    from aerynza_checkpoint_resume import (
        build_contract, prepare_run, select_checkpoint, write_checkpoint_manifest,
    )


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SHELLS = {"ap", "lifeos", "sprout", "nexus_home", "nexus_family", "core", "creation"}
SYSTEM_PROMPTS = {
    "ap": (
        "You are Aerynza, AerynzaLife's warm, perceptive personal intelligence. Conversation "
        "comes first. Use context quietly and ask only useful questions. A user statement "
        "is understood conversation context, not saved memory: say it is not saved unless a "
        "memory receipt confirms the write. You may prepare or draft an action, but preparation "
        "is not execution. Never say an action was submitted, paid, scheduled, saved, or completed "
        "without the matching provider receipt. A timeout or missing receipt means status unknown; "
        "check before retrying. Keep partial successes and failures separate."
    ),
    "lifeos": (
        "You are Aerynza operating inside AerynzaLife. Connect relevant life domains and distinguish "
        "known, inferred, prepared, requested, unknown, and confirmed states. Connection does not "
        "prove account or transaction sync; upload does not prove parsing or profile update. You may "
        "prepare actions, but never claim submission, payment, scheduling, saving, or completion "
        "without the matching provider receipt. A timeout means unknown, not success."
    ),
    "sprout": (
        "You are Aerynza AI operating through Aerynza Kids for a child or teen. Be warm, "
        "age-appropriate, guardian-aware, privacy-preserving, and explicit when an action "
        "needs parent permission or a verified receipt."
    ),
    "nexus_home": (
        "You are NexusHome, a privacy-aware household coordinator. Respect guardian, "
        "child, household, and private-memory boundaries and require action receipts."
    ),
    "nexus_family": (
        "You are NexusFamily, a family-enterprise coordinator. Help members coordinate without "
        "exposing private household data. When asked to stay quiet or wait until Nexus is addressed, "
        "acknowledge briefly without repeating the request, then remain silent."
    ),
    "core": (
        "You are Aerynza AI, the shared intelligence core. Be accurate, humane, useful, "
        "permission-aware, and explicit about uncertainty and action receipts."
    ),
    "creation": (
        "You are Aerynza AI in the Creation workspace. Give concrete creative development: propose "
        "a smallest useful version, decisions, and a test instead of restating the user's idea. Respect "
        "ownership and never claim publishing or other external action without a receipt."
    ),
}


def curriculum_files(pattern: str) -> list[Path]:
    files = sorted(ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no curriculum files matched {pattern!r}")
    return files


def load_curriculum(pattern: str, *, minimum_records: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = curriculum_files(pattern)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    digest = hashlib.sha256()
    shells: dict[str, int] = {}
    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"id", "shell", "user", "assistant"} - set(record)
            if missing:
                raise ValueError(f"{path.name}:{line_number} missing {sorted(missing)}")
            record_id = str(record["id"]).strip()
            shell = str(record["shell"]).strip()
            if not record_id or record_id in seen_ids:
                raise ValueError(f"duplicate or empty curriculum id: {record_id!r}")
            if shell not in SUPPORTED_SHELLS:
                raise ValueError(f"unsupported shell {shell!r} in {record_id}")
            if not str(record["user"]).strip() or not str(record["assistant"]).strip():
                raise ValueError(f"empty conversation field in {record_id}")
            history = record.get("history") or []
            if not isinstance(history, list):
                raise ValueError(f"history must be a list in {record_id}")
            for turn_index, turn in enumerate(history):
                if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
                    raise ValueError(f"invalid history turn {turn_index} in {record_id}")
                if not str(turn.get("content") or "").strip():
                    raise ValueError(f"empty history turn {turn_index} in {record_id}")
            seen_ids.add(record_id)
            shells[shell] = shells.get(shell, 0) + 1
            records.append(record)
    if len(records) < minimum_records:
        raise ValueError("curriculum is unexpectedly small; refusing to train")
    return records, {
        "sha256": digest.hexdigest(),
        "records": len(records),
        "shell_counts": dict(sorted(shells.items())),
        "files": [path.relative_to(ROOT).as_posix() for path in files],
    }


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_record(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    prefix_messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[record["shell"]]},
    ]
    prefix_messages.extend({
        "role": str(turn["role"]), "content": str(turn["content"])
    } for turn in (record.get("history") or []))
    prefix_messages.append({"role": "user", "content": str(record["user"])})
    # Build the exact inference prefix first.  Some Qwen templates render the
    # assistant header/thinking marker differently once an assistant message is
    # already present, so separately templating a completed conversation can
    # make the intended loss boundary ambiguous.
    prompt_text = apply_chat_template(
        tokenizer, prefix_messages, tokenize=False, add_generation_prompt=True
    )
    answer_text = str(record["assistant"]).strip() + (tokenizer.eos_token or "")
    prefix_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + answer_text, add_special_tokens=False)["input_ids"]
    prefix_length = len(prefix_ids)
    if full_ids[:prefix_length] != prefix_ids:
        raise ValueError(f"tokenizer changed the verified prompt prefix for {record['id']}")
    if prefix_length == 0 or prefix_length >= len(full_ids):
        raise ValueError(f"could not isolate assistant target for {record['id']}")
    if len(full_ids) > max_length:
        raise ValueError(
            f"{record['id']} requires {len(full_ids)} tokens, above max length {max_length}"
        )
    labels = [-100] * prefix_length + list(full_ids[prefix_length:])
    if all(label == -100 for label in labels):
        raise ValueError(f"no assistant labels for {record['id']}")
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "package": str(record.get("package") or "unclassified"),
        "shell": str(record.get("shell") or "core"),
    }


class AscensionDataset:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]

    @property
    def training_groups(self) -> list[str]:
        return [
            f'{row.get("package") or "unclassified"}:{row.get("shell") or "core"}'
            for row in self.rows
        ]


@dataclass
class AssistantOnlyCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def balanced_group_weights(groups: list[str]) -> list[float]:
    counts: dict[str, int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    return [1.0 / counts[group] for group in groups]


class BalancedPackageTrainerMixin:
    """Sample every curriculum package equally without duplicating records."""

    def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        groups = getattr(self.train_dataset, "training_groups", None)
        if not groups:
            return super()._get_train_sampler(*args, **kwargs)
        weights = torch.tensor(balanced_group_weights(groups), dtype=torch.double)
        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed))
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )


def training_arguments(**kwargs: Any) -> Any:
    from transformers import TrainingArguments

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in parameters and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an Ascension Qwen LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--model-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--output-dir", default="checkpoints/aerynza_qwen3_1_7b_v20_candidate")
    parser.add_argument("--curriculum", default="evals/training/aerynza_v20_balanced_master.jsonl")
    parser.add_argument("--eval-curriculum", default=None,
                        help="independent held-out JSONL pattern; never used for optimization")
    parser.add_argument("--resume-adapter", default=None,
                        help="start a NEW stage from reviewed adapter weights; resets optimizer")
    parser.add_argument("--resume-checkpoint", default=None,
                        help="exactly resume a saved run: auto or checkpoint path inside output-dir")
    parser.add_argument("--save-steps", type=int, default=2,
                        help="persist full resumable state every N optimizer steps")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="explicit override for diagnostic training only")
    args = parser.parse_args(argv)
    if args.resume_adapter and args.resume_checkpoint:
        parser.error("--resume-adapter starts a new training stage; do not combine it with --resume-checkpoint")
    if args.save_steps < 1:
        parser.error("--save-steps must be at least 1")

    records, curriculum = load_curriculum(args.curriculum)
    eval_records = None
    eval_curriculum = None
    if args.eval_curriculum:
        eval_records, eval_curriculum = load_curriculum(args.eval_curriculum, minimum_records=8)
        train_prompts = {str(row["user"]).strip().casefold() for row in records}
        overlap = train_prompts & {str(row["user"]).strip().casefold() for row in eval_records}
        if overlap:
            raise ValueError(f"train/eval prompt overlap: {len(overlap)}")
    print(json.dumps({"validated": True, "curriculum": curriculum,
                      "eval_curriculum": eval_curriculum}, indent=2), flush=True)
    if args.validate_only:
        return 0

    output_dir = (ROOT / args.output_dir).resolve()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, set_seed

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for v20 training; use --allow-cpu for explicitly approved small-model CPU runs")

    contract = build_contract(args, curriculum, eval_curriculum)
    prepare_run(output_dir, contract, resume=bool(args.resume_checkpoint))
    resume_checkpoint = select_checkpoint(output_dir, args.resume_checkpoint, contract)
    if resume_checkpoint:
        print(f"[Aerynza] verified exact training resume: {resume_checkpoint}", flush=True)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_record(tokenizer, record, args.max_length) for record in records]
    random.Random(args.seed).shuffle(encoded)
    if eval_records is not None:
        train_rows = encoded
        eval_rows = [encode_record(tokenizer, record, args.max_length) for record in eval_records]
    else:
        eval_size = max(1, round(len(encoded) * 0.1))
        eval_rows, train_rows = encoded[:eval_size], encoded[eval_size:]

    cuda_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if cuda_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.resume_adapter:
        model = PeftModel.from_pretrained(
            model, ROOT / args.resume_adapter, is_trainable=True
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_args = training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        warmup_steps=2,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        bf16=cuda_bf16,
        fp16=torch.cuda.is_available() and not cuda_bf16,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        save_only_model=False,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    class BalancedPackageTrainer(BalancedPackageTrainerMixin, Trainer):
        pass

    class VerifiedCheckpointCallback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.is_world_process_zero:
                write_checkpoint_manifest(Path(args.output_dir) / f"checkpoint-{state.global_step}", contract)
            return control

    trainer = BalancedPackageTrainer(
        model=model,
        args=train_args,
        train_dataset=AscensionDataset(train_rows),
        eval_dataset=AscensionDataset(eval_rows),
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        callbacks=[VerifiedCheckpointCallback()],
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    if trainer.is_world_process_zero():
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        receipt = {
            "status": "trained_not_promoted",
            "base_model": args.model,
            "base_model_revision": args.model_revision,
            "adapter_dir": (output_dir.relative_to(ROOT).as_posix()
                            if output_dir.is_relative_to(ROOT) else str(output_dir)),
            "resumed_from_adapter": args.resume_adapter,
            "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "global_step": trainer.state.global_step,
            "checkpoint_interval_steps": args.save_steps,
            "checkpoint_resume_contract": contract,
            "curriculum": curriculum,
            "eval_curriculum": eval_curriculum,
            "eval_is_independent": eval_curriculum is not None,
            "train_records": len(train_rows),
            "eval_records": len(eval_rows),
            "assistant_only_loss": True,
            "balanced_package_and_shell_sampling": True,
            "seed": args.seed,
            "max_length": args.max_length,
            "metrics": train_result.metrics,
            "production_replacement_ready": False,
            "human_review_required": True,
        }
        (output_dir / "ascension_training_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
