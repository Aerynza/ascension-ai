"""Fail-closed checkpoint identity and restoration preflight for Aerynza Qwen LoRA.

Starting a new curriculum from reviewed adapter weights is **not** exact
interruption recovery. Exact recovery must restore the optimizer, scheduler,
RNG and Trainer global step with identical data/configuration. This module
checks prerequisites before allocating the GPU; Trainer performs restoration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class CheckpointContractError(ValueError):
    pass


def contract_payload(*, base_model: str, revision: str, parent_adapter: str | None,
                     curriculum: dict[str, Any], eval_curriculum: dict[str, Any] | None,
                     max_length: int, epochs: float, learning_rate: float,
                     seed: int, save_steps: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_model": base_model,
        "revision": revision,
        "parent_adapter": parent_adapter,
        "curriculum_sha256": curriculum["sha256"],
        "curriculum_files": curriculum["files"],
        "curriculum_records": curriculum["records"],
        "independent_eval_sha256": (eval_curriculum or {}).get("sha256"),
        "eval_files": (eval_curriculum or {}).get("files"),
        "max_length": max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "save_steps": save_steps,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "scheduler": "cosine",
        "balanced_sampler": "fixed_seed_weighted_per_package_shell",
        "assistant_only_loss": True,
    }


def contract_hash(contract: dict[str, Any]) -> str:
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prepare_run(output_dir: Path, contract: dict[str, Any],
                resume_checkpoint: Path | None = None) -> dict[str, Any]:
    """Refuse nonempty output unless this is an exact, validated resume."""
    output_dir = output_dir.resolve()
    state_path = output_dir / "aerynza_exact_resume_contract.json"
    if resume_checkpoint is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise CheckpointContractError(
                "Existing output directory is nonempty. Use --resume-checkpoint "
                "with a verified complete checkpoint, or choose a new output directory."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return {"mode": "new_curriculum_stage", "contract_sha256": contract_hash(contract)}

    resume_checkpoint = resume_checkpoint.resolve()
    if resume_checkpoint.parent != output_dir:
        raise CheckpointContractError("Resume checkpoint must belong to the exact output directory")
    if not state_path.is_file():
        raise CheckpointContractError("Resume requested but the original run contract is missing")
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    if saved != contract:
        raise CheckpointContractError("Resume configuration, curriculum or base model identity mismatch")
    if not resume_checkpoint.is_dir() or not resume_checkpoint.name.startswith("checkpoint-"):
        raise CheckpointContractError("Resume requested but checkpoint directory does not exist")
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    missing = [name for name in required if not (resume_checkpoint / name).is_file()]
    if not any(resume_checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if not (resume_checkpoint / "adapter_model.safetensors").is_file() and not (
        resume_checkpoint / "pytorch_model.bin"
    ).is_file():
        missing.append("adapter_model.safetensors or pytorch_model.bin")
    if missing:
        raise CheckpointContractError("Incomplete optimizer/scheduler/RNG/model checkpoint: " + ", ".join(missing))
    trainer_state = json.loads((resume_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    step = trainer_state.get("global_step")
    if not isinstance(step, int) or step <= 0:
        raise CheckpointContractError("Checkpoint is missing a positive global_step")
    # Do not accept checkpoint-500 with metadata claiming step 1.
    if resume_checkpoint.name != f"checkpoint-{step}":
        raise CheckpointContractError("Checkpoint name and global step disagree")
    return {"mode": "exact_resume", "global_step": step, "contract_sha256": contract_hash(contract)}
