"""Validate and discover resumable Aerynza Trainer/PEFT checkpoints.

No ML dependencies: preflight can run on a CPU-only laptop. This module
deliberately refuses incomplete/mismatched states rather than restarting.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CONTRACT_FILE = "aerynza_run_contract.json"
CHECKPOINT_MANIFEST = "aerynza_checkpoint_manifest.json"


def sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_contract(args: Any, train: dict[str, Any], heldout: dict[str, Any] | None) -> dict[str, Any]:
    """All inputs that must remain unchanged for an exact optimizer resume."""
    return {
        "schema": 1,
        "model": args.model,
        "model_revision": args.model_revision,
        "curriculum_sha256": train["sha256"],
        "heldout_sha256": heldout["sha256"] if heldout else None,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "sampler": "balanced-weighted-v1",
        "lora": "r16-alpha32-dropout005-v1",
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    try:
        staged.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staged, path)
    finally:
        if staged.exists():
            staged.unlink()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_run(output: Path, contract: dict[str, Any], resume: bool) -> None:
    """No silent overwrite, dataset drift or fallback to step zero."""
    contract_file = output / CONTRACT_FILE
    if resume:
        if not contract_file.is_file():
            raise ValueError("no run contract found; cannot resume safely")
        saved = _load(contract_file)
        if saved != contract:
            raise ValueError("training configuration or dataset differs from saved run; use a new output directory and explicit --resume-adapter instead")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite previous training evidence: {output}")
        atomic_json(contract_file, contract)


def checkpoint_errors(path: Path, contract: dict[str, Any]) -> list[str]:
    errors = []
    if not path.is_dir() or not path.name.startswith("checkpoint-"):
        return ["not a checkpoint directory"]
    if (path / "checkpoint-is-incomplete.txt").exists():
        errors.append("incomplete checkpoint sentinel")
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json", CHECKPOINT_MANIFEST):
        if not (path / name).is_file():
            errors.append(f"missing {name}")
    if not ((path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()):
        errors.append("missing adapter weights")
    if not ((path / "rng_state.pth").is_file() or list(path.glob("rng_state_*.pth"))):
        errors.append("missing RNG state")
    if errors:
        return errors
    try:
        manifest = _load(path / CHECKPOINT_MANIFEST)
        state = _load(path / "trainer_state.json")
        step = int(path.name.removeprefix("checkpoint-"))
        if manifest.get("contract_sha256") != sha256_json(contract):
            errors.append("checkpoint contract mismatch")
        if int(manifest.get("global_step", -1)) != step or int(state.get("global_step", -1)) != step:
            errors.append("checkpoint step mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        errors.append("corrupt checkpoint metadata")
    return errors


def write_checkpoint_manifest(path: Path, contract: dict[str, Any]) -> None:
    """Called after Trainer finishes writing a complete checkpoint."""
    # File presence must be checked before the manifest is committed.
    expected = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json")
    missing = [name for name in expected if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"checkpoint state missing at save: {missing}")
    if not ((path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()):
        raise RuntimeError("checkpoint adapter weights missing")
    if not ((path / "rng_state.pth").exists() or list(path.glob("rng_state_*.pth"))):
        raise RuntimeError("checkpoint RNG state missing")
    step = int(path.name.removeprefix("checkpoint-"))
    state = _load(path / "trainer_state.json")
    if int(state.get("global_step", -1)) != step:
        raise RuntimeError("checkpoint trainer state not finished")
    atomic_json(path / CHECKPOINT_MANIFEST, {
        "global_step": step,
        "contract_sha256": sha256_json(contract),
        "status": "complete_not_promoted",
    })


def select_checkpoint(output: Path, requested: str | None, contract: dict[str, Any]) -> Path | None:
    if not requested:
        return None
    if requested != "auto":
        target = Path(requested).resolve()
        if target.parent != output.resolve():
            raise ValueError("checkpoint must be inside this output directory; copy a verified backup there before resuming")
        errors = checkpoint_errors(target, contract)
        if errors:
            raise ValueError(f"invalid checkpoint {target}: {', '.join(errors)}")
        return target

    if not output.is_dir():
        raise FileNotFoundError("no checkpoint directory to resume; starting from zero requires a new run")
    candidates = sorted(
        (p for p in output.glob("checkpoint-*") if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()),
        key=lambda p: int(p.name.removeprefix("checkpoint-")),
        reverse=True,
    )
    for path in candidates:
        if not checkpoint_errors(path, contract):
            return path
    raise FileNotFoundError("no complete compatible checkpoint found; refusing to restart silently")
