#!/usr/bin/env bash
# Aerynza v253 candidate launcher. Default is CPU-only package preflight.
# No hosted API calls, deploy, parent selection, provider actions or auto-promotion.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---preflight}"
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TRAIN="evals/training/aerynza_v253_gpu_mix_train.jsonl"
HELDOUT="evals/training/aerynza_v253_independent_heldout.jsonl"
OUTPUT="${AERYNZA_V253_OUTPUT_DIR:-checkpoints/aerynza_v253_self_reliance_candidate}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

python3 scripts/validate_v251_packages.py
python3 scripts/validate_v252_packages.py
python3 scripts/validate_v253_packages.py
python3 scripts/build_v253_gpu_mix.py
python3 scripts/train_qwen_ascension_lora.py \
  --model "$MODEL" --model-revision "$REVISION" \
  --curriculum "$TRAIN" --eval-curriculum "$HELDOUT" \
  --output-dir "$OUTPUT" --epochs 1.0 --learning-rate 2e-6 \
  --max-length 1024 --seed 3407 --validate-only

if [[ "$MODE" == "--preflight" ]]; then
  echo "[v253] Dataset and trainer schema preflight done; no GPU training or evaluation was run."
  echo "[v253] Human review of all train/heldout items and independent benchmarks is still required."
  exit 0
fi

if [[ "$MODE" == "--train" ]]; then
  if [[ "${AERYNZA_V253_DATA_REVIEW_APPROVED:-}" != "YES" ]]; then
    echo "[v253] BLOCKED: independent human dataset approval required" >&2
    exit 3
  fi
  if [[ -e "$OUTPUT" && -n "$(ls -A "$OUTPUT" 2>/dev/null)" ]]; then
    echo "[v253] BLOCKED: checkpoint directory is not empty; choose a new name" >&2
    exit 4
  fi
  if [[ -n "${AERYNZA_V253_PARENT_ADAPTER:-}" ]]; then
    if [[ "${AERYNZA_V253_PARENT_GATE_APPROVED:-}" != "YES" || ! -f "${AERYNZA_V253_PARENT_ADAPTER}/adapter_config.json" ]]; then
      echo "[v253] BLOCKED: parent adapter must exist and pass independent gate review" >&2
      exit 5
    fi
    PARENT=(--resume-adapter "$AERYNZA_V253_PARENT_ADAPTER")
  else
    PARENT=()
  fi
  # Fail early on an unsupported memory size instead of launching an OOM run.
  python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("BLOCKED: CUDA GPU not found")
vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"[v253] CUDA VRAM: {vram:.1f} GiB")
if vram < 12:
    raise SystemExit("BLOCKED: default LoRA profile requires >=12 GiB VRAM; select a reviewed low-memory profile")
PY
  echo "[v253] Starting one LoRA candidate; this does not authorize promotion."
  CUDA_VISIBLE_DEVICES="$GPU" python3 -u scripts/train_qwen_ascension_lora.py \
    --model "$MODEL" --model-revision "$REVISION" \
    --curriculum "$TRAIN" --eval-curriculum "$HELDOUT" \
    --output-dir "$OUTPUT" --epochs 1.0 --learning-rate 2e-6 \
    --max-length 1024 --seed 3407 "${PARENT[@]}"
  echo "[v253] Candidate trained; independent quality, safety, 640-ID evidence and human gates remain mandatory."
  exit 0
fi

if [[ "$MODE" == "--evaluate" ]]; then
  if [[ ! -f "$OUTPUT/adapter_config.json" ]]; then
    echo "[v253] BLOCKED: no saved adapter to evaluate" >&2
    exit 6
  fi
  mkdir -p evals/results
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluate_qwen_integrated_runtime.py \
    --model "$MODEL" --adapter "$OUTPUT" --tokens 128 \
    --output evals/results/v253_integrated_runtime.json
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluate_qwen_replacement_quality.py \
    --model "$MODEL" --adapter "$OUTPUT" --tokens 220 \
    --output evals/results/v253_replacement_quality.json
  CUDA_VISIBLE_DEVICES="$GPU" python3 scripts/evaluate_per_capability.py \
    --model "$MODEL" --model-revision "$REVISION" --adapter "$OUTPUT" --tokens 128 \
    --capability-gates evals/per_capability_gates.json --output evals/results/v253_per_capability.json
  python3 scripts/semantic_release_gate_v2.py \
    --report evals/results/v253_per_capability.json \
    --capability-gates evals/per_capability_gates.json \
    --output evals/results/v253_semantic_gate.json
  echo "[v253] Automatic gates were executed; real task fixtures and human review are STILL required."
  exit 0
fi

echo "Usage: bash scripts/launch_v253_self_reliance.sh [--preflight|--train|--evaluate]" >&2
exit 2
