#!/usr/bin/env bash
# Manual, fail-closed GPU handoff for the original v251 SFT package.
# Default execution is preflight only. Never deploy or replace production.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN="evals/training/aerynza_v251_foundation_train.jsonl"
EVAL="evals/training/aerynza_v251_foundation_heldout.jsonl"
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
OUTPUT="${AERYNZA_V251_OUTPUT_DIR:-checkpoints/aerynza_v251_foundation_candidate}"

python3 scripts/validate_v251_packages.py
python3 scripts/build_v251_capability_validation_queue.py
python3 scripts/train_qwen_ascension_lora.py \
  --model "$MODEL" --model-revision "$REVISION" \
  --curriculum "$TRAIN" --eval-curriculum "$EVAL" \
  --output-dir "$OUTPUT" --epochs 1.0 \
  --learning-rate 2e-6 --max-length 1024 --validate-only

if [[ "${1:-}" != "--run" ]]; then
  echo "[v251] Preflight completed. No GPU job, adapter, or deployment started."
  echo "[v251] To train: review all source data, then run with --run and AERYNZA_V251_REVIEW_APPROVED=YES."
  exit 0
fi
if [[ "${AERYNZA_V251_REVIEW_APPROVED:-}" != "YES" ]]; then
  echo "[v251] BLOCKED: written human approval of the examples is required" >&2
  exit 3
fi
if [[ -e "$OUTPUT" ]] && [[ -n "$(ls -A "$OUTPUT" 2>/dev/null)" ]]; then
  echo "[v251] BLOCKED: output exists; refusing to overwrite checkpoints" >&2
  exit 4
fi
if [[ -n "${AERYNZA_V251_PARENT_ADAPTER:-}" ]]; then
  if [[ ! -d "$AERYNZA_V251_PARENT_ADAPTER" || ! -f "$AERYNZA_V251_PARENT_ADAPTER/adapter_config.json" ]]; then
    echo "[v251] BLOCKED: selected parent adapter not present or invalid" >&2
    exit 5
  fi
  PARENT=(--resume-adapter "$AERYNZA_V251_PARENT_ADAPTER")
else
  PARENT=()
fi
echo "[v251] Starting an explicitly requested candidate run; never auto-promoting."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u scripts/train_qwen_ascension_lora.py \
  --model "$MODEL" --model-revision "$REVISION" \
  --curriculum "$TRAIN" --eval-curriculum "$EVAL" \
  --output-dir "$OUTPUT" --epochs 1.0 --learning-rate 2e-6 \
  --max-length 1024 "${PARENT[@]}"
echo "[v251] Candidate completed; canonical, receipt, 640-category, execution, latency and human gates still required."
