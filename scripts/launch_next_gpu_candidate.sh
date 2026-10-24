#!/usr/bin/env bash
# Aerynza next-GPU LoRA candidate. Preflight by default; no paid compute,
# model training, production deployment or automatic promotion.
set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE="training_packs/next_gpu/generated"
CURRICULUM="$BUNDLE/*.jsonl"
EVAL="evals/training/aerynza_v251_foundation_heldout.jsonl"
OUTPUT="${AERYNZA_NEXT_GPU_OUTPUT_DIR:-checkpoints/aerynza_next_gpu_candidate}"
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

python3 scripts/validate_v251_packages.py
python3 scripts/validate_v252_packages.py
python3 scripts/build_next_gpu_packs.py --dry-run

if [[ ! -f "$BUNDLE/manifest.json" ]]; then
  python3 scripts/build_next_gpu_packs.py --output "$BUNDLE"
fi
python3 scripts/verify_next_gpu_bundle.py --bundle "$BUNDLE"
python3 scripts/train_qwen_ascension_lora.py \
  --model "$MODEL" --model-revision "$REVISION" \
  --curriculum "$CURRICULUM" --eval-curriculum "$EVAL" \
  --output-dir "$OUTPUT" --epochs 1 --learning-rate 2e-6 \
  --max-length 1024 --save-steps 2 --validate-only

if [[ "${1:-}" != "--run" ]]; then
  echo "[Aerynza] CPU curriculum preflight passed; no GPU run launched."
  echo "[Aerynza] Review, sign bundle manifest, then explicitly run --run on approved hardware."
  exit 0
fi

if [[ "${AERYNZA_NEXT_GPU_TRAINING_APPROVED:-}" != "YES" ]]; then
  echo "[Aerynza] BLOCKED: explicit training approval is required" >&2
  exit 3
fi
if [[ -z "${AERYNZA_NEXT_GPU_REVIEW_RECEIPT:-}" ]]; then
  echo "[Aerynza] BLOCKED: a human review receipt for exact bundle bytes is required" >&2
  exit 4
fi
python3 scripts/verify_next_gpu_bundle.py \
  --bundle "$BUNDLE" --review-receipt "$AERYNZA_NEXT_GPU_REVIEW_RECEIPT"

if [[ -n "${AERYNZA_NEXT_GPU_RESUME:-}" && -n "${AERYNZA_NEXT_GPU_PARENT_ADAPTER:-}" ]]; then
  echo "[Aerynza] BLOCKED: exact checkpoint resume and adapter-stage continuation are distinct" >&2
  exit 5
fi

RESUME_ARGS=()
if [[ -n "${AERYNZA_NEXT_GPU_RESUME:-}" ]]; then
  RESUME_ARGS=(--resume-checkpoint "$AERYNZA_NEXT_GPU_RESUME")
fi
PARENT_ARGS=()
if [[ -n "${AERYNZA_NEXT_GPU_PARENT_ADAPTER:-}" ]]; then
  if [[ ! -f "$AERYNZA_NEXT_GPU_PARENT_ADAPTER/adapter_config.json" ]]; then
    echo "[Aerynza] BLOCKED: reviewed parent adapter path is missing" >&2
    exit 6
  fi
  PARENT_ARGS=(--resume-adapter "$AERYNZA_NEXT_GPU_PARENT_ADAPTER")
fi

echo "[Aerynza] Starting reviewed LoRA CANDIDATE; never auto-promoting."
python3 -u scripts/train_qwen_ascension_lora.py \
  --model "$MODEL" --model-revision "$REVISION" \
  --curriculum "$CURRICULUM" --eval-curriculum "$EVAL" \
  --output-dir "$OUTPUT" --epochs 1 --learning-rate 2e-6 \
  --max-length 1024 --save-steps 2 \
  "${RESUME_ARGS[@]}" "${PARENT_ARGS[@]}"

echo "[Aerynza] Candidate only: no production replacement; independent quality, 640-ID,"
echo "[Aerynza] action-receipt, privacy, child, uptime and release tests still required."
