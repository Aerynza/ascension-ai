#!/usr/bin/env bash
# Aerynza v256 GPU handoff. Default = CPU-only validation, never training.
# Preserves base and exact optimizer/scheduler/RNG resume contract.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---preflight}"
MODEL="Qwen/Qwen3-1.7B"
REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BUNDLE="${AERYNZA_V256_BUNDLE_DIR:-training_packs/next_gpu/bundles/v256}"
TRAIN="${BUNDLE}/train/*.jsonl"
EVAL="${BUNDLE}/eval/independent_supervised.jsonl"
OUTPUT="${AERYNZA_V256_OUTPUT_DIR:-}"

case "$MODE" in
  --preflight|--train|--resume|--evaluate) ;;
  *) echo "Usage: $0 [--preflight|--train|--resume|--evaluate]" >&2; exit 2 ;;
esac

python3 -m scripts.validate_v256_production_packs
python3 -m scripts.prepare_v256_gpu_bundle --output-root "$BUNDLE" --build
python3 scripts/train_qwen_ascension_lora.py \
  --model "$MODEL" --model-revision "$REVISION" \
  --curriculum "$TRAIN" --eval-curriculum "$EVAL" \
  --output-dir "${OUTPUT:-checkpoints/v256_preflight_only}" \
  --epochs 1.0 --learning-rate 2e-6 --seed 3407 \
  --max-length 1024 --save-steps 2 --validate-only

if [[ "$MODE" == "--preflight" ]]; then
  echo "[v256] CPU-only integrity preflight done. No model trained or deployed."
  echo "[v256] Review every training case and independent benchmark before GPU training."
  exit 0
fi
if [[ -z "$OUTPUT" ]]; then
  echo "[v256] BLOCKED: set AERYNZA_V256_OUTPUT_DIR on a persistent mounted checkpoint volume." >&2
  exit 3
fi
if [[ "${AERYNZA_V256_DATA_REVIEW_APPROVED:-}" != "YES" ]]; then
  echo "[v256] BLOCKED: independent review of data and heldout split required." >&2
  exit 4
fi
if [[ "${AERYNZA_V256_PERSISTENT_STORAGE_APPROVED:-}" != "YES" ]]; then
  echo "[v256] BLOCKED: persistent checkpoint storage and backup confirmation required." >&2
  exit 5
fi
if [[ "$MODE" != "--evaluate" ]]; then
  python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("BLOCKED: CUDA GPU not found; never silently train on CPU")
memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
print(f"[v256] Available GPU: {memory:.1f} GiB VRAM")
if memory < 12:
    raise SystemExit("BLOCKED: profile requires >=12 GiB; review a separate low-memory configuration")
PY
fi

if [[ "$MODE" == "--evaluate" ]]; then
  if [[ ! -f "$OUTPUT/adapter_config.json" ]]; then
    echo "[v256] BLOCKED: no trained adapter. Cannot evaluate non-existent model." >&2
    exit 6
  fi
  echo "[v256] Evaluation requires independent executable benchmarks and reviewer sign-off."
  python3 scripts/evaluate_qwen_replacement_quality.py --model "$MODEL" --adapter "$OUTPUT" \
    --tokens 220 --output evals/results/v256_replacement_quality.json
  python3 scripts/evaluate_per_capability.py --model "$MODEL" --model-revision "$REVISION" \
    --adapter "$OUTPUT" --tokens 128 --capability-gates evals/per_capability_gates.json \
    --output evals/results/v256_per_capability.json
  python3 scripts/semantic_release_gate_v2.py --report evals/results/v256_per_capability.json \
    --capability-gates evals/per_capability_gates.json \
    --output evals/results/v256_semantic_gate.json
  echo "[v256] Scoring does NOT authorize production replacement or pass unrun provider fixtures."
  exit 0
fi

if [[ "$MODE" == "--train" ]]; then
  if [[ -d "$OUTPUT" && -n "$(ls -A "$OUTPUT" 2>/dev/null)" ]]; then
    echo "[v256] BLOCKED: candidate output exists. Use --resume with complete matching state." >&2
    exit 7
  fi
  PARENT=()
  if [[ -n "${AERYNZA_V256_PARENT_ADAPTER:-}" ]]; then
    if [[ "${AERYNZA_V256_PARENT_GATE_APPROVED:-}" != "YES" ||
          ! -f "$AERYNZA_V256_PARENT_ADAPTER/adapter_config.json" ]]; then
      echo "[v256] BLOCKED: parent adapter needs verified independent gates and adapter_config." >&2
      exit 8
    fi
    PARENT=(--resume-adapter "$AERYNZA_V256_PARENT_ADAPTER")
  fi
  echo "[v256] Starting a NEW candidate training stage only; no automatic promotion."
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u scripts/train_qwen_ascension_lora.py \
    --model "$MODEL" --model-revision "$REVISION" \
    --curriculum "$TRAIN" --eval-curriculum "$EVAL" --output-dir "$OUTPUT" \
    --epochs 1.0 --learning-rate 2e-6 --seed 3407 --max-length 1024 --save-steps 2 \
    "${PARENT[@]}"
  exit 0
fi

if [[ "$MODE" == "--resume" ]]; then
  if [[ -n "${AERYNZA_V256_PARENT_ADAPTER:-}" ]]; then
    echo "[v256] BLOCKED: exact resume and parent-adapter stage selection are different." >&2
    exit 9
  fi
  CKPT="$(python3 -m scripts.select_v256_exact_resume --output-dir "$OUTPUT" \
          --bundle-dir "$BUNDLE" --checkpoint "${AERYNZA_V256_RESUME_CHECKPOINT:-auto}" --path-only)"
  echo "[v256] Exact resume checkpoint verified: $CKPT"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u scripts/train_qwen_ascension_lora.py \
    --model "$MODEL" --model-revision "$REVISION" \
    --curriculum "$TRAIN" --eval-curriculum "$EVAL" --output-dir "$OUTPUT" \
    --epochs 1.0 --learning-rate 2e-6 --seed 3407 --max-length 1024 --save-steps 2 \
    --resume-checkpoint "$CKPT"
fi
