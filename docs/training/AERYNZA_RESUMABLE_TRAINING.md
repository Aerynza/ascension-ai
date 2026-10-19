# Aerynza AI: zero-cost resumable training

**State:** proposed code change, not a completed model run, checkpoint migration, cloud deployment, or production approval. This branch extends the existing Qwen3 LoRA trainer. It does **not** create previously missing checkpoints, and it does **not** prove that the 640 listed product capabilities execute.

## Two kinds of continuation

- `--resume-checkpoint auto`: **exact training-run continuation** from the latest complete compatible Trainer checkpoint. Requires unchanged base model/revision, training and held-out bytes, learning rate, epochs, seed, sequence length, and effective training setup. Restores PEFT weights, Trainer step, optimizer, scheduler, RNG, and data progress where supported by the existing Transformers trainer. The most recent unsaved steps may need to be repeated.
- `--resume-adapter PATH`: **next-stage learning** from a selected, independently reviewed LoRA adapter. The learned adapter weights carry forward, but **optimizer/scheduler/global step reset**. Use this when moving from v250 to v251, changing the curriculum, or changing to a different hardware configuration. Always create a **new** candidate output directory. Never automatically pick the newest adapter as the best.

No checkpoint may be silently treated as compatible. The run contract hashes the source training and evaluation datasets. The actual weights and optimizer state are **not** committed to GitHub (which excludes `checkpoints/`).

## Review the training examples first

On the reviewed training branch:
```bash
python3 scripts/validate_v251_packages.py
python3 scripts/train_qwen_ascension_lora.py \
  --curriculum evals/training/aerynza_v251_foundation_train.jsonl \
  --eval-curriculum evals/training/aerynza_v251_foundation_heldout.jsonl \
  --output-dir checkpoints/aerynza_v251_foundation_candidate \
  --validate-only
python3 -m unittest discover -s tests -p 'test_aerynza_checkpoint_resume.py' -v
```
These CPU-only checks are not model training and do not replace independent review of the examples.

## Free laptop CPU experiment (Python 3.11, substantial RAM and disk space)

Install dependencies in a virtual environment and test the machine first. Full 1.7B training on a typical laptop can be impractical. Instead, begin with a **separate** Qwen3-0.6B adapter. This is a smaller **experiment**, not an exact continuation of the 1.7B model. The initial model download requires internet access. Run the command from the repository root, after reviewing the dataset.

```bash
python3 -m pip install -r requirements-training.txt
python3 -u scripts/train_qwen_ascension_lora.py \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --curriculum evals/training/aerynza_v251_foundation_train.jsonl \
  --eval-curriculum evals/training/aerynza_v251_foundation_heldout.jsonl \
  --output-dir checkpoints/aerynza_cpu_0_6b_v251_experiment \
  --epochs 1 --learning-rate 2e-6 --max-length 1024 \
  --save-steps 1 --allow-cpu
```

After an interruption, run the **same command**, adding:
```
  --resume-checkpoint auto
```
In a terminal, append that argument to the original Python command. Don't launch another run into the same candidate directory without a valid checkpoint; it will fail closed. CPU training can be extremely slow. Low-memory laptops may not be able to load the model for training.

If the original session completed and you want a new curriculum or more training epochs, use a **fresh output directory** with `--resume-adapter` pointing at a reviewed saved adapter, not `--resume-checkpoint`.

## Existing GPU v251 training launcher

After independent data review and approval:
```bash
export AERYNZA_V251_REVIEW_APPROVED=YES
bash scripts/launch_v251_foundation.sh --run
```
Resume a pre-empted v251 session:
```bash
export AERYNZA_V251_REVIEW_APPROVED=YES
export AERYNZA_V251_RESUME=auto
bash scripts/launch_v251_foundation.sh --run
```
The v251 launcher is for the **pinned 1.7B model** and still requires a suitable GPU. It uses a two-optimizer-step save interval; the CPU experiment saves every step. Saving costs disk space and time.

## Checkpoint location is critical

- **Laptop:** keep the entire `checkpoints/<run-name>` directory and backed-up copy, including `aerynza_run_contract.json` and all `checkpoint-*/*` files. Checkpoint data is not in GitHub.
- **Google Colab:** ordinary `/content` storage disappears when the runtime is removed. Mount your own persistent Google Drive and supply an absolute mounted `--output-dir`, e.g. `/content/drive/MyDrive/Aerynza/checkpoints/v251`. Writes to Drive can be slower/unreliable than local storage; keep multiple completed checkpoint copies and verify they are readable. Do not assume a session will finish or preserve data.
- **Kaggle:** `/kaggle/working` is not a guaranteed cross-session checkpoint store. Explicitly export/download the complete checkpoint tree before the session ends or use another genuinely durable mounted store. Keeping only `adapter_model.safetensors` is **not** an exact resume.
- **Moving between laptop and GPU:** optimizer/device/mixed-precision formats and library versions can differ. Verify a trial recovery; where exact restoration is incompatible, use adapter continuation in a separate stage and document the reset.

**A safe resumption requires:** `checkpoint-N/adapter_model.safetensors`, `adapter_config.json`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`, RNG state, `aerynza_checkpoint_manifest.json`, plus the parent `aerynza_run_contract.json`. A partially saved newest checkpoint is rejected; `auto` falls back to the latest fully validated one. Some distributed and other optimizer setups use different checkpoint layouts; these must be explicitly supported and tested before use.

## What happens next

1. Run CPU-only integrity tests and validate curriculum.
2. Check for actual usable previous v248-v251 adapters and evaluator receipts; do not invent a parent checkpoint.
3. Train one tiny local candidate or schedule a free GPU session. Choose permanent checkpoint storage first.
4. Resume after interruption and verify step counters increase instead of restarting.
5. Run held-out, hallucinated-execution, privacy/authorization, and product execution tests. Independently approve any candidate before production promotion.

No automatic model replacement; no autonomous user-data harvesting or implicit uploads.
