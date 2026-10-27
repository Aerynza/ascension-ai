# Aerynza AI — October 23, 2026 next-GPU training readiness and ChatGPT-class product gap map

**Status:** CPU preflight package; training, model selection, independent evaluation and production promotion are **not completed**. This branch combines the existing v251/v252 foundation/execution work, the rigorous resume contract, and the deterministic next-GPU capability curriculum from separate branches. Do not merge, buy GPU time, publish models or change production without founder approval.

## Source work reconciled

| Previous work | What transfers | Important limitation |
| --- | --- | --- |
| `codex/aerynza-v251-training-packages` | 60 human-review-pending original examples across 10 themes, 18 independent heldout examples, 640-ID validation queue, assistant-only LoRA/canonical evaluation | No GPU checkpoint proven; preflight is not model quality evidence |
| v252 execution extension | 24 execution/reasoning/recovery examples and 12 independently specified evaluation fixtures | No executable tool tasks or benchmark pass from examples alone |
| `codex/gpu-nextweek-training-packs` | Deterministic compiler of reviewed legacy v20 records, 640 distinct catalog-specific lessons, founder scenario seeds, source/heldout isolation manifest | These are catalog *lessons*, not 640 successful actions; older source approval/provenance needs review |
| `codex/aerynza-resumable-free-training` | Run contract bound to datasets/base/hyperparameters; Trainer checkpoint includes adapter, optimizer, scheduler, step and RNG; validate complete checkpoints and fail closed on mismatch | Must preserve full checkpoint tree on persistent storage; complete compatible checkpoint required for exact resume |
| v231 inference release on `master` | Full v231 GGUF is checkpointed and hash documented as an inference/runtime artifact | It is not a compatible LoRA parent or optimizer checkpoint for this Qwen3-1.7B training job |

**Observed successful technique:** The repository validates an *assistant-only loss* Qwen LoRA trainer with balanced package/shell sampling and separate heldout data. Recent targeted v249/v250 correction packages attempted to repair semantic failures with lower learning rates. **No trusted head-to-head receipt establishes which adapter scored highest**, so do not use the most recent model by default or claim ChatGPT-level parity.

## Scope of this bundle

- `scripts/build_next_gpu_packs.py` and `training_packs/next_gpu/seed_scenarios.jsonl` are the deterministic compiler + explicit synthetic founder lessons from the pre-existing GPU packs branch.
- `tests/test_next_gpu_packs.py` verifies exactly 640 unique catalog IDs, no known heldout-prompt overlap, deterministic compilation and support for all generated training shells.
- `scripts/verify_next_gpu_bundle.py` verifies current source hashes, compiled package checksums and exact review receipt digest.
- `scripts/launch_next_gpu_candidate.sh` runs CPU-only validation by default, never buys resources and never trains until `--run`, `AERYNZA_NEXT_GPU_TRAINING_APPROVED=YES` and a reviewer receipt for the **exact manifest bytes** exist.
- `scripts/train_qwen_ascension_lora.py` now understands Executive as an AP context distinct from the Core subscription tier. This change is to training prompts **only**; actual signed Executive runtime authorization must be separately implemented and tested.
- `.github/workflows/next-gpu-curriculum-preflight.yml` runs preflight on GitHub Actions without loading model weights or using a GPU.

## CPU-only preparation before the GPU is rented

From a reviewed checkout of this branch:

```bash
bash scripts/launch_next_gpu_candidate.sh
```

The command validates v251/v252 and the 640 inventory, generates a local compiled training bundle if none exists, verifies it, and checks the current trainer schema in CPU-only `--validate-only` mode. It does **not** call `trainer.train()` or produce model weights. Keep `training_packs/next_gpu/generated/manifest.json` together with all generated JSONL files.

To review a generated bundle, inspect its `manifest.json`, all sensitive and representative category examples, and its source-provenance ledger. The original sources are not universally marked human-approved. Check for incorrect answers, unlicensed imports, private data, false execution claims, inconsistencies in guardian/workspace boundaries and prompt leakage before approving. Do not sign an unseen/generated manifest.

Only **after** genuine review and independent approval, create an external, private receipt (do not upload secrets) with:

```json
{
  "approved": true,
  "reviewed_by": "human reviewer identifier",
  "reviewed_at": "YYYY-MM-DD",
  "bundle_manifest_sha256": "64-character SHA-256 of the actual generated/manifest.json bytes"
}
```

Then, on an approved GPU with **persistent** checkpoint storage:

```bash
export AERYNZA_NEXT_GPU_REVIEW_RECEIPT=/private/path/review-receipt.json
export AERYNZA_NEXT_GPU_TRAINING_APPROVED=YES
export AERYNZA_NEXT_GPU_OUTPUT_DIR=/durable/checkpoints/aerynza_next_gpu_candidate
bash scripts/launch_next_gpu_candidate.sh --run
```

The launcher pins `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` at one epoch, learning rate `2e-6`, max length 1024, and two-step complete checkpoint intervals. Actual GPU RAM/driver compatibility, model license, tokenizer, training runtime versions and disk requirements must be checked on the chosen hardware.

To **resume exactly** on the same compatible run/dataset/base/optimizer setup, keep the same command and environment and set `AERYNZA_NEXT_GPU_RESUME=auto`. The code refuses incomplete, different or absent checkpoints rather than silently restarting. To **start a new stage**, use a distinct output directory with `AERYNZA_NEXT_GPU_PARENT_ADAPTER=/reviewed/adapter`; this carries trained weights but resets optimizer/scheduler/step. Never use both modes at once.

## ChatGPT-class *product* parity, not merely model imitation

Fine-tuning 1.7B on a few thousand records does not train an independent frontier general-intelligence model. To approach a similar user experience, Aerynza needs **an evaluated system of components**:

1. **Conversation/reasoning:** multi-turn consistency, mathematical and code execution checks, source-aware retrieval, uncertainty, robust tool selection and concise direct communication.
2. **Execution/orchestration:** real browser, files, code, calendar, messages, finance, search and media adapters; each action needs a typed schema, signed identity, authorization, idempotency, timeout/retry semantics, verified artifact/receipt and truthful user-visible status.
3. **640 capabilities:** assign each ID the implementation type (local deterministic engine / model generation / connected API / future hardware / unsupported), a distinct positive and denial fixture, and evidence of **successful outcome**. A lesson referencing an ID is not a working capability.
4. **Persistent memory and Vortex:** consent-scoped facts, provenance, correction, retrieval freshness, quiet background synthesis, cross-shell isolation and explicit undo/delete semantics.
5. **AP and Nexus shells:** AP supports Life, Sprout and Executive; Nexus supports Home and Family. Core is a feature-access tier. Enforce guardian, organizational and resource boundaries in code; prompts alone cannot provide authorization.
6. **Multimodal interfaces:** image and audio processing, text-to-speech/speech-to-text, documents, search, device signals, presence and native/mobile clients require separate models, runtimes, hardware/permissions and tests.
7. **Self-hosted independence:** no hosted-LLM fallback **in the strict independence acceptance test**; local inference, retrieval index, database, job scheduler, adapters and observation stack must remain functional. External weather/email/payment services still require their providers; the system should degrade honestly when disconnected.
8. **Quality/operations:** independent hidden benchmark fixtures, blinded comparisons, numeric correctness, severe-safety/child/medical/financial risk gates, secret leakage checks, adversarial prompt injection, concurrency and latency budgets, persistent outcome logs, rollback, and reviewer-controlled promotion.

## How to optimize after release-gate failures

- Freeze the candidate and all evaluation receipts. Diagnose whether failure belongs to **model output**, **tool implementation**, **missing permission**, **data quality**, **stale memory**, **provider availability**, **hardware resource**, or **test defect**. Do not automatically train on every failure.
- Fix tool/schema/auth/database/runtime faults in code first. Only train semantic failures for which reviewers have written correct responses.
- Curate **new** reviewed recovery examples into a new curriculum stage. Preserve truly heldout benchmark prompts; any exposed evaluation item must be retired and replaced independently.
- Compare the candidate with the last verified adapter over the same test fixtures, plus fresh unrelated prompts. Reject any false “sent/paid/saved/verified” claims, private-data leakage, severe child/safety failures, or quality regressions.
- Do not automatically promote or switch hosted providers off. Require explicit review of adapter hash, dataset/manifests, evaluation receipts, performance/latency and rollback path.

## Current blockers and intended release gate

No human-signed next-GPU review receipt, compatible best-parent adapter decision, paid/available training GPU, or actual v251/v252/next-GPU trained adapter has been verified. GitHub Actions CPU checks establish **dataset/preflight integrity**, not general intelligence, successful real tools or 640 production capabilities.

A separate release readiness suite must show execution success and verified receipts for every advertised capability family, shell/tenant isolation, business logic invariants, resilient background jobs, and a passing security scan. Production model replacement is blocked until all required gates pass.
