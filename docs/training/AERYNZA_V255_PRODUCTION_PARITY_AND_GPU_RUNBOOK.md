# Aerynza v255 — next GPU training and ChatGPT-class product-parity acceptance

**Status: reviewed-style synthetic candidate; CPU-validated, NOT GPU-trained, NOT promoted.** This document extends [PR #15](https://github.com/Aerynza/ascension-ai/pull/15), which already combines v251/v252, v253, v254, 640 capability-ID lessons, exact checkpoint restoration and optimizer-driven repair triage. Do not re-implement the older v251 trainer on PR #7 over this newer pipeline.

## What has been added

- `training_packs/next_gpu/v255_production_parity.jsonl`: 36 original synthetic SFT tasks emphasizing useful general assistance and multi-domain execution truth (money, nutrition, fitness, grocery budgets, aspirations/Zillow, coding, reasoning, independent research, calendars, Executive products, Sprout, NexusHome and Family).
- `training_packs/next_gpu/v255_production_parity_heldout.jsonl`: 18 independently written hard cases; never used as optimizer targets.
- `scripts/build_v255_independent_eval.py`: builds a SHA-256-bound, fail-closed **60-case supervised evaluation union** from v251, v253 and v255, while retaining the separate v254 rubric-only cases for external execution scoring. Every prompt stays outside training. The rubric-only cases have no assistant answer and must never be fabricated as ground truth. No model runs in this compiler.
- The v255 training rows are included in the deterministic next-GPU bundle; the independent eval union is the **evaluation** split used by the launcher, not a fine-tuning dataset.
- `tests/test_v255_production_parity.py` and CPU-only GitHub Actions catch missing rows, missing shells, duplicated or leaked prompts, altered output and bundle format regressions.
- Original human review remains **pending** for every synthetic training/heldout example. Do not imply authoring is human sign-off or evaluation success.

## Existing evidence used to choose training style

The repository's actual earlier Qwen LoRA receipts show broad generic memorization is insufficient:
- `grow167_triple_proven_a` passed only 5/6 canonical and 5/8 receipt-truth prompts despite 7,914 training records.
- `grow168_generic_blast` passed 5/6 canonical and 7/8 receipts; blocked.
- `grow169_targeted_fix_b` passed 5/6 canonical and 7/8 receipts; blocked.
- A 640/640 registry-response gate did **not** imply 640 operational integrations; output could simply identify a feature and ask for a receipt.
- **No reliable head-to-head proof picks the best v249/v250/v251 adapter as universal champion.** Only independently reviewed gate data can choose the parent.

Use the existing pinned `Qwen/Qwen3-1.7B` LoRA adapter approach: assistant-only loss, balanced package/shell sampling, separate heldouts, lower learning rate for narrow corrective stages, immutable curriculum hashes, full optimizer/scheduler/RNG/step checkpoint backup, and independent quality gates. This is specialization of a small local model; it cannot guarantee frontier-model parity.

## Start next week — CPU-only preflight (no paid GPU)

From the repository root of the stacked PR branch after source review:

```bash
python3 scripts/validate_v251_packages.py
python3 scripts/validate_v252_packages.py
python3 scripts/build_next_gpu_packs.py --dry-run
python3 -m scripts.build_v255_independent_eval --validate-only
bash scripts/launch_next_gpu_candidate.sh
```

The launcher compiles or verifies the immutable training bundle, compiles the independent v251–v255 evaluation union, and runs `--validate-only` on the Qwen trainer. No training starts by default. The resulting dataset/review hashes must be preserved across the GPU handoff.

Only after the founder approves resource cost and a human reviewer signs the exact bundle hash off-repo should training begin with `AERYNZA_NEXT_GPU_TRAINING_APPROVED=YES`, a valid `AERYNZA_NEXT_GPU_REVIEW_RECEIPT`, and `--run` on an approved persistent GPU machine. Never copy secrets or personal health/financial records into the curriculum. For interrupted training, provide `AERYNZA_NEXT_GPU_RESUME` pointing to a full verified checkpoint from the same run and identical dataset/config. `AERYNZA_NEXT_GPU_PARENT_ADAPTER` begins a **new stage** from selected reviewed adapter weights; it is not exact continuation. Do not supply both. Preserve checkpoint trees and manifests in persistent protected storage before stopping any ephemeral GPU session.

## Product parity acceptance map (not just a model score)

| Area | Minimum demonstration before claiming production parity |
| --- | --- |
| Dialogue, reasoning and math | Multi-turn conversations; grounded calculations and explanations; adversarial uncertainty and source handling; no repetitive refusals |
| Research and browsing | Real connected retrieval, source provenance and citation freshness; no invented websites or news |
| Coding and technical work | Sandboxed file access, patch creation, test execution, diff and runtime evidence; no claiming tests/deployments ran without receipts |
| Documents/data/spreadsheets | Correct file parsing and generation, structured data integrity, usable exports and independent check of calculations |
| Multimodal | Actual supported image understanding/generation, transcription and speech output with declared engines; don't substitute text that merely describes an image |
| Durable agent execution | Authenticated tool adapter, per-resource authorization, pre-action confirmation, idempotency, retries, explicit state and provider receipt |
| Background presence | Durable schedules, source freshness, notification suppression, quiet hours, safe recovery and observable outcome signals |
| Personal memory and Human Model | Consent-scoped writes/reads, correction propagation, provenance and expiration; cross-shell negative privacy tests |
| AP/Nexus five shell contracts | AP serves Life/Sprout/Executive, Nexus serves Home/Family; Core is a paid tier. Child/guardian and group permission isolation |
| Commerce, wealth and finance | Correct budget math, real account freshness, idempotent authorized checkout and transfers, no duplicate subscriptions |
| 640 capabilities | One distinct registry test per ID **plus** evidence of each actual executor/provider artifact. Inventory lessons alone count only as curriculum coverage |
| Self-hosted operation | Block outside model APIs during canary; measure useful quality, VRAM/latency/concurrency and fallback. Train, serving and tools are separate components |

To call Aerynza fully self-reliant, independent tests must succeed **without outside inference** for the claimed offline features. Externally sourced real-time facts, banking, calendars, shopping carts, streaming media, websites and cloud services naturally require their own authorized connectors or local data; a model cannot wish those APIs into existence.

## Hard release blockers

- No live GPU candidate has been trained, checkpoint-restored and accepted from this v255 package.
- The 640 catalog is not a release certificate: verify each enabled operation with a positive receipt and negative unauthorized/replay cases.
- Main application draft PR #104 and related issues still require operational billing safety, cross-shell permissions, durable provider execution and real database/infrastructure evidence.
- Independent evaluator must review child/privacy safety, health and money boundaries, code execution claims, math accuracy, native reasoning and all critical denial scenarios. Reject any candidate that regresses an existing pass.
- Human approval is mandatory for model promotion or production traffic switch.

**Review outcome is currently BLOCKED PENDING HUMAN REVIEW, GPU training, independent external-action tests and deployment acceptance.**
