# Aerynza v256 — GPU-ready production intelligence and execution curriculum

**Prepared 2026-10-29. Status: structurally validated, NOT trained, NOT provider-verified, NOT production-ready.** This is a self-hosted candidate and test plan, not a claim of GPT-6-level reasoning or replacement. The native model cannot inherit proprietary weights, private reasoning traces, credentials, built-in tools, or paid model capabilities by imitation. All examples are original synthetic material and require independent human review.

## Why this method

Recent Aerynza v251/v252/v253/v255 work established a repeatable **pinned Qwen3 1.7B + LoRA rank 16 + assistant-only loss + balanced package/shell sampling + protected independent holdout** pipeline. The trained candidates shown in older gate reports were not universally successful: e.g. grow167 7,914 rows failed receipt and canon gates; grow168's generic blast and grow169's 150-row targeted fix likewise failed some gates. There is no independently established winning checkpoint or proof of model parity. This v256 extension **does not choose a parent adapter by version number**.

Preserve the validated structural style, but shift examples toward **usable output, actual tool choice, provenance, safe execution and recovery** instead of teaching hundreds of repetitive “I can help with X” responses.

## What is in GitHub

- `training_packs/next_gpu/v256_production_readiness_train.jsonl`: **60** synthetic assistant-answer lessons, 3 per each of **20** production skill families.
- `training_packs/next_gpu/v256_production_readiness_heldout.jsonl`: **20** distinct independent supervised evaluations; protected from optimization.
- `training_packs/next_gpu/v256_independent_execution_benchmarks.json`: **20** independent, unrun runtime tasks, needing real sandbox fixtures, provider adapters and receipts. They are explicitly **not** SFT data.
- `scripts/validate_v256_production_packs.py`: schema, provenance, family counts, unique IDs, train/heldout/benchmark leakage, 640 inventory and compiled output checks.
- `scripts/build_next_gpu_packs.py`: extended deterministic compiler; CI reports **2,286 GPU candidate rows** including older replay and one training lesson for each of the **640 catalog IDs**. That ID count does not mean 640 fully working integrations.
- `scripts/build_v255_independent_eval.py`: independent evaluation union expanded to **80 supervised heldout examples** (not generated answers). The 20 executable benchmarks remain separate.
- `scripts/prepare_v256_gpu_bundle.py`: deterministic immutable bundle with SHA-256 hashes for all source files, each compiled training package, and independent evaluation. Re-running against changed files refuses to overwrite.
- `scripts/launch_v256_production_readiness.sh`: CPU-first launcher, explicit human approval before training, no silent overwrite or fallback CPU training, persistent checkpoint requirement, and no production model replacement.
- `scripts/select_v256_exact_resume.py`: chooses only a compatible complete checkpoint with weights, Trainer state/step, optimizer, scheduler and RNG. Can use the exact same reviewed parent adapter when the stage originally started from one.
- `scripts/aerynza_training_optimizer.py` (existing): bounded CPU-only failure diagnosis, **not** automatic retraining or approval. See `docs/training/AERYNZA_GATE_FAILURE_OPTIMIZER.md`.

The training records are synthetic, not harvested from private chats or production user data. Their `review_status` is `pending_human_review`; do not flip this merely because CI passes.

### Twenty focused packages

| Family | Expected improvement | Separate working infrastructure/evidence |
| --- | --- | --- |
| context_dialogue | Multi-turn corrections, natural conversational continuity | Scoped real user state and retrieval |
| quantitative_reasoning | Accurate arithmetic and explicit assumptions | Deterministic calculators, financial source validation |
| source_grounding | Source freshness, citations, conflicting evidence | Real connected search and source records |
| code_debugging | Debugging and verified patches/tests | Isolated code runner, repository permissions, CI |
| artifact_quality | Useful plans, tickets, structured documents | Working document/spreadsheet/PDF renderers and file receipts |
| data_integrity | Reconciliation, provenance, exports | Reliable parsers, database queries and pagination |
| tool_selection | Choose least-authority tool and correct workflow | Signed connectors and typed tool registry |
| receipts_and_state | Distinguish proposed, accepted, attempted and verified | Independent provider status/receipt checks |
| failure_recovery | Retry safely, preserve partial progress | Durable idempotency keys, task journal, reconciliation |
| memory_provenance | Verified saved memory and correction propagation | Database consent and scoped storage |
| cross_domain_orchestration | Connect goals, money, nutrition, schedules | Graph permissions and multi-domain source freshness |
| ambient_agent | Quiet background planning and alert triggers | Durable jobs, local clock, token refresh, notification receipts |
| family_and_child_safety | Guardian/family permissions, child-safe replies | Guardian/household isolation and audit |
| multimodal_truth | Honest image/audio/video interpretation limits | Actual perception/rendering/transcription engines |
| timezone_and_scheduling | Overnight schedules, DST, travel zones | Timezone-aware calendar adapter |
| offline_and_self_reliance | Local operation and checkpoint honesty | Native model plus network-negative local test |
| security_and_tools | Resist injection, scoped access and migration risk | Auth, audit, tool sandbox, signed requests |
| production_evaluation | Never infer readiness from a catalog | Independent E2E and quality/latency acceptance suite |
| creative_originality | Original dialogue, scripts and style | Optional independent originality and media render review |
| careful_health_finance | Careful high-stakes boundaries and triage | Authorized data, safe escalation, no phantom transactions |

## Run now: CPU-only preflight

From the **`Aerynza/ascension-ai`** repository branch containing these files:

```bash
bash scripts/launch_v256_production_readiness.sh --preflight
```

This checks the 20 modules, 640-ID registry, independent data split and trainer schema, and creates the deterministic local bundle. It **does not** download Qwen weights or spend GPU credits. Alternatively:

```bash
python3 -m scripts.validate_v256_production_packs
python3 -m scripts.prepare_v256_gpu_bundle --build
python3 -m unittest discover -s tests -p 'test_v256_training_packages.py' -v
```

CI evidence: [v256 GPU preflight workflow](https://github.com/Aerynza/ascension-ai/actions/workflows/v256-gpu-training-preflight.yml) — checks are CPU structural evidence only.

## GPU training handoff (only after review and resource availability)

Use a **persistent mount and backup mechanism** accessible in subsequent sessions (mounted inside the repository, or a stable relative/absolute checkpoint path). A disposable cloud session with ephemeral `checkpoints/` **does not guarantee recovery**. Preserve entire `checkpoint-STEP` directories, including optimizer, scheduler, RNG, Trainer state and adapter weights; also preserve `aerynza_exact_resume_contract.json`, dataset bundle and any reviewed parent adapter. Do not save only a deployed GGUF or adapter final weights and call that an exact resume.

The full default LoRA profile requires **CUDA and at least 12 GiB VRAM**; this may not fit a laptop/free CPU or a small GPU. No cloud GPU is provisioned. Pin the Qwen revision and a reviewed environment lock before training.

After checking dataset quality, licensing, model tokenization length, hardware, storage, and independent benchmark separation:

```bash
export AERYNZA_V256_OUTPUT_DIR=/persistent/aerynza/checkpoints/v256_candidate_a
export AERYNZA_V256_DATA_REVIEW_APPROVED=YES
export AERYNZA_V256_PERSISTENT_STORAGE_APPROVED=YES
bash scripts/launch_v256_production_readiness.sh --train
```

The above starts a **new** training stage, only after those explicit approvals. It never claims a production release. To start a **new curriculum stage from an existing reviewed PEFT adapter**, select the actual best-performing parent based on signed independent evaluation—not a higher version number—and additionally set:

```bash
export AERYNZA_V256_PARENT_ADAPTER=/persistent/aerynza/checkpoints/REVIEWED_PARENT
export AERYNZA_V256_PARENT_GATE_APPROVED=YES
bash scripts/launch_v256_production_readiness.sh --train
```

That is **new-stage initialization**, not optimizer-state recovery. It requires a new empty output directory.

To continue an interrupted **same** stage, retain the identical dataset and parameters. Set the same environment, including `AERYNZA_V256_PARENT_ADAPTER` when the original stage used it, then:

```bash
bash scripts/launch_v256_production_readiness.sh --resume
```

This locates the newest complete compatible checkpoint, verifies exact run identity and state, and refuses to silently restart at step zero. If the best checkpoint is incomplete, an older compatible *complete* checkpoint may be chosen; that is an explicit, reported step rollback, not silent fresh training. New stages and exact resumes must never share the wrong output directory.

### Evaluation and optimizer loop

When a candidate exists, run unchanged historical gates and the new independent suite. The following launches existing automatic model gates but **does not run all 20 new runtime benchmarks**:

```bash
bash scripts/launch_v256_production_readiness.sh --evaluate
```

Then run the existing CPU-only gate failure analyzer on actual candidate receipts and actual canonical/capability/execution reports. **Do not fabricate gate reports or fill in human approvals.** For model behavior deficits, author *fresh* reviewed similar training examples, choose a new stage and preserve old heldout prompts; for missing tools, stale data, broken storage, latency and unsigned actions, fix the application runtime, not the LoRA curriculum.

## What still separates Aerynza from ChatGPT-class production

The local model alone cannot supply reliable tools. A true closed-owned intelligence stack needs: source-grounded browsing, code execution sandboxes, document and spreadsheet renderers, image/audio/video generation and perception engines, authenticated external apps, personal and family privacy scopes, a durable event/task ledger, idempotent action execution, accurate UTC/local-time services, retrieval/memory/knowledge freshness, prompt-injection defenses, token/cost/latency telemetry, user feedback, human oversight, reproducible deployment, and real E2E tests.

The **640 labels are a catalog**, not a claim that 640 capabilities are implemented or locally executable. Existing `scripts/build_v253_640_execution_map.py` already creates a separate **640-case evidence queue** with all gates initially `not_run`. Do not train on this evaluation queue. Complete these specific gates per capability: signed actor/shell/tenant; tier and guardian; tool engine/typed input; real or sandbox provider; receipt and output; timeout/retry/recovery; native quality; privacy; human review. Unknowns remain unsupported or provisional.

## Release decision

**HOLD.** No GPU training happened in this change. No adapter was trained or promoted, no ChatGPT parity was measured, no claims about live tools or user data were verified, and no production deployment or external spending was performed. This work is on a review branch and must not be blindly merged into `master`: the development branch currently contains stacked historical training work while current `master` contains newer native signed-request and release security changes. Reconcile both before model traffic or production changes.
