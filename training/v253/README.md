# Aerynza v253 — GPU-ready training packages for native self-reliance

**Status:** Synthetic SFT examples and independent benchmarks have been authored, not yet independently reviewed, GPU-trained, scored, or promoted. No license has been granted to copy another assistant's private model weights, training data, secret reasoning, or commercial tool APIs. The objective is an original, locally runnable Aerynza system with honest boundaries.

**Dependency:** This branch extends the [v251/v252 training foundation](https://github.com/acensionlifeos-dev/ascension-ai/pull/8). Do not train by merging v253 directly on `master` while v251/v252 files are missing. See `training/v253/manifest.json`.

## What we found from recent actual training receipts

- The v251/v252 GPU preflight was successful on October 10, 2026: synthetic dataset validation, negative privacy tests, trainer schema, and a 640-ID evidence queue **passed without GPU training**. Source: GitHub Actions run `38028452292`.
- Earlier attempts prove why bigger dumps are not automatically better: `grow167_triple_proven_a` trained 7,914 records but passed only **5/6 canonical** and **5/8 receipt truth** cases, blocking promotion. `grow168_generic_blast` trained 280 and reached **5/6, 7/8** but still failed. `grow169_targeted_fix_b` used 150 and reached **5/6, 7/8**, still failed. The independent gates reject all three.
- Therefore this package uses **specific worked tasks, corrective counterexamples, uncertainty, tool-state truth, and domain-specific receipts**, preserving separate tests. It avoids generic repetitive \"I can help with capability X\" filler, which previously failed semantic checks.
- The base is the repo's known pinned **Qwen/Qwen3-1.7B** revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, on a permissive **Apache-2.0** model license. The repository's model is roughly 2.0 billion actual parameters; a small LoRA is a specialization step, not frontier general reasoning parity.

## Twelve focused modules (12 × 6 = 72 new SFT conversations)

| Module | Desired skill | Proof still needed |
| --- | --- | --- |
| conversation_reasoning | Useful conversation, grounded arithmetic, counterarguments, causal uncertainty | Independent multi-turn and math tests |
| research_grounding | Cite source freshness, decline invented current facts, compare claims with primary documents | Live research evaluation with source tracing |
| coding_debugging | Diagnose code, tests, SQL binds, permissions, safe patches | Isolated coding benchmark, execution traces |
| documents_data | Parse whole data, create and check actual artifacts; separate missing metrics | Rendered files, parser coverage |
| tool_execution | Typed actions, authority, readbacks and honest final state | Real sandbox provider end-state |
| failure_recovery | Idempotency, partial progress, no blind retries | Timeout/restart scenarios with receipts |
| ambient_memory | Session versus saved memory, freshness, quiet proactive planning | Durable scoped storage, corrections |
| multi_shell_privacy | AP personal Life/Sprout/Executive vs NexusHome/Family and guardian rules | Negative tenant/guardian tests |
| health_financial_boundaries | Safety escalation and no unauthorized money/legal action | Safety and authorized provider review |
| multimodal_truth | Honest image/audio/video/device limits; no phantom outputs | Actual image/audio/device workloads |
| autonomous_planning | Background orchestration, quiet hours, cross-domain prioritization | Staging jobs and complete outcome loop |
| self_hosted_independence | Local-only inference and measured release gates | Network-blocked native canary, GPU load testing |

Each new row has `id`, canonical `shell`, `user`, `assistant`, `package`, `tags`, `provenance=ai_authored_synthetic`, `source_kind=synthetic` and **`review_status=pending_human_review`**. No copied user conversations or private notes are included.

The independent conversation holdout has **24 examples**, two per skill. The separate benchmark specification has **12 unseen task families**, each needing a separately prepared executable sandbox. **Never train on either the benchmark prompts or their solutions.** Once a benchmark prompt is exposed for fine-tuning, retire it and commission a new holdout.

## Deterministic retained curriculum

`python3 scripts/build_v253_gpu_mix.py` validates and interleaves:

- v251: 60 synthetic foundation examples,
- v252: 24 synthetic execution examples,
- v253: 72 targeted skill examples,

for **156 candidate training rows**. The generated mix has its own hash receipt. Prompt overlap checks compare v251 train/heldout, v252 train, v253 train/heldout, and independent benchmark prompts. The generated mix file is not shipped as a substitute for signed, immutable data review.

The local trainer computes **assistant-only loss**; these examples are not hidden chain-of-thought transcripts. Explanations are short and observable. A single training objective cannot itself make AP execute commands, browse, transcribe, render or operate finance accounts; those require authorized tools, middleware, APIs, storage and verification.

## GPU handoff for the week of October 19–25, 2026

Default does not train and can be run now:

```bash
python3 scripts/validate_v253_packages.py
python3 -m unittest discover -s tests -p 'test_v253_training_packages.py' -v
bash scripts/launch_v253_self_reliance.sh --preflight
```

When the GPU is available, an independent reviewer has inspected every sample and benchmark, actual GPU/VRAM is known, licenses and input hashes are approved, and a real inference quality baseline is captured:

```bash
export AERYNZA_V253_DATA_REVIEW_APPROVED=YES
export CUDA_VISIBLE_DEVICES=0
export AERYNZA_V253_OUTPUT_DIR=checkpoints/aerynza_v253_candidate_a
bash scripts/launch_v253_self_reliance.sh --train
bash scripts/launch_v253_self_reliance.sh --evaluate
```

For a resumed adapter, set `AERYNZA_V253_PARENT_ADAPTER` **and** `AERYNZA_V253_PARENT_GATE_APPROVED=YES` after verifying its adapter configuration, baseline gate results, exact model revision, license and fingerprint. **Do not pick a parent solely for its version number**. The default train profile requires CUDA and at least 12 GiB VRAM; 16+ GiB is preferable, and the actual training batch and length must be checked on the device. Smaller hardware needs a separately verified quantized training implementation; this script does not invent one. Never overwrite an existing adapter directory, or deploy a candidate automatically.

Pin the base model revision, tokenizer, dependency environment, seed, inputs SHA-256, parent hash (if any), GPU model/VRAM, CUDA/driver and train receipt before interpreting comparisons. `requirements-training.txt` pins the supported Transformers family, but for a repeatable run install from a tested, locked environment/image rather than assuming open ranges are bitwise reproducible.

## Closed-system definition and the 640 registry

**A local model is not the same as a closed functional product.** Aerynza needs self-hosted inference, local embedding/retrieval, persistent memory, model routing, authorized action planning, durable queues, sandboxed code/file/media tools, verified provider connectors, cross-shell permissions, context freshness and telemetry. Email, banks, calendar, location or public web data still require permitted external interfaces, unless the user supplies offline data. Never conceal external inference fallback when local-only mode is requested.

`python3 scripts/build_v251_capability_validation_queue.py` enumerates **all 640 existing catalog IDs** into a review queue. Catalog counts and historical flags, including `execution_ready`, do **not** constitute execution proof. For each ID, record: intended owner shell; supported reasoning; tool schema; signed identity/tier and resource scope; availability; test fixture; authorization; state change or artifact; trusted receipt; offline/recovery path; and production observation. Only promote evidence-backed categories, not 640 generic chatbot replies.

## Mandatory no-promotion gates

1. Dataset integrity and independent human editorial/license/privacy review.
2. Frozen Qwen/tokenizer run, assistant-only loss, candidate training receipt and SHA manifest.
3. Independent heldout quality compared with an approved baseline, including latency, task consistency and regression sampling.
4. **100% pass** for critical authorization/privacy/child-safety and receipt-truth tests; zero fabricated tool results or private-data disclosures.
5. All 640 entries assessed **separately**; no unsupported model/tool/readiness claim.
6. Twelve new executable benchmark fixtures plus coding, research, artifacts, background/autonomy and actual provider receipts, verified by an independent reviewer.
7. Signed AP/Nexus native runtime checks for personal, child, work and shared household/family scopes; adversarial tenant isolation.
8. Resource/retry testing on real hardware, no unsafe duplicate external actions after crash, offline-only network proof.
9. Founder-designated review and explicit operator approval before any default-model switch, release, or provider write.

This package is **prepared for review and preflight**, not approved for next week's GPU run until the actual device and independent approval are available.
