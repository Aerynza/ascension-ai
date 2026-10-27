# Aerynza AI GPU handoff — October 25, 2026

**State:** Dataset and safety preflight ready for independent review. **Not trained, promoted, or proven ChatGPT-class.** Repository: `Aerynza/ascension-ai`; draft branch `codex/aerynza-gpu-readiness-integrated` / PR #15. Do not merge to master or operate paid GPU services without the founder's explicit approval.

## What exists on the integrated branch

| Layer | Source | Evidence |
| --- | --- | --- |
| Legacy balanced conversational replay | `evals/training/aerynza_v20_balanced_master.jsonl` | Quarantines duplicate prompts, conflicting answers and unverified claims |
| Exactly 640 catalog-specific *lessons* | `evals/training/aerynza_product_v236_natural_action_correction.jsonl` + `evals/capability_status.json` | Explains plans and required tools, **does not prove execution** |
| Canonical seed scenarios | `training_packs/next_gpu/seed_scenarios.jsonl` | Consent, tools, AP/Nexus shells, memory, receipt truth |
| v253 targeted self-reliance | `evals/training/aerynza_v253_{reasoning_execution,ambient_sovereignty}_train.jsonl` | **72** synthetic supervised examples across **12** modules |
| v254 single-request coordination | `training_packs/next_gpu/cross_domain_v254.jsonl` | **32** synthetic examples across **8** modules: body, grocery, housing, family, Creation-to-Executive, recovery, local-only and systems |
| Independent evaluation | `evals/training/aerynza_v251_foundation_heldout.jsonl`, `evals/training/aerynza_v253_independent_heldout.jsonl`, `training_packs/next_gpu/cross_domain_v254_heldout.jsonl` | 18 + 24 + 12 heldout prompts. **Never merge into the training corpus.** |
| Executable benchmark *specifications* | `evals/benchmarks/aerynza_v253_independent_tasks.json` | **12** tasks need separate, trusted sandbox/provider fixtures and reviewer receipts; not run by dataset preflight |
| Honest 640-capability status | `scripts/build_v253_640_execution_map.py` | 640 individually blocked action/evidence rows; source classes are 605 domain prompts, 26 native action and 9 provider action (historical labels only) |
| Recovery / failed-gate trainer | `scripts/aerynza_checkpoint_resume.py`, `scripts/aerynza_training_optimizer.py`, `scripts/create_aerynza_repair_approval.py`, `scripts/run_aerynza_repair_stage.py` | Exact checkpoint identity and optimizer/scheduler/RNG validity, fail-closed bounded review of semantic retraining |

A recent GitHub Actions CPU preflight on this branch successfully built **2,190** combined training records, kept the 640 inventory complete, validated heldout isolation and checksum-bound bundle verification, tested the Qwen adapter loader without GPU, and ran optimizer/repair tests. The compiled output is generated and should not be treated as independently reviewed.

## What the latest training evidence supports

The historical report in `training/v253/README.md` records that a 7,914-example generic run still failed canonical (5/6) and receipt-truth (5/8) gates, while two smaller corrective runs each reached 5/6 canonical and 7/8 receipt-truth and **also failed**. The latest `docs/training/AERYNZA_NEXT_GPU_READINESS_2026-10-23.md` expressly notes there is **no trusted head-to-head receipt proving the highest-quality adapter**.

**Engineering choice:** prefer original, targeted worked examples with explicit provider state and uncertainty, balanced package/shell sampling, assistant-only loss, preserved independent holdouts, and comparison to an actual verified parent. Bigger or newer is not automatically better.

## No-GPU preflight

Run on a checkout of the branch, using Python 3.11 or the CI-pinned environment:

```bash
python3 scripts/validate_v251_packages.py
python3 scripts/validate_v252_packages.py
python3 scripts/validate_v253_packages.py
python3 -m unittest discover -s tests -p 'test_next_gpu_packs.py' -v
python3 -m unittest discover -s tests -p 'test_aerynza_checkpoint_resume.py' -v
python3 -m unittest discover -s tests -p 'test_aerynza_training_optimizer.py' -v
python3 -m unittest discover -s tests -p 'test_aerynza_repair_stage.py' -v
python3 -m unittest discover -s tests -p 'test_v253_640_execution_map.py' -v
bash scripts/launch_next_gpu_candidate.sh
```

The launcher refuses paid training without `--run`, `AERYNZA_NEXT_GPU_TRAINING_APPROVED=YES`, and a **genuine independent human review receipt** tied to the exact generated manifest SHA-256. Generated data changed with v253/v254. If an older `training_packs/next_gpu/generated` already exists, first preserve it as an archive, then recompile and review the new immutable output. Do not edit a signed manifest after approval.

## GPU run only after approval

1. Identify GPU model, usable VRAM, storage capacity, CUDA/driver, actual cost and tested environment image. The default Qwen3-1.7B LoRA profile has not been validated on an unknown GPU; start with a small VRAM/timing smoke test and a persistent checkpoint disk. **Never substitute the deployed v231 GGUF for a compatible Qwen LoRA optimizer checkpoint.**
2. Review all samples for math correctness, synthetic provenance, contamination with private material, unlicensed content, contradictory tool claims and age/organization boundaries. Independently prepare evaluation fixtures and capture an authentic review receipt.
3. Use the branch's pinned `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, seed 3407, assistant-only loss, LoRA r16/alpha32, one initial epoch and low initial learning rate (2e-6), subject to a documented hardware-compatible adjustment. Capture versions and all hashes.
4. To train **a fresh candidate** with approval, set a new durable output directory and invoke:

```bash
export AERYNZA_NEXT_GPU_REVIEW_RECEIPT=/private/real-review-receipt.json
export AERYNZA_NEXT_GPU_TRAINING_APPROVED=YES
export AERYNZA_NEXT_GPU_OUTPUT_DIR=/persistent/aerynza_candidate_run01
bash scripts/launch_next_gpu_candidate.sh --run
```

5. If interrupted, back up and restore the **complete** checkpoint directory including adapter weights, optimizer, scheduler, RNG, Trainer step and frozen run-contract identity. Use `AERYNZA_NEXT_GPU_RESUME=auto` with the original output/run identity. An adapter-only parent transfer starts a **new training stage** with fresh optimizer state and a new output directory; do not conflate them.
6. Evaluation after training must compare the candidate against the best *verified* parent on untouched holdouts, new independent tasks, truth/privacy/guardian negatives, actual tool execution receipts, latency/cost and local-only network-disabled operation. Do not automatically promote or silently switch hosted-provider fallback off.
7. A failed gate produces a read-only diagnosis via `scripts/aerynza_training_optimizer.py`; fix code/provider/identity/DB causes **without retraining**. For genuine semantic errors, independently author new reviewed examples, freeze report and adapter hashes, obtain manual approval, and allow at most two bounded corrective stages via `scripts/run_aerynza_repair_stage.py`.

## Product parity gaps requiring engineering, not examples

A fine-tuned 1.7B model alone is not ChatGPT-level capability. The minimum viable self-reliant product also requires: signed AP and Nexus inference scopes, local tokenizer/embedding/retrieval and persistent memory, source verification and current-information tools, sandboxed code execution, browser automation with explicit consent, document/spreadsheet/media creation engines, identity-aware connected apps, transaction and calendar receipts, multimodal model runtimes, robust GPU serving/concurrency, durable background jobs, guarded learning/feedback, timeouts/retries/idempotency, audit logs and rollback.

Every advertised capability needs an implementation, positive/negative fixture, shell/resource grant, tool state, result artifact/provider receipt and independent reviewer gate. A self-hosted-only model can function disconnected; weather, banks, retail carts, live property listings, email and search cannot provide **current** external data without legitimate external services. Keep those dependencies explicit and degrade gracefully.

## Shipping rule

**No production promotion until the actual 640 requested capabilities are classified and tested individually,** and the independently prioritized critical subset has passed E2E implementation, permission, outcome and recovery gates. Classifying 640 is not the same as implementing 640. Do not expose claims that Aerynza trained itself to ChatGPT parity without reproducible third-party evidence.
