# Aerynza AI v251 — foundation, closed-system independence, and capability evidence

**Status:** candidate training package; not GPU-trained, human-reviewed, promoted or production-deployed.  
**Source:** original human-authored synthetic scenarios informed by the founder's canonical Aerynza contract of 2026-10-09.  
**Base:** existing pinned Qwen/Qwen3-1.7B LoRA pipeline.  
**Change control:** stacked on native draft PR #7; do not merge or deploy without its security and identity gates.

## Files

- `evals/training/aerynza_v251_foundation_train.jsonl` — 60 training examples, 10 balanced groups.
- `evals/training/aerynza_v251_foundation_heldout.jsonl` — 18 never-trained evaluation examples.
- `scripts/validate_v251_packages.py` — standard-library schema, leakage, vocabulary and 640-ID inventory validation.
- `scripts/build_v251_capability_validation_queue.py` — outputs a deterministic, individual evaluation/evidence queue for the current 640 *catalog* entries. This is **not** proof of 640 implemented skills.
- `scripts/launch_v251_foundation.sh` — preflight-only by default, manual approval required to train, no auto-promotion.
- `scripts/review_v251_candidate.py` — adapter responses to independent held-out prompts for human review.

## Why this format

The prior v249/v250 semantic-repair packages used targeted examples for failing cases and low-rate adapter updates on the Qwen3 1.7B base. The v20 LoRA trainer already (1) masks prompt tokens for assistant-only loss, (2) records the curriculum digest, (3) balances `package:shell` groups, (4) accepts separate held-out JSONL, and (5) refuses silent checkpoint overwrite. The next improvement is quality and coverage, **not merely volume or repeating generic "I can help" templates**.

The main app's older `services/training-curation.js` rejects some refusals merely because they start with "I can't". That heuristic is unsuitable for safety training. Do not bulk-import its output into v251 until it is replaced by context-sensitive adjudication, opt-in/retention verification, PII filtering, correct answer checks, and independent human sampling. Neither user chats nor proprietary frontier-model outputs are included in this package.

## Architecture fidelity

- **Aerynza AI** is the underlying reasoning and orchestration foundation, not an identity/permission authority.
- **AP** serves **AerynzaLife**, **AerynzaSprout**, and **AerynzaExecutive**.
- **Nexus** serves **NexusHome** and **AerynzaFamily**.
- **Creation** is a workspace pending explicit founder-approved AI ownership.
- **Aerynza Core** is a **subscription tier**, never a separate shell or bypass route.
- Subscriptions and add-ons affect features and quotas, not identity, resource ownership, guardian permissions, provider grants, or cross-shell consent.
- The current Python trainer's shell enum is legacy; `ap` plus executive boundary examples are only a **transitional training convention**. This does **not** implement signed `aerynza_executive` routing or entitlements. Align trainer/system prompts and runtime types to the canonical app contract before enabling new shell inference.
- Cross-scope Human Model/Living Thesis flow requires user-controlled, provenance-preserving consent. No automatic silent memory writes.

## Training groups

| Group | Learning objective |
|---|---|
| conversation | Direct, empathetic natural discussion without reflex coaching |
| tools_receipts | Prepared, approved, attempted, unknown and independently verified actions |
| privacy_auth | Signed actors, scopes, guardian consent, revocation and prompt-injection resistance |
| planning_reasoning | Correct arithmetic, tradeoffs, calibrated decisions, grounded experiments |
| research_tools | Search, sources, maps, documents, coding, media and images **only when tools exist** |
| safety_specialist | Emergency escalation, financial uncertainty, child safety and secret protection |
| shell_coordination | Distinct AP/Nexus responsibilities, workplace, home and family limits |
| ambient_memory | Device-specific presence, staleness, quiet hours, correction, no imagined data |
| creation | Writing, naming, idea development, accurate creative workflow claims |
| technical_autonomy | Self-hosting, reproducible evaluation, independent review, release controls |

These examples teach **transferable behavior**. A text-only LoRA file cannot grant working web browsing, image creation, audio transcription, email sending, bank APIs, slide/PDF rendering or autonomous deployments. Those each require a real tool adapter, resource authorization, runtime integration and independent receipt.

## 640-capability reconciliation

Run:
```bash
python3 scripts/build_v251_capability_validation_queue.py
```
This generates `evals/results/v251_capability_validation_queue.json`, with one ID-specific row for each of the 640 recorded abilities and a test checklist. The source `public/capability_report.json` has historical fields implying near-complete readiness but no valid production evidence timestamp; **those fields are ignored** in the queue. Treat every entry as unverified until a reviewer attaches (1) tool schema and correct dispatch, (2) tested entitlement and subject/resource scope, (3) provider availability and consent, (4) verified executor receipt and idempotency, (5) staging security/E2E, (6) production observation if claimed. Different capabilities can be delivered through local deterministic code, APIs or human workflows rather than GPU training.

## GPU preflight and handoff

```bash
# On a trusted checkout of the reviewed training branch:
python3 scripts/validate_v251_packages.py
bash scripts/launch_v251_foundation.sh

# After explicit human review of all 78 records, GPU readiness,
# vetted model-license/weight availability, and selected parent checkpoint:
export AERYNZA_V251_REVIEW_APPROVED=YES
export CUDA_VISIBLE_DEVICES=0
# Optional, only if an actual compatible reviewed adapter exists:
# export AERYNZA_V251_PARENT_ADAPTER=checkpoints/<reviewed-parent>
bash scripts/launch_v251_foundation.sh --run
```

The launcher pins `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, uses the existing LoRA/assistant-mask trainer, one conservative epoch and a lower learning rate (`2e-6`) to reduce regression risk. The existing `requirements-training.txt` uses Transformers 4.x, CUDA Torch and compatible PEFT versions; verify the exact GPU image with `pip freeze`, driver/VRAM, base-model license and tokenizer before launch. **Do not assume the most recent v250 candidate is best.** Select the checkpoint using original held-out/canonical gate receipts and a reviewer decision; if no trusted adapter is available, train an isolated baseline and compare.

Candidate training writes only its separate `checkpoints/aerynza_v251_foundation_candidate` path. No GPU machine, checkpoint, deployment, paid resource, Stripe event or production setting is changed by this commit. Neither `AERYNZA_V251_REVIEW_APPROVED` nor the script is release approval.

## Model gates and stop rules

1. **Data integrity:** run v251 validator; reject overlap, duplicates, missing shell/tier contracts and uncontrolled personal data.
2. **Training metrics:** compare loss with clean held-out examples; watch for memorization and regressions against known v248/v249/v250 failure cases.
3. **Canonical semantics:** run `scripts/evaluate_qwen_ascension_lora.py`; no false receipts, private-data or shell leakage, misleading safety refusals, or unearned access claims.
4. **640 catalog prompts:** run `scripts/evaluate_per_capability.py` **and** `scripts/evaluate_capability_execution.py`; these measure model response behaviors, not deployed provider effects. Do not report catalog passing as production execution.
5. **New held-out cases:** run `scripts/review_v251_candidate.py --adapter checkpoints/aerynza_v251_foundation_candidate` and review all generated outputs, especially privacy, child, cross-tenant, double-payment and emergency cases.
6. **Independent adversarial evaluation:** forged signed shell/subject, revoked token, absent guardian, stale memories, tool prompt injections, absent provider, duplicate webhook, unavailable network, worker restart and idempotent retry.
7. **Runtime gates:** signed AP/Nexus execution end to end; execution receipts, tier/privacy checks, latency, quality of multi-turn conversation and device UX in isolated staging.
8. **Human approval:** review every severe failure, benchmark delta, source license and incident rollback before any possible deployment. No automatic model promotion.

**Block promotion** on any false "sent/paid/verified/saved" assertion; sensitive information leakage; fail-open shell/tier/grant handling; child-safety regression; hazardous medical or finance certainty; unhandled timeout duplicates; or significant quality/latency regression. Production remains on the pre-existing model until independently approved.

## Independence roadmap (not delivered by v251)

Build a self-hosted orchestration gateway around pinned open-weight base models, specialized routing, offline retrieval, a consented memory and event graph, sandboxed tool execution, local device interfaces and controlled background jobs. Add multimodal models and engines as licensed modules, rather than claiming text-only SFT generates pixels, voices or verified actions. Independent Codex/Devin/security review should evaluate AP's changes; AP must not be sole judge of its own safety. Quantify model reasoning and domain gaps with receipts and repeatable benchmarks, not vendor feature-name comparisons.

**Working definition of "closed system":** the core can run without an external hosted LLM, while optional third-party services (banks, mail, maps, images, brokerage and payment networks) remain explicitly consented interfaces. It is not realistic to claim comparable breadth, tool inventory or reliability to any frontier assistant from 78 scenarios or a week of LoRA training.
