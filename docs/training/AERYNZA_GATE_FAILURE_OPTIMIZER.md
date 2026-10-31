# Aerynza Gate-Failure Optimizer

Status: draft implementation; automated CPU-only preflight is not proof of real GPU training or production readiness.

## Recovery cycle

1. Train a candidate with durable checkpoints and saved model/dataset identity.
2. Run the existing canonical and receipt-truth gate; then capability text-response and execution-plan probes.
3. If a gate fails, create an immutable diagnosis that contains IDs and aggregated failure types only, not held-out prompts, expected answers, or model completions.
4. Triage critical truth/privacy failures separately from tokenizer/runtime failures, reasoning deficits and missing tool integrations.
5. Independently review root cause. If the model genuinely needs training, author new supervised examples from independent scenarios with approved native-domain replay; never copy from held-out tests.
6. Use a signed-off local approval manifest binding the report hash, parent adapter hash, reviewed curriculum hash and untouched heldout hash.
7. After explicit approval, run a SINGLE training stage from the old LoRA weights with reset optimizer state, in a new output directory, a compute budget, original gates, and follow-up failure report. No auto promotion.
8. At most two repair attempts per plan; critical security/receipt failures and exhausted budgets require manual review.

The runner uses a pinned Qwen3 1.7B model, and is not a replacement for the smaller 0.6B laptop experiment. No CPU/GPU hardware is provisioned or billed. It does not retrain just because an evaluation failed.

## Diagnose (CPU-only)

Run from the repository root, replacing bracketed names with the real directories:

    python3 scripts/aerynza_training_optimizer.py --training-receipt checkpoints/CANDIDATE/ascension_training_receipt.json --canonical evals/results/CANDIDATE_canonical.json --capability evals/results/CANDIDATE_capability.json --execution evals/results/CANDIDATE_execution.json --attempt 0 --max-attempts 2 --output evals/results/CANDIDATE_optimizer.json

If a gate was not executed, omit that report option; it will be recorded as not_run and NOT passed. Optional independent review is a separate report. The v20 pipeline calls the analyzer automatically on gate rejection.

## Create an independent reviewed repair dataset

Review the curriculum in evals/training/NEW_REPAIR.jsonl (minimum 20 records). Keep the evaluation dataset in evals/training/INDEPENDENT_HELDOUT.jsonl (minimum 8 records). The existing trainer checks schema and exact ID/prompt overlap; human review must also check semantic overlap, privacy, consent and retention.

NEVER use held-out questions, reference answers, generated failed responses, personal user conversations, or release-test rubrics as supervised labels. Use newly written analogous examples without copying the release fixtures.

## Generate an unapproved approval template

    python3 scripts/create_aerynza_repair_approval.py --plan evals/results/CANDIDATE_optimizer.json --parent-adapter checkpoints/CANDIDATE --training evals/training/NEW_REPAIR.jsonl --heldout evals/training/INDEPENDENT_HELDOUT.jsonl --output evals/results/CANDIDATE_repair_approval.json

It deliberately starts with approved=false and the three source/privacy attestations set to false. An independent reviewer must personally check and fill these. All paths/hashes are local; do not commit sensitive contents or private approval forms into the public repository.

## Dry-run the next candidate stage (no training)

    python3 scripts/run_aerynza_repair_stage.py --plan evals/results/CANDIDATE_optimizer.json --approval evals/results/CANDIDATE_repair_approval.json --parent-adapter checkpoints/CANDIDATE --training evals/training/NEW_REPAIR.jsonl --heldout evals/training/INDEPENDENT_HELDOUT.jsonl --output-dir checkpoints/NEXT_CANDIDATE --training-budget-minutes 120 --gate-budget-minutes 120

Add --run only after review and on an approved machine to execute ONE stage, then run existing unchanged release gates. Resource budgets constrain individual training and gate stages; there is no background daemon or infinite retraining.

An interrupted run should use the trainer's --resume-checkpoint auto to recover an exact same stage from a complete durable checkpoint, rather than invoking the new stage again. A new curriculum stage uses --resume-adapter to transfer trained weights but not the previous optimizer state.

## Boundaries

- Current 640 capability gates primarily evaluate text output and plans, not real tool execution. Genuine promotion requires actual implemented tools, receipts, timeouts, authorization, tier/guardian constraints, independent sandbox tests and human approval.
- Diagnostic hints need independent root-cause confirmation. Model fine-tuning cannot repair missing API providers or runtime defects.
- No automatic release. Do not relax thresholds or train on the evaluation suite to produce artificial passes.
- Local checkpoint storage and ignored evals/results must be backed up outside temporary cloud runtimes before they disappear.
- A model that passes all included gates remains blocked until separate production security and human approvals.
