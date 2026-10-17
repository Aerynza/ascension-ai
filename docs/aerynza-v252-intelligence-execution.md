# Aerynza v252: intelligence and execution targets

Status: 24 AI-authored synthetic SFT examples, pending independent review; 12 benchmark specifications requiring fixtures. No GPU training, inference comparison, production changes or parity proof.

The founder's target is assistant-level intelligence and execution across the Aerynza domains and the 640 capability catalog. This extension makes the target measurable; it cannot transfer a frontier model's weights, proprietary training data, internal reasoning or tool inventory. A 1.7B base plus a small LoRA curriculum is a specialization candidate. Whether a larger or different licensed base is needed must be determined from measured gaps, GPU memory and cost.

## Training contract

`evals/training/aerynza_v252_execution_train.jsonl` has four distinct examples in each of six groups: reasoning, execution, recovery, memory, verification, independence. The examples contain short worked explanations and observable procedures, not hidden reasoning traces. They use the existing trainer's user/assistant schema. Tool results inside examples are explicitly hypothetical or mock; this package does not implement executor dispatch.

`evals/training/aerynza_v252_execution_benchmark.json` is an independent prompt/rubric specification. Never feed it into SFT, checkpoint selection training, replay or failure-repair data. After evaluation prompts become exposed in repair data, retire them and author replacement holdouts. Prompt uniqueness checks detect exact normalized overlap only; separate reviewers must inspect semantic overlap and dataset quality.

Native Aerynza tasks come first. AP serves personal Life/Sprout/Executive contexts; Nexus serves Home/Family scope. The legacy trainer uses `ap` as transitional Executive context. Core is a subscription tier. Product runtime authority and signed routing remain independently enforced; training does not grant permissions.

## GPU handoff

1. Validate v251 and v252; review all new examples and benchmark families independently.
2. Choose the parent from actual v248/v249/v250/v251 evaluation receipts, not its version number. No result currently establishes the best checkpoint.
3. Use the existing pinned base/revision and locked training environment. Record model/license, tokenizer, hardware, dependency versions, seed, data hashes and adapter hash.
4. Use explicit curriculum paths. Validate v252 through `train_qwen_ascension_lora.py --curriculum evals/training/aerynza_v252_execution_train.jsonl --output-dir checkpoints/aerynza_v252_candidate --validate-only`. This checks schema, not tokenization or model quality.
5. Before any separately authorized GPU run, tokenize every example with the pinned tokenizer, enforce assistant-only masks and length limits, use an isolated output directory, and preserve earlier checkpoints. The v251 launcher does NOT train v252 automatically. Do not silently substitute this dataset into that launcher.
6. Review dataset mixing rather than training exclusively on 24 examples: replay reviewed native-domain examples to prevent forgetting, keep true holdouts separate, and compare single-domain improvements against regressions. Run the existing per-capability and execution model evaluations as described in v251.

## Fair comparison to this assistant

An independent reviewer should instantiate the 12 task specifications as executable sandbox fixtures and add unseen difficulty variants. Neither author-model answers nor reference texts alone are evidence of executor success. Separate the following tracks:

| Track | Evidence |
|---|---|
| Reasoning only | Same prompt/context and budget, arithmetic checked by code, blinded independent review for open answers |
| Agent execution | Same tool schemas, credentials/scopes, fixture state, time budget and allowed retries; inspect actual final state |
| Full product | Signed shell/runtime integration, scoped memories, entitlements, device continuation and provider receipts |
| Self-hosted independence | Block hosted LLM routes; inspect network/runtime evidence; distinguish optional external data services |

Record exact comparison model/version/date, candidate base/adapter hashes, dataset/fixture hash, tool versions, invocation settings and complete redacted action traces. Do not claim parity to an unnamed or unavailable reference model. Run multiple seeds/variants; show sample counts and uncertainty. Report reasoning correctness, verified task success, unnecessary clarification rate, recovery success, privacy/authorization failures, fabricated completion rate, latency percentiles and resource cost. Averages must not hide missing tools or severe failures.

Engineering goals for a future evaluated candidate: zero critical privacy/authorization/fabricated-completion failures in the evaluated sample; no native-domain regression; at least 90% verified success on the independently expanded Aerynza task suite; reference comparison within five percentage points on each tested family, with enough independent tasks to report confidence intervals. These are proposed gates, not measured scores or proof of universal intelligence. If confidence intervals are too wide, report insufficient evidence. Independent release review is still required.

## All 640 capabilities

Keep the v251 inventory queue. For every unique ID attach: task-specific inputs and expected outputs, actual tool schema/implementation, signed actor/resource/tier scope, positive fixture, denial fixture, timeout/retry fixture and independently verified result. Missing executors remain unsupported. Capability-ID accounting and model text quality do not establish production execution. Group families for reporting but never replace individual-ID evidence with one generic prompt.

## Remaining work

The new benchmark specifications need independent fixtures and an execution harness; they are not runnable end-to-end tests yet. Structured tool-call training requires a format compatible with the selected model and actual authenticated executor. Long-horizon tasks, multimodal engines, larger unseen evaluation suites, model comparison and GPU training are not delivered by this extension. Hosted fallback stays available until native replacement gates are proven. No automatic promotion.
