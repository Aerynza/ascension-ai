# Two-core v231 local inference speed gate

This is a **measured experiment**, not an assumption that a larger llama.cpp
prompt batch is automatically faster. No changes to paid Render plan,
quantization, model checksum, tokenizer, model weights or user credentials.

## Configuration

- Default: `n_batch=16` (production behavior preserved).
- Set `AERYNZA_NATIVE_FAST_PREFILL=true` to select `n_batch=64` **only when**
  `ASCENSION_MODEL_BATCH` and `AERYNZA_MODEL_BATCH` are not set.
- Explicit `ASCENSION_MODEL_BATCH=64` overrides the fast-prefill default.
- Operators can test 16, 32, 64 and 128; all values are bounded to 8–256,
  never larger than the configured context.
- Keep `ASCENSION_MODEL_THREADS=2` and `ASCENSION_MODEL_THREADS_BATCH=2`
  on the existing two-core instance until a benchmark supports a change.
- This config is **read only on model load**. Changing an environment variable
  causes a service restart/redeploy; don't rotate settings during live user
  conversations.

## Side-by-side benchmark

On a staging clone / isolated operator session with the **same 2c-4GB hardware,
same v231 GGUF and checksum, and no live AP conversations**:

```bash
ASCENSION_MODEL_PROFILE=pro_v231 python scripts/benchmark_native_prefill.py --batch 16 --trials 5
ASCENSION_MODEL_PROFILE=pro_v231 python scripts/benchmark_native_prefill.py --batch 64 --trials 5
```

**Run each command as a separate process**; never hold two copies of the
1.8 GB GGUF in 4 GB RAM. The benchmark calls the local model, does not call
any external providers, and prints aggregate latency/counts only. Its
derived throughput includes prefill time and is not labeled raw decode speed.

Record:

1. Last-deployed model and source commit; no silent model changes.
2. RAM peak and CPU utilization for each batch size.
3. Median and p95 total completion latency for the same synthetic prompt.
4. After safe native-stream PR #21 is deployed and AP client #558 is verified,
   compare actual user-visible first-token p50/p95, native queue wait, and
   total latency on separately consented, anonymized test cases.
5. Two account concurrent requests, correctness/receipt gates, and model
   context overflow / empty-response rate.

Keep 16 if 64 is slower, causes RAM pressure, hangs, or fails any regression
test. Restore `AERYNZA_NATIVE_FAST_PREFILL=false` / prior explicit
`ASCENSION_MODEL_BATCH` to roll back. Do not reduce prompt grounding or
model output quality merely to show a shorter latency number.

## Not a benchmark result

GitHub unit tests check parameter bounds and compatibility without loading
v231. **They do not establish a production latency improvement.** Do not
describe this as deployed or faster until comparative GGUF measurements exist.
