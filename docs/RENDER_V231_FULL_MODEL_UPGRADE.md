# Upgrade Aerynza AI to full v231 (requires explicit paid-plan approval)

**New service:** `aerynza-ai` in the new Render account (`https://aerynza-ai.onrender.com`). Do not restore or send credentials to the suspended old service.

**Verified source:** GitHub [Aerynza/ascension-ai release v231-release](https://github.com/Aerynza/ascension-ai/releases/tag/v231-release) has `v231_release.gguf`, size **1,834,426,112 bytes** and SHA-256 `c784d75e2625ff4a28138eec1fca256787942ac40bca72b013270e5cb7999d10`. `config/model_profiles.json` points to the verified asset. No fine-tuning/training is required to recover these model weights.

**Compute:** Render Free and Starter each have 512 MB; Standard has 2 GB. The pinned `pro_v231` GGUF is ~1.83GB **before** context, KV cache, Python, native runtime and concurrency, and manifest specifies Pro 4GB minimum. Lowest documented matching Render plan is **`2c-4g` (2 CPU / 4GB, $85 per month as of October 2026)**. Do **not** change the plan/billing without a deliberate founder decision.

### Execution after paid-plan approval

1. In the **existing** new Render web service `aerynza-ai`, change compute plan from `free` to `2c-4g`. Do not create a duplicate billed service. Apply the environment settings shown in `render-v231-4gb.yaml` to the *existing* service without replacing existing bearer tokens or other secrets; particularly `ASCENSION_MODEL_PROFILE=pro_v231`, `ASCENSION_MODEL_CONTEXT=2048`, `ASCENSION_MODEL_THREADS=2`, `AERYNZA_AI_RAM_BUDGET_BYTES=3221225472`, `ASCENSION_AI_RAM_BUDGET_BYTES=3221225472`.
2. Build using `pip install -r requirements.txt && python scripts/download_model.py`. Download the release model during the build and verify its full SHA-256 before boot; the manifest downloader already checks this. If checksum verification fails, do not load a partial/modified model.
3. Keep provider/device execution routes disabled and production bearer-token authentication enabled. Verify public `/health` reports `candidate_ready=true` with `runtime.profile=pro_v231` and `replacement_ready=true` before promoting.
4. Run **authenticated** `/v1/readiness` plus real `/v1/intelligence` test cases, record latency, memory headroom, model label, and error rates. Previous evaluation metadata is not proof that the new service passes the same release gates.
5. Once healthy, configure the *existing LifeOS* new Render service with `AERYNZA_AI_URL=https://aerynza-ai.onrender.com` and matching `AERYNZA_AI_SERVICE_TOKEN` from the AI service (don't post the token in GitHub or chat). Set `AERYNZA_NATIVE_MODE=canary` or `fallback` first, preserving outside-model fallback and bounded timeout; promote to `primary` only after quality gates and consent boundaries pass.
6. Verify LifeOS authenticated AP conversation and `/health`; watch latency, CPU, memory and billing. Keep old service suspended. Do not turn off external model fallbacks without testing all 640 capability and integrated action gates.

### Current status

- New Render Free service created and starter model verified/serving.
- Full model asset verified on GitHub, download URL corrected and paid-plan blueprint prepared.
- **4GB upgrade not purchased or applied**; full v231 not running on Free.
- The connected Render actions can create new services and change environment variables but currently **do not expose an existing-service compute-plan update**. Use the service dashboard (or approved connected browser workflow) after $85/month is expressly approved.
