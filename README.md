# Aerynza AI

Aerynza AI is the shared intelligence engine for AerynzaLife, AerynzaSprout, FamilyOS (household), and Nexus (extended network). It runs a pinned open-weight model locally through `llama.cpp`; the production runtime does not call OpenAI, Gemini, Anthropic, or another hosted model.

Current status: personal v19. The standalone service runs a pinned local model and is suitable for evaluation, local Windows/Android/iPhone control, and staged LifeOS routing. It is not yet approved as the only production intelligence provider; `/v1/readiness` deliberately reports that distinction.

## Product shells

- **AP** — one person’s conversational partner, assistant, task manager, coach, and mentor.
- **LifeOS** — domain intelligence that turns permissioned context into screen-ready insights and proposed next actions.
- **NexusHome** — household and co-parenting coordination with strict private-data boundaries.
- **NexusFamily** — FamilyOS and family-enterprise coordination; shared-chat behavior follows the FamilyOS consent contract.
- **Core** — neutral analysis without a relationship persona.

The same core recognizes FreeOS, LifeOS, LifeOS+, FounderOS, and LifeOS Infinite requests. The authenticated calling product—not the model—enforces subscriptions, permissions, and available actions.

## Executable capability domains

The native orchestration contract currently routes identity, scheduling, finance, health, learning, creation, relationships, family, home, career, emotional intelligence, documents, research, environment, business, and safety. These domains can be invoked through chat or as page/surface intelligence.

The AI proposes intelligence and actions. AerynzaLife and each authorized product shell own authentication, database access, entitlement checks, confirmations, action execution, evidence, and audit logs. A model response alone never proves that an action occurred.

## Local profiles

| Profile | Model | Intended host |
| --- | --- | --- |
| `starter` | SmolLM2 135M Q2 | 512 MB latency-first bootstrap validation |
| `standard` | Qwen2.5 0.5B Q4 | 2 GB service |
| `pro` | Qwen3 1.7B Q8 | 4 GB interactive service |
| `deep` | Qwen3 4B Q3_K_S | Separate 4 GB queued background service |

Every model file is pinned by repository revision and SHA-256 checksum in `config/model_profiles.json`.

The live and deep roles can be deployed as separate services. `render.deep-worker.example.yaml` prepares the 4B background worker without silently creating another paid instance. LifeOS routes `analysis` and `background` calls to it only when the owner supplies `ASCENSION_AI_DEEP_URL` and `ASCENSION_AI_DEEP_SERVICE_TOKEN`; live conversation remains on the faster service.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
python scripts/download_model.py
set ASCENSION_AI_TEST_TOKEN=choose-a-private-test-code
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and enter the private test code. The standalone page provides real token streaming, shell selection, and tier selection.

## Training environment

Model training is intentionally separated from the production web service so Render does not install the CUDA, Torch, Transformers, and dataset toolchain during every deploy.

```bash
pip install -r requirements-training.txt
python -u scripts/train_ascension_general_v5.py
```

Install both requirement files only on a machine that is meant to train models. The Render service uses `requirements.txt` exclusively for FastAPI and pinned GGUF inference.

Before continuing a completed v5 checkpoint, run its gate. The gate records
held-out loss and reviewable samples; it never promotes a model by itself.

```bash
python -u scripts/evaluate_native_checkpoint.py \
  --version ascension_causal_general_v7 \
  --output evals/results/v7_checkpoint_gate.json
```

Then build the reviewed, de-identified Ascension product corpus and continue
with the initialization selected by that exact gate receipt. A causal v2 base
that passes the gate normally resumes all learned weights. `transplant` is only
valid when the receipt explicitly recommends replacing incompatible attention.

```bash
python -u scripts/build_ascension_product_corpus.py
python -u scripts/train_ascension_product_v6.py \
  --base-version ascension_causal_general_v7 \
  --version ascension_product_v8 \
  --gate-result evals/results/v7_checkpoint_gate.json \
  --human-review-approved \
  --steps 20000
```

The product trainer writes a rotating recovery checkpoint every 2,500 steps.
Resume an interrupted run with `--resume-latest`. Training completion is not a
production-replacement receipt; held-out conversation, privacy, shell, action,
latency, and canary evaluations remain required.

## API

- `GET /health` — public readiness without secrets.
- `GET /model/info` — protected runtime metadata.
- `GET /v1/capabilities` — protected shell, tier, and domain registry.
- `POST /v1/intelligence` — protected native inference.
- `POST /v1/stream` — protected SSE inference with real token deltas.
- `POST /v1/surface-plan` — protected deterministic domain-to-surface proposal.
- `POST /chat` — compatibility alias for native inference.
- `POST /generate` — legacy prompt compatibility.

Example request:

```json
{
  "shell": "lifeos",
  "tier": "lifeos_plus",
  "messages": [{"role": "user", "content": "Review this creation idea."}],
  "context": {"workspace": "software", "verified_evidence": []},
  "surface": "creation/software",
  "mode": "planning",
  "allowed_capabilities": ["creation", "research", "business", "safety"]
}
```

## Connect LifeOS

Deploy this repository as its own Render service. Configure the LifeOS service with:

```text
ASCENSION_AI_URL=https://ascension-ai.onrender.com
ASCENSION_AI_SERVICE_TOKEN=<same private service token>
ASCENSION_NATIVE_MODE=primary
ASCENSION_AI_TIMEOUT_MS=120000
```

Routing modes:

- `primary` — Aerynza AI first, existing providers only if native inference fails.
- `fallback` — existing provider first, Aerynza AI available afterward.
- `only` — native intelligence only; use after replacement evaluations pass.
- `off` — native routing disabled.

The LifeOS router sends the relevant shell, subscription tier, product surface, allowed capability domains, and permission-scoped context. AP chat, Academy program generation, culinary intelligence, workout intelligence, content enrichment, translation, and Creation workspace requests share that router.

## Verification

```bash
python scripts/native_conversation_eval.py
python scripts/replacement_contract_eval.py
set ASCENSION_AI_BASE_URL=https://ascension-ai.onrender.com
set ASCENSION_AI_TEST_TOKEN=<private test token>
python scripts/replacement_readiness_eval.py
```

The smoke set contains 30 human prompts across AP, LifeOS, NexusHome, NexusFamily, and Core. The replacement suite adds explicit required concepts, forbidden claims, response-length limits, provider-integrity checks, hidden-reasoning checks, and an interactive-latency gate. Passing the smoke set alone does not authorize provider replacement.

## Security boundaries

- The public web page never receives the service token.
- Requests require a test token or server-to-server service token.
- Context is bounded and supplied per request; the model does not directly query LifeOS databases.
- The runtime makes no external model call and performs no external action.
- Model files are checksum-verified on boot.
- CORS is allow-list based.
- Production replacement remains disabled until quality, safety, privacy, action-integrity, and shell-boundary evaluations pass.

## What is not production-proven yet

The repository contains earlier experimental modules and roadmap concepts. They are not loaded by the native production path and must not be treated as working merely because a file or feature name exists. Multimodal perception, autonomous execution, long-term model training, distributed inference, and most legacy specialist modules still require implementation and evaluation.

© 2026 The B.E.I.N.G Group LLC. All rights reserved.
