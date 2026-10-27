# Aerynza AI — Render model-upgrade readiness

## Current safe production state

- Production service: **aerynza-ai**, branch `master`.
- Live Free service must remain on `ASCENSION_MODEL_PROFILE=starter`, with the
  256 MiB GGUF budget defined in `render.yaml`.
- The pinned `pro_v231` GGUF is approximately 1.83 GB and its manifest names
  `pro_4gb` as the minimum Render plan; a 512 MB Free instance cannot run it.
- Changing code or merging a PR never grants approval to start paid hosting.
- The current HTTP signing / durable nonce rollout is separate from the model
  upgrade. PR #7 remains subject to paired product/AI signing tests.

## Completed code prerequisites

1. `scripts/model_deploy_preflight.py` validates the pinned manifest, alias,
   SHA-256 declaration, model size and the explicit RAM budget **offline**.
2. `scripts/download_model.py` invokes the preflight before downloading any
   GGUF model. An oversized model is rejected before spending network/build
   resources. A downloaded artifact must match its pinned SHA-256.
3. `src/core/model_runtime.py` independently checks the budget again at boot
   and verifies the on-disk artifact checksum before loading.
4. CI runs `tests/test_model_deploy_preflight.py` alongside existing P0
   hardening checks. Passing the preflight means only *manifest and budget*
   pass. It does **not** prove benchmark quality, latency, provider access,
   tool execution, signing compatibility or business readiness.
5. The product's AP task receipts, persisted planning and permission checks
   retain their own validation independent of model identity.

## Upgrade-time checklist (not yet executed)

The owner explicitly stated they will upgrade Render **after code is ready**.
Do not resize or activate a paid instance until they do so.

1. Snapshot both current live Render commit SHAs, recent logs, database
   migration status and the existing AI environment (secret values must not
   enter logs, issues or commits).
2. Confirm current Render billing/pricing and plan selection with the owner.
   Plan names and quotas change; use the real Render workspace.
3. Upgrade **only** the `aerynza-ai` service to a verified sufficient-memory
   tier; do not assume an equivalent smaller-memory tier can run a 1.83 GB
   GGUF plus inference buffers. Check available RAM, CPU and disk.
4. Set `ASCENSION_MODEL_PROFILE=pro_v231` and align both
   `AERYNZA_AI_RAM_BUDGET_BYTES` and `ASCENSION_AI_RAM_BUDGET_BYTES` with
   the validated GGUF budget. Do not set a fictional budget larger than the
   instance's actual available memory.
5. Run `python scripts/model_deploy_preflight.py` before model download.
   Run the downloader in the build only after it passes. Verify pinned SHA-256
   against the real downloaded artifact, load health and inference-readiness.
6. Run replacement-quality gates on the deployed model, including safety,
   truthful external-action receipts, 640-capability evaluation, latency and
   crash recovery. Record exact model hash, configuration and results.
7. Verify all active product callers, signed request replay, permission
   revocation, and rollback. Stage changes to the product router separately;
   do not claim an AI response acted on external systems without receipts.
8. Only after a go/no-go review, enable the new production traffic.
   Keep the previous live Seed deployment SHA/config to roll back immediately.

## Inherent hosting limit for ambient AP

AerynzaLife's `AERYNZA_AMBIENT_ONLY_ENABLED=true` is separate from the
26-job global scheduler and runs only when the web service process is awake.
Render Free sleeps. To guarantee time-based planning without a user session,
provision an independent always-on scheduler/worker with Postgres locking
and least privilege after upgrading the relevant service. Do **not** solve
this with synthetic HTTP keep-alive traffic or skip consent checks.

## Email provider activation

AerynzaLife now supports Postmark directly over HTTPS, which is suitable for
Render Free network restrictions. Its `services/email-readiness.js` status
`configured_unverified` never means a message reached a mailbox.

Before activation, set a real `POSTMARK_SERVER_TOKEN` and a verified sender
`EMAIL_FROM` or `AERYNZA_EMAIL_FROM` in Render secrets. Do not use
`POSTMARK_API_TEST` for production delivery. Verify a single opt-in test
message and provider receipt before enabling automated transactional email.
Never commit tokens or silently fall back after ambiguous send failures.
