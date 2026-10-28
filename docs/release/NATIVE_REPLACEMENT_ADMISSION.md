# Native AI replacement admission — fail closed

The native service can load a model without proving that it is an acceptable
replacement for a provider-backed AP. **Candidate ready != replacement ready.**

**Staged, not deployed:** This draft now installs HMAC-signed HTTP admission,
user-and-shell validation, bounded request bodies, and an atomic PostgreSQL or
Redis nonce authority using the audited primitives from signed-admission
PR #7. Every private production request requires BOTH the service bearer
and a valid signed envelope; the public health route never certifies user
privacy or model replacement. Staging and production must configure
`ASCENSION_AI_REQUIRE_SIGNED_REQUESTS=true`,
`ASCENSION_AI_SERVICE_SHELLS`, the shared trusted issuer and signing key,
and exactly one durable nonce backend. The signed admission endpoint marks
its fields true **only after** a real verified request and nonce claim.
The live native deployment still runs an older commit without these changes.

LifeOS already contains the nonce-ledger and dedicated NOLOGIN holder-role
migrations, but migration execution and a least-privilege login credential
must be confirmed separately. Follow
`docs/security/POSTGRES_REPLAY_RUNBOOK.md` and do not share or reuse the
application's administrative database credentials.

**Remaining release HOLD:** No production promotion before signed LifeOS
integration, two isolated staging accounts, revoked-user and nonce-replay
testing, accessible private replay backend, load/queue recovery, and all
checkpoint-bound quality and safety evidence. The native-only background
cost fence in LifeOS PR #277 has not passed its full CI checks, so unattended
AI traffic remains disabled.

`/health` reports model load separately from `replacement_ready`; authenticated
`/v1/readiness` reports each actual evidence gate. Both must remain false by
default even after successful v231 startup.

## Operator-controlled, checkpoint-specific gate

Never commit a filled evidence report or operator deployment secrets.

1. Verify the exact `pro_v231` model file is boot-hash-checked, and record its
   configured GGUF SHA256 in a fresh JSON evidence report.
2. Independently test and document **each** gate against that checkpoint:
   `conversation_quality`, `shell_isolation`, `action_integrity`,
   `domain_reasoning`, `safety_privacy`, `interactive_latency`,
   `concurrency_recovery`, and `native_primary_canary`.
3. Write a private report with `profile`, `model_sha256`,
   `operator_approval: true`, ISO8601 UTC `verified_at`, and `gates` as a
   dictionary of `{"passed": true, "evidence_ref": "reviewed/test-artifact"}`
   for every gate. References must point to real reviewed evidence, not
   fabricated self-reported pass counts. Only approve after an independent
   review and the founder's separate traffic-cutover approval.
4. Provision environment variables **only on the intended deployment**:
   `AERYNZA_NATIVE_REPLACEMENT_APPROVED=1`,
   `AERYNZA_NATIVE_RELEASE_EVIDENCE_FILE=/private/path/release.json`, and
   `AERYNZA_NATIVE_RELEASE_EVIDENCE_SHA256=<sha256 of exact file bytes>`.
   The file must exist on the running instance. A path alone is insufficient.
5. Confirm the signed-in `/v1/readiness` response shows eight independent
   true gates and the actual live model fingerprint. Keep the provider-backed
   AP fallback and rollback available until observed canary quality, safety,
   latency and recovery all meet their preapproved thresholds.

A missing, stale (over 30 days), tampered, wrong-model or incomplete report
closes every unproved gate. Disabling model checksum verification on boot
also disables promotion. Passing a unit test or loading a GGUF does **not**
make the service production-replacement ready.

## Rollback

Unset `AERYNZA_NATIVE_REPLACEMENT_APPROVED` to close release gates
immediately. Existing production router/fallback settings are separate;
recheck them explicitly before any canary traffic or cutover. This change
does not enable an inference route, raise traffic, merge code, or deploy.
