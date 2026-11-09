# Distributed replay admission — bounded staging evidence

**Status:** Probe implementation only. The native model is not approved to replace outside AI. Neither an offline unit test nor a staging nonce acceptance test proves user/resource ownership, signed five-shell inference quality, cancellation recovery, or complete production admission.

## Existing authorized infrastructure only

The optional script `scripts/distributed_replay_probe.py` tests whether **separate backend client instances** atomically consume the same nonce. It accepts a real Redis/Redis-over-TLS URL or an authenticated PostgreSQL URL. The PostgreSQL client performs its existing least-privilege role preflight; no new schema is created. The probe writes two unique, **non-user** nonce records with a 120-second expiry, using the same prefix and nonce-store behavior as the serving code. PostgreSQL expired records use the existing bounded cleanup policy.

The probe intentionally **does not run during CI, server startup, deployment, or routine health checks**. Offline pytest regressions exercise its evidence logic, outage behavior, bounds and log redaction without opening a connection.

To use an **approved staging replay backend**, an operator first supplies the URL privately through the local process environment or a secret manager (never in a command pasted to chat, a PR, logs or an issue):

```bash
# Configure AERYNZA_AI_REPLAY_PROBE_URL outside shell history.
export AERYNZA_AI_REPLAY_PROBE_APPROVAL=staging-test-only
python scripts/distributed_replay_probe.py
```

The URL is deliberately excluded from console output and failure details. Redis uses distinct clients for competing claims. PostgreSQL uses distinct connections under the dedicated restricted role; the probe never changes end-user records, billing, subscriptions, model artifacts, or production configuration.

## What constitutes evidence

A valid run requires exactly **one** accepted nonce claim among eight concurrent independent clients, seven atomic rejections, rejection by a freshly created client, and acceptance of a **different** nonce. A connection error, non-boolean outcome, local per-client nonce store, privilege failure or timeout **fails the probe**. Keep the run's exact code commit, backend category, test environment, time, concurrency and summary; **do not** record connection URLs or credentials.

## Known limits and follow-up gates

- This probe uses fresh backend client instances, not two running native AI service replicas with signed HTTP envelopes. Real multi-replica request admission remains unproven until separately exercised.
- Redis outage fail-closed behavior requires an actual controlled staging outage test or correlated service logs; an offline mock is only a contract check.
- Proxy buffering, slow clients, connection quotas, body-limit handling, concurrent native inference, five-shell user ownership and consent, privacy-safe metrics, and checkpoint-bound release quality/safety/latency still belong to [issue #10](https://github.com/Aerynza/ascension-ai/issues/10).
- No outside-AI fallback removal or native release-flag change is authorized by these tests.
