# Native request admission hardening — October 14, 2026

Draft #7 only. No model training, deployment, merge, production configuration or traffic change.

The signed production middleware previously buffered the whole authenticated request before enforcing its 1 MiB limit. It now rejects excessive Content-Length before consuming the body and stops reading oversized chunked bodies while preserving the bounded bytes for downstream JSON validation. Signature, subject/shell, nonce replay, release gates and external fallback policy remain required.

The regression initially consumed four 600 KB chunks; after the fix it consumes two without Content-Length and zero with advertised oversize. A valid chunked signed request still reaches the handler. The full native `tests/` suite passes 87 tests and 12 subtests locally; replacement contract evaluation and `compileall` pass. CI now also runs the required model-free replacement contracts and full source/script compilation.

These tests use an in-memory nonce fake and ASGI transport. They do not prove real Redis concurrency, server/proxy buffering, slow-client timeouts, rate limiting, GPU inference quality, resource-owner authorization across five product shells, latency SLOs or provider fallback under live outages. The replacement readiness gate must remain evidence-bound; a catalog count or model load is not parity. See LifeOS #109 and the cross-repository continuation report on draft #104 for release blockers. Exact-head CI evidence is recorded on the PR and coordination issue #40.
