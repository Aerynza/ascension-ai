# Aerynza AI durable replay protection (existing PostgreSQL)

## Decision and scope

Use the existing Ascension-LifeOS Supabase PostgreSQL database as the shared
anti-replay authority. This needs no Render Key Value instance and adds no new
hosting subscription. Its availability and quota are nevertheless shared with
the product's database.

The private `aerynza_ai_security.signed_request_nonces` table is installed
by AerynzaLife migration `202610270001_ai_signed_replay_ledger.js`. It holds
only SHA-256 digests of `issuer:key-id:nonce`, a DB expiry time, and a creation
time. The unique primary key, not process-local memory, guarantees atomic
single-use consumption across all AI instances and service restarts.

A PostgreSQL outage, invalid credentials, privileged database role, or missing
table is **not** a reason to drop signing: production startup fails or signed
requests return HTTP 503. Reusing a verified nonce returns 401. The backend
must not automatically retry a non-idempotent request with a fresh nonce.

## Restricted database identity (one-time operator setup)

The migration creates NOLOGIN permission group `aerynza_ai_replay_writer`
with schema USAGE and select/insert/delete permissions restricted to the
private ledger under RLS. It intentionally does **not** create a stored
password or a broadly privileged AI credential.

The AerynzaLife migration `202610270002_ai_signed_replay_login_holder.js`
now creates the dedicated `aerynza_ai_replay_login` role as **NOLOGIN**
and grants only replay-writer membership. This is deliberately inert until
a release operator supplies a fresh strong secret from an approved secret
manager. After confirming the exact Supabase pooler host from Connect,
activate the existing role using an administrator connection:

```sql
ALTER ROLE aerynza_ai_replay_login
  LOGIN PASSWORD '<new random secret from your password manager>';
```

Do not execute this example with the literal placeholder. Do not grant
`postgres`, `service_role`, CREATEDB, CREATEROLE, or BYPASSRLS to the login.
Rotate the credential on exposure, and keep it out of GitHub, logs, and chat.

In Supabase Dashboard > Connect, copy the **session pooler** connection details
if the Render worker requires IPv4; do **not** guess the host from the region.
Supabase's pooler username format is `<role>.<project-ref>`. Use the
dedicated login role and real private password when constructing the URL.
Never reuse the application's admin-level `DATABASE_URL`.

Set these environment variables **on the AI service only**, while the PR
is still staged:

- `ASCENSION_AI_REPLAY_POSTGRES_URL`: connection URL for the dedicated
  replay LOGIN role; encrypted transport is forced with sslmode=require.
- `ASCENSION_AI_REPLAY_REDIS_URL`: **must be unset**. Both backends
  configured at once is a startup error.
- `ASCENSION_AI_TRUSTED_ISSUER`, `ASCENSION_AI_SIGNING_KEY_ID`,
  `ASCENSION_AI_SIGNING_KEY_BASE64`, `ASCENSION_AI_SERVICE_TOKEN`,
  and `ASCENSION_AI_SERVICE_SHELLS`: align with the product-side signer.
  Use new keys and keep them server-side.

No migration or environment setup automatically activates signing or switches
the AI model.

## Staged verification, in order

1. Confirm the migration appears in the live `_migrations` ledger and the
   private table has RLS enabled, exactly three writer policies, a unique
   primary key and an expiry index.
2. Run `python -m scripts.verify_postgres_replay` in a staging service with
   its private replay URL. It makes two requests with one generated nonce:
   first accepted, second denied, including over separate DB connections.
3. Run the native AI signing CI suite. Validate cross-runtime Node signature,
   wrong issuer/key/body, wrong shell, replay after service restart, two AI
   instances racing on a nonce, DB outage => 503, and expiry clock skew.
4. Audit **every** currently active product-side native AI caller, including
   readiness, streaming, media and legacy routes. Media GETs with unsigned
   query parameters are explicitly rejected by this receiver; either provide
   a signature-compatible endpoint or keep those flows on the previous path.
5. Coordinate product and AI deployments behind existing cutover flags with
   rollback to their currently verified live commits. Production signing
   requires an explicit operator go/no-go. Do not merge and restart the AI
   service before shared key material and the replay connection pass staging.

## Operations and cleanup

The client uses a new encrypted PostgreSQL connection per verified request,
with a three-second connect timeout, a transaction and atomic
`INSERT ... ON CONFLICT DO NOTHING RETURNING`. A bounded opportunistic cleanup
every 128 claims removes at most 250 rows whose expiry is more than five
minutes in the past. Cleanup cannot release unexpired nonces. Failed commits
never authorize requests. The service never logs connection strings or the
raw signed request payload.

If the shared DB is paused or unreachable, reject signed requests rather than
switching to in-memory replay tracking. The store grows with traffic until
successful cleanup; monitor row counts, DB quotas and connection saturation.

## Rollback

Leave the table and private role in place during a rollback; deleting them
does not make a running signed service safer. Restore the last verified
product and AI service release pair and disable signed cutover explicitly.
Do not force a new AI model or change Render billing as part of this work.
