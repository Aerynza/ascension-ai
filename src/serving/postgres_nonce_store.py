"""Durable PostgreSQL replay authority for signed Aerynza service requests.

A unique PostgreSQL primary key provides atomic insert-if-absent across all
AI processes and restarts. Only a digest is stored; never raw nonces,
service tokens, subjects, request bodies or envelopes.

Connection or transaction errors deliberately propagate to the middleware,
which returns HTTP 503 (fail closed).
"""
from __future__ import annotations

import hashlib
import threading
from urllib.parse import urlsplit


_INSERT = """
    INSERT INTO aerynza_ai_security.signed_request_nonces (nonce_hash, expires_at)
    VALUES (%s, clock_timestamp() + (%s::integer * INTERVAL '1 second'))
    ON CONFLICT (nonce_hash) DO NOTHING
    RETURNING nonce_hash
"""

# Opportunistic bounded cleanup cannot make a still-valid nonce reusable.
# The delete policy only permits rows expired by more than five minutes.
_PRUNE = """
    WITH expired AS (
      SELECT nonce_hash
      FROM aerynza_ai_security.signed_request_nonces
      WHERE expires_at < clock_timestamp() - INTERVAL '5 minutes'
      ORDER BY expires_at
      LIMIT 250
    )
    DELETE FROM aerynza_ai_security.signed_request_nonces AS used
    USING expired
    WHERE used.nonce_hash = expired.nonce_hash
"""


class PostgresNonceStore:
    """Atomic persistent nonce consumption using a private PostgreSQL table."""

    def __init__(self, database_url: str, *, connector=None, prune_every: int = 128):
        parsed = urlsplit(database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise RuntimeError("Invalid PostgreSQL replay store URL")
        if not parsed.username or not parsed.password:
            raise RuntimeError("Replay store requires a dedicated authenticated database role")
        self._database_url = database_url
        self._connector = connector
        self._prune_every = max(1, int(prune_every))
        self._count = 0
        self._lock = threading.Lock()

    def claim(self, key: str, ttl_seconds: int) -> bool:
        if not isinstance(key, str) or not key.startswith("aerynza:envelope:") or len(key) > 512:
            raise ValueError("Invalid replay nonce key")
        if type(ttl_seconds) is not int or ttl_seconds < 1 or ttl_seconds > 300:
            raise ValueError("Invalid replay nonce lifetime")
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        # Module is imported lazily so Redis-only and unit-test environments do
        # not require a DB driver until PostgreSQL replay mode is selected.
        if self._connector is None:
            import psycopg
            connector = psycopg.connect
        else:
            connector = self._connector
        with self._lock:
            self._count += 1
            cleanup = self._count % self._prune_every == 0
        # psycopg connection context commits on success, rolls back on any
        # exception, then closes. Never acknowledge a nonce before COMMIT.
        with connector(self._database_url, connect_timeout=3, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute(_INSERT, (digest, ttl_seconds))
                claimed = cur.fetchone() is not None
                if cleanup:
                    cur.execute(_PRUNE)
        return claimed
