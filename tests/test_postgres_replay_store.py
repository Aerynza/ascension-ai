"""Offline checks for durable cross-process Postgres anti-replay admission."""
import base64
import hashlib
import threading

import pytest

from src.serving import http_envelope
from src.serving.postgres_nonce_store import PostgresNonceStore
from src.serving.replay_guard import ReplayDetected, consume_verified_envelope


URL = "postgresql://replay_writer:test-secret@db.example.test:5432/postgres"


class FakePostgres:
    """Models the unique primary key of a real shared PostgreSQL database."""
    def __init__(self):
        self.claimed = set()
        self.lock = threading.Lock()
        self.statements = []
        self.closed = 0
        self.fail = False

    def connect(self, dsn, *, connect_timeout, sslmode):
        assert dsn == URL and connect_timeout == 3 and sslmode == "require"
        if self.fail:
            raise ConnectionError("Postgres is offline")
        outer = self

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                # This fake returns itself as both cursor and connection;
                # count the physical connection close only once.
                if not getattr(self, "_closed", False):
                    outer.closed += 1
                    self._closed = True

            def cursor(self):
                return self

            def execute(self, sql, params=None):
                outer.statements.append((sql, params))
                self.result = None
                if "INSERT INTO aerynza_ai_security.signed_request_nonces" in sql:
                    assert "ON CONFLICT (nonce_hash) DO NOTHING" in sql
                    assert "RETURNING nonce_hash" in sql
                    digest, ttl = params
                    assert isinstance(digest, bytes) and len(digest) == 32
                    assert isinstance(ttl, int) and 1 <= ttl <= 300
                    with outer.lock:
                        if digest not in outer.claimed:
                            outer.claimed.add(digest)
                            self.result = (digest,)

            def fetchone(self):
                return self.result

        return Connection()


def envelope(nonce="nonce-1"):
    return {"issuer": "app", "key_id": "key-1", "nonce": nonce, "expires_at": 1050}


def test_postgres_nonce_single_use_survives_new_store_instance_and_never_stores_plaintext():
    database = FakePostgres()
    first = PostgresNonceStore(URL, connector=database.connect)
    second = PostgresNonceStore(URL, connector=database.connect)
    assert consume_verified_envelope(envelope(), store=first, now=1000)
    with pytest.raises(ReplayDetected):
        consume_verified_envelope(envelope(), store=second, now=1000)
    assert database.closed == 2
    digest = next(iter(database.claimed))
    assert digest == hashlib.sha256(b"aerynza:envelope:app:key-1:nonce-1").digest()
    assert not any(b"nonce-1" in key for key in database.claimed)


def test_issuer_and_key_separate_nonce_identity_and_expired_requests_never_touch_db():
    database = FakePostgres()
    store = PostgresNonceStore(URL, connector=database.connect)
    assert consume_verified_envelope(envelope(), store=store, now=1000)
    assert consume_verified_envelope(dict(envelope(), issuer="another-issuer"), store=store, now=1000)
    assert consume_verified_envelope(dict(envelope(), key_id="new-key"), store=store, now=1000)
    with pytest.raises(Exception):
        consume_verified_envelope(envelope("late"), store=store, now=1050)
    assert database.closed == 3


def test_store_failure_propagates_and_never_downgrades_to_in_memory_replay():
    database = FakePostgres()
    store = PostgresNonceStore(URL, connector=database.connect)
    database.fail = True
    with pytest.raises(ConnectionError):
        consume_verified_envelope(envelope(), store=store, now=1000)
    assert not database.claimed


def test_cleanup_is_bounded_and_only_removes_rows_older_than_grace_period():
    database = FakePostgres()
    store = PostgresNonceStore(URL, connector=database.connect, prune_every=2)
    assert store.claim("aerynza:envelope:app:key-1:n1", 60)
    assert store.claim("aerynza:envelope:app:key-1:n2", 60)
    prune = [sql for sql, _ in database.statements if "WITH expired AS" in sql]
    assert len(prune) == 1
    assert "LIMIT 250" in prune[0]
    assert "INTERVAL '5 minutes'" in prune[0]


@pytest.mark.parametrize("ttl", [0, -1, 301, "20", True])
def test_invalid_ttl_rejected(ttl):
    store = PostgresNonceStore(URL, connector=lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError):
        store.claim("aerynza:envelope:app:key-1:n1", ttl)


def test_postgres_config_requires_exactly_one_backend_and_never_accepts_unsigned_downgrade(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_TRUSTED_ISSUER", "aerynza-product")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_ID", "current")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_BASE64", base64.b64encode(b"x" * 32).decode())
    monkeypatch.setenv("ASCENSION_AI_REPLAY_POSTGRES_URL", URL)
    monkeypatch.delenv("ASCENSION_AI_REPLAY_REDIS_URL", raising=False)
    issuer, keys, location = http_envelope.load_signing_configuration()
    assert issuer == "aerynza-product"
    assert len(keys["current"]) == 32
    assert location == URL
    fake = FakePostgres()
    monkeypatch.setattr(http_envelope, "postgres_nonce_store", lambda _: PostgresNonceStore(URL, connector=fake.connect))
    assert http_envelope.replay_nonce_store(location).claim("aerynza:envelope:app:key:n1", 60)
    monkeypatch.setenv("ASCENSION_AI_REPLAY_REDIS_URL", "redis://127.0.0.1:6379/0")
    with pytest.raises(RuntimeError, match="exactly one"):
        http_envelope.load_signing_configuration()
    monkeypatch.delenv("ASCENSION_AI_REPLAY_POSTGRES_URL")
    monkeypatch.delenv("ASCENSION_AI_REPLAY_REDIS_URL")
    with pytest.raises(RuntimeError, match="exactly one"):
        http_envelope.load_signing_configuration()
