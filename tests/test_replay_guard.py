import pytest

from src.serving.envelope_validation import InvalidEnvelope
from src.serving.replay_guard import ReplayDetected, consume_verified_envelope, RedisNonceStore


class FakeAtomicStore:
    def __init__(self):
        self.used = set()

    def claim(self, key, ttl_seconds):
        assert ttl_seconds > 0
        if key in self.used:
            return False
        self.used.add(key)
        return True


def envelope():
    return {"issuer": "app", "key_id": "key-1", "nonce": "nonce-1", "expires_at": 1050}


def test_claim_is_single_use():
    store = FakeAtomicStore()
    assert consume_verified_envelope(envelope(), store=store, now=1000)
    with pytest.raises(ReplayDetected):
        consume_verified_envelope(envelope(), store=store, now=1000)


def test_nonce_is_scoped_to_issuer_and_key():
    store = FakeAtomicStore()
    first = envelope()
    second = dict(first, issuer="other")
    assert consume_verified_envelope(first, store=store, now=1000)
    assert consume_verified_envelope(second, store=store, now=1000)


def test_expired_request_does_not_touch_store():
    store = FakeAtomicStore()
    with pytest.raises(InvalidEnvelope):
        consume_verified_envelope(envelope(), store=store, now=1050)
    assert not store.used


def test_persistence_failure_is_not_ignored():
    class Down:
        def claim(self, key, ttl_seconds):
            raise ConnectionError("nonce backend unavailable")
    with pytest.raises(ConnectionError):
        consume_verified_envelope(envelope(), store=Down(), now=1000)


def test_redis_uses_atomic_set_nx():
    class FakeRedis:
        def set(self, key, value, *, nx, ex):
            assert key.startswith("aerynza:envelope:")
            assert value == "1" and nx is True and ex > 0
            return True
    assert consume_verified_envelope(envelope(), store=RedisNonceStore(FakeRedis()), now=1000)
