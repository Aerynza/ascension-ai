"""Strict Redis nonce claim semantics; no Redis service required."""
import pytest

from src.serving.replay_guard import RedisNonceStore, ReplayDetected, consume_verified_envelope


class FakeRedis:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def set(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.value


KEY = "aerynza:envelope:issuer:active:unique"


def test_atomic_redis_success_uses_nx_and_bounded_expiration():
    backend = FakeRedis(True)
    assert RedisNonceStore(backend).claim(KEY, 90) is True
    assert backend.calls == [((KEY, "1"), {"nx": True, "ex": 90})]


@pytest.mark.parametrize("result", [None, False])
def test_existing_nonce_is_rejected(result):
    assert RedisNonceStore(FakeRedis(result)).claim(KEY, 90) is False


@pytest.mark.parametrize("result", ["OK", b"OK", "1", 1, {}, object()])
def test_unknown_truthy_redis_result_never_claims_success(result):
    with pytest.raises(RuntimeError, match="Unrecognized Redis"):
        RedisNonceStore(FakeRedis(result)).claim(KEY, 90)


@pytest.mark.parametrize("key", [None, "", "not-a-signed-nonce", "aerynza:envelope:" + "x" * 500])
def test_invalid_nonce_key_rejected_before_io(key):
    backend = FakeRedis(True)
    with pytest.raises(ValueError, match="nonce key"):
        RedisNonceStore(backend).claim(key, 30)
    assert backend.calls == []


@pytest.mark.parametrize("ttl", [-1, 0, 301, 1.5, "60", True, None])
def test_invalid_expiry_rejected_before_io(ttl):
    backend = FakeRedis(True)
    with pytest.raises(ValueError, match="lifetime"):
        RedisNonceStore(backend).claim(KEY, ttl)
    assert backend.calls == []


def test_backend_network_error_propagates_fail_closed():
    backend = FakeRedis(error=ConnectionError("Redis offline"))
    with pytest.raises(ConnectionError, match="offline"):
        RedisNonceStore(backend).claim(KEY, 90)


def test_verified_envelope_rejects_replay_and_unrecognized_store_response():
    envelope = {
        "issuer": "aerynza", "key_id": "current", "nonce": "unique",
        "expires_at": 1200,
    }
    with pytest.raises(ReplayDetected):
        consume_verified_envelope(
            envelope, store=RedisNonceStore(FakeRedis(None)), now=1140,
        )
    with pytest.raises(RuntimeError, match="Unrecognized Redis"):
        consume_verified_envelope(
            envelope, store=RedisNonceStore(FakeRedis("OK")), now=1140,
        )
