"""Atomic anti-replay gate for verified signed requests.

The store must implement atomic insert-if-absent with expiry across every
service process. Never use process memory as a production nonce authority.
"""
import time

from src.serving.envelope_validation import InvalidEnvelope


class ReplayDetected(InvalidEnvelope):
    pass


def consume_verified_envelope(envelope, *, store, now=None):
    """Claim a verified request nonce exactly once using a shared store.

    Store contract: claim(key: str, ttl_seconds: int) -> bool must be
    atomic, and raise on any persistence/network failure (fail closed).
    Only call after signature and issuer validation.
    """
    current = int(time.time()) if now is None else now
    expires_at = envelope["expires_at"]
    if type(expires_at) is not int or expires_at <= current:
        raise InvalidEnvelope("Expired envelope")
    key = "aerynza:envelope:" + envelope["issuer"] + ":" + envelope["key_id"] + ":" + envelope["nonce"]
    ttl = expires_at - current + 30
    if not store.claim(key, ttl):
        raise ReplayDetected("Request was already consumed")
    return True


class RedisNonceStore:
    """Redis SET NX EX provides atomic cross-worker nonce consumption."""

    def __init__(self, redis_client):
        self.redis = redis_client

    def claim(self, key, ttl_seconds):
        # Match PostgreSQL's strict contract. Bad keys/TTLs must not reach
        # Redis, and unknown server/client return types cannot authorize a
        # signed request just because they are truthy.
        if not isinstance(key, str) or not key.startswith("aerynza:envelope:") or len(key) > 512:
            raise ValueError("Invalid replay nonce key")
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 300:
            raise ValueError("Invalid replay nonce lifetime")
        result = self.redis.set(key, "1", nx=True, ex=ttl_seconds)
        # redis-py returns True on SET NX success and None when the key
        # already exists. False is also an explicit rejection. The driver
        # raises on timeouts/network failures; do not catch these errors.
        if result is True:
            return True
        if result is None or result is False:
            return False
        raise RuntimeError("Unrecognized Redis replay claim response")
