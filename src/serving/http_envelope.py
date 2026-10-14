"""HTTP admission for opt-in Aerynza service-to-service signed requests."""
import base64
import json
import os
import re

from src.serving.envelope_gateway import admit_request
from src.serving.envelope_validation import InvalidEnvelope
from src.serving.replay_guard import RedisNonceStore

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")


def _decode_signing_key(encoded):
    if not isinstance(encoded, str):
        raise RuntimeError("Invalid signed-request key encoding")
    try:
        key = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise RuntimeError("Invalid signed-request key encoding") from exc
    if len(key) < 32 or base64.b64encode(key).decode("ascii") != encoded:
        raise RuntimeError("Signed-request key must use canonical base64 and contain at least 32 bytes")
    return key


def signing_enabled() -> bool:
    """Reject misspelled security-mode settings instead of silently disabling signing."""
    raw = os.getenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "").strip().lower()
    if raw in ("", "false"):
        return False
    if raw == "true":
        return True
    raise RuntimeError("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS must be true or false")


def load_signing_configuration():
    """Read key material only from the server environment, never request data."""
    issuer = os.getenv("ASCENSION_AI_TRUSTED_ISSUER", "").strip()
    key_id = os.getenv("ASCENSION_AI_SIGNING_KEY_ID", "").strip()
    encoded = os.getenv("ASCENSION_AI_SIGNING_KEY_BASE64", "").strip()
    redis_url = os.getenv("ASCENSION_AI_REPLAY_REDIS_URL", "").strip()
    if not (issuer and key_id and encoded and redis_url):
        raise RuntimeError("Signed requests require issuer, key ID, key, and shared Redis")
    if not _IDENTIFIER.fullmatch(issuer) or not _IDENTIFIER.fullmatch(key_id):
        raise RuntimeError("Invalid signed-request issuer or key ID")
    keys = {key_id: _decode_signing_key(encoded)}
    # Optional verification-only overlap; the product signer still emits one
    # active key. Removing an old entry retires it without accepting unknown IDs.
    previous_raw = os.getenv("ASCENSION_AI_PREVIOUS_SIGNING_KEYS_JSON", "").strip()
    if previous_raw:
        try:
            if len(previous_raw) > 4096:
                raise ValueError("Oversized previous key map")
            previous = json.loads(previous_raw)
            if not isinstance(previous, dict) or len(previous) > 3:
                raise ValueError("Invalid previous key map")
            for previous_id, previous_encoded in previous.items():
                if not _IDENTIFIER.fullmatch(previous_id) or previous_id in keys:
                    raise ValueError("Invalid or duplicate previous key ID")
                keys[previous_id] = _decode_signing_key(previous_encoded)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Invalid previous signed-request key configuration") from exc
    if not redis_url.startswith(("redis://", "rediss://")):
        raise RuntimeError("Replay store must use a Redis URL")
    return issuer, keys, redis_url


def check_http_envelope(headers, payload, *, config, nonce_store, method, path):
    """Verify cryptography and consume single-use nonce; errors reject request."""
    raw = headers.get("x-aerynza-envelope", "")
    signature = headers.get("x-aerynza-signature", "")
    if not raw or len(raw) > 4096:
        raise InvalidEnvelope("Missing or oversized envelope")
    try:
        envelope = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidEnvelope("Malformed envelope JSON") from exc
    if not isinstance(envelope, dict):
        raise InvalidEnvelope("Envelope must be an object")
    if envelope.get("http_method") != method or envelope.get("http_path") != path:
        raise InvalidEnvelope("Signed HTTP target mismatch")
    issuer, keys, _ = config
    identity = admit_request(
        envelope, payload, signature, keys=keys,
        trusted_issuer=issuer, nonce_store=nonce_store,
    )
    if path == "/v1/thesis" and payload.get("scope") == "human":
        if payload.get("subject_id") != identity["subject"]:
            raise InvalidEnvelope("Personal thesis subject must match authenticated caller")
    return identity


def redis_nonce_store(redis_url):
    import redis
    client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    return RedisNonceStore(client)

