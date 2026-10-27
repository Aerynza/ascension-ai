import hashlib
import hmac

import pytest

from src.serving.envelope_signature import verify_signed_envelope
from src.serving.envelope_validation import InvalidEnvelope
from src.serving.request_integrity import canonical_payload, payload_sha256


def example():
    body = {"shell": "ap", "query": "test"}
    envelope = {
        "request_id": "request-1", "issuer": "aerynza-app", "subject": "user-1",
        "shell": "ap", "key_id": "key-1", "nonce": "unique-1",
        "issued_at": 1000, "expires_at": 1060, "payload_hash": payload_sha256(body),
        "http_method": "POST", "http_path": "/v1/memory/candidates",
    }
    key = b"a" * 32
    signature = hmac.new(key, canonical_payload(envelope), hashlib.sha256).hexdigest()
    return body, envelope, key, signature


def verify(body, envelope, key, signature):
    return verify_signed_envelope(
        envelope, body, signature, keys={"key-1": key},
        trusted_issuer="aerynza-app", now=1030,
    )


def test_valid_signature():
    assert verify(*example())


def test_tampered_payload_rejected():
    body, envelope, key, signature = example()
    with pytest.raises(InvalidEnvelope):
        verify({"shell": "nexus_family"}, envelope, key, signature)


def test_tampered_shell_rejected():
    body, envelope, key, signature = example()
    envelope["shell"] = "nexus_family"
    with pytest.raises(InvalidEnvelope):
        verify(body, envelope, key, signature)


def test_wrong_key_rejected():
    body, envelope, key, signature = example()
    with pytest.raises(InvalidEnvelope):
        verify(body, envelope, b"b" * 32, signature)


def test_expired_envelope_rejected():
    body, envelope, key, signature = example()
    envelope["expires_at"] = 1001
    with pytest.raises(InvalidEnvelope):
        verify(body, envelope, key, signature)


def test_unknown_issuer_rejected():
    body, envelope, key, signature = example()
    envelope["issuer"] = "unknown-service"
    with pytest.raises(InvalidEnvelope):
        verify(body, envelope, key, signature)
