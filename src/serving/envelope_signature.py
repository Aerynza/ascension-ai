"""HMAC verification primitive for authenticated Aerynza service envelopes.

Use only behind authenticated service access. Nonce consumption and caller policy
must be enforced by the HTTP boundary before enabling production traffic.
"""
import hashlib
import hmac

from src.serving.envelope_validation import InvalidEnvelope, validate_metadata
from src.serving.request_integrity import canonical_payload, payload_sha256


def verify_signed_envelope(envelope, payload, signature, *, keys, trusted_issuer,
                           now=None):
    validate_metadata(envelope, issuer=trusted_issuer, now=now)
    if not hmac.compare_digest(envelope["payload_hash"], payload_sha256(payload)):
        raise InvalidEnvelope("Payload changed")
    if "shell" in payload and payload["shell"] != envelope["shell"]:
        raise InvalidEnvelope("Shell identity mismatch")
    key = keys.get(envelope["key_id"])
    if not isinstance(key, bytes) or len(key) < 32:
        raise InvalidEnvelope("Missing or weak signing key")
    if not isinstance(signature, str) or len(signature) != 64:
        raise InvalidEnvelope("Malformed signature")
    expected = hmac.new(key, canonical_payload(envelope), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise InvalidEnvelope("Signature mismatch")
    return True
