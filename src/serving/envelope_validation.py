"""Validate service-envelope metadata before accepting scoped calls."""
import re
import time

_ID = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")
_FIELDS = ("request_id", "issuer", "subject", "shell", "key_id", "nonce",
           "issued_at", "expires_at", "payload_hash", "http_method", "http_path")


class InvalidEnvelope(ValueError):
    pass


def validate_metadata(envelope, *, issuer, now=None, lifetime=120, clock_skew=30):
    if not isinstance(envelope, dict):
        raise InvalidEnvelope("Envelope must be an object")
    if any(field not in envelope for field in _FIELDS):
        raise InvalidEnvelope("Missing envelope field")
    for field in ("request_id", "issuer", "subject", "shell", "key_id", "nonce"):
        value = envelope[field]
        if not isinstance(value, str) or not _ID.fullmatch(value):
            raise InvalidEnvelope("Invalid envelope identifier")
    if envelope["http_method"] not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
        raise InvalidEnvelope("Unsupported method")
    path = envelope["http_path"]
    if not isinstance(path, str) or not path.startswith("/") or len(path) > 512 or "?" in path:
        raise InvalidEnvelope("Invalid signed path")
    if envelope["issuer"] != issuer:
        raise InvalidEnvelope("Untrusted issuer")
    start, end = envelope["issued_at"], envelope["expires_at"]
    if type(start) is not int or type(end) is not int:
        raise InvalidEnvelope("Invalid timestamps")
    current = int(time.time()) if now is None else now
    if end <= start or end - start > lifetime:
        raise InvalidEnvelope("Invalid validity period")
    if start > current + clock_skew or end <= current - clock_skew:
        raise InvalidEnvelope("Envelope outside accepted time window")
    digest = envelope["payload_hash"]
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise InvalidEnvelope("Invalid payload digest")
    return True
