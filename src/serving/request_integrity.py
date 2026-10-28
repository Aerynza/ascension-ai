"""Canonical request payload integrity helpers for the planned service envelope."""
import hashlib
import rfc8785


def canonical_payload(payload: dict) -> bytes:
    # RFC 8785 uses ECMAScript numbers and UTF-16 key ordering on both runtimes.
    # This rejects unsafe integers, non-finite values and invalid Unicode.
    return rfc8785.dumps(payload)


def payload_sha256(payload: dict) -> str:
    return hashlib.sha256(canonical_payload(payload)).hexdigest()

