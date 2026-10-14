"""Strict service-envelope admission contract.

Call this only with the already authenticated request's parsed JSON body.
Caller authentication, subject ownership and permission scope remain separate
required production gates. Do not treat this primitive as an HTTP integration.
"""
from src.serving.envelope_signature import verify_signed_envelope
from src.serving.replay_guard import consume_verified_envelope


def admit_request(envelope, body, signature, *, keys, trusted_issuer, nonce_store, now=None):
    verify_signed_envelope(
        envelope, body, signature, keys=keys, trusted_issuer=trusted_issuer, now=now
    )
    consume_verified_envelope(envelope, store=nonce_store, now=now)
    return {
        "request_id": envelope["request_id"],
        "issuer": envelope["issuer"],
        "subject": envelope["subject"],
        "shell": envelope["shell"],
    }
