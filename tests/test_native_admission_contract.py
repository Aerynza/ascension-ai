"""Offline signing-key overlap and personal-subject admission checks."""
import base64
import hashlib
import hmac
import json
import os
import time
import unittest
from unittest.mock import patch

from src.serving.http_envelope import check_http_envelope, load_signing_configuration
from src.serving.envelope_validation import InvalidEnvelope
from src.serving.request_integrity import canonical_payload, payload_sha256


class NativeAdmissionContract(unittest.TestCase):
    def setUp(self):
        self.key = b"s" * 32
        self.environment = {
            "ASCENSION_AI_TRUSTED_ISSUER": "aerynza-product",
            "ASCENSION_AI_SIGNING_KEY_ID": "current",
            "ASCENSION_AI_SIGNING_KEY_BASE64": base64.b64encode(self.key).decode(),
            "ASCENSION_AI_REPLAY_REDIS_URL": "redis://offline.invalid/0",
        }

    def headers(self, payload, subject="user-123", key_id="current", key=None):
        now = int(time.time())
        envelope = {
            "request_id": "request-1", "issuer": "aerynza-product",
            "subject": subject, "shell": "lifeos", "key_id": key_id,
            "nonce": "nonce-1", "issued_at": now, "expires_at": now + 60,
            "http_method": "POST", "http_path": "/v1/thesis",
            "payload_hash": payload_sha256(payload),
        }
        signature = hmac.new(key or self.key, canonical_payload(envelope), hashlib.sha256).hexdigest()
        return {"x-aerynza-envelope": json.dumps(envelope), "x-aerynza-signature": signature}

    def admit(self, payload, headers, config):
        class Store:
            def claim(self, key, ttl_seconds):
                return True
        return check_http_envelope(headers, payload, config=config, nonce_store=Store(), method="POST", path="/v1/thesis")

    def test_personal_thesis_is_bound_to_authenticated_subject(self):
        body = {"shell": "lifeos", "scope": "human", "subject_id": "user-else", "context": {}}
        config = ("aerynza-product", {"current": self.key}, "redis://offline.invalid/0")
        with self.assertRaises(InvalidEnvelope):
            self.admit(body, self.headers(body), config)
        body["subject_id"] = "user-123"
        self.assertEqual(self.admit(body, self.headers(body), config)["subject"], "user-123")

    def test_key_overlap_accepts_old_then_retirement_rejects_old(self):
        previous_key = b"p" * 32
        body = {"shell": "lifeos", "scope": "human", "subject_id": "user-123"}
        env = {**self.environment, "ASCENSION_AI_PREVIOUS_SIGNING_KEYS_JSON": json.dumps({"previous": base64.b64encode(previous_key).decode()})}
        with patch.dict(os.environ, env, clear=True):
            config = load_signing_configuration()
        self.assertEqual(set(config[1]), {"current", "previous"})
        self.assertEqual(self.admit(body, self.headers(body, key_id="previous", key=previous_key), config)["subject"], "user-123")
        with patch.dict(os.environ, self.environment, clear=True):
            retired = load_signing_configuration()
        with self.assertRaises(InvalidEnvelope):
            self.admit(body, self.headers(body, key_id="previous", key=previous_key), retired)

    def test_malformed_previous_keys_fail_closed(self):
        for encoded in ("not-json", "[]", '{"previous":"bad"}', '{"current":"'+base64.b64encode(self.key).decode()+'"}'):
            with self.subTest(encoded=encoded), patch.dict(os.environ, {**self.environment, "ASCENSION_AI_PREVIOUS_SIGNING_KEYS_JSON": encoded}, clear=True):
                with self.assertRaises(RuntimeError):
                    load_signing_configuration()


if __name__ == "__main__":
    unittest.main()

