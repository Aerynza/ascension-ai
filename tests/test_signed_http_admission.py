"""Signed HTTP admission: bearer alone is insufficient for production personal AI."""
import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from src.serving import api
from src.serving.request_integrity import canonical_payload, payload_sha256


class NonceStore:
    def __init__(self):
        self.seen = set()

    def claim(self, key, ttl_seconds):
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


@pytest.fixture
def signed_service(monkeypatch):
    key = b"z" * 32
    monkeypatch.setenv("AERYNZA_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "true")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "internal-service-secret")
    monkeypatch.setenv("ASCENSION_AI_TRUSTED_ISSUER", "aerynza-product")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_ID", "current")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_BASE64", base64.b64encode(key).decode())
    monkeypatch.setenv("ASCENSION_AI_REPLAY_REDIS_URL", "redis://unreachable.invalid/0")
    monkeypatch.delenv("ASCENSION_AI_REPLAY_POSTGRES_URL", raising=False)
    store = NonceStore()
    monkeypatch.setattr(api, "replay_nonce_store", lambda _url: store)
    return key, store


def headers(body, key, *, path="/v1/memory/candidates", method="POST",
            subject="user:17", shell="ap", nonce="unique-nonce"):
    current = int(time.time())
    envelope = {
        "request_id": nonce, "issuer": "aerynza-product",
        "subject": subject, "shell": shell, "key_id": "current",
        "nonce": nonce, "issued_at": current - 1, "expires_at": current + 60,
        "http_method": method, "http_path": path,
        "payload_hash": payload_sha256(body),
    }
    signature = hmac.new(key, canonical_payload(envelope), hashlib.sha256).hexdigest()
    return {
        "Authorization": "Bearer internal-service-secret",
        "X-Aerynza-Envelope": json.dumps(envelope),
        "X-Aerynza-Signature": signature,
    }


def test_shared_bearer_without_signed_user_is_rejected(signed_service):
    with TestClient(api.app) as client:
        denied = client.post("/v1/memory/candidates", json={"text": "private"},
                             headers={"Authorization": "Bearer internal-service-secret"})
        assert denied.status_code == 401


def test_signed_account_is_admitted_once_and_replay_is_rejected(signed_service):
    key, _ = signed_service
    body = {"text": "My private goal"}
    signed = headers(body, key)
    with TestClient(api.app) as client:
        first = client.post("/v1/memory/candidates", json=body, headers=signed)
        second = client.post("/v1/memory/candidates", json=body, headers=signed)
        assert first.status_code == 200, first.text
        assert second.status_code == 401


def test_payload_and_route_tampering_are_rejected(signed_service):
    key, _ = signed_service
    signed = headers({"text": "authorized"}, key)
    with TestClient(api.app) as client:
        altered = client.post("/v1/memory/candidates", json={"text": "forged"}, headers=signed)
        assert altered.status_code == 401
        another_route = client.post("/v1/thesis", json={"text": "authorized"}, headers=signed)
        assert another_route.status_code == 401


def test_diagnostic_service_principal_cannot_access_private_inference(signed_service):
    key, _ = signed_service
    body = {"messages": [{"role": "user", "content": "Private"}], "shell": "ap"}
    signed = headers(body, key, subject="service:readiness", path="/v1/intelligence")
    with TestClient(api.app) as client:
        denied = client.post("/v1/intelligence", json=body, headers=signed)
        assert denied.status_code == 401


def test_unsupported_account_identifiers_are_rejected(signed_service):
    key, _ = signed_service
    body = {"text": "private"}
    with TestClient(api.app) as client:
        for index, subject in enumerate(("user:0", "user:01", "other:17", "service:worker", "0")):
            signed = headers(body, key, subject=subject, nonce=f"invalid-{index}")
            denied = client.post("/v1/memory/candidates", json=body, headers=signed)
            assert denied.status_code == 401, (subject, denied.status_code)


def test_shell_swap_is_rejected(signed_service):
    key, _ = signed_service
    body = {"text": "private"}
    signed = headers(body, key, shell="nexus_family")
    with TestClient(api.app) as client:
        denied = client.post("/v1/memory/candidates", json=body, headers=signed)
        assert denied.status_code == 401


def test_authentic_diagnostic_probe_attests_admission_not_model_quality(signed_service):
    key, _ = signed_service
    signed = headers({}, key, method="GET", path="/v1/readiness",
                     subject="service:readiness")
    with TestClient(api.app) as client:
        result = client.get("/v1/readiness", headers=signed)
        assert result.status_code == 200, result.text
        data = result.json()
        assert data["signed_actor_admission"] == {
            "enforced": True,
            "durable_replay_protection": True,
            "account_subject_verified": True,
        }
        assert data["native_inference_only"] is True
        assert data["outside_provider"] is False
        assert data["replacement_ready"] is False


def test_private_session_still_disallowed_even_with_valid_account_signature(signed_service):
    key, _ = signed_service
    body = {"session_id": "shared-id", "context": {"private": "account-17"}}
    signed = headers(body, key, path="/v1/session/context")
    with TestClient(api.app) as client:
        denied = client.post("/v1/session/context", json=body, headers=signed)
        assert denied.status_code == 403
        assert not api.SESSION_CONTEXTS


def test_nonce_backend_outage_never_falls_back_to_bearer(signed_service, monkeypatch):
    key, _ = signed_service
    def unavailable(_url):
        raise ConnectionError("Replay database unavailable")
    monkeypatch.setattr(api, "replay_nonce_store", unavailable)
    body = {"text": "private"}
    signed = headers(body, key)
    with TestClient(api.app) as client:
        result = client.post("/v1/memory/candidates", json=body, headers=signed)
        assert result.status_code == 503


def test_missing_signing_mode_in_production_fails_closed(monkeypatch):
    monkeypatch.setenv("AERYNZA_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "internal-service-secret")
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "false")
    with TestClient(api.app) as client:
        result = client.post("/v1/memory/candidates", json={"text": "private"},
                             headers={"Authorization": "Bearer internal-service-secret"})
        assert result.status_code == 503


def test_oversized_signed_payload_cannot_reach_handler(signed_service):
    key, _ = signed_service
    body = {"text": "x" * 1_100_000}
    signed = headers(body, key)
    with TestClient(api.app) as client:
        result = client.post("/v1/memory/candidates", json=body, headers=signed)
        assert result.status_code == 413


def test_shell_not_in_deployment_allowlist_cannot_admit_user(signed_service, monkeypatch):
    key, _ = signed_service
    body = {"text": "private"}
    signed = headers(body, key, nonce="shell-permission")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "nexus_family")
    with TestClient(api.app) as client:
        result = client.post("/v1/memory/candidates", json=body, headers=signed)
        assert result.status_code == 401


def test_missing_or_invalid_shell_allowlist_fails_closed(signed_service, monkeypatch):
    key, _ = signed_service
    body = {"text": "private"}
    for index, raw in enumerate(("", "ap,ap", "core", "unknown", "ap,")):
        monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", raw)
        signed = headers(body, key, nonce=f"bad-shells-{index}")
        with TestClient(api.app) as client:
            result = client.post("/v1/memory/candidates", json=body, headers=signed)
            assert result.status_code == 503, (raw, result.status_code)


def test_thesis_user_subject_cannot_be_cross_account(signed_service):
    key, _ = signed_service
    body = {"shell": "ap", "scope": "human", "subject_id": "user:18", "context": {}}
    signed = headers(body, key, path="/v1/thesis", nonce="cross-user-thesis",
                     subject="user:17")
    with TestClient(api.app) as client:
        result = client.post("/v1/thesis", json=body, headers=signed)
        assert result.status_code == 401


def test_numeric_lifeos_account_subject_is_compatible(signed_service):
    key, _ = signed_service
    body = {"text": "Never publish my private profile"}
    signed = headers(body, key, subject="17", nonce="numeric-lifeos-user")
    with TestClient(api.app) as client:
        result = client.post("/v1/memory/candidates", json=body, headers=signed)
        assert result.status_code == 200, result.text
