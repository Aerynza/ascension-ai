"""HTTP integration tests for staged signed request enforcement."""
import base64
import hashlib
import hmac
import json
import time

import httpx
import pytest

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
def signing(monkeypatch):
    key = b"s" * 32
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "true")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "test-service-token")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap")
    monkeypatch.setenv("ASCENSION_AI_TRUSTED_ISSUER", "aerynza-product")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_ID", "current")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_BASE64", base64.b64encode(key).decode())
    monkeypatch.setenv("ASCENSION_AI_REPLAY_REDIS_URL", "redis://localhost:6379/0")
    store = NonceStore()
    monkeypatch.setattr(api, "redis_nonce_store", lambda _url: store)
    return key


def auth_headers(body, key, *, nonce="nonce-1", shell="ap", method="POST", path="/v1/memory/candidates"):
    now = int(time.time())
    envelope = {
        "request_id": nonce, "issuer": "aerynza-product", "subject": "user-123",
        "shell": shell, "key_id": "current", "nonce": nonce,
        "issued_at": now - 1, "expires_at": now + 60,
        "http_method": method, "http_path": path,
        "payload_hash": payload_sha256(body),
    }
    signature = hmac.new(key, canonical_payload(envelope), hashlib.sha256).hexdigest()
    return {
        "Authorization": "Bearer test-service-token",
        "X-Aerynza-Envelope": json.dumps(envelope),
        "X-Aerynza-Signature": signature,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_length", [None, "1048577"])
async def test_oversized_body_stops_reading_before_remaining_chunks(signing, declared_length):
    consumed = []

    async def chunks():
        for index in range(4):
            consumed.append(index)
            yield b"x" * 600_000

    headers = {"Authorization": "Bearer test-service-token"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", content=chunks(), headers=headers)
    assert response.status_code == 413
    assert consumed == ([] if declared_length else [0, 1])


@pytest.mark.asyncio
async def test_small_chunked_signed_body_reaches_handler(signing):
    body = {"text": "Remember my preference"}
    encoded = json.dumps(body).encode()

    async def chunks():
        yield encoded[:8]
        yield encoded[8:]

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", content=chunks(), headers={
            **auth_headers(body, signing, nonce="chunked-body"), "Content-Type": "application/json",
        })
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_unsigned_production_post_rejected(signing):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/memory/candidates", json={"text": "hello"},
            headers={"Authorization": "Bearer test-service-token"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_request_accepted_once_only(signing):
    body = {"text": "Remember my preference"}
    headers = auth_headers(body, signing)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/v1/memory/candidates", json=body, headers=headers)
        replay = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert first.status_code == 200, first.text
    assert replay.status_code == 401


@pytest.mark.asyncio
async def test_payload_tampering_rejected(signing):
    headers = auth_headers({"text": "authorized"}, signing)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/memory/candidates", json={"text": "altered"}, headers=headers,
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_target_cannot_be_reused_on_different_route(signing):
    body = {"text": "hello"}
    headers = auth_headers(body, signing, nonce="different-route")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/iphone/inbox", json=body, headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_shell_conflict_rejected(signing):
    body = {"text": "hello", "shell": "nexus_family"}
    headers = auth_headers(body, signing, nonce="shell-conflict", shell="ap")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_replay_store_failure_fails_closed(signing, monkeypatch):
    class UnavailableNonceStore:
        def claim(self, key, ttl_seconds):
            raise ConnectionError("Replay store unavailable")

    monkeypatch.setattr(api, "redis_nonce_store", lambda _url: UnavailableNonceStore())
    body = {"text": "hello"}
    headers = auth_headers(body, signing, nonce="redis-outage")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_invalid_json_rejected_without_handler_execution(signing):
    body = {"text": "hello"}
    headers = auth_headers(body, signing, nonce="invalid-json")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/memory/candidates", content=b"{not valid json",
            headers={**headers, "Content-Type": "application/json"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unsigned_get_shell_catalog_denied_in_signed_mode(signing):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/v1/actions/catalog/ap",
            headers={"Authorization": "Bearer test-service-token"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_disallowed_shell_rejected_even_without_body_shell(signing):
    body = {"text": "hello"}
    headers = auth_headers(body, signing, nonce="disallowed-shell", shell="nexus_family")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_allowed_shell_succeeds_without_body_shell(signing):
    body = {"text": "hello"}
    headers = auth_headers(body, signing, nonce="allowed-shell", shell="ap")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert response.status_code == 200


@pytest.mark.parametrize("value", ["tru", "yes", "1", "TRUEE", "enabled"])
def test_malformed_signing_configuration_fails_closed(monkeypatch, value):
    from src.serving.http_envelope import signing_enabled
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", value)
    with pytest.raises(RuntimeError, match="ASCENSION_AI_REQUIRE_SIGNED_REQUESTS"):
        signing_enabled()


def test_explicit_signing_modes(monkeypatch):
    from src.serving.http_envelope import signing_enabled
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "true")
    assert signing_enabled() is True
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "false")
    assert signing_enabled() is False


@pytest.mark.asyncio
async def test_non_object_envelope_rejected(signing):
    body = {"text": "hello"}
    headers = auth_headers(body, signing, nonce="not-object")
    headers["X-Aerynza-Envelope"] = "[]"
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_get_shell_catalog_allowed(signing):
    path = "/v1/actions/catalog/ap"
    headers = auth_headers({}, signing, nonce="signed-get-catalog", method="GET", path=path)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path, headers=headers)
    assert response.status_code == 200, response.text


@pytest.mark.asyncio
async def test_signed_get_cannot_be_replayed(signing):
    path = "/v1/actions/catalog/ap"
    headers = auth_headers({}, signing, nonce="signed-get-replay", method="GET", path=path)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get(path, headers=headers)
        second = await client.get(path, headers=headers)
    assert first.status_code == 200, first.text
    assert second.status_code == 401


@pytest.mark.asyncio
async def test_unsigned_readiness_denied_in_signed_mode(signing):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/v1/readiness", headers={"Authorization": "Bearer test-service-token"}
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_signed_catalog_cannot_access_other_allowed_shell(signing, monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap,nexus_family")
    path = "/v1/actions/catalog/nexus_family"
    headers = auth_headers({}, signing, nonce="cross-shell-get", shell="ap", method="GET", path=path)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path, headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unsupported_method_rejected_in_signed_production(signing):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            "/v1/memory/candidates", json={"text": "hello"},
            headers={"Authorization": "Bearer test-service-token"},
        )
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_cors_preflight_rejects_untrusted_origin(signing):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.options(
            "/v1/memory/candidates",
            headers={
                "Origin": "https://not-authorized.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,x-aerynza-envelope,x-aerynza-signature",
            },
        )
    assert response.status_code in (400, 405)
    assert response.headers.get("access-control-allow-origin") != "https://not-authorized.example"


@pytest.mark.asyncio
async def test_private_get_query_string_cannot_escape_signed_target(signing):
    path = "/v1/actions/catalog/ap"
    headers = auth_headers({}, signing, nonce="query-string", method="GET", path=path)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path + "?resource=other-user", headers=headers)
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
async def test_production_hides_api_schema(signing, path):
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cross_runtime_node_signature_is_admitted_and_replay_rejected(signing, monkeypatch):
    from src.serving import envelope_validation, replay_guard
    monkeypatch.setattr(envelope_validation.time, "time", lambda: 1760000030)
    monkeypatch.setattr(replay_guard.time, "time", lambda: 1760000030)
    body = {"text": "Remember my preference"}
    envelope = {
        "request_id": "node-http-123", "issuer": "aerynza-product",
        "subject": "user:123", "shell": "ap", "key_id": "current",
        "nonce": "node-http-123", "issued_at": 1759999999,
        "expires_at": 1760000060, "http_method": "POST",
        "http_path": "/v1/memory/candidates",
        "payload_hash": "fa0ca8723bd848cf83f572154f5f86ae3559b73d20bf402d5316851b03ec7631",
    }
    headers = {
        "Authorization": "Bearer test-service-token",
        "X-Aerynza-Envelope": json.dumps(envelope),
        "X-Aerynza-Signature": "5b79d5c4cd531f32b5889f4da76fda25f2e844b93c642bbfc7320b71c858c253",
    }
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        accepted = await client.post("/v1/memory/candidates", json=body, headers=headers)
        replay = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert accepted.status_code == 200, accepted.text
    assert replay.status_code == 401


@pytest.mark.asyncio
async def test_memory_candidate_route_rejects_other_allowed_shell(signing, monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap,nexus_family,sprout")
    body = {"text": "Private preference"}
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for shell in ("nexus_family", "sprout"):
            headers = auth_headers(body, signing, nonce="memory-shell-" + shell, shell=shell)
            result = await client.post("/v1/memory/candidates", json=body, headers=headers)
            assert result.status_code == 401


@pytest.mark.asyncio
async def test_shell_aware_route_rejects_missing_or_forged_shell(signing, monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap,nexus_family")
    path = "/v1/retrieve"
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        body = {"query": "private information"}
        headers = auth_headers(body, signing, nonce="missing-shell", shell="nexus_family", path=path)
        missing = await client.post(path, json=body, headers=headers)
        assert missing.status_code == 401

        body = {"query": "private information", "shell": "ap"}
        headers = auth_headers(body, signing, nonce="forged-shell", shell="nexus_family", path=path)
        forged = await client.post(path, json=body, headers=headers)
        assert forged.status_code == 401

        body = {"query": "hello", "shell": "ap"}
        headers = auth_headers(body, signing, nonce="matched-shell", shell="ap", path=path)
        allowed = await client.post(path, json=body, headers=headers)
        assert allowed.status_code == 200, allowed.text


@pytest.mark.asyncio
async def test_legacy_chat_alias_requires_explicit_matching_shell(signing):
    body = {"messages": [{"role": "user", "content": "hello"}]}
    headers = auth_headers(body, signing, nonce="legacy-chat", path="/chat")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/chat", json=body, headers=headers)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_legacy_generate_denied_even_with_valid_signed_request(signing):
    body = {"prompt": "hello"}
    headers = auth_headers(body, signing, nonce="legacy-generate", path="/generate")
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/generate", json=body, headers=headers)
    assert response.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_shell", ["core", "creation"])
async def test_tier_and_workspace_cannot_authenticate_as_signed_shell(signing, monkeypatch, invalid_shell):
    # Even if an operator accidentally includes a non-shell in the configured
    # allowlist, it must not become a trusted user/workspace identity.
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap,core,creation")
    body = {"text": "private"}
    headers = auth_headers(
        body, signing, nonce=invalid_shell + "-not-shell",
        shell=invalid_shell, path="/v1/memory/candidates"
    )
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/memory/candidates", json=body, headers=headers)
    assert response.status_code == 401
