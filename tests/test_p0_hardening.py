"""HTTP and boot guards for the free-plan service boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.model_runtime import ModelRamBudgetExceeded, assert_gguf_within_ram_budget, runtime
from src.serving.api import SESSION_CONTEXTS, app, get_session_context


ROOT = Path(__file__).resolve().parents[1]
SERVICE_TOKEN = "svc-token-aaa"
ALIAS_TOKEN = "alias-token-bbb"
FREE_RAM_BUDGET = 268_435_456


@pytest.fixture(autouse=True)
def _clear_sessions(monkeypatch):
    # Existing bearer-only P0 tests exercise the explicitly local development
    # compatibility contract. Production requires independent signed admission.
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("AERYNZA_AI_AUTH_MODE", raising=False)
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "development")
    monkeypatch.delenv("ASCENSION_AI_LOCAL_DEV_BYPASS", raising=False)
    monkeypatch.delenv("ASCENSION_AI_ALLOW_TEST_TOKEN", raising=False)
    monkeypatch.delenv("AERYNZA_AI_ENABLE_DEVICE_ROUTES", raising=False)
    monkeypatch.delenv("AERYNZA_AI_ENABLE_PROVIDER_ROUTES", raising=False)
    SESSION_CONTEXTS.clear()
    yield
    SESSION_CONTEXTS.clear()


@pytest.fixture
def service_token(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", SERVICE_TOKEN)
    monkeypatch.delenv("AERYNZA_AI_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("AERYNZA_AI_RAM_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("ASCENSION_AI_RAM_BUDGET_BYTES", raising=False)
    return SERVICE_TOKEN


@pytest.fixture
def client(service_token):
    with TestClient(app) as test_client:
        yield test_client


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_protected_route_rejects_missing_and_wrong_bearer(client):
    assert client.get("/model/info").status_code == 401
    assert client.get("/model/info", headers=_auth("wrong-token")).status_code == 401


def test_protected_route_accepts_service_token(client):
    response = client.get("/model/info", headers=_auth(SERVICE_TOKEN))
    assert response.status_code == 200
    assert response.json()["name"] == "Aerynza AI Native Core"


def test_aerynza_service_token_alias_is_accepted(monkeypatch):
    monkeypatch.delenv("ASCENSION_AI_SERVICE_TOKEN", raising=False)
    monkeypatch.setenv("AERYNZA_AI_SERVICE_TOKEN", ALIAS_TOKEN)
    monkeypatch.delenv("AERYNZA_AI_RAM_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("ASCENSION_AI_RAM_BUDGET_BYTES", raising=False)
    with TestClient(app) as test_client:
        assert test_client.get("/v1/capabilities").status_code == 401
        response = test_client.get("/v1/capabilities", headers=_auth(ALIAS_TOKEN))
    assert response.status_code == 200
    assert "capabilities" in response.json()


def test_email_and_lab_token_are_not_secrets(client, monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_ALLOWED_EMAIL", "founder@example.com")
    monkeypatch.setenv("ASCENSION_AI_TEST_TOKEN", "lab-token")
    assert client.get("/model/info", headers=_auth("founder@example.com")).status_code == 401
    assert client.get("/model/info", headers=_auth("lab-token")).status_code == 401
    assert client.get("/model/info", headers=_auth(SERVICE_TOKEN)).status_code == 200


def test_production_without_service_token_fails_closed(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.delenv("ASCENSION_AI_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("AERYNZA_AI_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("AERYNZA_AI_RAM_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("ASCENSION_AI_RAM_BUDGET_BYTES", raising=False)
    # Missing signing configuration fails startup rather than falling back
    # to accepting a bearer without the required production envelope.
    with pytest.raises(RuntimeError, match="SIGNED_REQUESTS"):
        with TestClient(app):
            pass


def test_thesis_returns_a_thesis(client):
    response = client.post(
        "/v1/thesis",
        headers=_auth(SERVICE_TOKEN),
        json={
            "scope": "human",
            "subject_id": "user-1",
            "shell": "ap",
            "context": {"goals": [{"name": "Build consistency", "confidence": 0.9}]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert body["scope"] == "human"
    assert body["subject_id"] == "user-1"
    assert body["shell"] == "ap"
    assert body["claims"]


def test_thesis_rejects_wrong_shell(client):
    response = client.post(
        "/v1/thesis",
        headers=_auth(SERVICE_TOKEN),
        json={"scope": "home", "subject_id": "home-1", "shell": "ap", "context": {}},
    )
    assert response.status_code == 403


def test_retrieve_accepts_shell(client):
    response = client.post(
        "/v1/retrieve",
        headers=_auth(SERVICE_TOKEN),
        json={
            "query": "dentist",
            "shell": "lifeos",
            "context": {"memories": ["Dentist Friday at 9"]},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "dentist"
    assert isinstance(body["results"], list)


def test_session_is_keyed_to_token_and_strips_provider_keys(client, monkeypatch):
    monkeypatch.setenv("AERYNZA_AI_SERVICE_TOKEN", ALIAS_TOKEN)
    stored = client.post(
        "/v1/session/context",
        headers=_auth(SERVICE_TOKEN),
        json={
            "session_id": "s1",
            "merge": False,
            "context": {
                "note": "hello",
                "provider_keys": {"openai": {"api_key": "sk-live"}},
            },
        },
    )
    assert stored.status_code == 200
    assert "provider_keys" not in stored.json()["keys"]
    visible = get_session_context("s1", SERVICE_TOKEN)
    assert visible["note"] == "hello"
    assert "provider_keys" not in visible
    assert get_session_context("s1", ALIAS_TOKEN) == {}


def test_persist_context_defaults_false(client):
    response = client.post(
        "/v1/intelligence",
        headers=_auth(SERVICE_TOKEN),
        json={
            "messages": [{"role": "user", "content": "Hello there"}],
            "session_id": "persist-me",
            "context": {"note": "should-not-stick", "provider_keys": {"openai": {"api_key": "sk"}}},
        },
    )
    assert response.status_code in {200, 502, 503}
    assert get_session_context("persist-me", SERVICE_TOKEN) == {}


def test_device_and_provider_routes_are_disabled(client):
    headers = _auth(SERVICE_TOKEN)
    assert client.post("/v1/windows/execute", headers=headers, json={"action": "screenshot"}).status_code == 403
    assert client.post("/v1/android/execute", headers=headers, json={"action": "home"}).status_code == 403
    assert client.post("/v1/iphone/execute", headers=headers, json={"action": "list"}).status_code == 403
    assert client.post("/v1/iphone/inbox", headers=headers, json={"battery": 80}).status_code == 403
    assert client.post("/v1/media/generate", headers=headers, json={"prompt": "a cat"}).status_code == 403
    assert client.post("/v1/windows/execute", json={"action": "screenshot"}).status_code == 401


def test_boot_guard_refuses_oversized_gguf():
    profiles = json.loads((ROOT / "config" / "model_profiles.json").read_text(encoding="utf-8"))
    assert_gguf_within_ram_budget(profiles["starter"], budget=FREE_RAM_BUDGET)
    with pytest.raises(ModelRamBudgetExceeded):
        assert_gguf_within_ram_budget(profiles["pro_v231"], budget=FREE_RAM_BUDGET)


def test_load_refuses_boot_over_budget(monkeypatch):
    monkeypatch.setenv("AERYNZA_AI_RAM_BUDGET_BYTES", "1000")
    monkeypatch.setattr(
        runtime,
        "_read_profile",
        lambda: {
            "filename": "too-big.gguf",
            "size_bytes": 1_834_426_112,
            "sha256": "abc",
            "context_tokens": 768,
            "threads": 1,
        },
    )
    with pytest.raises(ModelRamBudgetExceeded):
        runtime.load()


def test_invalid_ram_budget_fails_closed(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_RAM_BUDGET_BYTES", "many")
    with pytest.raises(ModelRamBudgetExceeded):
        assert_gguf_within_ram_budget({"size_bytes": 1})


def test_render_blueprint_selects_free_starter():
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "plan: free" in text
    assert "value: starter" in text
    assert 'value: "768"' in text
    assert 'value: "16"' in text
    assert 'value: "268435456"' in text
    assert "value: production" in text
