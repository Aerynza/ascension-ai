"""Bound ephemeral native AI session memory without treating service tokens as user IDs."""
import time

import pytest

from src.serving import api


@pytest.fixture(autouse=True)
def clear_session_contexts():
    with api._SESSION_LOCK:
        api.SESSION_CONTEXTS.clear()
    yield
    with api._SESSION_LOCK:
        api.SESSION_CONTEXTS.clear()


def test_contexts_remain_isolated_across_distinct_bearers_and_are_copied():
    api.set_session_context("same-session", {
        "profile": {"details": ["private"], "provider_keys": {"secret": "NEVER_STORE"}}
    }, "bearer-one")
    assert api.get_session_context("same-session", "bearer-two") == {}
    data = api.get_session_context("same-session", "bearer-one")
    assert data == {"profile": {"details": ["private"]}}
    data["profile"]["details"].append("mutated")
    assert api.get_session_context("same-session", "bearer-one")["profile"]["details"] == ["private"]


def test_expired_context_cannot_be_retrieved_or_merged():
    api.set_session_context("expired", {"old": "stale"}, "token")
    key = api._session_key("expired", "token")
    with api._SESSION_LOCK:
        _, stored = api.SESSION_CONTEXTS[key]
        api.SESSION_CONTEXTS[key] = (time.monotonic() - 1, stored)
    assert api.get_session_context("expired", "token") == {}
    assert api.set_session_context("expired", {"new": "fresh"}, "token", merge=True) == {"new": "fresh"}


def test_lru_capacity_evicts_oldest_not_recently_read_context():
    for n in range(api.MAX_ACTIVE_SESSION_CONTEXTS):
        api.set_session_context(f"s-{n}", {"n": n}, "token")
    assert api.get_session_context("s-0", "token") == {"n": 0}
    api.set_session_context("new", {"n": "new"}, "token")
    assert len(api.SESSION_CONTEXTS) == api.MAX_ACTIVE_SESSION_CONTEXTS
    assert api.get_session_context("s-1", "token") == {}
    assert api.get_session_context("s-0", "token") == {"n": 0}
    assert api.get_session_context("new", "token") == {"n": "new"}


def test_oversized_context_rejected_without_persisting():
    with pytest.raises(ValueError, match="size limit"):
        api.set_session_context("large", {"content": "x" * 200_001}, "token")
    assert api.get_session_context("large", "token") == {}


def test_production_shared_bearer_cannot_store_or_read_session(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setenv("AERYNZA_AI_AUTH_MODE", "production")
    with pytest.raises(HTTPException) as read_error:
        api.get_session_context("same-session", "shared-service-token")
    assert read_error.value.status_code == 403
    with pytest.raises(HTTPException) as write_error:
        api.set_session_context("same-session", {"private": "account-A"}, "shared-service-token")
    assert write_error.value.status_code == 403
    assert not api.SESSION_CONTEXTS


def test_production_stateless_context_still_works_without_session(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    monkeypatch.setenv("RENDER", "true")
    request = SimpleNamespace(session_id="", persist_context=False, context={"owner": "LifeOS"})
    assert api.resolve_session_context(request, "shared-service-token") == {"owner": "LifeOS"}
    for session_id, persist_context in [("user-picked-session", False), ("", True), ("user-picked-session", True)]:
        with pytest.raises(HTTPException) as error:
            api.resolve_session_context(SimpleNamespace(
                session_id=session_id, persist_context=persist_context, context={"private": True}
            ), "shared-service-token")
        assert error.value.status_code == 403
    assert not api.SESSION_CONTEXTS


def test_production_http_personal_session_endpoints_deny_before_storage(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("AERYNZA_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "shared-service-token")
    headers = {"Authorization": "Bearer shared-service-token"}
    client = TestClient(api.app)
    write = client.post("/v1/session/context", json={
        "session_id": "same-for-two-users", "context": {"owner": "A"}
    }, headers=headers)
    read = client.post("/v1/session/data-panels", json={
        "session_id": "same-for-two-users"
    }, headers=headers)
    inference = client.post("/v1/intelligence", json={
        "messages": [{"role": "user", "content": "hello"}],
        "session_id": "same-for-two-users",
        "context": {"owner": "B"}
    }, headers=headers)
    # Unsigned production callers now fail at the gateway before the session guard.
    assert [write.status_code, read.status_code, inference.status_code] == [503, 503, 503]
    assert not api.SESSION_CONTEXTS


def test_production_flags_never_authorize_unscoped_native_device_or_provider_actions(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("AERYNZA_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "shared-service-token")
    monkeypatch.setenv("AERYNZA_AI_ENABLE_DEVICE_ROUTES", "true")
    monkeypatch.setenv("AERYNZA_AI_ENABLE_PROVIDER_ROUTES", "true")
    assert api.device_routes_enabled() is False
    assert api.provider_routes_enabled() is False
    headers = {"Authorization": "Bearer shared-service-token"}
    client = TestClient(api.app)
    cases = (
        ("/v1/windows/execute", {"action": "screenshot"}),
        ("/v1/android/execute", {"action": "home"}),
        ("/v1/iphone/execute", {"action": "list"}),
        ("/v1/iphone/inbox", {"location": "private"}),
        ("/v1/media/generate", {"prompt": "private image"}),
    )
    for path, body in cases:
        result = client.post(path, json=body, headers=headers)
        assert result.status_code == 503, (path, result.status_code)
    lab_login = client.post("/v1/login", json={
        "email": "lab@example.com", "password": "no-keyring-call"
    })
    assert lab_login.status_code == 503


def test_development_explicit_flags_remain_available_for_local_lab(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("AERYNZA_AI_AUTH_MODE", "development")
    monkeypatch.setenv("AERYNZA_AI_ENABLE_DEVICE_ROUTES", "true")
    monkeypatch.setenv("AERYNZA_AI_ENABLE_PROVIDER_ROUTES", "true")
    assert api.device_routes_enabled() is True
    assert api.provider_routes_enabled() is True


def test_native_background_yields_when_foreground_or_queue_is_busy(monkeypatch):
    from fastapi import HTTPException
    states = [
        {"ready": True, "queue": {"active_requests": 1, "queue_depth": 0}},
        {"ready": True, "queue": {"active_requests": 0, "queue_depth": 2}},
        {"ready": True, "queue": {"active_requests": 0, "queue_depth": 0, "inference_hung": True}},
        {"ready": False, "queue": {"active_requests": 0, "queue_depth": 0}},
        {"ready": True, "queue": {}},
    ]
    for state in states:
        monkeypatch.setattr(api.runtime, "status", lambda state=state: state)
        for mode in ("background", "proactive"):
            with pytest.raises(HTTPException) as result:
                api.require_idle_for_background(mode)
            assert result.value.status_code == 409
            assert result.value.headers["Retry-After"] == "2"
        # Interactive requests retain their normal queue admission path.
        api.require_idle_for_background("conversation")


def test_native_background_allows_idle_model_then_uses_existing_lock(monkeypatch):
    monkeypatch.setattr(api.runtime, "status", lambda: {
        "ready": True,
        "queue": {"active_requests": 0, "queue_depth": 0, "inference_hung": False},
    })
    for mode in ("background", "proactive", "planning", "conversation"):
        api.require_idle_for_background(mode)


def test_unattended_streaming_cannot_bypass_native_priority_lock():
    import asyncio
    from fastapi import HTTPException
    from src.serving.api import IntelligenceRequest, stream_intelligence
    for mode in ("background", "proactive"):
        request = IntelligenceRequest(messages=[{"role": "user", "content": "Prepare privately"}], mode=mode)
        with pytest.raises(HTTPException) as result:
            asyncio.run(stream_intelligence(request, token="test-only"))
        assert result.value.status_code == 409
        assert result.value.headers["Retry-After"] == "2"

def test_background_runtime_race_returns_409_not_provider_outage(monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from src.core.model_runtime import NativeInferenceQueueTimeout
    monkeypatch.setattr(api, "require_idle_for_background", lambda _mode: None)
    monkeypatch.setattr(api.runtime, "status", lambda: {"ready": True, "error": None})
    def busy_model(**_kwargs):
        raise NativeInferenceQueueTimeout("busy foreground")
    monkeypatch.setattr(api, "respond", busy_model)
    monkeypatch.setattr(api, "provider_routes_enabled", lambda: False)
    request = api.IntelligenceRequest(messages=[
        {"role": "user", "content": "Plan a private goal"}
    ], mode="background")
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.intelligence(request, token="test-only"))
    assert error.value.status_code == 409
    assert error.value.headers["Retry-After"] == "2"
