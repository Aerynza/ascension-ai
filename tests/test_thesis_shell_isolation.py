"""Regression tests for previously unreachable thesis authorization and retrieval shell field."""
import pytest
from fastapi import HTTPException

from src.core.contracts import Shell
from src.serving import api


@pytest.fixture(autouse=True)
def explicit_development_shell_mode(monkeypatch):
    """Route unit tests exercise thesis scope rules independently of deployment credentials."""
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "development")


@pytest.mark.asyncio
async def test_thesis_rejects_wrong_shell():
    request = api.ThesisRequest(
        scope="family", subject_id="family-1", shell=Shell.AP, context={}
    )
    with pytest.raises(HTTPException) as error:
        await api.thesis(request)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_thesis_authorized_shell_returns_result(monkeypatch):
    monkeypatch.setattr(api, "build_thesis", lambda scope, subject_id, context: {"ok": True})
    request = api.ThesisRequest(
        scope="family", subject_id="family-1", shell=Shell.NEXUS_FAMILY, context={}
    )
    result = await api.thesis(request)
    assert result["ok"] is True
    assert result["shell"] == Shell.NEXUS_FAMILY.value


@pytest.mark.asyncio
async def test_retrieve_scopes_to_declared_shell(monkeypatch):
    seen = {}
    def fake_scope(context, shell):
        seen["shell"] = shell
        return {"safe": True}
    monkeypatch.setattr(api, "scope_context", fake_scope)
    monkeypatch.setattr(api, "hybrid_retrieve", lambda query, context, top_k: [])
    request = api.RetrievalRequest(shell=Shell.NEXUS_HOME, query="hello", context={"private": "value"})
    result = await api.retrieve(request)
    assert seen["shell"] == Shell.NEXUS_HOME
    assert result["results"] == []


@pytest.mark.asyncio
async def test_production_denies_unlisted_thesis_shell(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", Shell.AP.value)
    request = api.ThesisRequest(scope="family", subject_id="family-1", shell=Shell.NEXUS_FAMILY, context={})
    with pytest.raises(HTTPException) as error:
        await api.thesis(request)
    assert error.value.status_code == 403


def test_founder_ai_shell_tier_canon():
    from src.core.contracts import Shell, Tier, SHELL_CONTRACTS
    from src.core.action_runtime import shell_allows_action
    assert Tier.CORE.value == "core"
    assert Shell.EXECUTIVE.value == "executive"
    assert Shell.EXECUTIVE in SHELL_CONTRACTS
    assert Shell.SPROUT in SHELL_CONTRACTS
    assert Shell.NEXUS_HOME in SHELL_CONTRACTS
    assert Shell.NEXUS_FAMILY in SHELL_CONTRACTS
    assert shell_allows_action(Shell.EXECUTIVE, "documents.draft")
    assert not shell_allows_action(Shell.EXECUTIVE, "trading.submit_prediction_order")
