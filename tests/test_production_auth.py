"""Regression tests for the native API production authentication boundary."""
import pytest
from fastapi import HTTPException

from src.serving import api


@pytest.fixture(autouse=True)
def signed_production_config(monkeypatch):
    """Supply signing config so auth tests isolate the property under test."""
    import base64
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "true")
    monkeypatch.setenv("ASCENSION_AI_TRUSTED_ISSUER", "aerynza-test")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_ID", "test-key")
    monkeypatch.setenv("ASCENSION_AI_SIGNING_KEY_BASE64", base64.b64encode(b"k" * 32).decode())
    monkeypatch.setenv("ASCENSION_AI_REPLAY_REDIS_URL", "redis://localhost:6379/0")


def test_production_requires_service_secret(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.delenv("ASCENSION_AI_SERVICE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ASCENSION_AI_SERVICE_TOKEN"):
        api.validate_auth_configuration()


def test_production_rejects_email_test_and_session_tokens(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "real-service-secret")
    monkeypatch.setenv("ASCENSION_AI_ALLOWED_EMAIL", "owner@example.test")
    monkeypatch.setenv("ASCENSION_AI_TEST_TOKEN", "test-secret")
    monkeypatch.setenv("ASCENSION_AI_LOCAL_DEV_BYPASS", "true")
    api.SESSIONS.add("desktop-session-secret")
    try:
        for token in ("owner@example.test", "test-secret", "desktop-session-secret"):
            with pytest.raises(HTTPException) as error:
                api.require_access("Bearer " + token)
            assert error.value.status_code == 401
        api.require_access("Bearer real-service-secret")
    finally:
        api.SESSIONS.discard("desktop-session-secret")


def test_development_bypass_requires_explicit_mode(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "development")
    monkeypatch.setenv("ASCENSION_AI_LOCAL_DEV_BYPASS", "true")
    api.require_access(None)
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    with pytest.raises(HTTPException) as error:
        api.require_access(None)
    assert error.value.status_code == 401


def test_malformed_auth_and_unknown_mode_fail_closed(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "real-service-secret")
    for value in (None, "", "Basic real-service-secret", "Bearer wrong"):
        with pytest.raises(HTTPException):
            api.require_access(value)
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "invalid")
    with pytest.raises(RuntimeError):
        api.validate_auth_configuration()


@pytest.mark.asyncio
async def test_production_disables_desktop_login(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    with pytest.raises(HTTPException) as error:
        await api.login(api.LoginRequest(email="owner@example.test", password="not-a-secret"))
    assert error.value.status_code == 404


def test_production_shell_allowlist_denies_unlisted_shell(monkeypatch):
    from src.core.contracts import Shell
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "service-secret")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", Shell.AP.value)
    api.validate_auth_configuration()
    api.enforce_shell_access(Shell.AP)
    with pytest.raises(HTTPException) as error:
        api.enforce_shell_access(Shell.NEXUS_FAMILY)
    assert error.value.status_code == 403


def test_production_requires_explicit_shell_allowlist(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "service-secret")
    monkeypatch.delenv("ASCENSION_AI_SERVICE_SHELLS", raising=False)
    with pytest.raises(RuntimeError, match="ASCENSION_AI_SERVICE_SHELLS"):
        api.validate_auth_configuration()


def test_production_direct_action_guard(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    with pytest.raises(HTTPException) as error:
        api.require_certified_action_gateway()
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_action_catalog_requires_authorized_shell(monkeypatch):
    from src.core.contracts import Shell
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", Shell.AP.value)
    with pytest.raises(HTTPException) as error:
        await api.actions_catalog(Shell.NEXUS_HOME)
    assert error.value.status_code == 403


def test_production_action_guard_blocks_all_device_dispatch(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    for handler in (api.windows_execute, api.android_execute, api.iphone_execute):
        assert "require_certified_action_gateway" in handler.__code__.co_names


def test_production_rejects_shared_token_session_reads_and_writes(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    with pytest.raises(HTTPException) as read_error:
        api.get_session_context("same-session-across-users", "Bearer service-secret")
    assert read_error.value.status_code == 403
    with pytest.raises(HTTPException) as write_error:
        api.set_session_context("same-session-across-users", {"private": True}, "Bearer service-secret")
    assert write_error.value.status_code == 403


@pytest.mark.asyncio
async def test_production_rejects_iphone_inbox_persistence(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    with pytest.raises(HTTPException) as error:
        await api.iphone_inbox({"location": "private"})
    assert error.value.status_code == 403


def test_production_rejects_unsigned_mode(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "service-secret")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap")
    monkeypatch.setenv("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS", "false")
    with pytest.raises(RuntimeError, match="ASCENSION_AI_REQUIRE_SIGNED_REQUESTS"):
        api.validate_auth_configuration()


def test_production_rejects_missing_signing_key(monkeypatch):
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_TOKEN", "service-secret")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap")
    monkeypatch.delenv("ASCENSION_AI_SIGNING_KEY_BASE64", raising=False)
    with pytest.raises(RuntimeError, match="Signed requests require"):
        api.validate_auth_configuration()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,shell", [
    ("sprout", "ap"),
    ("home", "nexus_home"),
    ("family", "nexus_family"),
])
async def test_sensitive_thesis_requires_resource_ownership(monkeypatch, scope, shell):
    from src.core.contracts import Shell
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap,nexus_home,nexus_family")
    request = api.ThesisRequest(
        scope=scope, subject_id="other-users-resource", shell=Shell(shell), context={}
    )
    with pytest.raises(HTTPException) as error:
        await api.thesis(request)
    assert error.value.status_code == 403
    assert "ownership" in error.value.detail.lower()


@pytest.mark.asyncio
async def test_thesis_contribution_requires_verified_consent_in_production(monkeypatch):
    from src.core.contracts import Shell
    monkeypatch.setenv("ASCENSION_AI_AUTH_MODE", "production")
    monkeypatch.setenv("ASCENSION_AI_SERVICE_SHELLS", "ap")
    request = api.ThesisContributionRequest(
        member_id="member-1", target_scope="nexus_family",
        shell=Shell.AP, human_thesis={},
        selections=[{"domain": "wellness"}], consent_receipt_id="unverified-receipt"
    )
    with pytest.raises(HTTPException) as error:
        await api.thesis_contribution(request)
    assert error.value.status_code == 403
