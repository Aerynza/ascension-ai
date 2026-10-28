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
