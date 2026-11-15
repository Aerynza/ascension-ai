"""Model-free distributed replay-probe regressions; no network or database."""
from __future__ import annotations

import threading

import pytest

from scripts import distributed_replay_probe as probe


class SharedMemoryAuthority:
    def __init__(self):
        self.lock = threading.Lock()
        self.claimed = set()

    def client(self):
        authority = self

        class IndependentClient:
            def claim(self, key, ttl_seconds):
                assert key.startswith("aerynza:envelope:probe:")
                assert ttl_seconds == probe.NONCE_TTL_SECONDS
                with authority.lock:
                    if key in authority.claimed:
                        return False
                    authority.claimed.add(key)
                    return True

        return IndependentClient()


def test_probe_requires_two_distinct_clients_and_exactly_one_claim():
    authority = SharedMemoryAuthority()
    result = probe.collect_replay_evidence(
        authority.client, attempts=6, nonce="testnonce"
    )
    assert result == {
        "concurrent_attempts": 6,
        "accepted": 1,
        "replays_rejected": 6,
        "new_nonce_accepted": True,
    }


def test_probe_catches_nonshared_per_process_replay_authority():
    # A separate in-memory store per service replica is not a valid nonce
    # authority: every process can accept the same signed request.
    with pytest.raises(RuntimeError, match="expected one accepted claim"):
        probe.collect_replay_evidence(
            lambda: SharedMemoryAuthority().client(),
            attempts=4,
            nonce="unshared",
        )


def test_probe_fails_on_network_outage_instead_of_marking_a_replay_success():
    def no_network():
        class OfflineClient:
            def claim(self, _key, _ttl):
                raise ConnectionError("backend unavailable")

        return OfflineClient()

    with pytest.raises(ConnectionError):
        probe.collect_replay_evidence(no_network, attempts=2, nonce="offline")


def test_probe_rejects_non_boolean_claim_outcomes():
    class Broken:
        def claim(self, _key, _ttl):
            return None

    with pytest.raises(RuntimeError, match="non-boolean"):
        probe.collect_replay_evidence(lambda: Broken(), attempts=2, nonce="unknown")


@pytest.mark.parametrize("attempts", [0, 1, 13, 2.5, True])
def test_probe_bounds_concurrent_clients(attempts):
    with pytest.raises(ValueError, match="2 and 12"):
        probe.collect_replay_evidence(lambda: object(), attempts=attempts)


@pytest.mark.parametrize("nonce", ["", "not safe", "a/b", "X\nY"])
def test_probe_rejects_unsafe_nonce_names(nonce):
    with pytest.raises(ValueError, match="simple opaque"):
        probe.collect_replay_evidence(lambda: object(), attempts=2, nonce=nonce)


def test_no_probe_runs_without_explicit_approval(monkeypatch, capsys):
    monkeypatch.delenv("AERYNZA_AI_REPLAY_PROBE_APPROVAL", raising=False)
    monkeypatch.setenv("AERYNZA_AI_REPLAY_PROBE_URL", "rediss://example.invalid:6379")
    def forbidden(_url):
        raise AssertionError("Cannot create network clients without approval")
    monkeypatch.setattr(probe, "store_factory_for_url", forbidden)
    assert probe.main() == 2
    assert "approval" in capsys.readouterr().out.lower()


def test_auth_credentials_are_redacted_from_failure_output(monkeypatch, capsys):
    monkeypatch.setenv("AERYNZA_AI_REPLAY_PROBE_APPROVAL", "staging-test-only")
    monkeypatch.setenv("AERYNZA_AI_REPLAY_PROBE_URL",
                       "postgresql://role:NEVER-LOG-ME@db.invalid/aerynza")
    def fails(_url):
        raise RuntimeError("postgresql://role:NEVER-LOG-ME@db.invalid/aerynza")
    monkeypatch.setattr(probe, "store_factory_for_url", fails)
    assert probe.main() == 1
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "NEVER-LOG-ME" not in output
    assert "postgresql://" not in output
