"""Opt-in cross-process replay-store evidence probe for staging.

This DOES NOT certify the native AI service or test signed shell ownership.
It validates only atomic nonce claims across separate backend client instances.
A real shared staging Redis/PostgreSQL URL is required. Never log it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import secrets
import sys
from threading import Barrier
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving.postgres_nonce_store import PostgresNonceStore
from src.serving.replay_guard import RedisNonceStore

MAX_ATTEMPTS = 12
NONCE_TTL_SECONDS = 120


def store_factory_for_url(url: str):
    """Return *new* store clients each time; never reuse a process-local fake."""
    parsed = urlsplit(url)
    if parsed.scheme in {"redis", "rediss"} and parsed.hostname:
        def create():
            import redis
            client = redis.Redis.from_url(
                url, socket_connect_timeout=3, socket_timeout=3
            )
            return RedisNonceStore(client)
        return "redis", create
    if (parsed.scheme in {"postgres", "postgresql"} and parsed.hostname
            and parsed.username and parsed.password):
        # Preflight the least-privilege role before any ledger write.
        PostgresNonceStore(url).preflight()
        return "postgres", lambda: PostgresNonceStore(url)
    raise ValueError("A supported authenticated replay staging URL is required")


def collect_replay_evidence(store_factory, *, attempts=8, nonce=None):
    """Race independent clients for one nonce; error on any false-positive.

    A backend outage must raise, not be silently counted as a consumed nonce.
    The keys are random, short-lived, and contain no account or request body.
    """
    if type(attempts) is not int or not 2 <= attempts <= MAX_ATTEMPTS:
        raise ValueError("Choose between 2 and 12 concurrent clients")
    identity = secrets.token_hex(20) if nonce is None else nonce
    if not isinstance(identity, str) or not identity.isascii() or not identity.isalnum():
        raise ValueError("Probe nonce must be a simple opaque identifier")
    first_key = "aerynza:envelope:probe:" + identity
    second_key = "aerynza:envelope:probe:" + identity + "second"
    barrier = Barrier(attempts, timeout=15)

    def consume(_index):
        client = store_factory()
        barrier.wait()
        accepted = client.claim(first_key, NONCE_TTL_SECONDS)
        if type(accepted) is not bool:
            raise RuntimeError("Replay backend returned a non-boolean claim outcome")
        return accepted

    with ThreadPoolExecutor(max_workers=attempts) as workers:
        outcomes = list(workers.map(consume, range(attempts)))

    if outcomes.count(True) != 1 or outcomes.count(False) != attempts - 1:
        raise RuntimeError("Two-client nonce isolation failed: expected one accepted claim")

    # A new independent client (analogous to another replica/restart) must
    # still reject the used nonce. A different unique nonce must still work.
    replay = store_factory().claim(first_key, NONCE_TTL_SECONDS)
    fresh = store_factory().claim(second_key, NONCE_TTL_SECONDS)
    if replay is not False or fresh is not True:
        raise RuntimeError("Replay persistence or independent-nonce acceptance failed")

    return {"concurrent_attempts": attempts, "accepted": 1,
            "replays_rejected": attempts, "new_nonce_accepted": True}


def main() -> int:
    # Never connect automatically during CI or a deployment boot.
    if os.environ.get("AERYNZA_AI_REPLAY_PROBE_APPROVAL") != "staging-test-only":
        print("Replay probe inactive; explicit staging approval is required.")
        return 2
    url = os.environ.get("AERYNZA_AI_REPLAY_PROBE_URL", "")
    if not url:
        print("Replay probe inactive; staging backend URL is not configured.")
        return 2
    try:
        backend, factory = store_factory_for_url(url)
        result = collect_replay_evidence(factory)
    except Exception as error:
        # Do not echo exception strings: drivers may embed connection URIs
        # containing usernames/passwords in network failures.
        print("FAIL: distributed replay probe did not pass (details redacted).")
        return 1
    print("PASS: independent " + backend + " clients rejected replay "
          "and admitted one fresh nonce. Attempts: "
          + str(result["concurrent_attempts"]) + ".")
    print("Evidence applies only to this staging backend, not live AI readiness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
