"""One-off staging verifier for the private PostgreSQL nonce ledger.

Uses ASCENSION_AI_REPLAY_POSTGRES_URL from the process environment.
Outputs only PASS/FAIL; does not print credentials, database host, or nonce.
"""
import os
import secrets

from src.serving.postgres_nonce_store import PostgresNonceStore


def main():
    url = os.environ.get("ASCENSION_AI_REPLAY_POSTGRES_URL", "").strip()
    if not url:
        raise SystemExit("FAIL: replay PostgreSQL URL is not configured")
    try:
        store = PostgresNonceStore(url)
        store.preflight()
        key = "aerynza:envelope:preflight:verify:" + secrets.token_hex(24)
        if not store.claim(key, 90):
            raise RuntimeError("Unexpected first-claim collision")
        if store.claim(key, 90):
            raise RuntimeError("Replay was accepted")
    except Exception:
        raise SystemExit("FAIL: private replay store checks did not pass") from None
    print("PASS: restricted PostgreSQL replay store accepted one claim and rejected its replay")


if __name__ == "__main__":
    main()
