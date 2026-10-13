"""Evidence-bound native replacement gate.

Loading a model is not evidence that it can replace an outside provider.
A validated, recent, HMAC-attested evaluation tied to the exact model digest
plus explicit operator approval is necessary before promoting native inference.
There is intentionally no default approval or built-in pass count.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

REQUIRED_GATES = (
    "conversation_quality",
    "shell_isolation",
    "action_integrity",
    "domain_reasoning",
    "safety_privacy",
    "interactive_latency",
    "concurrency_recovery",
    "native_primary_canary",
)
READINESS_FILE = Path(__file__).resolve().parents[2] / "evals" / "results" / "production_replacement_attestation.json"
ALLOWED_PROFILE = "pro_v231"
SUITE = "aerynza_native_replacement_v1"
MAX_EVIDENCE_AGE_SECONDS = 7 * 86400


def _blank(ready: bool, status: str) -> dict[str, Any]:
    return {
        "candidate_ready": ready,
        "replacement_ready": False,
        "evidence_status": status,
        "gates": {gate: False for gate in REQUIRED_GATES},
        "evidence": {},
    }


def release_status(
    runtime: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    evidence_path: Path | None = None,
    now: int | None = None,
) -> dict[str, Any]:
    """Return a conservative release status with no inference/model side effects."""
    settings = os.environ if env is None else env
    current_time = int(time.time()) if now is None else now
    loaded = runtime.get("ready") is True
    if not loaded:
        return _blank(False, "model_not_loaded")
    if runtime.get("profile") != ALLOWED_PROFILE:
        return _blank(True, "profile_not_evaluated")
    expected_sha = str(profile.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        return _blank(True, "model_digest_unavailable")

    key_encoded = settings.get("ASCENSION_AI_READINESS_ATTESTATION_KEY_BASE64", "")
    try:
        key = base64.b64decode(key_encoded, validate=True)
    except (binascii.Error, ValueError):
        key = b""
    if len(key) < 32:
        return _blank(True, "attestation_key_not_configured")

    try:
        source = READINESS_FILE if evidence_path is None else evidence_path
        envelope = json.loads(source.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        signature = envelope["signature"]
        if not isinstance(payload, dict) or not isinstance(signature, str) or not re.fullmatch(r"[a-f0-9]{64}", signature):
            raise ValueError("invalid attestation")
        signed_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        actual = hmac.new(key, signed_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(actual, signature):
            raise ValueError("attestation signature mismatch")
        issued = payload["issued_at"]
        expires = payload["expires_at"]
        if type(issued) is not int or type(expires) is not int:
            raise ValueError("invalid time fields")
        if issued > current_time + 300 or current_time > expires or expires <= issued:
            raise ValueError("attestation expired or future dated")
        if current_time - issued > MAX_EVIDENCE_AGE_SECONDS or expires - issued > MAX_EVIDENCE_AGE_SECONDS:
            raise ValueError("attestation exceeds freshness window")
        if payload.get("suite") != SUITE or payload.get("profile") != ALLOWED_PROFILE:
            raise ValueError("wrong evidence suite or model profile")
        if payload.get("model_sha256") != expected_sha:
            raise ValueError("evidence belongs to a different model")
        run_url = payload.get("ci_run_url", "")
        if not isinstance(run_url, str) or not re.fullmatch(
            r"https://github\.com/acensionlifeos-dev/ascension-ai/actions/runs/[1-9][0-9]*", run_url
        ):
            raise ValueError("missing trusted CI evidence")
        cases = payload.get("evaluation_cases")
        if type(cases) is not int or cases < 20:
            raise ValueError("insufficient evaluation coverage")
        gates = payload.get("gates")
        if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES) or any(type(value) is not bool for value in gates.values()):
            raise ValueError("incomplete or invalid gates")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return _blank(True, "attestation_missing_or_invalid")

    evidence = {
        "suite": SUITE,
        "profile": ALLOWED_PROFILE,
        "model_sha256": expected_sha,
        "evaluation_cases": cases,
        "ci_run_url": run_url,
        "issued_at": issued,
        "expires_at": expires,
    }
    approved = settings.get("ASCENSION_NATIVE_REPLACEMENT_APPROVED", "").lower() == "true"
    every_gate_passed = all(gates.values())
    return {
        "candidate_ready": True,
        "replacement_ready": bool(approved and every_gate_passed),
        "evidence_status": "approved" if approved and every_gate_passed else ("evaluation_failed" if not every_gate_passed else "awaiting_operator_approval"),
        "gates": gates,
        "evidence": evidence,
    }
