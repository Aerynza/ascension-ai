"""Fail-closed, checkpoint-bound evidence gate for native AI promotion.

A loaded model is a *candidate*, never evidence that the user experience,
privacy, safety, recovery, or latency passed an independent release review.
The evidence file and its SHA256 are operator-supplied at deployment time.
No production service should commit that file or an approval key to Git.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re

REQUIRED_RELEASE_GATES = (
    "conversation_quality", "shell_isolation", "action_integrity",
    "domain_reasoning", "safety_privacy", "interactive_latency",
    "concurrency_recovery", "native_primary_canary",
)
MAX_REPORT_BYTES = 262_144
MAX_EVIDENCE_AGE = timedelta(days=30)


def evaluate_release_gates(model: dict, profile: dict, *, signed_admission_active: bool = False) -> dict[str, bool]:
    """Return per-gate verified values; no inputs means all gates closed."""
    closed = {name: False for name in REQUIRED_RELEASE_GATES}
    # A reviewed model cannot replace AP while the native HTTP server still
    # trusts a shared bearer with no verified per-user signed actor admission.
    # Only a tested, installed middleware integration may supply True here.
    if signed_admission_active is not True:
        return closed
    if not model.get("ready") or model.get("profile") != "pro_v231":
        return closed
    if os.getenv("AERYNZA_NATIVE_REPLACEMENT_APPROVED") != "1":
        return closed
    # An operator may disable hashing for development, but never for promotion.
    if os.getenv("ASCENSION_VERIFY_MODEL_ON_BOOT", "1") == "0":
        return closed
    expected = os.getenv("AERYNZA_NATIVE_RELEASE_EVIDENCE_SHA256", "").strip().lower()
    path = os.getenv("AERYNZA_NATIVE_RELEASE_EVIDENCE_FILE", "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or not path:
        return closed
    try:
        source = Path(path)
        if not source.is_file() or source.stat().st_size > MAX_REPORT_BYTES:
            return closed
        raw = source.read_bytes()
        if len(raw) > MAX_REPORT_BYTES or hashlib.sha256(raw).hexdigest() != expected:
            return closed
        report = json.loads(raw)
        if not isinstance(report, dict):
            return closed
        if report.get("profile") != model.get("profile"):
            return closed
        # Pin evidence to the exact model artifact configured and checked on boot.
        profile_sha = profile.get("sha256") if isinstance(profile, dict) else None
        if not isinstance(profile_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", profile_sha):
            return closed
        if report.get("model_sha256", "").lower() != profile_sha.lower():
            return closed
        if report.get("operator_approval") is not True:
            return closed
        stamp = report.get("verified_at")
        if not isinstance(stamp, str):
            return closed
        verified = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if verified.tzinfo is None:
            return closed
        age = datetime.now(timezone.utc) - verified.astimezone(timezone.utc)
        if age < timedelta(minutes=-5) or age > MAX_EVIDENCE_AGE:
            return closed
        checks = report.get("gates")
        if not isinstance(checks, dict):
            return closed
        return {
            name: isinstance(checks.get(name), dict)
            and checks[name].get("passed") is True
            and isinstance(checks[name].get("evidence_ref"), str)
            and 8 <= len(checks[name]["evidence_ref"].strip()) <= 256
            for name in REQUIRED_RELEASE_GATES
        }
    except (OSError, ValueError, TypeError, OverflowError, AttributeError):
        return closed
