"""Offline deployment guard for pinned Aerynza model profiles.

Run before downloading GGUF artifacts on Render: wrong profile, insufficient
RAM budget, or unverifiable manifest must fail before any network request.
This does not establish model quality or authorize promotion by itself.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PreflightRejected(RuntimeError):
    pass


def preflight(env=None, profiles_path=None, aliases_path=None):
    env = os.environ if env is None else env
    profiles_path = Path(profiles_path) if profiles_path else ROOT / "config/model_profiles.json"
    aliases_path = Path(aliases_path) if aliases_path else ROOT / "config/profile_aliases.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    aliases = json.loads(aliases_path.read_text(encoding="utf-8")) if aliases_path.is_file() else {}
    requested = str(env.get("ASCENSION_MODEL_PROFILE") or env.get("AERYNZA_MODEL_PROFILE") or "starter").strip().lower()
    resolved = str(aliases.get(requested, requested)).strip().lower()
    if resolved not in profiles:
        raise PreflightRejected("Unknown model profile")
    profile = profiles[resolved]
    if not isinstance(profile, dict) or not isinstance(profile.get("filename"), str):
        raise PreflightRejected("Invalid model manifest")
    declared = profile.get("size_bytes")
    digest = profile.get("sha256")
    if not isinstance(declared, int) or isinstance(declared, bool) or declared <= 0:
        raise PreflightRejected("Model manifest is missing a valid size")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise PreflightRejected("Model manifest is missing a pinned SHA-256")
    raw_budget = env.get("AERYNZA_AI_RAM_BUDGET_BYTES") or env.get("ASCENSION_AI_RAM_BUDGET_BYTES")
    if raw_budget is None or str(raw_budget).strip() == "":
        if str(env.get("RENDER", "")).lower() == "true" and resolved != "starter":
            raise PreflightRejected("Non-starter model on Render requires explicit verified RAM budget")
        budget = None
    else:
        try:
            budget = int(str(raw_budget).strip())
        except ValueError as exc:
            raise PreflightRejected("Invalid RAM budget") from exc
        if budget <= 0:
            raise PreflightRejected("RAM budget must be positive")
        if declared > budget:
            raise PreflightRejected("Pinned model is larger than available GGUF RAM budget")
    return {
        "requested_profile": requested,
        "profile": resolved,
        "label": str(profile.get("label") or resolved),
        "size_bytes": declared,
        "ram_budget_bytes": budget,
        "artifact_sha256": digest,
        "minimum_render_plan": profile.get("minimum_render_plan"),
        "release_quality_verified": False,
    }


def main():
    try:
        result = preflight()
    except (PreflightRejected, OSError, ValueError) as exc:
        raise SystemExit("MODEL PREFLIGHT FAILED: " + str(exc)) from None
    print(json.dumps({"status": "manifest_and_memory_ok", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
