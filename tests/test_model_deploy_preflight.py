"""No network required: prove model promotions cannot outrun free-tier memory."""
import json
from pathlib import Path

import pytest

from scripts.model_deploy_preflight import PreflightRejected, preflight

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "config/model_profiles.json"
ALIASES = ROOT / "config/profile_aliases.json"


def test_starter_fits_free_render_budget():
    result = preflight({
        "RENDER": "true",
        "ASCENSION_MODEL_PROFILE": "starter",
        "AERYNZA_AI_RAM_BUDGET_BYTES": "268435456",
    }, PROFILES, ALIASES)
    assert result["profile"] == "starter"
    assert result["size_bytes"] < result["ram_budget_bytes"]
    assert result["release_quality_verified"] is False


def test_v231_is_rejected_on_free_tier_before_any_download():
    with pytest.raises(PreflightRejected, match="larger than available"):
        preflight({
            "RENDER": "true",
            "ASCENSION_MODEL_PROFILE": "pro_v231",
            "AERYNZA_AI_RAM_BUDGET_BYTES": "268435456",
        }, PROFILES, ALIASES)


def test_v231_requires_explicit_budget_on_render_even_if_gguf_manifest_is_pinned():
    with pytest.raises(PreflightRejected, match="requires explicit"):
        preflight({"RENDER": "true", "ASCENSION_MODEL_PROFILE": "pro_v231"}, PROFILES, ALIASES)


def test_paid_plan_can_preflight_profile_without_claiming_quality():
    result = preflight({
        "RENDER": "true",
        "ASCENSION_MODEL_PROFILE": "pro_v231",
        "AERYNZA_AI_RAM_BUDGET_BYTES": "2684354560",
    }, PROFILES, ALIASES)
    assert result["profile"] == "pro_v231"
    assert result["minimum_render_plan"] == "pro_4gb"
    assert result["release_quality_verified"] is False


def test_bad_budget_manifest_and_profile_fail_closed(tmp_path):
    for budget in ["unknown", "-100", "0"]:
        with pytest.raises(PreflightRejected):
            preflight({"AERYNZA_AI_RAM_BUDGET_BYTES": budget}, PROFILES, ALIASES)
    with pytest.raises(PreflightRejected, match="Unknown"):
        preflight({"ASCENSION_MODEL_PROFILE": "not-a-profile"}, PROFILES, ALIASES)
    bad = tmp_path / "profiles.json"
    bad.write_text(json.dumps({"starter": {
        "filename": "bootstrap.gguf", "size_bytes": 1, "sha256": "invalid"
    }}), encoding="utf-8")
    with pytest.raises(PreflightRejected, match="SHA-256"):
        preflight({"ASCENSION_MODEL_PROFILE": "starter"}, bad, tmp_path / "missing.json")


def test_source_manifest_sha_and_filename_are_required():
    result = preflight({"ASCENSION_MODEL_PROFILE": "starter"}, PROFILES, ALIASES)
    assert len(result["artifact_sha256"]) == 64
    assert result["minimum_render_plan"] == "free_or_starter_512mb"
