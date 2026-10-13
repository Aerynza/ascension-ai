"""Native replacement is an attested release decision, never merely model_loaded."""
import base64
import hashlib
import hmac
import json

from src.core.release_gate import REQUIRED_GATES, SUITE, release_status


MODEL_SHA = "ab" * 32
KEY = b"readiness-release-key-for-test-only--32bytes"
ISSUED = 1_780_000_000
RUNTIME = {"ready": True, "profile": "pro_v231"}
PROFILE = {"sha256": MODEL_SHA}


def write_attestation(tmp_path, payload_overrides=None, **kwargs):
    payload = {
        "suite": SUITE,
        "profile": "pro_v231",
        "model_sha256": MODEL_SHA,
        "issued_at": ISSUED,
        "expires_at": ISSUED + 3600,
        "evaluation_cases": 30,
        "ci_run_url": "https://github.com/acensionlifeos-dev/ascension-ai/actions/runs/12345",
        "gates": {gate: True for gate in REQUIRED_GATES},
    }
    payload.update(payload_overrides or {})
    signature = hmac.new(KEY, json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(), hashlib.sha256).hexdigest()
    path = tmp_path / "production_replacement_attestation.json"
    path.write_text(json.dumps({"payload": payload, "signature": signature}), encoding="utf-8")
    env = {
        "ASCENSION_AI_READINESS_ATTESTATION_KEY_BASE64": base64.b64encode(KEY).decode(),
        "ASCENSION_NATIVE_REPLACEMENT_APPROVED": "true",
    }
    env.update(kwargs)
    return path, env


def check(tmp_path, overrides=None, runtime=None, profile=None, now=ISSUED + 120, **env_overrides):
    path, env = write_attestation(tmp_path, overrides, **env_overrides)
    return release_status(runtime or RUNTIME, profile or PROFILE, env=env, evidence_path=path, now=now)


def test_unattested_model_cannot_claim_replacement(tmp_path):
    result = release_status(RUNTIME, PROFILE, env={}, evidence_path=tmp_path / "missing.json", now=ISSUED + 30)
    assert result["candidate_ready"] is True
    assert result["replacement_ready"] is False
    assert result["evidence"] == {}
    assert not any(result["gates"].values())


def test_valid_signed_evaluation_and_operator_approval_can_pass(tmp_path):
    result = check(tmp_path)
    assert result["candidate_ready"] is True
    assert result["replacement_ready"] is True
    assert result["evidence_status"] == "approved"
    assert all(result["gates"].values())
    assert result["evidence"]["model_sha256"] == MODEL_SHA
    assert result["evidence"]["evaluation_cases"] == 30


def test_loaded_model_is_not_a_qa_pass(tmp_path):
    result = check(tmp_path, runtime={"ready": False, "profile": "pro_v231"})
    assert result["candidate_ready"] is False
    assert result["replacement_ready"] is False


def test_signature_tampering_or_mismatched_model_rejected(tmp_path):
    path, env = write_attestation(tmp_path)
    doc = json.loads(path.read_text())
    doc["payload"]["evaluation_cases"] = 999999
    path.write_text(json.dumps(doc))
    result = release_status(RUNTIME, PROFILE, env=env, evidence_path=path, now=ISSUED + 100)
    assert result["replacement_ready"] is False
    assert result["evidence"] == {}
    assert check(tmp_path, {"model_sha256": "cd" * 32})["replacement_ready"] is False


def test_stale_future_dated_or_overlong_evidence_rejected(tmp_path):
    assert check(tmp_path, now=ISSUED + 4000)["replacement_ready"] is False
    assert check(tmp_path, now=ISSUED - 400)["replacement_ready"] is False
    assert check(tmp_path, {"expires_at": ISSUED + 8 * 86400})["replacement_ready"] is False


def test_missing_or_failing_required_gate_remains_not_ready(tmp_path):
    gates = {gate: True for gate in REQUIRED_GATES}
    gates["action_integrity"] = False
    result = check(tmp_path, {"gates": gates})
    assert result["replacement_ready"] is False
    assert result["evidence_status"] == "evaluation_failed"
    gates.pop("native_primary_canary")
    assert check(tmp_path, {"gates": gates})["replacement_ready"] is False


def test_operator_approval_cannot_be_inferred_from_evaluation(tmp_path):
    result = check(tmp_path, ASCENSION_NATIVE_REPLACEMENT_APPROVED="false")
    assert result["replacement_ready"] is False
    assert result["evidence_status"] == "awaiting_operator_approval"
    assert all(result["gates"].values())


def test_invalid_evaluation_metadata_or_untrusted_ci_url_rejected(tmp_path):
    for override in (
        {"evaluation_cases": 0},
        {"ci_run_url": "https://attacker.example/eval"},
        {"profile": "another_checkpoint"},
        {"suite": "unknown_suite"},
        {"issued_at": True},
    ):
        assert check(tmp_path, override)["replacement_ready"] is False
