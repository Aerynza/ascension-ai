"""Regression tests: loading v231 must never certify release by itself."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.core.release_readiness import REQUIRED_RELEASE_GATES, evaluate_release_gates

MODEL = {"ready": True, "profile": "pro_v231"}
PROFILE = {"sha256": "a" * 64}


def report(age_days=0, profile="pro_v231", model_sha=None):
    return {
        "profile": profile, "model_sha256": model_sha or PROFILE["sha256"],
        "operator_approval": True,
        "verified_at": (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat(),
        "gates": {
            name: {"passed": True, "evidence_ref": "release-review/" + name}
            for name in REQUIRED_RELEASE_GATES
        },
    }


class ReleaseReadinessTests(unittest.TestCase):
    def test_readiness_does_not_advertise_verified_actor_before_middleware(self):
        import asyncio
        from src.serving.api import replacement_readiness
        state = asyncio.run(replacement_readiness())
        self.assertEqual(state["signed_actor_admission"], {
            "enforced": False,
            "durable_replay_protection": False,
            "account_subject_verified": False,
        })
        self.assertFalse(state["replacement_ready"])
        with patch.dict(os.environ, {
            "AERYNZA_AI_AUTH_MODE": "production",
            "AERYNZA_AI_ENABLE_PROVIDER_ROUTES": "true",
        }):
            production = asyncio.run(replacement_readiness())
            self.assertTrue(production["native_inference_only"])
            self.assertFalse(production["outside_provider"])
            self.assertFalse(production["signed_actor_admission"]["enforced"])

    def test_model_loaded_without_evidence_is_not_qualified(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(any(evaluate_release_gates(MODEL, PROFILE, signed_admission_active=True).values()))

    def test_operator_review_and_pinned_evidence_all_required(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "review.json"
            raw = json.dumps(report()).encode()
            path.write_bytes(raw)
            env = {
                "AERYNZA_NATIVE_REPLACEMENT_APPROVED": "1",
                "AERYNZA_NATIVE_RELEASE_EVIDENCE_FILE": str(path),
                "AERYNZA_NATIVE_RELEASE_EVIDENCE_SHA256": hashlib.sha256(raw).hexdigest(),
            }
            with patch.dict(os.environ, env, clear=True):
                self.assertFalse(any(evaluate_release_gates(MODEL, PROFILE).values()),
                                 "Shared-bearer runtime cannot self-certify even with a signed evidence report")
                self.assertTrue(all(evaluate_release_gates(MODEL, PROFILE, signed_admission_active=True).values()))
                self.assertFalse(any(evaluate_release_gates({"ready": False, "profile": "pro_v231"}, PROFILE, signed_admission_active=True).values()))
                self.assertFalse(any(evaluate_release_gates({"ready": True, "profile": "starter"}, PROFILE, signed_admission_active=True).values()))
                os.environ["ASCENSION_VERIFY_MODEL_ON_BOOT"] = "0"
                self.assertFalse(any(evaluate_release_gates(MODEL, PROFILE, signed_admission_active=True).values()))
                os.environ.pop("ASCENSION_VERIFY_MODEL_ON_BOOT")
                os.environ["AERYNZA_NATIVE_RELEASE_EVIDENCE_SHA256"] = "b" * 64
                self.assertFalse(any(evaluate_release_gates(MODEL, PROFILE, signed_admission_active=True).values()))

    def test_missing_gate_or_false_claim_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            item = report()
            item["gates"]["interactive_latency"]["passed"] = False
            item["gates"].pop("shell_isolation")
            raw = json.dumps(item).encode()
            path = Path(temp) / "review.json"
            path.write_bytes(raw)
            env = {
                "AERYNZA_NATIVE_REPLACEMENT_APPROVED": "1",
                "AERYNZA_NATIVE_RELEASE_EVIDENCE_FILE": str(path),
                "AERYNZA_NATIVE_RELEASE_EVIDENCE_SHA256": hashlib.sha256(raw).hexdigest(),
            }
            with patch.dict(os.environ, env, clear=True):
                gates = evaluate_release_gates(MODEL, PROFILE, signed_admission_active=True)
                self.assertFalse(gates["shell_isolation"])
                self.assertFalse(gates["interactive_latency"])
                self.assertTrue(gates["conversation_quality"])

    def test_stale_or_wrong_model_attestation_fails_closed(self):
        for item in [report(age_days=31), report(model_sha="f" * 64), report(profile="starter")]:
            with self.subTest(item=item["profile"], date=item["verified_at"]):
                with tempfile.TemporaryDirectory() as temp:
                    raw = json.dumps(item).encode()
                    path = Path(temp) / "review.json"
                    path.write_bytes(raw)
                    env = {
                        "AERYNZA_NATIVE_REPLACEMENT_APPROVED": "1",
                        "AERYNZA_NATIVE_RELEASE_EVIDENCE_FILE": str(path),
                        "AERYNZA_NATIVE_RELEASE_EVIDENCE_SHA256": hashlib.sha256(raw).hexdigest(),
                    }
                    with patch.dict(os.environ, env, clear=True):
                        self.assertFalse(any(evaluate_release_gates(MODEL, PROFILE, signed_admission_active=True).values()))


if __name__ == "__main__":
    unittest.main()
