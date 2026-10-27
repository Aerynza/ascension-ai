"""Run the real local LifeOS signer against native admission and ASGI middleware.

No model, live HTTP endpoint, Redis service or secret is used. If desktop-only
keyring is absent, a test substitute raises on any attempted keyring operation.
Run from the artifact root: PYTHONPATH=native:offline-deps python offline_cross_runtime.py
"""
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from threading import Lock
import time
import types
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = Path(os.getenv("AERYNZA_NATIVE_SOURCE_ROOT", str(Path(__file__).resolve().parents[1])))
PRODUCT_ROOT = Path(os.getenv("AERYNZA_PRODUCT_SOURCE_ROOT", str(ROOT / "lifeos")))
sys.path.insert(0, str(NATIVE_ROOT))
SIGNER = PRODUCT_ROOT / "services/aerynza-native-envelope.js"
NODE_EXECUTABLE = shutil.which("node")
if not NODE_EXECUTABLE:
    raise RuntimeError("Node is required for the cross-runtime contract")
KEY = b"s" * 32
ENVIRONMENT = {
    "ASCENSION_AI_AUTH_MODE": "production",
    "ASCENSION_AI_REQUIRE_SIGNED_REQUESTS": "true",
    "ASCENSION_AI_SERVICE_TOKEN": "offline-service-token",
    "ASCENSION_AI_SERVICE_SHELLS": "ap,lifeos,nexus_home,nexus_family,sprout,executive",
    "ASCENSION_AI_TRUSTED_ISSUER": "aerynza-product",
    "ASCENSION_AI_SIGNING_KEY_ID": "current",
    "ASCENSION_AI_SIGNING_KEY_BASE64": base64.b64encode(KEY).decode(),
    "ASCENSION_AI_REPLAY_REDIS_URL": "redis://offline.invalid/0",
}
if importlib.util.find_spec("keyring") is None:
    desktop_keyring = types.ModuleType("keyring")
    def unavailable(*args, **kwargs):
        raise AssertionError("Desktop keyring must not be used in native admission tests")
    desktop_keyring.get_password = unavailable
    desktop_keyring.set_password = unavailable
    sys.modules["keyring"] = desktop_keyring

from src.serving import api
from src.serving.envelope_gateway import admit_request
from src.serving.envelope_validation import InvalidEnvelope
from src.serving.replay_guard import RedisNonceStore, ReplayDetected
from src.serving.request_integrity import canonical_payload


def sign(body, *, shell="ap", subject="user-123", method="POST",
         path="/v1/memory/candidates", nonce=None, now=None):
    data = {"body": json.dumps(body, ensure_ascii=False), "shell": shell,
            "subject": subject, "method": method, "path": path,
            "nonce": nonce or os.urandom(12).hex(),
            "now": int(time.time()) if now is None else now}
    script = "const {signedNativeHeaders}=require(process.argv[1]); process.stdout.write(JSON.stringify(signedNativeHeaders(JSON.parse(process.argv[2]))));"
    env = {**os.environ, **ENVIRONMENT}
    return json.loads(subprocess.check_output([NODE_EXECUTABLE, "-e", script, str(SIGNER), json.dumps(data, ensure_ascii=False)], env=env, text=True))


class AtomicNonceStore:
    def __init__(self):
        self.seen = set()
        self.lock = Lock()

    def claim(self, key, ttl_seconds):
        with self.lock:
            if key in self.seen:
                return False
            self.seen.add(key)
            return True


class RealSignerVectors(unittest.TestCase):
    def test_real_signer_to_native_admission(self):
        fixtures = json.loads((NATIVE_ROOT / "tests/fixtures/native-canonicalization.json").read_text())
        for fixture in fixtures:
            with self.subTest(fixture=fixture["id"]):
                headers = sign(fixture["payload"], path="/v1/intelligence")
                envelope = json.loads(headers["X-Aerynza-Envelope"])
                identity = admit_request(envelope, fixture["payload"], headers["X-Aerynza-Signature"],
                                         keys={"current": KEY}, trusted_issuer="aerynza-product",
                                         nonce_store=AtomicNonceStore())
                self.assertEqual(canonical_payload(fixture["payload"]).decode(), fixture["canonical"])
                self.assertEqual(identity["subject"], "user-123")

        # PR104 also has a separately mounted consented memory route. Keep its
        # strict user:<id> mapping and integer-only signer compatible with the
        # same native admission boundary when that client is present.
        memory_client = PRODUCT_ROOT / "lib/native-ai-client.js"
        if memory_client.is_file():
            payload = {"text": "Remember my preference: café"}
            config = {**ENVIRONMENT, "AERYNZA_NATIVE_AI_URL": "https://ascension-ai.onrender.com"}
            script = "const {callNativeAi}=require(process.argv[1]); callNativeAi({route:'memory-candidates',userId:123,payload:JSON.parse(process.argv[2]),env:JSON.parse(process.argv[3]),fetchImpl:async(url,options)=>{process.stdout.write(JSON.stringify(options));return {ok:true,headers:{get:()=>null},text:async()=>'{\\\"candidates\\\":[]}'}}}).catch(error=>{process.stderr.write(error.message);process.exitCode=1});"
            signed = json.loads(subprocess.check_output([NODE_EXECUTABLE, "-e", script, str(memory_client), json.dumps(payload), json.dumps(config)], text=True))
            identity = admit_request(json.loads(signed["headers"]["X-Aerynza-Envelope"]), json.loads(signed["body"]), signed["headers"]["X-Aerynza-Signature"],
                                     keys={"current": KEY}, trusted_issuer="aerynza-product", nonce_store=AtomicNonceStore())
            self.assertEqual(identity["subject"], "user:123")

    def test_concurrent_claim_has_one_winner_and_redis_uses_nx_expiry(self):
        class RedisSubstitute:
            def __init__(self):
                self.store = AtomicNonceStore()
                self.calls = []
            def set(self, key, value, *, nx, ex):
                self.calls.append((value, nx, ex))
                return self.store.claim(key, ex)
        client = RedisSubstitute()
        store = RedisNonceStore(client)
        body = {"text": "hi"}
        headers = sign(body, nonce="same-across-workers")
        envelope = json.loads(headers["X-Aerynza-Envelope"])
        def admit():
            try:
                admit_request(envelope, body, headers["X-Aerynza-Signature"],
                              keys={"current": KEY}, trusted_issuer="aerynza-product", nonce_store=store)
                return True
            except ReplayDetected:
                return False
        with ThreadPoolExecutor(max_workers=12) as workers:
            self.assertEqual(sum(workers.map(lambda _: admit(), range(12))), 1)
        self.assertTrue(all(value == "1" and nx is True and 1 <= ex <= 90 for value, nx, ex in client.calls))


class RealSignerHttpBoundary(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # The managed network sandbox can suppress asyncio's thread wake-up
        # socket. A local timer keeps the loop moving; verification still runs
        # in the real worker, and no service/network operation is substituted.
        loop = asyncio.get_running_loop()
        def tick():
            if not loop.is_closed():
                loop.call_later(0.01, tick)
        loop.call_later(0.01, tick)
        self.environment = patch.dict(os.environ, ENVIRONMENT, clear=True)
        self.environment.start()
        self.store = AtomicNonceStore()
        self.nonce_patch = patch.object(api, "redis_nonce_store", lambda _: self.store)
        self.nonce_patch.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://offline")
    async def asyncTearDown(self):
        await self.client.aclose()
        self.nonce_patch.stop()
        self.environment.stop()
    def auth(self, headers):
        return {**headers, "Authorization": "Bearer offline-service-token"}

    async def test_memory_request_is_accepted_once(self):
        body = {"text": "Remember my preference"}
        headers = self.auth(sign(body))
        first = await self.client.post("/v1/memory/candidates", json=body, headers=headers)
        replay = await self.client.post("/v1/memory/candidates", json=body, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(replay.status_code, 401, replay.text)

    async def test_signed_intelligence_reaches_model_not_ready_guard(self):
        body = {"shell": "ap", "messages": [{"role": "user", "content": "hello"}],
                "temperature": 1e-6, "max_tokens": 32,
                "context": json.loads('{"10":"ten","2":"two","__proto__":{"fact":"preserved"},"\\ue000":"bmp","\\ud800\\udc00":"astral"}')}
        response = await self.client.post("/v1/intelligence", json=body, headers=self.auth(sign(body, path="/v1/intelligence")))
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(response.json()["detail"], "Aerynza native model is not ready.")

    async def test_signed_readiness_get(self):
        response = await self.client.get("/v1/readiness", headers=self.auth(sign({}, method="GET", path="/v1/readiness")))
        self.assertEqual(response.status_code, 200, response.text)

    async def test_body_tamper_is_rejected(self):
        headers = self.auth(sign({"text": "authorized"}))
        response = await self.client.post("/v1/memory/candidates", json={"text": "altered"}, headers=headers)
        self.assertEqual(response.status_code, 401)

    async def test_wrong_method_and_path_are_rejected(self):
        body = {"text": "hello"}
        headers = self.auth(sign(body))
        path_response = await self.client.post("/v1/iphone/inbox", json=body, headers=headers)
        method_response = await self.client.get("/v1/memory/candidates", headers=headers)
        self.assertEqual(path_response.status_code, 401)
        self.assertEqual(method_response.status_code, 401)

    async def test_shell_cannot_default_into_other_handler_identity(self):
        body = {"messages": [{"role": "user", "content": "hello"}]}
        response = await self.client.post("/v1/intelligence", json=body, headers=self.auth(sign(body, shell="nexus_family", path="/v1/intelligence")))
        self.assertEqual(response.status_code, 401)

    async def test_expired_signature_is_rejected(self):
        body = {"text": "hello"}
        headers = self.auth(sign(body, now=int(time.time()) - 100))
        response = await self.client.post("/v1/memory/candidates", json=body, headers=headers)
        self.assertEqual(response.status_code, 401)

    async def test_nonce_store_outage_is_rejected(self):
        class FailedStore:
            def claim(self, key, ttl_seconds):
                raise ConnectionError("offline outage")
        body = {"text": "hello"}
        with patch.object(api, "redis_nonce_store", lambda _: FailedStore()):
            response = await self.client.post("/v1/memory/candidates", json=body, headers=self.auth(sign(body)))
        self.assertEqual(response.status_code, 503)

    async def test_personal_subject_mismatch_is_rejected(self):
        body = {"shell": "lifeos", "scope": "human", "subject_id": "someone-else", "context": {}}
        response = await self.client.post("/v1/thesis", json=body, headers=self.auth(sign(body, shell="lifeos", path="/v1/thesis")))
        self.assertEqual(response.status_code, 401)

    async def test_personal_subject_match_is_accepted(self):
        body = {"shell": "lifeos", "scope": "human", "subject_id": "user-123", "context": {}}
        response = await self.client.post("/v1/thesis", json=body, headers=self.auth(sign(body, shell="lifeos", path="/v1/thesis")))
        self.assertEqual(response.status_code, 200, response.text)

    async def test_sensitive_resource_thesis_remains_denied(self):
        for scope, shell in (("home", "nexus_home"), ("family", "nexus_family"), ("sprout", "sprout")):
            with self.subTest(scope=scope):
                body = {"shell": shell, "scope": scope, "subject_id": "unverified-resource", "context": {}}
                response = await self.client.post("/v1/thesis", json=body, headers=self.auth(sign(body, shell=shell, path="/v1/thesis")))
                self.assertEqual(response.status_code, 403, response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

