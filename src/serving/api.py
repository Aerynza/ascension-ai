"""Standalone self-contained Aerynza AI service."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal


SESSIONS = set()
KEYRING_SERVICE = "Aerynza AI"
SESSION_CONTEXTS: dict[str, dict] = {}
_SESSION_LOCK = Lock()
MAX_SESSION_CONTEXT_BYTES = 200_000


def _session_key(session_id: str, token: str | None) -> str:
    """Bind a nonproduction session to its verified caller token."""
    return hashlib.sha256(f"{token or ''}:{session_id}".encode("utf-8")).hexdigest()


def _strip_provider_keys(value):
    """Drop provider credentials before anything is persisted."""
    if isinstance(value, dict):
        return {
            key: _strip_provider_keys(item)
            for key, item in value.items()
            if key != "provider_keys"
        }
    if isinstance(value, list):
        return [_strip_provider_keys(item) for item in value]
    return value


def get_session_context(session_id: str, authorization: str | None) -> dict:
    if _auth_mode() == "production":
        raise HTTPException(status_code=403, detail="Shared-token session storage disabled pending caller-bound identity")
    if not session_id:
        return {}
    key = _session_key(session_id, authorization)
    with _SESSION_LOCK:
        return SESSION_CONTEXTS.get(key, {}).copy()


def set_session_context(session_id: str, context: dict, authorization: str | None, merge: bool = False) -> dict:
    if _auth_mode() == "production":
        raise HTTPException(status_code=403, detail="Shared-token session storage disabled pending caller-bound identity")
    safe_context = _strip_provider_keys(context)
    if not session_id:
        return safe_context
    key = _session_key(session_id, authorization)
    with _SESSION_LOCK:
        existing = SESSION_CONTEXTS.get(key, {}) if merge else {}
        merged = _merge_context(existing, safe_context)
        if len(json.dumps(merged, default=str)) > MAX_SESSION_CONTEXT_BYTES:
            raise ValueError("session context exceeds size limit")
        SESSION_CONTEXTS[key] = merged
    return merged


def _merge_context(existing: dict, new: dict) -> dict:
    result = {**existing}
    for key, value in new.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_context(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


def resolve_session_context(request, authorization: str | None) -> dict:
    """Merge persisted session context with request context; persist the result when asked."""
    if not getattr(request, "session_id", None):
        return request.context
    stored = get_session_context(request.session_id, authorization)
    merged = _merge_context(stored, request.context)
    if getattr(request, "persist_context", False):
        set_session_context(request.session_id, request.context, authorization, merge=True)
    return merged


from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.serving.http_envelope import signing_enabled, load_signing_configuration, check_http_envelope, redis_nonce_store
from src.serving.envelope_validation import InvalidEnvelope
from src.core.capabilities import CAPABILITIES
from src.core.action_runtime import shell_action_catalog, shell_allows_action, validate_action_receipt
from src.core.cognition import TALENTS, build_action_execution_contract, build_cognitive_packet, extract_memory_candidates, hybrid_retrieve
from src.core.contracts import Shell, Tier
from src.core.media_executor import generate_image, parse_image_request
from src.core.model_runtime import ModelRamBudgetExceeded, NativeInferenceQueueTimeout, runtime
from src.core.release_gate import REQUIRED_GATES, release_status
from src.financial.plaid_client import get_balances, parse_balance_query
from src.core.orchestrator import deterministic_response, prepare_inference, respond, scope_context, surface_plan
from src.core.safety import medical_emergency_response
from src.core.thesis import build_member_thesis_contribution, build_thesis
from src.phone import android_bridge, iphone_bridge
from src.windows import executor


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public"
APP_VERSION = "2.3.2-native-production"
MAX_MESSAGES = 24
MAX_MESSAGE_LENGTH = 12_000
MAX_SIGNED_BODY_BYTES = 1_048_576
def production_replacement_enabled() -> bool:
    # A loaded checkpoint alone cannot certify outside-provider replacement.
    # This remains false without current, model-bound attested evaluation gates
    # and explicit operator approval.
    return bool(release_status(runtime.status(), runtime.profile)["replacement_ready"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_auth_configuration()
    try:
        await asyncio.to_thread(runtime.load)
    except ModelRamBudgetExceeded:
        raise
    except Exception as error:
        print(f"Aerynza native model did not load: {error}")
    yield


app = FastAPI(title="Aerynza AI Native Core", version=APP_VERSION, lifespan=lifespan)
allowed_origins = [value.strip() for value in os.getenv("ASCENSION_AI_ALLOWED_ORIGINS", "").split(",") if value.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Aerynza-Envelope", "X-Aerynza-Signature"],
)
app.mount("/static", StaticFiles(directory=str(PUBLIC)), name="static")


@app.middleware("http")
async def signed_request_boundary(request, call_next):
    """Staged signing enforcement; requires real Redis and matching product signer."""
    if _auth_mode() != "production" or not signing_enabled():
        return await call_next(request)
    if request.url.path in {"/docs", "/redoc", "/openapi.json"}:
        return JSONResponse(status_code=404, content={"detail": "Not found"})
    if request.url.query and request.url.path.startswith("/v1/"):
        return JSONResponse(status_code=400, content={"detail": "Unsigned query parameters are not supported"})
    protected_get = request.method == "GET" and (
        request.url.path.startswith("/v1/") or request.url.path == "/model/info"
    )
    if request.method not in ("POST", "GET"):
        return JSONResponse(status_code=405, content={"detail": "Method not allowed"})
    if request.method == "GET" and not protected_get:
        return await call_next(request)
    if not _authorized_token(request.headers.get("authorization")):
        return JSONResponse(status_code=401, content={"detail": "Authentication required"})
    try:
        declared_length = request.headers.get("content-length")
        if declared_length is not None and (not declared_length.isascii() or not declared_length.isdigit()):
            return JSONResponse(status_code=400, content={"detail": "Invalid content length"})
        if declared_length is not None and int(declared_length) > MAX_SIGNED_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "Request too large"})
        # Bound accumulation even when Content-Length is absent or understates
        # a chunked body. Do not drain an oversized stream before rejecting it.
        chunks = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_SIGNED_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request too large"})
            chunks.append(chunk)
        body_bytes = b"".join(chunks)
        # Starlette's cached middleware request replays this bounded body to
        # downstream JSON validation after signature verification.
        request._body = body_bytes
        payload = {} if protected_get and not body_bytes else json.loads(body_bytes)
        if not isinstance(payload, dict):
            raise InvalidEnvelope("Expected JSON object body")
        config = load_signing_configuration()
        nonce_store = redis_nonce_store(config[2])
        identity = await asyncio.to_thread(
            check_http_envelope, request.headers, payload, config=config,
            nonce_store=nonce_store, method=request.method, path=request.url.path,
        )
        # Shell-aware operations must explicitly declare the same shell as the signed envelope.
        # Otherwise a non-AP signature could reach a handler whose Pydantic default is AP.
        shell_bound_routes = {
            "/v1/intelligence", "/v1/stream", "/chat", "/v1/cognition",
            "/v1/agent/plan", "/v1/retrieve", "/v1/surface-plan",
            "/v1/relationships/feed", "/v1/session/data-panels",
            "/v1/thesis", "/v1/thesis/contribution",
            "/v1/actions/receipt/validate",
        }
        if request.url.path in shell_bound_routes and payload.get("shell") != identity["shell"]:
            raise InvalidEnvelope("Explicit matching shell required")
        declared_shell = payload.get("shell")
        if declared_shell is not None and declared_shell != identity["shell"]:
            raise InvalidEnvelope("Shell identity mismatch")
        # Core is an entitlement tier; Creation is an AP workspace.
        # Neither is an independent authenticated production shell.
        if identity["shell"] in {"core", "creation"}:
            raise InvalidEnvelope("Tier or workspace is not an AI shell")
        if identity["shell"] not in _authorized_shells():
            raise InvalidEnvelope("Signed shell not authorized for this service")
        # Memory extraction is personal AP work, never a family, home, or child shell operation.
        if request.url.path == "/v1/memory/candidates" and identity["shell"] != "ap":
            raise InvalidEnvelope("Memory candidate route requires AP shell")
        if request.url.path.startswith("/v1/actions/catalog/"):
            catalog_shell = request.url.path.rsplit("/", 1)[-1]
            if catalog_shell != identity["shell"]:
                raise InvalidEnvelope("Catalog shell mismatch")
        request.state.signed_identity = identity
    except (InvalidEnvelope, ValueError, TypeError):
        return JSONResponse(status_code=401, content={"detail": "Invalid signed request"})
    except Exception:
        return JSONResponse(status_code=503, content={"detail": "Signed request verification unavailable"})
    return await call_next(request)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
    response.headers["Cache-Control"] = "no-store"
    return response


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be empty")
        return cleaned


class IntelligenceRequest(BaseModel):
    shell: Shell = Shell.AP
    tier: Tier = Tier.LIFE_OS
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)
    context: dict = Field(default_factory=dict)
    surface: str = Field(default="chat", max_length=100)
    mode: Literal["conversation", "analysis", "planning", "proactive", "background"] = "conversation"
    allowed_capabilities: list[str] = Field(default_factory=list, max_length=100)
    available_actions: list[str] = Field(default_factory=list, max_length=100)
    temperature: float = Field(default=0.65, ge=0.0, le=1.2)
    max_tokens: int = Field(default=500, ge=32, le=1600)
    session_id: str = Field(default="", max_length=120)
    persist_context: bool = False

    @field_validator("messages")
    @classmethod
    def require_user_turn(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if value[-1].role != "user":
            raise ValueError("the final message must be from the user")
        return value

    @field_validator("context")
    @classmethod
    def bound_context(cls, value: dict) -> dict:
        if len(json.dumps(value, default=str)) > 50_000:
            raise ValueError("context packet is too large")
        return value

    @field_validator("allowed_capabilities")
    @classmethod
    def validate_capabilities(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip().lower() for item in value if item.strip()))
        unknown = [item for item in cleaned if item not in CAPABILITIES]
        if unknown:
            raise ValueError(f"unknown capabilities: {', '.join(unknown)}")
        return cleaned


class SurfacePlanRequest(BaseModel):
    shell: Shell = Shell.LIFE_OS
    tier: Tier = Tier.LIFE_OS
    trigger: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    context: dict = Field(default_factory=dict)
    available_actions: list[str] = Field(default_factory=list, max_length=100)
    allowed_capabilities: list[str] = Field(default_factory=list, max_length=100)
    session_id: str = Field(default="", max_length=120)
    persist_context: bool = True


class CognitionRequest(SurfacePlanRequest):
    pass


class RetrievalRequest(BaseModel):
    shell: Shell = Shell.AP
    query: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    context: dict = Field(default_factory=dict)
    top_k: int = Field(default=6, ge=1, le=10)


class MemoryCandidateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class SessionContextRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    context: dict = Field(default_factory=dict)
    merge: bool = True


class SessionRefreshRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    shell: Shell = Shell.AP
    tier: Tier = Tier.LIFE_OS
    trigger: str = Field(default="refresh", min_length=1, max_length=MAX_MESSAGE_LENGTH)
    surface: str = Field(default="chat", max_length=100)
    allowed_capabilities: list[str] = Field(default_factory=list, max_length=100)
    available_actions: list[str] = Field(default_factory=list, max_length=100)


class RelationshipFeedRequest(BaseModel):
    shell: Shell = Shell.AP
    tier: Tier = Tier.LIFE_OS
    context: dict = Field(default_factory=dict)
    surface: str = Field(default="relationship_feed", max_length=100)
    session_id: str = Field(default="", max_length=120)
    persist_context: bool = True


class ThesisRequest(BaseModel):
    scope: Literal["human", "sprout", "home", "family", "product"]
    subject_id: str = Field(min_length=1, max_length=200)
    shell: Shell
    context: dict = Field(default_factory=dict)

    @field_validator("context")
    @classmethod
    def bound_thesis_context(cls, value: dict) -> dict:
        if len(json.dumps(value, default=str)) > 200_000:
            raise ValueError("thesis context packet is too large")
        return value


class ThesisContributionRequest(BaseModel):
    member_id: str = Field(min_length=1, max_length=200)
    target_scope: Literal["nexus_home", "nexus_family"]
    shell: Shell
    human_thesis: dict
    selections: list[dict] = Field(min_length=1, max_length=100)
    consent_receipt_id: str = Field(min_length=1, max_length=300)


class LegacyGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    max_new_tokens: int = Field(default=300, ge=32, le=1000)
    temperature: float = Field(default=0.65, ge=0.0, le=1.2)


class ActionReceiptRequest(BaseModel):
    shell: Shell
    action: str = Field(min_length=1, max_length=120)
    receipt_fields: list[str] = Field(default_factory=list, max_length=40)
    receipt: dict = Field(default_factory=dict)


def _auth_mode() -> str:
    if os.getenv("RENDER", "").strip().lower() == "true":
        return "production"
    mode = os.getenv("AERYNZA_AI_AUTH_MODE", os.getenv("ASCENSION_AI_AUTH_MODE", "production")).strip().lower()
    if mode not in {"production", "development"}:
        raise RuntimeError("ASCENSION_AI_AUTH_MODE must be production or development")
    return mode


def _authorized_token(authorization: str | None) -> bool:
    if not authorization or not authorization.lower().startswith("bearer "):
        return False
    supplied = authorization[7:].strip()
    if not supplied:
        return False
    service_tokens = [os.getenv(name, "").strip() for name in (
        "ASCENSION_AI_SERVICE_TOKEN", "AERYNZA_AI_SERVICE_TOKEN"
    )]
    supplied_digest = hashlib.sha256(supplied.encode("utf-8")).digest()
    if any(token and hmac.compare_digest(supplied_digest, hashlib.sha256(token.encode("utf-8")).digest()) for token in service_tokens):
        return True
    if _auth_mode() == "development":
        if supplied in SESSIONS:
            return True
        if os.getenv("ASCENSION_AI_ALLOW_TEST_TOKEN", "").strip().lower() != "true":
            return False
        test_token = os.getenv("ASCENSION_AI_TEST_TOKEN", "").strip()
        return bool(test_token and hmac.compare_digest(supplied_digest, hashlib.sha256(test_token.encode("utf-8")).digest()))
    return False


def _authorized_shells() -> set[str]:
    """Production caller's allowed shell identities, configured server-side."""
    raw = os.getenv("ASCENSION_AI_SERVICE_SHELLS", "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def enforce_shell_access(shell: Shell) -> None:
    """Never trust a request body to grant a shell identity."""
    if _auth_mode() == "development":
        return
    if shell.value.lower() not in _authorized_shells():
        raise HTTPException(status_code=403, detail="Shell not authorized")


def require_access(authorization: str | None = Header(default=None)) -> str:
    """Require a caller token; never share development sessions across callers."""
    if _auth_mode() == "development" and os.getenv("ASCENSION_AI_LOCAL_DEV_BYPASS", "").lower() in {"1", "true"}:
        return 'explicit-local-development-bypass'
    if not _authorized_token(authorization):
        raise HTTPException(status_code=401, detail="Authentication required")
    return authorization[7:].strip()


def validate_auth_configuration() -> None:
    if _auth_mode() == "production":
        if not signing_enabled():
            raise RuntimeError("ASCENSION_AI_REQUIRE_SIGNED_REQUESTS=true is required in production")
        load_signing_configuration()
        if not os.getenv("ASCENSION_AI_SERVICE_TOKEN", "").strip():
            raise RuntimeError("ASCENSION_AI_SERVICE_TOKEN is required in production")
        if not _authorized_shells():
            raise RuntimeError("ASCENSION_AI_SERVICE_SHELLS is required in production")


def require_certified_action_gateway() -> None:
    """Fail closed until production action authorization and receipts are certified."""
    if _auth_mode() == "production" or os.getenv("AERYNZA_AI_ENABLE_DEVICE_ROUTES", "").strip().lower() != "true":
        raise HTTPException(status_code=403, detail="Direct device execution requires a verified development enablement")


def require_native_ready() -> None:
    if not runtime.status()["ready"]:
        raise HTTPException(status_code=503, detail=runtime.status()["error"] or "Aerynza native model is not ready.")


def effective_max_tokens(requested: int, mode: str = "conversation") -> int:
    """Bound interactive latency while preserving deeper queued reasoning."""
    profile_limits = {"starter": 72, "standard": 420, "pro": 1200, "candidate": 1200, "deep": 1400}
    mode_limits = {"conversation": 160, "proactive": 240, "planning": 420, "analysis": 680, "background": 1200}
    return min(requested, profile_limits.get(runtime.profile_name, requested), mode_limits.get(mode, 240))


def stream_error_payload(error: Exception) -> dict:
    """Return a stable public SSE error contract after response headers are sent."""
    if isinstance(error, NativeInferenceQueueTimeout):
        return {
            "code": "native_inference_queue_timeout",
            "message": "Aerynza AI is busy. Retry this request shortly.",
            "retryable": True,
            "http_equivalent": 504,
        }
    return {
        "code": "native_inference_failed",
        "message": "Aerynza AI could not complete this stream.",
        "retryable": True,
        "http_equivalent": 502,
    }


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(PUBLIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/capabilities")
async def capabilities_page() -> FileResponse:
    return FileResponse(PUBLIC / "capabilities.html")


@app.get("/health")
async def health() -> dict:
    model = runtime.status()
    replacement_ready = model["profile"] in RELEASE_READY_PROFILES and bool(model["ready"])
    return {
        "status": "healthy" if model["ready"] else "loading_or_degraded",
        "version": APP_VERSION,
        "mode": "ascension_native_local",
        "candidate_ready": model["ready"],
        "replacement_ready": replacement_ready,
        "provider": "Aerynza-Native" if model["ready"] else None,
        "model": model["model"],
        "outside_provider": False,
        "test_access_configured": bool(os.getenv("ASCENSION_AI_TEST_TOKEN", "").strip()),
        "service_access_configured": bool(os.getenv("ASCENSION_AI_SERVICE_TOKEN", "").strip()),
        "runtime": model,
    }


@app.get("/v1/readiness")
async def replacement_readiness(_: None = Depends(require_access)) -> dict:
    status = release_status(runtime.status(), runtime.profile)
    return {
        **status,
        "evaluation_suite": "aerynza_native_replacement_v1",
        "required_gates": list(REQUIRED_GATES),
        "promotion_rule": "A recent signed evaluation for this exact model and explicit operator approval are required before native primary replacement.",
        "runtime": runtime.status(),
    }


@app.get("/model/info")
async def model_info(_: None = Depends(require_access)) -> dict:
    return {
        "name": "Aerynza AI Native Core",
        "version": APP_VERSION,
        "runtime": runtime.status(),
        "shells": [shell.value for shell in Shell],
        "tiers": [tier.value for tier in Tier],
        "capability_domains": list(CAPABILITIES),
        "outside_provider": False,
        "production_replacement_enabled": production_replacement_enabled(),
        "promotion_rule": "Enable production replacement only after shell-specific evaluation gates pass.",
    }


@app.get("/v1/capability-report")
async def capability_report(_: None = Depends(require_access)) -> FileResponse:
    return FileResponse(PUBLIC / "capability_report.json")


@app.get("/v1/capabilities")
async def capabilities(_: None = Depends(require_access)) -> dict:
    return {
        "capabilities": CAPABILITIES,
        "shells": [shell.value for shell in Shell],
        "tiers": [tier.value for tier in Tier],
        "tier_scope": "all_ascension_tiers",
        "entitlement_rule": "The authenticated calling shell supplies and enforces tier permissions; the shared native core retains one capability model.",
    }


@app.get("/v1/actions/catalog/{shell}")
async def actions_catalog(shell: Shell, _: None = Depends(require_access)) -> dict:
    enforce_shell_access(shell)
    return {
        "shell": shell.value,
        "actions": shell_action_catalog(shell),
        "advertisement_rule": "The authenticated product shell advertises only currently connected executors in available_actions.",
        "model_readiness_rule": "A missing provider prevents dispatch but does not prevent planning or model qualification.",
    }


@app.post("/v1/actions/receipt/validate")
async def action_receipt_validate(request: ActionReceiptRequest, _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    if not shell_allows_action(request.shell, request.action):
        raise HTTPException(status_code=403, detail=f"{request.shell.value} cannot execute {request.action}")
    action = {"action": request.action, "receipt_fields": request.receipt_fields}
    result = validate_action_receipt(action, request.receipt)
    return {
        **result,
        "shell": request.shell.value,
        "completion_claim_allowed": result["valid"],
        "rule": "Only a valid receipt may be returned to AP as confirmed execution context.",
    }


@app.get("/capabilities")
async def capabilities_compat(access: None = Depends(require_access)) -> dict:
    """Compatibility alias retained for Devin's original SDK consumers."""
    return await capabilities(access)


@app.get("/v1/talents")
async def talents(_: None = Depends(require_access)) -> dict:
    active = [key for key, value in TALENTS.items() if value["state"] == "active"]
    shell_required = [key for key, value in TALENTS.items() if value["state"] == "shell_required"]
    return {
        "talents": TALENTS,
        "counts": {"active": len(active), "shell_required": len(shell_required)},
        "active": active,
        "shell_required": shell_required,
        "truth_rule": "A talent is active only when this service can produce its structured intelligence now; shell-required talents depend on permissioned LifeOS integrations.",
    }


@app.post("/v1/cognition")
async def cognition(request: CognitionRequest, authorization: str | None = Header(default=None), _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    context = resolve_session_context(request, authorization)
    scoped_context = scope_context(context, request.shell)
    packet = build_cognitive_packet(
        request.trigger,
        scoped_context,
        request.allowed_capabilities,
        request.available_actions,
    )
    execution_contract = build_action_execution_contract(packet, request.shell)
    return {
        "shell": request.shell.value,
        "tier": request.tier.value,
        "surface": context.get("surface", "chat"),
        **packet,
        "execution_contract": execution_contract,
        "outside_provider": False,
    }


@app.post("/v1/agent/plan")
async def agent_plan(request: CognitionRequest, access: None = Depends(require_access)) -> dict:
    """Plan but never silently execute; the authenticated product shell owns execution."""
    return await cognition(request, access)


@app.post("/v1/retrieve")
async def retrieve(request: RetrievalRequest, _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    scoped_context = scope_context(request.context, request.shell)
    return {
        "query": request.query,
        "results": hybrid_retrieve(request.query, scoped_context, request.top_k),
        "scope": "request_supplied_permissioned_context",
        "outside_provider": False,
    }


@app.post("/v1/memory/candidates")
async def memory_candidates(request: MemoryCandidateRequest, _: None = Depends(require_access)) -> dict:
    return {
        "candidates": extract_memory_candidates(request.text),
        "persistence": "calling_shell_validates_and_persists",
        "outside_provider": False,
    }


@app.post("/v1/thesis")
async def thesis(request: ThesisRequest, _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    if _auth_mode() == "production" and request.scope in {"sprout", "home", "family"}:
        raise HTTPException(status_code=403, detail="Resource ownership verification required for this thesis scope")
    required_shell = {
        "human": {Shell.AP, Shell.LIFE_OS},
        "sprout": {Shell.AP, Shell.LIFE_OS},
        "home": {Shell.NEXUS_HOME},
        "family": {Shell.NEXUS_FAMILY},
        "product": {Shell.CORE},
    }
    if request.shell not in required_shell[request.scope]:
        raise HTTPException(status_code=403, detail=f"{request.shell.value} cannot build the {request.scope} thesis")
    try:
        result = build_thesis(request.scope, request.subject_id, request.context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**result, "shell": request.shell.value, "outside_provider": False}


@app.post("/v1/thesis/contribution")
async def thesis_contribution(request: ThesisContributionRequest, _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    if _auth_mode() == "production":
        raise HTTPException(status_code=403, detail="Member consent and resource ownership verification required")
    if request.shell not in {Shell.AP, Shell.LIFE_OS}:
        raise HTTPException(status_code=403, detail="only the member's AP or LifeOS shell can prepare a thesis contribution")
    try:
        contribution = build_member_thesis_contribution(
            request.member_id,
            request.target_scope,
            request.human_thesis,
            request.selections,
            request.consent_receipt_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**contribution, "shell": request.shell.value, "outside_provider": False}


@app.post("/v1/surface-plan")
async def plan_surfaces(request: SurfacePlanRequest, authorization: str | None = Header(default=None), _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    context = resolve_session_context(request, authorization)
    return surface_plan(
        shell=request.shell,
        tier=request.tier,
        trigger=request.trigger,
        context=context,
        available_actions=request.available_actions,
        allowed_capabilities=request.allowed_capabilities,
    )


@app.post("/v1/session/context")
async def set_session(request: SessionContextRequest, access: str = Depends(require_access)) -> dict:
    merged = set_session_context(request.session_id, request.context, access, merge=request.merge)
    return {
        "session_id": request.session_id,
        "keys": list(merged.keys()),
        "size_bytes": len(json.dumps(merged, default=str)),
        "rule": "Session context is merged with any previously stored context for this token/session.",
    }


@app.post("/v1/session/data-panels")
async def session_data_panels(request: SessionRefreshRequest, access: str = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    context = get_session_context(request.session_id, access)
    scoped_context = scope_context(context, request.shell)
    packet = build_cognitive_packet(
        request.trigger,
        scoped_context,
        request.allowed_capabilities,
        request.available_actions,
    )
    return {
        "session_id": request.session_id,
        "shell": request.shell.value,
        "tier": request.tier.value,
        "surface": request.surface,
        "data_panels": packet.get("data_panels", []),
        "surface_recommendations": packet.get("surface_recommendations", []),
        "domains": packet.get("domains", []),
        "outside_provider": False,
    }


def _build_relationships_feed(context: dict, shell: Shell) -> dict:
    """Combine social, Aerynza, family, home, and dating entries into a mixed feed.

    Full relationship profiles stay in AP; lightweight social acquaintances are
    labelled so the screen can distinguish real friends from social-only contacts.
    """
    feed: list[dict] = []
    tabs: list[dict] = []
    full_profiles: list[dict] = []
    opportunities: list[dict] = []
    source_map = {
        "social_contacts": ("Social Media", "social", False),
        "aerynza_social": ("Aerynza Social", "aerynza", False),
        "family_members": ("Family", "family", True),
        "home_members": ("Home", "home", True),
        "dating_profiles": ("Aerynza Dating", "dating", True),
        "actual_relationships": ("Actual Relationships", "actual", True),
    }
    for key, (label, tab, is_real) in source_map.items():
        items = context.get(key, []) if isinstance(context.get(key), (list, tuple)) else []
        if not items:
            continue
        tabs.append({"id": tab, "label": label, "count": len(items)})
        for item in items[:20]:
            if isinstance(item, str):
                row = {"title": item, "summary": "", "real_relationship": is_real, "source": tab}
            elif isinstance(item, dict):
                row = {
                    "title": item.get("name") or item.get("title") or str(item)[:80],
                    "summary": item.get("summary") or item.get("note") or "",
                    "real_relationship": is_real,
                    "source": tab,
                }
            else:
                continue
            feed.append(row)
            if is_real:
                full_profiles.append({**row, "domain": tab})
            elif tab in {"social", "aerynza"}:
                opportunities.append(row)
    return {
        "feed": feed[:30],
        "tabs": tabs,
        "opportunities": opportunities[:6],
        "full_profiles": full_profiles[:12],
        "rule": "Social acquaintances are not real friendships. Full profiles are for family, home, dating, and explicitly marked actual relationships.",
    }


@app.post("/v1/relationships/feed")
async def relationships_feed(request: RelationshipFeedRequest, authorization: str | None = Header(default=None), _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    context = resolve_session_context(request, authorization)
    scoped_context = scope_context(context, request.shell)
    packet = build_cognitive_packet("mixed social feed", scoped_context, [])
    feed = _build_relationships_feed(scoped_context, request.shell)
    return {
        "shell": request.shell.value,
        "tier": request.tier.value,
        "surface": request.surface,
        **feed,
        "data_panels": packet.get("data_panels", []),
        "surface_recommendations": packet.get("surface_recommendations", []),
        "outside_provider": False,
    }


@app.post("/v1/intelligence")
async def intelligence(request: IntelligenceRequest, authorization: str | None = Header(default=None), _: None = Depends(require_access)) -> dict:
    enforce_shell_access(request.shell)
    context = resolve_session_context(request, authorization)
    emergency = medical_emergency_response(request.messages[-1].content)
    if emergency:
        return {
            "content": emergency,
            "model": "Aerynza Emergency Guard",
            "provider": "aerynza-deterministic-safety",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
            "queue_wait_ms": 0,
            "inference_ms": 0,
            "shell": request.shell.value,
            "tier": request.tier.value,
            "mode": request.mode,
            "surface": request.surface,
            "domains": ["safety", "health"],
            "capabilities": {},
            "outside_provider": False,
            "production_replacement_enabled": production_replacement_enabled(),
            "safety_intercept": "medical_emergency",
        }

    # Natural image-generation requests go straight to the configured media provider.
    image_prompt = parse_image_request(request.messages[-1].content)
    if image_prompt:
        started = time.perf_counter()
        result = await asyncio.to_thread(generate_image, image_prompt, context)
        latency_ms = round((time.perf_counter() - started) * 1000)
        if result["status"] == "image":
            return {
                "content": result["message"],
                "imageUrl": result["url"],
                "model": "dall-e-3",
                "provider": "openai",
                "outside_provider": True,
                "production_replacement_enabled": production_replacement_enabled(),
                "shell": request.shell.value,
                "tier": request.tier.value,
                "mode": request.mode,
                "surface": request.surface,
                "latency_ms": latency_ms,
            }
        return {
            "content": result["message"],
            "model": "dall-e-3",
            "provider": "openai" if result["status"] != "no_key" else "Aerynza-Native",
            "outside_provider": result["status"] != "no_key",
            "production_replacement_enabled": production_replacement_enabled(),
            "shell": request.shell.value,
            "tier": request.tier.value,
            "mode": request.mode,
            "surface": request.surface,
            "latency_ms": latency_ms,
        }

    # Balance / fund queries go to Plaid if configured.
    if parse_balance_query(request.messages[-1].content):
        started = time.perf_counter()
        result = await asyncio.to_thread(get_balances, context)
        latency_ms = round((time.perf_counter() - started) * 1000)
        return {
            "content": result["message"],
            "model": "plaid",
            "provider": "plaid" if result["status"] != "no_key" else "Aerynza-Native",
            "outside_provider": result["status"] != "no_key",
            "production_replacement_enabled": production_replacement_enabled(),
            "shell": request.shell.value,
            "tier": request.tier.value,
            "mode": request.mode,
            "surface": request.surface,
            "latency_ms": latency_ms,
        }

    require_native_ready()
    try:
        return await asyncio.to_thread(
            respond,
            shell=request.shell,
            tier=request.tier,
            messages=[message.model_dump() for message in request.messages],
            context=context,
            surface=request.surface,
            mode=request.mode,
            allowed_capabilities=request.allowed_capabilities,
            available_actions=request.available_actions,
            temperature=request.temperature,
            max_tokens=effective_max_tokens(request.max_tokens, request.mode),
        )
    except NativeInferenceQueueTimeout as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/chat")
async def chat(request: IntelligenceRequest, access: None = Depends(require_access)) -> dict:
    return await intelligence(request, access)


@app.post("/v1/stream")
async def stream_intelligence(request: IntelligenceRequest, authorization: str | None = Header(default=None), _: None = Depends(require_access)):
    enforce_shell_access(request.shell)
    context = resolve_session_context(request, authorization)
    emergency = medical_emergency_response(request.messages[-1].content)
    if emergency:
        def emergency_events():
            meta = {
                "model": "Aerynza Emergency Guard",
                "provider": "aerynza-deterministic-safety",
                "outside_provider": False,
                "safety_intercept": "medical_emergency",
            }
            yield f"event: meta\ndata: {json.dumps(meta, separators=(',', ':'))}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': emergency}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps({'latency_ms': 0, 'production_replacement_enabled': production_replacement_enabled()}, separators=(',', ':'))}\n\n"
        return StreamingResponse(emergency_events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})

    # Natural image-generation requests are sent to the configured media provider.
    image_prompt = parse_image_request(request.messages[-1].content)
    if image_prompt:
        started = time.perf_counter()
        result = await asyncio.to_thread(generate_image, image_prompt, context)
        latency_ms = round((time.perf_counter() - started) * 1000)

        def media_events():
            meta = {
                "shell": request.shell.value,
                "tier": request.tier.value,
                "model": "dall-e-3",
                "provider": "openai",
                "outside_provider": result["status"] == "image" or result["status"] != "no_key",
            }
            yield f"event: meta\ndata: {json.dumps(meta, separators=(',', ':'))}\n\n"
            if result["status"] == "image":
                media_data = {"url": result["url"], "message": result["message"]}
                yield f"event: media\ndata: {json.dumps(media_data, ensure_ascii=False)}\n\n"
            else:
                yield f"event: token\ndata: {json.dumps({'token': result['message']}, ensure_ascii=False)}\n\n"
            done = {
                "latency_ms": latency_ms,
                "production_replacement_enabled": production_replacement_enabled(),
                "outside_provider": result["status"] == "image" or result["status"] != "no_key",
            }
            yield f"event: done\ndata: {json.dumps(done, separators=(',', ':'))}\n\n"

        return StreamingResponse(
            media_events(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    # Balance / fund queries go to Plaid if configured.
    if parse_balance_query(request.messages[-1].content):
        started = time.perf_counter()
        result = await asyncio.to_thread(get_balances, context)
        latency_ms = round((time.perf_counter() - started) * 1000)

        def balance_events():
            meta = {
                "shell": request.shell.value,
                "tier": request.tier.value,
                "model": "plaid",
                "provider": "plaid" if result["status"] != "no_key" else "Aerynza-Native",
                "outside_provider": result["status"] != "no_key",
            }
            yield f"event: meta\ndata: {json.dumps(meta, separators=(',', ':'))}\n\n"
            yield f"event: token\ndata: {json.dumps({'token': result['message']}, ensure_ascii=False)}\n\n"
            done = {
                "latency_ms": latency_ms,
                "production_replacement_enabled": production_replacement_enabled(),
                "outside_provider": result["status"] != "no_key",
            }
            yield f"event: done\ndata: {json.dumps(done, separators=(',', ':'))}\n\n"

        return StreamingResponse(
            balance_events(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    require_native_ready()
    prepared = prepare_inference(
        shell=request.shell,
        tier=request.tier,
        messages=[message.model_dump() for message in request.messages],
        context=context,
        surface=request.surface,
        mode=request.mode,
        allowed_capabilities=request.allowed_capabilities,
        available_actions=request.available_actions,
    )
    latest = request.messages[-1].content
    first_pass = deterministic_response(request.shell, latest, request.mode, prepared["cognition"])

    def events():
        started = time.perf_counter()
        cognition_meta = prepared.get("cognition", {})
        meta = {
            key: value for key, value in prepared.items()
            if key not in {"messages", "cognition"}
        }
        meta["cognition"] = {
            "domains": cognition_meta.get("domains", []),
            "talents": [item.get("key") for item in cognition_meta.get("talents", [])],
            "memory_candidates": cognition_meta.get("memory_candidates", []),
            "action_proposals": cognition_meta.get("action_proposals", []),
            "surface_recommendations": cognition_meta.get("surface_recommendations", []),
            "retrieval": (cognition_meta.get("retrieval", []) or [])[:3],
            "data_panels": cognition_meta.get("data_panels", [])[:12],
        }
        meta.update({"model": runtime.status()["model"], "provider": "Aerynza-Native", "outside_provider": False})
        yield f"event: meta\ndata: {json.dumps(meta, separators=(',', ':'))}\n\n"
        try:
            if first_pass:
                yield f"event: token\ndata: {json.dumps({'token': first_pass}, ensure_ascii=False)}\n\n"
                done = {
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                    "production_replacement_enabled": production_replacement_enabled(),
                    "contract_engine": True,
                }
                yield f"event: done\ndata: {json.dumps(done, separators=(',', ':'))}\n\n"
                return
            for token in runtime.stream_chat(prepared["messages"], request.temperature, effective_max_tokens(request.max_tokens, request.mode)):
                yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            done = {"latency_ms": round((time.perf_counter() - started) * 1000), "production_replacement_enabled": production_replacement_enabled()}
            yield f"event: done\ndata: {json.dumps(done, separators=(',', ':'))}\n\n"
        except Exception as error:
            yield f"event: error\ndata: {json.dumps(stream_error_payload(error), separators=(',', ':'))}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})


@app.post("/generate")
async def generate(request: LegacyGenerationRequest, access: None = Depends(require_access)) -> dict:
    if _auth_mode() == "production":
        raise HTTPException(status_code=403, detail="Legacy generation route disabled pending explicit shell identity")
    result = await intelligence(
        IntelligenceRequest(
            shell=Shell.CORE,
            messages=[ChatMessage(role="user", content=request.prompt)],
            surface="legacy_generate",
            mode="analysis",
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
        ),
        access,
    )
    return {
        "content": result["content"],
        "model": result["model"],
        "provider": result["provider"],
        "tokens_generated": result.get("usage", {}).get("completion_tokens", 0),
        "generation_time_ms": result["latency_ms"],
        "mode": "ascension_native_local",
    }


class WindowsActionRequest(BaseModel):
    action: str = Field(min_length=1, max_length=40)
    params: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)


@app.post("/v1/windows/execute")
async def windows_execute(request: WindowsActionRequest, access: None = Depends(require_access)) -> dict:
    """Execute one allowed local Windows desktop action.

    Only the actions listed in `src/windows/executor.py` are allowed.
    The request is authenticated and the shell remains responsible for
    deciding when an action is appropriate.
    """
    require_certified_action_gateway()
    return executor.run(request.action, **request.params)


class AndroidActionRequest(BaseModel):
    action: str = Field(min_length=1, max_length=40)
    params: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)


@app.post("/v1/android/execute")
async def android_execute(request: AndroidActionRequest, access: None = Depends(require_access)) -> dict:
    """Execute one allowed Android action via a connected adb device.

    The phone must have USB debugging enabled and be authorized.
    The bridge uses the local `tools/adb/platform-tools/adb.exe` binary.
    """
    require_certified_action_gateway()
    return android_bridge.run(request.action, **request.params)


class iPhoneActionRequest(BaseModel):
    action: str = Field(min_length=1, max_length=40)
    params: dict = Field(default_factory=dict)
    context: dict = Field(default_factory=dict)


@app.post("/v1/iphone/execute")
async def iphone_execute(request: iPhoneActionRequest, access: None = Depends(require_access)) -> dict:
    """Execute one allowed iPhone action through a user-configured iOS Shortcut webhook."""
    require_certified_action_gateway()
    return iphone_bridge.run(request.action, context=request.context, **request.params)


class MediaRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    provider: str = Field(default="dall-e-3")
    context: dict = Field(default_factory=dict)


@app.post("/v1/media/generate")
async def media_generate(request: MediaRequest, _: None = Depends(require_access)) -> dict:
    """Generate media through the configured outside provider. DALL-E 3 is the first supported provider."""
    if os.getenv("AERYNZA_AI_ENABLE_PROVIDER_ROUTES", "").strip().lower() != "true":
        raise HTTPException(status_code=403, detail="Direct provider execution is disabled")
    return await asyncio.to_thread(generate_image, request.prompt, request.context)


@app.post("/v1/iphone/inbox")
async def iphone_inbox(payload: dict, access: None = Depends(require_access)) -> dict:
    """Receive a JSON payload from an iOS Shortcut.

    The iPhone can POST here to send battery, location, or any other
    data it is allowed to share. Data is stored locally in data/iphone_inbox.json.
    """
    require_certified_action_gateway()
    return iphone_bridge.receive(payload)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=1, max_length=120)


@app.post("/v1/login")
async def login(request: LoginRequest) -> dict:
    """Local desktop login is unavailable in production service mode."""
    if _auth_mode() != "development":
        raise HTTPException(status_code=404, detail="Not found")
    try:
        import keyring
    except ImportError as error:
        raise HTTPException(status_code=503, detail="Desktop credential storage is unavailable") from error
    stored = keyring.get_password(KEYRING_SERVICE, request.email)
    if not stored or not hmac.compare_digest(stored, request.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    session_token = secrets.token_urlsafe(32)
    SESSIONS.add(session_token)
    return {"status": "authenticated", "session_token": session_token}
