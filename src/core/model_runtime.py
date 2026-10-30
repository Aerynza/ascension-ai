"""Thread-safe local GGUF inference runtime for Aerynza AI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = ROOT / "config" / "model_profiles.json"
PROFILE_ALIASES_PATH = ROOT / "config" / "profile_aliases.json"


def resolve_profile_name(requested: str) -> str:
    """Resolve source-controlled deployment aliases without mutating profiles."""
    normalized = str(requested or "").strip().lower()
    if not PROFILE_ALIASES_PATH.is_file():
        return normalized
    aliases = json.loads(PROFILE_ALIASES_PATH.read_text(encoding="utf-8"))
    return str(aliases.get(normalized, normalized)).strip().lower()


class NativeInferenceQueueTimeout(RuntimeError):
    """The serialized native runtime could not start within its queue budget."""


class ModelRamBudgetExceeded(RuntimeError):
    """The pinned GGUF is larger than the configured host RAM budget."""


def configured_ram_budget_bytes() -> int | None:
    """Return the GGUF byte budget, or None when no budget is configured.

    AERYNZA_AI_RAM_BUDGET_BYTES is the primary name. ASCENSION_AI_RAM_BUDGET_BYTES
    is accepted as an alias. An invalid value fails closed.
    """
    raw = os.getenv("AERYNZA_AI_RAM_BUDGET_BYTES", os.getenv("ASCENSION_AI_RAM_BUDGET_BYTES", "")).strip()
    if not raw:
        return None
    try:
        budget = int(raw)
    except ValueError as exc:
        raise ModelRamBudgetExceeded("RAM budget is not an integer byte count") from exc
    if budget <= 0:
        raise ModelRamBudgetExceeded("RAM budget must be a positive byte count")
    return budget


def assert_gguf_within_ram_budget(profile: dict, file_size: int | None = None, budget: int | None = None) -> None:
    """Refuse to boot when the GGUF exceeds the RAM budget."""
    if budget is None:
        budget = configured_ram_budget_bytes()
    if budget is None:
        return
    declared = int(profile.get("size_bytes") or 0)
    observed = declared if file_size is None else max(declared, int(file_size))
    if observed > budget:
        raise ModelRamBudgetExceeded(f"GGUF size {observed} bytes exceeds RAM budget {budget} bytes")


class NativeModelRuntime:
    def __init__(self, exit_callback=None) -> None:
        self.model = None
        self.requested_profile_name = (
            os.getenv("ASCENSION_MODEL_PROFILE")
            or os.getenv("AERYNZA_MODEL_PROFILE")
            or "starter"
        ).strip().lower()
        self.profile_name = resolve_profile_name(self.requested_profile_name)
        self.profile: dict = {}
        self.loaded_at: float | None = None
        self.load_error: str | None = None
        self.lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.queue_depth = 0
        self.active_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.timeout_requests = 0
        self.last_queue_wait_ms = 0
        self.max_queue_wait_ms = 0
        self.last_inference_ms = 0
        self.last_stream_first_token_ms = 0
        self.last_stream_inference_ms = 0
        self.completed_stream_requests = 0
        self.queue_timeout_seconds = float(os.getenv("ASCENSION_QUEUE_TIMEOUT_SECONDS", "95"))
        self.inference_timeout_seconds = float(os.getenv("ASCENSION_INFERENCE_TIMEOUT_SECONDS", "180"))
        self.inference_started_at: float | None = None
        self.watchdog_enabled = os.getenv("ASCENSION_INFERENCE_WATCHDOG_EXIT", "0") == "1"
        self._exit_callback = exit_callback or (self._default_exit if self.watchdog_enabled else lambda: None)
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_running = False

    def _default_exit(self) -> None:
        """Last-resort process exit when the only worker is stuck.

        Render or the container runtime will restart the service.
        No prompt, context, or token data is ever written here.
        """
        os._exit(1)

    def _read_profile(self) -> dict:
        profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        if self.profile_name not in profiles:
            raise RuntimeError(f"Unknown model profile: {self.profile_name}")
        return profiles[self.profile_name]

    def _model_path(self) -> Path:
        override = os.getenv("ASCENSION_MODEL_PATH", "").strip()
        return Path(override) if override else ROOT / "models" / self.profile["filename"]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _lora_path(self) -> Path | None:
        lora = self.profile.get("lora_path", os.getenv("ASCENSION_MODEL_LORA_PATH", "")).strip()
        if not lora:
            return None
        if os.path.isabs(lora):
            return Path(lora)
        return ROOT / "models" / lora

    def load(self) -> None:
        try:
            self.profile = self._read_profile()
            assert_gguf_within_ram_budget(self.profile)
            path = self._model_path()
            if not path.is_file():
                raise RuntimeError("Pinned model file is missing. Run scripts/download_model.py during the build.")
            assert_gguf_within_ram_budget(self.profile, path.stat().st_size)
            if os.getenv("ASCENSION_VERIFY_MODEL_ON_BOOT", "1") != "0" and self._sha256(path) != self.profile["sha256"]:
                raise RuntimeError("Pinned model checksum verification failed.")
            from llama_cpp import Llama
            lora_path = self._lora_path()
            model_kwargs = {
                "model_path": str(path),
                "n_ctx": int(os.getenv("ASCENSION_MODEL_CONTEXT", os.getenv("AERYNZA_MODEL_CONTEXT", "768"))),
                "n_threads": int(os.getenv("ASCENSION_MODEL_THREADS", self.profile["threads"])),
                "n_threads_batch": int(os.getenv("ASCENSION_MODEL_THREADS_BATCH", self.profile["threads"])),
                "n_batch": int(os.getenv("ASCENSION_MODEL_BATCH", os.getenv("AERYNZA_MODEL_BATCH", "16"))),
                "use_mmap": True,
                "use_mlock": False,
                "verbose": False,
            }
            if lora_path and lora_path.is_file():
                model_kwargs["lora_path"] = str(lora_path)
                model_kwargs["lora_scale"] = float(os.getenv("ASCENSION_MODEL_LORA_SCALE", self.profile.get("lora_scale", "1.0")))
            self.model = Llama(**model_kwargs)
            self.loaded_at = time.time()
            self.load_error = None
            if not self._watchdog_running:
                self._watchdog_running = True
                self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
                self._watchdog_thread.start()
        except Exception as error:
            self.model = None
            self.load_error = str(error)
            raise

    def status(self) -> dict:
        with self.metrics_lock:
            queue = {
                "queue_depth": self.queue_depth,
                "active_requests": self.active_requests,
                "completed_requests": self.completed_requests,
                "failed_requests": self.failed_requests,
                "timeout_requests": self.timeout_requests,
                "last_queue_wait_ms": self.last_queue_wait_ms,
                "max_queue_wait_ms": self.max_queue_wait_ms,
                "last_inference_ms": self.last_inference_ms,
                "last_stream_first_token_ms": self.last_stream_first_token_ms,
                "last_stream_inference_ms": self.last_stream_inference_ms,
                "completed_stream_requests": self.completed_stream_requests,
                "queue_timeout_seconds": self.queue_timeout_seconds,
                "inference_timeout_seconds": self.inference_timeout_seconds,
                "inference_started_at": self.inference_started_at,
                "inference_hung": (
                    self.inference_started_at is not None
                    and (time.perf_counter() - self.inference_started_at) > self.inference_timeout_seconds
                ),
                "watchdog_enabled": self.watchdog_enabled,
            }
        return {
            "ready": self.model is not None,
            "profile": self.profile_name,
            "requested_profile": self.requested_profile_name,
            "model": self.profile.get("label"),
            "repo_id": self.profile.get("repo_id"),
            "filename": self.profile.get("filename"),
            "context_tokens": self.profile.get("context_tokens"),
            "threads": self.profile.get("threads"),
            "minimum_render_plan": self.profile.get("minimum_render_plan"),
            "loaded_at": self.loaded_at,
            "error": self.load_error,
            "inference": "local_gguf_llama_cpp",
            "outside_provider": False,
            "queue": queue,
        }

    def chat(self, messages: list[dict], temperature: float, max_tokens: int,
             mode: str = "conversation") -> dict:
        if self.model is None:
            raise RuntimeError(self.load_error or "Native model is not loaded.")
        started = time.perf_counter()
        queued_at = started
        inference_messages = self._prepare_messages(messages)
        unattended = mode in {"background", "proactive"}
        # An unattended model turn cannot queue in front of a foreground AP
        # response. These counters are advisory; the lock acquisition below
        # is the actual atomic admission boundary.
        with self.metrics_lock:
            if unattended and (self.queue_depth > 0 or self.active_requests > 0):
                raise NativeInferenceQueueTimeout("Native background yielded to foreground")
            self.queue_depth += 1
        dequeued = False
        try:
            acquired = (
                self.lock.acquire(blocking=False)
                if unattended
                else self.lock.acquire(timeout=self.queue_timeout_seconds)
            )
            if not acquired:
                raise NativeInferenceQueueTimeout(
                    "Native background yielded to foreground"
                    if unattended else "Native inference queue timeout"
                )
            dequeued = True
            inference_started = time.perf_counter()
            queue_wait_ms = round((inference_started - queued_at) * 1000)
            with self.metrics_lock:
                self.queue_depth = max(0, self.queue_depth - 1)
                self.active_requests += 1
                self.last_queue_wait_ms = queue_wait_ms
                self.max_queue_wait_ms = max(self.max_queue_wait_ms, queue_wait_ms)
                self.inference_started_at = inference_started
            try:
                result = self.model.create_chat_completion(
                    messages=inference_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.9,
                    repeat_penalty=1.08,
                    stop=["<|im_end|>", "<|endoftext|>"],
                )
                content = self._clean_content(
                    str(result.get("choices", [{}])[0].get("message", {}).get("content", ""))
                )
                usage = result.get("usage") or {}
                if not content:
                    # Never run raw fallback outside the same per-model
                    # inference lock: llama.cpp is shared between accounts.
                    last_user = ""
                    for message in reversed(inference_messages):
                        if message.get("role") == "user":
                            last_user = str(message.get("content", "")).strip()
                            break
                    prompt = f"User: {last_user}\nAssistant:"
                    raw_result = self.model.create_completion(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=0.9,
                        repeat_penalty=1.08,
                        stop=["\n", "User:", "Assistant:", "<|endoftext|>"],
                    )
                    content = self._clean_content(
                        str(raw_result.get("choices", [{}])[0].get("text", ""))
                    )
                    usage = raw_result.get("usage") or usage
                if not content:
                    raise RuntimeError("Native model returned an empty response.")
                inference_ms = round((time.perf_counter() - inference_started) * 1000)
                with self.metrics_lock:
                    self.completed_requests += 1
                    self.last_inference_ms = inference_ms
            finally:
                self.lock.release()
                with self.metrics_lock:
                    self.active_requests = max(0, self.active_requests - 1)
                    self.inference_started_at = None
        except Exception:
            with self.metrics_lock:
                if not dequeued:
                    self.queue_depth = max(0, self.queue_depth - 1)
                    self.timeout_requests += 1
                self.failed_requests += 1
            raise
        return {
            "content": content,
            "model": self.profile.get("label") or self.profile_name,
            "provider": "ascension-native",
            "usage": usage,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "queue_wait_ms": queue_wait_ms,
            "inference_ms": inference_ms,
        }

    def stream_chat(self, messages: list[dict], temperature: float, max_tokens: int):
        if self.model is None:
            raise RuntimeError(self.load_error or "Native model is not loaded.")
        queued_at = time.perf_counter()
        inference_messages = self._prepare_messages(messages)
        with self.metrics_lock:
            self.queue_depth += 1
        acquired = False
        completed = False
        inference_started = None
        try:
            acquired = self.lock.acquire(timeout=self.queue_timeout_seconds)
            if not acquired:
                raise NativeInferenceQueueTimeout("Native inference queue timeout")
            inference_started = time.perf_counter()
            with self.metrics_lock:
                self.queue_depth = max(0, self.queue_depth - 1)
                self.last_queue_wait_ms = round((inference_started - queued_at) * 1000)
                self.max_queue_wait_ms = max(self.max_queue_wait_ms, self.last_queue_wait_ms)
                self.active_requests += 1
                self.inference_started_at = inference_started
            chunks = self.model.create_chat_completion(
                messages=inference_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                repeat_penalty=1.08,
                stop=["<|im_end|>", "<|endoftext|>"],
                stream=True,
            )
            received_first_token = False
            for chunk in chunks:
                token = str(chunk.get("choices", [{}])[0].get("delta", {}).get("content", ""))
                if token:
                    if not received_first_token:
                        with self.metrics_lock:
                            self.last_stream_first_token_ms = round((time.perf_counter() - queued_at) * 1000)
                        received_first_token = True
                    yield token
            completed = True
        except Exception:
            with self.metrics_lock:
                self.failed_requests += 1
                if not acquired:
                    self.timeout_requests += 1
            raise
        finally:
            if acquired:
                self.lock.release()
            with self.metrics_lock:
                if not acquired:
                    self.queue_depth = max(0, self.queue_depth - 1)
                else:
                    self.active_requests = max(0, self.active_requests - 1)
                    self.inference_started_at = None
                    if inference_started is not None:
                        self.last_stream_inference_ms = round((time.perf_counter() - inference_started) * 1000)
                    if completed:
                        self.completed_stream_requests += 1

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Keep Qwen3 in fast non-thinking mode without exposing control tokens in UI."""
        prepared = [dict(message) for message in messages]
        if "Qwen3" not in str(self.profile.get("repo_id", "")):
            return prepared
        for index in range(len(prepared) - 1, -1, -1):
            if prepared[index].get("role") == "user":
                content = str(prepared[index].get("content", ""))
                if "/no_think" not in content:
                    prepared[index]["content"] = f"{content}\n/no_think"
                break
        return prepared

    @staticmethod
    def _clean_content(content: str) -> str:
        """Never surface hidden reasoning or model control tags to users."""
        cleaned = content.strip()
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[1].strip()
        if cleaned.startswith("<think>"):
            return ""
        cleaned = re.sub(r"<\|[^>]+\|>", "", cleaned).strip()
        cleaned = re.sub(r"^\s*(?:analysis|reasoning)\s*:\s*", "", cleaned, flags=re.I)
        return cleaned


    def _watchdog_tick(self) -> None:
        """Check for a hung active inference and trigger the configured exit.

        The callback is mocked in tests; in production it exits the process.
        No prompt, token, or context data is logged or exposed.
        """
        with self.metrics_lock:
            should_exit = (
                self.watchdog_enabled
                and self.inference_started_at is not None
                and (time.perf_counter() - self.inference_started_at) > self.inference_timeout_seconds
            )
        if should_exit:
            self._exit_callback()

    def _watchdog_loop(self) -> None:
        while self._watchdog_running:
            time.sleep(1.0)
            self._watchdog_tick()


runtime = NativeModelRuntime()
