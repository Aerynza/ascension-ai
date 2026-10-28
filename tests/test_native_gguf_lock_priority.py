"""Native local GGUF serialization and foreground-priority regressions.

No GPU, model download, network, accounts or real secrets are required.
"""
import threading
import time

import pytest

from src.core.model_runtime import NativeInferenceQueueTimeout, NativeModelRuntime


MESSAGES = [{"role": "user", "content": "Please privately answer."}]


def local_runtime(model):
    runtime = NativeModelRuntime()
    runtime.model = model
    runtime.profile = {"label": "mocked-local-gguf"}
    return runtime


def test_raw_fallback_keeps_shared_gguf_lock_until_final_token():
    entered_fallback = threading.Event()
    allow_fallback_to_finish = threading.Event()

    class Model:
        def __init__(self):
            self.chat_calls = 0
            self.raw_calls = 0

        def create_chat_completion(self, **_kwargs):
            self.chat_calls += 1
            return {"choices": [{"message": {"content": ""}}], "usage": {}}

        def create_completion(self, **_kwargs):
            self.raw_calls += 1
            entered_fallback.set()
            if not allow_fallback_to_finish.wait(timeout=3):
                raise RuntimeError("mock fallback did not release")
            return {
                "choices": [{"text": "Recovered privately"}],
                "usage": {"completion_tokens": 4},
            }

    model = Model()
    runtime = local_runtime(model)
    output = []
    errors = []

    def foreground_call():
        try:
            output.append(runtime.chat(MESSAGES, temperature=0.2,
                                       max_tokens=32, mode="conversation"))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=foreground_call, daemon=True)
    thread.start()
    try:
        assert entered_fallback.wait(timeout=2), "Fallback never started"
        assert not runtime.lock.acquire(blocking=False), (
            "A second account could enter llama.cpp while raw fallback runs"
        )
        # Background cannot wait behind or interrupt the in-flight AP turn.
        with pytest.raises(NativeInferenceQueueTimeout):
            runtime.chat(MESSAGES, temperature=0.2,
                         max_tokens=32, mode="background")
        assert model.chat_calls == 1
        assert model.raw_calls == 1
        assert runtime.active_requests == 1
    finally:
        allow_fallback_to_finish.set()
        thread.join(timeout=3)
    assert not thread.is_alive(), "Model request must finish and release the lock"
    assert not errors
    assert output[0]["content"] == "Recovered privately"
    assert output[0]["usage"]["completion_tokens"] == 4
    assert runtime.active_requests == 0
    assert runtime.queue_depth == 0
    assert runtime.completed_requests == 1
    assert runtime.lock.acquire(blocking=False)
    runtime.lock.release()


@pytest.mark.parametrize("mode", ["background", "proactive"])
def test_unattended_generation_never_waits_in_foreground_queue(mode):
    class Model:
        def create_chat_completion(self, **_kwargs):
            raise AssertionError("Busy native model must not receive unattended work")
    runtime = local_runtime(Model())
    assert runtime.lock.acquire(blocking=False)
    try:
        started = time.perf_counter()
        with pytest.raises(NativeInferenceQueueTimeout):
            runtime.chat(MESSAGES, temperature=0.1, max_tokens=64, mode=mode)
        assert time.perf_counter() - started < 0.25
        assert runtime.queue_depth == 0
        assert runtime.active_requests == 0
    finally:
        runtime.lock.release()


def test_queued_interactive_turn_prevents_background_admission():
    class Model:
        def create_chat_completion(self, **_kwargs):
            raise AssertionError("Queued foreground must be served before background")
    runtime = local_runtime(Model())
    with runtime.metrics_lock:
        runtime.queue_depth = 1
    with pytest.raises(NativeInferenceQueueTimeout):
        runtime.chat(MESSAGES, temperature=0.1, max_tokens=64, mode="proactive")
    assert runtime.queue_depth == 1
    assert runtime.active_requests == 0


def test_failed_raw_fallback_releases_lock_and_counts_generation_failure():
    class Model:
        def create_chat_completion(self, **_kwargs):
            return {"choices": [{"message": {"content": ""}}]}

        def create_completion(self, **_kwargs):
            raise RuntimeError("mock fallback failed")
    runtime = local_runtime(Model())
    with pytest.raises(RuntimeError, match="mock fallback failed"):
        runtime.chat(MESSAGES, temperature=0.1, max_tokens=64)
    assert runtime.completed_requests == 0
    assert runtime.failed_requests == 1
    assert runtime.active_requests == 0
    assert runtime.queue_depth == 0
    assert runtime.lock.acquire(blocking=False)
    runtime.lock.release()


def test_foreground_generation_remains_available_after_background_yield():
    class Model:
        def create_chat_completion(self, **_kwargs):
            return {
                "choices": [{"message": {"content": "Direct answer"}}],
                "usage": {"completion_tokens": 3},
            }
    runtime = local_runtime(Model())
    runtime.lock.acquire()
    try:
        with pytest.raises(NativeInferenceQueueTimeout):
            runtime.chat(MESSAGES, temperature=0.1,
                         max_tokens=64, mode="background")
    finally:
        runtime.lock.release()
    result = runtime.chat(MESSAGES, temperature=0.1,
                          max_tokens=64, mode="conversation")
    assert result["content"] == "Direct answer"
    assert runtime.completed_requests == 1
    assert runtime.active_requests == 0
