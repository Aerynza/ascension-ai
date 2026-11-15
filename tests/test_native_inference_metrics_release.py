"""Regression: local GGUF admission metrics must be retired before lock release.

No model download, GPU, secret, external account, or network needed.
"""
import threading
import unittest

from src.core.model_runtime import NativeModelRuntime


class ProbeLock:
    """Inspect the public admission state exactly when the model lock releases."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._lock = threading.Lock()
        self.at_release = []

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking=blocking, timeout=timeout)

    def release(self):
        with self.runtime.metrics_lock:
            self.at_release.append((
                self.runtime.active_requests,
                self.runtime.inference_started_at,
                self.runtime.queue_depth,
            ))
        self._lock.release()


class GoodModel:
    def create_chat_completion(self, **kwargs):
        if kwargs.get("stream"):
            return iter([
                {"choices": [{"delta": {"content": "First"}}]},
                {"choices": [{"delta": {"content": " second"}}]},
            ])
        return {
            "choices": [{"message": {"content": "Ready"}}],
            "usage": {"completion_tokens": 1},
        }


def make_runtime(model=None):
    runtime = NativeModelRuntime(exit_callback=lambda: None)
    runtime.model = model if model is not None else GoodModel()
    runtime.profile = {"label": "isolated-test"}
    runtime.lock = ProbeLock(runtime)
    return runtime


class NativeMetricsReleaseContract(unittest.TestCase):
    def assert_released_idle(self, runtime, expected_count=1):
        self.assertEqual(len(runtime.lock.at_release), expected_count)
        for active, started, queued in runtime.lock.at_release:
            self.assertEqual(active, 0, "An earlier request released the lock while still marked active")
            self.assertIsNone(started, "An earlier request may erase the new owner's start time")
            self.assertEqual(queued, 0)
        queue = runtime.status()["queue"]
        self.assertEqual(queue["active_requests"], 0)
        self.assertIsNone(queue["inference_started_at"])
        self.assertEqual(queue["queue_depth"], 0)

    def test_regular_chat_clears_owner_metrics_before_release(self):
        runtime = make_runtime()
        result = runtime.chat([{"role": "user", "content": "Hi"}], 0.2, 32)
        self.assertEqual(result["content"], "Ready")
        self.assert_released_idle(runtime)
        self.assertEqual(runtime.completed_requests, 1)

    def test_failed_chat_also_retires_owner_before_release(self):
        class FailedModel:
            def create_chat_completion(self, **_kwargs):
                raise RuntimeError("mock model failed")
        runtime = make_runtime(FailedModel())
        with self.assertRaises(RuntimeError):
            runtime.chat([{"role": "user", "content": "Hi"}], 0.2, 32)
        self.assert_released_idle(runtime)
        self.assertEqual(runtime.failed_requests, 1)

    def test_full_stream_retires_owner_before_release(self):
        runtime = make_runtime()
        tokens = list(runtime.stream_chat([{"role": "user", "content": "Hi"}], 0.2, 32))
        self.assertEqual("".join(tokens), "First second")
        self.assert_released_idle(runtime)
        self.assertEqual(runtime.completed_stream_requests, 1)

    def test_aborted_stream_retires_owner_before_release_without_fake_success(self):
        runtime = make_runtime()
        stream = runtime.stream_chat([{"role": "user", "content": "Hi"}], 0.2, 32)
        self.assertEqual(next(stream), "First")
        stream.close()
        self.assert_released_idle(runtime)
        self.assertEqual(runtime.completed_stream_requests, 0)

    def test_back_to_back_chat_and_stream_stay_account_neutral(self):
        runtime = make_runtime()
        runtime.chat([{"role": "user", "content": "One"}], 0.2, 32)
        list(runtime.stream_chat([{"role": "user", "content": "Two"}], 0.2, 32))
        self.assert_released_idle(runtime, expected_count=2)
        self.assertEqual(runtime.completed_requests, 1)
        self.assertEqual(runtime.completed_stream_requests, 1)


if __name__ == "__main__":
    unittest.main()
