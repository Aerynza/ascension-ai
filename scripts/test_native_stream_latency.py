"""Model-free first-token and native queue instrumentation checks."""
import unittest
from unittest.mock import MagicMock
from src.core.model_runtime import NativeModelRuntime


class TestStreamLatencyMetrics(unittest.TestCase):
    def test_first_token_and_completed_stream_are_counted_without_content(self):
        native = NativeModelRuntime(exit_callback=lambda: None)
        native.model = MagicMock()
        native.model.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        output = "".join(native.stream_chat([{"role": "user", "content": "Hi"}], 0.7, 32))
        self.assertEqual(output, "Hello world")
        queue = native.status()["queue"]
        self.assertEqual(queue["completed_stream_requests"], 1)
        self.assertEqual(queue["queue_depth"], 0)
        self.assertEqual(queue["active_requests"], 0)
        self.assertIsNone(queue["inference_started_at"])
        self.assertGreaterEqual(queue["last_stream_first_token_ms"], 0)
        self.assertGreaterEqual(queue["last_stream_inference_ms"], 0)
        self.assertNotIn("Hello", str(queue))
        self.assertNotIn("Hi", str(queue))

    def test_closed_generator_releases_lock_without_success_receipt(self):
        native = NativeModelRuntime(exit_callback=lambda: None)
        native.model = MagicMock()
        native.model.create_chat_completion.return_value = [
            {"choices": [{"delta": {"content": "Partial"}}]},
            {"choices": [{"delta": {"content": " final"}}]},
        ]
        stream = native.stream_chat([{"role": "user", "content": "Hi"}], 0.7, 32)
        self.assertEqual(next(stream), "Partial")
        stream.close()
        queue = native.status()["queue"]
        self.assertEqual(queue["completed_stream_requests"], 0)
        self.assertEqual(queue["queue_depth"], 0)
        self.assertEqual(queue["active_requests"], 0)
        self.assertIsNone(queue["inference_started_at"])
        self.assertTrue(native.lock.acquire(blocking=False))
        native.lock.release()


if __name__ == "__main__":
    unittest.main()
