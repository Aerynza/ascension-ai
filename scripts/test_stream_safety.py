"""Focused safety tests for incremental Qwen GGUF streaming."""
import unittest
from src.core.stream_safety import safe_model_chunks


class TestSafeNativeStream(unittest.TestCase):
    def test_normal_tokens_are_visible_incrementally(self):
        source = (chunk for chunk in ["Hello", " ", "AP", "!"])
        result = safe_model_chunks(source)
        self.assertEqual(next(result), "Hello")
        self.assertEqual("".join(result), " AP!")

    def test_hidden_reasoning_never_appears_even_with_split_tags(self):
        tokens = ["<thi", "nk>private thought", "</thi", "nk>Hello", " there"]
        self.assertEqual("".join(safe_model_chunks(tokens)), "Hello there")

    def test_non_terminated_thought_is_not_emitted(self):
        self.assertEqual("".join(safe_model_chunks(["<think>", "hidden only"])), "")

    def test_interspersed_control_tokens_filtered(self):
        chunks = ["A<|im_", "end|>B<|endoftext|>", "C"]
        self.assertEqual("".join(safe_model_chunks(chunks)), "ABC")

    def test_literal_angle_brackets_kept(self):
        chunks = ["Use x < 5 and y > 2"]
        self.assertEqual("".join(safe_model_chunks(chunks)), "Use x < 5 and y > 2")

    def test_split_noncontrol_text(self):
        self.assertEqual("".join(safe_model_chunks(["<ap", "ple>"])), "<apple>")


if __name__ == "__main__":
    unittest.main()
