"""Tests: native SSE cannot bypass verified receipt and personal data claims."""
import unittest

from src.core.stream_contract import requires_full_validation, verified_output_chunks


class NativeStreamContractTest(unittest.TestCase):
    def verified(self, pieces, *, latest="hello", mode="conversation", cognition=None, context=None):
        return verified_output_chunks(
            pieces, cognitive=cognition or {}, context=context or {},
            mode=mode, latest=latest,
        )

    def test_casual_sentence_streams_before_response_finishes(self):
        consumed = []
        def source():
            consumed.append(1)
            yield "Hello"
            consumed.append(2)
            yield " there. "
            consumed.append(3)
            yield "How are you?"
        stream = self.verified(source())
        self.assertEqual(next(stream), "Hello there. ")
        self.assertEqual(consumed, [1, 2], "first completed sentence is visible before last generation")
        self.assertEqual("".join(stream), "How are you?")

    def test_unreceipted_completion_claim_never_streams_verbatim(self):
        text = "".join(self.verified(["I've saved your account. ", "Let me know."]))
        self.assertNotIn("I've saved your account", text)
        self.assertIn("Nothing is confirmed by a provider receipt yet", text)
        self.assertIn("Let me know.", text)

    def test_invented_balance_does_not_reach_user(self):
        text = "".join(self.verified(["Your balance is $900. ", "I cannot verify your bank here."]))
        self.assertNotIn("Your balance is $900", text)
        self.assertIn("I can't confirm those financial details", text)

    def test_sensitive_request_is_fully_validated_before_first_output(self):
        consumed = []
        def source():
            consumed.append("first")
            yield "I've saved it. "
            consumed.append("last")
            yield "All done."
        stream = self.verified(source(), latest="Remember this for next week")
        result = next(stream)
        self.assertEqual(consumed, ["first", "last"])
        self.assertNotIn("I've saved", result)
        self.assertIn("not been saved", result)

    def test_action_proposal_and_nonconversation_buffer_full_model_turn(self):
        for mode, latest, cognitive in [
            ("planning", "hello", {}),
            ("analysis", "hello", {}),
            ("conversation", "book a meeting", {}),
            ("conversation", "hi", {"action_proposals": [{"action": "finance.payment"}]}),
        ]:
            with self.subTest(mode=mode, latest=latest):
                self.assertTrue(requires_full_validation(mode, latest, cognitive))
                chunks = self.verified(["I can draft. ", "Nothing has been sent."],
                                       latest=latest, mode=mode, cognition=cognitive)
                self.assertEqual(len(list(chunks)), 1)

    def test_non_sensitive_stream_has_bounded_unfinished_sentence(self):
        with self.assertRaises(RuntimeError):
            list(self.verified(["A" * 32001]))

    def test_no_cross_tenant_receipts_are_assumed(self):
        self.assertFalse(requires_full_validation("conversation", "howdy", {}))
        output = "".join(self.verified(["I sent the payment. "], latest="hello", context={}))
        self.assertNotIn("I sent the payment", output)


if __name__ == "__main__":
    unittest.main()
