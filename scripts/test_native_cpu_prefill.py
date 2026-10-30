"""Prefill settings and synthetic benchmark contract: no model download required."""
import unittest
from src.core.inference_tuning import llama_cpu_settings
from scripts.benchmark_native_prefill import fixed_messages, percentile


class CpuPrefillSettingsTest(unittest.TestCase):
    def setUp(self):
        self.profile = {"threads": 2}

    def test_existing_baseline_is_unchanged_until_opt_in(self):
        value = llama_cpu_settings(self.profile, {})
        self.assertEqual(value, {
            "n_ctx": 768, "n_threads": 2, "n_threads_batch": 2,
            "n_batch": 16,
        })

    def test_opt_in_fast_prefill_uses_bounded_sixty_four_token_batch(self):
        value = llama_cpu_settings(self.profile, {"AERYNZA_NATIVE_FAST_PREFILL": "true"})
        self.assertEqual(value["n_batch"], 64)
        self.assertEqual(value["n_ctx"], 768)
        self.assertEqual(value["n_threads"], 2)

    def test_explicit_operator_batch_setting_wins_over_fast_default(self):
        value = llama_cpu_settings(self.profile, {
            "AERYNZA_NATIVE_FAST_PREFILL": "1",
            "ASCENSION_MODEL_BATCH": "32",
            "ASCENSION_MODEL_THREADS": "2",
            "ASCENSION_MODEL_THREADS_BATCH": "2",
        })
        self.assertEqual(value["n_batch"], 32)

    def test_batch_cannot_exceed_model_context_or_safety_cap(self):
        value = llama_cpu_settings(self.profile, {
            "ASCENSION_MODEL_CONTEXT": "256", "ASCENSION_MODEL_BATCH": "256",
        })
        self.assertEqual(value["n_batch"], 256)
        for raw in ("0", "-1", "300", "abc", "999999"):
            with self.subTest(value=raw):
                with self.assertRaises(ValueError):
                    llama_cpu_settings(self.profile, {"ASCENSION_MODEL_BATCH": raw})
        with self.assertRaises(ValueError):
            llama_cpu_settings(self.profile, {"ASCENSION_MODEL_THREADS": "0"})

    def test_token_averages_are_not_mislabeled_as_raw_decode_speed(self):
        self.assertEqual(percentile([100, 200, 300], 0.5), 200)
        self.assertEqual(percentile([100, 200], 0.95), 195)

    def test_benchmark_contains_no_personal_information(self):
        example = fixed_messages(2)
        self.assertEqual([x["role"] for x in example], ["system", "user"])
        self.assertIn("Synthetic benchmark", example[1]["content"])
        self.assertNotIn("API key", str(example))
        self.assertNotIn("your bank", str(example).lower())


if __name__ == "__main__":
    unittest.main()
