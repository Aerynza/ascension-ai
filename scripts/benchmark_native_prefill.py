"""Local, zero-provider-cost GGUF prefill benchmark.

Run on the SAME 2c/4GB host with the pinned v231 model:
  python scripts/benchmark_native_prefill.py --batch 16 --trials 3
  python scripts/benchmark_native_prefill.py --batch 64 --trials 3
The model is loaded once per process (never two copies in RAM). No secrets,
personal conversations, API tokens, network requests or model output are logged.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time


def percentile(samples: list[float], fraction: float) -> float:
    values = sorted(samples)
    if not values:
        return 0.0
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def fixed_messages(trial: int) -> list[dict]:
    # Vary an innocuous suffix to avoid measuring only cached KV prefill.
    return [
        {"role": "system", "content": (
            "You are an assistant. Be accurate. Distinguish facts from assumptions. "
            "Do not claim to have saved data or executed actions without a verified "
            "receipt. Answer concisely, without exposing internal reasoning. "
            "If information is unavailable, say so rather than inventing it."
        )},
        {"role": "user", "content": (
            "A fictional shop has three boxes of notebooks. Each box holds twelve "
            "notebooks. They give away four notebooks and buy eight more. Explain "
            "the arithmetic in a sentence. No real account data exists here. "
            f"Synthetic benchmark sequence {trial}."
        )}
    ]


def benchmark(batch: int, trials: int, warmup: int) -> dict:
    os.environ["ASCENSION_MODEL_BATCH"] = str(batch)
    from src.core.model_runtime import NativeModelRuntime
    instance = NativeModelRuntime()
    instance.load()
    samples = []
    first_token_proxy = []
    completions = []
    for n in range(-warmup, trials):
        started = time.perf_counter()
        result = instance.chat(fixed_messages(n), temperature=0.0, max_tokens=48)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        if n < 0:
            continue
        samples.append(elapsed)
        first_token_proxy.append(result.get("queue_wait_ms", 0))
        completions.append(int((result.get("usage") or {}).get("completion_tokens") or 0))
    return {
        "profile": instance.profile_name,
        "batch": batch,
        "model_load_parameters": __import__("src.core.inference_tuning", fromlist=["llama_cpu_settings"]).llama_cpu_settings(instance.profile),
        "ctx": os.getenv("ASCENSION_MODEL_CONTEXT", "768"),
        "warmup": warmup,
        "samples": len(samples),
        "total_latency_ms": {
            "p50": round(statistics.median(samples), 1),
            "p95": round(percentile(samples, 0.95), 1),
        },
        "queue_wait_ms_p50": round(statistics.median(first_token_proxy), 1),
        "completion_tokens_median": statistics.median(completions),
        "effective_output_tokens_per_second_including_prefill": (
            round(sum(completions) / (sum(samples) / 1000), 2) if sum(samples) else 0
        ),
        "measurement_note": (
            "No token-first timing is claimed by this non-streaming benchmark. "
            "The derived throughput includes prompt prefill and is not raw decode speed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pinned CPU GGUF with fixed synthetic prompts")
    parser.add_argument("--batch", type=int, choices=(16, 32, 64, 128), required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.trials <= 12 or not 0 <= args.warmup <= 2:
        parser.error("trials must be 1..12 and warmup 0..2")
    print(json.dumps(benchmark(args.batch, args.trials, args.warmup), sort_keys=True))


if __name__ == "__main__":
    main()
