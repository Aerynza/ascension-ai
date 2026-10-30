"""Bounded, reversible llama.cpp CPU prefill knobs for 2c/4GB model profiles.

Opt-in tuning leaves existing production settings unchanged until benchmarked.
No prompt, transcript, personal context or model weights are written here.
"""
from __future__ import annotations

import os

FAST_BATCH_TOKENS = 64
DEFAULT_BATCH_TOKENS = 16
MAX_BATCH_TOKENS = 256


def _positive_int(value: str | int, name: str, low: int, high: int) -> int:
    try:
        result = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not low <= result <= high:
        raise ValueError(f"{name} out of permitted range ({low}-{high})")
    return result


def llama_cpu_settings(profile: dict, environ: dict | None = None) -> dict:
    env = os.environ if environ is None else environ
    context = _positive_int(env.get("ASCENSION_MODEL_CONTEXT",
                                    env.get("AERYNZA_MODEL_CONTEXT", "768")),
                            "model context", 256, 8192)
    threads = _positive_int(env.get("ASCENSION_MODEL_THREADS", profile.get("threads", 2)),
                            "inference threads", 1, 16)
    batch_threads = _positive_int(env.get("ASCENSION_MODEL_THREADS_BATCH", threads),
                                  "prefill threads", 1, 16)
    fast_mode = str(env.get("AERYNZA_NATIVE_FAST_PREFILL", "0")).lower() in ("1", "true", "yes")
    default_batch = FAST_BATCH_TOKENS if fast_mode else DEFAULT_BATCH_TOKENS
    batch = _positive_int(env.get("ASCENSION_MODEL_BATCH",
                                  env.get("AERYNZA_MODEL_BATCH", default_batch)),
                          "prefill batch", 8, MAX_BATCH_TOKENS)
    # llama.cpp prefill never needs to configure more tokens per batch than
    # the model's allocated context. This also keeps memory bounded.
    return {
        "n_ctx": context,
        "n_threads": threads,
        "n_threads_batch": batch_threads,
        "n_batch": min(batch, context),
    }
