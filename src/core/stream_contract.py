"""Receipt-safe output gating for native incremental model responses.

A response fragment may be shown only after its sentence can pass the same
truth contract as a full native reply. Sensitive requests defer disclosure until
the whole model answer is verified. Do not log prompts, tokens or private data.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from .contracts import Shell
from .orchestrator import (
    EXPLICIT_PERSISTENCE_REQUEST,
    TIMEOUT_RESULT,
    enforce_response_contract,
)

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_MAX_BUFFER_CHARS = 32000
_SENSITIVE_TOPIC = re.compile(
    r"\b(?:payment|payroll|transfer|purchase|bank|balance|credit card|"
    r"medical|diagnos|prescription|health record|saved|schedule|"
    r"send|sent|delete|erase|calendar|booked|reservation)\b",
    re.I,
)


def requires_full_validation(mode: str, latest: str, cognitive: dict) -> bool:
    """Conservatively prioritize truth over streaming for consequential turns."""
    if mode != "conversation":
        return True
    if EXPLICIT_PERSISTENCE_REQUEST.search(latest) or TIMEOUT_RESULT.search(latest):
        return True
    if _SENSITIVE_TOPIC.search(latest):
        return True
    if (cognitive or {}).get("action_proposals"):
        return True
    return False


def verified_output_chunks(
    chunks: Iterable[str], *, cognitive: dict, context: dict,
    mode: str, latest: str,
) -> Iterator[str]:
    """Return sentence-validated pieces or one fully verified sensitive reply.

    Never emit arbitrary text before the contract filter can reject an
    unreceipted completion. When no sentence boundary exists, buffer the
    remaining bounded text rather than leaking a potentially unsafe prefix.
    """
    full_validation = requires_full_validation(mode, latest, cognitive)
    pending = ""
    for raw in chunks:
        if not raw:
            continue
        pending += str(raw)
        if len(pending) > _MAX_BUFFER_CHARS:
            raise RuntimeError("Native stream exceeded safe buffering limit")
        if full_validation:
            continue
        while True:
            boundary = _BOUNDARY.search(pending)
            if boundary is None:
                break
            fragment = pending[:boundary.start()]
            separator = pending[boundary.start():boundary.end()]
            pending = pending[boundary.end():]
            verified = enforce_response_contract(fragment, cognitive, context, mode, latest)
            if verified:
                yield verified + separator

    if pending.strip():
        verified = enforce_response_contract(pending, cognitive, context, mode, latest)
        if verified:
            yield verified
