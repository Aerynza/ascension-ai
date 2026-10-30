"""Safe incremental output filter for local GGUF stream chunks.

Keep hidden thinking and model control markers out of user-facing SSE without
buffering the entire response. The caller retains the native model lock.
"""
from __future__ import annotations

_CONTROL_PREFIXES = ("<think>", "</think>", "<|")
_CLOSE_THINK = "</think>"
_MAX_CONTROL_TAG = 80


def safe_model_chunks(chunks):
    pending = ""
    in_thought = False
    for raw in chunks:
        if not raw:
            continue
        pending += str(raw)
        while pending:
            if in_thought:
                close_at = pending.find(_CLOSE_THINK)
                if close_at < 0:
                    # Only preserve enough bytes for a split closing tag.
                    pending = pending[-(len(_CLOSE_THINK) - 1):]
                    break
                pending = pending[close_at + len(_CLOSE_THINK):]
                in_thought = False
                continue

            first_tag = pending.find("<")
            if first_tag < 0:
                yield pending
                pending = ""
                break
            if first_tag > 0:
                yield pending[:first_tag]
                pending = pending[first_tag:]
                continue

            if pending.startswith("<think>"):
                pending = pending[len("<think>"):]
                in_thought = True
                continue
            if pending.startswith(_CLOSE_THINK):
                pending = pending[len(_CLOSE_THINK):]
                continue
            if pending.startswith("<|"):
                end = pending.find("|>")
                if end >= 0 and end < _MAX_CONTROL_TAG:
                    pending = pending[end + 2:]
                    continue
                if len(pending) < _MAX_CONTROL_TAG:
                    break
            if any(prefix.startswith(pending) for prefix in _CONTROL_PREFIXES):
                break

            # Not a recognized marker. Emit one character and keep streaming.
            yield pending[0]
            pending = pending[1:]

    # Never release a truncated hidden-thought section or partial control tag.
    if pending and not in_thought and not any(
        prefix.startswith(pending) for prefix in _CONTROL_PREFIXES
    ):
        yield pending
