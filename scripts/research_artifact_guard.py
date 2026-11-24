"""Verify research input files against a previously recorded manifest."""
import hashlib
import hmac
from pathlib import Path

def fingerprint(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("regular non-symlink artifact required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def snapshot(artifacts):
    required = {"checkpoint", "metadata", "tokenizer", "dataset", "config", "gate"}
    if not required.issubset(artifacts):
        raise ValueError("incomplete research artifact manifest")
    return {name: fingerprint(path) for name, path in artifacts.items()}

def verify(artifacts, expected):
    actual = snapshot(artifacts)
    if actual.keys() != expected.keys():
        raise ValueError("artifact set changed")
    for name, digest in actual.items():
        if not isinstance(expected[name], str) or not hmac.compare_digest(digest, expected[name]):
            raise ValueError(f"artifact identity changed: {name}")
    return actual
