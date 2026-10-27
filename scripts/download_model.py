"""Download and verify the pinned local model selected for this deployment."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import hf_hub_download
from model_deploy_preflight import preflight, PreflightRejected


ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "config" / "model_profiles.json"
PROFILE_ALIASES_PATH = ROOT / "config" / "profile_aliases.json"
MODELS_DIR = ROOT / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, path: Path, expected_sha256: str | None = None) -> Path:
    req = Request(url, headers={"User-Agent": "ascension-ai-download/1.0"})
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urlopen(req, timeout=60) as source, path.open("wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected_sha256 and actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise SystemExit("Downloaded model checksum did not match the pinned manifest.")
    return path


def main() -> None:
    # Fail before any network activity if a costly pinned artifact cannot fit
    # the selected service's explicitly configured model RAM budget.
    try:
        selected = preflight()
    except (PreflightRejected, OSError, ValueError) as exc:
        raise SystemExit("MODEL PREFLIGHT FAILED: " + str(exc)) from None
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    requested_name = os.getenv("ASCENSION_MODEL_PROFILE", "starter").strip().lower()
    aliases = json.loads(PROFILE_ALIASES_PATH.read_text(encoding="utf-8")) if PROFILE_ALIASES_PATH.is_file() else {}
    name = str(aliases.get(requested_name, requested_name)).strip().lower()
    if name not in profiles:
        raise SystemExit(f"Unknown ASCENSION_MODEL_PROFILE: {name}")
    profile = profiles[name]
    if name != selected["profile"]:
        raise SystemExit("Model profile changed during deployment preflight")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    filename = profile["filename"]
    if "download_url" in profile:
        downloaded = MODELS_DIR / filename
        if not downloaded.is_file() or sha256(downloaded) != profile["sha256"]:
            downloaded = download_url(profile["download_url"], downloaded, profile.get("sha256"))
    else:
        downloaded = Path(hf_hub_download(
            repo_id=profile["repo_id"],
            filename=filename,
            revision=profile["revision"],
            local_dir=MODELS_DIR,
        ))
        actual = sha256(downloaded)
        if actual != profile["sha256"]:
            downloaded.unlink(missing_ok=True)
            raise SystemExit("Downloaded model checksum did not match the pinned manifest.")

    print(json.dumps({
        "status": "verified",
        "profile": name,
        "requested_profile": requested_name,
        "model": profile["label"],
        "path": str(downloaded.relative_to(ROOT)),
        "size_bytes": downloaded.stat().st_size,
    }))


if __name__ == "__main__":
    main()
