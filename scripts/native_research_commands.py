"""Build native trainer argv; does not execute or authorize continuation."""
import math
import re
from pathlib import Path
import sys

def build_command(repo, base_version, experiment_version, gate, steps, rate, *, reviewed=False):
    for version in (base_version, experiment_version):
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", version):
            raise ValueError("invalid version")
    if not experiment_version.startswith("research_") or experiment_version == base_version:
        raise ValueError("distinct research version required")
    if reviewed is not True:
        raise ValueError("human review required")
    if type(steps) is not int or not 1 <= steps <= 20000:
        raise ValueError("invalid steps")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 0.001:
        raise ValueError("invalid learning rate")
    trainer = Path(repo).resolve() / "scripts/train_ascension_product_v6.py"
    gate = Path(gate).resolve()
    if not trainer.is_file() or not gate.is_file():
        raise FileNotFoundError("trainer and continuation gate required")
    return [sys.executable, str(trainer), "--base-version", base_version,
            "--version", experiment_version, "--gate-result", str(gate),
            "--human-review-approved", "--steps", str(steps),
            "--learning-rate", str(rate), "--save-every", str(min(steps, 250)),
            "--assistant-only-loss"]
