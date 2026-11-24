"""Run one explicitly configured local trial in a fresh output directory."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def run(command, output, seconds):
    if type(seconds) is not int or not 1 <= seconds <= 7200:
        raise ValueError("invalid time budget")
    if not command or not all(isinstance(x, str) for x in command):
        raise ValueError("explicit argument list required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, AERYNZA_TRIAL_OUTPUT=str(output),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               WANDB_MODE="disabled")
    receipt = {"status": "failed", "production_promotion_authorized": False}
    with (output / "trial.log").open("wb") as log:
        try:
            result = subprocess.run(command, cwd=output, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=seconds, check=False)
            receipt.update(status="completed" if result.returncode == 0 else "failed",
                           returncode=result.returncode)
        except subprocess.TimeoutExpired:
            receipt["status"] = "timed_out"
        finally:
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", required=True, type=int)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    result = run(command, a.output, a.seconds)
    print(json.dumps(result))
    sys.exit(0 if result["status"] == "completed" else 1)
