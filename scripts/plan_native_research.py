"""Generate offline experiment plans without invoking training or paid compute."""
import argparse
import hashlib
import json
from pathlib import Path

def plan(checkpoint, dataset, config, minutes, trials):
    if type(minutes) is not int or type(trials) is not int or not 1 <= trials <= 12 or not trials <= minutes <= 120:
        raise ValueError("invalid research budget")
    identities = {}
    for name, path in (("checkpoint", checkpoint), ("dataset", dataset), ("config", config)):
        with Path(path).open("rb") as stream:
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1048576), b""):
                h.update(chunk)
        identities[name] = h.hexdigest()
    candidates = [{"kind": "baseline"}, *[
        {"kind": "learning_rate_multiplier", "value": x}
        for x in (0.75, 1.25, 0.5, 1.5, 0.9, 1.1, 0.6, 1.4, 0.8, 1.2, 0.7)
    ]]
    return {"identities": identities, "budget_minutes": minutes,
            "paid_compute_authorized": False, "production_promotion_authorized": False,
            "mode": "new_stage_from_verified_checkpoint",
            "trials": [{"id": i, "max_minutes": minutes // trials, **candidate}
                       for i, candidate in enumerate(candidates[:trials])],
            "requirements": ["isolated output per trial", "fixed held-out suite",
                             "preserve full recovery state", "reject regressions"]}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "config"):
        p.add_argument(name)
    p.add_argument("--minutes", type=int, default=30)
    p.add_argument("--trials", type=int, default=3)
    a = p.parse_args()
    print(json.dumps(plan(a.checkpoint, a.dataset, a.config, a.minutes, a.trials), indent=2))
