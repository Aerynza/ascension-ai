"""Rank research candidates without promoting any checkpoint."""
import math

def select(baseline, candidates):
    def validate(row):
        for key in ("suite_sha256", "input_manifest_sha256"):
            v = row[key]
            if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
                raise ValueError("invalid evidence identity")
        cases = row["cases"]
        if not isinstance(cases, dict) or not cases:
            raise ValueError("missing cases")
        for case in cases.values():
            if type(case["passed"]) is not bool or type(case["safety_passed"]) is not bool:
                raise ValueError("explicit boolean evidence required")
            score = case["score"]
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid score")
        return cases
    base = validate(baseline)
    eligible, rejected, seen = [], {}, set()
    for candidate in candidates:
        name = candidate["id"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("invalid or duplicate trial ID")
        seen.add(name)
        cases = validate(candidate)
        if any(candidate[k] != baseline[k] for k in ("suite_sha256", "input_manifest_sha256")) or cases.keys() != base.keys():
            rejected[name] = "unmatched evidence"
            continue
        if any(not c["safety_passed"] or (base[k]["passed"] and not c["passed"]) or c["score"] < base[k]["score"] for k,c in cases.items()):
            rejected[name] = "safety failure or regression"
            continue
        gain = sum(c["score"] - base[k]["score"] for k,c in cases.items()) / len(base)
        if gain <= 0:
            rejected[name] = "no measured improvement"
            continue
        eligible.append((gain, name))
    eligible.sort(key=lambda x: (-x[0], x[1]))
    return {"recommended_trial": eligible[0][1] if eligible else None,
            "ranked": [{"id": name, "mean_gain": gain} for gain,name in eligible],
            "rejected": rejected, "production_promotion_authorized": False}
