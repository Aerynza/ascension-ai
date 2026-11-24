"""Compare held-out capability receipts offline; never changes routing."""
import argparse
import json
import math
from pathlib import Path

def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("suite_sha256"), str):
        raise ValueError("suite identity required")
    digest = data["suite_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid suite digest")
    rows = {}
    for row in data["cases"]:
        key = row["id"]
        if not isinstance(key, str) or not key or key in rows:
            raise ValueError("invalid or duplicate case ID")
        for field in ("passed", "safety_passed"):
            if type(row[field]) is not bool:
                raise ValueError("explicit boolean results required")
        latency = row["latency_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid latency")
        rows[key] = row
    if not rows:
        raise ValueError("empty evaluation")
    return digest, rows

def compare(native, supplemental):
    nd, n = native
    sd, s = supplemental
    if nd != sd or n.keys() != s.keys():
        raise ValueError("evaluation suites differ")
    regressions = sorted(k for k in n if s[k]["passed"] and not n[k]["passed"])
    failures = sorted(k for k in n if not n[k]["passed"])
    unsafe = sorted(k for k in n if not n[k]["safety_passed"])
    return {"suite_sha256": nd, "case_count": len(n),
            "native_failures": failures, "regressions": regressions,
            "native_safety_failures": unsafe,
            "native_mean_latency_ms": sum(x["latency_ms"] for x in n.values()) / len(n),
            "supplemental_mean_latency_ms": sum(x["latency_ms"] for x in s.values()) / len(s),
            "eligible_for_further_review": not (failures or unsafe or regressions),
            "production_replacement_authorized": False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native")
    parser.add_argument("supplemental")
    args = parser.parse_args()
    result = compare(load(args.native), load(args.supplemental))
    print(json.dumps(result, indent=2))
    return 0 if result["eligible_for_further_review"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
