"""Offline graph planner and scope validator for isolated engineering workers.

This module never runs a model, executes commands, or modifies a repository.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
PROTECTED = {"AGENTS.md", "CLAUDE.md", ".gitleaks.toml", ".gitleaksignore", "scripts/agent_work_graph.py"}


def safe_path(value: object) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or value.startswith("/") or re.match(r"^[A-Za-z]:", value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"invalid repository path: {value!r}")
    if (value in PROTECTED or value == ".env" or value.startswith(".env.")
            or value == ".git" or value.startswith(".git/")
            or value == ".github" or value.startswith(".github/")):
        raise ValueError(f"protected repository path: {value}")
    return value


def paths(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise ValueError(f"{label} needs 1-64 paths")
    result = [safe_path(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate paths")
    return result


def validate_plan(plan: object) -> dict:
    if (not isinstance(plan, dict) or type(plan.get("version")) is not int or plan["version"] != 1
            or not isinstance(plan.get("units"), list) or not 1 <= len(plan["units"]) <= 32):
        raise ValueError("plan must use version 1 and contain 1-32 units")
    max_parallel = plan.get("max_parallel", 2)
    if type(max_parallel) is not int or not 1 <= max_parallel <= 3:
        raise ValueError("max_parallel must be 1-3")
    units: dict[str, dict] = {}
    owners: dict[str, str] = {}
    for unit in plan["units"]:
        if (not isinstance(unit, dict) or not isinstance(unit.get("id"), str)
                or ID.fullmatch(unit["id"]) is None or unit["id"] in units):
            raise ValueError("invalid or repeated unit id")
        unit_files = paths(unit.get("files"), f"unit {unit['id']}")
        needs = unit.get("needs", [])
        if not isinstance(needs, list):
            raise ValueError(f"{unit['id']}: needs must be an array")
        for file in unit_files:
            if file in owners:
                raise ValueError(f"file ownership collision: {file}")
            owners[file] = unit["id"]
        units[unit["id"]] = {"files": unit_files, "needs": needs}
    downstream: dict[str, list[str]] = {key: [] for key in units}
    indegree = {key: 0 for key in units}
    for unit_id, unit in units.items():
        seen: set[str] = set()
        for edge in unit["needs"]:
            if (not isinstance(edge, dict) or not isinstance(edge.get("unit"), str)
                    or edge["unit"] not in units or edge["unit"] == unit_id
                    or edge["unit"] in seen):
                raise ValueError(f"{unit_id}: invalid dependency")
            seen.add(edge["unit"])
            reads = paths(edge.get("reads"), f"{unit_id} dependency on {edge['unit']}")
            for read in reads:
                if read not in units[edge["unit"]]["files"]:
                    raise ValueError(f"{unit_id}: fake edge to {edge['unit']}; {read} is not produced by it")
            indegree[unit_id] += 1
            downstream[edge["unit"]].append(unit_id)
    waves: list[list[str]] = []
    ready = sorted(key for key in units if indegree[key] == 0)
    visited = 0
    while ready:
        batch, ready = ready[:max_parallel], ready[max_parallel:]
        waves.append(batch)
        visited += len(batch)
        newly_ready: list[str] = []
        for unit_id in batch:
            for consumer in downstream[unit_id]:
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    newly_ready.append(consumer)
        ready = sorted(ready + newly_ready)
    if visited != len(units):
        raise ValueError("dependency cycle detected")
    return {"version": 1, "max_parallel": max_parallel, "waves": waves, "units": sorted(units)}


def check_scope(plan: dict, unit_id: str, changed_paths: object) -> dict:
    validate_plan(plan)
    unit = next((unit for unit in plan["units"] if unit["id"] == unit_id), None)
    if unit is None:
        raise ValueError(f"unknown unit: {unit_id}")
    changed = paths(changed_paths, "changed paths")
    outside = [file for file in changed if file not in unit["files"]]
    if outside:
        raise ValueError(f"out-of-scope changes for {unit_id}: {', '.join(outside)}")
    return {"unit": unit_id, "changed": len(changed), "status": "GREEN_SCOPE_ONLY"}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("validate", "check-scope"):
        raise ValueError("usage: python scripts/agent_work_graph.py validate PLAN.json | check-scope PLAN.json UNIT CHANGES.json")
    plan = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    if argv[0] == "validate":
        result = validate_plan(plan)
    elif len(argv) == 4:
        result = check_scope(plan, argv[2], json.loads(Path(argv[3]).read_text(encoding="utf-8")))
    else:
        raise ValueError("check-scope requires unit and changes JSON")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"RED: {error}", file=sys.stderr)
        sys.exit(1)
