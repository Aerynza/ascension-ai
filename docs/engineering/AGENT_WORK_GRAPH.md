# Ascension AI engineering agent graph (review-only foundation)

This adds a standard-library-only **offline task planner and scope gate**. It does not run Claude/Codex/Devin, download a model, train AI, spend inference tokens, change Git refs, or deploy anything. NightWatch is excluded. It does not affect runtime user shells or training.

## Example plan

```json
{
  "version": 1,
  "max_parallel": 2,
  "units": [
    { "id": "api", "files": ["src/api.py"], "needs": [] },
    { "id": "test", "files": ["tests/test_api.py"], "needs": [
      { "unit": "api", "reads": ["src/api.py"] }
    ] }
  ]
}
```

`files` specify exactly owned repository-relative files. `needs` describe actual producer-owned paths a downstream worker reads. The planner rejects cycles, fake edges, duplicate IDs, two workers owning the same file, path traversal, control-file claims, and more than three concurrent units. The waves output is a schedule **proposal**, not worker execution.

```bash
python scripts/agent_work_graph.py validate /tmp/agent-plan.json
python -m unittest scripts.test_agent_work_graph
python -m compileall -q src scripts
```

After a worker finishes in an isolated worktree, export its `git diff` and untracked file paths as a JSON string array, and run:

```bash
python scripts/agent_work_graph.py check-scope /tmp/agent-plan.json api /tmp/changed-files.json
```

`GREEN_SCOPE_ONLY` proves only that supplied changed paths are inside declared ownership. The executor must independently gather changed paths (never trust worker-provided claims), run production CI/security checks, capture receipts, and enforce bounded retries and compute cost. Safety/release model benchmarks are independent of a passing scope gate. Do not let worker-generated rules change verifiers or approve a merge.

## Future executor (separate reviewed PR)

Once tested, optionally connect isolated workers to this contract with frozen task inputs, local worktrees, replayable job receipts, hardened tool and secret boundaries, and explicit spending limits. Human approval is still required for merges and deployments. No agent may elevate permissions, override native replacement gates, or turn a simulated result into a verified achievement.

Coordination: [engineering issue #40](https://github.com/Aerynza/Aerynza/issues/40).
