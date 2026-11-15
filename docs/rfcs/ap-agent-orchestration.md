# AP agent workforce and durable execution

Status: proposed; implementation gated by tests and review.

## Goal
AP is the sole user-facing coordinator. A request becomes a durable outcome-oriented job rather than a chain of manual prompts.

## Job lifecycle
1. Capture goal, explicit authorization, constraints, and acceptance criteria.
2. Planner creates bounded tasks with dependencies and budgets.
3. Delegate to specialist workers (research, builder, finance, family, verifier) through existing infrastructure; no second competing orchestrator.
4. Each worker runs with least-privilege scoped capabilities, isolated workspace, time and spend limits, redacted logs, and idempotency keys.
5. Checkpoint job/task state and tool outputs durably. Retry recoverable errors with backoff; never silently restart a completed side effect.
6. Independent verifier checks evidence against acceptance criteria before marking done. Report blockers and request approvals only when necessary.

## Safety boundaries
- Never expose credentials to model prompts or untrusted content.
- External communications, purchases, money movement, permission changes, deployments, and destructive operations require explicit user authorization appropriate to the action.
- Prompt injection in retrieved material cannot grant new permissions.
- Maintain immutable audit trail, cancellation, rate limits, tenant isolation, and human override.
- Family/child data remains scoped to guardian permissions; no cross-household access.

## Interfaces (proposal)
- Job: id, owner, tenant, goal, criteria, status, created_at, updated_at, policy_version.
- Task: id, job_id, worker_type, dependencies, capability_scope, status, attempts, checkpoint_ref, verification_ref.
- Approval: action digest, user, scope, expiry, decision, audit reference.
- Verification: criteria, evidence references, pass/fail/unknown, verifier version.

## Acceptance tests
- Resume a partially completed job without repeating successful external side effects.
- Reject unauthorized tool invocation and cross-tenant access.
- Surface missing permissions as a single actionable blocker.
- Reconcile failed and uncertain actions rather than claiming success.
- Keep AP Home and dashboard separate; show job progress without blocking conversation.

## Rollout
Audit existing workers and queues first; implement behind a feature flag; run dry-run integration tests; enable limited cohorts before general release.

## Source note
Inspired by user-shared X links; exact post content not verified. No third-party claim is treated as a confirmed requirement.
