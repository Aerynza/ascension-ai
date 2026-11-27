"""Pure recovery policy for product-verified task snapshots.

The caller must authenticate the actor, validate receipt provenance and
perform atomic version checks at execution time. This module never executes
actions or grants permissions.
"""
from dataclasses import dataclass
from typing import Literal

Status = Literal["pending", "running", "unknown", "failed", "succeeded"]
Decision = Literal["blocked", "reconcile", "complete", "propose", "wait"]

@dataclass(frozen=True)
class TaskSnapshot:
    status: Status
    version: int
    attempts: int
    max_attempts: int
    receipt_verified: bool = False
    retryable: bool = False
    dependencies_complete: bool = False

def recovery_decision(task: TaskSnapshot, *, authorized: bool,
                      expected_version: int) -> Decision:
    """Return a proposal only; authorization is a trusted caller input."""
    if authorized is not True or task.version != expected_version:
        return "blocked"
    if any(type(v) is not int or v < 0 for v in
           (task.version, expected_version, task.attempts, task.max_attempts)):
        return "blocked"
    if task.max_attempts == 0 or task.attempts > task.max_attempts:
        return "blocked"
    if task.status == "succeeded":
        return "complete" if task.receipt_verified is True else "reconcile"
    if task.status == "unknown":
        return "reconcile"
    if task.status == "running":
        return "wait"
    if task.receipt_verified is not False:
        return "reconcile"
    if task.attempts >= task.max_attempts or task.dependencies_complete is not True:
        return "blocked"
    if task.status == "pending" and task.attempts == 0:
        return "propose"
    if task.status == "failed" and task.retryable is True:
        return "propose"
    return "blocked"
