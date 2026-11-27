"""Product-side adapter; never dispatches or accepts model-provided authority."""
from .task_recovery import TaskSnapshot, recovery_decision

def propose_recovery(task: TaskSnapshot, *, authorized: bool,
                     expected_version: int, action: str,
                     execution_contract: dict) -> dict:
    """Caller supplies a freshly built, authenticated execution contract.

    Match exactly one action; readiness is necessary but never sufficient
    for dispatch. The product must recheck approval, actor and version.
    """
    decision = recovery_decision(task, authorized=authorized,
                                 expected_version=expected_version)
    if decision == "propose":
        matches = [item for item in execution_contract.get("proposed_actions", [])
                   if item.get("action") == action]
        if len(matches) != 1 or matches[0].get("can_dispatch_now") is not True:
            decision = "blocked"
        elif matches[0].get("guardian_required") or matches[0].get("approval") == "explicit_confirmation":
            decision = "blocked"
    return {"action": action, "decision": decision,
            "execution_state": "proposal_only", "task_version": task.version}
