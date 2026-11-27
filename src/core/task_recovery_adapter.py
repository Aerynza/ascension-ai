"""Product-side adapter; never dispatches or accepts model-provided authority."""
from .task_recovery import TaskSnapshot, recovery_decision

_SAFE_APPROVALS = frozenset({"safe_internal_auto", "safe_read", "safe_research"})

def propose_recovery(task: TaskSnapshot, *, authorized: bool,
                     expected_version: int, action: str,
                     execution_contract: dict) -> dict:
    """Caller supplies a freshly built, authenticated execution contract.
    Readiness is necessary but never sufficient for dispatch.
    The product must recheck approval, actor and version.
    """
    decision = recovery_decision(task, authorized=authorized,
                                 expected_version=expected_version)
    if decision == "propose":
        actions = execution_contract.get("proposed_actions") if isinstance(execution_contract, dict) else None
        if not isinstance(actions, list) or not all(isinstance(item, dict) for item in actions):
            decision = "blocked"
        else:
            matches = [item for item in actions if item.get("action") == action]
            if not isinstance(action, str) or not action.strip() or len(matches) != 1:
                decision = "blocked"
            else:
                item = matches[0]
                approval = item.get("approval")
                if (item.get("can_dispatch_now") is not True
                    or item.get("guardian_required") is not False
                    or not isinstance(approval, str)
                    or approval not in _SAFE_APPROVALS):
                    decision = "blocked"
    return {"action": action, "decision": decision,
            "execution_state": "proposal_only", "task_version": task.version}
