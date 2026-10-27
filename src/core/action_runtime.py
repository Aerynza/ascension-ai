"""Provider-independent action readiness and receipt validation for every shell.

The native model plans and explains. Authenticated product shells advertise the
actions they can currently execute, perform those actions, and return receipts.
This module keeps provider availability separate from model competence.
"""
from __future__ import annotations

from typing import Any

from .contracts import Shell


ACTION_EXECUTION_MODES = {
    "schedule.upsert_recurring_work": "internal_write",
    "schedule.prepare_week": "internal_write",
    "finance.refresh_cashflow": "provider_read",
    "finance.prepare_budget": "internal_write",
    "trading.refresh_prediction_markets": "provider_read",
    "trading.prepare_prediction_position": "internal_write",
    "trading.submit_prediction_order": "provider_write",
    "housing.search_options": "provider_read",
    "creation.save_seed": "internal_write",
    "creation.prepare_project": "internal_write",
    "immersive.prepare_world": "internal_write",
    "nutrition.research_recipes": "provider_read",
    "nutrition.prepare_meal_plan": "internal_write",
    "learning.prepare_course": "internal_write",
    "career.research_jobs": "provider_read",
    "documents.prepare_draft": "internal_write",
    "messages.send": "provider_write",
    "calendar.external_write": "provider_write",
    "finance.payment": "future_provider_write",
    "task.create_quest": "internal_write",
}

SHELL_ACTION_PREFIXES = {
    Shell.CORE: ("schedule.", "finance.", "trading.", "housing.", "creation.", "immersive.",
                 "nutrition.", "learning.", "career.", "documents.", "messages.", "calendar.", "task."),
    Shell.AP: ("schedule.", "finance.", "trading.", "housing.", "creation.", "immersive.",
               "nutrition.", "learning.", "career.", "documents.", "messages.", "calendar.", "task."),
    Shell.LIFE_OS: ("schedule.", "finance.", "trading.", "housing.", "creation.", "immersive.",
                    "nutrition.", "learning.", "career.", "documents.", "messages.", "calendar.", "task."),
    Shell.SPROUT: ("schedule.", "creation.", "nutrition.", "learning.", "documents.", "task."),
    Shell.NEXUS_HOME: ("schedule.", "finance.refresh_", "finance.prepare_", "housing.", "creation.",
                       "nutrition.", "learning.", "documents.", "messages.", "calendar.", "task."),
    Shell.NEXUS_FAMILY: ("schedule.", "finance.refresh_", "finance.prepare_", "housing.", "creation.",
                         "nutrition.", "learning.", "career.", "documents.", "messages.", "calendar.", "task."),
    Shell.EXECUTIVE: ("schedule.", "finance.refresh_", "finance.prepare_", "creation.", "learning.", "career.", "documents.", "messages.", "calendar.", "task."),
    Shell.CREATION: ("creation.", "immersive.", "documents.", "messages.", "learning.", "career."),
}

SUCCESS_STATES = {"completed", "confirmed", "success", "succeeded"}


def shell_allows_action(shell: Shell | None, action: str) -> bool:
    if shell is None:
        return False
    return any(action.startswith(prefix) for prefix in SHELL_ACTION_PREFIXES.get(shell, ()))


def shell_action_catalog(shell: Shell) -> list[dict[str, Any]]:
    """Expose the stable action vocabulary a product shell may advertise."""
    return [
        {"action": action, "execution_mode": mode, "shell_allowed": True}
        for action, mode in sorted(ACTION_EXECUTION_MODES.items())
        if shell_allows_action(shell, action)
    ]


def classify_action_readiness(
    action: dict[str, Any], shell: Shell | None, available_actions: list[str] | None
) -> dict[str, Any]:
    """Describe whether a shell could dispatch an action without executing it."""
    action_id = str(action.get("action") or "")
    mode = ACTION_EXECUTION_MODES.get(action_id, "unknown")
    advertised = {str(item).strip() for item in (available_actions or []) if str(item).strip()}
    shell_allowed = shell_allows_action(shell, action_id)
    internal = mode == "internal_write"
    provider_available = internal or action_id in advertised
    future_nonblocking = mode == "future_provider_write" and action_id not in advertised
    missing = list(action.get("missing_variables") or [])
    explicit = action.get("approval") == "explicit_confirmation"
    guardian = bool(action.get("guardian_required"))

    if not shell_allowed:
        state, reason = "blocked_shell_scope", "This shell is not authorized to dispatch this action."
    elif future_nonblocking:
        state, reason = "future_capability_unavailable", "The company/provider capability is not enabled; model readiness is non-blocking."
    elif not provider_available:
        state, reason = "provider_not_connected", "The shell did not advertise a connected executor for this action."
    elif missing:
        state, reason = "awaiting_context", "Material action inputs are still missing."
    elif guardian:
        state, reason = "awaiting_guardian_confirmation", "Guardian authority must be confirmed by the authenticated shell."
    elif explicit:
        state, reason = "awaiting_explicit_confirmation", "The exact action requires final user confirmation."
    else:
        state, reason = "ready_for_shell_execution", "The authenticated shell may execute and return a receipt."

    return {
        "execution_mode": mode,
        "shell_allowed": shell_allowed,
        "provider_advertised": action_id in advertised,
        "provider_available": provider_available,
        "future_capability_nonblocking": future_nonblocking,
        "dispatch_state": state,
        "dispatch_reason": reason,
        "can_dispatch_now": state == "ready_for_shell_execution",
    }


def validate_action_receipt(action: dict[str, Any], receipt: dict[str, Any] | None) -> dict[str, Any]:
    """Validate shell/provider evidence before any completion claim is allowed."""
    receipt = receipt if isinstance(receipt, dict) else {}
    action_id = str(action.get("action") or "")
    errors: list[str] = []
    if str(receipt.get("action") or "") != action_id:
        errors.append("receipt_action_mismatch")
    if str(receipt.get("status") or "").casefold() not in SUCCESS_STATES:
        errors.append("receipt_not_successful")
    if receipt.get("verified") is not True:
        errors.append("receipt_not_verified_by_authenticated_shell")
    if not (receipt.get("id") or receipt.get("reference") or receipt.get("provider_receipt_id")):
        errors.append("receipt_identifier_missing")
    missing_fields = [field for field in action.get("receipt_fields", []) if receipt.get(field) in (None, "", [])]
    if missing_fields:
        errors.append("receipt_fields_missing")
    return {
        "action": action_id,
        "valid": not errors,
        "errors": errors,
        "missing_fields": missing_fields,
        "execution_state": "confirmed" if not errors else "unconfirmed",
        "receipt_id": receipt.get("id") or receipt.get("reference") or receipt.get("provider_receipt_id"),
    }
