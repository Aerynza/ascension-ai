"""CPU-only recovery adapter regression tests."""
import unittest
from dataclasses import replace
from src.core.task_recovery import TaskSnapshot
from src.core.task_recovery_adapter import propose_recovery

class AdapterTests(unittest.TestCase):
    def test_boundaries(self):
        task = TaskSnapshot("pending", 1, 0, 3, dependencies_complete=True)
        action = "documents.prepare_draft"
        ready = {"action": action, "can_dispatch_now": True,
                 "guardian_required": False, "approval": "safe_internal_auto"}
        cases = [
            (task, True, 1, [ready], "propose"),
            (task, False, 1, [ready], "blocked"),
            (task, True, 2, [ready], "blocked"),
            (task, True, 1, [], "blocked"),
            (task, True, 1, [ready, ready], "blocked"),
            (task, True, 1, [dict(ready, action="other")], "blocked"),
            (task, True, 1, [dict(ready, can_dispatch_now=False)], "blocked"),
            (task, True, 1, [dict(ready, guardian_required=True)], "blocked"),
            (task, True, 1, [dict(ready, approval="explicit_confirmation")], "blocked"),
            (replace(task, status="unknown"), True, 1, [], "reconcile"),
            (replace(task, status="succeeded", receipt_verified=True), True, 1, [], "complete"),
        ]
        for snapshot, authorized, version, actions, expected in cases:
            with self.subTest(expected=expected, actions=actions):
                result = propose_recovery(snapshot, authorized=authorized,
                    expected_version=version, action=action,
                    execution_contract={"proposed_actions": actions})
                self.assertEqual(result["decision"], expected)
                self.assertEqual(result["execution_state"], "proposal_only")

    def test_malformed_contracts(self):
        task = TaskSnapshot("pending", 1, 0, 3, dependencies_complete=True)
        action = "documents.prepare_draft"
        ready = {"action": action, "can_dispatch_now": True,
                 "guardian_required": False, "approval": "safe_internal_auto"}
        malformed = [None, [], {}, {"proposed_actions": None},
                     {"proposed_actions": "invalid"}, {"proposed_actions": [None]},
                     {"proposed_actions": [ready, None]}]
        for key, values in {
            "approval": [None, [], "unknown", "explicit_confirmation"],
            "guardian_required": [None, 0, "false"],
            "can_dispatch_now": [None, 1, "true"],
        }.items():
            malformed.append({"proposed_actions": [{k: v for k, v in ready.items() if k != key}]})
            malformed.extend({"proposed_actions": [dict(ready, **{key: value})]} for value in values)
        for contract in malformed:
            with self.subTest(contract=contract):
                result = propose_recovery(task, authorized=True, expected_version=1,
                    action=action, execution_contract=contract)
                self.assertEqual(result["decision"], "blocked")
                self.assertEqual(result["execution_state"], "proposal_only")

if __name__ == "__main__":
    unittest.main()
