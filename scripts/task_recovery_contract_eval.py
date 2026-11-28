"""CPU-only task recovery policy checks."""
import unittest
from dataclasses import replace
from src.core.task_recovery import TaskSnapshot, recovery_decision

class RecoveryTests(unittest.TestCase):
    def test_decisions(self):
        base = TaskSnapshot("pending", 1, 0, 3, dependencies_complete=True)
        cases = [
            (base, True, 1, "propose"),
            (base, False, 1, "blocked"),
            (base, True, 2, "blocked"),
            (replace(base, dependencies_complete=False), True, 1, "blocked"),
            (replace(base, status="unknown", attempts=3), True, 1, "reconcile"),
            (replace(base, status="running"), True, 1, "wait"),
            (replace(base, status="succeeded"), True, 1, "reconcile"),
            (replace(base, status="succeeded", receipt_verified=True), True, 1, "complete"),
            (replace(base, status="failed", retryable=True, attempts=1), True, 1, "propose"),
            (replace(base, status="failed", retryable=True, attempts=3), True, 1, "blocked"),
            (replace(base, status="failed", attempts=1), True, 1, "blocked"),
            (replace(base, attempts=-1), True, 1, "blocked"),
            (replace(base, receipt_verified=True), True, 1, "reconcile"),
            (replace(base, max_attempts=0), True, 1, "blocked"),
            (replace(base, status="unexpected"), True, 1, "blocked"),
            (replace(base, status="unexpected", receipt_verified=True), True, 1, "blocked"),
            (replace(base, status="failed", retryable=True, attempts=1, dependencies_complete=False), True, 1, "blocked"),
            (replace(base, dependencies_complete=1), True, 1, "blocked"),
            (replace(base, status="failed", retryable=1, attempts=1), True, 1, "blocked"),
        ]
        for snapshot, authorized, version, expected in cases:
            with self.subTest(snapshot=snapshot, authorized=authorized, version=version):
                self.assertEqual(recovery_decision(snapshot, authorized=authorized,
                                                   expected_version=version), expected)

if __name__ == "__main__":
    unittest.main()
