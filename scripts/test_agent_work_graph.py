"""No-dependency contract tests for the offline agent graph."""
import unittest

try:
    from scripts.agent_work_graph import validate_plan, check_scope
except ModuleNotFoundError:
    from agent_work_graph import validate_plan, check_scope


def sample_plan():
    return {"version": 1, "max_parallel": 2, "units": [
        {"id": "api", "files": ["src/api.py"], "needs": []},
        {"id": "docs", "files": ["docs/api.md"], "needs": []},
        {"id": "consumer", "files": ["tests/test_api.py"],
         "needs": [{"unit": "api", "reads": ["src/api.py"]}]},
    ]}


class GraphContractTests(unittest.TestCase):
    def test_order_and_parallel_limit(self):
        self.assertEqual(validate_plan(sample_plan())["waves"], [["api", "docs"], ["consumer"]])

    def test_scope(self):
        self.assertEqual(check_scope(sample_plan(), "api", ["src/api.py"])["status"], "GREEN_SCOPE_ONLY")
        with self.assertRaisesRegex(ValueError, "out-of-scope"):
            check_scope(sample_plan(), "api", ["src/another.py"])
        with self.assertRaisesRegex(ValueError, "changed paths"):
            check_scope(sample_plan(), "api", [])

    def test_fake_dependency(self):
        plan = sample_plan()
        plan["units"][2]["needs"][0]["reads"] = ["docs/api.md"]
        with self.assertRaisesRegex(ValueError, "fake edge"):
            validate_plan(plan)

    def test_shared_ownership(self):
        plan = sample_plan()
        plan["units"][1]["files"] = ["src/api.py"]
        with self.assertRaisesRegex(ValueError, "collision"):
            validate_plan(plan)

    def test_cycle(self):
        plan = sample_plan()
        plan["units"][0]["needs"] = [{"unit": "consumer", "reads": ["tests/test_api.py"]}]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_plan(plan)

    def test_bad_paths(self):
        for file in ("../escape", "/tmp/secret", ".github/workflows/p0-checks.yml", ".env", "AGENTS.md", "a//b", "a\\b"):
            with self.subTest(file=file):
                plan = sample_plan()
                plan["units"][0]["files"] = [file]
                with self.assertRaisesRegex(ValueError, "path"):
                    validate_plan(plan)

    def test_unknown_dependency_and_repeated_ids(self):
        plan = sample_plan()
        plan["units"][2]["needs"][0]["unit"] = "unknown"
        with self.assertRaisesRegex(ValueError, "dependency"):
            validate_plan(plan)
        plan = sample_plan()
        plan["units"][1]["id"] = "api"
        with self.assertRaisesRegex(ValueError, "unit id"):
            validate_plan(plan)

    def test_parallel_limit(self):
        plan = sample_plan()
        plan["max_parallel"] = 4
        with self.assertRaisesRegex(ValueError, "max_parallel"):
            validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
