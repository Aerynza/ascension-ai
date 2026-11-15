"""Native Sprout shell must never inherit adult human-companion mode language."""
import unittest

from src.core.contracts import Shell, Tier, response_contract, system_contract


class SproutLearningGuideContractTests(unittest.TestCase):
    def test_sprout_system_is_nonhuman_and_guardian_scoped(self):
        child = system_contract(Shell.SPROUT, Tier.LIFE_OS, ["learning"])
        self.assertIn("clearly nonhuman AI learning guide", child)
        self.assertIn("Never solicit loyalty, affection, secrecy", child)
        self.assertIn("Never import a guardian's private LifeOS memory", child)
        self.assertIn("A reading quiz is practice, not mastery", child)
        self.assertIn("purpose-specific revocation", child)
        self.assertIn("verified receipt", child)

    def test_child_conversation_overrides_adult_companion_contract_only(self):
        child = response_contract("conversation", shell=Shell.SPROUT)
        self.assertIn("nonhuman AI learning guide", child)
        self.assertNotIn("Reply like a present human companion", child)
        self.assertIn("trusted adults", child)
        adult = response_contract("conversation", shell=Shell.AP)
        legacy = response_contract("conversation")
        self.assertEqual(adult, legacy)
        self.assertIn("Do not force advice", adult)

    def test_non_conversation_modes_are_unchanged(self):
        self.assertEqual(response_contract("planning", shell=Shell.SPROUT),
                         response_contract("planning"))
        self.assertEqual(response_contract("analysis", shell=Shell.SPROUT),
                         response_contract("analysis"))


if __name__ == "__main__":
    unittest.main()
