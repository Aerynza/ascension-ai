"""Regression checks for canonical product-to-shell prompt mapping.

Internal enum values intentionally stay stable while customer-facing products
follow FamilyOS = household, Nexus = extended network.
"""
import unittest

from src.core.contracts import FOUNDATION, SHELL_CONTRACTS, Shell, Tier, system_contract


class CanonicalShellBrandContracts(unittest.TestCase):
    def test_household_shell_maps_to_familyos_without_privacy_expansion(self):
        self.assertEqual(Shell.NEXUS_HOME.value, "nexus_home")
        prompt = SHELL_CONTRACTS[Shell.NEXUS_HOME]
        self.assertIn("Operate as FamilyOS", prompt)
        self.assertIn("household", prompt)
        self.assertIn("explicitly shared household evidence", prompt)
        self.assertIn("private Human Thesis", prompt)
        self.assertIn("Membership changes must be authorized", prompt)
        self.assertNotIn("Operate as NexusHome", prompt)

    def test_extended_network_maps_to_nexus_and_keeps_member_consent(self):
        self.assertEqual(Shell.NEXUS_FAMILY.value, "nexus_family")
        prompt = SHELL_CONTRACTS[Shell.NEXUS_FAMILY]
        self.assertIn("Operate as Nexus", prompt)
        self.assertIn("extended family, friends, relationships", prompt)
        self.assertIn("explicitly consented to share with Nexus", prompt)
        self.assertIn("Never expose private household, child, or personal information", prompt)
        self.assertNotIn("Operate as NexusFamily", prompt)

    def test_sprout_retains_guardian_boundary(self):
        prompt = SHELL_CONTRACTS[Shell.SPROUT]
        self.assertIn("AerynzaSprout", prompt)
        self.assertIn("guardian-aware", prompt)
        self.assertIn("verified receipt", prompt)
        self.assertIn("child's request", prompt)

    def test_no_request_protocol_or_action_receipt_regression(self):
        self.assertIn("Understanding is not saving", FOUNDATION)
        self.assertIn("A proposal is not execution", FOUNDATION)
        prompt = system_contract(Shell.NEXUS_HOME, Tier.LIFE_OS, ["schedule"])
        self.assertIn("FamilyOS", prompt)
        self.assertIn("authenticated shell is the authority", prompt)


if __name__ == "__main__":
    unittest.main()
