"""Aerynza AI public branding contract (legacy runtime IDs remain stable)."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AerynzaAIBrandContractTests(unittest.TestCase):
    def test_npm_package_and_lockfile_agree(self):
        package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT / "package-lock.json").read_text(encoding="utf-8"))
        self.assertEqual("aerynza-ai", package["name"])
        self.assertEqual(package["name"], lock["name"])
        self.assertEqual(package["name"], lock["packages"][""]["name"])

    def test_readme_identity(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(readme.startswith("# Aerynza AI\n"))
        self.assertIn("AerynzaLife", readme)
        self.assertIn("AerynzaSprout", readme)
        self.assertIn("FamilyOS", readme)
        self.assertIn("Nexus", readme)

    def test_public_shell_labels_and_protocol_stability(self):
        page = (ROOT / "public/index.html").read_text(encoding="utf-8")
        for shell, label in (
            ("nexus_home", "FamilyOS · Household"),
            ("nexus_family", "Nexus · Extended network"),
            ("sprout", "AerynzaSprout · Learning"),
        ):
            self.assertIn(f'<option value="{shell}">{label}</option>', page)
        self.assertIn("<strong>AERYNZA AI</strong>", page)

    def test_legacy_environment_contract_retained(self):
        blueprint = (ROOT / "render.yaml").read_text(encoding="utf-8")
        self.assertIn("ASCENSION_AI_SERVICE_TOKEN", blueprint)
        self.assertIn("ASCENSION_AI_AUTH_MODE", blueprint)


if __name__ == "__main__":
    unittest.main()
