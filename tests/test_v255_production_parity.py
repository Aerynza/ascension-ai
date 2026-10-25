"""Independent v255 production-parity curriculum and holdout safety contract."""
import unittest

from scripts.build_next_gpu_packs import build, heldout_set, normalize, read_jsonl, SOURCES


class V255CurriculumContract(unittest.TestCase):
    def test_all_v255_cases_enter_training_without_using_independent_holdout(self):
        rows, manifest = build()
        train = read_jsonl(SOURCES["v255_parity_train"])
        heldout = read_jsonl(SOURCES["v255_parity_heldout"])
        self.assertEqual(len(train), 36)
        self.assertEqual(len(heldout), 18)
        corpus_ids = {row["id"] for row in rows}
        self.assertEqual(len(corpus_ids), len(rows))
        self.assertTrue({row["id"] for row in train}.issubset(corpus_ids))
        self.assertFalse({row["id"] for row in heldout} & corpus_ids)
        self.assertTrue({normalize(row["user"]) for row in heldout}.issubset(heldout_set()))
        self.assertFalse({normalize(row["user"]) for row in heldout} &
                         {normalize(row["user"]) for row in rows})
        self.assertEqual(manifest["catalog_training_coverage"], 640)
        self.assertFalse(manifest["production_replacement_ready"])
        self.assertTrue(manifest["human_review_required"])

    def test_canon_scopes_and_provenance_survive_compilation(self):
        rows, manifest = build()
        v255 = [row for row in rows if row.get("source") == "v255_production_parity"]
        self.assertEqual(len(v255), 36)
        shells = {row["shell"] for row in v255}
        for shell in ("ap", "lifeos", "executive", "sprout", "nexus_home", "nexus_family"):
            self.assertIn(shell, shells)
        self.assertNotIn("core", shells)
        self.assertTrue(all(row.get("review_status") == "pending_human_review"
                            and row.get("source_kind") == "ai_authored_synthetic" for row in v255))
        self.assertIn("v255_parity_train", manifest["source_sha256"])
        self.assertIn("v255_parity_heldout", manifest["source_sha256"])
        self.assertIn("ngpu_v255_generalist", manifest["package_counts"])

    def test_heldout_is_never_an_approved_training_source(self):
        self.assertNotEqual(SOURCES["v255_parity_train"], SOURCES["v255_parity_heldout"])
        for row in read_jsonl(SOURCES["v255_parity_heldout"]):
            self.assertEqual(row["reserved_for"], "independent_evaluation_never_train")
            self.assertEqual(row["review_status"], "pending_human_review")


if __name__ == "__main__":
    unittest.main()
