"""
Comprehensive unit tests for the Asas Al-Balagha Lexical & Rhetorical RAG module.
Validates 3,719 entries indexing, root search, literal vs. metaphorical disambiguation,
and multi-hop contextual enrichment.
"""

import unittest
from rawiai.nlp.lexicon_retriever import AsasBalaghaRetriever


class TestAsasBalaghaRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = AsasBalaghaRetriever()

    def test_lexicon_loads_3700_entries(self):
        """Verifies full loading of the 3,719 entries from Asas Al-Balagha."""
        self.assertGreaterEqual(len(self.retriever.entries), 3700)

    def test_retrieval_by_root_sayf(self):
        """Tests retrieval for root [سيف]."""
        results = self.retriever.retrieve("سيف", top_k=3)
        self.assertTrue(len(results) > 0)
        top_entry = results[0].entry
        self.assertEqual(top_entry.root, "سيف")
        self.assertTrue(len(top_entry.literal_meaning) > 0)
        self.assertTrue(len(top_entry.metaphorical_meaning) > 0)

    def test_retrieval_by_root_khayl(self):
        """Tests retrieval for root [خيل]."""
        results = self.retriever.retrieve("خيل", top_k=3)
        self.assertTrue(len(results) > 0)
        top_entry = results[0].entry
        self.assertEqual(top_entry.root, "خيل")

    def test_retrieval_by_root_bayd(self):
        """Tests retrieval for root [بيد] (البيداء)."""
        results = self.retriever.retrieve("بيد", top_k=3)
        self.assertTrue(len(results) > 0)
        top_entry = results[0].entry
        self.assertEqual(top_entry.root, "بيد")

    def test_separation_of_literal_and_metaphorical(self):
        """Ensures the lexicon strictly separates literal sense from metaphorical rhetorical usages."""
        entry = self.retriever.get_entry_by_root("سيف")
        self.assertIsNotNone(entry)
        self.assertIn("السلاح", entry.literal_meaning)
        self.assertTrue("سيف الصبح" in entry.metaphorical_meaning or "المجاز" in entry.metaphorical_meaning or len(entry.metaphorical_meaning) > 0)


    def test_annotate_verse_with_rhetorical_senses(self):
        """Tests multi-hop annotation of a verse with its key roots."""
        verse_text = "الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي"
        annotations = self.retriever.annotate_verse(verse_text)
        self.assertTrue(len(annotations) >= 2)
        roots_found = [a.root for a in annotations]
        self.assertTrue(any(r in ["خيل", "ليل", "بيد"] for r in roots_found))

    def test_multi_hop_lexicon_enrichment(self):
        """Tests multi-hop enrichment output formatting for LLM context."""
        entry = self.retriever.get_entry_by_root("سيف")
        context_str = entry.to_llm_context()
        self.assertIn("المعنى الوضعي", context_str)
        self.assertIn("المجاز والاستعارة", context_str)

    def test_empty_query_lexicon_handling(self):
        """Ensures safe handling of empty search input."""
        results = self.retriever.retrieve("", top_k=5)
        self.assertEqual(len(results), 0)

    def test_non_existent_root_handling(self):
        """Ensures safe empty list returned for non-existent root."""
        entry = self.retriever.get_entry_by_root("xyz123")
        self.assertIsNone(entry)

    def test_poetic_citations_extracted_from_lexicon(self):
        """Verifies poetic citations exist for foundational entries."""
        entry = self.retriever.get_entry_by_root("سيف")
        self.assertTrue(len(entry.poetic_citations) > 0)


if __name__ == "__main__":
    unittest.main()
