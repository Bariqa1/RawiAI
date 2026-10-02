"""
Tests for Asas Al-Balagha Lexicon Retriever.
"""

import unittest
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.nlp.normalizer import ArabicNormalizer


class TestAsasLexiconRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = AsasLexiconRetriever()

    def test_entries_loaded(self):
        self.assertGreater(len(self.retriever.entries), 10)

    def test_lookup_by_root(self):
        entry = self.retriever.lookup_by_root("عذل")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.lemma, "العَذْل")
        clean_meaning = ArabicNormalizer.remove_tashkeel(entry.literal_meaning)
        self.assertIn("اللوم", clean_meaning)
        self.assertIn("رقاب العُذَّل", entry.metaphorical_meaning)

    def test_lookup_by_lemma(self):
        entry = self.retriever.lookup_by_lemma("البيداء")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.root, "بيد")
        self.assertIn("الصحراء", entry.literal_meaning)

    def test_retrieve_metaphor(self):
        results = self.retriever.retrieve("استعارة الصحراء والبيداء", top_k=2)
        self.assertTrue(len(results) >= 1)
        found_roots = [r.entry.root for r in results]
        self.assertIn("بيد", found_roots)

    def test_annotate_verse(self):
        verse = "حكم سيوفك في رقاب العذل وإذا نزلت بدار ذل فارحل"
        entries = self.retriever.annotate_verse(verse)
        self.assertTrue(len(entries) >= 1)
        roots = [e.root for e in entries]
        self.assertIn("عذل", roots)
        self.assertIn("سيف", roots)


if __name__ == "__main__":
    unittest.main()
