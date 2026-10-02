"""
Tests for Classical Arabic Poetry Lexicon.
"""

import unittest
from rawiai.nlp.lexicon import PoetryLexicon


class TestPoetryLexicon(unittest.TestCase):

    def setUp(self):
        self.lexicon = PoetryLexicon()

    def test_direct_lookup(self):
        entry = self.lexicon.lookup("العذل")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.root, "عذل")
        from rawiai.nlp.normalizer import ArabicNormalizer
        self.assertIn("اللوم", ArabicNormalizer.remove_tashkeel(entry.meaning))

    def test_annotate_verse(self):
        verse = "حكم سيوفك في رقاب العذل وإذا نزلت بدار ذل فارحل"
        annotations = self.lexicon.annotate_verse(verse)
        self.assertTrue(len(annotations) >= 1)
        roots = [a.root for a in annotations]
        self.assertIn("عذل", roots)

    def test_annotate_with_affixes(self):
        verse = "وسل السيف من غمده في الوغى"
        annotations = self.lexicon.annotate_verse(verse)
        roots = [a.root for a in annotations]
        self.assertIn("وغي", roots)


if __name__ == "__main__":
    unittest.main()
