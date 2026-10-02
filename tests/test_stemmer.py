"""
Tests for Poetry-Aware Arabic Stemmer.
"""

import unittest
from rawiai.nlp.stemmer import PoetryArabicStemmer


class TestPoetryStemmer(unittest.TestCase):

    def test_strip_compound_prefixes(self):
        self.assertEqual(PoetryArabicStemmer.stem_word("والسيف"), "سيف")
        self.assertEqual(PoetryArabicStemmer.stem_word("بالحق"), "حق")
        self.assertEqual(PoetryArabicStemmer.stem_word("فالعدل"), "عدل")
        self.assertEqual(PoetryArabicStemmer.stem_word("كالرياح"), "رياح")
        self.assertEqual(PoetryArabicStemmer.stem_word("للناس"), "ناس")

    def test_radical_protection(self):
        self.assertEqual(PoetryArabicStemmer.stem_word("ولد"), "ولد")
        self.assertEqual(PoetryArabicStemmer.stem_word("كتب"), "كتب")
        self.assertEqual(PoetryArabicStemmer.stem_word("بلاد"), "بلاد")
        self.assertEqual(PoetryArabicStemmer.stem_word("يد"), "يد")
        self.assertEqual(PoetryArabicStemmer.stem_word("دم"), "دم")

    def test_strip_suffixes(self):
        self.assertEqual(PoetryArabicStemmer.stem_word("سيوفها"), "سيف")
        self.assertEqual(PoetryArabicStemmer.stem_word("رقابهم"), "رقاب")
        # عيونك stems to singular canonical 'عين'
        self.assertIn(PoetryArabicStemmer.stem_word("عيونك"), ["عين", "عيون"])

    def test_tokenize_and_stem(self):
        verse = "حكم سيوفك في رقاب العذل وإذا نزلت بدار ذل فارحل"
        stems = PoetryArabicStemmer.tokenize_and_stem(verse, remove_stopwords=True)
        # Stopwords like 'في', 'اذا' should be removed
        self.assertNotIn("في", stems)
        self.assertNotIn("اذا", stems)
        # Content words should be stemmed
        self.assertIn("سيف", stems)
        self.assertIn("عذل", stems)


if __name__ == "__main__":
    unittest.main()
