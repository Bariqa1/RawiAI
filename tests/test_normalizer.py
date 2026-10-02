"""
Tests for Arabic Normalizer.
"""

import unittest
from rawiai.nlp.normalizer import ArabicNormalizer


class TestArabicNormalizer(unittest.TestCase):

    def test_remove_tashkeel(self):
        vocalized = "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ"
        stripped = ArabicNormalizer.remove_tashkeel(vocalized)
        self.assertEqual(stripped, "حكم سيوفك في رقاب العذل")

    def test_remove_tatweel(self):
        stretched = "ســــيوفك فــــي رقــــاب"
        cleaned = ArabicNormalizer.remove_tatweel(stretched)
        self.assertEqual(cleaned, "سيوفك في رقاب")

    def test_normalize_search(self):
        text = "إِذَا أَنْتَ أَكْرَمْتَ الكَرِيمَ مَلَكْتَهُ... وَإِنْ أَنْتَ أَكْرَمْتَ اللَّئِيمَ تَمَرَّدَا!"
        norm = ArabicNormalizer.normalize_search(text)
        # Should unify Alef to ا, remove diacritics and punctuation
        self.assertIn("اذا", norm)
        self.assertIn("انت", norm)
        self.assertIn("اكرمت", norm)
        self.assertNotIn("!", norm)
        self.assertNotIn("...", norm)

    def test_normalize_phonetic_rhyme(self):
        # Keeps distinction for Taa Marbuta and Alif Maqsura
        text = "فَلاَ تَحْسَبَنَّ الحَرْبَ أُمَّاً حَنُونَةً"
        rhyme_norm = ArabicNormalizer.normalize_phonetic_rhyme(text)
        self.assertTrue(rhyme_norm.endswith("حنونة") or rhyme_norm.endswith("حنونةً") or "حنونة" in rhyme_norm)

    def test_has_arabic(self):
        self.assertTrue(ArabicNormalizer.has_arabic("شعر عربي"))
        self.assertTrue(ArabicNormalizer.has_arabic("123 بيت"))
        self.assertFalse(ArabicNormalizer.has_arabic("English only text 123!"))


if __name__ == "__main__":
    unittest.main()
