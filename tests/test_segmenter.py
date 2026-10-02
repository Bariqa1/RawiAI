"""
Tests for Verse Segmenter.
"""

import unittest
from rawiai.nlp.segmenter import VerseSegmenter


class TestVerseSegmenter(unittest.TestCase):

    def test_split_explicit_separators(self):
        cases = [
            ("حكم سيوفك في رقاب العذل # وإذا نزلت بدار ذل فارحل", "حكم سيوفك في رقاب العذل", "وإذا نزلت بدار ذل فارحل"),
            ("الخيل والليل والبيداء تعرفني ... والسيف والرمح والقرطاس والقلم", "الخيل والليل والبيداء تعرفني", "والسيف والرمح والقرطاس والقلم"),
            ("على قدر أهل العزم تأتي العزائم   وتأتي على قدر الكرام المكارم", "على قدر أهل العزم تأتي العزائم", "وتأتي على قدر الكرام المكارم"),
            ("أنا الذي نظر الأعمى إلى أدبي / وأسمعت كلماتي من به صمم", "أنا الذي نظر الأعمى إلى أدبي", "وأسمعت كلماتي من به صمم"),
        ]

        for line, expected_sadr, expected_ajuz in cases:
            sadr, ajuz = VerseSegmenter.split_hemistichs(line)
            self.assertEqual(sadr, expected_sadr)
            self.assertEqual(ajuz, expected_ajuz)

    def test_clean_verse_prefix_numbers(self):
        line = "1- ولد الهدى فالكائنات ضياء ... وفم الزمان تبسم وثناء"
        cleaned = VerseSegmenter.clean_verse_text(line)
        self.assertFalse(cleaned.startswith("1-"))
        self.assertTrue(cleaned.startswith("ولد الهدى"))

        line2 = "(١٢) قفا نبك من ذكرى حبيب ومنزل"
        cleaned2 = VerseSegmenter.clean_verse_text(line2)
        self.assertTrue(cleaned2.startswith("قفا نبك"))

    def test_extract_verses_from_poem(self):
        poem = """
        ديوان المتنبي
        الخيل والليل والبيداء تعرفني ... والسيف والرمح والقرطاس والقلم
        صحوت فما يغرني سراب ... ونمت فما يؤرقني منام
        """
        verses = VerseSegmenter.extract_verses_from_poem(poem)
        # Should extract 2 valid verses and skip the short title 'ديوان المتنبي'
        self.assertEqual(len(verses), 2)
        self.assertIn("الخيل", verses[0][0])
        self.assertIn("صحوت", verses[1][0])


if __name__ == "__main__":
    unittest.main()
