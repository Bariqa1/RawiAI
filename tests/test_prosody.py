"""
Tests for Prosody & Rhyme Analyzer.
"""

import unittest
from rawiai.nlp.prosody import ProsodyAnalyzer


class TestProsodyAnalyzer(unittest.TestCase):

    def test_extract_rawiyy(self):
        cases = [
            ("وإذا نزلت بدار ذل فارحل", "ل", "مقيّدة"),
            ("في رقاب العذل", "ل", "مقيّدة"),
            ("فتكلمَا", "م", "مطلقة"),         # Alif of Itlaaq
            ("ورأيت سيوفَها", "ف", "مطلقة"),      # Pronoun suffix 'ها'
            ("من ذكرى حبيب ومنزل", "ل", "مقيّدة"),
            ("فاصبحِينَا", "ن", "مطلقة"),        # Suffix 'نا'
        ]

        for ajuz, expected_rawiyy, _ in cases:
            rawiyy, rhyme_type = ProsodyAnalyzer.extract_rawiyy(ajuz)
            self.assertEqual(rawiyy, expected_rawiyy, f"Failed for ajuz: '{ajuz}' - expected '{expected_rawiyy}', got '{rawiyy}'")

    def test_estimate_meter(self):
        # Al-Taweel example:
        # قفا نبك من ذكرى حبيب ومنزل ... بسقط اللوى بين الدخول فحومل
        sadr = "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"
        ajuz = "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
        meter, tafail, conf = ProsodyAnalyzer.estimate_meter(sadr, ajuz)
        self.assertIn("الطويل", meter)
        self.assertGreater(conf, 0.4)


if __name__ == "__main__":
    unittest.main()
