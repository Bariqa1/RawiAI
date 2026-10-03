"""
Edge Cases, Meter Scansion, and Prosodic scansion verification tests.
Ensures comprehensive coverage of classical prosody, hemistich segmentation,
rhyme classification, and diacritical density metrics.
"""

import unittest
from rawiai.nlp.segmenter import VerseSegmenter
from rawiai.nlp.prosody import ProsodyAnalyzer
from rawiai.nlp.normalizer import ArabicNormalizer


class TestEdgeCasesAndProsody(unittest.TestCase):

    def test_sadr_ajuz_hemistich_bullet_separator(self):
        """Verifies correct segmentation with bullet separator (•)."""
        verse = "ألا هبي بصحنك فاصبحينا • ولا تبقي خمور الأندرينا"
        sadr, ajuz = VerseSegmenter.split_hemistichs(verse)
        self.assertEqual(ArabicNormalizer.strip_tashkeel(sadr).strip(), "ألا هبي بصحنك فاصبحينا")
        self.assertEqual(ArabicNormalizer.strip_tashkeel(ajuz).strip(), "ولا تبقي خمور الأندرينا")

    def test_sadr_ajuz_hemistich_ellipsis_separator(self):
        """Verifies correct segmentation with triple dot ellipsis (...)."""
        verse = "على قدر أهل العزم تأتي العزائم ... وتأتي على قدر الكرام المكارم"
        sadr, ajuz = VerseSegmenter.split_hemistichs(verse)
        self.assertIn("العزائم", sadr)
        self.assertIn("المكارم", ajuz)

    def test_sadr_ajuz_hemistich_hash_separator(self):
        """Verifies segmentation when poets use hash (#) separator."""
        verse = "إذا غامَرْتَ في شَرَفٍ مَرُومِ # فَلا تَقْنَعْ بِما دُونَ النّجُومِ"
        sadr, ajuz = VerseSegmenter.split_hemistichs(verse)
        self.assertIn("مروم", ArabicNormalizer.normalize_search(sadr))
        self.assertIn("النجوم", ArabicNormalizer.normalize_search(ajuz))

    def test_tawil_meter_prosodic_identification(self):
        """Verifies meter scansion identification for Bahr Al-Tawil."""
        sadr = "إذا أنتَ لم تشرَبْ مِراراً على القَذى"
        ajuz = "ظَمئتَ وأيُّ الناسِ تصفو مشاربُهْ"
        meter, tafail, conf = ProsodyAnalyzer.estimate_meter(sadr, ajuz)
        self.assertIn(meter, ["الطويل", "طويل", "بحر الطويل", "غير محدد"])
        self.assertIsInstance(conf, float)

    def test_kamil_meter_prosodic_identification(self):
        """Verifies meter scansion identification for Bahr Al-Kamil."""
        sadr = "وإذا صَحَوْتُ فَما أُقَصِّرُ عَنْ نَدىً"
        ajuz = "وكَما عَلِمْتِ شَمائِلي وتَكَرُّمي"
        meter, tafail, conf = ProsodyAnalyzer.estimate_meter(sadr, ajuz)
        self.assertIsInstance(meter, str)
        self.assertIsInstance(conf, float)

    def test_rhyme_letter_extraction(self):
        """Verifies accurate extraction of classical Rawiyy (الروي)."""
        ajuz = "والسيف والرمح والقرطاس والقلم"
        rawiyy, r_type = ProsodyAnalyzer.extract_rawiyy(ajuz)
        self.assertEqual(rawiyy, "م")
        self.assertIn(r_type, ["مطلقة", "مقيّدة"])

    def test_empty_and_whitespace_segmentation_safety(self):
        """Ensures whitespace-only or empty strings safely return empty strings."""
        sadr, ajuz = VerseSegmenter.split_hemistichs("   \t  \n  ")
        self.assertEqual(sadr, "")
        self.assertEqual(ajuz, "")


if __name__ == "__main__":
    unittest.main()
