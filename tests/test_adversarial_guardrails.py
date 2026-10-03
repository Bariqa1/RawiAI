"""
Adversarial & Guardrail Security Tests for RawiAI Dual-RAG.
Verifies strictly enforced zero-hallucination refusal, out-of-domain rejection,
tatweel/kashida resilience, and punctuation noise robustness.
"""

import unittest
from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.nlp.normalizer import ArabicNormalizer


class TestAdversarialGuardrails(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sample_verse = EnrichedVerse(
            id="v1",
            original_text="حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            sadr="حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ",
            ajuz="وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            poet="عنترة بن شداد",
            era="العصر الجاهلي",
            theme="شجاعة وعزة",
            poem_title="معلقة عنترة",
            normalized_search="حكم سيوفك في رقاب العذل واذا نزلت بدار ذل فارحل",
            prosody=ProsodyInfo(
                phonetic_sadr="",
                phonetic_ajuz="",
                meter="الطويل",
                meter_confidence=0.95,
                rhyme_letter="ل"
            )
        )
        cls.orchestrator = DualRAGOrchestrator(poetry_records=[sample_verse])

    def test_refusal_for_modern_technology_query(self):
        """Rejects modern anachronistic queries (smartphones/spaceships)."""
        q = "ما هي الأبيات التي كتبها المتنبي عن ركوب الطائرات والسفر بالفضاء؟"
        ans = self.orchestrator.answer_literary_inquiry(q)
        self.assertIn("غير متوفرة", ans["answer"])
        self.assertEqual(len(ans["retrieved_verses"]), 0)

    def test_refusal_for_fictional_poet_query(self):
        """Rejects queries mentioning fictional or non-existent poets."""
        q = "من الشاعر الملقب بأبي الفلافل في العصر الجاهلي؟"
        ans = self.orchestrator.answer_literary_inquiry(q)
        self.assertIn("غير متوفرة", ans["answer"])
        self.assertEqual(len(ans["retrieved_verses"]), 0)

    def test_refusal_for_modern_programming_query(self):
        """Rejects modern tech concepts (Python programming, websites)."""
        q = "أريد قصيدة عن لغة بايثون وتطوير مواقع الويب"
        ans = self.orchestrator.answer_literary_inquiry(q)
        self.assertIn("غير متوفرة", ans["answer"])

    def test_refusal_for_nonsense_gibberish(self):
        """Rejects nonsensical character sequences."""
        q = "من قائل: سشيبكستنم ويكسنتنب صثقفغع؟"
        ans = self.orchestrator.answer_literary_inquiry(q)
        self.assertIn("غير متوفرة", ans["answer"])

    def test_tatweel_kashida_resilience(self):
        """Ensures queries with Arabic tatweel (ـ) are normalized and retrieved correctly."""
        q_tatweel = "من قائل: حـــكِّـــمْ سُـــيُـــوفَـــكَ فِي رِقَابِ العُذَّلِ؟"
        norm = ArabicNormalizer.normalize_search(q_tatweel)
        self.assertNotIn("ـ", norm)
        ans = self.orchestrator.answer_literary_inquiry(q_tatweel)
        self.assertIn("عنترة", ans["answer"])

    def test_punctuation_noise_resilience(self):
        """Ensures search works despite heavy exclamation/question marks or symbols."""
        q_noisy = "من قائل: حَكِّم سُيُوفَكَ في رِقَابِ العُذَّلِ؟!؟!!؟!؟***"
        ans = self.orchestrator.answer_literary_inquiry(q_noisy)
        self.assertIn("عنترة", ans["answer"])

    def test_short_ambiguous_queries(self):
        """Ensures single character or extremely short queries do not crash."""
        ans = self.orchestrator.answer_literary_inquiry("؟")
        self.assertIn("غير متوفرة", ans["answer"])

    def test_diacritics_stripping(self):
        """Verifies ArabicNormalizer removes all diacritics including tanween, shaddah, sukoon."""
        vocalized = "فَوَدِدْتُ تَقْبِيلَ السُّيُوفِ لأَنَّهَا"
        stripped = ArabicNormalizer.strip_tashkeel(vocalized)
        self.assertNotIn("َ", stripped)
        self.assertNotIn("ُ", stripped)
        self.assertNotIn("ّ", stripped)
        self.assertNotIn("ْ", stripped)

    def test_empty_string_safety(self):
        """Ensures passing empty string is handled gracefully without exception."""
        ans = self.orchestrator.answer_literary_inquiry("")
        self.assertIn("غير متوفرة", ans["answer"])

    def test_intent_classification_safety(self):
        """Ensures intent classification returns valid default 'search' for unknown patterns."""
        intent = self.orchestrator.classify_intent("كلام عام جداً بدون أي كلمة مفتاحية")
        self.assertEqual(intent, "search")


if __name__ == "__main__":
    unittest.main()
