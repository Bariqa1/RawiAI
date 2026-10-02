"""
Tests for Dual-RAG Multi-Hop Orchestrator.
"""

import unittest
from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator


class TestDualRAGOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orchestrator = DualRAGOrchestrator()
        self.sample_verse = EnrichedVerse(
            id="v1",
            original_text="حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            sadr="حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ",
            ajuz="وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            normalized_search="حكم سيوفك في رقاب العذل واذا نزلت بدار ذل فارحل",
            stemmed_tokens=["حكم", "سيف", "رقاب", "عذل", "دار", "ذل", "ارحل"],
            prosody=ProsodyInfo(meter="الطويل", rhyme_letter="ل", confidence=0.85),
            poet="عنترة بن شداد",
            era="جاهلي",
            theme="فخر وشجاعة"
        )

    def test_classify_intent(self):
        self.assertEqual(self.orchestrator.classify_intent("ما معنى حكم سيوفك؟"), "meaning")
        self.assertEqual(self.orchestrator.classify_intent("اشرح البلاغة في هذا البيت"), "meaning")
        self.assertEqual(self.orchestrator.classify_intent("من قائل: الخيل والليل؟"), "author")
        self.assertEqual(self.orchestrator.classify_intent("أكمل البيت: إذا المرء لم يدنس"), "completion")
        self.assertEqual(self.orchestrator.classify_intent("أبيات عن الشجاعة والكرم"), "search")

    def test_clean_search_query(self):
        cleaned = self.orchestrator.clean_search_query("ما معنى: حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ؟")
        self.assertEqual(cleaned, "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ")

    def test_dual_rag_context_generation(self):
        context = self.orchestrator.build_dual_rag_context(
            query="ما معنى حكم سيوفك في رقاب العذل؟",
            retrieved_verses=[self.sample_verse],
            intent="meaning"
        )
        # Should include Poetry info
        self.assertIn("عنترة بن شداد", context)
        self.assertIn("جاهلي", context)
        self.assertIn("حرف ل", context)
        # Should include Asas Al-Balagha rhetoric section
        self.assertIn("أساس البلاغة", context)
        self.assertIn("الزمخشري", context)
        self.assertIn("المعنى الحقيقي", context)
        self.assertIn("الاستعمال المجازي", context)
        self.assertIn("العَذْل", context)

    def test_formulate_prompt(self):
        prompts = self.orchestrator.formulate_grounded_prompt(
            query="اشرح بلاغة: حكّم سيوفك في رقاب العذل",
            retrieved_verses=[self.sample_verse],
            intent="meaning"
        )
        self.assertIn("system_prompt", prompts)
        self.assertIn("user_prompt", prompts)
        self.assertIn("أساس البلاغة", prompts["system_prompt"])
        self.assertIn("الصور البيانية", prompts["user_prompt"])


if __name__ == "__main__":
    unittest.main()
