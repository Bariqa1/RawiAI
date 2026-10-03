"""
Comprehensive unit tests for the Verse Attribution and Q&A Council RAG module.
Validates poet identification, hemistich completion, era metadata, and search normalization.
"""

import unittest
import json
import os
from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.nlp.normalizer import ArabicNormalizer


class TestAttributionRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        corpus_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "rawiai", "data", "classical_poetry_corpus.json"
        )
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        cls.corpus = []
        for item in raw_data:
            p = item.get("prosody", {})
            prosody = ProsodyInfo(
                phonetic_sadr="",
                phonetic_ajuz="",
                meter=p.get("meter", "غير محدد"),
                meter_confidence=p.get("confidence", 0.9),
                rhyme_letter=p.get("rhyme_letter", "")
            )
            v = EnrichedVerse(
                id=item.get("id", "verse_1"),
                original_text=item.get("original_text", ""),
                sadr=item.get("sadr", ""),
                ajuz=item.get("ajuz", ""),
                poet=item.get("poet", ""),
                era=item.get("era", ""),
                theme=item.get("theme", ""),
                poem_title=item.get("poem_title", ""),
                normalized_search=item.get("normalized_search", ""),
                stemmed_tokens=item.get("stemmed_tokens", []),
                prosody=prosody,
                difficult_words=item.get("difficult_words", []),
                source_row=item.get("source_row")
            )
            cls.corpus.append(v)

        cls.orchestrator = DualRAGOrchestrator(poetry_records=cls.corpus)

    def test_attribution_al_mutanabbi(self):
        """Attribution test for Al-Mutanabbi iconic verse."""
        query = "من قائل: الخيل والليل والبيداء تعرفني؟"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertIn("المتنبي", ans["answer"])
        self.assertEqual(ans["intent"], "author")
        self.assertTrue(len(ans["retrieved_verses"]) > 0)
        self.assertEqual(ans["retrieved_verses"][0].poet, "أبو الطيب المتنبي")

    def test_attribution_antarah(self):
        """Attribution test for Antarah ibn Shaddad."""
        query = "من قائل: حكم سيوفك في رقاب العذل؟"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertIn("عنترة", ans["answer"])
        self.assertEqual(ans["intent"], "author")
        self.assertEqual(ans["retrieved_verses"][0].poet, "عنترة بن شداد")

    def test_attribution_imru_al_qais(self):
        """Attribution test for Imru' al-Qais mu'allaqah."""
        query = "من قائل: قفا نبك من ذكرى حبيب ومنزل؟"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertIn("امرؤ القيس", ans["answer"])
        self.assertEqual(ans["retrieved_verses"][0].era, "العصر الجاهلي")

    def test_attribution_al_shafi(self):
        """Attribution test for Imam Al-Shafi'i wisdom poetry."""
        query = "من صاحب بيت: دع الأيام تفعل ما تشاء؟"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertIn("الشافعي", ans["answer"])
        self.assertEqual(ans["retrieved_verses"][0].poet, "الإمام الشافعي")

    def test_attribution_al_shabbi(self):
        """Attribution test for Abu Al-Qasim Al-Shabbi."""
        query = "من قائل: إذا الشعب يوما أراد الحياة؟"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertIn("الشابي", ans["answer"])
        self.assertEqual(ans["retrieved_verses"][0].era, "العصر الحديث")

    def test_attribution_ahmad_shawqi(self):
        """Attribution test for Ahmad Shawqi."""
        query = "من صاحب البيت: قم للمعلم وفه التبجيلا؟"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertIn("شوقي", ans["answer"])

    def test_attribution_al_khansa(self):
        """Attribution test for Al-Khansa elegy."""
        query = "من القائل: وإن صخرا لتأتم الهداة به؟"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertIn("الخنساء", ans["answer"])

    def test_completion_sadr_to_ajuz(self):
        """Hemistich completion test."""
        query = "أكمل: قفا نبك من ذكرى حبيب ومنزل"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertEqual(ans["intent"], "completion")
        self.assertIn("بسقط اللوى", ArabicNormalizer.strip_tashkeel(ans["answer"]))

    def test_completion_mutanabbi(self):
        """Verse continuation test for Al-Mutanabbi."""
        query = "أكمل البيت: الخيل والليل والبيداء تعرفني"
        ans = self.orchestrator.answer_literary_inquiry(query)
        self.assertEqual(ans["intent"], "completion")
        self.assertIn("السيف والرمح", ArabicNormalizer.strip_tashkeel(ans["answer"]))

    def test_normalization_invariance_diacritics(self):
        """Ensures query matching is identical with or without full tashkeel."""
        q_vocalized = "من قائل: الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي؟"
        q_plain = "من قائل: الخيل والليل والبيداء تعرفني؟"
        res_vocalized = self.orchestrator.answer_literary_inquiry(q_vocalized)
        res_plain = self.orchestrator.answer_literary_inquiry(q_plain)
        self.assertEqual(res_vocalized["retrieved_verses"][0].id, res_plain["retrieved_verses"][0].id)

    def test_normalization_invariance_alif_variants(self):
        """Ensures query matching is resilient across Alif variations (أ، إ، آ، ا)."""
        q1 = "إذا المرء لم يدنس"
        q2 = "اذا المرء لم يدنس"
        clean1 = ArabicNormalizer.normalize_search(q1)
        clean2 = ArabicNormalizer.normalize_search(q2)
        self.assertEqual(clean1, clean2)

    def test_clean_search_query_prefixes(self):
        """Tests removing prefixes like 'من قائل', 'أكمل', 'ما معنى'."""
        self.assertEqual(self.orchestrator.clean_search_query("من قائل: حكم سيوفك"), "حكم سيوفك")
        self.assertEqual(self.orchestrator.clean_search_query("أكمل البيت: قفا نبك"), "قفا نبك")
        self.assertEqual(self.orchestrator.clean_search_query("ما معنى: الخيل والليل"), "الخيل والليل")


if __name__ == "__main__":
    unittest.main()
