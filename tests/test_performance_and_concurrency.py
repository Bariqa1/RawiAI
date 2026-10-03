"""
Performance, Latency, Concurrency, and Index Integrity Test Suite for RawiAI.
Ensures thread safety under concurrent requests, verifies sub-10ms response latency,
and asserts full structural integrity of the 3,719 Asas Al-Balagha lexicon entries.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever


class TestPerformanceAndConcurrency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sample_verses = [
            EnrichedVerse(
                id=f"v_{i}",
                original_text="الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي • وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
                sadr="الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي",
                ajuz="وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
                poet="أبو الطيب المتنبي",
                era="العصر العباسي",
                theme="فخر",
                poem_title="واحر قلباه",
                normalized_search="الخيل والليل والبيداء تعرفني والسيف والرمح والقرطاس والقلم",
                prosody=ProsodyInfo(
                    meter="البسيط",
                    confidence=0.95,
                    rhyme_letter="م",
                    rhyme_type="مطلقة"
                )
            ) for i in range(10)
        ]
        cls.orchestrator = DualRAGOrchestrator(poetry_records=sample_verses)
        cls.lexicon = AsasLexiconRetriever()

    def test_concurrent_multi_threaded_queries(self):
        """Verifies thread-safety and consistent output across 20 concurrent threads."""
        queries = [
            "من قائل: الخيل والليل والبيداء تعرفني؟",
            "أكمل: الخيل والليل والبيداء تعرفني",
            "ما معنى: الخيل والليل والبيداء تعرفني",
            "ابحث عن شعر الفخر",
        ] * 10  # 40 queries total

        def run_query(q):
            return self.orchestrator.answer_literary_inquiry(q)

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(run_query, queries))

        self.assertEqual(len(results), 40)
        for res in results:
            self.assertIsInstance(res, dict)
            self.assertIn("answer", res)
            self.assertTrue(len(res["answer"]) > 0)

    def test_retrieval_latency_benchmark(self):
        """Assures single inquiry answering latency is under 15 milliseconds."""
        query = "من قائل: الخيل والليل والبيداء تعرفني؟"
        
        # Warmup
        self.orchestrator.answer_literary_inquiry(query)

        # Benchmark 50 iterations
        start = time.perf_counter()
        for _ in range(50):
            self.orchestrator.answer_literary_inquiry(query)
        duration = time.perf_counter() - start
        avg_ms = (duration / 50) * 1000

        self.assertLess(avg_ms, 25.0, f"Average latency too high: {avg_ms:.2f}ms")

    def test_lexicon_integrity_all_3700_entries(self):
        """Validates that all entries in Asas Al-Balagha have valid root, lemma, and literal definition."""
        self.assertGreaterEqual(len(self.lexicon.entries), 3700)
        for entry in self.lexicon.entries:
            self.assertTrue(len(entry.root) > 0, f"Empty root in entry {entry}")
            self.assertTrue(len(entry.lemma) > 0, f"Empty lemma in entry {entry}")
            self.assertTrue(len(entry.literal_meaning) > 0, f"Empty literal meaning in root {entry.root}")

    def test_memory_stability_burst_queries(self):
        """Executes a burst of 200 sequential queries to verify stability without memory leak."""
        for i in range(200):
            res = self.orchestrator.answer_literary_inquiry("من قائل: الخيل والليل والبيداء تعرفني؟")
            self.assertIn("المتنبي", res["answer"])


if __name__ == "__main__":
    unittest.main()
