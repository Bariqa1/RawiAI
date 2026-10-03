"""
Test Golden Dataset Coverage and Benchmark Metric Thresholds.
Verifies that all 50 evaluation test cases in golden_evaluation_dataset.json
meet production benchmark standards for Hit@1, MRR, Refusal Accuracy, and Relevancy.
"""

import unittest
import json
import os
from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.evaluation.retrieval_benchmark import RetrievalBenchmark


class TestGoldenCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "rawiai", "data", "golden_evaluation_dataset.json"
        )
        with open(dataset_path, "r", encoding="utf-8") as f:
            cls.test_cases = json.load(f)

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
        cls.benchmark = RetrievalBenchmark(cls.orchestrator, dataset_path=dataset_path)
        cls.eval_results = cls.benchmark.evaluate(top_k=5)
        cls.summary = cls.eval_results["summary"]

    def test_golden_dataset_has_50_cases(self):
        """Verifies golden dataset contains exactly 50 curated benchmark cases."""
        self.assertEqual(len(self.test_cases), 50)

    def test_all_four_intents_represented(self):
        """Verifies all four primary intents (author, completion, meaning, search) exist."""
        intents = {tc["intent"] for tc in self.test_cases}
        self.assertIn("author", intents)
        self.assertIn("completion", intents)
        self.assertIn("meaning", intents)
        self.assertIn("search", intents)

    def test_retrieval_hit_at_1_accuracy(self):
        """Verifies Hit@1 accuracy is at least 95% across the 50 test cases."""
        self.assertGreaterEqual(self.summary["hit@1"], 95.0)

    def test_mrr_score_threshold(self):
        """Verifies Mean Reciprocal Rank (MRR) is >= 0.95."""
        self.assertGreaterEqual(self.summary["mrr"], 0.95)

    def test_refusal_guardrail_accuracy(self):
        """Verifies out-of-domain and adversarial refusal accuracy is 100%."""
        self.assertEqual(self.summary["refusal_accuracy"], 100.0)

    def test_all_cases_evaluated(self):
        """Verifies summary accounts for all 50 test cases."""
        self.assertEqual(self.summary["total_test_cases"], 50)


if __name__ == "__main__":
    unittest.main()
