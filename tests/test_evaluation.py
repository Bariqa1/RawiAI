"""
Unit tests for the Evaluation module (Retrieval Benchmark, DeepEval runner, and metrics).
"""

import unittest
import os
import json
from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.evaluation.retrieval_benchmark import RetrievalBenchmark
from rawiai.evaluation.deepeval_runner import DeepEvalRunner
from rawiai.evaluation.metrics import (
    PoeticFaithfulnessMetric,
    PoeticRelevancyMetric,
    PoeticRefusalMetric
)

try:
    from deepeval.test_case import LLMTestCase
except ImportError:
    class LLMTestCase:
        def __init__(self, **kw):
            self.__dict__.update(kw)


class TestEvaluationSuite(unittest.TestCase):

    def setUp(self):
        self.sample_verse = EnrichedVerse(
            id="verse_1",
            original_text="حكم سيوفك في رقاب العذل ... وإذا نزلت بدار ذل فارحل",
            sadr="حكم سيوفك في رقاب العذل",
            ajuz="وإذا نزلت بدار ذل فارحل",
            poet="عنترة بن شداد",
            era="جاهلي",
            theme="شجاعة",
            poem_title="معلقة عنترة",
            normalized_search="حكم سيوفك في رقاب العذل واذا نزلت بدار ذل فارحل",
            prosody=ProsodyInfo(
                phonetic_sadr="101010",
                phonetic_ajuz="101010",
                meter="الكامل",
                meter_confidence=0.95,
                rhyme_letter="ل"
            )
        )
        self.orchestrator = DualRAGOrchestrator(poetry_records=[self.sample_verse])
        self.runner = DeepEvalRunner(self.orchestrator)

    def test_golden_dataset_structure(self):
        """Ensures golden dataset has required schema fields."""
        dataset_path = self.runner.dataset_path
        self.assertTrue(os.path.exists(dataset_path))
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertGreaterEqual(len(data), 5)
        for item in data:
            self.assertIn("id", item)
            self.assertIn("query", item)
            self.assertIn("intent", item)
            self.assertIn("ground_truth", item)
            self.assertIn(item["intent"], ["meaning", "author", "completion", "search"])

    def test_poetic_faithfulness_metric_grounded(self):
        """Tests that well-grounded outputs score high."""
        metric = PoeticFaithfulnessMetric(threshold=0.7)
        tc = LLMTestCase(
            input="ما معنى حكم سيوفك في رقاب العذل؟",
            actual_output="معنى حكم سيوفك في رقاب العذل للشاعر عنترة بن شداد",
            retrieval_context=["البيت: حكم سيوفك في رقاب العذل الشاعر: عنترة بن شداد العصر: جاهلي"]
        )
        score = metric.measure(tc)
        self.assertGreaterEqual(score, 0.7)
        self.assertTrue(metric.is_successful())

    def test_poetic_faithfulness_metric_refusal(self):
        """Tests that safe refusal scores 1.0 when no context is available."""
        metric = PoeticFaithfulnessMetric()
        tc = LLMTestCase(
            input="من قائل هذا البيت المعاصر الخيالي؟",
            actual_output="المعلومة المطلوبة غير متوفرة في قاعدة بيانات الشعر.",
            retrieval_context=["لا توجد شواهد أو أبيات مطابقة في قاعدة البيانات لهذا الاستعلام."]
        )
        score = metric.measure(tc)
        self.assertEqual(score, 1.0)
        self.assertTrue(metric.is_successful())

    def test_poetic_relevancy_metric_author(self):
        """Tests author question relevancy scoring."""
        metric = PoeticRelevancyMetric(threshold=0.65)
        tc = LLMTestCase(
            input="من قائل حكم سيوفك؟",
            actual_output="الشاعر القائل هو: عنترة بن شداد (جاهلي).",
            expected_output="عنترة بن شداد"
        )
        score = metric.measure(tc)
        self.assertEqual(score, 1.0)
        self.assertTrue(metric.is_successful())

    def test_poetic_relevancy_metric_completion(self):
        """Tests completion question relevancy scoring."""
        metric = PoeticRelevancyMetric(threshold=0.65)
        tc = LLMTestCase(
            input="أكمل البيت: حكم سيوفك في رقاب العذل",
            actual_output="تكملة البيت: وإذا نزلت بدار ذل فارحل",
            expected_output="وإذا نزلت بدار ذل فارحل"
        )
        score = metric.measure(tc)
        self.assertEqual(score, 1.0)
        self.assertTrue(metric.is_successful())

    def test_poetic_refusal_metric(self):
        """Tests that guardrail flags hallucinations and rewards correct refusals."""
        metric = PoeticRefusalMetric()

        # Correct refusal
        tc_safe = LLMTestCase(
            input="شعر المتنبي عن الطائرات",
            actual_output="المعلومة غير متوفرة في التراث.",
            expected_output="المعلومة المطلوبة غير متوفرة في قاعدة بيانات الشعر."
        )
        self.assertEqual(metric.measure(tc_safe), 1.0)
        self.assertTrue(metric.is_successful())

        # Hallucinated answer for impossible query
        tc_hallucinated = LLMTestCase(
            input="شعر المتنبي عن الطائرات",
            actual_output="قال المتنبي: ركبت طيارة في الجو سابحة...",
            expected_output="المعلومة المطلوبة غير متوفرة في قاعدة بيانات الشعر."
        )
        self.assertEqual(metric.measure(tc_hallucinated), 0.0)
        self.assertFalse(metric.is_successful())

    def test_retrieval_benchmark_execution(self):
        """Ensures the retrieval benchmark executes without error."""
        bench = RetrievalBenchmark(self.orchestrator)
        results = bench.evaluate(top_k=3)
        self.assertIn("summary", results)
        self.assertIn("hit@1", results["summary"])
        self.assertIn("mrr", results["summary"])


if __name__ == "__main__":
    unittest.main()
