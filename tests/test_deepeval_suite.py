"""
Official DeepEval Test Suite for RawiAI Dual-RAG System.
Can be executed via:
  1) deepeval test run tests/test_deepeval_suite.py
  2) pytest tests/test_deepeval_suite.py
  3) python3 -m unittest discover tests
"""

import os
import json
import pytest
import unittest
from typing import Dict, Any

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from rawiai.processors.corpus_processor import PoetryCorpusProcessor
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.evaluation.deepeval_runner import DeepEvalRunner
from rawiai.evaluation.metrics import (
    PoeticFaithfulnessMetric,
    PoeticRelevancyMetric,
    PoeticRefusalMetric
)


def _init_orchestrator_and_runner():
    processor = PoetryCorpusProcessor()
    raw_samples = [
        {
            "text": "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            "poet": "عنترة بن شداد",
            "era": "جاهلي",
            "theme": "شجاعة وفخر وعزة",
            "title": "معلقة عنترة"
        },
        {
            "text": "الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي # وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
            "poet": "أبو الطيب المتنبي",
            "era": "عباسي",
            "theme": "فخر واعتداد بالنفس وشجاعة",
            "title": "وا حر قلباه"
        },
        {
            "text": "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ ... بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
            "poet": "امرؤ القيس",
            "era": "جاهلي",
            "theme": "غزل ووقوف على الأطلال",
            "title": "معلقة امرئ القيس"
        },
        {
            "text": "واختر لنفسك منزلاً تعلو به ... أو مت كريماً تحت ظل القسطلِ",
            "poet": "عنترة بن شداد",
            "era": "جاهلي",
            "theme": "عزة وشجاعة وإقدام",
            "title": "ديوان عنترة"
        },
        {
            "text": "إِذَا المَرْءُ لَمْ يَدْنَسْ مِنَ اللُّؤْمِ عِرْضُهُ ... فَكُلُّ رِدَاءٍ يَرْتَدِيهِ جَمِيلُ",
            "poet": "السموأل",
            "era": "جاهلي",
            "theme": "حكمة ومروءة وشرف",
            "title": "لامية السموأل"
        },
        {
            "text": "أَلاَ هُبِّي بِصَحْنِكِ فَاصْبَحِينَا ... وَلاَ تُبْقِي خُمُورَ الأَنْدَرِينَا",
            "poet": "عمرو بن كلثوم",
            "era": "جاهلي",
            "theme": "فخر وحماسة",
            "title": "معلقة عمرو بن كلثوم"
        }
    ]

    corpus = []
    for s in raw_samples:
        rec = processor.process_verse_text(
            raw_text=s["text"],
            poet=s["poet"],
            era=s["era"],
            theme=s["theme"],
            poem_title=s["title"]
        )
        if rec:
            corpus.append(rec)

    orchestrator = DualRAGOrchestrator(poetry_records=corpus)
    runner = DeepEvalRunner(orchestrator)
    return orchestrator, runner


_ORCHESTRATOR, _RUNNER = _init_orchestrator_and_runner()
_GOLDEN_CASES = _RUNNER.test_cases


# --- Pytest / DeepEval CLI Native Test Cases ---

@pytest.mark.parametrize("case_data", _GOLDEN_CASES, ids=[c["id"] for c in _GOLDEN_CASES])
def test_deepeval_golden_cases(case_data: Dict[str, Any]):
    """
    Executes DeepEval assertion on each golden dataset case.
    Tests Faithfulness, Relevancy, and Refusal guardrails.
    """
    bundle = _RUNNER.build_test_case_bundle(case_data)

    test_case = LLMTestCase(
        input=bundle["query"],
        actual_output=bundle["actual_output"],
        expected_output=bundle["expected_output"],
        retrieval_context=bundle["retrieval_context"]
    )

    faithfulness = PoeticFaithfulnessMetric(threshold=0.70)
    relevancy = PoeticRelevancyMetric(threshold=0.65)
    refusal = PoeticRefusalMetric()

    assert_test(test_case, [faithfulness, relevancy, refusal])


# --- Standard Python unittest Runner Support ---

class TestDeepEvalSuite(unittest.TestCase):
    """Allows running DeepEval tests via python3 -m unittest."""

    @classmethod
    def setUpClass(cls):
        cls.orchestrator, cls.runner = _init_orchestrator_and_runner()

    def test_all_cases_pass_deepeval_metrics(self):
        eval_result = self.runner.run_evaluation(max_cases=10, use_live_api=False)
        self.assertEqual(eval_result["pass_rate"], 100.0)
        self.assertGreaterEqual(eval_result["average_faithfulness"], 0.80)
        self.assertGreaterEqual(eval_result["average_relevancy"], 0.90)

    def test_refusal_guardrails(self):
        refusal_cases = [c for c in self.runner.test_cases if c["target_verse_substring"] is None]
        self.assertGreaterEqual(len(refusal_cases), 2)
        refusal_metric = PoeticRefusalMetric()

        for c in refusal_cases:
            bundle = self.runner.build_test_case_bundle(c)
            tc = LLMTestCase(
                input=bundle["query"],
                actual_output=bundle["actual_output"],
                expected_output=bundle["expected_output"],
                retrieval_context=bundle["retrieval_context"]
            )
            score = refusal_metric.measure(tc)
            self.assertEqual(score, 1.0, f"Refusal guardrail failed for query: {c['query']}")


if __name__ == "__main__":
    unittest.main()
