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

from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.processors.corpus_processor import PoetryCorpusProcessor
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.evaluation.deepeval_runner import DeepEvalRunner
from rawiai.evaluation.metrics import (
    PoeticFaithfulnessMetric,
    PoeticRelevancyMetric,
    PoeticRefusalMetric
)


def _init_orchestrator_and_runner():
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "rawiai", "data", "classical_poetry_corpus.json"
    )
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        corpus = []
        for item in raw_data:
            p = item.get("prosody", {})
            prosody = ProsodyInfo(
                meter=p.get("meter", "غير محدد"),
                confidence=p.get("confidence", 0.9),
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
            corpus.append(v)
    else:
        corpus = []

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
