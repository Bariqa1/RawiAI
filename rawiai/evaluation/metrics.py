"""
Custom DeepEval Metrics for Classical Arabic Poetry and Rhetorical Dual-RAG.
Inherits from DeepEval's BaseMetric to provide domain-aware evaluation
for Arabic literary retrieval, faithfulness, and hallucination prevention.
"""

from typing import List, Optional, Any
try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    # Fallback base class if deepeval is not installed
    class BaseMetric:
        def __init__(self):
            self.score = 0.0
            self.success = False
            self.reason = ""
            self.threshold = 0.7

from rawiai.nlp.normalizer import ArabicNormalizer


class PoeticFaithfulnessMetric(BaseMetric):
    """
    Evaluates whether the generated response is strictly grounded in the
    retrieved poetry records and 'Asas Al-Balagha' rhetorical citations.
    Penalizes hallucinated classical attributions or unsupported interpretations.
    """

    def __init__(self, threshold: float = 0.70):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: Any) -> float:
        context_text = " ".join(test_case.retrieval_context or [])
        output_text = test_case.actual_output or ""

        # Safe refusal handling: If context indicates unavailable data and output refuses, 100% faithful
        if "لا توجد شواهد" in context_text or not test_case.retrieval_context:
            if "غير متوفرة" in output_text or "لا تتوفر" in output_text or "غير موجود" in output_text:
                self.score = 1.0
                self.success = True
                self.reason = "Safe refusal verified: model avoided hallucinating verses for out-of-domain query."
                return self.score

        # Normalize texts for semantic grounding
        norm_context = ArabicNormalizer.normalize_search(context_text)
        norm_output = ArabicNormalizer.normalize_search(output_text)

        output_words = [w for w in norm_output.split() if len(w) > 2]
        if not output_words:
            self.score = 1.0
            self.success = True
            self.reason = "Output is concise and within supported length."
            return self.score

        # Count grounded meaningful words
        grounded_count = 0
        unsupported = []
        for word in output_words:
            if word in norm_context:
                grounded_count += 1
            else:
                unsupported.append(word)

        # Base grounding score + bonus for citing Asas Al-Balagha when present
        raw_score = grounded_count / len(output_words)
        if "اساس البلاغة" in norm_context and "الزمخشري" in norm_output:
            raw_score = min(1.0, raw_score + 0.10)

        self.score = round(min(1.0, max(0.0, raw_score)), 2)
        self.success = self.score >= self.threshold

        if self.success:
            self.reason = f"Output is strictly grounded in literary context ({int(self.score * 100)}%)."
        else:
            self.reason = f"Output contains unsupported vocabulary: {', '.join(unsupported[:5])}"

        return self.score

    async def a_measure(self, test_case: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class PoeticRelevancyMetric(BaseMetric):
    """
    Evaluates whether the generated response directly answers the user's specific
    literary inquiry (Author identification, verse completion, metaphorical explanation,
    thematic search, or refusal).
    """

    def __init__(self, threshold: float = 0.65):
        super().__init__()
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: Any) -> float:
        query = test_case.input or ""
        output = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        norm_query = ArabicNormalizer.normalize_search(query)
        norm_output = ArabicNormalizer.normalize_search(output)
        norm_expected = ArabicNormalizer.normalize_search(expected)

        # 1. Author Identification queries
        if any(kw in norm_query for kw in ["من قائل", "من الشاعر", "صاحب البيت"]):
            if norm_expected and norm_expected in norm_output:
                self.score = 1.0
                self.success = True
                self.reason = "Poet author correctly identified and matched expected target."
                return self.score

        # 2. Verse Completion queries
        if any(kw in norm_query for kw in ["اكمل", "تكملة", "تتمة"]):
            if norm_expected and (norm_expected in norm_output or norm_output in norm_expected):
                self.score = 1.0
                self.success = True
                self.reason = "Verse hemistich completion matched expected text accurately."
                return self.score

        # 3. Refusal queries (unanswerable / out of domain)
        if "غير متوفرة" in norm_expected:
            if "غير متوفرة" in norm_output or "لا تتوفر" in norm_output:
                self.score = 1.0
                self.success = True
                self.reason = "Non-classical query successfully refused to prevent hallucination."
                return self.score

        # 4. Metaphor / Meaning / Search queries
        expected_words = [w for w in norm_expected.split() if len(w) > 2]
        if not expected_words:
            self.score = 1.0
            self.success = True
            return self.score

        overlap = sum(1 for w in expected_words if w in norm_output)
        ratio = overlap / len(expected_words)
        
        # Scale to 0.60..1.0 for relevant responses
        self.score = round(min(1.0, 0.55 + ratio * 0.45), 2)
        self.success = self.score >= self.threshold

        if self.success:
            self.reason = f"Response is semantically relevant to query intent ({int(self.score * 100)}%)."
        else:
            self.reason = f"Response did not satisfy semantic relevancy threshold (score: {self.score})."

        return self.score

    async def a_measure(self, test_case: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


class PoeticRefusalMetric(BaseMetric):
    """
    Dedicated guardrail metric for out-of-domain or modern queries
    masquerading as classical Arabic poetry.
    """

    def __init__(self):
        super().__init__()
        self.threshold = 1.0
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: Any) -> float:
        output = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        if "غير متوفرة" in expected or "غير موجود" in expected:
            if "غير متوفرة" in output or "لا تتوفر" in output or "غير موجود" in output:
                self.score = 1.0
                self.success = True
                self.reason = "Guardrail passed: response correctly refused due to absence of classical evidence."
            else:
                self.score = 0.0
                self.success = False
                self.reason = "Guardrail failed: model hallucinated output for out-of-domain query!"
        else:
            # Not a refusal case
            self.score = 1.0
            self.success = True
            self.reason = "Standard in-domain test case."

        return self.score

    async def a_measure(self, test_case: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success
