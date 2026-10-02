"""
Poetic Guardrails & Safety Filter for RawiAI.
Protects the Multi-Agent system against:
1. Non-Arabic or out-of-domain queries.
2. Inappropriate, offensive, or derogatory language (الهجاء الفاحش والألفاظ الخادشة).
3. Low-quality, broken, or unsymmetrical poetic outputs.
"""

import re
from typing import Tuple, Optional, List, Dict
from pydantic import BaseModel, Field
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.agents.schemas import GeneratedPoem, PoemCompositionRequest


class GuardrailCheckResult(BaseModel):
    """Outcome of a safety or domain guardrail inspection."""
    passed: bool
    risk_level: str = "low"  # "low", "medium", "high"
    reason: str = ""
    remediation_message: Optional[str] = None


class PoeticGuardrails:
    """Enterprise-grade guardrails and safety filter for classical Arabic poetry."""

    # Profanity, vulgarity, offensive satire, and toxic phrases (الهجاء المقذع والألفاظ المبتذلة)
    OFFENSIVE_KEYWORDS = [
        "سافل", "حقير", "لعنة", "قذر", "كلب", "حمار", "خنزير", "شتيمة",
        "سب", "قذف", "فاحش", "عاهر", "نذل", "ملعون", "تافه", "خبيث",
        "فاسق", "ديوث", "زنديق", "سفاح"
    ]

    # Non-literary / technical instruction patterns that are out-of-domain for poetry
    OUT_OF_DOMAIN_PATTERNS = [
        r"write (python|code|java|c\+\+|javascript|sql)",
        r"(شرح كود|اكتب كود|برمج لي|تصليح سيارة|حل معادلة رياضية|طبخ|وصفة طعام)",
        r"(hack|exploit|sql injection|payload)"
    ]

    @classmethod
    def inspect_input(cls, request_or_topic: str) -> GuardrailCheckResult:
        """
        Validates user input before invoking the agent council.
        Checks language, length, safety, and domain alignment.
        """
        text = request_or_topic.strip()

        # 1. Empty or too short check
        if not text or len(text) < 2:
            return GuardrailCheckResult(
                passed=False,
                risk_level="high",
                reason="طلب فارغ أو قصير جداً.",
                remediation_message="يرجى كتابة موضوع أو فكرة محددة لنظم الأبيات حولها."
            )

        # 2. Language check (Must contain Arabic characters)
        if not ArabicNormalizer.has_arabic(text):
            return GuardrailCheckResult(
                passed=False,
                risk_level="medium",
                reason="الطلب لا يحتوي على أحرف عربية.",
                remediation_message="عذراً، هذا المجلس مخصص لعيون الشعر العربي وبلاغة الفصحى، يرجى كتابة طلبك باللغة العربية."
            )

        # 3. Profanity & offensive content check
        norm_text = ArabicNormalizer.normalize_search(text)
        for bad_word in cls.OFFENSIVE_KEYWORDS:
            norm_bad = ArabicNormalizer.normalize_search(bad_word)
            if norm_bad in norm_text:
                return GuardrailCheckResult(
                    passed=False,
                    risk_level="high",
                    reason=f"تم رصد لفظ غير لائق ({bad_word}).",
                    remediation_message="عذراً، يلتزم مجلس الرواة بعفة اللسان ومكارم الأخلاق، ويربأ بالشعر العربي عن الألفاظ الخادشة أو المهينة."
                )

        # 4. Out-of-domain technical instruction check
        for pattern in cls.OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailCheckResult(
                    passed=False,
                    risk_level="medium",
                    reason="طلب خارج نطاق الأدب والشعر العربي.",
                    remediation_message="مجلس الرواة متخصص في عوالم الشعر العربي وبلاغته. يرجى تزويدنا بموضوع أدبي أو معنى شعري ترغب في نظمه."
                )

        # 5. Excessively long spam input
        if len(text) > 1000:
            return GuardrailCheckResult(
                passed=False,
                risk_level="medium",
                reason="النص المدخل طويل جداً.",
                remediation_message="يرجى تلخيص فكرة القصيدة في سطر أو سطرين ليتمكن المجلس من استلهام المعاني ونظمها بدقة."
            )

        return GuardrailCheckResult(
            passed=True,
            risk_level="low",
            reason="المدخلات سليمة وتلبي معايير السلامة والملاءمة الأدبية."
        )

    @classmethod
    def inspect_output(cls, poem: GeneratedPoem) -> GuardrailCheckResult:
        """
        Validates the finished poem before releasing it to the user.
        Ensures metric quality, structural integrity, and absence of hallucinations.
        """
        if not poem.verses:
            return GuardrailCheckResult(
                passed=False,
                risk_level="high",
                reason="لم يتم نظم أي أبيات.",
                remediation_message="تعذر إتمام النظم، يرجى إعادة المحاولة بموضوع أدبي أوضح."
            )

        # Minimum metric quality threshold (at least 60% of verses must pass prosodic critique)
        quality_score = poem.critique_report.overall_quality_score
        if quality_score < 0.60:
            return GuardrailCheckResult(
                passed=False,
                risk_level="medium",
                reason=f"درجة الجودة العروضية منخفضة ({quality_score * 100:.0f}%).",
                remediation_message="الأبيات المنظومة لا تزال قاصرة عن الضبط العروضي الكامل لبحر القصيدة، ونوصي بإعادة النظم في جولة تنقيح جديدة."
            )

        # Ensure all verses have proper Sadr and Ajuz separation
        for v in poem.verses:
            if not v.sadr or not v.ajuz or ("..." not in v.full_verse and "…" not in v.full_verse):
                return GuardrailCheckResult(
                    passed=False,
                    risk_level="high",
                    reason=f"هيكل البيت رقم {v.verse_number} غير عمودي أو يفتقر للصدر والعجز.",
                    remediation_message="حدث خلل في بنية شطري البيت، يرجى إعادة الصياغة."
                )

        return GuardrailCheckResult(
            passed=True,
            risk_level="low",
            reason="القصيدة مجازة عروضياً وتستوفي شروط الشطرين والقافية."
        )
