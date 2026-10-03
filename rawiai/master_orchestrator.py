"""
Rawi Master Orchestrator (المايسترو الموحد لمنظومة راوي).
Unifies:
1. Multi-Agent Poetic Council (نظم الشعر وتوليده وتحكيمه)
2. Dual-RAG Literary Orchestrator (استرجاع الشواهد ومعجم أساس البلاغة)
3. Poetic Guardrails (حواجز الأمان وفحص السلامة والملاءمة الأدبية)
"""

import os
import re
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field

from rawiai.agents.schemas import PoemCompositionRequest, GeneratedPoem, PoeticEvaluationReport
from rawiai.agents.poetic_council import PoeticCouncil
from rawiai.agents.evaluator_agent import HumanPoetryEvaluator
from rawiai.agents.guardrails import PoeticGuardrails, GuardrailCheckResult
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.rag.prosody_rules_retriever import ProsodyRulesRetriever
from rawiai.models.schemas import EnrichedVerse
from rawiai.nlp.normalizer import ArabicNormalizer


class RawiIntent(str):
    """Unified intent taxonomy for RawiAI."""
    COMPOSE = "compose"          # طلب نظم أو تأليف شعر
    CRITIQUE = "critique"        # طلب تقييم شعر أو فحص وزنه وقافيته ونقده
    EXPLAIN = "meaning"          # طلب شرح بيت أو بيان مجاز أو مفردة
    AUTHOR = "author"            # طلب معرفة قائل البيت
    COMPLETION = "completion"    # طلب إكمال شطر أو بيت ناقص
    SEARCH = "search"            # بحث عام أو استعلام موضوعي
    BLOCKED = "blocked"          # طلب مخالف لمعايير الأمان ومحجوب


class RawiResponse(BaseModel):
    """Unified response envelope from RawiMasterOrchestrator."""
    query: str
    intent: str
    success: bool
    response_text: str
    poem: Optional[GeneratedPoem] = None
    evaluation_report: Optional[PoeticEvaluationReport] = None
    prompt_bundle: Optional[Dict[str, str]] = None
    retrieved_verses: List[Dict[str, Any]] = Field(default_factory=list)
    guardrail_result: Optional[GuardrailCheckResult] = None


class RawiMasterOrchestrator:
    """
    Enterprise-grade Master Orchestrator for RawiAI.
    Serves as the central gateway connecting users, safety guardrails,
    the Multi-Agent Poetic Council, Human Poetry Evaluator, and the Dual-RAG Knowledge Engine.
    """

    CRITIQUE_TRIGGERS = [
        "قيم هذا الشعر", "قيم لي هذا الشعر", "قيم الشعر", "قيم هذا البيت", "قيم لي هذا البيت",
        "قيم الأبيات", "قيم الابيات", "تقييم الشعر", "تقييم هذا البيت", "تقييم هذه القصيدة",
        "انقد هذا الشعر", "انقد هذا البيت", "انقد قصيدتي", "نقد الشعر", "تحكيم هذا الشعر",
        "هل هذا البيت موزون", "هل هذه الأبيات موزونة", "هل هذه الابيات موزونة", "هل هذه القصيدة موزونة",
        "هل فيه كسر", "هل فيها كسر", "فحص الوزن", "فحص القافية", "صحح لي هذا الشعر", "صحح هذا البيت",
        "عيوب هذه القصيدة", "ما عيوب هذا الشعر", "وزن هذا البيت", "عروض هذا البيت"
    ]

    COMPOSITION_TRIGGERS = [
        "اكتب شعر", "اكتب لي شعر", "اكتب قصيدة", "اكتب لي قصيدة",
        "انظم", "انظم لي", "نظم", "أبيات في", "ابيات في", "أبيات عن", "ابيات عن",
        "قصيدة في", "قصيدة عن", "شعر عن", "شعر في", "ألف شعر", "تأليف قصيدة",
        "أنشئ قصيدة", "هات شعرا", "قرض الشعر", "بحر الطويل", "بحر الكامل", "بحر البسيط"
    ]

    def __init__(
        self,
        council: Optional[PoeticCouncil] = None,
        dual_rag: Optional[DualRAGOrchestrator] = None,
        poetry_records: Optional[List[EnrichedVerse]] = None,
        lexicon_retriever: Optional[AsasLexiconRetriever] = None,
        prosody_retriever: Optional[ProsodyRulesRetriever] = None,
        evaluator: Optional[HumanPoetryEvaluator] = None,
        use_llm: bool = False
    ):
        lex = lexicon_retriever or AsasLexiconRetriever()
        pros_ret = prosody_retriever or ProsodyRulesRetriever()

        self.use_llm = use_llm
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = None
        if self.use_llm and self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[RawiMasterOrchestrator] Failed to init OpenAI client: {e}")
                self.client = None

        self.rules_retriever = pros_ret
        self.council = council or PoeticCouncil(lexicon_retriever=lex, use_llm=use_llm)
        self.evaluator = evaluator or HumanPoetryEvaluator(
            prosody_retriever=pros_ret,
            use_llm=use_llm
        )
        self.dual_rag = dual_rag or DualRAGOrchestrator(
            poetry_records=poetry_records or [],
            lexicon_retriever=lex
        )
        self.guardrails = PoeticGuardrails

    def classify_intent(self, query: str) -> str:
        """
        Classifies incoming user queries into critique, composition, or RAG retrieval intents.
        """
        norm_q = ArabicNormalizer.normalize_search(query)

        # 1. Check critique & evaluation patterns first
        for trigger in self.CRITIQUE_TRIGGERS:
            norm_trig = ArabicNormalizer.normalize_search(trigger)
            if norm_trig in norm_q:
                return RawiIntent.CRITIQUE

        # 2. Check composition patterns
        for trigger in self.COMPOSITION_TRIGGERS:
            norm_trig = ArabicNormalizer.normalize_search(trigger)
            if norm_trig in norm_q:
                return RawiIntent.COMPOSE

        # 3. Delegate to DualRAG intent classifier for analysis & retrieval queries
        rag_intent = self.dual_rag.classify_intent(query)
        if rag_intent == "meaning":
            return RawiIntent.EXPLAIN
        elif rag_intent == "author":
            return RawiIntent.AUTHOR
        elif rag_intent == "completion":
            return RawiIntent.COMPLETION
        return RawiIntent.SEARCH

    def evaluate_poem(self, text: str) -> PoeticEvaluationReport:
        """Direct API to evaluate human or external poetry."""
        return self.evaluator.evaluate_text(text)

    def _extract_critique_text(self, query: str) -> str:
        """Extracts the raw poetry text from a critique request."""
        cleaned = query
        prefixes = [
            r"^(قيم\s+(لي\s+)?(هذا\s+|هذه\s+)?(الشعر|البيت|الأبيات|الابيات|القصيدة)?\s*[:\-\.]?)",
            r"^(انقد\s+(لي\s+)?(هذا\s+|هذه\s+)?(الشعر|البيت|الأبيات|الابيات|القصيدة|قصيدتي)?\s*[:\-\.]?)",
            r"^(تحكيم\s+(هذا\s+|هذه\s+)?(الشعر|البيت|الأبيات|الابيات|القصيدة)?\s*[:\-\.]?)",
            r"^(تقييم\s+(هذا\s+|هذه\s+)?(الشعر|البيت|الأبيات|الابيات|القصيدة)?\s*[:\-\.]?)",
            r"^(هل\s+(هذا\s+|هذه\s+)?(البيت|الأبيات|الابيات|القصيدة)\s+موزون[ة]?\s*[:\-\.]?)",
            r"^(صحح\s+(لي\s+)?(هذا\s+|هذه\s+)?(الشعر|البيت|الأبيات|الابيات|القصيدة)?\s*[:\-\.]?)",
            r"^(فحص\s+(وزن|قافية)\s+(هذا\s+|هذه\s+)?(الشعر|البيت|الأبيات|الابيات|القصيدة)?\s*[:\-\.]?)"
        ]
        for p in prefixes:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else query

    def _extract_composition_params(self, query: str) -> PoemCompositionRequest:
        """Heuristically extracts target meter, rhyme, and topic from prompt."""
        norm_q = ArabicNormalizer.normalize_search(query)

        # Extract meter if specified
        target_meter = None
        for m in ["الطويل", "الكامل", "البسيط", "الخفيف", "الوافر", "الرمل", "السريع"]:
            if ArabicNormalizer.normalize_search(m) in norm_q:
                target_meter = m
                break

        # Extract rhyme letter if specified
        target_rhyme = None
        rhyme_match = re.search(r"قافية\s*(?:بحرف|حرف|على)?\s*([ء-ي])", query)
        if rhyme_match:
            target_rhyme = rhyme_match.group(1)

        # Clean topic from common prefixes
        cleaned_topic = query
        prefixes = [
            r"^(اكتب\s+(لي\s+)?(قصيدة|شعر|أبيات|ابيات)\s*(عن|في|حول)?)",
            r"^(انظم\s+(لي\s+)?(قصيدة|شعر|أبيات|ابيات)\s*(عن|في|حول)?)",
            r"^(أبيات\s*(عن|في|حول)?)",
            r"^(قصيدة\s*(عن|في|حول)?)"
        ]
        for p in prefixes:
            cleaned_topic = re.sub(p, "", cleaned_topic, flags=re.IGNORECASE).strip()

        # Remove trailing meter/rhyme specification from topic text
        cleaned_topic = re.sub(r"(على\s+بحر\s+[\u0621-\u064A]+).*", "", cleaned_topic).strip()
        cleaned_topic = re.sub(r"(بقافية\s+[\u0621-\u064A]+).*", "", cleaned_topic).strip()

        if not cleaned_topic:
            cleaned_topic = query

        return PoemCompositionRequest(
            topic=cleaned_topic,
            target_meter=target_meter,
            target_rhyme=target_rhyme,
            verse_count=3
        )

    def process(self, query: str, **kwargs) -> RawiResponse:
        """
        Master processing pipeline:
        1. Guardrail input validation (Language, Safety, Toxicity, Domain).
        2. Intent classification (Critique vs. Compose vs. RAG Explanation).
        3. Routing to HumanPoetryEvaluator, PoeticCouncil, or DualRAGOrchestrator.
        4. Output guardrail validation and assembly into unified RawiResponse.
        """
        # Step 1: Input Guardrail Check
        input_check = self.guardrails.inspect_input(query)
        if not input_check.passed:
            return RawiResponse(
                query=query,
                intent=RawiIntent.BLOCKED,
                success=False,
                response_text=f"⚠️ [تنبيه أمان RawiAI]: {input_check.remediation_message}",
                guardrail_result=input_check
            )

        # Step 2: Intent Classification
        intent = self.classify_intent(query)

        # Step 3: Branch to Human Poetry Evaluator for Critique
        if intent == RawiIntent.CRITIQUE:
            critique_input = self._extract_critique_text(query)
            report = self.evaluator.evaluate_text(critique_input)
            return RawiResponse(
                query=query,
                intent=RawiIntent.CRITIQUE,
                success=True,
                response_text=report.format_display(),
                evaluation_report=report,
                guardrail_result=input_check
            )

        # Step 4: Branch to Poetic Council for Composition
        if intent == RawiIntent.COMPOSE:
            req = self._extract_composition_params(query)
            poem = self.council.compose_poem(req, **kwargs)
            return RawiResponse(
                query=query,
                intent=RawiIntent.COMPOSE,
                success=poem.safety_passed,
                response_text=poem.full_text,
                poem=poem,
                guardrail_result=input_check
            )

        # Step 4: Branch to Dual-RAG for Literary Q&A / Lexicon Search
        retrieved_verses = self.dual_rag.retrieve_poetry(query, top_k=kwargs.get("top_k", 3))
        prompt_bundle = self.dual_rag.formulate_grounded_prompt(
            query=query,
            retrieved_verses=retrieved_verses,
            intent=intent
        )

        formatted_verses = [
            {
                "verse_text": getattr(v, "original_text", getattr(v, "text", "")),
                "poet": getattr(v, "poet", None),
                "era": getattr(v, "era", None),
                "meter": getattr(v.prosody, "meter", getattr(v.prosody, "meter_name", None)) if v.prosody else None,
                "rhyme": getattr(v.prosody, "rhyme_letter", None) if v.prosody else None,
                "sadr": getattr(v, "sadr", ""),
                "ajuz": getattr(v, "ajuz", ""),
                "title": getattr(v, "poem_title", ""),
                "theme": getattr(v, "theme", "")
            }
            for v in retrieved_verses
        ]

        answer = self._generate_rag_answer(query, prompt_bundle, intent, retrieved_verses)

        return RawiResponse(
            query=query,
            intent=intent,
            success=bool(retrieved_verses),
            response_text=answer,
            prompt_bundle=prompt_bundle,
            retrieved_verses=formatted_verses,
            guardrail_result=input_check
        )

    def _generate_rag_answer(
        self,
        query: str,
        prompt_bundle: Dict[str, str],
        intent: str,
        retrieved_verses: List[EnrichedVerse]
    ) -> str:
        """
        Executes grounded generation via OpenAI LLM (gpt-4o-mini) matching advanced-arabic-poetry-rag.
        Falls back to structured deterministic extraction if offline or if no LLM configured.
        """
        fallback = "عذراً، الشاهد أو المعلومة المطلوبة غير متوفرة في قاعدة الشواهد الحالية."

        # 1. Try Live OpenAI Grounded Generation if enabled
        if self.client:
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": prompt_bundle.get("system_prompt", "")},
                        {"role": "user", "content": prompt_bundle.get("user_prompt", "")}
                    ]
                )
                answer = resp.choices[0].message.content or ""
                if answer.strip():
                    return answer.strip()
            except Exception as e:
                print(f"[RawiMasterOrchestrator] OpenAI RAG generation error: {e}")

        # 2. Deterministic grounded fallback
        if not retrieved_verses:
            return fallback

        top = retrieved_verses[0]
        if intent == RawiIntent.AUTHOR:
            ans = f"قائل البيت هو الشاعر {top.poet}"
            if top.era:
                ans += f" ({top.era})"
            if top.poem_title:
                ans += f" من قصيدة «{top.poem_title}»"
            ans += f".\nالشاهد كاملاً: {top.formatted_bayt()}"
            if top.prosody and top.prosody.meter and top.prosody.meter != "غير محدد":
                ans += f" (بحر {top.prosody.meter})"
            return ans
        elif intent == RawiIntent.COMPLETION:
            return f"تكملة البيت:\n{top.formatted_bayt()}\nالشاعر: {top.poet or 'غير محدد'}"
        elif intent == RawiIntent.EXPLAIN:
            ans = f"البيت للشاعر {top.poet} ({top.era or 'تراثي'}):\n{top.formatted_bayt()}\n"
            if top.theme:
                ans += f"الغرض الشعري: {top.theme}.\n"
            ans += "شرح المعنى: البيت يعكس فصاحة المعنى والبيان العربي الكلاسيكي الأصيل."
            return ans
        else:  # search
            verses_lines = [f"• {v.formatted_bayt()} — {v.poet} ({v.era or ''})" for v in retrieved_verses]
            return "الشواهد المستخرجة المطابقة للاستعلام:\n" + "\n".join(verses_lines)
