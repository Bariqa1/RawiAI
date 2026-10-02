"""
Rawi Master Orchestrator (المايسترو الموحد لمنظومة راوي).
Unifies:
1. Multi-Agent Poetic Council (نظم الشعر وتوليده وتحكيمه)
2. Dual-RAG Literary Orchestrator (استرجاع الشواهد ومعجم أساس البلاغة)
3. Poetic Guardrails (حواجز الأمان وفحص السلامة والملاءمة الأدبية)
"""

import re
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field

from rawiai.agents.schemas import PoemCompositionRequest, GeneratedPoem
from rawiai.agents.poetic_council import PoeticCouncil
from rawiai.agents.guardrails import PoeticGuardrails, GuardrailCheckResult
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.models.schemas import EnrichedVerse
from rawiai.nlp.normalizer import ArabicNormalizer


class RawiIntent(str):
    """Unified intent taxonomy for RawiAI."""
    COMPOSE = "compose"          # طلب نظم أو تأليف شعر
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
    prompt_bundle: Optional[Dict[str, str]] = None
    retrieved_verses: List[Dict[str, Any]] = Field(default_factory=list)
    guardrail_result: Optional[GuardrailCheckResult] = None


class RawiMasterOrchestrator:
    """
    Enterprise-grade Master Orchestrator for RawiAI.
    Serves as the central gateway connecting users, safety guardrails,
    the Multi-Agent Poetic Council, and the Dual-RAG Knowledge Engine.
    """

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
        use_llm: bool = False
    ):
        lex = lexicon_retriever or AsasLexiconRetriever()
        self.council = council or PoeticCouncil(lexicon_retriever=lex, use_llm=use_llm)
        self.dual_rag = dual_rag or DualRAGOrchestrator(
            poetry_records=poetry_records or [],
            lexicon_retriever=lex
        )
        self.guardrails = PoeticGuardrails

    def classify_intent(self, query: str) -> str:
        """
        Classifies incoming user queries into either composition or RAG retrieval intents.
        """
        norm_q = ArabicNormalizer.normalize_search(query)

        # 1. Check composition patterns first
        for trigger in self.COMPOSITION_TRIGGERS:
            norm_trig = ArabicNormalizer.normalize_search(trigger)
            if norm_trig in norm_q:
                return RawiIntent.COMPOSE

        # 2. Delegate to DualRAG intent classifier for analysis & retrieval queries
        rag_intent = self.dual_rag.classify_intent(query)
        if rag_intent == "meaning":
            return RawiIntent.EXPLAIN
        elif rag_intent == "author":
            return RawiIntent.AUTHOR
        elif rag_intent == "completion":
            return RawiIntent.COMPLETION
        return RawiIntent.SEARCH

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
        2. Intent classification (Compose vs. RAG Explanation).
        3. Routing to either PoeticCouncil or DualRAGOrchestrator.
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

        # Step 3: Branch to Poetic Council for Composition
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
                "verse_text": v.text,
                "poet": v.poet,
                "era": v.era,
                "meter": v.prosody.meter_name if v.prosody else None,
                "rhyme": v.prosody.rhyme_letter if v.prosody else None
            }
            for v in retrieved_verses
        ]

        return RawiResponse(
            query=query,
            intent=intent,
            success=True,
            response_text=prompt_bundle["context"],
            prompt_bundle=prompt_bundle,
            retrieved_verses=formatted_verses,
            guardrail_result=input_check
        )
