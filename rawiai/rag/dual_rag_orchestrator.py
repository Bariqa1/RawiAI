"""
Dual-RAG Orchestrator for Classical Arabic Poetry & Rhetoric.
Coordinates between the Poetry RAG index and the 'Asas Al-Balagha' Lexicon RAG index,
performing Multi-Hop retrieval for in-depth literary explanations.
"""

import os
import json
from typing import List, Dict, Optional, Any
from rawiai.models.schemas import EnrichedVerse
from rawiai.models.lexicon_schema import AsasEntry, LexiconSearchResult
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.nlp.stemmer import PoetryArabicStemmer


class DualRAGOrchestrator:
    """
    Hierarchical multi-hop orchestrator that connects Poetry Retrieval
    with Rhetorical Lexicon Retrieval (Asas Al-Balagha).
    """

    DEFAULT_CORPUS_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "classical_poetry_corpus.json"
    )

    def __init__(
        self,
        poetry_records: Optional[List[EnrichedVerse]] = None,
        lexicon_retriever: Optional[AsasLexiconRetriever] = None
    ):
        if poetry_records:
            self.poetry_records = poetry_records
        elif os.path.exists(self.DEFAULT_CORPUS_PATH):
            try:
                with open(self.DEFAULT_CORPUS_PATH, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self.poetry_records = [EnrichedVerse(**item) for item in raw_data]
            except Exception:
                self.poetry_records = []
        else:
            self.poetry_records = []

        self.lexicon_retriever = lexicon_retriever or AsasLexiconRetriever()

    def set_poetry_records(self, records: List[EnrichedVerse]):
        """Sets or updates indexed poetry records."""
        self.poetry_records = records

    def retrieve_poetry(self, query: str, top_k: int = 3) -> List[EnrichedVerse]:
        """
        Retrieves matching poetry records from self.poetry_records based on lexical overlap.
        """
        if not self.poetry_records:
            return []

        clean_q = self.clean_search_query(query)
        norm_q = ArabicNormalizer.normalize_search(clean_q)
        if not norm_q or len(norm_q) < 2:
            return []

        arabic_stopwords = {
            "في", "من", "عن", "مع", "على", "الى", "إلى", "ان", "أن", "إن",
            "ما", "هو", "هي", "هل", "لا", "لم", "لن", "قد", "ثم", "أو", "او",
            "ذا", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "الذين", "كان", "كانت",
            "شاعر", "الشاعر", "قائل", "القائل", "صاحب", "الصاحب", "بيت", "البيت", "شعر", "الشعر"
        }

        raw_tokens = norm_q.split()
        meaningful_tokens = [t for t in raw_tokens if t not in arabic_stopwords and len(t) >= 2]
        expanded_q_tokens = set(meaningful_tokens)
        for t in list(meaningful_tokens):
            if t.startswith("ال") and len(t) > 3:
                expanded_q_tokens.add(t[2:])
            if t.startswith("و") and len(t) > 3:
                expanded_q_tokens.add(t[1:])
            if t.startswith("وال") and len(t) > 4:
                expanded_q_tokens.add(t[3:])
            if t.startswith("ب") and len(t) > 3:
                expanded_q_tokens.add(t[1:])

        scored = []
        for verse in self.poetry_records:
            score = 0.0
            norm_verse = verse.normalized_search or ArabicNormalizer.normalize_search(verse.original_text)

            # Exact substring match
            if len(norm_q) >= 4 and norm_q in norm_verse:
                score += 15.0

            # Token overlap (using meaningful content tokens)
            v_tokens = {t for t in norm_verse.split() if t not in arabic_stopwords and len(t) >= 2}
            overlap = len(expanded_q_tokens & v_tokens)
            score += overlap * 2.5

            # Poet match
            if verse.poet and ArabicNormalizer.normalize_search(verse.poet) in norm_q:
                score += 8.0

            # Theme match
            if verse.theme:
                norm_theme = ArabicNormalizer.normalize_search(verse.theme)
                for t in expanded_q_tokens:
                    if len(t) >= 3 and (t in norm_theme or norm_theme in t):
                        score += 5.0

            # Poem title match
            if verse.poem_title:
                norm_title = ArabicNormalizer.normalize_search(verse.poem_title)
                for t in expanded_q_tokens:
                    if len(t) >= 3 and (t in norm_title or norm_title in t):
                        score += 4.0

            if score >= 2.0:
                scored.append((score, verse))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def classify_intent(self, query: str) -> str:
        """
        Classifies user query into one of four task categories:
        - 'meaning': requests explanation of a verse, phrase, or metaphor.
        - 'author': requests poet or author identification.
        - 'completion': requests verse continuation.
        - 'search': general retrieval or thematic search.
        """
        norm_q = ArabicNormalizer.normalize_search(query)

        meaning_keywords = [
            "ما معنى", "معنى", "اشرح", "شرح", "تفسير", "فسر",
            "ماذا يعني", "ماذا يقصد", "المجاز في", "بلاغة", "استعارة", "تشبيه"
        ]
        for kw in meaning_keywords:
            if ArabicNormalizer.normalize_search(kw) in norm_q:
                return "meaning"

        author_keywords = [
            "من قائل", "من الشاعر", "قائل البيت", "صاحب البيت", "لمن هذا البيت", "من صاحب هذا البيت"
        ]
        for kw in author_keywords:
            if ArabicNormalizer.normalize_search(kw) in norm_q:
                return "author"

        completion_keywords = [
            "اكمل", "أكمل", "تكملة", "تتمة", "اكمل البيت", "أكمل البيت"
        ]
        for kw in completion_keywords:
            if ArabicNormalizer.normalize_search(kw) in norm_q:
                return "completion"

        return "search"

    def clean_search_query(self, query: str) -> str:
        """Removes question prefixes from query to isolate core verse or term."""
        q = query.strip()
        prefixes = [
            "من قائل:", "من قائل", "من الشاعر:", "من الشاعر",
            "قائل البيت:", "قائل البيت", "صاحب البيت:", "صاحب البيت",
            "من صاحب هذا البيت:", "من صاحب هذا البيت",
            "ما معنى:", "ما معنى", "معنى البيت:", "معنى البيت",
            "اشرح:", "اشرح", "فسر:", "فسر", "ما تفسير:", "ما تفسير",
            "أكمل البيت:", "أكمل البيت", "اكمل البيت:", "اكمل البيت",
            "أكمل:", "أكمل", "اكمل:", "اكمل", "تكملة البيت:", "تكملة البيت",
            "أريد بيتاً عن", "اريد بيتا عن", "أريد شعر عن", "اريد شعر عن",
            "ابحث عن شعر في", "ابحث عن بيت في"
        ]
        # Sort prefixes descending by length so longer prefixes match first
        prefixes.sort(key=len, reverse=True)
        for prefix in prefixes:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
                break
        return q.strip(":؟? ")

    def answer_literary_inquiry(self, query: str) -> Dict[str, Any]:
        """
        Produces a strictly grounded response directly from retrieved evidence,
        handling author identification, verse completion, meaning explanation, or safe refusal.
        """
        if not query or not query.strip():
            return {
                "answer": "المعلومة المطلوبة غير متوفرة في قاعدة بيانات الشعر.",
                "intent": "search",
                "retrieved_verses": []
            }

        intent = self.classify_intent(query)
        retrieved = self.retrieve_poetry(query, top_k=3)

        if not retrieved:
            return {
                "answer": "المعلومة المطلوبة غير متوفرة في قاعدة بيانات الشعر.",
                "intent": intent,
                "retrieved_verses": []
            }

        top_v = retrieved[0]
        if intent == "author":
            ans = f"الشاعر القائل هو: {top_v.poet} ({top_v.era}). الشاهد: {top_v.formatted_bayt()}"
        elif intent == "completion":
            ans = f"تكملة البيت: {top_v.ajuz} (الصدر: {top_v.sadr}) للشاعر {top_v.poet}."
        elif intent == "meaning":
            ans = f"معنى وبلاغة قول {top_v.poet} في بيته ({top_v.formatted_bayt()}): من بحر {top_v.prosody.meter}."
        else:
            ans = f"البيت المنشود للشاعر {top_v.poet} ({top_v.era}) هو:\n{top_v.formatted_bayt()}"

        return {
            "answer": ans,
            "intent": intent,
            "retrieved_verses": retrieved
        }

    def retrieve_lexicon_for_verse(self, verse: EnrichedVerse) -> List[AsasEntry]:
        """
        Hop 2 of the Multi-Hop:
        Given a retrieved verse, extracts its key terms and retrieves
        corresponding metaphorical entries from Asas Al-Balagha.
        """
        return self.lexicon_retriever.annotate_verse(verse.original_text)

    def build_dual_rag_context(
        self,
        query: str,
        retrieved_verses: List[EnrichedVerse],
        intent: str
    ) -> str:
        """
        Assembles a comprehensive, grounded context block incorporating:
        1. Primary Poetry evidence (Verse, Poet, Era, Meter, Rhyme).
        2. Secondary Rhetorical evidence from Asas Al-Balagha (Literal & Metaphorical usages).
        """
        context_blocks = []

        # 1. Poetry Evidence
        context_blocks.append("=== أولاً: الشواهد والأبيات الشعرية المسترجعة من قاعدة البيانات ===")
        if not retrieved_verses:
            context_blocks.append("لا توجد شواهد أو أبيات مطابقة في قاعدة البيانات لهذا الاستعلام.")
        else:
            for idx, verse in enumerate(retrieved_verses, start=1):
                block = [
                    f"[{idx}] البيت: {verse.formatted_bayt()}",
                ]
                if verse.poet:
                    block.append(f"    الشاعر: {verse.poet}")
                if verse.era:
                    block.append(f"    العصر الأدبي: {verse.era}")
                if verse.prosody.meter and verse.prosody.meter != "غير محدد":
                    block.append(f"    البحر العروضي: {verse.prosody.meter}")
                if verse.prosody.rhyme_letter:
                    block.append(f"    القافية (الروي): حرف {verse.prosody.rhyme_letter}")

                context_blocks.append("\n".join(block))

        # 2. Lexicon Evidence (Multi-Hop for 'meaning' or literary queries)
        if intent == "meaning" and retrieved_verses:
            context_blocks.append("\n=== ثانياً: شواهد ومجازات الألفاظ من معجم «أساس البلاغة» للزمخشري ===")
            all_lexicon_entries: List[AsasEntry] = []
            seen_roots = set()

            for verse in retrieved_verses:
                entries = self.retrieve_lexicon_for_verse(verse)
                for ent in entries:
                    if ent.root not in seen_roots:
                        all_lexicon_entries.append(ent)
                        seen_roots.add(ent.root)

            if all_lexicon_entries:
                for ent in all_lexicon_entries:
                    context_blocks.append(ent.to_llm_context())
            else:
                context_blocks.append("لم تُسجل مفردات غامضة في المعجم لهذا الشاهد.")

        return "\n\n".join(context_blocks)

    def formulate_grounded_prompt(
        self,
        query: str,
        retrieved_verses: List[EnrichedVerse],
        intent: str
    ) -> Dict[str, str]:
        """
        Constructs system and user prompts with balanced grounding rules,
        preventing hallucination while enabling eloquent literary interpretation.
        Matches the 4 core tasks of advanced-arabic-poetry-rag.
        """
        context = self.build_dual_rag_context(query, retrieved_verses, intent)
        fallback = "عذراً، المعلومة المطلوبة غير متوفرة في قاعدة الشواهد الحالية."

        system_prompt = (
            "أنت «راوي» (RawiAI)، خبير أدبي متخصص في الشعر العربي الكلاسيكي وبلاغة اللغة العربية وفق معجم «أساس البلاغة» للزمخشري. "
            "The retrieved context is the only source of truth. "
            "Do not hallucinate, do not invent verses, and do not use outside knowledge."
        )

        header = f"""استعلام المستخدم:
{query}

سياق الأدلة الموثقة (الشعر + أساس البلاغة للزمخشري):
{context}
"""

        if intent == "author":
            instructions = header + f"""
التعليمات حسب نية الاستعلام ({intent}):
1. حدد قائل البيت وعصره الأدبي ومصدره حصراً من حقل (الشاعر) و(العصر الأدبي) الموجود في سياق الأدلة الموثقة المسترجعة.
2. لا تخمن ولا تستخدم أي معلومات خارجية إطلاقاً.
3. إذا كان الشاعر متوفراً في السجلات المسترجعة، اذكر الإجابة باللغة العربية بدقة:
- الشاعر: [اسم الشاعر]
- العصر الأدبي: [العصر الأدبي]
- القصيدة أو الديوان: [المصدر إن وجد]
- الشاهد كاملاً: [البيت بعجز وصدر]
- البحر العروضي: [البحر العروضي إن وجد]
4. إذا لم تجد الشاهد في السجلات، أجب بالضبط:
"{fallback}"
"""
        elif intent == "completion":
            instructions = header + f"""
التعليمات حسب نية الاستعلام ({intent}):
1. أكمل الشطر أو البيت المطلوب حصراً باستخدام النص الموثق في سياق الأدلة الموثقة المسترجعة.
2. لا تخترع أي كلمة مفقودة والتزم بالكلمات وحركات التشكيل المسترجعة تماماً.
3. اعرض البيت مكتملاً بوضوح: الصدر ... العجز، واذكر الشاعر والبحر العروضي.
4. إذا لم تجد البيت في السجلات، أجب بالضبط:
"{fallback}"
"""
        elif intent == "meaning":
            instructions = header + f"""
التعليمات حسب نية الاستعلام ({intent}):
1. اشرح المعنى بدقة بليغة مستنداً إلى المعاني الحقيقية والمجازية المذكورة في سياق أساس البلاغة.
2. وضح الصور البيانية (الاستعارة، التشبيه، الكناية) التي استعملها الشاعر إن وُجدت.
3. التزم فقط بالمعلومات المسترجعة ولا تؤلف سياقات تاريخية غير موجودة في الأدلة.
4. إذا لم تجد البيت في السجلات، أجب بالضبط:
"{fallback}"
"""
        else:  # search
            instructions = header + f"""
التعليمات حسب نية الاستعلام ({intent}):
1. استخرج الأبيات المطابقة لموضوع الاستعلام أو كلماته من واقع سياق الأدلة الموثقة حصراً.
2. اعرض الأبيات بدقة مع قائلها وبحرها العروضي وغرضها الشعري.
3. لا تؤلف أبياتاً جديدة ولا تستخدم معلومات من خارج الشواهد المسترجعة.
4. إذا لم تجد شواهد مطابقة، أجب بالضبط:
"{fallback}"
"""

        return {
            "system_prompt": system_prompt,
            "user_prompt": instructions.strip(),
            "context": context
        }
