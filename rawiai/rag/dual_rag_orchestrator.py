"""
Dual-RAG Orchestrator for Classical Arabic Poetry & Rhetoric.
Coordinates between the Poetry RAG index and the 'Asas Al-Balagha' Lexicon RAG index,
performing Multi-Hop retrieval for in-depth literary explanations.
"""

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

    def __init__(
        self,
        poetry_records: Optional[List[EnrichedVerse]] = None,
        lexicon_retriever: Optional[AsasLexiconRetriever] = None
    ):
        self.poetry_records = poetry_records or []
        self.lexicon_retriever = lexicon_retriever or AsasLexiconRetriever()

    def set_poetry_records(self, records: List[EnrichedVerse]):
        """Sets or updates indexed poetry records."""
        self.poetry_records = records

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
            "من قائل", "من الشاعر", "قائل البيت", "صاحب البيت", "لمن هذا البيت"
        ]
        for kw in author_keywords:
            if ArabicNormalizer.normalize_search(kw) in norm_q:
                return "author"

        completion_keywords = [
            "اكمل", "تكملة", "تتمة", "اكمل البيت"
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
            "ما معنى:", "ما معنى", "معنى البيت:", "معنى البيت",
            "اشرح:", "اشرح", "فسر:", "فسر", "ما تفسير:", "ما تفسير",
            "أكمل:", "أكمل", "اكمل:", "اكمل"
        ]
        for prefix in prefixes:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
                break
        return q.strip(":؟? ")

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
        """
        context = self.build_dual_rag_context(query, retrieved_verses, intent)

        system_prompt = (
            "أنت «راوي» (RawiAI)، خبير أدبي متخصص في الشعر العربي الكلاسيكي وبلاغة اللغة العربية. "
            "تعتمد في إجاباتك حصراً على الشواهد الشعرية المسترجعة وعلى معجم «أساس البلاغة» للزمخشري. "
            "لا تختلق أبياتاً ولا تنسب شعراً لغير قائله، واشرح الاستعارات والمجازات بالاستناد إلى شواهد المعجم المرفقة."
        )

        user_prompt = f"""
استعلام المستخدم:
{query}

سياق الأدلة الموثقة (الشعر + أساس البلاغة للزمخشري):
{context}

التعليمات حسب نية الاستعلام ({intent}):
1. اشرح المعنى بدقة بليغة مستنداً إلى المعاني الحقيقية والمجازية المذكورة في سياق أساس البلاغة.
2. وضح الصور البيانية (الاستعارة، التشبيه، الكناية) التي استعملها الشاعر إن وُجدت.
3. التزم فقط بالمعلومات المسترجعة ولا تؤلف سياقات تاريخية غير موجودة في الأدلة.

قدم الإجابة باللغة العربية الفصحى الراقية.
"""
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt.strip(),
            "context": context
        }
