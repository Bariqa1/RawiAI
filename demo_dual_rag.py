#!/usr/bin/env python3
"""
RawiAI - Demo for Dual-RAG: Poetry RAG + Asas Al-Balagha Lexicon RAG.
Demonstrates Multi-Hop retrieval for in-depth rhetorical and metaphorical poetic explanations.
"""

import warnings
warnings.filterwarnings("ignore")

from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_dual_rag_demo():
    print_banner("[DEMO] RawiAI Dual-RAG: Classical Poetry + Asas Al-Balagha (Al-Zamakhshari)")

    # Sample indexed poetry records in RawiAI
    indexed_verses = [
        EnrichedVerse(
            id="v1",
            original_text="حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            sadr="حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ",
            ajuz="وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            normalized_search="حكم سيوفك في رقاب العذل واذا نزلت بدار ذل فارحل",
            stemmed_tokens=["حكم", "سيف", "رقاب", "عذل", "دار", "ذل", "ارحل"],
            prosody=ProsodyInfo(meter="الكامل", meter_tafail="متفاعلن متفاعلن متفاعلن", rhyme_letter="ل", confidence=0.85),
            poet="عنترة بن شداد",
            era="جاهلي",
            theme="شجاعة وفخر",
            poem_title="معلقة عنترة"
        ),
        EnrichedVerse(
            id="v2",
            original_text="الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي # وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
            sadr="الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي",
            ajuz="وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
            normalized_search="الخيل والليل والبيداء تعرفني والسيف والرمح والقرطاس والقلم",
            stemmed_tokens=["خيل", "ليل", "بيداء", "تعرف", "سيف", "رمح", "قرطاس", "قلم"],
            prosody=ProsodyInfo(meter="الطويل", meter_tafail="فعولن مفاعيلن فعولن مفاعيلن", rhyme_letter="م", confidence=0.90),
            poet="أبو الطيب المتنبي",
            era="عباسي",
            theme="فخر واعتداد بالنفس",
            poem_title="وا حر قلباه"
        ),
        EnrichedVerse(
            id="v3",
            original_text="واختر لنفسك منزلاً تعلو به ... أو مت كريماً تحت ظل القسطلِ",
            sadr="واختر لنفسك منزلاً تعلو به",
            ajuz="أو مت كريماً تحت ظل القسطلِ",
            normalized_search="واختر لنفسك منزلا تعلو به او مت كريما تحت ظل القسطل",
            stemmed_tokens=["اختر", "نفس", "منزل", "تعلو", "مت", "كريم", "ظل", "قسطل"],
            prosody=ProsodyInfo(meter="الكامل", rhyme_letter="ل", confidence=0.80),
            poet="عنترة بن شداد",
            era="جاهلي",
            theme="شجاعة وعزة"
        )
    ]

    orchestrator = DualRAGOrchestrator(poetry_records=indexed_verses)

    test_queries = [
        "ما معنى حَكِّم سُيُوفَكَ في رِقَابِ العُذَّلِ وما الاستعارة فيها؟",
        "اشرح المجاز والبلاغة في: الخيل والليل والبيداء تعرفني",
        "من قائل: حكم سيوفك في رقاب العذل؟",
    ]

    for q_idx, query in enumerate(test_queries, start=1):
        print_banner(f"Query {q_idx}: {query}")

        # 1. Intent & Search Text
        intent = orchestrator.classify_intent(query)
        cleaned_text = orchestrator.clean_search_query(query)
        print(f"- Intent Detected   : [{intent}]")
        print(f"- Cleaned Query     : '{cleaned_text}'")

        # 2. Hop 1: Poetry Retrieval
        matched_verses = []
        for v in indexed_verses:
            if any(term in v.normalized_search for term in cleaned_text.split() if len(term) >= 3):
                matched_verses.append(v)

        if not matched_verses:
            matched_verses = [indexed_verses[0]]

        print(f"\n[Hop 1: Poetry RAG] Retrieved Poetic Evidence:")
        for mv in matched_verses:
            print(f"   * {mv.formatted_bayt()} -- {mv.poet} ({mv.era}) [Meter: {mv.prosody.meter}]")

        # 3. Hop 2: Lexicon RAG (Asas Al-Balagha) if intent == 'meaning'
        if intent == "meaning":
            print(f"\n[Hop 2: Asas Al-Balagha RAG] Rhetorical & Metaphorical Evidence:")
            for mv in matched_verses:
                lex_entries = orchestrator.retrieve_lexicon_for_verse(mv)
                for le in lex_entries:
                    print(f"   * Lemma [{le.lemma}] (Root: {le.root}):")
                    print(f"      - Literal Meaning     : {le.literal_meaning[:90]}...")
                    print(f"      - Metaphorical (Majaz): {le.metaphorical_meaning}")

        # 4. Formulate Dual Grounded Prompt
        prompt_data = orchestrator.formulate_grounded_prompt(query, matched_verses, intent)
        print(f"\n[Prompt Assembly] Grounded Context for LLM Generation:")
        print("-" * 65)
        print(prompt_data["context"][:800] + "\n... [Additional Asas Al-Balagha citations]")
        print("-" * 65)


if __name__ == "__main__":
    run_dual_rag_demo()
