#!/usr/bin/env python3
"""
RawiAI - Demo for Domain-Specific Classical Arabic Poetry NLP
Demonstrates segmentation, prosody, rhyme, stemming, and heritage lexicon enrichment.
"""

import warnings
warnings.filterwarnings("ignore")

from rawiai.models.schemas import EnrichedVerse
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.nlp.segmenter import VerseSegmenter
from rawiai.nlp.stemmer import PoetryArabicStemmer
from rawiai.nlp.prosody import ProsodyAnalyzer
from rawiai.nlp.lexicon import PoetryLexicon
from rawiai.processors.corpus_processor import PoetryCorpusProcessor


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def run_demo():
    print_banner("[DEMO] RawiAI - Classical Arabic Poetry NLP Engine")

    test_verses = [
        {
            "raw": "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            "poet": "عنترة بن شداد",
            "era": "جاهلي",
            "title": "معلقة عنترة",
        },
        {
            "raw": "الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي # وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
            "poet": "أبو الطيب المتنبي",
            "era": "عباسي",
            "title": "وا حر قلباه",
        },
        {
            "raw": "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ ... بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
            "poet": "امرؤ القيس",
            "era": "جاهلي",
            "title": "معلقة امرئ القيس",
        },
        {
            "raw": "أَلاَ هُبِّي بِصَحْنِكِ فَاصْبَحِينَا ... وَلاَ تُبْقِي خُمُورَ الأَنْدَرِينَا",
            "poet": "عمرو بن كلثوم",
            "era": "جاهلي",
            "title": "معلقة عمرو بن كلثوم",
        },
        {
            "raw": "إِذَا المَرْءُ لَمْ يَدْنَسْ مِنَ اللُّؤْمِ عِرْضُهُ ... فَكُلُّ رِدَاءٍ يَرْتَدِيهِ جَمِيلُ",
            "poet": "السموأل",
            "era": "جاهلي",
            "title": "لامية السموأل",
        },
    ]

    processor = PoetryCorpusProcessor()

    for idx, sample in enumerate(test_verses, start=1):
        print_banner(f"Sample {idx}: {sample['poet']} ({sample['era']})")
        
        enriched = processor.process_verse_text(
            raw_text=sample["raw"],
            poet=sample["poet"],
            era=sample["era"],
            poem_title=sample["title"]
        )

        if not enriched:
            print("[ERROR] Failed to process verse")
            continue

        print(f"Original Vocalized Text:\n   {sample['raw']}")
        print(f"\nHemistich Segmentation (Sadr & Ajuz):")
        print(f"   - Sadr (First Hemistich) : {enriched.sadr}")
        print(f"   - Ajuz (Second Hemistich): {enriched.ajuz}")

        print(f"\nProsodic Analysis & Rhyme (Arud):")
        print(f"   - Detected Meter    : {enriched.prosody.meter}")
        if enriched.prosody.meter_tafail:
            print(f"   - Metrical Pattern  : {enriched.prosody.meter_tafail}")
        print(f"   - Rawiyy (Rhyme)    : {enriched.prosody.rhyme_letter}")
        print(f"   - Rhyme Type        : {enriched.prosody.rhyme_type}")
        print(f"   - Confidence Score  : {enriched.prosody.confidence * 100:.0f}%")

        print(f"\nBM25 Poetry Stemmed Tokens:")
        print(f"   {enriched.stemmed_tokens}")

        print(f"\nClassical Lexicon Annotations:")
        if enriched.difficult_words:
            for dw in enriched.difficult_words:
                print(f"   * [{dw.word}] (Root: {dw.root} | Lexicon: {dw.source_lexicon}):")
                print(f"     {dw.meaning}")
        else:
            print("   No archaic/difficult terms flagged for this verse.")

        print(f"\nGrounded RAG Context for LLM:")
        print("-" * 55)
        print(enriched.to_rag_context())
        print("-" * 55)


if __name__ == "__main__":
    run_demo()
