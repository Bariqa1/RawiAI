"""
Interactive Demonstration: RawiAI Multi-Agent Poetic Council (مجلس الشعراء).
Demonstrates autonomous multi-agent poem generation with deterministic prosody verification.
"""

from rawiai.agents import PoeticCouncil, PoemCompositionRequest


def main():
    print("=" * 80)
    print("  [RawiAI] Multi-Agent Poetic Council Showcase (مجلس الرواة والشعراء)")
    print("=" * 80)

    council = PoeticCouncil()

    topics = [
        {
            "topic": "الذكاء الاصطناعي والحضارة المعاصرة",
            "verse_count": 3
        },
        {
            "topic": "الفروسية ومقارعة الأهوال في المعركة",
            "verse_count": 3
        },
        {
            "topic": "حكمة الزمان وتقلبات الدهر",
            "verse_count": 3
        }
    ]

    for idx, t in enumerate(topics, 1):
        print(f"\n{'=' * 80}")
        print(f"  [تجربة {idx}] طلب المستخدم: {t['topic']}")
        print(f"{'=' * 80}")

        request = PoemCompositionRequest(
            topic=t["topic"],
            verse_count=t["verse_count"]
        )

        poem = council.compose_poem(request)

        print("\n[تقرير المجلس الشعري]")
        print(f"• البحر المعتمد : {poem.meter} ({poem.meter_tafail})")
        print(f"• حرف القافية   : حرف {poem.rhyme}")
        print(f"• درجة الجودة   : {poem.critique_report.overall_quality_score * 100:.0f}%")
        print(f"• جولات التنقيح : {poem.iterations_used}")
        print(f"• الملاحظات     : {poem.critique_report.general_notes}")

        print("\n[القصيدة المنظومة النهائية]:")
        print("-" * 75)
        for v in poem.verses:
            print(f"[{v.verse_number}] {v.sadr:<35} ... {v.ajuz}")
        print("-" * 75)

        if poem.metaphor_sources:
            print("\n[استلهام من مجازات أساس البلاغة للزمخشري]:")
            for m in poem.metaphor_sources:
                print(f"  * مادة [{m['lemma']}] ({m['root']}): {m['metaphorical_meaning'][:140]}...")

    print("\n" + "=" * 80)
    print("  [نجاح] اكتمل عرض مجلس الشعراء ومنظومة التوليد والتحكيم العروضي.")
    print("=" * 80)


if __name__ == "__main__":
    main()
