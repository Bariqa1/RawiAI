"""
Demonstration: RawiAI Master Orchestrator (المايسترو الموحد).
Showcases unified intent routing, safety guardrails, and seamless switching
between Multi-Agent Poetry Composition and Dual-RAG Literary Analysis.
"""

from rawiai import RawiMasterOrchestrator, RawiIntent


def main():
    print("=" * 80)
    print("  [RawiAI] Master Orchestrator & Safety Guardrails Showcase")
    print("  المايسترو الموحد لمنظومة راوي الأدبية وحواجز الأمان")
    print("=" * 80)

    orchestrator = RawiMasterOrchestrator(use_llm=False)

    test_scenarios = [
        {
            "name": "سيناريو 1: طلب نظم شعر (Composition -> Poetic Council)",
            "query": "انظم لي ثلاثة أبيات في الصبر على المكاره على بحر الكامل"
        },
        {
            "name": "سيناريو 2: استفسار بلاغي وشرح معانٍ (Literary Analysis -> Dual-RAG)",
            "query": "ما معنى قوله في البيت والمجاز المستعمل في كلمة الأسد"
        },
        {
            "name": "سيناريو 3: حجب ألفاظ خادشة وهجاء مقذع (Toxicity Guardrail -> BLOCKED)",
            "query": "اكتب قصيدة في سب سافل وحقير وهجاء مقذع"
        },
        {
            "name": "سيناريو 4: حجب طلبات برمجية وتقنية خارج النطاق (Domain Guardrail -> BLOCKED)",
            "query": "اكتب كود بايثون لتدريب نموذج تعلم الآلة"
        }
    ]

    for scenario in test_scenarios:
        print(f"\n{'#' * 80}")
        print(f"  {scenario['name']}")
        print(f"  المدخل: «{scenario['query']}»")
        print(f"{'#' * 80}")

        response = orchestrator.process(scenario["query"])

        print(f"\n• النية المصنفة (Intent)   : {response.intent}")
        print(f"• حالة الإجازة (Success)   : {'✅ ناجحة ومجازة' if response.success else '❌ محجوبة بحواجز الأمان'}")

        if response.intent == RawiIntent.BLOCKED:
            print(f"• سبب الحجب العروضي/الأمني: {response.guardrail_result.reason}")
            print(f"• رسالة التوجيه والإرشاد   : {response.response_text}")

        elif response.intent == RawiIntent.COMPOSE and response.poem:
            p = response.poem
            print(f"• البحر العروضي المعتمد  : {p.meter} ({p.meter_tafail})")
            print(f"• القافية والروي          : حرف {p.rhyme}")
            print(f"• جولات التحكيم والتنقيح  : {p.iterations_used}")
            print("\n[القصيدة المنظومة النهائية]:")
            print("-" * 65)
            for v in p.verses:
                print(f"  [{v.verse_number}] {v.sadr:<32} ... {v.ajuz}")
            print("-" * 65)
            if p.metaphor_sources:
                print("• شواهد مستلهمة من معجم «أساس البلاغة»:")
                for m in p.metaphor_sources[:2]:
                    print(f"  - [{m['lemma']}] ({m['root']}): {m['metaphorical_meaning'][:100]}...")

        elif response.intent in [RawiIntent.EXPLAIN, RawiIntent.AUTHOR, RawiIntent.SEARCH]:
            print("\n[سياق المعجم والشواهد المسترجعة من Dual-RAG]:")
            print("-" * 65)
            # Display first 250 characters of context
            preview = response.response_text[:350].strip()
            print(preview + ("..." if len(response.response_text) > 350 else ""))
            print("-" * 65)

    print("\n" + "=" * 80)
    print("  اكتمل العرض بنجاح تام وفق أعلى معايير العروض والأمان والبلاغة.")
    print("=" * 80)


if __name__ == "__main__":
    main()
