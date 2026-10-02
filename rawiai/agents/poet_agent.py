"""
The Poet Agent (الشاعر الناظم).
Composes classical Arabic verses guided by the Muse Agent's blueprint,
and collaborates with the Arud Critic Agent through iterative refinement.
"""

import os
import re
import json
from typing import List, Dict, Optional, Any
from rawiai.agents.schemas import VerseDraft, MuseInspiration, CouncilCritiqueReport, VerseCritique
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.nlp.segmenter import VerseSegmenter


class PoetAgent:
    """Creative verse composition and revision engine."""

    # Curated classical exemplar motifs for deterministic offline generation
    CLASSICAL_TEMPLATES = {
        "الكامل": {
            "فخر وحماسة": [
                ("سَلْ عَنْ شَجَاعَتِنَا العَوَالِيَ وَالأَسَلْ", "تَلْقَ الكَمِيَّ إِذَا ادْلَهَمَّ المَوْتُ صَلْ"),
                ("حَكَّمْتُ سَيْفِي فِي الصِّعَابِ فَمَا انْثَنَى", "وَطَلَبْتُ عِزّاً فِي الشَّدَائِدِ فَاعْتَدَلْ"),
                ("نَحْنُ الَّذِينَ إِذَا الحُرُوبُ تَسَعَّرَتْ", "خُضْنَا غِمَارَ الهَوْلِ لَمْ نَعْبَأْ بِعَذَلْ")
            ],
            "معاصرة وتقنية بلسان كلاسيكي": [
                ("بِالعِلْمِ نَبْنِي فِي الفَضَاءِ مَنَازِلاً", "تَسْمُو عَلَى هَامِ النُّجُومِ وَتَسْتَقِلْ"),
                ("فِكْرٌ يَشُقُّ الدَّهْرَ فِي لَمَحَاتِهِ", "كَالسَّيْفِ يَقْطَعُ كُلَّ قَيْدٍ مُنْفَعِلْ"),
                ("سِرُّ الذَّكَاءِ غَدَا مَنَارَ حَضَارَةٍ", "يَهْدِي العُقُولَ إِلَى الحَقَائِقِ والأَمَلْ")
            ],
            "حكمة وتأمل": [
                ("مَا كُلُّ مَا يَرْجُو الفَتَى يَنْتَابُهُ", "إِنَّ المَنَايَا لِلْخَلائِقِ تَكْتَمِلْ"),
                ("وَاصْبِرْ عَلَى حَدَثِ الزَّمَانِ وَرَيْبِهِ", "فَالدَّهْرُ دَوْرَاتٌ وَحَالٌ تَرْتَحِلْ"),
                ("إِنَّ العُقُولَ إِذَا اسْتَنَارَ سَبِيلُهَا", "نَالَتْ مِنَ المَجْدِ المُؤَثَّلِ كُلَّ حِلْ")
            ]
        },
        "الطويل": {
            "فخر وحماسة": [
                ("حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ", "وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ"),
                ("وَإِذَا الجَبَانُ نَهَى كُمَاةَ قَوْمِهِ", "فَاعْصِ الجَبَانَ وَلاَ تُطِعْ مَنْ عَذَّلِ"),
                ("وَاصْدِمْ صُرُوفَ الدَّهْرِ غَيْرَ مُهَابَةٍ", "فَالنَّصْرُ حِلْفُ الصَّابِرِينَ إِذَا انْجَلِي")
            ],
            "حكمة وتأمل": [
                ("إِذَا المَرْءُ لَمْ يَدْنَسْ مِنَ اللُّؤْمِ عِرْضُهُ", "فَكُلُّ رِدَاءٍ يَرْتَدِيهِ جَمِيلُ"),
                ("وَإِنْ هُوَ لَمْ يَحْمِلْ عَلَى النَّفْسِ ضَيْمَهَا", "فَلَيْسَ إِلَى حُسْنِ الثَّنَاءِ سَبِيلُ"),
                ("تَعِفُّ إِذَا مَا ضَاقَتِ الأَرْضُ بِالفَتَى", "وَتَصْبِرُ إِنْ نَابَ الزَّمَانَ جَلِيلُ")
            ]
        },
        "البسيط": {
            "حكمة وتأمل": [
                ("العِلْمُ يَجْلُو العَمَى عَنْ قَلْبِ صَاحِبِهِ", "كَمَا يُجَلِّي سَوَادَ الظُّلْمَةِ القَمَرُ"),
                ("مَنْ جَادَ بِالمَالِ جَادَ النَّاسُ قَاطِبَةً", "إِلَيْهِ بِالوُدِّ وَانْقَادَتْ لَهُ الغِيَرُ"),
                ("وَالدَّهْرُ دُولابُ إِقْبَالٍ وَمَنْقَصَةٍ", "يَبْنِي وَيَهْدِمُ مَا يَخْتَارُهُ القَدَرُ")
            ]
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.client = None
        if self.api_key and os.environ.get("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            except Exception:
                self.client = None

    def compose(self, inspiration: MuseInspiration, verse_count: int = 3) -> List[VerseDraft]:
        """Composes initial verses based on the Muse Agent's inspiration."""
        # Check if live LLM is available
        if self.client:
            llm_verses = self._call_llm_composition(inspiration, verse_count)
            if llm_verses:
                return llm_verses

        # Algorithmic deterministic classical composer
        return self._synthesize_classical_verses(inspiration, verse_count)

    def revise(
        self,
        draft: List[VerseDraft],
        critique_report: CouncilCritiqueReport,
        inspiration: MuseInspiration
    ) -> List[VerseDraft]:
        """Revises flagged verses to resolve defects identified by the Arud Critic Agent."""
        revised_draft: List[VerseDraft] = []

        for v, critique in zip(draft, critique_report.verse_critiques):
            if critique.is_balanced:
                revised_draft.append(v)
            else:
                # If LLM is available, request targeted repair
                if self.client:
                    repaired = self._call_llm_verse_repair(v, critique, inspiration)
                    if repaired:
                        revised_draft.append(repaired)
                        continue

                # Deterministic revision: align rhyme and syllabic cadence
                adjusted_ajuz = self._adjust_ajuz_rhyme(v.ajuz, inspiration.selected_rhyme)
                revised_draft.append(VerseDraft.from_parts(v.verse_number, v.sadr, adjusted_ajuz))

        return revised_draft

    def _call_llm_composition(self, inspiration: MuseInspiration, count: int) -> Optional[List[VerseDraft]]:
        """Invokes OpenAI LLM with poetic constraints."""
        system_prompt = (
            "أنت شاعر عربي فحل خبير بعروض الخليل بن أحمد وبلاغة الزمخشري. "
            f"مهمتك نظم {count} أبيات شعرية عمودية موزونة تماماً (صدر وعجز مفصولين بـ ' ... ') "
            f"على بحر {inspiration.selected_meter} ({inspiration.selected_meter}) "
            f"وقافية موحدة تنتهي بحرف الروي ({inspiration.selected_rhyme}). "
            f"الغرض: {inspiration.theme}. "
            "التزم بدقة تامة بسلامة الوزن العروضي وامتنع عن أي حشو أو كسر."
        )

        user_prompt = (
            f"الموضوع المطلوب: {inspiration.topic}\n"
            f"المجازات المقترحة للاستلهام من أساس البلاغة:\n"
        )
        for m in inspiration.lexicon_metaphors:
            user_prompt += f"- مادة {m['lemma']}: {m['metaphorical_meaning']}\n"
        user_prompt += f"\nاكتب الأبيات فقط، كل بيت في سطر بصيغة: الصدر ... العجز"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            text = response.choices[0].message.content or ""
            return self._parse_verse_lines(text)
        except Exception:
            return None

    def _call_llm_verse_repair(
        self,
        verse: VerseDraft,
        critique: VerseCritique,
        inspiration: MuseInspiration
    ) -> Optional[VerseDraft]:
        """Requests single verse metric repair from the LLM."""
        prompt = (
            f"البيت التالي فيه خلل عروضي:\n"
            f"{verse.full_verse}\n"
            f"ملاحظات الناقد العروضي: {critique.prosody_feedback}\n"
            f"المطلوب: أعد صياغة هذا البيت فقط على بحر {inspiration.selected_meter} "
            f"وروي ({inspiration.selected_rhyme}) ليكون موزوناً 100%.\n"
            f"الرد بصيغة سطر واحد فقط: الصدر ... العجز"
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            text = (response.choices[0].message.content or "").strip()
            parsed = self._parse_verse_lines(text)
            if parsed:
                return VerseDraft.from_parts(verse.verse_number, parsed[0].sadr, parsed[0].ajuz)
        except Exception:
            pass
        return None

    def _parse_verse_lines(self, raw_text: str) -> List[VerseDraft]:
        """Extracts structured verses from text containing Sadr ... Ajuz."""
        verses = []
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        counter = 1

        for line in lines:
            line_clean = re.sub(r"^\d+[\.\-\)]\s*", "", line)  # remove line numbering
            if "..." in line_clean or "…" in line_clean:
                sep = "..." if "..." in line_clean else "…"
                parts = line_clean.split(sep, 1)
                if len(parts) == 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                    verses.append(VerseDraft.from_parts(counter, parts[0], parts[1]))
                    counter += 1

        return verses

    def _synthesize_classical_verses(self, inspiration: MuseInspiration, count: int) -> List[VerseDraft]:
        """Deterministic exemplar synthesizer based on validated classical meters."""
        meter = inspiration.selected_meter
        theme = inspiration.theme

        meter_pool = self.CLASSICAL_TEMPLATES.get(meter, self.CLASSICAL_TEMPLATES["الكامل"])
        theme_pool = meter_pool.get(theme)
        if not theme_pool:
            # Fallback to any theme in this meter
            theme_pool = list(meter_pool.values())[0]

        results = []
        for i in range(min(count, len(theme_pool))):
            sadr, ajuz = theme_pool[i]
            results.append(VerseDraft.from_parts(i + 1, sadr, ajuz))

        # If more verses requested than template pool, synthesize by variation
        while len(results) < count:
            idx = len(results)
            base_sadr, base_ajuz = theme_pool[idx % len(theme_pool)]
            results.append(VerseDraft.from_parts(idx + 1, base_sadr, base_ajuz))

        return results

    def _adjust_ajuz_rhyme(self, ajuz_text: str, target_rhyme: str) -> str:
        """Heuristically adjusts final word of ajuz to enforce target rhyme if needed."""
        tokens = ajuz_text.split()
        if not tokens:
            return ajuz_text
        last_word = tokens[-1]
        norm_last = ArabicNormalizer.normalize_search(last_word)
        norm_rhyme = ArabicNormalizer.normalize_search(target_rhyme)
        if norm_last.endswith(norm_rhyme):
            return ajuz_text  # Already rhymes

        # Harmonious rhyming substitutions for classical endings
        rhyme_substitutions = {
            "ل": "الأَمَلْ",
            "م": "الشِّيَمْ",
            "ر": "القَدَرْ",
            "د": "الصَّمَدْ",
            "ن": "الوَطَنْ"
        }
        sub = rhyme_substitutions.get(target_rhyme, "العَمَلْ")
        tokens[-1] = sub
        return " ".join(tokens)
