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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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
            ],
            "بر الوالدين ووجدان الأمومة": [
                ("أُمِّي رِيَاضُ الحُبِّ نَبْعُ حَنَانِنَا", "فِي فَيْئِهَا نَلْقَى الأَمَانَ وَنَسْتَظِلْ"),
                ("حَمَلَتْ فُؤَادِي فِي الشَّدَائِدِ رَحْمَةً", "تَسْقِي حَيَاتِي بِالوِدَادِ وَتَشْتَمِلْ"),
                ("نَفْدِي الرَّءُومَ بِكُلِّ غَالٍ عِنْدَنَا", "فَالجُودُ فِي كَفِّ الحَبِيبَةِ مُكْتَمِلْ")
            ],
            "وصف وطبيعة": [
                ("قَدْ فَاحَ عِطْرُ الرَّوْضِ فِي إِشْرَاقِهِ", "وَاسْتَيْقَظَتْ بَيْنَ الغُصُونِ بَلاَبِلُ"),
                ("وَالصَّفْوُ فِي كَأْسِ الصَّبَاحِ سَعَادَةٌ", "تَجْلُو الهُمُومَ عَنِ الفُؤَادِ وَتَشْمَلُ"),
                ("قَهْوَى الصَّبَاحِ سَلِيلَةٌ لِصَفَائِنَا", "تَشْفِي النُّفُوسَ مِنَ العَنَاءِ وَتَفْضُلُ")
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
            ],
            "بر الوالدين ووجدان الأمومة": [
                ("أَحِنُّ إِلَى أُمِّي إِذَا جَنَّنِي الدُّجَى", "وَأَرْجُو رِضَاهَا فَهْوَ فِي الحَشْرِ مَوْئِلِي"),
                ("لَقَدْ غَذَّتِ النَّفْسَ الكَرِيمَةَ بِالهُدَى", "وَفَاضَتْ بِعَطْفٍ كَالغَمَامِ المُهَلَّلِ"),
                ("فَيَا رَبِّ بَارِكْ فِي مَدَى عُمْرِهَا وَزِدْ", "لَهَا مِنْ جَزِيلِ الفَضْلِ فِي كُلِّ مَنْزِلِ")
            ],
            "وصف وطبيعة": [
                ("صَبَاحٌ بَدَا فِيهِ السُّرُورُ وَأَشْرَقَا", "وَفِنْجَانُ قَهْوٍ طَابَ ذَوْقاً وَرَاقَا"),
                ("يَفُوحُ شَذَاهَا بِالعَبِيرِ كَأَنَّهَا", "سُلافَةُ صَفْوٍ لَمْ تُخَالِطْ نِفَاقَا"),
                ("تُجَدِّدُ عَزْمَ المَرْءِ فِي كُلِّ بُكْرَةٍ", "وَتَجْلُو سَوَادَ الهَمِّ حِينَ تَلاَقَى")
            ]
        },
        "البسيط": {
            "حكمة وتأمل": [
                ("العِلْمُ يَجْلُو العَمَى عَنْ قَلْبِ صَاحِبِهِ", "كَمَا يُجَلِّي سَوَادَ الظُّلْمَةِ القَمَرُ"),
                ("مَنْ جَادَ بِالمَالِ جَادَ النَّاسُ قَاطِبَةً", "إِلَيْهِ بِالوُدِّ وَانْقَادَتْ لَهُ الغِيَرُ"),
                ("وَالدَّهْرُ دُولابُ إِقْبَالٍ وَمَنْقَصَةٍ", "يَبْنِي وَيَهْدِمُ مَا يَخْتَارُهُ القَدَرُ")
            ],
            "بر الوالدين ووجدان الأمومة": [
                ("أُمِّي ضِيَاءُ حَيَاتِي وَهْيَ نُورُ دَمِي", "وَفَيْضُ جُودٍ مِنَ الرَّحْمَنِ مُنْسَكِبُ"),
                ("فِي حِضْنِهَا نَشَأَتْ رُوحِي عَلَى كَرَمٍ", "وَكُلُّ خَيْرٍ أَتَى مِنْهَا لَهُ سَبَبُ"),
                ("بِرُّ الرَّءُومِ جِهَادٌ طَابَ مَغْنَمُهُ", "يُهْدِي الجِنَانَ وَتُجْلَى عِنْدَهُ الكُرَبُ")
            ]
        }
    }

    def __init__(self, api_key: Optional[str] = None, use_llm: bool = True):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        env_llm = os.environ.get("USE_LIVE_LLM", "").lower()
        if env_llm in ["0", "false", "no"]:
            self.use_llm = False
        elif not use_llm:
            self.use_llm = False
        else:
            self.use_llm = bool(self.api_key)

        self.client = None
        if self.use_llm and self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                print(f"[PoetAgent] Failed to init OpenAI client: {e}")
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
        """Invokes OpenAI LLM with strict poetic constraints, exemplars, and low temperature."""
        exemplar_verse = ""
        meter_pool = self.CLASSICAL_TEMPLATES.get(inspiration.selected_meter, {})
        for theme_verses in meter_pool.values():
            if theme_verses:
                exemplar_verse = f"{theme_verses[0][0]} ... {theme_verses[0][1]}"
                break

        system_prompt = (
            "أنت شاعر عربي فحل ومحكم خبير في بلاغة التراث وعروض الخليل بن أحمد.\n"
            f"مهمتك نظم {count} أبيات شعرية عمودية فصيحة، رصينة، وموزونة تماماً (صدر وعجز مفصولين بـ ' ... ')\n"
            f"على بحر {inspiration.selected_meter} (تفعيلاته لكل شطر: {inspiration.meter_tafail})\n"
            f"وقافية موحدة تنتهي بحرف الروي ({inspiration.selected_rhyme}).\n"
            f"الغرض الشعري: {inspiration.theme}.\n\n"
            "ضوابط الصياغة والمنطق (إلزامية صارمة):\n"
            "1. سلامة المنطق والمعنى: يجب أن تكون الأبيات منطقية، عذبة، ومترابطة وجدانياً مع الموضوع المطلوب. إياك والتشبيهات المتناقضة أو غير المعقولة (ممنوع القول بأن الأحجار تصدأ، أو تشبيه لسان الأخ بالسيف القاطع في الخصومة!). صب اهتمامك على جوهر الموضوع (المحبة، الوفاء، الأخوة، العقل، الحكمة).\n"
            "2. طبيعة القافية والكلمات: اختر كلمات فصيحة، مألوفة، وجميلة لقافية كل بيت بما يخدم المعنى طبيعياً. ممنوع منعاً كلياً حشر كلمات شاذة أو مبتذلة أو غير لائقة لمجرد التقفيل بحرف الروي (مثل: العلل، خفل، زفل).\n"
            "3. الوزن العروضي التام: طابق تفعيلات البحر بدقة تامة وبلا أي كسر.\n"
            "4. كمال الجملة: ممنوع قطع الشطر عند حرف جر (مثل: على، في، من، إلى) أو أداة عطف أو اسم موصول؛ يجب أن يكون كل شطر جملة فصيحة تامة ومكتملة المعنى.\n"
            "5. الشكل: ضع التشكيل الكامل بالحركات والسكنات على الكلمات لتأكيد الوزن الموسيقي.\n"
            "6. الصيغة: كل بيت في سطر مستقل مفصولاً بثلاث نقاط: الصدر ... العجز\n"
            "7. ممنوع كتابة أي مقدمات أو تحيات أو هوامش إطلاقاً؛ ابدأ فوراً بنص البيت الأول مباشرة."
        )
        if exemplar_verse:
            system_prompt += f"\n\nنموذج عروضي موزون على بحر {inspiration.selected_meter} للمحاكاة:\n{exemplar_verse}"

        user_prompt = (
            f"الموضوع المطلوب: {inspiration.topic}\n"
            "تنبيه حول الاستلهام: صب اهتمامك وإبداعك على جوهر الموضوع وعاطفته ومفرداته اللائقة به حصراً.\n"
        )
        if inspiration.lexicon_metaphors:
            user_prompt += "شواهد معجمية للاستئناس البلاغي (اختيارية وغير ملزمة إن لم تخدم المعنى الطبيعي):\n"
            for m in inspiration.lexicon_metaphors:
                user_prompt += f"- مادة {m['lemma']}: {m['metaphorical_meaning']}\n"
        user_prompt += f"\nانظم {count} أبيات مترابطة، بليغة، وموزونة تماماً الآن بالتشكيل التام."

        try:
            model_name = os.environ.get("POET_LLM_MODEL", "gpt-4o")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            text = response.choices[0].message.content or ""
            verses = self._parse_verse_lines(text)
            if verses and len(verses) >= count:
                return verses[:count]
            return verses if verses else None
        except Exception:
            return None

    def _call_llm_verse_repair(
        self,
        verse: VerseDraft,
        critique: VerseCritique,
        inspiration: MuseInspiration
    ) -> Optional[VerseDraft]:
        """Requests single verse metric repair from the LLM with low temperature."""
        prompt = (
            f"البيت التالي فيه خلل عروضي أو نحوي:\n"
            f"{verse.full_verse}\n"
            f"ملاحظات الناقد العروضي: {critique.prosody_feedback}\n"
            f"المطلوب: أعد صياغة هذا البيت فقط ليكون موزوناً 100% على بحر {inspiration.selected_meter} "
            f"(تفعيلاته: {inspiration.meter_tafail}) "
            f"وروي ({inspiration.selected_rhyme}) مع التشكيل التام بالحركات والالتزام بالنحو السليم واكتمال المعنى دون أي قطع في الكلام.\n"
            f"الرد بصيغة سطر واحد فقط: الصدر ... العجز"
        )
        try:
            model_name = os.environ.get("POET_LLM_MODEL", "gpt-4o")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            text = (response.choices[0].message.content or "").strip()
            parsed = self._parse_verse_lines(text)
            if parsed:
                return VerseDraft.from_parts(verse.verse_number, parsed[0].sadr, parsed[0].ajuz)
        except Exception:
            pass
        return None

    def _parse_verse_lines(self, raw_text: str) -> List[VerseDraft]:
        """Extracts structured verses from text containing Sadr ... Ajuz or separators."""
        verses = []
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        counter = 1

        conversational_markers = [
            "إليك", "اليك", "تفضل", "أبيات", "ابيات", "قصيدة", "شعر", "بحر", "قافية", "ملاحظة"
        ]

        for line in lines:
            line_clean = re.sub(r"^[0-9٠-٩]+[\.\-\)\:\s]*", "", line).strip()  # remove line numbering
            line_clean = re.sub(r"[\*\_]", "", line_clean).strip()
            line_clean = re.sub(r"^(?:البيت\s*[0-9٠-٩]*\s*[:\-]\s*)", "", line_clean).strip()

            # Skip conversational introductions and headers
            if line_clean.endswith(":") and any(m in line_clean for m in conversational_markers):
                continue
            if any(line_clean.startswith(m) for m in ["إليك", "اليك", "تفضل", "هاك"]):
                continue

            # Split on common classical separators
            parts = re.split(r"\s*(?:\.{3,}|…|#|/|\||\s-\s|\s—\s|\s؛\s)\s*", line_clean)
            if len(parts) >= 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                sadr_part = re.sub(r"^[0-9٠-٩]+[\.\-\)\:\s]*", "", parts[0].strip()).strip()
                ajuz_part = re.sub(r"^[0-9٠-٩]+[\.\-\)\:\s]*", "", parts[1].strip()).strip()
                verses.append(VerseDraft.from_parts(counter, sadr_part, ajuz_part))
                counter += 1
            else:
                words = line_clean.split()
                if len(words) >= 6 and not any(m in line_clean for m in conversational_markers):
                    mid = len(words) // 2
                    sadr = re.sub(r"^[0-9٠-٩]+[\.\-\)\:\s]*", "", " ".join(words[:mid])).strip()
                    ajuz = re.sub(r"^[0-9٠-٩]+[\.\-\)\:\s]*", "", " ".join(words[mid:])).strip()
                    verses.append(VerseDraft.from_parts(counter, sadr, ajuz))
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
