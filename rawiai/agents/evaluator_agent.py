"""
Human Poetry Evaluator & Literary Critic Agent (محكم الشعر والناقد الأدبي).
Evaluates poetry submitted by human authors or external systems,
benchmarking meter soundness, rhyme precision, and poetic defects against
'Mizan Al-Dhahab fi Sina'at Shi'r Al-Arab' by Ahmad Al-Hashimi.
"""

import os
import re
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

from rawiai.agents.schemas import SingleVerseEvaluation, PoeticEvaluationReport
from rawiai.rag.prosody_rules_retriever import ProsodyRulesRetriever
from rawiai.nlp.prosody import ProsodyAnalyzer
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.nlp.segmenter import VerseSegmenter


class HumanPoetryEvaluator:
    """
    Expert Literary & Prosodic Judge for Arabic Poetry.
    Provides diagnostic audits, defect detection, and remediation
    grounded in 'Mizan Al-Dhahab'.
    """

    def __init__(
        self,
        prosody_retriever: Optional[ProsodyRulesRetriever] = None,
        prosody_analyzer: Optional[ProsodyAnalyzer] = None,
        use_llm: bool = False,
        api_key: Optional[str] = None
    ):
        self.rules = prosody_retriever or ProsodyRulesRetriever()
        self.prosody = prosody_analyzer or ProsodyAnalyzer()
        self.use_llm = use_llm or (os.environ.get("USE_LIVE_LLM", "").lower() in ["1", "true", "yes"])
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None

        if self.use_llm and self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def evaluate_text(self, text: str) -> PoeticEvaluationReport:
        """
        Conducts a comprehensive, end-to-end evaluation of poetry submitted by a user.
        """
        # Step 1: Segment text into verses (Sadr, Ajuz)
        parsed_verses = self._extract_user_verses(text)
        if not parsed_verses:
            # Fallback if no explicit separator: try to treat lines or commas
            lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 5]
            for idx, l in enumerate(lines, 1):
                # If space separated with many words, split in half
                words = l.split()
                mid = len(words) // 2
                sadr = " ".join(words[:mid]) if mid > 0 else l
                ajuz = " ".join(words[mid:]) if mid > 0 else l
                parsed_verses.append((idx, sadr, ajuz))

        if not parsed_verses:
            return self._build_empty_report(text)

        # Step 2: Per-verse prosodic and rhyme inspection
        verse_evals: List[SingleVerseEvaluation] = []
        meter_votes = []
        rhyme_votes = []

        for num, sadr, ajuz in parsed_verses:
            s_meter, _, s_conf = self.prosody.estimate_meter(sadr)
            a_meter, _, a_conf = self.prosody.estimate_meter(ajuz)
            full_verse = f"{sadr} ... {ajuz}"
            f_meter, _, f_conf = self.prosody.estimate_meter(sadr, ajuz)

            # Determine likely meter for this verse
            chosen_meter = f_meter if f_conf >= 0.70 else (s_meter if s_conf >= a_conf else a_meter)
            avg_conf = f_conf
            meter_votes.append(chosen_meter)

            # Extract rhyme
            rhyme_res = self.prosody.extract_rawiyy(ajuz)
            rhyme_letter = rhyme_res[0] if isinstance(rhyme_res, tuple) else str(rhyme_res)
            rhyme_votes.append(rhyme_letter)

            verse_evals.append(SingleVerseEvaluation(
                verse_number=num,
                sadr=sadr,
                ajuz=ajuz,
                full_verse=full_verse,
                detected_meter=chosen_meter,
                meter_confidence=round(avg_conf, 2),
                is_meter_valid=False,  # Will be finalized once dominant meter is determined
                sadr_meter=s_meter,
                ajuz_meter=a_meter,
                detected_rhyme=rhyme_letter,
                is_rhyme_valid=False,
                prosodic_status="",
                defects=[],
                remediation_suggestion=None
            ))

        # Step 3: Determine Dominant Meter & Dominant Rhyme of the poem
        dominant_meter = Counter(meter_votes).most_common(1)[0][0]
        dominant_rhyme = Counter(rhyme_votes).most_common(1)[0][0]

        # Retrieve authoritative rules from Mizan Al-Dhahab
        meter_info = self.rules.get_meter(dominant_meter) or {}
        meter_tafail = meter_info.get("standard_tafail", "تفاعيل قياسية")
        mnemonic_key = meter_info.get("mnemonic_key", "غير متوفر")

        # Step 4: Finalize audits against dominant meter and rhyme, diagnose defects
        sound_count = 0
        defects_manifest: Dict[str, Dict[str, str]] = {}
        seen_rhyme_words = {}

        for v in verse_evals:
            len_s = len(v.sadr.split())
            len_a = len(v.ajuz.split())
            symmetry = min(len_s, len_a) / max(len_s, len_a, 1)

            # A verse is metrically sound if its combined metric confidence meets classical threshold (>= 0.80),
            # matches the dominant meter, and exhibits balanced hemistich symmetry (>= 0.60).
            meter_valid = (
                (v.detected_meter == dominant_meter)
                and (v.meter_confidence >= 0.80)
                and (symmetry >= 0.60)
            )
            v.is_meter_valid = meter_valid

            # Check rhyme validity
            norm_v_rhyme = ArabicNormalizer.normalize_search(v.detected_rhyme)
            norm_dom_rhyme = ArabicNormalizer.normalize_search(dominant_rhyme)
            rhyme_valid = (norm_v_rhyme == norm_dom_rhyme)
            v.is_rhyme_valid = rhyme_valid

            # Diagnose defects
            if not meter_valid:
                v.defects.append("كسر عروضي")
                v.prosodic_status = f"خلل في تفعيلات بحر {dominant_meter}"
                defect_data = self.rules.get_defect_info("الكسر العروضي")
                if defect_data:
                    defects_manifest["الكسر العروضي"] = defect_data
                v.remediation_suggestion = f"إعادة صياغة الألفاظ لتطابق تفعيلات {meter_tafail}"
            else:
                v.prosodic_status = f"مستقيم وموزون على بحر {dominant_meter}"

            if not rhyme_valid:
                v.defects.append("اختلاف حرف الروي (إكفاء)")
                defect_data = self.rules.get_defect_info("الإكفاء")
                if defect_data:
                    defects_manifest["الإكفاء"] = defect_data
                v.remediation_suggestion = (
                    (v.remediation_suggestion or "") + f" | ضبط نهاية العجز بالروي '{dominant_rhyme}'"
                ).strip(" | ")

            # Check for Ita (تكرار القافية)
            ajuz_tokens = v.ajuz.split()
            if ajuz_tokens:
                last_w = ArabicNormalizer.normalize_search(ajuz_tokens[-1])
                if last_w in seen_rhyme_words:
                    prev_line = seen_rhyme_words[last_w]
                    if v.verse_number - prev_line < 7:
                        v.defects.append("إيطاء (تكرار كلمة القافية)")
                        defect_data = self.rules.get_defect_info("الإيطاء")
                        if defect_data:
                            defects_manifest["الإيطاء"] = defect_data
                seen_rhyme_words[last_w] = v.verse_number

            if v.is_meter_valid and v.is_rhyme_valid:
                sound_count += 1

        total = len(verse_evals)
        broken_count = total - sound_count
        score = round((sound_count / max(total, 1)) * 100.0, 1)
        is_fully_sound = (sound_count == total) and total > 0

        # Step 5: Rhetorical analysis & general critique
        general_notes, rhetorical_critique, remedies = self._synthesize_literary_critique(
            verse_evals=verse_evals,
            dominant_meter=dominant_meter,
            dominant_rhyme=dominant_rhyme,
            score=score,
            is_sound=is_fully_sound,
            defects=list(defects_manifest.keys())
        )

        return PoeticEvaluationReport(
            original_input=text,
            total_verses=total,
            dominant_meter=dominant_meter,
            meter_tafail=meter_tafail,
            meter_mnemonic_key=mnemonic_key,
            dominant_rhyme=dominant_rhyme,
            sound_verses_count=sound_count,
            broken_verses_count=broken_count,
            overall_score=score,
            is_fully_sound=is_fully_sound,
            verse_evaluations=verse_evals,
            poetic_defects_found=list(defects_manifest.values()),
            general_critique=general_notes,
            rhetorical_analysis=rhetorical_critique,
            remedy_recommendations=remedies
        )

    def _extract_user_verses(self, raw_text: str) -> List[Tuple[int, str, str]]:
        """Parses verses using explicit delimiters ('...', '…', '=', '*', or newline splits)."""
        verses = []
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        counter = 1

        for line in lines:
            # Remove any leading numbers (e.g., 1. or 1-)
            line_clean = re.sub(r"^\d+[\.\-\)]\s*", "", line).strip()
            # Check standard separators
            for sep in ["...", "…", " - ", " = ", " * ", " # "]:
                if sep in line_clean:
                    parts = line_clean.split(sep, 1)
                    if len(parts) == 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                        verses.append((counter, parts[0].strip(), parts[1].strip()))
                        counter += 1
                        break
        return verses

    def _synthesize_literary_critique(
        self,
        verse_evals: List[SingleVerseEvaluation],
        dominant_meter: str,
        dominant_rhyme: str,
        score: float,
        is_sound: bool,
        defects: List[str]
    ) -> Tuple[str, str, List[str]]:
        """Synthesizes human-readable critique and remediation guidance."""
        remedies = []
        if is_sound:
            general = (
                f"أحسنت القول! الأبيات جارية على سنن العرب وموزونة تماماً على بحر {dominant_meter}، "
                f"والتزمت روي ({dominant_rhyme}) دون نشوز عروضي أو عيب من عيوب القافية."
            )
            rhetoric = (
                "الألفاظ متناسقة وجرس الموسيقى الداخلية منسجم مع وقار البحر. "
                "القصيدة تتمتع بسبك متماسك ووضوح في الدلالة."
            )
            remedies.append("القصيدة معتمدة عروضياً ومكتملة الأركان.")
        else:
            general = (
                f"القصيدة مبنية في أصلها على بحر {dominant_meter}، لكن رُصد خلل عروضي "
                f"في {len(verse_evals) - int(score / 100 * len(verse_evals))} شطر/بيت. "
                f"الأبيات تحتاج إلى إعادة موازنة للحركات والسكنات لتستقيم تفعيلاتها."
            )
            rhetoric = (
                "تراكيب المعاني واضحة ومقاصد الأبيات نبيلة، ولكن التفاوت في عدد المقاطع الصوتية "
                "أدى إلى انكسار الجرس الصوتي في مواضع محددة تم تبيانها في التقرير."
            )
            if "كسر عروضي" in defects:
                remedies.append(f"مراجعة التفعيلات المكسورة ومطابقتها لميزان بحر {dominant_meter}.")
            if "الإكفاء" in defects:
                remedies.append(f"توحيد حرف الروي ليكون ({dominant_rhyme}) في سائر الأبيات.")
            if "الإيطاء" in defects:
                remedies.append("تنويع قوافي الأعجاز وتجنب تكرار الكلمة نفسها قبل 7 أبيات.")

        # If LLM is available, enrich with AI literary touch
        if self.client:
            llm_critique = self._call_llm_literary_critique(verse_evals, dominant_meter, dominant_rhyme)
            if llm_critique:
                rhetoric = llm_critique

        return general, rhetoric, remedies

    def _call_llm_literary_critique(
        self,
        verse_evals: List[SingleVerseEvaluation],
        meter: str,
        rhyme: str
    ) -> Optional[str]:
        """Calls OpenAI LLM to provide rhetorical and stylistic literary review."""
        verses_str = "\n".join([f"{v.sadr} ... {v.ajuz}" for v in verse_evals])
        prompt = (
            "أنت ناقد أدبي خبير في بلاغة الشعر العربي ومناهجه التراثية (كابن رشيق والزمخشري). "
            f"إليك هذه الأبيات المنظومة على بحر {meter} وروي {rhyme}:\n"
            f"{verses_str}\n\n"
            "قدم نقداً بلاغياً وأسلوبياً موجزاً ومركزاً في فقرة واحدة (3-4 أسطر): "
            "تناول فصاحة الألفاظ، حسن الاستعارة والتشبيه، ومدى تماسك السبك الشعري."
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

    def _build_empty_report(self, text: str) -> PoeticEvaluationReport:
        """Returns a polite notification when input cannot be parsed as verse."""
        return PoeticEvaluationReport(
            original_input=text,
            total_verses=0,
            dominant_meter="غير محدد",
            meter_tafail="غير محدد",
            meter_mnemonic_key="غير محدد",
            dominant_rhyme="غير محدد",
            sound_verses_count=0,
            broken_verses_count=0,
            overall_score=0.0,
            is_fully_sound=False,
            verse_evaluations=[],
            poetic_defects_found=[],
            general_critique="لم يتم العثور على أبيات شعرية واضحة (يُفضل كتابة كل بيت بصيغة: الصدر ... العجز).",
            rhetorical_analysis="",
            remedy_recommendations=["يرجى إدخال الشعر مفصولاً بنقاط أو أسطر مستقلة ليتمكن الناقد من تحكيمه بدقة."]
        )
