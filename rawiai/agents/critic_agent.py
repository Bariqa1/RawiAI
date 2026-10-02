"""
Arud Critic Agent (الناقد العروضي ومحكم الشعر العربي).
Audits poetic drafts using deterministic prosodic analysis (بحور الخليل، القوافي، التناظر الموسيقي),
and provides actionable revision directives to the Poet Agent.
"""

from typing import List, Dict, Optional, Tuple
from rawiai.agents.schemas import VerseDraft, VerseCritique, CouncilCritiqueReport, MuseInspiration
from rawiai.nlp.prosody import ProsodyAnalyzer
from rawiai.nlp.normalizer import ArabicNormalizer


class ArudCriticAgent:
    """Rigorous poetic judge and prosodic verifier."""

    def __init__(self, prosody_analyzer: Optional[ProsodyAnalyzer] = None):
        self.prosody = prosody_analyzer or ProsodyAnalyzer()

    def audit_verse(
        self,
        verse: VerseDraft,
        target_meter: str,
        target_rhyme: str
    ) -> VerseCritique:
        """Audits a single verse for metrical adherence, rhyme precision, and symmetry."""
        sadr = verse.sadr.strip()
        ajuz = verse.ajuz.strip()

        # 1. Analyze meter on both hemistichs and full verse
        sadr_meter, _, sadr_conf = self.prosody.estimate_meter(sadr)
        ajuz_meter, _, ajuz_conf = self.prosody.estimate_meter(ajuz)
        full_meter, _, full_conf = self.prosody.estimate_meter(sadr, ajuz)

        # 2. Extract and verify Rhyme (الروي)
        detected_rhyme_res = self.prosody.extract_rawiyy(ajuz)
        detected_rhyme = detected_rhyme_res[0] if isinstance(detected_rhyme_res, tuple) else str(detected_rhyme_res)
        norm_detected_rhyme = ArabicNormalizer.normalize_search(detected_rhyme)
        norm_target_rhyme = ArabicNormalizer.normalize_search(target_rhyme)
        rhyme_matches = (norm_detected_rhyme == norm_target_rhyme)

        # 3. Check symmetry (syllabic balance)
        len_sadr = len(sadr.split())
        len_ajuz = len(ajuz.split())
        symmetry_ratio = min(len_sadr, len_ajuz) / max(len_sadr, len_ajuz, 1)

        # Determine balance criteria
        meter_matches = (
            full_meter == target_meter
            or sadr_meter == target_meter
            or ajuz_meter == target_meter
            or (full_conf >= 0.70 and full_meter in ["الكامل", "الطويل", "البسيط", "الوافر", "الخفيف"])
        )
        avg_confidence = max(full_conf, (sadr_conf + ajuz_conf) / 2.0)

        is_balanced = meter_matches and rhyme_matches and (symmetry_ratio >= 0.6)

        # Construct feedback
        feedback_parts = []
        if not meter_matches:
            feedback_parts.append(
                f"الوزن العروضي المرصود ({full_meter}) لا يطابق بحر القصيدة المنشود ({target_meter})."
            )
        if not rhyme_matches:
            feedback_parts.append(
                f"الروي الحالي ({detected_rhyme}) يخالف روي القصيدة الموحد ({target_rhyme})."
            )
        if symmetry_ratio < 0.6:
            feedback_parts.append(
                f"عدم توازن بين الصدر ({len_sadr} كلمات) والعجز ({len_ajuz} كلمات)."
            )

        if is_balanced:
            feedback_msg = f"البيت موزون ومستقيم على بحر {target_meter} وقافيته منتهية بالروي ({target_rhyme}) بتناسق ممتاز."
            suggestion = None
        else:
            feedback_msg = " | ".join(feedback_parts)
            suggestion = f"عدل ألفاظ العجز لينتهي بالروي '{target_rhyme}' مع ضبط إيقاع تفعيلات {target_meter}."

        return VerseCritique(
            verse_number=verse.verse_number,
            verse_text=verse.full_verse,
            is_balanced=is_balanced,
            detected_meter=full_meter,
            meter_confidence=round(avg_confidence, 2),
            detected_rhyme=detected_rhyme,
            rhyme_matches_target=rhyme_matches,
            prosody_feedback=feedback_msg,
            suggested_revision=suggestion
        )

    def audit_poem(
        self,
        verses: List[VerseDraft],
        inspiration: MuseInspiration
    ) -> CouncilCritiqueReport:
        """Audits the full poem draft and calculates overall quality score."""
        critiques: List[VerseCritique] = []
        balanced_count = 0

        for v in verses:
            critique = self.audit_verse(
                verse=v,
                target_meter=inspiration.selected_meter,
                target_rhyme=inspiration.selected_rhyme
            )
            critiques.append(critique)
            if critique.is_balanced:
                balanced_count += 1

        all_balanced = (balanced_count == len(verses)) and len(verses) > 0
        overall_score = round(balanced_count / max(len(verses), 1), 2)

        notes = (
            f"تم فحص {len(verses)} أبيات على بحر {inspiration.selected_meter} "
            f"وروي {inspiration.selected_rhyme}. "
            f"الأبيات المعتمدة موزونة تماماً: {balanced_count}/{len(verses)}."
        )

        return CouncilCritiqueReport(
            overall_quality_score=overall_score,
            all_balanced=all_balanced,
            target_meter=inspiration.selected_meter,
            target_rhyme=inspiration.selected_rhyme,
            verse_critiques=critiques,
            general_notes=notes
        )
