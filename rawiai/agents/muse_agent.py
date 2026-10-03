"""
The Muse Agent (عميل الإلهام وبلاغة المعجم).
Analyzes user topics, determines poetic themes, selects classical meters and rhymes,
and extracts rhetorical motifs from 'Asas Al-Balagha'.
"""

from typing import List, Dict, Optional, Any
from rawiai.agents.schemas import PoemCompositionRequest, MuseInspiration, PoeticTheme
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.rag.prosody_rules_retriever import ProsodyRulesRetriever
from rawiai.nlp.normalizer import ArabicNormalizer


class MuseAgent:
    """Agent responsible for conceptual inspiration, meter selection, and lexicon enrichment."""


    METERS_REGISTRY = {
        "الكامل": {
            "tafail": "مُتَفَاعِلُنْ مُتَفَاعِلُنْ مُتَفَاعِلُنْ",
            "ideal_themes": ["فخر وحماسة", "معاصرة وتقنية بلسان كلاسيكي", "حكمة وتأمل"],
            "characteristics": "بحر فخم رنان واسع العبارة، يستوعب المعاني المعاصرة والجليلة."
        },
        "الطويل": {
            "tafail": "فَعُولُنْ مَفَاعِيلُنْ فَعُولُنْ مَفَاعِيلُنْ",
            "ideal_themes": ["فخر وحماسة", "غزل ووجدان", "حكمة وتأمل"],
            "characteristics": "بحر المعلقات والقصائد الملحمية الكبرى، يمنح الجزالة والوقار."
        },
        "البسيط": {
            "tafail": "مُسْتَفْعِلُنْ فَاعِلُنْ مُسْتَفْعِلُنْ فَاعِلُنْ",
            "ideal_themes": ["حكمة وتأمل", "وصف وطبيعة", "رثاء وشجن"],
            "characteristics": "بحر سلس ممتد النغم، يلائم التفصيل والتأمل الفلسفي."
        },
        "الخفيف": {
            "tafail": "فَاعِلاتُنْ مُسْتَفْعِ لُنْ فَاعِلاتُنْ",
            "ideal_themes": ["غزل ووجدان", "وصف وطبيعة"],
            "characteristics": "بحر رقيق رشيق الإيقاع."
        }
    }

    THEME_KEYWORDS = {
        PoeticTheme.MODERN: ["ذكاء", "تقنية", "حاسب", "كود", "برمجة", "آلة", "روبوت", "بيانات", "مستقبل", "عصر", "علم"],
        PoeticTheme.FAKHR: ["شجاعة", "سيف", "فروسية", "خيل", "وغى", "حرب", "بأس", "عزيمة", "فخر", "مجد", "صمود"],
        PoeticTheme.HIKMA: ["حكمة", "زمان", "دهر", "عقل", "صبر", "حق", "عدل", "تأمل", "نصيحة", "دنيا"],
        PoeticTheme.GHAZAL: ["حب", "شوق", "حنين", "غرام", "دمع", "فراق", "قلب", "هوى", "جمال"],
        PoeticTheme.WASF: ["طبيعة", "صحراء", "مطر", "نجوم", "ليل", "بيداء", "جبل", "سماء", "صباح", "فجر", "قهوة", "بستان", "روض"],
        PoeticTheme.BIRR: ["ام", "امي", "والدة", "والدتي", "والدين", "اب", "ابي", "حنان", "عطف", "بر", "امومة", "رضا", "مهد", "اخ", "اخي", "اخوة", "شقيق", "صديق", "صاحب", "رفيق", "وفاء", "قربى", "صلة"]
    }

    ARABIC_PREFIXES = ["", "ال", "و", "وال", "ف", "فال", "ب", "بال", "ك", "كال", "ل", "لل"]

    def __init__(
        self,
        lexicon_retriever: Optional[AsasLexiconRetriever] = None,
        prosody_rules: Optional[ProsodyRulesRetriever] = None
    ):
        self.lexicon_retriever = lexicon_retriever or AsasLexiconRetriever()
        self.prosody_rules = prosody_rules or ProsodyRulesRetriever()

    def infer_theme(self, topic: str, user_theme: Optional[str] = None) -> str:
        """Determines the appropriate classical poetic purpose (غرض القصيدة)."""
        if user_theme:
            return user_theme

        norm_topic = ArabicNormalizer.normalize_search(topic)
        tokens = set(norm_topic.split())

        for theme_name, keywords in self.THEME_KEYWORDS.items():
            for kw in keywords:
                norm_kw = ArabicNormalizer.normalize_search(kw)
                valid_forms = {p + norm_kw for p in self.ARABIC_PREFIXES}
                if tokens & valid_forms:
                    return theme_name

        return PoeticTheme.FAKHR  # Default noble classical theme

    def select_meter(self, theme: str, user_meter: Optional[str] = None) -> str:
        """Selects the most harmonious meter for the theme."""
        if user_meter and user_meter in self.METERS_REGISTRY:
            return user_meter

        if theme in [PoeticTheme.MODERN, PoeticTheme.BIRR]:
            return "الكامل"
        elif theme == PoeticTheme.HIKMA:
            return "البسيط"
        elif theme == PoeticTheme.GHAZAL:
            return "الخفيف"
        else:
            return "الطويل"

    def select_rhyme(self, meter: str, user_rhyme: Optional[str] = None) -> str:
        """Picks a resounding classical rawiyy letter harmonious with the meter."""
        if user_rhyme and len(user_rhyme.strip()) == 1:
            return user_rhyme.strip()
        if meter == "البسيط":
            return "ر"
        return "ل"

    def retrieve_lexicon_motifs(self, topic: str, theme: str) -> List[Dict[str, Any]]:
        """Queries Asas Al-Balagha for metaphorical seeds and poetic rhetoric."""
        search_terms = []
        if theme == PoeticTheme.MODERN:
            search_terms = ["فكر", "قلم", "سيف", "نور", "علم"]
        elif theme == PoeticTheme.FAKHR:
            search_terms = ["سيف", "خيل", "بيد", "مجد", "صمصام"]
        elif theme == PoeticTheme.HIKMA:
            search_terms = ["عقل", "صبر", "دهر", "حزم", "بصر"]
        elif theme == PoeticTheme.BIRR:
            search_terms = ["عضد", "عطف", "ودد", "وصل", "كرم", "برر"]
        elif theme == PoeticTheme.WASF:
            search_terms = ["قهو", "صبح", "روض", "عطر", "شمس"]
        elif theme == PoeticTheme.GHAZAL:
            search_terms = ["شوق", "وجد", "هوى", "قلب", "بدر"]
        else:
            search_terms = ["صبح", "نور", "فضل", "كرم"]

        motifs = []
        for term in search_terms:
            results = self.lexicon_retriever.retrieve(term, top_k=1)
            if results:
                entry = results[0].entry
                motifs.append({
                    "lemma": entry.lemma,
                    "root": entry.root,
                    "literal_meaning": entry.literal_meaning[:120],
                    "metaphorical_meaning": entry.metaphorical_meaning[:200] if entry.metaphorical_meaning else entry.literal_meaning[:200],
                    "poetic_citations": entry.poetic_citations[:2]
                })

        return motifs[:3]

    def inspire(self, request: PoemCompositionRequest) -> MuseInspiration:
        """Synthesizes the complete creative blueprint for the Poet Agent."""
        theme = self.infer_theme(request.topic, request.theme)
        meter = self.select_meter(theme, request.target_meter)
        meter_info = self.METERS_REGISTRY.get(meter, self.METERS_REGISTRY["الطويل"])
        rhyme = self.select_rhyme(meter, request.target_rhyme)
        metaphors = self.retrieve_lexicon_motifs(request.topic, theme)

        mnemonic_key = self.prosody_rules.get_meter_key(meter) or ""
        zihafat = self.prosody_rules.get_permissible_zihafat(meter)
        zihafat_str = "، ".join([z["name"] for z in zihafat]) if zihafat else "لا زحاف غالب"

        guidance = (
            f"نظم {request.verse_count} أبيات على بحر {meter} ({meter_info['tafail']}). "
            f"مفتاح البحر من ميزان الذهب: [{mnemonic_key}]، والزحافات الجائزة: [{zihafat_str}]. "
            f"بقوافي منتهية بحرف الروي ({rhyme}). "
            f"الغرض الشعري: {theme}. استعن بالمجازات المرفقة من أساس البلاغة للزمخشري."
        )

        return MuseInspiration(
            topic=request.topic,
            theme=theme,
            selected_meter=meter,
            meter_tafail=meter_info["tafail"],
            selected_rhyme=rhyme,
            imagery_concepts=[m["lemma"] for m in metaphors],
            lexicon_metaphors=metaphors,
            guidance_notes=guidance
        )
