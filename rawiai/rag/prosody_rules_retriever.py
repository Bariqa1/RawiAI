"""
Prosody Rules Retriever for 'Mizan Al-Dhahab fi Sina'at Shi'r Al-Arab' by Ahmad Al-Hashimi.
Provides deterministic, structured retrieval for the 16 Classical Arabic meters,
mnemonic keys, permissible zihafat/ilal, rhyme laws, and poetic defect diagnoses.
"""

import os
import json
from typing import List, Dict, Optional, Any
from rawiai.nlp.normalizer import ArabicNormalizer


class ProsodyRulesRetriever:
    """
    Authoritative retrieval engine for Arabic prosody and poetic craft
    grounded in 'Mizan Al-Dhahab' (ميزان الذهب في صناعة شعر العرب).
    """

    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "mizan_al_dhahab.json"
    )
    DEFAULT_FULL_JSON_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "mizan_al_dhahab_full.json"
    )
    DEFAULT_FULL_TXT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "mizan_al_dhahab_full.txt"
    )
    DEFAULT_PDF_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "mizan_al_dhahab.pdf"
    )

    def __init__(self, data_path: Optional[str] = None, full_json_path: Optional[str] = None):
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self.full_json_path = full_json_path or self.DEFAULT_FULL_JSON_PATH
        self.treatise_title = "ميزان الذهب في صناعة شعر العرب"
        self.author = "أحمد الهاشمي"
        self.meters_by_name: Dict[str, Dict[str, Any]] = {}
        self.rhyme_rules: Dict[str, Any] = {}
        self.defects_by_name: Dict[str, Dict[str, str]] = {}
        self.craft_principles: List[Dict[str, str]] = []
        self.full_book_data: Dict[str, Any] = {}
        self.meters_full_lessons: Dict[str, Dict[str, Any]] = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Loads and indexes the Mizan Al-Dhahab JSON file and full book text."""
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.treatise_title = data.get("treatise_title", self.treatise_title)
            self.author = data.get("author", self.author)

            for m in data.get("meters", []):
                norm_name = ArabicNormalizer.normalize_search(m["name"])
                self.meters_by_name[norm_name] = m

            self.rhyme_rules = data.get("rhyme_rules", {})

            for d in data.get("poetic_defects", []):
                norm_defect = ArabicNormalizer.normalize_search(d["name"])
                self.defects_by_name[norm_defect] = d

            self.craft_principles = data.get("craft_principles", [])

        # Load unabridged full book JSON if available
        if os.path.exists(self.full_json_path):
            try:
                with open(self.full_json_path, "r", encoding="utf-8") as f:
                    self.full_book_data = json.load(f)
                    self.meters_full_lessons = self.full_book_data.get("meters_lessons", {})
            except Exception:
                pass

    def has_full_book_loaded(self) -> bool:
        """Checks if the complete unabridged book text is loaded."""
        return bool(self.full_book_data) or os.path.exists(self.DEFAULT_FULL_TXT_PATH)

    def get_meter_full_lesson(self, meter_name: str) -> Optional[str]:
        """Returns the unabridged complete lesson text for a specific meter from the book."""
        norm_name = ArabicNormalizer.normalize_search(meter_name)
        for m, data in self.meters_full_lessons.items():
            if ArabicNormalizer.normalize_search(m) in norm_name or norm_name in ArabicNormalizer.normalize_search(m):
                return data.get("lesson_full_text")
        return None

    def get_rhyme_full_lesson(self) -> Optional[str]:
        """Returns the unabridged complete text of Bab 2 (علم القافية) from the book."""
        if self.full_book_data:
            return self.full_book_data.get("rhyme_lessons", {}).get("full_text")
        return None

    def get_book_full_text(self) -> str:
        """Returns the entire unabridged text of the book."""
        if os.path.exists(self.DEFAULT_FULL_TXT_PATH):
            with open(self.DEFAULT_FULL_TXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return ""

    def get_meter(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves exact meter specification and rules by name."""
        norm_name = ArabicNormalizer.normalize_search(name)
        # Check direct match
        if norm_name in self.meters_by_name:
            return self.meters_by_name[norm_name]
        # Check prefix/suffix match (e.g. "بحر الطويل" -> "طويل")
        for k, v in self.meters_by_name.items():
            if k in norm_name or norm_name in k:
                return v
        return None

    def list_all_meters(self) -> List[str]:
        """Returns names of all 16 canonical Arabic meters."""
        return [m["name"] for m in self.meters_by_name.values()]

    def get_meter_key(self, meter_name: str) -> Optional[str]:
        """Returns the mnemonic key verse (مفتاح البحر) for a given meter."""
        m = self.get_meter(meter_name)
        return m.get("mnemonic_key") if m else None

    def get_permissible_zihafat(self, meter_name: str) -> List[Dict[str, str]]:
        """Returns permissible metrical variations (الزحافات الجائزة) for the meter."""
        m = self.get_meter(meter_name)
        return m.get("permissible_zihafat", []) if m else []

    def get_defect_info(self, defect_name: str) -> Optional[Dict[str, str]]:
        """Retrieves diagnosis and remedy for a poetic defect (e.g., إقواء, إكفاء, إيطاء, كسر)."""
        norm_name = ArabicNormalizer.normalize_search(defect_name)
        if norm_name in self.defects_by_name:
            return self.defects_by_name[norm_name]
        for k, v in self.defects_by_name.items():
            if k in norm_name or norm_name in k:
                return v
        return None

    def recommend_meter_for_theme(self, theme: str) -> List[Dict[str, Any]]:
        """Recommends canonical meters historically tailored to a specific theme."""
        norm_theme = ArabicNormalizer.normalize_search(theme)
        recommended = []
        for m in self.meters_by_name.values():
            for st in m.get("suitable_themes", []):
                norm_st = ArabicNormalizer.normalize_search(st)
                if norm_theme in norm_st or norm_st in norm_theme:
                    recommended.append(m)
                    break
        return recommended if recommended else [self.get_meter("الكامل"), self.get_meter("الطويل")]

    def format_prosody_reference_context(
        self,
        meter_name: Optional[str] = None,
        theme: Optional[str] = None
    ) -> str:
        """
        Synthesizes an authoritative briefing from 'Mizan Al-Dhahab'
        to inject into Agent instructions or LLM generation prompts.
        """
        lines = [f"=== مستند من «{self.treatise_title}» للشيخ {self.author} ==="]

        target_m = self.get_meter(meter_name) if meter_name else None
        if target_m:
            lines.append(f"• البحر: {target_m['name']}")
            lines.append(f"• مفتاح البحر: {target_m['mnemonic_key']}")
            lines.append(f"• وزنه المعياري: {target_m['standard_tafail']}")
            lines.append(f"• الطابع البلاغي: {target_m['aesthetic_notes']}")
            if target_m.get("permissible_zihafat"):
                lines.append("• الزحافات المستحسنة والجائزة:")
                for z in target_m["permissible_zihafat"]:
                    lines.append(f"  - {z['name']}: {z['application']} ({z['status']})")
        elif theme:
            recs = self.recommend_meter_for_theme(theme)
            lines.append(f"• البحور الأنسب لغرض ({theme}): " + "، ".join(r["name"] for r in recs if r))

        # Add rhyme definition summary
        if self.rhyme_rules:
            def_text = self.rhyme_rules.get("definition", "")
            if def_text:
                lines.append(f"• قاعدة القافية: {def_text}")

        return "\n".join(lines)
