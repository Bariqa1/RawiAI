"""
Parser and Ingestion Engine for 'Mizan Al-Dhahab fi Sina'at Shi'r Al-Arab' by Ahmad Al-Hashimi.
Processes full unabridged raw text, indexing chapters, sections, meters, and poetic rules into
a searchable JSON knowledge base.
"""

import os
import re
import json
from typing import List, Dict, Optional, Any


class MizanAlDhahabParser:
    """Parser for the complete text of Mizan Al-Dhahab."""

    DEFAULT_RAW_TXT = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "mizan_al_dhahab_full.txt"
    )

    DEFAULT_OUTPUT_JSON = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "mizan_al_dhahab_full.json"
    )

    METERS_CANONICAL = [
        "الطويل", "المديد", "البسيط", "الوافر", "الكامل", "الهزج",
        "الرجز", "الرمل", "السريع", "المنسرح", "الخفيف", "المضارع",
        "المقتضب", "المجتث", "المتقارب", "المتدارك"
    ]

    def __init__(self, raw_path: Optional[str] = None):
        self.raw_path = raw_path or self.DEFAULT_RAW_TXT

    def parse(self) -> Dict[str, Any]:
        """Parses the full text file into structured chapters and sections."""
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"File not found: {self.raw_path}")

        with open(self.raw_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        # Clean text
        cleaned_text = self._clean_ocr(raw_text)

        # Build book structure
        book_data = {
            "title": "ميزان الذهب في صناعة شعر العرب",
            "author": "أحمد الهاشمي",
            "edition": "مؤسسة هنداوي للتعليم والثقافة",
            "full_text_char_count": len(cleaned_text),
            "chapters": self._extract_chapters(cleaned_text),
            "meters_lessons": self._extract_meter_lessons(cleaned_text),
            "rhyme_lessons": self._extract_rhyme_lessons(cleaned_text)
        }

        return book_data

    def _clean_ocr(self, text: str) -> str:
        """Removes page numbers, publisher boilerplate headers, and OCR artifacts."""
        lines = text.splitlines()
        cleaned_lines = []
        skip_header = False

        for line in lines:
            l = line.strip()
            # Skip page headers / footers
            if l == "ميزان الذهب في صناعة شعر العرب" or l == "ميزان الذهب ق صناعة شعر العرب":
                continue
            if l == "أحمد الهاشمي" or l == "أحمد الهاشمى":
                continue
            if re.match(r"^[0-9٠-٩]+$", l):  # Standalone page number
                continue
            if "مؤسسة هنداوي للتعليم والثقافة" in l:
                continue
            if "عمارات الفتح" in l or "تلیفون:" in l:
                continue
            if l.startswith("www.") or "http" in l:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _extract_chapters(self, text: str) -> List[Dict[str, Any]]:
        """Splits full text into main thematic chapters."""
        chapters = []
        patterns = [
            ("مقدمة المؤلف", r"(مقدمة المؤلف\s+حمدًا لمن شيد معالم البيان.*?)(?=الباب الأول|$)"),
            ("الباب الأول: علم العروض", r"(الباب الأول\s+علم العروض.*?)(?=الباب الثاني|$)"),
            ("الباب الثاني: علم القافية", r"(الباب الثاني\s+علم القافية.*?)(?=الباب الثالث|$)"),
            ("الباب الثالث: خواطر في فنون الشعر ومحاسنه", r"(الباب الثالث\s+خواطر في فنون الشعر.*?)(?=$)")
        ]

        for idx, (title, pat) in enumerate(patterns, 1):
            match = re.search(pat, text, flags=re.DOTALL)
            if match:
                content = match.group(1).strip()
                chapters.append({
                    "chapter_number": idx,
                    "title": title,
                    "character_count": len(content),
                    "text": content[:50000]  # Store rich chunk
                })

        return chapters

    def _extract_meter_lessons(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extracts complete detailed lessons for each of the 16 meters."""
        lessons = {}

        for meter in self.METERS_CANONICAL:
            # Search for meter header and extract lesson until next meter
            pat = rf"(?:البحر\s+[^\n]*:\s*{meter}|بحر\s+{meter})(.*?)(?=(?:\([0-9٠-٩]+\)\s*البحر|بحر\s+(?:الكامل|الطويل|البسيط|الوافر|الخفيف|الرمل|الرجز|السريع|المنسرح|الهزج|المتقارب|المتدارك|المديد|المضارع|المقتضب|المجتث)|الباب الثاني|$))"
            match = re.search(pat, text, flags=re.DOTALL)
            if match:
                lesson_text = match.group(0).strip()
                lessons[meter] = {
                    "meter": meter,
                    "lesson_full_text": lesson_text,
                    "length": len(lesson_text)
                }

        return lessons

    def _extract_rhyme_lessons(self, text: str) -> Dict[str, Any]:
        """Extracts lessons on Rhyme (حروف القافية، حركاتها، عيوبها) from Bab 2."""
        match_bab2 = re.search(r"الباب الثاني\s+علم القافية(.*?)(?=الباب الثالث|$)", text, flags=re.DOTALL)
        if not match_bab2:
            return {}

        bab2_text = match_bab2.group(1).strip()
        return {
            "title": "الباب الثاني: علم القافية",
            "full_text": bab2_text,
            "has_defects_section": "عيوب القافية" in bab2_text,
            "has_letters_section": "حروف القافية" in bab2_text,
            "has_vowels_section": "حركات القافية" in bab2_text
        }

    def save_json(self, output_path: Optional[str] = None) -> str:
        """Parses and serializes the complete book to JSON."""
        target = output_path or self.DEFAULT_OUTPUT_JSON
        data = self.parse()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return target


if __name__ == "__main__":
    parser = MizanAlDhahabParser()
    out = parser.save_json()
    print("Parsed and saved Mizan Al-Dhahab successfully to:", out)
