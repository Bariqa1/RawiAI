"""
Poetry Verse Segmenter (التقطيع الشعري والفصل بين الصدر والعجز).
Handles real-world Arabic poetry formats, noisy separators, line numbers, and hemistich balancing.
"""

import re
from typing import Tuple, List, Optional
from rawiai.nlp.normalizer import ArabicNormalizer


class VerseSegmenter:
    """Intelligent segmentation of classical Arabic poetry lines into Sadr (الصدر) and Ajuz (العجز)."""

    # Primary explicit separators commonly used in Arabic poetry corpora
    EXPLICIT_SEPARATORS = [
        re.compile(r"\s*#\s*"),
        re.compile(r"\s*\t\s*"),
        re.compile(r"\s*\.{3,}\s*"),
        re.compile(r"\s*…\s*"),
        re.compile(r"\s*\*{1,5}\s*"),
        re.compile(r"\s*//\s*"),
        re.compile(r"\s*/\s*"),
        re.compile(r"\s*\|\s*"),
        re.compile(r"\s*[—–-]{2,}\s*"),
        re.compile(r"[ ]{3,}"),  # 3 or more consecutive spaces
    ]

    # Pattern to strip leading/trailing verse numbering (e.g. "1-", "[12]", "(٣)")
    VERSE_NUMBER_PREFIX = re.compile(
        r"^\s*([0-9\u0660-\u0669]+[-.)\]\:]|\([0-9\u0660-\u0669]+\)|\[[0-9\u0660-\u0669]+\])\s*"
    )

    @classmethod
    def clean_verse_text(cls, text: str) -> str:
        """Removes verse index numbering and extraneous leading/trailing junk."""
        if not text:
            return ""
        s = text.strip()
        s = cls.VERSE_NUMBER_PREFIX.sub("", s)
        return s.strip()

    @classmethod
    def split_hemistichs(cls, verse_line: str) -> Tuple[str, str]:
        """
        Splits a single verse line into (Sadr, Ajuz).
        If explicit separators exist, splits cleanly.
        If no explicit separator exists, attempts a balanced midpoint split
        if the word count conforms to classical poetry standards (>= 6 words).
        """
        line = cls.clean_verse_text(verse_line)
        if not line:
            return "", ""

        # 1. Try explicit regex separators
        for sep_pattern in cls.EXPLICIT_SEPARATORS:
            parts = sep_pattern.split(line)
            # Filter non-empty parts
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) == 2:
                sadr, ajuz = parts[0], parts[1]
                # Validate that both sides have substantive words
                if len(sadr.split()) >= 2 and len(ajuz.split()) >= 2:
                    return sadr, ajuz

        # 2. Check for single hyphen/dash surrounded by spaces
        if " - " in line or " – " in line:
            for d in [" - ", " – "]:
                if d in line:
                    parts = line.split(d, 1)
                    if len(parts) == 2 and len(parts[0].split()) >= 2 and len(parts[1].split()) >= 2:
                        return parts[0].strip(), parts[1].strip()

        # 3. Fallback: Midpoint balanced heuristic split if word count >= 6
        words = line.split()
        if len(words) >= 6:
            mid = len(words) // 2
            sadr = " ".join(words[:mid]).strip()
            ajuz = " ".join(words[mid:]).strip()
            return sadr, ajuz

        # Single hemistich or short phrase
        return line, ""

    @classmethod
    def extract_verses_from_poem(cls, poem_text: str, min_words: int = 3) -> List[Tuple[str, str, str]]:
        """
        Extracts structured verses from a poem string (multi-line).
        Returns a list of tuples: (full_line, sadr, ajuz).
        Filters out noise, poem titles, and metadata headers.
        """
        if not poem_text:
            return []

        raw_lines = str(poem_text).split("\n")
        results = []

        for raw_line in raw_lines:
            line = cls.clean_verse_text(raw_line)
            if not line:
                continue

            # Must contain Arabic characters
            if not ArabicNormalizer.has_arabic(line):
                continue

            words = line.split()
            if len(words) < min_words:
                continue

            sadr, ajuz = cls.split_hemistichs(line)
            if sadr and ajuz:
                full_verse = f"{sadr} ... {ajuz}"
            else:
                full_verse = line

            results.append((full_verse, sadr, ajuz))

        return results
