"""
Arabic Text Normalization Suite for Classical Literature & Poetry.
Supports search normalization, display normalization, and phonetically-aware rhyme normalization.
"""

import re
from typing import Optional

# Comprehensive Arabic diacritics regex (harakat + tanween + shadda + sukun + Quranic symbols)
ARABIC_DIACRITICS_PATTERN = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Standard and Arabic punctuation
ARABIC_PUNCTUATION_PATTERN = re.compile(
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~،؛؟«»…—–•\t\r]"""
)

# Tatweel / Kashida
TATWEEL = "ـ"


class ArabicNormalizer:
    """Specialized normalizer designed for classical Arabic and poetry."""

    @staticmethod
    def remove_tashkeel(text: str) -> str:
        """Strips all Arabic diacritical marks (Tashkeel)."""
        if not text:
            return ""
        return ARABIC_DIACRITICS_PATTERN.sub("", text)

    @staticmethod
    def remove_tatweel(text: str) -> str:
        """Removes the tatweel (kashida) character used in poetic calligraphic stretching."""
        if not text:
            return ""
        return text.replace(TATWEEL, "")

    @staticmethod
    def normalize_search(text: str) -> str:
        """
        Normalizes Arabic text for lexical and semantic retrieval.
        - Strips diacritics and tatweel
        - Unifies Alef variants (إ, أ, آ, ٱ -> ا)
        - Unifies Hamza carriers (ؤ -> و, ئ -> ي)
        - Normalizes Taa Marbuta (ة -> ه)
        - Normalizes Alef Maqsura (ى -> ي)
        - Cleans punctuation and excessive whitespace
        """
        if not text:
            return ""

        s = str(text)
        s = ARABIC_DIACRITICS_PATTERN.sub("", s)
        s = s.replace(TATWEEL, "")

        # Alef normalization
        s = re.sub(r"[إأآٱ]", "ا", s)

        # Hamza normalization for search
        s = s.replace("ؤ", "و")
        s = s.replace("ئ", "ي")

        # Taa Marbuta and Alef Maqsura
        s = s.replace("ة", "ه")
        s = s.replace("ى", "ي")

        # Punctuation to spaces
        s = ARABIC_PUNCTUATION_PATTERN.sub(" ", s)

        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def normalize_phonetic_rhyme(text: str) -> str:
        """
        Phonetic normalization for rhyme extraction (الرويّ).
        Unlike search normalization:
        - Strips diacritics and tanween so letters are clean
        - Keeps distinctions between 'ة' (Taa Marbuta) and 'ه' (Haa)
        - Keeps distinctions between 'ى' (Alif Maqsura phonetically an Alif) and 'ي' (Yaa)
        - Removes tatweel and punctuation
        """
        if not text:
            return ""

        s = str(text)
        s = ARABIC_DIACRITICS_PATTERN.sub("", s)
        s = s.replace(TATWEEL, "")
        s = ARABIC_PUNCTUATION_PATTERN.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def normalize_display(text: str) -> str:
        """
        Cleans text for user presentation without destroying diacritics or letter aesthetics.
        """
        if not text:
            return ""

        s = str(text)
        s = s.replace("\r", "")
        # Remove repeated tatweel beyond single elongation if any
        s = re.sub(r"ـ{2,}", "ـ", s)
        # Collapse multiple spaces but preserve single newlines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n")]
        return "\n".join(line for line in lines if line)

    @staticmethod
    def has_arabic(text: str) -> bool:
        """Checks if text contains Arabic characters."""
        if not text:
            return False
        return bool(re.search(r"[\u0600-\u06FF]", text))
