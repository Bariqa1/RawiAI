"""
Prosody & Rhyme Analyzer for Classical Arabic Poetry (العروض والقافية).
Extracts the Rhyme consonant (حرف الرويّ), detects wasl/madd letters,
and estimates the classical poetic meter (البحر العروضي وتفعيلاته).
"""

import re
from typing import Tuple, Optional, Dict
from rawiai.nlp.normalizer import ArabicNormalizer


class ProsodyAnalyzer:
    """
    Classical Arabic prosody analysis engine.
    Computes poetic meters (بحور الشعر) and isolates the exact rhyming consonant (الرويّ).
    """

    WEAK_VOWELS = {"ا", "ى", "و", "ي"}
    
    # Suffixes where the rawiyy precedes the suffix (هاء الغائب، ضمائر المخاطب والغائب)
    RHYME_SUFFIXES = [
        "هما", "كما", "هن", "هم", "كم", "كن",
        "ها", "تم", "تن"
    ]

    # Canonical meters with their core rhythmic feet patterns
    METERS_DEFINITIONS = {
        "الطويل": {
            "tafail": "فعولن مفاعيلن فعولن مفاعيلن",
            "regex": r"(11010|1100|11011).*(11010|1100)",
            "signature": "11010",
            "syllable_len": (22, 32)
        },
        "الكامل": {
            "tafail": "متفاعلن متفاعلن متفاعلن",
            "regex": r"(1110110|1010110)",
            "signature": "1110110",
            "syllable_len": (18, 28)
        },
        "الوافر": {
            "tafail": "مفاعلتن مفاعلتن فعولن",
            "regex": r"(1101110|1101010)",
            "signature": "1101110",
            "syllable_len": (16, 26)
        },
        "البسيط": {
            "tafail": "مستفعلن فاعلن مستفعلن فاعلن",
            "regex": r"(1010110|100110).*10110",
            "signature": "1010110",
            "syllable_len": (19, 28)
        },
        "الخفيف": {
            "tafail": "فاعلاتن مستفعلن فاعلاتن",
            "regex": r"(1011010|11010).*1010110",
            "signature": "1011010",
            "syllable_len": (17, 26)
        },
        "الرمل": {
            "tafail": "فاعلاتن فاعلاتن فاعلاتن",
            "regex": r"(1011010|10110){2,}",
            "signature": "1011010",
            "syllable_len": (16, 24)
        },
        "المتقارب": {
            "tafail": "فعولن فعولن فعولن فعولن",
            "regex": r"(11010|1100){3,}",
            "signature": "11010",
            "syllable_len": (15, 24)
        },
        "الرجز": {
            "tafail": "مستفعلن مستفعلن مستفعلن",
            "regex": r"(1010110|100110){2,}",
            "signature": "1010110",
            "syllable_len": (16, 24)
        },
        "السريع": {
            "tafail": "مستفعلن مستفعلن فاعلن",
            "regex": r"(1010110|100110)",
            "signature": "1010110",
            "syllable_len": (16, 24)
        }
    }

    @classmethod
    def extract_rawiyy(cls, ajuz_text: str) -> Tuple[str, str]:
        """
        Extracts the true rhyming consonant (حرف الرويّ) and rhyme type (مطلقة أو مقيّدة)
        from the second hemistich (العجز).
        """
        if not ajuz_text:
            return "", "غير محدد"

        clean_text = ArabicNormalizer.normalize_phonetic_rhyme(ajuz_text)
        words = clean_text.split()
        if not words:
            return "", "غير محدد"

        last_word = words[-1]
        word_letters = ArabicNormalizer.remove_tashkeel(last_word)

        if len(word_letters) == 0:
            return "", "غير محدد"

        if len(word_letters) == 1:
            return word_letters, "مقيّدة"

        # Check pronoun suffixes
        for suffix in cls.RHYME_SUFFIXES:
            if word_letters.endswith(suffix) and len(word_letters) - len(suffix) >= 2:
                consonant = word_letters[-len(suffix) - 1]
                return consonant, "مطلقة"

        last_char = word_letters[-1]
        prev_char = word_letters[-2]

        # Case 1: Ends with Alif of Itlaaq / Madd (e.g. حكما، تكلما، منزلا، فاصبحينا)
        if last_char in {"ا", "ى"}:
            return prev_char, "مطلقة"

        # Case 2: Ends with Haa of Wasl
        if last_char == "ه" and len(word_letters) >= 3:
            if prev_char in cls.WEAK_VOWELS and len(word_letters) >= 4:
                return word_letters[-3], "مطلقة"
            return prev_char, "مطلقة"

        # Case 3: Ends with Taa Marbuta (ة)
        if last_char == "ة":
            return "ة", "مطلقة"

        # Case 4: Ends with Waw or Yaa acting as release vowel (إشباع)
        if last_char in {"و", "ي"} and len(word_letters) >= 3:
            return prev_char, "مطلقة"

        # Case 5: Standard consonant at end
        return last_char, "مقيّدة"

    @classmethod
    def arud_phonetize_text(cls, text: str) -> str:
        """
        Transforms Arabic text into phonetically rendered Arabic prosodic text (الكتابة العروضية).
        - Resolves Tanween into Noon Sakinah
        - Resolves Hamzat Al-Wasl in definite articles (ال)
        - Eliminates silent Alif
        """
        s = text.strip()
        # Tanween -> Noon sakinah
        s = re.sub(r"[\u064B\u064C\u064D]", "نْ", s)

        # Definite articles
        sun_letters = "تثدذرزسشصضطظلن"
        for sl in sun_letters:
            s = re.sub(r"\s+ال" + sl, " " + sl + "ّ", s)
        s = re.sub(r"\s+ال", " لْ", s)
        return s

    @classmethod
    def phonetize_for_prosody(cls, text: str) -> str:
        """
        Converts vocalized Arabic text into prosodic binary representation:
        '1' for Mutaharrik (متحرك), '0' for Sakin (ساكن / حرف مد).
        """
        if not text:
            return ""

        transformed = cls.arud_phonetize_text(text)
        diacritics_set = {"َ", "ُ", "ِ", "ً", "ٌ", "ٍ"}
        sukun = "ْ"
        shadda = "ّ"

        pattern = []
        i = 0
        n = len(transformed)

        while i < n:
            char = transformed[i]

            if char in " \t\n\r" or char in ArabicNormalizer.remove_tashkeel(char) == "":
                i += 1
                continue

            if char in diacritics_set or char == sukun:
                i += 1
                continue

            has_shadda = False
            has_vowel = False
            has_sukun = False

            j = i + 1
            while j < n and (transformed[j] in diacritics_set or transformed[j] in {shadda, sukun}):
                if transformed[j] == shadda:
                    has_shadda = True
                elif transformed[j] in diacritics_set:
                    has_vowel = True
                elif transformed[j] == sukun:
                    has_sukun = True
                j += 1

            if char in {"ا", "و", "ي", "ى"} and not has_vowel and not has_shadda:
                pattern.append("0")
            elif has_shadda:
                pattern.append("0")
                pattern.append("1")
                if has_sukun:
                    pattern.append("0")
            elif has_sukun:
                pattern.append("1")
                pattern.append("0")
            elif has_vowel:
                pattern.append("1")
            else:
                pattern.append("1")

            i = j if j > i + 1 else i + 1

        return "".join(pattern)

    @classmethod
    def estimate_meter(cls, sadr: str, ajuz: str = "") -> Tuple[str, Optional[str], float]:
        """
        Estimates the poetic meter (البحر العروضي) using syllable count and metric matching.
        Returns: (meter_name, tafail, confidence).
        """
        sample_text = sadr.strip() if sadr.strip() else ajuz.strip()
        if not sample_text:
            return "غير محدد", None, 0.0

        pattern = cls.phonetize_for_prosody(sample_text)
        sadr_len = len(pattern)

        best_meter = "غير محدد"
        best_tafail = None
        best_score = 0.0

        for meter_name, info in cls.METERS_DEFINITIONS.items():
            min_len, max_len = info["syllable_len"]

            if min_len <= sadr_len <= max_len:
                score = 0.65
                # Check regex matching
                if re.search(info["regex"], pattern):
                    score = 0.85
                elif info["signature"] in pattern:
                    score = 0.75

                if score > best_score:
                    best_score = score
                    best_meter = meter_name
                    best_tafail = info["tafail"]

        # Default fallback for classical poetry if length matches Al-Taweel
        if best_meter == "غير محدد" and 22 <= sadr_len <= 32:
            best_meter = "الطويل (تقديري)"
            best_tafail = cls.METERS_DEFINITIONS["الطويل"]["tafail"]
            best_score = 0.60

        return best_meter, best_tafail, round(best_score, 2)
