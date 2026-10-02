"""
Poetry-Aware Arabic Light Stemmer & Morphological Processor.
Optimized for Classical Arabic Poetry information retrieval (BM25),
handling clitics (affixes) while strictly preserving core radicals.
"""

import re
from typing import List, Set
from rawiai.nlp.normalizer import ArabicNormalizer


class PoetryArabicStemmer:
    """
    Light stemmer designed to bridge the lexical gap in classical poetry
    without over-stemming or conflating distinct Arabic roots.
    """

    # Multi-letter prefixes to strip first (order of length descending)
    COMPOUND_PREFIXES = [
        "وال", "فال", "بال", "كال", "ولل", "فلل",
        "ال", "لل", "فس", "وس",
    ]

    # Short valid roots (2 letters) that should be protected from over-stripping
    SHORT_VALID_WORDS: Set[str] = {
        "يد", "دم", "أب", "اب", "أخ", "اخ", "فم", "ابن",
        "عز", "حق", "رب", "حر", "مر", "سر", "بر", "حب",
        "عد", "فل", "قل", "كل", "عم", "ام", "أم",
        "دار", "نار", "جار", "غار",
    }

    # Protected classical words starting with ب, ك, ل, س, و that should never lose their initial radical
    PROTECTED_INITIALS: Set[str] = {
        "بلاد", "بلدة", "بلدان", "بحر", "بحار", "بيت", "بيوت", "بدر", "بدور",
        "بأس", "باس", "باب", "ابواب", "بطل", "ابطال", "بيد", "بيداء",
        "كتاب", "كتائب", "كتب", "كف", "كفوف", "كرم", "كريم", "كاس", "كأس",
        "لسان", "لحم", "ليل", "ليال", "ليالي", "ليت", "لعل", "لباب",
        "سيف", "سيوف", "سياط", "سماء", "سلام", "سهم", "سهام", "سنة", "سنوات",
        "سراب", "سحر", "ساحر", "سير", "سيرة", "سيل", "سيول", "سور", "سوار",
        "سلطان", "سلاح", "سفينة", "سفائن", "سبيل", "سبل", "سوء",
        "ولد", "وليد", "ورد", "ورود", "وقت", "اوقات", "وطن", "اوطان", "وحش",
        "وعد", "وعود", "وفاء", "وجوه", "وجه", "وداد", "وادي", "وديان", "وغى",
    }

    # Irregular / broken plural stems commonly found in poetry mapped to canonical form
    PLURAL_STEM_MAPPINGS = {
        "سيوف": "سيف",
        "اسياف": "سيف",
        "رماح": "رمح",
        "جياد": "جواد",
        "خيول": "خيل",
        "فرسان": "فارس",
        "اعناق": "عنق",
        "عيون": "عين",
        "قلوب": "قلب",
        "منازل": "منزل",
        "اطراف": "طرف",
        "شجعان": "شجاع",
    }

    # Curated classical poetry stop words (grammatical particles, prepositions, demonstratives)
    POETRY_STOPWORDS: Set[str] = {
        "في", "من", "على", "عن", "إلى", "الى", "حتى", "مذ", "منذ", "رب",
        "إن", "ان", "أن", "كأن", "كان", "كانت", "لكن", "لعل", "ليت",
        "لا", "لم", "لن", "ما", "ماذا", "لماذا", "هل", "ألا", "الا", "أما",
        "كيف", "أين", "متى", "أنى", "إذ", "إذا", "لو", "لولا", "لوما",
        "واذا", "وإذا", "فإذا", "فاذا", "فإن", "وان", "وأن", "وإن",
        "غير", "سوى", "بلى", "نعم", "أجل", "هو", "هي", "هما", "هم", "هن",
        "أنت", "انت", "أنتم", "أنتما", "أنتن", "أنا", "انا", "نحن",
        "هذا", "هذه", "هذان", "هاتان", "هؤلاء", "ذلك", "تلك", "أولئك",
        "الذي", "التي", "اللذان", "اللتان", "الذين", "اللاتي", "اللائي",
        "حيث", "حين", "لدى", "دون", "مع", "بين", "عند", "خلف", "أمام",
        "فوق", "تحت", "قد", "لقد", "بل", "ثم", "أو", "او", "أم", "ام",
    }

    @classmethod
    def stem_word(cls, word: str) -> str:
        """
        Stems an individual Arabic token by removing proclitics and enclitics
        while respecting radical root constraints and poetry plural forms.
        """
        w = ArabicNormalizer.normalize_search(word)
        if len(w) <= 2:
            return w

        if w in cls.SHORT_VALID_WORDS or w in cls.PROTECTED_INITIALS:
            return cls.PLURAL_STEM_MAPPINGS.get(w, w)

        # --- Phase 1: Strip Compound Prefixes (وال, بال, فال, كال, لل, ال) ---
        for pfx in cls.COMPOUND_PREFIXES:
            if w.startswith(pfx):
                rem = w[len(pfx):]
                if len(rem) >= 3 or rem in cls.SHORT_VALID_WORDS:
                    w = rem
                    break

        if w in cls.SHORT_VALID_WORDS or w in cls.PROTECTED_INITIALS:
            return cls.PLURAL_STEM_MAPPINGS.get(w, w)

        # --- Phase 2: Strip Compound Suffixes (هما, كما, هن, هم, كم, نا, ها, ون, ين, ات, ان) ---
        compound_suffixes = [
            "كما", "تما", "هما", "هن", "هم",
            "كم", "كن", "نا", "ني", "ها",
            "ون", "ين", "ات", "ان"
        ]
        for sfx in compound_suffixes:
            if w.endswith(sfx):
                rem = w[:-len(sfx)]
                if len(rem) >= 3 or rem in cls.SHORT_VALID_WORDS:
                    w = rem
                    break

        # --- Phase 3: Strip Single Suffixes (ك, ه, ة, ي) FIRST before single prefixes ---
        if len(w) >= 4 and w not in cls.SHORT_VALID_WORDS:
            for sfx in ["ك", "ه", "ة", "ي"]:
                if w.endswith(sfx):
                    rem = w[:-1]
                    if len(rem) >= 3 or rem in cls.SHORT_VALID_WORDS:
                        w = rem
                        break

        if w in cls.SHORT_VALID_WORDS or w in cls.PROTECTED_INITIALS:
            return cls.PLURAL_STEM_MAPPINGS.get(w, w)

        # --- Phase 4: Strip Single Prefixes (و, ف, ب, ك, ل) after suffixes are stripped ---
        if len(w) >= 4 and w not in cls.PROTECTED_INITIALS:
            # Waw conjunction (و): strip if remaining length >= 3 and not starting with و
            if w.startswith("و") and len(w) - 1 >= 3 and not w.startswith("وو") and w not in cls.PROTECTED_INITIALS:
                cand = w[1:]
                if cand in cls.PROTECTED_INITIALS or len(cand) >= 3:
                    w = cand

            # Fa conjunction (ف): strip if remaining >= 3
            elif w.startswith("ف") and len(w) - 1 >= 3 and w not in cls.PROTECTED_INITIALS:
                cand = w[1:]
                if cand in cls.PROTECTED_INITIALS or len(cand) >= 3:
                    w = cand

            # Ba preposition (ب): e.g. بدار -> دار, بحق -> حق
            elif w.startswith("ب") and len(w) - 1 >= 3 and w not in cls.PROTECTED_INITIALS:
                cand = w[1:]
                if cand in cls.SHORT_VALID_WORDS or cand in cls.PROTECTED_INITIALS:
                    w = cand

        return cls.PLURAL_STEM_MAPPINGS.get(w, w)

    @classmethod
    def tokenize_and_stem(
        cls,
        text: str,
        remove_stopwords: bool = True,
        min_token_len: int = 2
    ) -> List[str]:
        """
        Tokenizes text, strips stopwords before and after stemming,
        and returns cleaned stems for BM25 indexing.
        """
        if not text:
            return []

        norm_text = ArabicNormalizer.normalize_search(text)
        tokens = norm_text.split()

        stems = []
        for token in tokens:
            if len(token) < min_token_len:
                continue

            if remove_stopwords and token in cls.POETRY_STOPWORDS:
                continue

            stemmed = cls.stem_word(token)
            if not stemmed or len(stemmed) < min_token_len:
                continue

            if remove_stopwords and stemmed in cls.POETRY_STOPWORDS:
                continue

            stems.append(stemmed)

        return stems
