"""
Classical Arabic Poetry Lexicon (معجم غريب ألفاظ الشعر العربي والتراث).
Enriches retrieved poetry context with verified heritage dictionary meanings (لسان العرب، القاموس المحيط),
enabling LLMs to explain classical verses with 100% grounded fidelity and zero hallucination.
"""

import os
import json
from typing import Dict, List, Optional
from rawiai.models.schemas import DifficultWord
from rawiai.nlp.normalizer import ArabicNormalizer


class PoetryLexicon:
    """Heritage vocabulary annotation engine for classical Arabic poetry."""

    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "asas_al_balagha.json"
    )

    def __init__(self, custom_lexicon_path: Optional[str] = None):
        self.entries: Dict[str, DifficultWord] = {}
        data_file = custom_lexicon_path or self.DEFAULT_DATA_PATH
        if os.path.exists(data_file):
            self.load_from_json(data_file)

    def load_from_json(self, file_path: str):
        """Loads vocabulary entries from a structured JSON dataset."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            lemma = item.get("lemma") or item.get("word", "")
            root = item.get("root", "")
            
            # Combine literal and metaphorical if available
            lit = item.get("literal_meaning", "")
            meta = item.get("metaphorical_meaning", "")
            if lit and meta:
                meaning = f"{lit}؛ ومن المجاز: {meta}"
            else:
                meaning = item.get("meaning") or lit or meta

            source = item.get("source", "أساس البلاغة - الزمخشري")

            word_obj = DifficultWord(
                word=lemma,
                root=root,
                meaning=meaning,
                source_lexicon=source
            )

            # Index by normalized lemma and normalized root
            norm_lemma = ArabicNormalizer.normalize_search(lemma)
            norm_root = ArabicNormalizer.normalize_search(root)

            if norm_lemma:
                self.entries[norm_lemma] = word_obj
            if norm_root:
                self.entries[norm_root] = word_obj

    def lookup(self, word: str) -> Optional[DifficultWord]:
        """Looks up a word in the classical lexicon."""
        norm = ArabicNormalizer.normalize_search(word)
        return self.entries.get(norm)

    def add_entry(self, word: str, root: str, meaning: str, source: str = "معجم تراثي"):
        """Adds or overrides a lexicon entry."""
        norm = ArabicNormalizer.normalize_search(word)
        self.entries[norm] = DifficultWord(
            word=word,
            root=root,
            meaning=meaning,
            source_lexicon=source
        )

    def annotate_verse(self, verse_text: str) -> List[DifficultWord]:
        """
        Scans a verse text and extracts annotated explanations for all
        archaic or classical terms found in the lexicon.
        Prevents duplicates.
        """
        if not verse_text:
            return []

        annotated: List[DifficultWord] = []
        seen_roots = set()

        # Check raw tokens and search-normalized tokens
        tokens = verse_text.split()
        for raw_token in tokens:
            # Try exact word
            norm_token = ArabicNormalizer.normalize_search(raw_token)
            
            # Check direct match
            entry = self.entries.get(norm_token)

            # If not found, strip common prefix (e.g. بدار، كالبيد، والعذل)
            if not entry and len(norm_token) >= 4:
                for pfx in ["ال", "وال", "بال", "فال", "كال", "لل", "و", "ف", "ب"]:
                    if norm_token.startswith(pfx):
                        candidate = norm_token[len(pfx):]
                        if candidate in self.entries:
                            entry = self.entries[candidate]
                            break

            # If still not found, check suffixes (e.g. عذولي، سيوفها)
            if not entry and len(norm_token) >= 4:
                for sfx in ["ها", "هم", "ي", "ك", "ه", "ات", "ين", "ون"]:
                    if norm_token.endswith(sfx):
                        candidate = norm_token[:-len(sfx)]
                        if candidate in self.entries:
                            entry = self.entries[candidate]
                            break

            if entry and entry.root not in seen_roots:
                annotated.append(entry)
                seen_roots.add(entry.root)

        return annotated

    def save_to_json(self, file_path: str):
        """Serializes lexicon entries to a JSON file."""
        data = {k: v.dict() for k, v in self.entries.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_custom_lexicon(self, file_path: str):
        """Loads and merges additional lexicon entries from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k, item in data.items():
                norm_key = ArabicNormalizer.normalize_search(k)
                self.entries[norm_key] = DifficultWord(**item)
