"""
Asas Al-Balagha Text Parser & Ingestion Pipeline.
Converts digitized OpenITI / Maktaba Shamela raw texts into structured AsasEntry
dictionaries separating literal semantic definitions (الحقيقة) from metaphorical
usages (المجاز) and poetic citations (الشواهد الشعرية).
"""

import os
import sys
import re
import json
from typing import List, Dict, Optional, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rawiai.models.lexicon_schema import AsasEntry
from rawiai.nlp.normalizer import ArabicNormalizer


class AsasBalaghaParser:
    """Parser and tokenizer for Al-Zamakhshari's Asas Al-Balagha dictionary."""

    DEFAULT_RAW_TEXT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "asas_al_balagha_shamela_full.txt"
    )

    DEFAULT_OUTPUT_JSON_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "asas_al_balagha_full.json"
    )

    def __init__(self, raw_path: Optional[str] = None):
        self.raw_path = raw_path or self.DEFAULT_RAW_TEXT_PATH

    @classmethod
    def clean_text_stream(cls, raw_content: str) -> str:
        """Pre-processes OpenITI mARkdown format, fixing wrapped headers and tags."""
        text = raw_content

        # 1. Fix OpenITI wrapped headers where last root letter is on the next line:
        # e.g.: '### | أ ب\n# و\n' -> '### | أ ب و\n'
        pattern_wrapped = re.compile(r'### \|+ ([\u0600-\u06FF\s]+)\n# ([\u0600-\u06FF])\n')
        text = pattern_wrapped.sub(r'### | \1 \2\n', text)

        # 2. Strip page markers, milestones and mARkdown artifacts
        text = re.sub(r'# PageV\d+P\d+', '', text)
        text = re.sub(r'ms\d+', '', text)
        text = text.replace('~~', '')  # OpenITI line continuations

        return text

    def parse(self, raw_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parses the raw text file and returns a list of dictionary entries."""
        source_file = raw_path or self.raw_path
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found at: {source_file}")

        with open(source_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text = self.clean_text_stream(raw_text)

        # Split into sections based on '### | [Root letters]'
        raw_sections = re.split(r'\n### \|+ +([^\n]+)\n', cleaned_text)
        entries: List[Dict[str, Any]] = []
        seen_roots = set()

        for i in range(1, len(raw_sections), 2):
            header = raw_sections[i].strip()
            content = raw_sections[i + 1].strip() if i + 1 < len(raw_sections) else ""

            # Filter out non-root sections (preface, chapters)
            clean_root = header.replace(' ', '').replace('·', '').replace('ـ', '').strip()
            clean_root = ArabicNormalizer.remove_tashkeel(clean_root)

            if (
                not clean_root
                or clean_root.startswith("كتاب")
                or clean_root.startswith("باب")
                or "مقدمة" in clean_root
                or len(clean_root) < 2
            ):
                continue

            # Process content lines
            content_lines = []
            citations = []
            for line in content.split("\n"):
                l = line.strip()
                if l.startswith("#"):
                    l = l.lstrip("#").strip()
                if not l:
                    continue

                content_lines.append(l)

                # Capture classical poetic hemistichs / verses (separated by ...)
                if "..." in l or "…" in l:
                    citations.append(l)

            full_entry_text = " ".join(content_lines)
            if not full_entry_text:
                continue

            # Differentiate between Literal (الحقيقة) and Metaphorical (المجاز)
            parts = re.split(
                r'(ومن المجاز|ومن المستعار|ومجاز ذلك|ومن الاستعارة)[\s:؛]',
                full_entry_text,
                maxsplit=1
            )

            if len(parts) >= 3:
                literal_text = parts[0].strip()
                metaphorical_text = (parts[1] + ": " + parts[2]).strip()
            else:
                literal_text = full_entry_text
                metaphorical_text = ""

            # Ensure unique roots in dataset
            if clean_root in seen_roots:
                # Append to existing root if duplicate
                for existing in entries:
                    if existing["root"] == clean_root:
                        if metaphorical_text and not existing["metaphorical_meaning"]:
                            existing["metaphorical_meaning"] = metaphorical_text
                        break
                continue

            seen_roots.add(clean_root)

            entry_dict = {
                "root": clean_root,
                "lemma": clean_root,
                "literal_meaning": literal_text,
                "metaphorical_meaning": metaphorical_text,
                "poetic_citations": citations[:6],
                "source": "أساس البلاغة - الزمخشري"
            }
            entries.append(entry_dict)

        return entries

    def export_json(self, output_path: Optional[str] = None) -> str:
        """Parses the entire book and exports a structured JSON file."""
        out_file = output_path or self.DEFAULT_OUTPUT_JSON_PATH
        parsed_entries = self.parse()

        # If base curated dataset exists, merge it to preserve gold standard annotations
        curated_path = os.path.join(os.path.dirname(out_file), "asas_al_balagha.json")
        merged_entries = []
        curated_roots = set()

        if os.path.exists(curated_path):
            try:
                with open(curated_path, "r", encoding="utf-8") as f:
                    curated_data = json.load(f)
                for item in curated_data:
                    curated_roots.add(item.get("root", ""))
                    merged_entries.append(item)
            except Exception:
                pass

        for entry in parsed_entries:
            if entry["root"] not in curated_roots:
                merged_entries.append(entry)

        # Ensure directory exists
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(merged_entries, f, ensure_ascii=False, indent=2)

        return out_file


if __name__ == "__main__":
    parser = AsasBalaghaParser()
    out = parser.export_json()
    print(f"Exported Asas Al-Balagha complete JSON to: {out}")
