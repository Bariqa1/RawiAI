"""
Corpus Processor for Arabic Poetry Datasets.
Ingests, segments, enriches, deduplicates, and structures classical poetry records
for production-grade Hybrid RAG pipelines.
"""

import os
import uuid
import json
import pickle
from typing import List, Dict, Optional, Any
import pandas as pd

from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.nlp.segmenter import VerseSegmenter
from rawiai.nlp.stemmer import PoetryArabicStemmer
from rawiai.nlp.prosody import ProsodyAnalyzer
from rawiai.nlp.lexicon import PoetryLexicon


class PoetryCorpusProcessor:
    """End-to-end dataset transformation engine for classical Arabic poetry."""

    # Candidate column names for automatic schema detection
    POEM_COL_CANDIDATES = ["poem", "poem_text", "text", "قصيدة", "القصيدة", "البيت", "verse", "verses", "content"]
    POET_COL_CANDIDATES = ["poet", "author", "الشاعر", "شاعر", "قائل", "writer"]
    TITLE_COL_CANDIDATES = ["title", "poem_title", "العنوان", "عنوان", "name"]
    ERA_COL_CANDIDATES = ["era", "period", "العصر", "عصر", "الفترة"]
    THEME_COL_CANDIDATES = ["theme", "genre", "الغرض", "غرض", "topic", "category"]

    def __init__(self, lexicon: Optional[PoetryLexicon] = None):
        self.lexicon = lexicon or PoetryLexicon()

    @classmethod
    def detect_column(cls, columns: List[str], candidates: List[str]) -> Optional[str]:
        """Detects column names using exact and partial normalized matching."""
        norm_cols = {ArabicNormalizer.normalize_search(str(c)): c for c in columns}
        
        # 1. Exact normalized match
        for cand in candidates:
            norm_cand = ArabicNormalizer.normalize_search(cand)
            if norm_cand in norm_cols:
                return norm_cols[norm_cand]

        # 2. Substring match
        for col_name in columns:
            col_norm = ArabicNormalizer.normalize_search(str(col_name))
            for cand in candidates:
                cand_norm = ArabicNormalizer.normalize_search(cand)
                if cand_norm in col_norm or col_norm in cand_norm:
                    return col_name

        return None

    def process_verse_text(
        self,
        raw_text: str,
        poet: Optional[str] = None,
        era: Optional[str] = None,
        theme: Optional[str] = None,
        poem_title: Optional[str] = None,
        source_row: Optional[int] = None,
        verse_id: Optional[str] = None
    ) -> Optional[EnrichedVerse]:
        """
        Transforms a raw verse line into a fully enriched, validated EnrichedVerse record.
        """
        line = VerseSegmenter.clean_verse_text(raw_text)
        if not line or not ArabicNormalizer.has_arabic(line):
            return None

        # 1. Segment into Sadr and Ajuz
        sadr, ajuz = VerseSegmenter.split_hemistichs(line)
        if sadr and ajuz:
            full_text = f"{sadr} ... {ajuz}"
        else:
            full_text = line
            sadr = line
            ajuz = ""

        # 2. Search normalization
        normalized_search = ArabicNormalizer.normalize_search(full_text)
        if len(normalized_search.split()) < 2:
            return None

        # 3. Stemming for BM25
        stemmed_tokens = PoetryArabicStemmer.tokenize_and_stem(normalized_search)

        # 4. Prosody & Rhyme analysis
        rhyme_letter, rhyme_type = ProsodyAnalyzer.extract_rawiyy(ajuz if ajuz else sadr)
        meter_name, tafail, confidence = ProsodyAnalyzer.estimate_meter(sadr, ajuz)
        
        prosody = ProsodyInfo(
            meter=meter_name,
            meter_tafail=tafail,
            rhyme_letter=rhyme_letter,
            rhyme_type=rhyme_type,
            confidence=confidence
        )

        # 5. Lexicon annotation (extract heritage word definitions)
        annotated_words = self.lexicon.annotate_verse(full_text)

        vid = verse_id or str(uuid.uuid4())[:8]

        return EnrichedVerse(
            id=vid,
            original_text=full_text,
            sadr=sadr,
            ajuz=ajuz,
            normalized_search=normalized_search,
            stemmed_tokens=stemmed_tokens,
            prosody=prosody,
            difficult_words=annotated_words,
            poet=poet.strip() if poet and not pd.isna(poet) else None,
            era=era.strip() if era and not pd.isna(era) else None,
            theme=theme.strip() if theme and not pd.isna(theme) else None,
            poem_title=poem_title.strip() if poem_title and not pd.isna(poem_title) else None,
            source_row=source_row
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        limit_rows: Optional[int] = None,
        deduplicate: bool = True
    ) -> List[EnrichedVerse]:
        """
        Processes an entire poetry DataFrame, splitting poems into verses
        and producing a list of EnrichedVerse objects.
        """
        cols = list(df.columns)
        poem_col = self.detect_column(cols, self.POEM_COL_CANDIDATES)
        poet_col = self.detect_column(cols, self.POET_COL_CANDIDATES)
        title_col = self.detect_column(cols, self.TITLE_COL_CANDIDATES)
        era_col = self.detect_column(cols, self.ERA_COL_CANDIDATES)
        theme_col = self.detect_column(cols, self.THEME_COL_CANDIDATES)

        if not poem_col:
            raise ValueError(f"Could not automatically detect poem text column in {cols}")

        print(f"Detected columns -> Poem: '{poem_col}', Poet: '{poet_col}', Title: '{title_col}', Era: '{era_col}'")

        records: List[EnrichedVerse] = []
        unique_keys = set()
        count = 0

        target_df = df if limit_rows is None else df.head(limit_rows)

        for row_idx, row in target_df.iterrows():
            poem_content = row[poem_col]
            if pd.isna(poem_content):
                continue

            poet_val = str(row[poet_col]) if poet_col and not pd.isna(row[poet_col]) else None
            title_val = str(row[title_col]) if title_col and not pd.isna(row[title_col]) else None
            era_val = str(row[era_col]) if era_col and not pd.isna(row[era_col]) else None
            theme_val = str(row[theme_col]) if theme_col and not pd.isna(row[theme_col]) else None

            # Split poem into individual verses
            extracted = VerseSegmenter.extract_verses_from_poem(str(poem_content))
            for full_v, sadr_v, ajuz_v in extracted:
                enriched = self.process_verse_text(
                    raw_text=full_v,
                    poet=poet_val,
                    era=era_val,
                    theme=theme_val,
                    poem_title=title_val,
                    source_row=int(row_idx)
                )

                if not enriched:
                    continue

                if deduplicate:
                    key = enriched.normalized_search
                    if key in unique_keys:
                        continue
                    unique_keys.add(key)

                records.append(enriched)
                count += 1

        print(f"Successfully processed {len(records):,} enriched verses.")
        return records

    @staticmethod
    def save_records_to_jsonl(records: List[EnrichedVerse], file_path: str):
        """Saves enriched records to high-efficiency JSONL format."""
        with open(file_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.json(ensure_ascii=False) + "\n")
        print(f"Saved {len(records):,} records to {file_path}")

    @staticmethod
    def load_records_from_jsonl(file_path: str) -> List[EnrichedVerse]:
        """Loads enriched records from JSONL file."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(EnrichedVerse.parse_raw(line))
        return records
