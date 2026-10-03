"""
Lexicon Hybrid Retriever for 'Asas Al-Balagha' by Al-Zamakhshari.
Supports exact root/lemma matching, zero-dependency BM25 rhetorical search,
and metaphorical context retrieval for classical Arabic poetry.
"""

import os
import math
import json
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import numpy as np

from rawiai.models.lexicon_schema import AsasEntry, LexiconSearchResult
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.nlp.stemmer import PoetryArabicStemmer


class PurePythonBM25:
    """Lightweight, standalone Okapi BM25 implementation with zero external C dependencies."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / (self.corpus_size or 1)
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}

        df = Counter()
        for doc in corpus:
            counts = Counter(doc)
            self.doc_freqs.append(counts)
            for word in counts:
                df[word] += 1

        for word, freq in df.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Calculates BM25 relevance scores for all documents given query tokens."""
        scores = [0.0] * self.corpus_size
        for i, doc_counts in enumerate(self.doc_freqs):
            dl = self.doc_lens[i]
            for token in query_tokens:
                if token in doc_counts:
                    f = doc_counts[token]
                    idf = self.idf.get(token, 0.0)
                    scores[i] += idf * (f * (self.k1 + 1.0)) / (
                        f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    )
        return scores


class AsasLexiconRetriever:
    """
    Retrieval engine dedicated to rhetorical and metaphorical definitions
    from 'Asas Al-Balagha' by Al-Zamakhshari.
    """

    DEFAULT_FULL_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "asas_al_balagha_full.json"
    )
    DEFAULT_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "asas_al_balagha.json"
    )

    def __init__(self, data_path: Optional[str] = None):
        if data_path:
            self.data_path = data_path
        elif os.path.exists(self.DEFAULT_FULL_DATA_PATH):
            self.data_path = self.DEFAULT_FULL_DATA_PATH
        else:
            self.data_path = self.DEFAULT_DATA_PATH
        self.entries: List[AsasEntry] = []
        self.root_index: Dict[str, AsasEntry] = {}
        self.lemma_index: Dict[str, AsasEntry] = {}
        self.bm25_corpus: List[List[str]] = []
        self.bm25: Optional[PurePythonBM25] = None

        self._load_and_index()

    def _load_and_index(self):
        """Loads entries from JSON and builds exact and BM25 indices."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Asas Al-Balagha data file not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)

        self.entries = [AsasEntry(**item) for item in raw_entries]

        tokenized_corpus = []
        for entry in self.entries:
            # Index by normalized root
            norm_root = ArabicNormalizer.normalize_search(entry.root)
            self.root_index[norm_root] = entry

            # Index by normalized lemma
            norm_lemma = ArabicNormalizer.normalize_search(entry.lemma)
            self.lemma_index[norm_lemma] = entry

            # Build BM25 chunk
            chunk_text = (
                f"{entry.lemma} {entry.root} {entry.literal_meaning} "
                f"{entry.metaphorical_meaning} {' '.join(entry.poetic_citations)}"
            )
            tokens = PoetryArabicStemmer.tokenize_and_stem(chunk_text, remove_stopwords=True)
            tokenized_corpus.append(tokens)

        self.bm25_corpus = tokenized_corpus
        if self.bm25_corpus:
            self.bm25 = PurePythonBM25(self.bm25_corpus)

        print(f"Loaded {len(self.entries)} entries from Asas Al-Balagha into retrieval index.")

    def lookup_by_root(self, root: str) -> Optional[AsasEntry]:
        """Direct O(1) lookup by Arabic root."""
        norm = ArabicNormalizer.normalize_search(root)
        return self.root_index.get(norm)

    get_entry_by_root = lookup_by_root

    def lookup_by_lemma(self, lemma: str) -> Optional[AsasEntry]:
        """Direct O(1) lookup by lemma/word."""
        norm = ArabicNormalizer.normalize_search(lemma)
        return self.lemma_index.get(norm)

    def retrieve(self, query: str, top_k: int = 3) -> List[LexiconSearchResult]:
        """
        Multi-stage retrieval over Asas Al-Balagha:
        1. Exact root/lemma matching.
        2. BM25 lexical match over metaphorical explanations.
        3. Rank fusion.
        """
        results: List[LexiconSearchResult] = []
        seen_roots = set()

        # Step 1: Extract candidate roots and words from query
        query_stems = PoetryArabicStemmer.tokenize_and_stem(query, remove_stopwords=False)
        query_words = [ArabicNormalizer.normalize_search(w) for w in query.split()]

        # Check exact root or lemma matches
        for candidate in query_stems + query_words:
            # Check roots
            if candidate in self.root_index and candidate not in seen_roots:
                entry = self.root_index[candidate]
                results.append(LexiconSearchResult(entry=entry, score=1.0, match_type="exact_root"))
                seen_roots.add(candidate)

            # Check lemmas
            if candidate in self.lemma_index and self.lemma_index[candidate].root not in seen_roots:
                entry = self.lemma_index[candidate]
                results.append(LexiconSearchResult(entry=entry, score=0.95, match_type="exact_lemma"))
                seen_roots.add(entry.root)

        # Step 2: BM25 search for metaphorical context
        if self.bm25 and len(results) < top_k:
            query_tokens = PoetryArabicStemmer.tokenize_and_stem(query, remove_stopwords=True)
            if query_tokens:
                scores = self.bm25.get_scores(query_tokens)
                ranked_indices = np.argsort(scores)[::-1]

                for idx in ranked_indices:
                    if len(results) >= top_k:
                        break
                    score = float(scores[idx])
                    if score <= 0.0:
                        continue

                    entry = self.entries[idx]
                    if entry.root not in seen_roots:
                        results.append(LexiconSearchResult(
                            entry=entry,
                            score=round(score, 3),
                            match_type="bm25_metaphor"
                        ))
                        seen_roots.add(entry.root)

        return results[:top_k]

    def annotate_verse(self, verse_text: str) -> List[AsasEntry]:
        """
        Scans all tokens in a verse and retrieves matching entries from Asas Al-Balagha,
        extracting the exact metaphorical contexts relevant to the poem.
        """
        matched_entries: List[AsasEntry] = []
        seen_roots = set()

        raw_words = verse_text.split()
        for w in raw_words:
            norm_w = ArabicNormalizer.normalize_search(w)
            stem_w = PoetryArabicStemmer.stem_word(w)

            candidates = [norm_w, stem_w]
            # Strip common clitics
            for p in ["وال", "فال", "بال", "كال", "ولل", "فلل", "ال", "لل", "و", "ف", "ب"]:
                if norm_w.startswith(p) and len(norm_w) > len(p) + 1:
                    candidates.append(norm_w[len(p):])

            # Morphological variations (suffixes & verb prefixes)
            expanded = list(candidates)
            for c in expanded:
                for s in ["اء", "ني", "ها", "هم", "كم", "نا", "ات", "ون", "ين", "ية", "ة", "ي"]:
                    if c.endswith(s) and len(c) > len(s) + 2:
                        candidates.append(c[:-len(s)])
                if len(c) >= 4 and c[0] in "تينأ":
                    candidates.append(c[1:])

            for cand in candidates:
                entry = self.lemma_index.get(cand) or self.root_index.get(cand)
                if entry and entry.root not in seen_roots:
                    matched_entries.append(entry)
                    seen_roots.add(entry.root)
                    break

        return matched_entries



# Alias for backwards compatibility and test imports
AsasBalaghaRetriever = AsasLexiconRetriever
