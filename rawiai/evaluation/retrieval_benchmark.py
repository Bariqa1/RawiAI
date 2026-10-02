"""
Retrieval Layer Benchmark for Arabic Poetry & Dual-RAG.
Measures Hit@K (Hit@1, Hit@3, Hit@5), MRR (Mean Reciprocal Rank),
Refusal Precision, and Multi-Hop Lexicon Annotation Accuracy.
"""

import os
import json
from typing import List, Dict, Any, Optional

from rawiai.models.schemas import EnrichedVerse
from rawiai.nlp.normalizer import ArabicNormalizer
from rawiai.nlp.stemmer import PoetryArabicStemmer
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator


class RetrievalBenchmark:
    """Benchmark suite for validating the retrieval layer independently from generation."""

    def __init__(
        self,
        orchestrator: DualRAGOrchestrator,
        dataset_path: Optional[str] = None
    ):
        self.orchestrator = orchestrator
        self.dataset_path = dataset_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "golden_evaluation_dataset.json"
        )
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.test_cases: List[Dict[str, Any]] = json.load(f)

    def evaluate(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes benchmark over all golden test cases.
        Returns detailed scores and metric summaries.
        """
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_ranks = []
        valid_search_cases = 0

        refusal_tests = 0
        refusal_successes = 0

        lexicon_recall_scores = []
        case_results = []

        for tc in self.test_cases:
            tc_id = tc["id"]
            query = tc["query"]
            target_sub = tc.get("target_verse_substring")
            expected_roots = tc.get("expected_lexicon_roots", [])
            expected_intent = tc.get("intent")

            # 1. Intent check
            detected_intent = self.orchestrator.classify_intent(query)
            intent_correct = (detected_intent == expected_intent)

            # 2. Search query cleaning
            search_query = self.orchestrator.clean_search_query(query)
            norm_search = ArabicNormalizer.normalize_search(search_query)
            query_stems = PoetryArabicStemmer.tokenize_and_stem(search_query, remove_stopwords=True)

            # 3. Hybrid scoring across text, poet, theme, and stems
            ranked_matches = []
            for record in self.orchestrator.poetry_records:
                score = 0.0
                norm_v = record.normalized_search
                norm_poet = ArabicNormalizer.normalize_search(record.poet or "")
                norm_theme = ArabicNormalizer.normalize_search(record.theme or "")
                
                # Exact phrase match has top priority
                if norm_search and norm_search in norm_v:
                    score = 1.0
                else:
                    # Token overlap with verse
                    token_matches = sum(1 for t in query_stems if t in record.stemmed_tokens)
                    # Poet name match
                    poet_match = sum(1 for t in query_stems if t in norm_poet) * 1.5
                    # Theme match
                    theme_match = sum(1 for t in query_stems if t in norm_theme) * 1.0

                    total_points = token_matches + poet_match + theme_match
                    if query_stems:
                        score = min(0.99, total_points / len(query_stems))

                if score > 0.15:
                    ranked_matches.append((score, record))

            ranked_matches.sort(key=lambda x: x[0], reverse=True)
            retrieved_records = [r for _, r in ranked_matches[:top_k]]

            # Refusal test case (target_sub is None)
            if target_sub is None:
                refusal_tests += 1
                top_score = ranked_matches[0][0] if ranked_matches else 0.0
                passed_refusal = (top_score < 0.4)
                if passed_refusal:
                    refusal_successes += 1
                case_results.append({
                    "id": tc_id,
                    "query": query,
                    "type": "refusal",
                    "passed": passed_refusal,
                    "intent": detected_intent
                })
                continue

            # Standard retrieval test case
            valid_search_cases += 1
            norm_target = ArabicNormalizer.normalize_search(target_sub)
            rank_found = None

            for rank_idx, rec in enumerate(retrieved_records, start=1):
                if norm_target in rec.normalized_search or rec.normalized_search in norm_target:
                    rank_found = rank_idx
                    break

            if rank_found is not None:
                if rank_found == 1:
                    hits_at_1 += 1
                if rank_found <= 3:
                    hits_at_3 += 1
                if rank_found <= 5:
                    hits_at_5 += 1
                reciprocal_ranks.append(1.0 / rank_found)
            else:
                reciprocal_ranks.append(0.0)

            # Lexicon Multi-Hop check if roots expected
            lex_score = 1.0
            if expected_roots and rank_found is not None:
                matched_verse = retrieved_records[rank_found - 1]
                lex_entries = self.orchestrator.retrieve_lexicon_for_verse(matched_verse)
                found_roots = {e.root for e in lex_entries}
                matched_roots = sum(1 for r in expected_roots if r in found_roots)
                lex_score = matched_roots / len(expected_roots)
                lexicon_recall_scores.append(lex_score)

            case_results.append({
                "id": tc_id,
                "query": query,
                "type": "search",
                "rank": rank_found,
                "hit@1": (rank_found == 1),
                "hit@3": (rank_found is not None and rank_found <= 3),
                "mrr": (1.0 / rank_found) if rank_found else 0.0,
                "intent_correct": intent_correct,
                "lexicon_score": lex_score
            })

        hit_1_rate = (hits_at_1 / valid_search_cases) if valid_search_cases else 0.0
        hit_3_rate = (hits_at_3 / valid_search_cases) if valid_search_cases else 0.0
        hit_5_rate = (hits_at_5 / valid_search_cases) if valid_search_cases else 0.0
        mean_mrr = (sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0
        refusal_rate = (refusal_successes / refusal_tests) if refusal_tests else 1.0
        avg_lex_score = (sum(lexicon_recall_scores) / len(lexicon_recall_scores)) if lexicon_recall_scores else 1.0

        return {
            "summary": {
                "total_test_cases": len(self.test_cases),
                "valid_search_cases": valid_search_cases,
                "refusal_cases": refusal_tests,
                "hit@1": round(hit_1_rate * 100, 1),
                "hit@3": round(hit_3_rate * 100, 1),
                "hit@5": round(hit_5_rate * 100, 1),
                "mrr": round(mean_mrr, 3),
                "refusal_accuracy": round(refusal_rate * 100, 1),
                "lexicon_multi_hop_recall": round(avg_lex_score * 100, 1),
            },
            "cases": case_results
        }
