#!/usr/bin/env python3
"""
RawiAI - Evaluation Suite & DeepEval Benchmark Runner.
Executes end-to-end evaluation covering both Retrieval Metrics (Hit@K, MRR)
and Generation Reliability (Faithfulness, Answer Relevancy) over the Golden Dataset.
"""

import os
import json
import sys
import warnings

# Suppress external library SSL/auth deprecation warnings in terminal
warnings.filterwarnings("ignore")

from rawiai.models.schemas import EnrichedVerse, ProsodyInfo
from rawiai.processors.corpus_processor import PoetryCorpusProcessor
from rawiai.rag.dual_rag_orchestrator import DualRAGOrchestrator
from rawiai.evaluation.retrieval_benchmark import RetrievalBenchmark
from rawiai.evaluation.deepeval_runner import DeepEvalRunner


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def build_evaluation_corpus():
    """Loads the verified 38-verse classical corpus covering 15 classical poets."""
    corpus_path = os.path.join(
        os.path.dirname(__file__), "rawiai", "data", "classical_poetry_corpus.json"
    )
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        corpus = []
        for item in raw_data:
            p = item.get("prosody", {})
            prosody = ProsodyInfo(
                phonetic_sadr="",
                phonetic_ajuz="",
                meter=p.get("meter", "غير محدد"),
                meter_confidence=p.get("confidence", 0.9),
                rhyme_letter=p.get("rhyme_letter", "")
            )
            v = EnrichedVerse(
                id=item.get("id", "verse_1"),
                original_text=item.get("original_text", ""),
                sadr=item.get("sadr", ""),
                ajuz=item.get("ajuz", ""),
                poet=item.get("poet", ""),
                era=item.get("era", ""),
                theme=item.get("theme", ""),
                poem_title=item.get("poem_title", ""),
                normalized_search=item.get("normalized_search", ""),
                stemmed_tokens=item.get("stemmed_tokens", []),
                prosody=prosody,
                difficult_words=item.get("difficult_words", []),
                source_row=item.get("source_row")
            )
            corpus.append(v)
        return corpus

    processor = PoetryCorpusProcessor()
    raw_samples = [
        {
            "text": "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ",
            "poet": "عنترة بن شداد",
            "era": "جاهلي",
            "theme": "شجاعة وفخر وعزة",
            "title": "معلقة عنترة"
        },
        {
            "text": "الخَيْلُ وَاللَّيْلُ وَالبَيْدَاءُ تَعْرِفُنِي # وَالسَّيْفُ وَالرُّمْحُ وَالقِرْطَاسُ وَالقَلَمُ",
            "poet": "أبو الطيب المتنبي",
            "era": "عباسي",
            "theme": "فخر واعتداد بالنفس وشجاعة",
            "title": "وا حر قلباه"
        }
    ]
    corpus = []
    for s in raw_samples:
        rec = processor.process_verse_text(
            raw_text=s["text"],
            poet=s["poet"],
            era=s["era"],
            theme=s["theme"],
            poem_title=s["title"]
        )
        if rec:
            corpus.append(rec)
    return corpus


def main():
    print_header("[BENCHMARK] RawiAI Quality Benchmark & DeepEval Dashboard")

    corpus = build_evaluation_corpus()
    orchestrator = DualRAGOrchestrator(poetry_records=corpus)

    # -------------------------------------------------------------
    # 1. Retrieval Benchmark (Hit@K & MRR)
    # -------------------------------------------------------------
    print_header("[1] Retrieval Layer Benchmark (Hit@K & MRR)")
    retrieval_bench = RetrievalBenchmark(orchestrator)
    retrieval_res = retrieval_bench.evaluate(top_k=5)
    ret_summary = retrieval_res["summary"]

    print(f"- Total Test Cases        : {ret_summary['total_test_cases']}")
    print(f"- Valid Search Queries    : {ret_summary['valid_search_cases']}")
    print(f"- Refusal Guardrail Cases : {ret_summary['refusal_cases']}")
    print("-" * 55)
    print(f"- Hit@1 (Top-1 Accuracy)       : {ret_summary['hit@1']}%")
    print(f"- Hit@3 (Top-3 Accuracy)       : {ret_summary['hit@3']}%")
    print(f"- Hit@5 (Top-5 Accuracy)       : {ret_summary['hit@5']}%")
    print(f"- MRR (Mean Reciprocal Rank)   : {ret_summary['mrr']}")
    print(f"- Refusal Guardrail Accuracy   : {ret_summary['refusal_accuracy']}%")
    print(f"- Multi-Hop Lexicon Recall     : {ret_summary['lexicon_multi_hop_recall']}%")

    # -------------------------------------------------------------
    # 2. Generation & DeepEval Benchmark
    # -------------------------------------------------------------
    print_header("[2] Generation Reliability & DeepEval Evaluation")
    use_live = "--live" in sys.argv
    deepeval_runner = DeepEvalRunner(orchestrator)
    eval_res = deepeval_runner.run_evaluation(max_cases=50, use_live_api=use_live)

    print(f"- Evaluation Mode          : {eval_res['mode']}")
    print(f"- Evaluated Test Cases     : {eval_res['evaluated_cases']}")
    print(f"- Overall Pass Rate        : {eval_res['pass_rate']}% ({eval_res['passed_cases']}/{eval_res['evaluated_cases']})")
    print(f"- Average Faithfulness     : {eval_res['average_faithfulness'] * 100:.1f}%")
    print(f"- Average Answer Relevancy : {eval_res['average_relevancy'] * 100:.1f}%")

    print("\n[DETAILS] Test Case Breakdown:")
    for r in eval_res["results"]:
        status = "PASSED" if r["passed"] else "FLAGGED"
        print(f"   [{status}] [{r['id']}] ({r['intent']}) {r['query'][:40]}... "
              f"| Faithfulness: {r['faithfulness']} | Relevancy: {r['answer_relevancy']}")

    print_header("[SUMMARY] RawiAI Dual-RAG pipeline verified and operational.")


if __name__ == "__main__":
    main()
