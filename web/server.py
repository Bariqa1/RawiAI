"""
Web Server and REST API for RawiAI.
Provides live endpoints connecting the modern web UI to:
- RawiMasterOrchestrator (Poetic Council & Composition)
- HumanPoetryEvaluator (Poetry Critique & Metric Diagnosis)
- ProsodyRulesRetriever (Mizan Al-Dhahab Encyclopedia)
- AsasLexiconRetriever (Asas Al-Balagha Metaphors)
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rawiai.master_orchestrator import RawiMasterOrchestrator, RawiIntent
from rawiai.agents.evaluator_agent import HumanPoetryEvaluator
from rawiai.rag.prosody_rules_retriever import ProsodyRulesRetriever
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever

from dotenv import load_dotenv
load_dotenv()

PORT = 8080
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


class RawiAPIHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving static files and API endpoints."""

    # Singletons
    orchestrator = None
    evaluator = None
    prosody_rules = None
    lexicon = None

    @classmethod
    def get_services(cls):
        if cls.orchestrator is None:
            load_dotenv()
            cls.prosody_rules = ProsodyRulesRetriever()
            cls.lexicon = AsasLexiconRetriever()
            cls.evaluator = HumanPoetryEvaluator(prosody_retriever=cls.prosody_rules, use_llm=True)
            cls.orchestrator = RawiMasterOrchestrator(
                lexicon_retriever=cls.lexicon,
                prosody_retriever=cls.prosody_rules,
                evaluator=cls.evaluator,
                use_llm=True
            )
        return cls.orchestrator, cls.evaluator, cls.prosody_rules, cls.lexicon

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/meters":
            self._handle_get_meters()
        elif path.startswith("/api/meter/"):
            meter_name = path.replace("/api/meter/", "")
            self._handle_get_meter_detail(meter_name)
        elif path == "/api/search_lexicon":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            self._handle_search_lexicon(query)
        elif path == "/api/lookup_poet":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            self._handle_lookup_poet(query)
        elif path == "/api/ask":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            self._handle_ask(query)
        elif path == "/api/stats":
            self._handle_get_stats()
        else:
            # Fallback to serving static files
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(post_data)
        except Exception:
            payload = {}

        if path == "/api/compose":
            self._handle_compose(payload)
        elif path == "/api/evaluate":
            self._handle_evaluate(payload)
        elif path == "/api/ask":
            self._handle_ask(payload)
        else:
            self._send_json({"error": "Endpoint not found"}, status=404)

    def _handle_get_stats(self):
        _, _, rules, lex = self.get_services()
        data = {
            "meters_count": len(rules.list_all_meters()),
            "lexicon_entries": len(lex.entries),
            "reference_treatise": rules.treatise_title,
            "lexicon_title": "أساس البلاغة للزمخشري"
        }
        self._send_json(data)

    def _handle_get_meters(self):
        _, _, rules, _ = self.get_services()
        meters = []
        for name in rules.list_all_meters():
            m = rules.get_meter(name)
            if m:
                meters.append({
                    "name": m["name"],
                    "mnemonic_key": m.get("mnemonic_key", ""),
                    "standard_tafail": m.get("standard_tafail", ""),
                    "suitable_themes": m.get("suitable_themes", []),
                    "aesthetic_notes": m.get("aesthetic_notes", "")
                })
        self._send_json({"meters": meters})

    def _handle_get_meter_detail(self, name: str):
        from urllib.parse import unquote
        meter_name = unquote(name)
        _, _, rules, _ = self.get_services()
        m = rules.get_meter(meter_name)
        if not m:
            self._send_json({"error": "Meter not found"}, status=404)
            return

        lesson = rules.get_meter_full_lesson(meter_name)
        self._send_json({
            "meter": m,
            "full_lesson_text": lesson or "النص الكامل متوفر في كتاب ميزان الذهب."
        })

    def _handle_search_lexicon(self, query: str):
        _, _, _, lex = self.get_services()
        results = lex.retrieve(query, top_k=5)
        serialized = []
        for r in results:
            e = r.entry
            serialized.append({
                "lemma": e.lemma,
                "root": e.root,
                "literal_meaning": e.literal_meaning,
                "metaphorical_meaning": e.metaphorical_meaning,
                "poetic_citations": e.poetic_citations,
                "score": round(r.score, 2)
            })
        self._send_json({"query": query, "results": serialized})

    def _handle_compose(self, payload: Dict[str, Any]):
        topic = payload.get("topic", "").strip()
        meter = payload.get("meter")
        rhyme = payload.get("rhyme")
        verse_count = int(payload.get("verse_count", 3))

        if not topic:
            self._send_json({"error": "يرجى إدخال موضوع القصيدة"}, status=400)
            return

        orchestrator, _, _, _ = self.get_services()
        prompt = f"اكتب لي شعر عن {topic}"
        if meter:
            prompt += f" على بحر {meter}"
        if rhyme:
            prompt += f" بقافية {rhyme}"

        resp = orchestrator.process(prompt, verse_count=verse_count)
        if resp.poem:
            p = resp.poem
            verses = [
                {
                    "number": v.verse_number,
                    "sadr": re.sub(r"^[0-9٠-٩]+[\.\-\)\:\s]*", "", v.sadr).strip(),
                    "ajuz": re.sub(r"^[0-9٠-٩]+[\.\-\)\:\s]*", "", v.ajuz).strip(),
                    "full": v.full_verse
                }
                for v in p.verses
            ]
            metaphor_sources = [
                {
                    "root": m.get("root", ""),
                    "lemma": m.get("lemma", ""),
                    "metaphor": m.get("metaphorical_meaning") or m.get("metaphor") or m.get("literal_meaning", ""),
                    "metaphorical_meaning": m.get("metaphorical_meaning", ""),
                    "literal_meaning": m.get("literal_meaning", "")
                }
                for m in p.metaphor_sources
            ]
            self._send_json({
                "success": True,
                "title": p.title,
                "topic": p.topic,
                "theme": p.theme,
                "meter": p.meter,
                "meter_tafail": p.meter_tafail,
                "rhyme": p.rhyme,
                "iterations": p.iterations_used,
                "verses": verses,
                "formatted_text": p.format_display(),
                "metaphor_sources": metaphor_sources,
                "quality_score": p.critique_report.overall_quality_score
            })
        else:
            self._send_json({
                "success": False,
                "message": resp.response_text
            })

    def _handle_evaluate(self, payload: Dict[str, Any]):
        text = payload.get("text", "").strip()
        if not text:
            self._send_json({"error": "يرجى إدخال أبيات الشعر للتحكيم"}, status=400)
            return

        orchestrator, evaluator, _, _ = self.get_services()
        report = evaluator.evaluate_text(text)

        # Classical corpus attribution (Kaggle/Heritage records)
        matches = orchestrator.dual_rag.retrieve_poetry(text, top_k=1)
        attribution = None
        if matches:
            top_m = matches[0]
            from rawiai.nlp.normalizer import ArabicNormalizer
            norm_q = ArabicNormalizer.normalize_search(text)
            norm_m = top_m.normalized_search or ArabicNormalizer.normalize_search(top_m.original_text)
            q_words = set(norm_q.split())
            m_words = set(norm_m.split())
            overlap = len(q_words & m_words)
            if overlap >= 2 or norm_q in norm_m or norm_m in norm_q:
                attribution = {
                    "poet": top_m.poet,
                    "era": top_m.era,
                    "title": top_m.poem_title,
                    "theme": top_m.theme,
                    "matched_verse": top_m.original_text
                }

        verse_evals = [
            {
                "number": v.verse_number,
                "sadr": v.sadr,
                "ajuz": v.ajuz,
                "detected_meter": v.detected_meter,
                "confidence": v.meter_confidence,
                "is_meter_valid": v.is_meter_valid,
                "detected_rhyme": v.detected_rhyme,
                "is_rhyme_valid": v.is_rhyme_valid,
                "status": v.prosodic_status,
                "defects": v.defects,
                "suggestion": v.remediation_suggestion
            }
            for v in report.verse_evaluations
        ]

        self._send_json({
            "success": True,
            "overall_score": report.overall_score,
            "is_fully_sound": report.is_fully_sound,
            "dominant_meter": report.dominant_meter,
            "meter_tafail": report.meter_tafail,
            "meter_mnemonic_key": report.meter_mnemonic_key,
            "dominant_rhyme": report.dominant_rhyme,
            "sound_verses_count": report.sound_verses_count,
            "total_verses": report.total_verses,
            "attribution": attribution,
            "verse_evaluations": verse_evals,
            "poetic_defects": report.poetic_defects_found,
            "general_critique": report.general_critique,
            "rhetorical_analysis": report.rhetorical_analysis,
            "remedies": report.remedy_recommendations,
            "formatted_display": report.format_display()
        })

    def _handle_lookup_poet(self, query: str):
        if not query.strip():
            self._send_json({"error": "يرجى إدخال نص البيت أو اسم الشاعر للبحث"}, status=400)
            return

        orchestrator, _, _, lex = self.get_services()
        matches = orchestrator.dual_rag.retrieve_poetry(query, top_k=6)
        results = []
        for m in matches:
            lex_notes = []
            for w in (m.difficult_words or []):
                w_str = w.word if hasattr(w, "word") else (w.get("word") if isinstance(w, dict) else str(w))
                lex_res = lex.retrieve(w_str, top_k=1)
                if lex_res:
                    e = lex_res[0].entry
                    lex_notes.append({
                        "word": w_str,
                        "meaning": e.metaphorical_meaning or e.literal_meaning
                    })

            results.append({
                "poet": m.poet or "غير محدد",
                "era": m.era or "تراثي",
                "title": m.poem_title or "ديوان الشاعر",
                "theme": m.theme or "عام",
                "verse": m.original_text,
                "sadr": m.sadr,
                "ajuz": m.ajuz,
                "meter": m.prosody.meter if m.prosody else "غير محدد",
                "tafail": m.prosody.meter_tafail if m.prosody else "",
                "rhyme": m.prosody.rhyme_letter if m.prosody else "",
                "lexicon_notes": lex_notes
            })

        self._send_json({
            "query": query,
            "total_matches": len(results),
            "results": results
        })

    def _handle_ask(self, payload_or_query: Any):
        if isinstance(payload_or_query, dict):
            query = payload_or_query.get("query", "").strip()
        else:
            query = str(payload_or_query).strip()

        if not query:
            self._send_json({"error": "يرجى كتابة سؤالك أو استعلامك"}, status=400)
            return

        orchestrator, _, _, lex = self.get_services()
        resp = orchestrator.process(query)

        verses_data = []
        for v in resp.retrieved_verses:
            lex_notes = []
            text = v.get("verse_text", "")
            tokens = text.split()
            for token in tokens[:6]:
                clean_tok = token.strip("،.؛:؟!* ")
                if len(clean_tok) >= 3:
                    matches = lex.retrieve(clean_tok, top_k=1)
                    if matches and matches[0].score >= 1.5:
                        e = matches[0].entry
                        lex_notes.append({
                            "word": clean_tok,
                            "meaning": e.metaphorical_meaning or e.literal_meaning
                        })

            verses_data.append({
                "verse_text": v.get("verse_text", ""),
                "sadr": v.get("sadr", ""),
                "ajuz": v.get("ajuz", ""),
                "poet": v.get("poet", "غير محدد"),
                "era": v.get("era", "تراثي"),
                "meter": v.get("meter", "غير محدد"),
                "rhyme": v.get("rhyme", ""),
                "title": v.get("title", "ديوان الشاعر"),
                "theme": v.get("theme", "عام"),
                "lexicon_notes": lex_notes
            })

        intent_labels = {
            "author": "معرفة قائل البيت وتوثيق نسبته",
            "completion": "إكمال الشطر أو البيت التراثي",
            "meaning": "شرح البيت ومجازاته وبلاغته",
            "search": "استخراج الشواهد من ديوان الشعر",
            "critique": "تحكيم عروضي ونقدي",
            "compose": "نظم شعر جديد",
            "blocked": "مخالف لضوابط الأمان"
        }

        self._send_json({
            "success": resp.success,
            "query": query,
            "intent": resp.intent,
            "intent_label": intent_labels.get(resp.intent, "استعلام شعري"),
            "answer": resp.response_text,
            "retrieved_verses": verses_data
        })

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)


def run(port=PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, RawiAPIHandler)
    print(f"RawiAI Web Server running on: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run(port)
