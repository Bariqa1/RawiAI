"""
Tests for AsasBalaghaParser and Full Lexicon Ingestion.
"""

import os
import pytest
from rawiai.processors.asas_parser import AsasBalaghaParser
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.nlp.lexicon import PoetryLexicon


class TestAsasBalaghaParser:
    """Test suite for parsing and structuring Al-Zamakhshari's Asas Al-Balagha."""

    SAMPLE_RAW_STREAM = """
### | ع ذ ل
# رجل عذلة خذلة وعذالة خذالة. قال تأبط شرا:
# يا من لعذالة خذالة أشب ... حرق باللوم جلدي أي تحراق
# وعذلته فاعتذل أي عذل نفسه وأعتب ورمى فأخطأ ثم اعتذل.
# ومن المجاز: قول الراعي:
# ثم انصرفت وظل الحلم يعذلني ... قد طال ما قادني جهلي وعناني
# وقد اعتذل يومنا إذا اشتد حره.
### | س ي ف
# سافه وتسيفه: ضربه بالسيف، وسايفه وتسايفوا.
# ومن المجاز: بين فكيه سيف صارم.
"""

    def test_clean_text_stream(self):
        wrapped = "### | أ ب\n# و\n# content here"
        cleaned = AsasBalaghaParser.clean_text_stream(wrapped)
        assert "### | أ ب و" in cleaned

    def test_parse_sample(self, tmp_path):
        sample_file = tmp_path / "sample_asas.txt"
        sample_file.write_text(self.SAMPLE_RAW_STREAM, encoding="utf-8")

        parser = AsasBalaghaParser(raw_path=str(sample_file))
        entries = parser.parse()

        assert len(entries) == 2
        roots = [e["root"] for e in entries]
        assert "عذل" in roots
        assert "سيف" in roots

        azl_entry = next(e for e in entries if e["root"] == "عذل")
        assert "رجل عذلة" in azl_entry["literal_meaning"]
        assert "ومن المجاز:" in azl_entry["metaphorical_meaning"]
        assert len(azl_entry["poetic_citations"]) >= 1

    def test_full_dataset_loaded_in_retriever(self):
        full_json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "rawiai",
            "data",
            "asas_al_balagha_full.json"
        )
        if os.path.exists(full_json_path):
            retriever = AsasLexiconRetriever(data_path=full_json_path)
            assert len(retriever.entries) >= 3000

            # Test high-frequency roots
            for root_query in ["سيف", "عذل", "خيل", "بيد"]:
                results = retriever.retrieve(root_query, top_k=1)
                assert len(results) > 0
                assert results[0].entry.root == root_query

    def test_full_dataset_loaded_in_poetry_lexicon(self):
        full_json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "rawiai",
            "data",
            "asas_al_balagha_full.json"
        )
        if os.path.exists(full_json_path):
            lex = PoetryLexicon(custom_lexicon_path=full_json_path)
            assert len(lex.entries) >= 3000
            word = lex.lookup("عذل")
            assert word is not None
            assert word.root == "عذل"
