"""
Tests for Corpus Processor.
"""

import unittest
import pandas as pd
from rawiai.processors.corpus_processor import PoetryCorpusProcessor


class TestCorpusProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = PoetryCorpusProcessor()

    def test_process_dataframe(self):
        sample_data = {
            "القصيدة": [
                "حكم سيوفك في رقاب العذل ... وإذا نزلت بدار ذل فارحل\nواختر لنفسك منزلا تعلو به ... أو مت كريما تحت ظل القسطل",
                "الخيل والليل والبيداء تعرفني ... والسيف والرمح والقرطاس والقلم"
            ],
            "الشاعر": ["عنترة بن شداد", "المتنبي"],
            "العصر": ["جاهلي", "عباسي"],
            "عنوان القصيدة": ["معلقة عنترة", "مدح سيف الدولة"]
        }
        df = pd.DataFrame(sample_data)

        records = self.processor.process_dataframe(df)

        # Expected 3 distinct verses
        self.assertEqual(len(records), 3)

        # First verse checks
        v1 = records[0]
        self.assertEqual(v1.poet, "عنترة بن شداد")
        self.assertEqual(v1.era, "جاهلي")
        self.assertEqual(v1.prosody.rhyme_letter, "ل")
        self.assertIn("سيف", v1.stemmed_tokens)
        self.assertIn("عذل", v1.stemmed_tokens)

        # Check RAG context generation
        context_str = v1.to_rag_context()
        self.assertIn("البيت:", context_str)
        self.assertIn("عنترة بن شداد", context_str)
        self.assertIn("حرف ل", context_str)
        # Should include classical vocabulary definition for العذل
        self.assertIn("المفردات التراثية وغريب الألفاظ:", context_str)


if __name__ == "__main__":
    unittest.main()
