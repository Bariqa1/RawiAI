"""
Unit tests for ProsodyRulesRetriever (Mizan Al-Dhahab Knowledge Engine).
"""

import pytest
from rawiai.rag.prosody_rules_retriever import ProsodyRulesRetriever


class TestProsodyRulesRetriever:
    """Validates the Mizan Al-Dhahab prosody retrieval engine."""

    @pytest.fixture(autouse=True)
    def setup_retriever(self):
        self.retriever = ProsodyRulesRetriever()

    def test_all_16_meters_loaded(self):
        meters = self.retriever.list_all_meters()
        assert len(meters) == 16
        expected_sample = ["الطويل", "الكامل", "البسيط", "الوافر", "الخفيف", "الرمل", "الرجز", "المتقارب"]
        for m in expected_sample:
            assert m in meters

    def test_get_meter_details(self):
        tawil = self.retriever.get_meter("الطويل")
        assert tawil is not None
        assert "طَوِيلٌ لَهُ دُونَ البُحُورِ فَضَائِلُ" in tawil["mnemonic_key"]
        assert "فَعُولُنْ" in tawil["standard_tafail"]
        assert len(tawil["permissible_zihafat"]) > 0

        kamil = self.retriever.get_meter("الكامل")
        assert kamil is not None
        assert "مُتَفَاعِلُنْ" in kamil["standard_tafail"]

    def test_get_meter_mnemonic_key(self):
        key = self.retriever.get_meter_key("الوافر")
        assert key is not None
        assert "وَافِرُهَا جَمِيلُ" in key

    def test_get_defect_diagnoses(self):
        iqwa = self.retriever.get_defect_info("الإقواء")
        assert iqwa is not None
        assert "اختلاف حركة الروي" in iqwa["definition"]
        assert "توحيد حركة الإعراب" in iqwa["remedy"]

        ikfa = self.retriever.get_defect_info("الإكفاء")
        assert ikfa is not None
        assert "المخرج الصوتي" in ikfa["definition"]

        ita = self.retriever.get_defect_info("الإيطاء")
        assert ita is not None
        assert "تكرار" in ita["definition"]

        kasr = self.retriever.get_defect_info("الكسر العروضي")
        assert kasr is not None
        assert "تفعيلات" in kasr["definition"]

    def test_recommend_meter_for_theme(self):
        fakhr_meters = self.retriever.recommend_meter_for_theme("فخر وحماسة")
        assert len(fakhr_meters) > 0
        names = [m["name"] for m in fakhr_meters]
        assert "الطويل" in names or "الكامل" in names

    def test_format_prosody_reference_context(self):
        context = self.retriever.format_prosody_reference_context(meter_name="الطويل")
        assert "ميزان الذهب" in context
        assert "أحمد الهاشمي" in context
        assert "الطويل" in context
        assert "طَوِيلٌ لَهُ دُونَ البُحُورِ فَضَائِلُ" in context
