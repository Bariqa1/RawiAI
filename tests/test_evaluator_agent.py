"""
Unit tests for HumanPoetryEvaluator and MasterOrchestrator Critique Intent.
"""

import pytest
from rawiai.agents.evaluator_agent import HumanPoetryEvaluator
from rawiai.master_orchestrator import RawiMasterOrchestrator, RawiIntent


class TestHumanPoetryEvaluator:
    """Validates the Human Poetry Evaluator against Mizan Al-Dhahab criteria."""

    @pytest.fixture(autouse=True)
    def setup_evaluator(self):
        self.evaluator = HumanPoetryEvaluator()
        self.orchestrator = RawiMasterOrchestrator()

    def test_evaluate_sound_classical_verse(self):
        # Sound verse on Al-Tawil from Al-Mutanabbi
        verse = "عَلى قَدْرِ أَهْلِ العَزْمِ تَأْتِي العَزائِمُ ... وَتَأْتِي عَلَى قَدْرِ الكِرامِ المَكارِمُ"
        report = self.evaluator.evaluate_text(verse)

        assert report.total_verses == 1
        assert report.is_fully_sound is True
        assert report.overall_score == 100.0
        assert report.dominant_meter == "الطويل"
        assert report.dominant_rhyme == "م"
        assert len(report.poetic_defects_found) == 0
        assert "طَوِيلٌ لَهُ دُونَ البُحُورِ فَضَائِلُ" in report.meter_mnemonic_key

    def test_evaluate_broken_prose_verse(self):
        # Prose text lacking poetic cadence
        prose = "اليوم ذهبت إلى السوق مسرعا ... واشتريت تفاحا وأشياء أخرى"
        report = self.evaluator.evaluate_text(prose)

        assert report.total_verses == 1
        assert report.is_fully_sound is False
        assert report.overall_score < 50.0
        assert report.broken_verses_count >= 1
        defect_names = [d["name"] for d in report.poetic_defects_found]
        assert "الكسر العروضي" in defect_names

    def test_evaluate_rhyme_mismatch_ikfa(self):
        # Two verses with conflicting rhyme letters (Lam vs Raa)
        verses = (
            "سَلْ عَنْ شَجَاعَتِنَا العَوَالِيَ وَالأَسَلْ ... تَلْقَ الكَمِيَّ إِذَا ادْلَهَمَّ المَوْتُ صَلْ\n"
            "حَكَّمْتُ سَيْفِي فِي الصِّعَابِ فَمَا انْثَنَى ... وَطَلَبْتُ عِزّاً فِي الشَّدَائِدِ فَانْتَصَرْ"
        )
        report = self.evaluator.evaluate_text(verses)

        assert report.total_verses == 2
        assert report.is_fully_sound is False
        defect_names = [d["name"] for d in report.poetic_defects_found]
        assert "الإكفاء" in defect_names

    def test_evaluate_rhyme_repetition_ita(self):
        # Same rhyme word repeated in consecutive verses
        verses = (
            "سَلْ عَنْ شَجَاعَتِنَا العَوَالِيَ وَالأَسَلْ ... تَلْقَ الكَمِيَّ إِذَا ادْلَهَمَّ المَوْتُ صَلْ\n"
            "حَكَّمْتُ سَيْفِي فِي الصِّعَابِ فَمَا انْثَنَى ... فَرَمَى العِدَى فِي مَوْقِفِ الهَيْجَاءِ صَلْ"
        )
        report = self.evaluator.evaluate_text(verses)

        assert report.total_verses == 2
        defect_names = [d["name"] for d in report.poetic_defects_found]
        assert "الإيطاء" in defect_names

    def test_orchestrator_routes_critique_intent(self):
        query = "قيم لي هذا البيت: عَلى قَدْرِ أَهْلِ العَزْمِ تَأْتِي العَزائِمُ ... وَتَأْتِي عَلَى قَدْرِ الكِرامِ المَكارِمُ"
        response = self.orchestrator.process(query)

        assert response.intent == RawiIntent.CRITIQUE
        assert response.success is True
        assert response.evaluation_report is not None
        assert response.evaluation_report.dominant_meter == "الطويل"
        assert "تقرير التحكيم والنقد الشعري" in response.response_text

    def test_orchestrator_direct_evaluation_method(self):
        verse = "حَكِّمْ سُيُوفَكَ فِي رِقَابِ العُذَّلِ ... وَإِذَا نَزَلْتَ بِدَارِ ذُلٍّ فَارْحَلِ"
        report = self.orchestrator.evaluate_poem(verse)

        assert report.total_verses == 1
        assert report.dominant_rhyme == "ل"
        assert report.is_fully_sound is True
