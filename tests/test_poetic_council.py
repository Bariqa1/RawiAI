"""
Unit and Integration Tests for RawiAI Multi-Agent Poetic Council.
Validates MuseAgent, PoetAgent, ArudCriticAgent, and PoeticCouncil orchestrator.
"""

import pytest
from rawiai.agents.schemas import (
    PoemCompositionRequest,
    MuseInspiration,
    VerseDraft,
    CouncilCritiqueReport,
    GeneratedPoem,
    PoeticTheme
)
from rawiai.agents.muse_agent import MuseAgent
from rawiai.agents.poet_agent import PoetAgent
from rawiai.agents.critic_agent import ArudCriticAgent
from rawiai.agents.poetic_council import PoeticCouncil


class TestPoeticCouncilSuite:
    """Test suite for the Multi-Agent Poetic Council."""

    @pytest.fixture
    def council(self):
        return PoeticCouncil()

    def test_muse_agent_theme_and_meter_inference(self, council):
        muse = council.muse
        # Modern tech topic -> Modern theme & Kamil meter
        theme1 = muse.infer_theme("الذكاء الاصطناعي وهندسة البرمجيات")
        assert theme1 == PoeticTheme.MODERN
        meter1 = muse.select_meter(theme1)
        assert meter1 == "الكامل"

        # Chivalry topic -> Fakhr & Tawil
        theme2 = muse.infer_theme("فروسية وشجاعة في معمعة القتال")
        assert theme2 == PoeticTheme.FAKHR
        meter2 = muse.select_meter(theme2)
        assert meter2 == "الطويل"

        # Inspiration blueprint
        req = PoemCompositionRequest(topic="الشجاعة والصبر", verse_count=3)
        insp = muse.inspire(req)
        assert insp.selected_meter in ["الطويل", "الكامل", "البسيط"]
        assert len(insp.lexicon_metaphors) > 0

    def test_critic_agent_approves_balanced_verse(self, council):
        critic = council.critic
        # Known balanced verse on Kamil, rhyme L
        verse = VerseDraft.from_parts(
            verse_number=1,
            sadr="بِالعِلْمِ نَبْنِي فِي الفَضَاءِ مَنَازِلاً",
            ajuz="تَسْمُو عَلَى هَامِ النُّجُومِ وَتَسْتَقِلْ"
        )
        critique = critic.audit_verse(verse, target_meter="الكامل", target_rhyme="ل")
        assert critique.is_balanced is True
        assert critique.rhyme_matches_target is True
        assert critique.detected_rhyme == "ل"

    def test_critic_agent_flags_rhyme_mismatch(self, council):
        critic = council.critic
        # Verse ending with 'م' instead of 'ل'
        verse = VerseDraft.from_parts(
            verse_number=1,
            sadr="بِالعِلْمِ نَبْنِي فِي الفَضَاءِ مَنَازِلاً",
            ajuz="تَسْمُو عَلَى هَامِ النُّجُومِ وَتَسْتَقِيمْ"
        )
        critique = critic.audit_verse(verse, target_meter="الكامل", target_rhyme="ل")
        assert critique.rhyme_matches_target is False
        assert critique.is_balanced is False
        assert "يخالف روي القصيدة" in critique.prosody_feedback

    def test_poet_agent_composition_and_revision(self, council):
        poet = council.poet
        muse = council.muse
        req = PoemCompositionRequest(topic="الذكاء الاصطناعي", verse_count=3)
        insp = muse.inspire(req)

        verses = poet.compose(insp, verse_count=3)
        assert len(verses) == 3
        for v in verses:
            assert len(v.sadr) > 5
            assert len(v.ajuz) > 5
            assert "..." in v.full_verse

    def test_poetic_council_end_to_end_composition(self, council):
        poem = council.compose_poem(
            request_or_topic="حكمة الزمان والوفاء",
            verse_count=3
        )
        assert isinstance(poem, GeneratedPoem)
        assert len(poem.verses) == 3
        assert poem.meter in ["الطويل", "الكامل", "البسيط"]
        assert poem.critique_report.overall_quality_score >= 0.8
        assert "===" in poem.full_text
        assert len(poem.metaphor_sources) > 0
