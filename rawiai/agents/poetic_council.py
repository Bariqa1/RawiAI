"""
Poetic Council Orchestrator (مجلس الرواة والشعراء).
Coordinates the Multi-Agent collaboration loop between:
1. Muse Agent (الإلهام والمعجم والبحر)
2. Poet Agent (الشاعر الناظم)
3. Arud Critic Agent (الناقد العروضي ومحكم الشعر)
"""

from typing import List, Dict, Optional, Any, Union
from rawiai.agents.schemas import (
    PoemCompositionRequest,
    MuseInspiration,
    VerseDraft,
    CouncilCritiqueReport,
    GeneratedPoem
)
from rawiai.agents.muse_agent import MuseAgent
from rawiai.agents.poet_agent import PoetAgent
from rawiai.agents.critic_agent import ArudCriticAgent
from rawiai.rag.lexicon_retriever import AsasLexiconRetriever
from rawiai.nlp.prosody import ProsodyAnalyzer


class PoeticCouncil:
    """Master Multi-Agent Council for Arabic Poetry Generation and Verification."""

    def __init__(
        self,
        muse_agent: Optional[MuseAgent] = None,
        poet_agent: Optional[PoetAgent] = None,
        critic_agent: Optional[ArudCriticAgent] = None,
        lexicon_retriever: Optional[AsasLexiconRetriever] = None,
        prosody_analyzer: Optional[ProsodyAnalyzer] = None
    ):
        lex = lexicon_retriever or AsasLexiconRetriever()
        pros = prosody_analyzer or ProsodyAnalyzer()

        self.muse = muse_agent or MuseAgent(lexicon_retriever=lex)
        self.poet = poet_agent or PoetAgent()
        self.critic = critic_agent or ArudCriticAgent(prosody_analyzer=pros)

    def compose_poem(
        self,
        request_or_topic: Union[PoemCompositionRequest, str],
        max_refinement_rounds: int = 3,
        **kwargs
    ) -> GeneratedPoem:
        """
        Orchestrates the end-to-end multi-agent poetry composition lifecycle:
        1. Inspiration & Rhetoric (Muse)
        2. Initial Verse Drafting (Poet)
        3. Prosodic Auditing & Feedback (Critic)
        4. Iterative Refinement Loop (Actor-Critic)
        5. Final Classical Assembly
        """
        if isinstance(request_or_topic, str):
            request = PoemCompositionRequest(topic=request_or_topic, **kwargs)
        else:
            request = request_or_topic

        # Step 1: Muse Inspiration
        inspiration = self.muse.inspire(request)

        # Step 2: Initial Composition by Poet
        current_verses = self.poet.compose(inspiration, request.verse_count)

        # Step 3: Iterative Actor-Critic Refinement Loop
        final_critique = None
        rounds_used = 1

        for r in range(max_refinement_rounds):
            final_critique = self.critic.audit_poem(current_verses, inspiration)
            if final_critique.all_balanced:
                rounds_used = r + 1
                break

            # Send back to Poet with Critic's directives
            current_verses = self.poet.revise(current_verses, final_critique, inspiration)
            rounds_used = r + 1

        # Final audit after revisions
        final_critique = self.critic.audit_poem(current_verses, inspiration)

        # Title Formulation
        title = f"قصيدة في ({request.topic}) على بحر {inspiration.selected_meter}"

        poem = GeneratedPoem(
            title=title,
            topic=request.topic,
            theme=inspiration.theme,
            meter=inspiration.selected_meter,
            meter_tafail=inspiration.meter_tafail,
            rhyme=inspiration.selected_rhyme,
            verses=current_verses,
            iterations_used=rounds_used,
            critique_report=final_critique,
            metaphor_sources=inspiration.lexicon_metaphors
        )
        poem.full_text = poem.format_display()
        return poem
