"""
RawiAI (راوي) - Advanced Classical Arabic Poetry NLP & RAG Suite
Designed for heritage Arabic literature, prosody analysis, and grounded RAG.
"""

__version__ = "0.2.0"

from rawiai.master_orchestrator import RawiMasterOrchestrator, RawiResponse, RawiIntent
from rawiai.agents.poetic_council import PoeticCouncil
from rawiai.agents.guardrails import PoeticGuardrails

__all__ = [
    "RawiMasterOrchestrator",
    "RawiResponse",
    "RawiIntent",
    "PoeticCouncil",
    "PoeticGuardrails"
]
