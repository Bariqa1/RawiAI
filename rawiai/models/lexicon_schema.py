"""
Data models for Classical Lexicon entries (specifically Asas Al-Balagha by Al-Zamakhshari).
Separates literal semantic definitions (الحقيقة) from metaphorical and poetic usages (المجاز).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AsasEntry(BaseModel):
    """
    Structured entry representing a root in 'Asas Al-Balagha' by Al-Zamakhshari.
    Differentiates between literal language and metaphorical rhetorical usage in poetry.
    """
    root: str = Field(description="The Arabic root (الجذر: عذل، سيف، بيد، إلخ)")
    lemma: str = Field(description="Canonical headword or display lemma (المادة اللغوية)")
    literal_meaning: str = Field(description="The primary literal meaning (ومن الحقيقة)")
    metaphorical_meaning: str = Field(description="The metaphorical/figurative poetic usages (ومن المجاز)")
    poetic_citations: List[str] = Field(default_factory=list, description="Classical poetic verses or Arab sayings cited by Al-Zamakhshari")
    source: str = Field(default="أساس البلاغة - الزمخشري", description="Reference dictionary")

    def to_retrieval_chunk(self) -> str:
        """Compact text representation for embedding and sparse retrieval."""
        parts = [
            f"مادة ({self.lemma}) - جذر [{self.root}]:",
            f"الحقيقة: {self.literal_meaning}",
            f"المجاز: {self.metaphorical_meaning}",
        ]
        if self.poetic_citations:
            citations_str = " | ".join(self.poetic_citations)
            parts.append(f"شواهد: {citations_str}")
        return "\n".join(parts)

    def to_llm_context(self) -> str:
        """Formatted block for injection into LLM prompt context."""
        lines = [
            f"• مادة [{self.lemma}] (الجذر: {self.root}) من {self.source}:",
            f"  - المعنى الحقيقي: {self.literal_meaning}",
            f"  - الاستعمال المجازي والبلاغي: {self.metaphorical_meaning}",
        ]
        if self.poetic_citations:
            lines.append("  - شواهد شعرية وبلاغية من كلام العرب:")
            for cite in self.poetic_citations:
                lines.append(f"    * {cite}")
        return "\n".join(lines)


class LexiconSearchResult(BaseModel):
    """Represents a scored candidate retrieved from the Lexicon RAG index."""
    entry: AsasEntry
    score: float = 0.0
    match_type: str = "hybrid"  # "exact_root", "bm25", "semantic", "hybrid"
