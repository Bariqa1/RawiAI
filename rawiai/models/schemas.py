"""
Data schemas and domain models for Classical Arabic Poetry.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DifficultWord(BaseModel):
    """Represents an annotated archaic or classical vocabulary item with dictionary meaning."""
    word: str = Field(description="The word as found in the verse")
    root: str = Field(description="The triconsonantal or quadriconsonantal root (الجذر)")
    meaning: str = Field(description="Classical explanation / definition (الشرح المعجمي)")
    source_lexicon: str = Field(default="لسان العرب", description="Reference lexicon / dictionary")


class ProsodyInfo(BaseModel):
    """Prosodic metrics for classical Arabic poetry (العروض والقافية)."""
    meter: str = Field(default="غير محدد", description="Poetic meter (البحر العروضي: الطويل، الكامل، إلخ)")
    meter_tafail: Optional[str] = Field(default=None, description="Taf'ilah pattern (التفاعيل)")
    rhyme_letter: str = Field(default="", description="Rhyming consonant (حرف الرويّ)")
    rhyme_type: str = Field(default="مطلقة", description="Rhyme release type (مطلقة أو مقيّدة)")
    confidence: float = Field(default=0.0, description="Prosody detection confidence score between 0.0 and 1.0")


class Hemistich(BaseModel):
    """Represents a single poetic hemistich (شطر: صدر أو عجز)."""
    text: str = Field(description="Raw text of the hemistich")
    normalized: str = Field(description="Search-normalized text")
    token_count: int = Field(default=0, description="Word count")


class EnrichedVerse(BaseModel):
    """
    Comprehensive structured record for a classical Arabic verse.
    Enriched with hemistich segmentation, prosody, search stems, and classical lexicon annotations.
    """
    id: str = Field(description="Unique verse identifier")
    original_text: str = Field(description="Complete raw line/verse with original vocalization if present")
    sadr: str = Field(description="First hemistich (صدر البيت)")
    ajuz: str = Field(description="Second hemistich (عجز البيت)")
    normalized_search: str = Field(description="Normalized form for lexical and dense retrieval")
    stemmed_tokens: List[str] = Field(default_factory=list, description="Stemmed tokens for BM25 retrieval")
    prosody: ProsodyInfo = Field(default_factory=ProsodyInfo, description="Prosody and rhyme information")
    difficult_words: List[DifficultWord] = Field(default_factory=list, description="Annotated classical vocabulary")
    poet: Optional[str] = Field(default=None, description="Name of the poet (الشاعر)")
    era: Optional[str] = Field(default=None, description="Literary era (العصر: جاهلي، إسلامي، أموي، عباسي، أندلسي، حديث)")
    theme: Optional[str] = Field(default=None, description="Poetic genre/theme (الغرض: فخر، حكمة، غزل، رثاء، هجاء...)")
    poem_title: Optional[str] = Field(default=None, description="Title or opening of the poem (عنوان القصيدة)")
    source_row: Optional[int] = Field(default=None, description="Row index in original dataset")

    def formatted_bayt(self, separator: str = " ... ") -> str:
        """Returns the verse formatted as classical two hemistichs."""
        if self.sadr and self.ajuz:
            return f"{self.sadr}{separator}{self.ajuz}"
        return self.original_text

    def to_rag_context(self) -> str:
        """
        Builds a rich, grounded context block for LLM prompts.
        Crucially includes classical lexicon definitions so the LLM can explain
        archaic words without breaking grounded retrieval constraints.
        """
        lines = [
            f"البيت: {self.formatted_bayt()}",
        ]
        if self.poet:
            lines.append(f"الشاعر: {self.poet}")
        if self.era:
            lines.append(f"العصر: {self.era}")
        if self.prosody.meter and self.prosody.meter != "غير محدد":
            lines.append(f"البحر: {self.prosody.meter}")
        if self.prosody.rhyme_letter:
            lines.append(f"القافية (الروي): حرف {self.prosody.rhyme_letter}")
        if self.difficult_words:
            lex_entries = [f"- {w.word} ({w.root}): {w.meaning}" for w in self.difficult_words]
            lines.append("المفردات التراثية وغريب الألفاظ:")
            lines.extend(lex_entries)
        return "\n".join(lines)
