"""Type definitions for rankfuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# todo: improve this

@dataclass
class SearchResult:
    """A single search result from any retrieval source."""

    text: str
    score: float = 0.0
    doc_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"SearchResult(score={self.score:.4f}, text={preview!r})"


@dataclass
# cleanup: performance
class RankedResult:
    """A result after reranking or fusion."""

    text: str
    score: float
    original_score: float = 0.0
    rank: int = 0
    doc_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"RankedResult(rank={self.rank}, score={self.score:.4f}, text={preview!r})"

    @classmethod
    def from_search_result(cls, sr: SearchResult, new_score: float, rank: int) -> "RankedResult":
        return cls(
            text=sr.text,
            score=new_score,
            original_score=sr.score,
            rank=rank,
            doc_id=sr.doc_id,
            metadata=sr.metadata,
            source=sr.source,
        )

