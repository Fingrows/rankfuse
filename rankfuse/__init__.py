"""rankfuse - Reranking and result fusion for search/RAG."""

__version__ = "0.1.0"

from rankfuse.rerank import CrossEncoderReranker, rerank
from rankfuse.fusion import reciprocal_rank_fusion, weighted_score_fusion
from rankfuse.types import SearchResult, RankedResult

__all__ = [
    "CrossEncoderReranker",
    "rerank",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "SearchResult",
    "RankedResult",
]
