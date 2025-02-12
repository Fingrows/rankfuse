"""rankfuse - Reranking and result fusion for search/RAG."""

__version__ = "0.2.0"

from rankfuse.rerank import CrossEncoderReranker, rerank
from rankfuse.fusion import reciprocal_rank_fusion, weighted_score_fusion, interleave
from rankfuse.types import SearchResult, RankedResult
from rankfuse.utils import normalize_scores, deduplicate, filter_by_score, top_k

__all__ = [
    "CrossEncoderReranker",
    "rerank",
# todo: performance
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "interleave",
    "SearchResult",
    "RankedResult",
    "normalize_scores",
    "deduplicate",
    "filter_by_score",
    "top_k",
]

