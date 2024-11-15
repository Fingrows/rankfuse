"""Cross-encoder reranking."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from rankfuse.types import SearchResult, RankedResult
from rankfuse.utils import normalize_scores

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks results using a cross-encoder model.

    Cross-encoders score (query, document) pairs jointly,
    producing much more accurate relevance scores than
    bi-encoder similarity.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: Optional[int] = None,
        normalize: bool = True,
    ) -> list[RankedResult]:
        """Rerank search results for a query.

        Args:
            query: The search query.
            results: Search results to rerank.
            top_k: Return only top-k results (None = all).
            normalize: Normalize scores to [0, 1].

        Returns:
            Reranked results sorted by relevance.
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size).tolist()

        if normalize:
            scores = normalize_scores(scores, method="minmax")

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            scored = scored[:top_k]

        ranked = []
        for rank, (result, score) in enumerate(scored, 1):
            ranked.append(RankedResult.from_search_result(result, score, rank))

        return ranked


def rerank(
    query: str,
    results: list[SearchResult],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None,
    normalize: bool = True,
) -> list[RankedResult]:
    """Convenience function for one-off reranking."""
    ranker = CrossEncoderReranker(model_name=model_name)
    return ranker.rerank(query, results, top_k=top_k, normalize=normalize)

