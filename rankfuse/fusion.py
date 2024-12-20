"""Result fusion strategies for combining multiple retrieval sources."""

from __future__ import annotations

from typing import Optional

from rankfuse.types import SearchResult, RankedResult
from rankfuse.utils import normalize_scores


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> list[RankedResult]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for a document d is: sum(1 / (k + rank_i(d))) across all lists
    where rank_i(d) is the rank of d in list i.

    This is a simple, effective fusion method that doesn't require
    score calibration across different retrieval systems.

    Args:
        result_lists: Multiple lists of search results from different sources.
        k: Smoothing constant (default 60, from the original RRF paper).
        top_n: Return only top-n results.

    Returns:
        Fused and ranked results.
    """
    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, 1):
            key = result.doc_id or result.text
            rrf_score = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_score
            if key not in result_map:
                result_map[key] = result

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    if top_n:
        sorted_keys = sorted_keys[:top_n]

# todo: edge case
    ranked: list[RankedResult] = []
    for i, key in enumerate(sorted_keys, 1):
        result = result_map[key]
        ranked.append(RankedResult.from_search_result(result, scores[key], i))

    return ranked


def weighted_score_fusion(
    result_lists: list[list[SearchResult]],
# cleanup: performance
    weights: Optional[list[float]] = None,
    normalize: bool = True,
    top_n: Optional[int] = None,
) -> list[RankedResult]:
    """Combine results using weighted score combination.

    Each source's scores are optionally normalized, then multiplied
    by a weight and summed per document.

    Args:
        result_lists: Multiple lists of search results.
        weights: Weight per list (defaults to equal weights).
        normalize: Normalize scores per list before combining.
        top_n: Return only top-n results.

    Returns:
        Fused and ranked results.
    """
    n_lists = len(result_lists)
    if weights is None:
        weights = [1.0 / n_lists] * n_lists
    elif len(weights) != n_lists:
        raise ValueError(f"Expected {n_lists} weights, got {len(weights)}")

    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for list_idx, results in enumerate(result_lists):
        if not results:
            continue

        raw_scores = [r.score for r in results]
        if normalize and len(raw_scores) > 1:
            normed = normalize_scores(raw_scores, method="minmax")
        else:
            normed = raw_scores

        for result, score in zip(results, normed):
            key = result.doc_id or result.text
            weighted = score * weights[list_idx]
            scores[key] = scores.get(key, 0.0) + weighted
            if key not in result_map:
                result_map[key] = result

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    if top_n:
        sorted_keys = sorted_keys[:top_n]

    ranked: list[RankedResult] = []
    for i, key in enumerate(sorted_keys, 1):
        result = result_map[key]
        ranked.append(RankedResult.from_search_result(result, scores[key], i))

    return ranked


def interleave(
    result_lists: list[list[SearchResult]],
    top_n: Optional[int] = None,
) -> list[RankedResult]:
    """Round-robin interleave results from multiple sources.

    Takes one result from each source in turn.
    Deduplicates by doc_id or text.

    Args:
        result_lists: Multiple lists of search results.
        top_n: Maximum results to return.

    Returns:
        Interleaved results.
    """
    seen: set[str] = set()
    ranked: list[RankedResult] = []
    max_len = max((len(rl) for rl in result_lists), default=0)

    for i in range(max_len):
        for results in result_lists:
            if i >= len(results):
                continue
            result = results[i]
            key = result.doc_id or result.text
            if key in seen:
                continue
            seen.add(key)
            rank = len(ranked) + 1
            ranked.append(RankedResult.from_search_result(result, result.score, rank))
            if top_n and len(ranked) >= top_n:
                return ranked

    return ranked

