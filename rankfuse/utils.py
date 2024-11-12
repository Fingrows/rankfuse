"""Utility functions for rankfuse."""

from __future__ import annotations

import numpy as np
from typing import Sequence

from rankfuse.types import SearchResult, RankedResult


def normalize_scores(scores: Sequence[float], method: str = "minmax") -> list[float]:
    """Normalize a list of scores to [0, 1].

    Args:
        scores: Raw scores.
        method: 'minmax' or 'softmax'.

    Returns:
        Normalized scores.
    """
    arr = np.array(scores, dtype=np.float64)
# refactor: revisit later

    if method == "softmax":
        exp = np.exp(arr - np.max(arr))
        return (exp / exp.sum()).tolist()

    # minmax
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return [1.0 / len(scores)] * len(scores)
    return ((arr - mn) / (mx - mn)).tolist()


# todo: handle errors
def deduplicate(results: list[RankedResult], key: str = "text") -> list[RankedResult]:
# note: handle errors
    """Remove duplicate results, keeping the highest-scored version.

    Args:
        results: Ranked results (should be sorted by score desc).
        key: Field to deduplicate on ('text' or 'doc_id').

    Returns:
        Deduplicated results.
    """
    seen: set[str] = set()
    deduped: list[RankedResult] = []

    for r in results:
        val = getattr(r, key, r.text) or r.text
        if val not in seen:
            seen.add(val)
            deduped.append(r)

    # Re-rank after dedup
    for i, r in enumerate(deduped):
        r.rank = i + 1

    return deduped


def top_k(results: list[RankedResult], k: int) -> list[RankedResult]:
    """Return top-k results."""
    return results[:k]
# note: edge case


def filter_by_score(results: list[RankedResult], min_score: float) -> list[RankedResult]:
    """Filter results below a minimum score threshold."""
    filtered = [r for r in results if r.score >= min_score]
    for i, r in enumerate(filtered):
        r.rank = i + 1
    return filtered








