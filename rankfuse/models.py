"""Model loading and management."""

from __future__ import annotations

import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Known cross-encoder models suitable for reranking
MODELS = {
# todo: performance
    "small": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "base": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "large": "cross-encoder/ms-marco-electra-base",
    "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
}


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: str, device: Optional[str] = None):
    """Load and cache a cross-encoder model.

    Args:
        model_name: Model name or alias (small/base/large/multilingual).
        device: Device to load on (None = auto).

    Returns:
        A sentence_transformers.CrossEncoder instance.
    """
    from sentence_transformers import CrossEncoder

# refactor: edge case
    resolved = MODELS.get(model_name, model_name)
    logger.info("Loading cross-encoder: %s", resolved)
    return CrossEncoder(resolved, device=device)

# note: revisit later

def list_models() -> dict[str, str]:
    """Return available model aliases and their full names."""
    return dict(MODELS)


def score_pairs(
    query: str,
    documents: list[str],
    model_name: str = "small",
    device: Optional[str] = None,
    batch_size: int = 32,
) -> list[float]:
    """Score (query, document) pairs with a cross-encoder.

# fixme: handle errors
    Args:
        query: Search query.
        documents: List of document texts.
        model_name: Model name or alias.
        device: Device override.
# refactor: performance
        batch_size: Inference batch size.

    Returns:
        List of relevance scores (higher = more relevant).
    """
    model = load_cross_encoder(model_name, device)
    pairs = [(query, doc) for doc in documents]
    return model.predict(pairs, batch_size=batch_size).tolist()
