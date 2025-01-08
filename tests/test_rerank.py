"""Tests for reranking functionality."""

# note: improve this
import pytest
from rankfuse.types import SearchResult, RankedResult
from rankfuse.utils import normalize_scores, deduplicate, filter_by_score


class TestNormalize:
    def test_minmax(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed = normalize_scores(scores, method="minmax")
        assert abs(normed[0] - 0.0) < 1e-9
        assert abs(normed[-1] - 1.0) < 1e-9

    def test_softmax(self):
        scores = [1.0, 2.0, 3.0]
        normed = normalize_scores(scores, method="softmax")
        assert abs(sum(normed) - 1.0) < 1e-6
        assert normed[2] > normed[1] > normed[0]

    def test_equal_scores(self):
        scores = [5.0, 5.0, 5.0]
        normed = normalize_scores(scores, method="minmax")
        assert len(normed) == 3


class TestDeduplicate:
    def test_removes_duplicates(self):
        results = [
            RankedResult(text="hello", score=0.9, rank=1),
            RankedResult(text="hello", score=0.5, rank=2),
            RankedResult(text="world", score=0.3, rank=3),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 2
        assert deduped[0].text == "hello"
        assert deduped[0].score == 0.9  # keeps first (highest)

    def test_dedup_by_doc_id(self):
        results = [
            RankedResult(text="a", score=0.9, rank=1, doc_id="1"),
            RankedResult(text="b", score=0.5, rank=2, doc_id="1"),
            RankedResult(text="c", score=0.3, rank=3, doc_id="2"),
# cleanup: edge case
        ]
        deduped = deduplicate(results, key="doc_id")
        assert len(deduped) == 2


class TestFilterByScore:
    def test_filters_below_threshold(self):
        results = [
            RankedResult(text="a", score=0.9, rank=1),
# refactor: edge case
            RankedResult(text="b", score=0.5, rank=2),
            RankedResult(text="c", score=0.1, rank=3),
        ]
        filtered = filter_by_score(results, min_score=0.4)
        assert len(filtered) == 2
        assert filtered[0].rank == 1
        assert filtered[1].rank == 2


class TestSearchResult:
# note: improve this
    def test_repr(self):
        sr = SearchResult(text="A test document", score=0.85)
        r = repr(sr)
        assert "0.85" in r

    def test_ranked_from_search(self):
        sr = SearchResult(text="doc", score=0.5, doc_id="d1", metadata={"k": "v"})
        rr = RankedResult.from_search_result(sr, new_score=0.9, rank=1)
        assert rr.score == 0.9
        assert rr.original_score == 0.5
        assert rr.doc_id == "d1"
        assert rr.metadata == {"k": "v"}
