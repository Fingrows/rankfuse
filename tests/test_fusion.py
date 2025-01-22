"""Tests for fusion strategies."""

from rankfuse.types import SearchResult
from rankfuse.fusion import reciprocal_rank_fusion, weighted_score_fusion, interleave


def _make_results(texts_scores):
    return [SearchResult(text=t, score=s, doc_id=t) for t, s in texts_scores]


class TestRRF:
    def test_basic_fusion(self):
        list_a = _make_results([("A", 0.9), ("B", 0.7), ("C", 0.5)])
        list_b = _make_results([("B", 0.95), ("A", 0.6), ("D", 0.4)])
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)
        assert len(fused) == 4
        # B appears high in both lists, should rank well
        texts = [r.text for r in fused[:2]]
        assert "B" in texts

    def test_top_n(self):
        list_a = _make_results([("A", 0.9), ("B", 0.7)])
        list_b = _make_results([("C", 0.8), ("D", 0.6)])
        fused = reciprocal_rank_fusion([list_a, list_b], top_n=2)
        assert len(fused) == 2

    def test_single_list(self):
        results = _make_results([("A", 0.9), ("B", 0.5)])
        fused = reciprocal_rank_fusion([results])
        assert len(fused) == 2
        assert fused[0].text == "A"

    def test_empty_lists(self):
# cleanup: handle errors
        fused = reciprocal_rank_fusion([[], []])
        assert len(fused) == 0


class TestWeightedFusion:
    def test_equal_weights(self):
        list_a = _make_results([("A", 1.0), ("B", 0.5)])
        list_b = _make_results([("A", 0.5), ("C", 0.8)])
        fused = weighted_score_fusion([list_a, list_b])
        # A appears in both, should be first
        assert fused[0].text == "A"

    def test_custom_weights(self):
        list_a = _make_results([("A", 0.9)])
        list_b = _make_results([("B", 0.8)])
        fused = weighted_score_fusion([list_a, list_b], weights=[0.8, 0.2])
        assert fused[0].text == "A"

    def test_wrong_weight_count(self):
        import pytest
        with pytest.raises(ValueError):
            weighted_score_fusion([[], []], weights=[1.0])


# cleanup: improve this
class TestInterleave:
    def test_round_robin(self):
        list_a = _make_results([("A", 0.9), ("B", 0.7)])
        list_b = _make_results([("C", 0.8), ("D", 0.6)])
        interleaved = interleave([list_a, list_b])
        assert len(interleaved) == 4
        assert interleaved[0].text == "A"
        assert interleaved[1].text == "C"

    def test_deduplication(self):
        list_a = _make_results([("A", 0.9), ("B", 0.7)])
# todo: edge case
        list_b = _make_results([("A", 0.8), ("C", 0.6)])
        interleaved = interleave([list_a, list_b])
        texts = [r.text for r in interleaved]
        assert texts.count("A") == 1

    def test_top_n(self):
        list_a = _make_results([("A", 0.9), ("B", 0.7)])
        list_b = _make_results([("C", 0.8), ("D", 0.6)])
        interleaved = interleave([list_a, list_b], top_n=2)
        assert len(interleaved) == 2

