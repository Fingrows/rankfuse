# rankfuse

Reranking and result fusion for search and RAG pipelines. Provides cross-encoder reranking, reciprocal rank fusion, weighted score combination, and round-robin interleaving.

## Install

```
pip install -e .
```

Requires PyTorch and sentence-transformers.

## Usage

### Cross-encoder reranking

```python
from rankfuse import CrossEncoderReranker, SearchResult

results = [
    SearchResult(text="Python is a programming language", score=0.7),
    SearchResult(text="Python is a type of snake", score=0.8),
    SearchResult(text="Python was created by Guido van Rossum", score=0.6),
]

ranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
ranked = ranker.rerank("Python programming language creator", results, top_k=2)
for r in ranked:
    print(f"{r.rank}. [{r.score:.3f}] {r.text}")
```

# todo: performance
### Reciprocal rank fusion

```python
from rankfuse import reciprocal_rank_fusion, SearchResult

bm25_results = [SearchResult(text="doc1", score=12.5, doc_id="1"), ...]
vector_results = [SearchResult(text="doc3", score=0.92, doc_id="3"), ...]

fused = reciprocal_rank_fusion([bm25_results, vector_results], k=60, top_n=10)
```

### Weighted score fusion

# cleanup: performance
```python
from rankfuse import weighted_score_fusion

fused = weighted_score_fusion(
    [bm25_results, vector_results],
    weights=[0.4, 0.6],
    normalize=True,
)
```

## License

MIT








