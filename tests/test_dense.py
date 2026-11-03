import numpy as np

from hre.analysis import Analyzer
from hre.dense import DenseIndex
from hre.embedders import HashingLSAEmbedder

DOCS = [
    ("d0", "vector embedding nearest neighbour similarity search"),
    ("d1", "bm25 inverted index term frequency lexical ranking"),
    ("d2", "retrieval augmented generation grounding cited answer"),
    ("d3", "vector semantic dense retrieval proximity embedding"),
    ("d4", "forecast horizon seasonality demand time series"),
]


def _index():
    emb = HashingLSAEmbedder(dim=16, analyzer=Analyzer())
    emb.fit([t for _, t in DOCS])
    return DenseIndex(emb).fit([i for i, _ in DOCS], [t for _, t in DOCS])


def test_vectors_are_normalized():
    emb = HashingLSAEmbedder(dim=16)
    emb.fit([t for _, t in DOCS])
    vecs = emb.encode([t for _, t in DOCS])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_self_similarity_ranks_first():
    idx = _index()
    hits = idx.search("vector embedding nearest neighbour similarity search", k=3)
    assert hits[0][0] == "d0"


def test_topk_is_sorted_and_bounded():
    idx = _index()
    hits = idx.search("vector dense retrieval", k=10)
    assert len(hits) == len(DOCS)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_k_zero():
    idx = _index()
    assert idx.search("anything", k=0) == []
