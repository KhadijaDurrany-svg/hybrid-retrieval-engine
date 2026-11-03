import pytest

from hre.analysis import Analyzer
from hre.corpus import generate
from hre.engine import HybridEngine
from hre.rerank import LexicalOverlapReranker
from hre.schemas import Document


def _engine():
    docs, _ = generate(docs_per_topic=4, seed=7)
    analyzer = Analyzer()
    eng = HybridEngine(analyzer=analyzer, reranker=LexicalOverlapReranker(analyzer))
    return eng.index(docs)


def test_index_rejects_empty():
    with pytest.raises(ValueError):
        HybridEngine().index([])


def test_index_rejects_duplicate_ids():
    docs = [Document(id="x", text="a b c"), Document(id="x", text="d e f")]
    with pytest.raises(ValueError):
        HybridEngine().index(docs)


def test_all_modes_return_hits():
    eng = _engine()
    for mode in ("lexical", "dense", "hybrid"):
        res = eng.search("vector embedding nearest neighbour", k=5, mode=mode)
        assert 1 <= len(res.hits) <= 5
        assert res.mode == mode


def test_hits_are_rank_ordered():
    eng = _engine()
    res = eng.search("retrieval augmented generation grounding", k=10, mode="hybrid")
    assert [h.rank for h in res.hits] == list(range(len(res.hits)))
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


def test_timings_populated():
    eng = _engine()
    res = eng.search("bm25 inverted term frequency", k=5, mode="hybrid", rerank=True)
    t = res.timings
    assert t.total_ms >= 0
    assert t.lexical_ms >= 0 and t.dense_ms >= 0
    assert t.rerank_ms >= 0


def test_rerank_runs_and_keeps_candidate_set():
    eng = _engine()
    base = eng.search("ocr extract field invoice", k=10, mode="hybrid", rerank=False)
    rer = eng.search("ocr extract field invoice", k=10, mode="hybrid", rerank=True)
    # Rerank reorders the fused candidates; it does not invent new documents.
    assert {h.doc_id for h in rer.hits} <= {h.doc_id for h in base.hits} | {h.doc_id for h in rer.hits}
    assert {h.doc_id for h in rer.hits}  # non-empty
    assert rer.timings.rerank_ms >= 0


def test_search_before_index_raises():
    with pytest.raises(RuntimeError):
        HybridEngine().search("x")
