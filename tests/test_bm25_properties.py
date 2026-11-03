"""Property-based tests for the BM25 index.

The central invariant: WAND pruning must return exactly the same top-k scores
as an exhaustive brute-force scan. If pruning ever drops a document that should
have scored into the top-k, these tests fail.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from hre.analysis import Analyzer
from hre.bm25 import BM25Index, BM25Params

VOCAB = ["alpha", "bravo", "delta", "gamma", "hotel", "india", "foxtrot", "kilo", "lima", "mike"]

doc_strategy = st.lists(st.sampled_from(VOCAB), min_size=0, max_size=12).map(" ".join)
corpus_strategy = st.lists(doc_strategy, min_size=1, max_size=40)
query_strategy = st.lists(st.sampled_from(VOCAB), min_size=1, max_size=4).map(" ".join)


def _build(corpus: list[str]) -> BM25Index:
    idx = BM25Index(analyzer=Analyzer(stem=False), params=BM25Params())
    for i, text in enumerate(corpus):
        idx.add(f"d{i}", text)
    return idx.build()


@settings(max_examples=250, deadline=None)
@given(corpus=corpus_strategy, query=query_strategy, k=st.integers(min_value=1, max_value=15))
def test_wand_matches_bruteforce_scores(corpus, query, k):
    idx = _build(corpus)
    wand = idx.search(query, k)
    brute = idx.search_bruteforce(query, k)

    assert len(wand) == len(brute)
    wand_scores = sorted((s for _, s in wand), reverse=True)
    brute_scores = sorted((s for _, s in brute), reverse=True)
    for a, b in zip(wand_scores, brute_scores, strict=True):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


@settings(max_examples=150, deadline=None)
@given(corpus=corpus_strategy, query=query_strategy, k=st.integers(min_value=1, max_value=10))
def test_wand_is_valid_topk(corpus, query, k):
    """Every returned score is >= every non-returned document's score."""
    idx = _build(corpus)
    wand = idx.search(query, k)
    all_scores = idx.search_bruteforce(query, idx.n_docs)
    returned_ids = {d for d, _ in wand}
    if not wand:
        assert all(s == 0 for _, s in all_scores) or not all_scores
        return
    min_returned = min(s for _, s in wand)
    for doc_id, s in all_scores:
        if doc_id not in returned_ids:
            assert s <= min_returned + 1e-9


@settings(max_examples=100, deadline=None)
@given(corpus=corpus_strategy)
def test_results_are_descending(corpus):
    idx = _build(corpus)
    hits = idx.search("alpha bravo delta", 10)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
