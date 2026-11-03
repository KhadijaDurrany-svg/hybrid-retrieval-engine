import math

from hre.metrics import bootstrap_ci, ndcg_at_k, recall_at_k, reciprocal_rank


def test_ndcg_perfect_is_one():
    rel = {"a": 2, "b": 1, "c": 1}
    assert math.isclose(ndcg_at_k(["a", "b", "c"], rel, 10), 1.0)


def test_ndcg_worse_order_is_lower():
    rel = {"a": 2, "b": 1, "c": 1}
    good = ndcg_at_k(["a", "b", "c"], rel, 10)
    bad = ndcg_at_k(["c", "b", "a"], rel, 10)
    assert bad < good


def test_recall_counts_relevant():
    rel = {"a": 1, "b": 1, "c": 0}
    assert math.isclose(recall_at_k(["a", "x", "y"], rel, 10), 0.5)


def test_reciprocal_rank_position():
    rel = {"b": 1}
    assert math.isclose(reciprocal_rank(["a", "b", "c"], rel), 0.5)
    assert reciprocal_rank(["a", "c"], rel) == 0.0


def test_bootstrap_ci_brackets_mean():
    vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    mean, lo, hi = bootstrap_ci(vals, n_resamples=500, seed=1)
    assert math.isclose(mean, 0.6)
    assert lo <= mean <= hi
