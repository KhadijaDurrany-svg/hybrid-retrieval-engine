import math

from hypothesis import given
from hypothesis import strategies as st

from hre.fusion import reciprocal_rank_fusion, weighted_fusion


def test_rrf_rewards_consistent_high_rank():
    lex = [("a", 9.0), ("b", 8.0), ("c", 1.0)]
    den = [("a", 0.9), ("c", 0.8), ("b", 0.1)]
    fused = reciprocal_rank_fusion([lex, den], k=3)
    assert fused[0][0] == "a"  # top of both lists wins


def test_rrf_scores_match_formula():
    lex = [("a", 5.0), ("b", 4.0)]
    den = [("b", 0.9), ("a", 0.1)]
    fused = dict(reciprocal_rank_fusion([lex, den], k=10, c=60.0))
    assert math.isclose(fused["a"], 1 / 60 + 1 / 61)
    assert math.isclose(fused["b"], 1 / 61 + 1 / 60)


def test_weighted_alpha_extremes_recover_single_list():
    lex = [("a", 10.0), ("b", 5.0), ("c", 0.0)]
    den = [("c", 1.0), ("b", 0.5), ("a", 0.0)]
    only_lex = [d for d, _ in weighted_fusion(lex, den, k=3, alpha=1.0)]
    only_den = [d for d, _ in weighted_fusion(lex, den, k=3, alpha=0.0)]
    assert only_lex[0] == "a"
    assert only_den[0] == "c"


@given(
    st.lists(st.tuples(st.text(min_size=1, max_size=3), st.floats(0, 1)), max_size=10),
)
def test_rrf_scores_non_negative(lst):
    fused = reciprocal_rank_fusion([lst], k=10)
    assert all(s >= 0 for _, s in fused)
